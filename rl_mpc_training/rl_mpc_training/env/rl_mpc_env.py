#!/usr/bin/env python3

"""Step 4.2.1 - Gymnasium Environment with episode reset."""

import math
import threading
import time

import gymnasium as gym
import numpy as np
import rclpy

from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformException

from rl_mpc_training.ros.ros_bridge import RosBridge

from rl_mpc_training.utils.path_geometry import (
    compute_path_geometry
)

from rl_mpc_training.utils.lidar_features import (
    compute_lidar_features
)

from rl_mpc_training.utils.motion_features import (
    compute_motion_features
)

from rl_mpc_training.utils.observation_builder import (
    build_observation,
    OBSERVATION_DIM
)

ERROR_PHASES = {
    'RESET_ERROR',
    'LOCALIZATION_ERROR',
    'EPISODE_ERROR',
    'EPISODE_END_ERROR',
}

TERMINAL_PHASES = {
    'EPISODE_SUCCESS',
    'EPISODE_TIMEOUT',
    'EPISODE_COLLISION',
    'EPISODE_PLANNING_FAILURE',
}


class RLMPCEnv(gym.Env):

    metadata = {'render_modes': []}

    def __init__(self):

        super().__init__()

        # ============================================
        # Temporary spaces for reset integration test
        # ============================================

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        # ============================================
        # ROS2 initialization
        # ============================================

        if rclpy.ok():
            raise RuntimeError(
                'RLMPCEnv requires its own ROS2 process'
            )

        self._closed = False
        self._sequence_touched = False

        rclpy.init()

        self.bridge = RosBridge()

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.bridge)

        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True
        )

        self.spin_thread.start()

    # ================================================
    # Wait for ROS2 interfaces
    # ================================================

    def _wait_for_services(self, timeout_sec=45.0):

        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:

            missing = [
                name
                for name, client in self.bridge.required_clients.items()
                if not client.service_is_ready()
            ]

            if not missing:
                return

            time.sleep(0.2)

        raise TimeoutError(
            f'ROS2 services unavailable: {missing}'
        )

    # ================================================
    # Service helpers
    # ================================================

    def _trigger(self, client, name):

        return self.bridge.call_service_wait(
            client,
            Trigger.Request(),
            name,
            timeout_sec=10.0
        )

    def _check_ack(self, response, name):

        if not response.success:

            raise RuntimeError(
                f'{name} rejected: {response.message}'
            )

    def _get_phase(self, client):

        response = self._trigger(
            client,
            'status_poll'
        )

        status = self.bridge.parse_service_message(
            response.message
        )

        if 'phase' not in status:

            raise RuntimeError(
                'Episode Manager response has no phase'
            )

        return status

    def _wait_phase(
        self,
        client,
        expected_phase,
        timeout_sec
    ):

        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:

            status = self._get_phase(client)

            phase = status['phase']

            if phase == expected_phase:

                print(
                    f'[PASS] {expected_phase}',
                    flush=True
                )

                return status

            if phase in ERROR_PHASES:

                raise RuntimeError(
                    f'Episode Manager error: {status}'
                )

            if (
                expected_phase == 'EPISODE_RUNNING'
                and phase in TERMINAL_PHASES
            ):

                raise RuntimeError(
                    f'Episode ended before reset returned: '
                    f'{status}'
                )

            time.sleep(0.2)

        raise TimeoutError(
            f'Timeout waiting for {expected_phase}'
        )

    # ================================================
    # Get synchronized initial observation
    # ================================================

    def _initial_observation(self, baseline, ack_baseline, episode_start_ns, timeout_sec=8.0):

        def stamp_ns(stamp):

            return (
                int(stamp.sec) * 1_000_000_000
                + int(stamp.nanosec)
            )

        def yaw_from_quaternion(q):

            return math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

        deadline = time.monotonic() + timeout_sec

        last_reason = 'Waiting for observation sources'

        while time.monotonic() < deadline:

            # --------------------------------------------
            # 1. Get consistent snapshot
            # --------------------------------------------

            snapshot = self.bridge.get_observation_snapshot()

            path = snapshot['accepted_path']
            reference = snapshot['reference']
            odom = snapshot['odom']
            scan = snapshot['scan']
            params = snapshot['adaptive_params']

            # --------------------------------------------
            # 2. Check available data
            # --------------------------------------------

            if (
                path is None
                or reference is None
                or odom is None
                or scan is None
                or params is None
            ):

                last_reason = 'Missing observation source'
                time.sleep(0.05)
                continue

            if snapshot['accepted_path_invalidated']:

                last_reason = 'Accepted path invalidated'
                time.sleep(0.05)
                continue

            if len(path.poses) < 2:

                last_reason = 'Accepted path too short'
                time.sleep(0.05)
                continue

            # --------------------------------------------
            # 3. Require messages newer than baselines
            # --------------------------------------------

            counters = (
                'accepted_path_count',
                'reference_count',
                'odom_count',
                'scan_count'
            )

            fresh = all(
                snapshot[key] > baseline[key]
                for key in counters
            )

            if not fresh:

                last_reason = 'Waiting for fresh episode data'
                time.sleep(0.05)
                continue

            # MPC must have published a new ACK since
            # the reset sequence started.
            if snapshot['adaptive_ack_seq'] <= ack_baseline:

                last_reason = 'Waiting for reset MPC ACK'
                time.sleep(0.05)
                continue

            # --------------------------------------------
            # 4. Check default MPC parameters
            # --------------------------------------------

            expected_params = [
                1.0, 1.0, 1.0, 1.0, 0.5
            ]

            if not np.allclose(
                params,
                expected_params,
                rtol=0.0,
                atol=1e-8
            ):

                last_reason = 'MPC parameters not reset'
                time.sleep(0.05)
                continue

            # --------------------------------------------
            # 5. Check coordinate frames
            # --------------------------------------------

            if (
                path.header.frame_id != 'map'
                or reference.header.frame_id != 'map'
                or odom.child_frame_id != self.bridge.robot_frame
            ):

                raise RuntimeError(
                    'Unexpected observation coordinate frame'
                )

            # --------------------------------------------
            # 6. Check message timestamps
            # --------------------------------------------

            path_ns = stamp_ns(path.header.stamp)
            reference_ns = stamp_ns(reference.header.stamp)
            odom_ns = stamp_ns(odom.header.stamp)
            scan_ns = stamp_ns(scan.header.stamp)

            if min(
                path_ns,
                reference_ns,
                odom_ns,
                scan_ns
            ) <= 0:

                last_reason = 'Invalid observation timestamp'
                time.sleep(0.05)
                continue

            # Reject data predating episode startup.
            if (
                path_ns < episode_start_ns
                or reference_ns < episode_start_ns
                or odom_ns < episode_start_ns
                or scan_ns < episode_start_ns
            ):

                last_reason = 'Data predates current episode'
                time.sleep(0.05)
                continue

            # MPC reference must not predate accepted path.
            if reference_ns < path_ns:

                last_reason = 'MPC reference older than path'
                time.sleep(0.05)
                continue

            now_ns = self.bridge.get_clock().now().nanoseconds

            sensor_timestamps = [
                reference_ns,
                odom_ns,
                scan_ns
            ]

            ages = {
                'reference': (now_ns - reference_ns) / 1e9,
                'odom': (now_ns - odom_ns) / 1e9,
                'scan': (now_ns - scan_ns) / 1e9,
            }

            if any(
                age < -0.1 or age > 0.5
                for age in ages.values()
            ):

                last_reason = 'Stale or future sensor data'
                time.sleep(0.05)
                continue

            skew_sec = (
                max(sensor_timestamps)
                - min(sensor_timestamps)
            ) / 1e9

            if skew_sec > 0.3:

                last_reason = 'Sensor timestamp skew too large'
                time.sleep(0.05)
                continue

            # --------------------------------------------
            # 7. Robot pose from TF
            # --------------------------------------------

            try:

                transform = self.bridge.tf_buffer.lookup_transform(
                    'map',
                    self.bridge.robot_frame,
                    Time()
                )

            except TransformException:

                last_reason = 'TF unavailable'
                time.sleep(0.05)
                continue

            tf_ns = stamp_ns(transform.header.stamp)

            tf_age_sec = (now_ns - tf_ns) / 1e9

            # Allow some TF timestamp offset due to
            # AMCL transform publication tolerance.
            if (
                tf_ns <= 0
                or tf_age_sec < -1.5
                or tf_age_sec > 1.5
            ):

                last_reason = 'TF timestamp outside tolerance'
                time.sleep(0.05)
                continue

            p = transform.transform.translation
            q = transform.transform.rotation

            robot_pose = [
                float(p.x),
                float(p.y),
                yaw_from_quaternion(q)
            ]

            # --------------------------------------------
            # 8. MPC reference
            # --------------------------------------------

            rp = reference.pose.position
            rq = reference.pose.orientation

            reference_pose = [
                float(rp.x),
                float(rp.y),
                yaw_from_quaternion(rq)
            ]

            # --------------------------------------------
            # 9. Accepted path
            # --------------------------------------------

            path_xy = np.array(
                [
                    [
                        pose.pose.position.x,
                        pose.pose.position.y
                    ]
                    for pose in path.poses
                ],
                dtype=np.float64
            )

            # --------------------------------------------
            # 10. Compute geometric features
            # --------------------------------------------

            geometry = compute_path_geometry(
                robot_pose,
                reference_pose,
                path_xy,
                lookahead_m=1.0
            )

            # --------------------------------------------
            # 11. Compute LiDAR features
            # --------------------------------------------

            lidar = compute_lidar_features(scan)

            # --------------------------------------------
            # 12. Compute motion features
            # --------------------------------------------

            motion = compute_motion_features(
                float(odom.twist.twist.linear.x),
                float(odom.twist.twist.angular.z),
                params
            )

            # --------------------------------------------
            # 13. Build normalized observation
            # --------------------------------------------

            result = build_observation(
                geometry,
                lidar,
                motion,
                lidar_max_m=float(scan.range_max)
            )

            # --------------------------------------------
            # 14. Check whether path changed during work
            # --------------------------------------------

            latest = self.bridge.get_observation_snapshot()

            if (
                latest['accepted_path_count']
                != snapshot['accepted_path_count']
                or latest['accepted_path_invalidated']
            ):

                last_reason = 'Path changed during observation'
                time.sleep(0.05)
                continue

            observation = result['normalized']
            raw_observation = result['raw']

            metadata = {
                'sensor_ages_s': ages,
                'sensor_skew_s': skew_sec,
                'tf_age_s': tf_age_sec,
                'path_stamp_ns': path_ns,
                'observation_counts': {
                    key: snapshot[key]
                    for key in counters
                }
            }

            return observation, raw_observation, metadata

        raise TimeoutError(
            'Synchronized observation unavailable: '
            f'{last_reason}'
        )

    # ================================================
    # Safety HOLD
    # ================================================

    def _safety_hold(self):

        if not self._sequence_touched:
            return

        try:

            request = SetBool.Request()
            request.data = True

            response = self.bridge.call_service_wait(
                self.bridge.mpc_hold_client,
                request,
                'mpc_hold',
                timeout_sec=10.0
            )

            if not response.success:

                raise RuntimeError(
                    response.message
                )

            self._sequence_touched = False

            print(
                '[SAFETY] MPC HOLD confirmed',
                flush=True
            )

        except Exception as exc:

            print(
                f'[SAFETY] HOLD unconfirmed: {exc}',
                flush=True
            )

    # ================================================
    # Gymnasium reset()
    # ================================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        if self._closed:

            raise RuntimeError(
                'RLMPCEnv has already been closed'
            )

        try:

            # ----------------------------------------
            # 1. Check interfaces
            # ----------------------------------------

            print('\n[RESET] Check ROS2 services')

            self._wait_for_services()

            status = self._get_phase(
                self.bridge.episode_status_client
            )

            allowed = (
                {'IDLE', 'EPISODE_RUNNING'}
                | TERMINAL_PHASES
            )

            if status['phase'] not in allowed:

                raise RuntimeError(
                    f'Cannot reset from phase={status["phase"]}'
                )

            # Record MPC ACK sequence before resetting.
            pre_reset_snapshot = (
                self.bridge.get_observation_snapshot()
            )

            ack_baseline = (
                pre_reset_snapshot['adaptive_ack_seq']
            )

            # Any following request may affect robot state.
            self._sequence_touched = True

            # ----------------------------------------
            # 2. Reset control pipeline
            # ----------------------------------------

            print('[RESET] Reset Control Pipeline')

            response = self._trigger(
                self.bridge.reset_control_client,
                'reset_control_pipeline'
            )

            self._check_ack(
                response,
                'reset_control_pipeline'
            )

            self._wait_phase(
                self.bridge.reset_status_client,
                'RESET_READY',
                30.0
            )

            # ----------------------------------------
            # 3. Reset localization
            # ----------------------------------------

            print('[RESET] Reset Localization')

            response = self._trigger(
                self.bridge.reset_localization_client,
                'reset_localization'
            )

            self._check_ack(
                response,
                'reset_localization'
            )

            self._wait_phase(
                self.bridge.reset_status_client,
                'LOCALIZATION_READY',
                45.0
            )

            # ----------------------------------------
            # 4. Start episode
            # ----------------------------------------

            baseline = self.bridge.get_observation_snapshot()

            episode_start_ns = (
                self.bridge.get_clock().now().nanoseconds
            )

            if episode_start_ns <= 0:
                raise RuntimeError(
                    'Simulation clock unavailable'
                )

            print('[RESET] Start Episode')

            response = self._trigger(
                self.bridge.start_episode_client,
                'start_episode'
            )

            self._check_ack(
                response,
                'start_episode'
            )

            status = self._wait_phase(
                self.bridge.episode_status_client,
                'EPISODE_RUNNING',
                50.0
            )

            # ----------------------------------------
            # 5. Initial observation
            # ----------------------------------------

            observation, raw_observation, metadata = (
                self._initial_observation(
                    baseline=baseline,
                    ack_baseline=ack_baseline,
                    episode_start_ns=episode_start_ns,
                    timeout_sec=8.0
                )
            )

            if not self.observation_space.contains(observation):

                raise RuntimeError(
                    'Initial observation outside observation_space'
                )

            # Ensure the episode is still running.
            latest_status = self._get_phase(
                self.bridge.episode_status_client
            )

            if latest_status['phase'] != 'EPISODE_RUNNING':

                raise RuntimeError(
                    'Episode changed phase during observation: '
                    f'{latest_status}'
                )

            info = {
                'phase': latest_status['phase'],
                'attempt': int(latest_status['attempt']),

                'raw_observation': raw_observation,

                'sensor_ages_s': metadata['sensor_ages_s'],
                'sensor_skew_s': metadata['sensor_skew_s'],
                'tf_age_s': metadata['tf_age_s'],

                'path_stamp_ns': metadata['path_stamp_ns'],

                'observation_counts':
                    metadata['observation_counts'],

                'diagnostic_observation_only': True
            }

            print('[PASS] Gymnasium reset completed')

            return observation, info

        except Exception:

            self._safety_hold()
            raise

    # ================================================
    # Gymnasium step() - Not yet implemented
    # ================================================

    def step(self, action):

        raise NotImplementedError(
            'step() has not been implemented yet'
        )

    # ================================================
    # Close environment
    # ================================================

    def close(self):

        if self._closed:
            return

        self._safety_hold()

        self.executor.shutdown()

        self.spin_thread.join(timeout=2.0)

        self.bridge.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        self._closed = True
