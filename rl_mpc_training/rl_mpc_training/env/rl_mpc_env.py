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

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)

from rl_mpc_training.utils.step_utils import (
    RL_PERIOD_NS,
    classify_episode_status,
    schedule_next_tick_ns
)

from rl_mpc_training.utils.terminal_observation import (
    build_terminal_observation_candidate
)

from rl_mpc_training.utils.terminal_transition import (
    build_terminal_transition
)

from rl_mpc_training.utils.reward_function import (
    compute_reward,
    TERMINAL_REWARDS
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
        # RL step state
        # ============================================

        self._episode_done = True
        self._active_attempt = None

        self._episode_path_stamp_ns = None

        self._last_observation = None
        self._last_raw_observation = None

        self._last_return_sim_ns = None
        self._next_rl_tick_ns = None

        self._step_count = 0

        # Last action actually applied during the episode.
        self._last_action = np.array(
            [-1.0, -1.0, -1.0, -1.0, 1.0],
            dtype=np.float64
        )

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

    def _initial_observation(
        self,
        baseline,
        ack_baseline,
        episode_start_ns,
        timeout_sec=8.0,
        expected_params=None,
        require_new_path=True,
        min_sensor_stamp_ns=None
    ):

        # Default parameter check for Gymnasium reset().
        if expected_params is None:
            expected_params = [
                1.0, 1.0, 1.0, 1.0, 0.5
            ]

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

            # During reset, require a newly accepted path.
            # During a running episode, an unchanged path is valid.
            required_counters = (
                counters
                if require_new_path
                else (
                    'reference_count',
                    'odom_count',
                    'scan_count'
                )
            )

            fresh = all(
                snapshot[key] > baseline[key]
                for key in required_counters
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

            if not np.allclose(
                params,
                expected_params,
                rtol=0.0,
                atol=1e-8
            ):

                last_reason = (
                    'MPC parameters mismatch: '
                    f'expected={expected_params}, '
                    f'actual={params}'
                )

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

            # Require sensor measurements after the RL sample tick
            if min_sensor_stamp_ns is not None:

                if min(reference_ns, odom_ns, scan_ns) < min_sensor_stamp_ns:

                    last_reason = (
                        'Waiting for sensor data after RL tick'
                    )

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

        self._episode_done = True

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

            # ========================================
            # 2-4. Reset and start episode
            #
            # Retry the FULL reset sequence only when
            # start_episode rejects transient localization.
            #
            # The Episode Manager remains responsible
            # for checking localization validity.
            # ========================================

            MAX_LOCALIZATION_ATTEMPTS = 3
            RESET_RETRY_DELAY_SEC = 0.5

            for reset_attempt in range(
                1,
                MAX_LOCALIZATION_ATTEMPTS + 1
            ):

                print(
                    f'\n[RESET ATTEMPT '
                    f'{reset_attempt}/'
                    f'{MAX_LOCALIZATION_ATTEMPTS}]',
                    flush=True
                )

                # ------------------------------------
                # 2. Reset control pipeline
                # ------------------------------------

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

                # ------------------------------------
                # 3. Reset localization
                # ------------------------------------

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

                # ------------------------------------
                # 4. Prepare episode start
                # ------------------------------------

                baseline = (
                    self.bridge.get_observation_snapshot()
                )

                episode_start_ns = (
                    self.bridge.get_clock()
                    .now()
                    .nanoseconds
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

                # ------------------------------------
                # Accepted: continue normal reset
                # ------------------------------------

                if response.success:

                    print(
                        '[RESET] Start Episode accepted',
                        flush=True
                    )

                    break

                # ------------------------------------
                # Rejected: classify the reason
                # ------------------------------------

                error_message = str(
                    response.message
                )

                LOCALIZATION_ERROR = (
                    'Start rejected: '
                    'localization no longer valid'
                )

                # Any other rejection is NOT retried.
                if error_message.strip() != LOCALIZATION_ERROR:

                    self._check_ack(
                        response,
                        'start_episode'
                    )

                # ------------------------------------
                # Confirm the Episode Manager is still
                # in LOCALIZATION_READY.
                # ------------------------------------

                status = self._get_phase(
                    self.bridge.reset_status_client
                )

                if status['phase'] != 'LOCALIZATION_READY':

                    raise RuntimeError(
                        'Localization rejection changed '
                        'Episode Manager phase: '
                        f'{status}'
                    )

                # ------------------------------------
                # Retry only while attempts remain.
                # ------------------------------------

                print(
                    '[WARNING] start_episode rejected: '
                    'localization no longer valid',
                    flush=True
                )

                if (
                    reset_attempt
                    >= MAX_LOCALIZATION_ATTEMPTS
                ):

                    raise RuntimeError(
                        'start_episode failed after '
                        f'{MAX_LOCALIZATION_ATTEMPTS} '
                        'complete localization resets. '
                        'Inspect EKF, AMCL and TF '
                        'in Episode Manager.'
                    )

                print(
                    '[RECOVERY] Restarting full '
                    'control/localization reset; '
                    'MPC remains HOLD',
                    flush=True
                )

                time.sleep(
                    RESET_RETRY_DELAY_SEC
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

            # ----------------------------------------
            # Initialize RL step state
            # ----------------------------------------

            now_sim_ns = (
                self.bridge.get_clock().now().nanoseconds
            )

            self._episode_done = False

            self._active_attempt = int(
                latest_status['attempt']
            )

            self._episode_path_stamp_ns = (
                metadata['path_stamp_ns']
            )

            self._last_observation = observation.copy()
            self._last_raw_observation = raw_observation.copy()

            self._last_return_sim_ns = now_sim_ns

            self._next_rl_tick_ns = (
                now_sim_ns + RL_PERIOD_NS
            )

            self._step_count = 0

            # The MPC has been reset to its default parameters.
            self._last_action = np.array(
                [-1.0, -1.0, -1.0, -1.0, 1.0],
                dtype=np.float64
            )

            print('[PASS] Gymnasium reset completed')

            return observation, info

        except Exception:

            self._safety_hold()
            raise

    def _step_episode_status(self):

        status = self._get_phase(
            self.bridge.episode_status_client
        )

        state = classify_episode_status(status)

        attempt = int(
            status.get('attempt', -1)
        )

        if attempt != self._active_attempt:

            raise RuntimeError(
                'Episode identity changed during step: '
                f'expected={self._active_attempt}, '
                f'actual={attempt}'
            )

        return status, state

    def _compute_transition_reward(
        self,
        current_raw,
        current_action,
        dt_s,
        phase,
        reward_mode='full'
    ):
        """
        Compute reward for one Gymnasium transition.

        full:
            Use current and previous raw observations.

        event_only:
            Use only the terminal event reward.
        """

        if reward_mode == 'full':

            if dt_s <= 0.0:
                raise ValueError(
                    'Full reward requires positive dt'
                )

            return compute_reward(
                previous_raw=self._last_raw_observation,
                current_raw=current_raw,
                previous_action=self._last_action,
                current_action=current_action,
                dt_s=dt_s,
                phase=phase
            )

        if reward_mode != 'event_only':

            raise ValueError(
                f'Unknown reward mode: {reward_mode}'
            )

        if phase not in TERMINAL_REWARDS:

            raise ValueError(
                f'Unknown terminal phase: {phase}'
            )

        if phase == 'EPISODE_RUNNING':

            raise ValueError(
                'Event-only reward requires terminal phase'
            )

        terminal_reward = float(
            TERMINAL_REWARDS[phase]
        )

        components = {
            'progress': 0.0,
            'path': 0.0,
            'heading': 0.0,
            'clearance': 0.0,
            'unknown_lidar': 0.0,
            'action_change': 0.0,
            'time': 0.0,
            'terminal': terminal_reward
        }

        return {
            'total': terminal_reward,
            'components': components,
            'metrics': {
                'dt_s': float(dt_s)
            }
        }

    def _collect_terminal_observation(self, timeout_sec=1.5):
        """
        Try to collect a fresh observation after terminal.

        Returns:
            (candidate, reason)

        candidate is None when no verified observation
        can be constructed.
        """

        bridge = self.bridge

        deadline = time.monotonic() + timeout_sec

        last_reason = 'Waiting for terminal observation'

        while time.monotonic() < deadline:

            snapshot = bridge.get_observation_snapshot()

            now_ns = bridge.get_clock().now().nanoseconds

            try:

                transform = bridge.tf_buffer.lookup_transform(
                    'map',
                    bridge.robot_frame,
                    Time()
                )

                candidate = build_terminal_observation_candidate(
                    snapshot=snapshot,
                    transform=transform,
                    now_ns=now_ns,
                    last_sample_ns=self._last_return_sim_ns,
                    episode_start_ns=self._episode_path_stamp_ns,
                    robot_frame=bridge.robot_frame
                )

                # Ensure the path and MPC parameter feedback
                # did not change while constructing the
                # terminal observation.
                latest = bridge.get_observation_snapshot()

                if (
                    latest['accepted_path_count']
                    != snapshot['accepted_path_count']
                    or latest['accepted_path_invalidated']
                ):
                    raise ValueError(
                        'Path changed during terminal observation'
                    )

                if (
                    latest['adaptive_ack_seq']
                    != snapshot['adaptive_ack_seq']
                ):
                    raise ValueError(
                        'MPC parameters changed during terminal observation'
                    )

                return candidate, 'fresh_terminal_snapshot'

            except (TransformException, ValueError) as exc:

                last_reason = str(exc)

            time.sleep(0.05)

        return None, last_reason

    def _terminal_step_result(
        self,
        status,
        state,
        action_applied,
        skipped_ticks,
        action
    ):

        self._episode_done = True

        # --------------------------------------------------
        # 1. Try to acquire a fresh terminal observation
        # --------------------------------------------------

        candidate, reason = self._collect_terminal_observation(
            timeout_sec=1.5
        )

        # --------------------------------------------------
        # 2. Apply terminal transition policy
        # --------------------------------------------------

        if candidate is not None:

            terminal = build_terminal_transition(
                status=status,

                last_observation=self._last_observation,
                last_raw_observation=self._last_raw_observation,
                last_sample_ns=self._last_return_sim_ns,

                terminal_observation=candidate['observation'],
                terminal_raw_observation=candidate['raw_observation'],
                terminal_sample_ns=candidate['sample_ns'],
                terminal_verified=True
            )

            terminal_sample_ns = candidate['sample_ns']

            terminal_metadata = candidate['metadata']

        else:

            terminal = build_terminal_transition(
                status=status,

                last_observation=self._last_observation,
                last_raw_observation=self._last_raw_observation,
                last_sample_ns=self._last_return_sim_ns,
                terminal_verified=False
            )

            terminal_sample_ns = None
            terminal_metadata = {}

        # --------------------------------------------------
        # 3. Determine terminal reward interval
        # --------------------------------------------------

        now_sim_ns = (
            self.bridge.get_clock().now().nanoseconds
        )

        if now_sim_ns < self._last_return_sim_ns:

            raise RuntimeError(
                'Simulation clock moved backwards at terminal'
            )

        if candidate is not None:

            reward_dt_s = (
                terminal_sample_ns
                - self._last_return_sim_ns
            ) / 1e9

        else:

            reward_dt_s = (
                now_sim_ns
                - self._last_return_sim_ns
            ) / 1e9


        # --------------------------------------------------
        # 4. Determine terminal reward mode
        # --------------------------------------------------

        if candidate is not None and action_applied:

            reward_mode = 'full'

            reward_result = self._compute_transition_reward(
                current_raw=terminal['raw_observation'],
                current_action=action,
                dt_s=reward_dt_s,
                phase=terminal['phase'],
                reward_mode='full'
            )

        else:

            # Do not calculate geometric progress from
            # reused observations or unapplied actions.
            reward_mode = 'event_only'

            reward_result = self._compute_transition_reward(
                current_raw=terminal['raw_observation'],
                current_action=self._last_action,
                dt_s=reward_dt_s,
                phase=terminal['phase'],
                reward_mode='event_only'
            )

        reward = float(
            reward_result['total']
        )

        # --------------------------------------------------
        # 5. Build Gymnasium info
        # --------------------------------------------------

        info = {
            'phase': terminal['phase'],
            'attempt': self._active_attempt,
            'event': status.get('event'),

            'action_applied': bool(action_applied),

            'step_count': self._step_count,
            'skipped_ticks': int(skipped_ticks),

            'raw_observation':
                terminal['raw_observation'].copy(),

            'terminal_observation_reused':
                terminal['terminal_observation_reused'],

            'terminal_observation_source':
                terminal['terminal_observation_source'],

            'terminal_sample_ns': terminal_sample_ns,

            'terminal_observation_reason': reason,

            'terminal_metadata': terminal_metadata,

            # A transition must not be added to Replay Buffer
            # if the requested action was never applied.
            'replay_eligible': bool(
                terminal['replay_eligible']
                and action_applied
            ),

            'bootstrap_allowed':
                terminal['bootstrap_allowed'],

            'reward_mode': reward_mode,

            'reward_components':
                reward_result['components'],

            'reward_metrics':
                reward_result['metrics'],

            'reward_dt_s': float(reward_dt_s),

            'diagnostic_reward_only': False,

            'episode_status': status,
        }

        # --------------------------------------------------
        # 6. Keep diagnostic reward until Step 4.2.4c
        # --------------------------------------------------

        return (
            terminal['observation'].copy(),
            reward,
            terminal['terminated'],
            terminal['truncated'],
            info
        )

    # ================================================
    # Gymnasium step() - Not yet implemented
    # ================================================

    def step(self, action):

        # ================================================
        # 1. Validate environment and action
        # ================================================

        if self._closed:
            raise RuntimeError('Environment is closed')

        if self._episode_done:
            raise RuntimeError(
                'Episode is not active. Call reset() first.'
            )

        # Map action before interacting with ROS2.
        params = map_action_to_mpc(action)

        # ================================================
        # 2. Initialize step timing
        # ================================================

        bridge = self.bridge

        deadline = time.monotonic() + 25.0

        action_applied = False
        skipped_ticks = 0

        try:

            # ============================================
            # 3. Wait until episode/path is ready
            # ============================================

            while time.monotonic() < deadline:

                status, state = self._step_episode_status()

                if state['stage'] == 'terminal':

                    return self._terminal_step_result(
                        status,
                        state,
                        action_applied,
                        skipped_ticks,
                        action
                    )

                if state['stage'] == 'finalizing':

                    time.sleep(0.05)
                    continue

                snapshot = (
                    bridge.get_observation_snapshot()
                )

                path = snapshot['accepted_path']

                path_valid = (
                    path is not None
                    and not snapshot[
                        'accepted_path_invalidated'
                    ]
                    and len(path.poses) >= 2
                )

                if path_valid:
                    break

                # Replanning: do not send a new action.
                time.sleep(0.05)

            else:

                raise TimeoutError(
                    'Episode/path readiness timeout'
                )

            # ============================================
            # 4. Determine next simulation-time tick
            # ============================================

            now_ns = (
                bridge.get_clock().now().nanoseconds
            )

            if now_ns < self._last_return_sim_ns:

                raise RuntimeError(
                    'Simulation clock moved backwards'
                )

            target_ns = self._next_rl_tick_ns

            # If step() was called too late, schedule
            # the next future tick.
            if now_ns >= target_ns:

                target_ns, skipped = (
                    schedule_next_tick_ns(
                        target_ns - RL_PERIOD_NS,
                        now_ns
                    )
                )

                skipped_ticks += skipped

            # ============================================
            # 5. Send adaptive MPC parameters
            # ============================================

            feedback = bridge.send_adaptive_params_wait(
                params,
                timeout_sec=5.0
            )

            action_applied = True

            np.testing.assert_allclose(
                feedback,
                params,
                rtol=0.0,
                atol=1e-8
            )

            # Snapshot after the action ACK.
            baseline = bridge.get_observation_snapshot()

            ack_seq = baseline['adaptive_ack_seq']

            solver_baseline = (
                bridge.solver_success_count
            )

            ack_sim_ns = (
                bridge.get_clock().now().nanoseconds
            )

            # ACK may arrive after the intended tick.
            # Move to the next future tick if necessary.
            if ack_sim_ns >= target_ns:

                target_ns, skipped = (
                    schedule_next_tick_ns(
                        target_ns - RL_PERIOD_NS,
                        ack_sim_ns
                    )
                )

                skipped_ticks += skipped

            # ============================================
            # 6. Wait for RL tick and MPC solver
            # ============================================

            while time.monotonic() < deadline:

                status, state = self._step_episode_status()

                if state['stage'] == 'terminal':

                    return self._terminal_step_result(
                        status,
                        state,
                        action_applied,
                        skipped_ticks,
                        action
                    )

                if state['stage'] == 'finalizing':

                    time.sleep(0.05)
                    continue

                now_ns = (
                    bridge.get_clock().now().nanoseconds
                )

                if now_ns < ack_sim_ns:

                    raise RuntimeError(
                        'Simulation clock moved backwards'
                    )

                snapshot = (
                    bridge.get_observation_snapshot()
                )

                path = snapshot['accepted_path']

                path_valid = (
                    path is not None
                    and not snapshot[
                        'accepted_path_invalidated'
                    ]
                    and len(path.poses) >= 2
                )

                solver_updated = (
                    bridge.solver_success_count
                    > solver_baseline
                )

                if (
                    now_ns >= target_ns
                    and path_valid
                    and solver_updated
                ):
                    break

                time.sleep(0.05)

            else:

                raise TimeoutError(
                    'RL tick, path or MPC solver timeout'
                )

            # ============================================
            # 7. Collect observation after the RL tick
            # ============================================

            while time.monotonic() < deadline:

                status, state = self._step_episode_status()

                if state['stage'] == 'terminal':

                    return self._terminal_step_result(
                        status,
                        state,
                        action_applied,
                        skipped_ticks,
                        action
                    )

                if state['stage'] == 'finalizing':

                    time.sleep(0.05)
                    continue

                try:

                    observation, raw, metadata = (
                        self._initial_observation(
                            baseline=baseline,
                            ack_baseline=ack_seq - 1,
                            episode_start_ns=(
                                self._episode_path_stamp_ns
                            ),
                            timeout_sec=0.6,
                            expected_params=params,
                            require_new_path=False,
                            min_sensor_stamp_ns=target_ns
                        )
                    )

                except TimeoutError:

                    # Allow short delays during replanning
                    # or sensor updates.
                    continue

                # ========================================
                # 8. Confirm episode did not finish
                # ========================================

                status, state = self._step_episode_status()

                if state['stage'] == 'terminal':

                    return self._terminal_step_result(
                        status,
                        state,
                        action_applied,
                        skipped_ticks,
                        action
                    )

                if state['stage'] == 'finalizing':

                    time.sleep(0.05)
                    continue

                if not self.observation_space.contains(
                    observation
                ):

                    raise RuntimeError(
                        'Observation outside observation_space'
                    )

                # ========================================
                # 9. Update RL scheduling
                # ========================================

                sample_sim_ns = (
                    bridge.get_clock().now().nanoseconds
                )

                if sample_sim_ns < target_ns:

                    raise RuntimeError(
                        'Observation obtained before RL tick'
                    )

                actual_dt_s = (
                    sample_sim_ns
                    - self._last_return_sim_ns
                ) / 1e9

                reward_result = self._compute_transition_reward(
                    current_raw=raw,
                    current_action=action,
                    dt_s=actual_dt_s,
                    phase='EPISODE_RUNNING',
                    reward_mode='full'
                )

                reward = float(
                    reward_result['total']
                )

                next_tick_ns, skipped = (
                    schedule_next_tick_ns(
                        target_ns,
                        sample_sim_ns
                    )
                )

                skipped_ticks += skipped

                self._next_rl_tick_ns = next_tick_ns

                self._last_return_sim_ns = sample_sim_ns

                # Update last valid observation.
                self._last_observation = (
                    observation.copy()
                )

                self._last_raw_observation = raw.copy()

                # Save the action only after a successful step.
                self._last_action = np.asarray(
                    action,
                    dtype=np.float64
                ).copy()

                self._step_count += 1

                # ========================================
                # 10. Gymnasium info
                # ========================================

                info = {
                    'phase': state['phase'],
                    'attempt': self._active_attempt,

                    'action_applied': True,

                    'mpc_params': list(params),

                    'raw_observation': raw.copy(),

                    'sensor_ages_s':
                        metadata['sensor_ages_s'],

                    'sensor_skew_s':
                        metadata['sensor_skew_s'],

                    'tf_age_s':
                        metadata['tf_age_s'],

                    'path_stamp_ns':
                        metadata['path_stamp_ns'],

                    'observation_counts':
                        metadata['observation_counts'],

                    'actual_dt_s': actual_dt_s,

                    'skipped_ticks': int(skipped_ticks),

                    'step_count': self._step_count,

                    'terminal_observation_reused': False,

                    'reward_mode': 'full',

                    'reward_components':
                        reward_result['components'],

                    'reward_metrics':
                        reward_result['metrics'],

                    'diagnostic_reward_only': False,
                }

                # ========================================
                # 11. Return diagnostic Gymnasium step
                # ========================================

                terminated = False
                truncated = False

                return (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info
                )

            raise TimeoutError(
                'Synchronized step observation timeout'
            )

        except Exception:

            # Stop robot if step fails unexpectedly.
            self._episode_done = True

            self._safety_hold()

            raise

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
