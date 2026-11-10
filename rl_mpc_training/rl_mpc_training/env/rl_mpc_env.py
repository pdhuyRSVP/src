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
            low=-np.inf,
            high=np.inf,
            shape=(5,),
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
    # Get initial observation
    # ================================================

    def _initial_observation(
        self,
        baseline_odom_count,
        timeout_sec=5.0
    ):

        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:

            odom = self.bridge.odom

            if (
                odom is None
                or self.bridge.odom_count <= baseline_odom_count
            ):

                time.sleep(0.05)
                continue

            try:

                transform = (
                    self.bridge.tf_buffer.lookup_transform(
                        self.bridge.global_frame,
                        self.bridge.robot_frame,
                        Time()
                    )
                )

            except TransformException:

                time.sleep(0.05)
                continue

            q = transform.transform.rotation

            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            observation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    yaw,
                    odom.twist.twist.linear.x,
                    odom.twist.twist.angular.z
                ],
                dtype=np.float32
            )

            if np.isfinite(observation).all():

                return observation

            time.sleep(0.05)

        raise TimeoutError(
            'Initial odometry or TF unavailable'
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

            baseline = self.bridge.odom_count

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

            observation = self._initial_observation(
                baseline
            )

            info = {
                'phase': status['phase'],
                'attempt': int(status['attempt']),
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
