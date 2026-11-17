#!/usr/bin/env python3

"""Step 4.1.2b - Test ROS Bridge episode service sequence."""

import threading
import time

import rclpy

from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger, SetBool

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

# ========================================================
# Wait for ROS2 service discovery
# ========================================================

def wait_for_required_services(bridge, timeout_sec=45.0):

    start = time.monotonic()

    last_missing = None

    while rclpy.ok():

        missing = [
            name
            for name, client in bridge.required_clients.items()
            if not client.service_is_ready()
        ]

        if not missing:

            print(
                '[PASS] All required ROS2 services ready '
                f'({len(bridge.required_clients)}/'
                f'{len(bridge.required_clients)})',
                flush=True
            )

            return

        if missing != last_missing:

            print(
                f'[WAIT] Missing services: {missing}',
                flush=True
            )

            last_missing = missing

        elapsed = time.monotonic() - start

        if elapsed >= timeout_sec:

            raise TimeoutError(
                f'Service discovery timeout after '
                f'{timeout_sec:.1f} s. '
                f'Missing: {missing}'
            )

        time.sleep(0.2)

    raise RuntimeError('ROS2 shutdown during service discovery')

def call_trigger(bridge, client, name):

    response = bridge.call_service_wait(
        client,
        Trigger.Request(),
        name,
        timeout_sec=10.0
    )

    return response


def check_ack(response, name):

    if not response.success:
        raise RuntimeError(
            f'{name} rejected: {response.message}'
        )


def read_phase(bridge, client, name):

    response = call_trigger(bridge, client, name)

    status = bridge.parse_service_message(
        response.message
    )

    phase = status.get('phase')

    if not phase:
        raise RuntimeError(
            f'{name}: missing phase field'
        )

    return status


def wait_for_phase(
    bridge,
    client,
    expected_phase,
    timeout_sec
):

    start = time.monotonic()

    while time.monotonic() - start < timeout_sec:

        status = read_phase(
            bridge,
            client,
            'status_poll'
        )

        phase = status['phase']

        if phase == expected_phase:

            print(f'[PASS] {expected_phase}')
            return status

        if phase in ERROR_PHASES:
            raise RuntimeError(
                f'Episode Manager error: {status}'
            )

        if phase in TERMINAL_PHASES:
            raise RuntimeError(
                f'Unexpected terminal phase: {status}'
            )

        time.sleep(0.3)

    raise TimeoutError(
        f'Timeout waiting for {expected_phase}'
    )


def run_episode(bridge):

    # ----------------------------------------
    # 1. Reset control pipeline
    # ----------------------------------------

    print('\n[1] Reset Control Pipeline')

    response = call_trigger(
        bridge,
        bridge.reset_control_client,
        'reset_control_pipeline'
    )

    check_ack(response, 'reset_control_pipeline')

    wait_for_phase(
        bridge,
        bridge.reset_status_client,
        'RESET_READY',
        30.0
    )

    # ----------------------------------------
    # 2. Reset localization
    # ----------------------------------------

    print('\n[2] Reset Localization')

    response = call_trigger(
        bridge,
        bridge.reset_localization_client,
        'reset_localization'
    )

    check_ack(response, 'reset_localization')

    wait_for_phase(
        bridge,
        bridge.reset_status_client,
        'LOCALIZATION_READY',
        45.0
    )

    # ----------------------------------------
    # 3. Start episode
    # ----------------------------------------

    print('\n[3] Start Episode')

    response = call_trigger(
        bridge,
        bridge.start_episode_client,
        'start_episode'
    )

    check_ack(response, 'start_episode')

    wait_for_phase(
        bridge,
        bridge.episode_status_client,
        'EPISODE_RUNNING',
        50.0
    )

    # ----------------------------------------
    # 4. Monitor terminal event
    # ----------------------------------------

    print('\n[4] Monitor Episode')

    start = time.monotonic()

    while time.monotonic() - start < 240.0:

        status = read_phase(
            bridge,
            bridge.episode_status_client,
            'episode_status'
        )

        phase = status['phase']

        if phase in TERMINAL_PHASES:

            print('\n[PASS] Episode terminal received')
            print(status)

            return status

        if phase in ERROR_PHASES:
            raise RuntimeError(
                f'Episode failed: {status}'
            )

        if phase not in (
            'EPISODE_RUNNING',
            'EP_WAIT_GOAL_HOLD'
        ):
            raise RuntimeError(
                f'Unexpected running phase: {phase}'
            )

        time.sleep(0.5)

    raise TimeoutError(
        'Episode monitoring exceeded 240 seconds'
    )


def request_safety_hold(bridge):

    print('[SAFETY] Requesting MPC HOLD')

    request = SetBool.Request()
    request.data = True

    try:

        response = bridge.call_service_wait(
            bridge.mpc_hold_client,
            request,
            'safety_hold',
            timeout_sec=10.0
        )

        if response.success:
            print('[SAFETY] MPC HOLD confirmed')
        else:
            print('[SAFETY] MPC HOLD rejected')

    except Exception as exc:

        print(f'[SAFETY] HOLD unconfirmed: {exc}')
        print('[SAFETY] Manual MPC state check required')


def main():

    rclpy.init()

    bridge = RosBridge()

    executor = SingleThreadedExecutor()
    executor.add_node(bridge)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )

    spin_thread.start()

    try:

        print('\nROS Bridge service test')

        # ----------------------------------------
        # Wait for ROS2 service discovery
        # ----------------------------------------

        print('\n[0] Waiting for required ROS2 services')

        wait_for_required_services(
            bridge,
            timeout_sec=45.0
        )

        # ----------------------------------------
        # Read initial Episode Manager status
        # ----------------------------------------

        status = read_phase(
            bridge,
            bridge.episode_status_client,
            'episode_status'
        )

        print('Initial manager status:', status)

        answer = input(
            '\nType START to execute one episode: '
        )

        if answer.strip() != 'START':
            print('Test cancelled - no reset/start sent')
            return

        try:

            result = run_episode(bridge)

            print('\nTEST COMPLETED')
            print('Final phase:', result['phase'])

        except (Exception, KeyboardInterrupt):

            request_safety_hold(bridge)
            raise

    finally:

        executor.shutdown()
        spin_thread.join(timeout=2.0)

        bridge.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
