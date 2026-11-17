#!/usr/bin/env python3

"""Step 4.1.3a - Test adaptive MPC parameter acknowledgement."""

import threading
import time

import rclpy

from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import SetBool

from rl_mpc_training.ros.ros_bridge import RosBridge


DEFAULT_PARAMS = [
    1.0, 1.0, 1.0, 1.0, 0.5
]

TEST_PARAMS = [
    1.5, 2.0, 2.0, 3.0, 0.35
]


def wait_for_connections(bridge, timeout_sec=45.0):

    start = time.monotonic()

    while rclpy.ok():

        hold_ready = (
            bridge.mpc_hold_client.service_is_ready()
        )

        mpc_subscribers = (
            bridge.adaptive_param_pub.get_subscription_count()
        )

        if hold_ready and mpc_subscribers >= 1:

            print(
                '[PASS] MPC HOLD service and '
                'adaptive subscriber ready',
                flush=True
            )

            return

        if time.monotonic() - start >= timeout_sec:

            raise TimeoutError(
                'MPC interface discovery timeout'
            )

        time.sleep(0.2)

    raise RuntimeError('ROS2 shutdown during discovery')


def activate_hold(bridge):

    request = SetBool.Request()
    request.data = True

    response = bridge.call_service_wait(
        bridge.mpc_hold_client,
        request,
        'mpc_hold',
        timeout_sec=10.0
    )

    if not response.success:

        raise RuntimeError(
            f'MPC HOLD rejected: {response.message}'
        )

    print('[PASS] MPC HOLD confirmed')


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

    hold_confirmed = False

    try:

        print('\nAdaptive MPC Parameter Test')

        wait_for_connections(bridge)

        answer = input(
            '\nType TEST to update MPC parameters: '
        )

        if answer.strip() != 'TEST':

            print('Test cancelled')
            return

        # --------------------------------------------
        # 1. Activate safety HOLD
        # --------------------------------------------

        print('\n[1] Activate MPC HOLD')

        activate_hold(bridge)
        hold_confirmed = True

        # --------------------------------------------
        # 2. Send test parameters
        # --------------------------------------------

        print('\n[2] Send test parameters')

        print('Sent:', TEST_PARAMS)

        feedback = bridge.send_adaptive_params_wait(
            TEST_PARAMS,
            timeout_sec=5.0
        )

        print('[PASS] MPC parameter ACK received')
        print('Feedback:', feedback)

        print('\n[PASS] Adaptive parameter test completed')

    finally:

        # --------------------------------------------
        # 3. Restore default parameters
        # --------------------------------------------

        if hold_confirmed:

            print('\n[3] Restore default parameters')

            try:

                feedback = bridge.send_adaptive_params_wait(
                    DEFAULT_PARAMS,
                    timeout_sec=5.0
                )

                print('[PASS] Default parameters restored')
                print('Feedback:', feedback)

            except Exception as exc:

                print(
                    '[ERROR] Parameter restoration '
                    f'not confirmed: {exc}'
                )

                print(
                    'MPC remains in HOLD. '
                    'Check parameters before next mission.'
                )

        executor.shutdown()

        spin_thread.join(timeout=2.0)

        bridge.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
