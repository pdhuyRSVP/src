#!/usr/bin/env python3

"""Step 4.1.3b - Test adaptive MPC updates during an episode."""

import threading
import time

import rclpy

from rclpy.executors import SingleThreadedExecutor

from rl_mpc_training.ros.ros_bridge import RosBridge

from rl_mpc_training.training.test_bridge_sequence import (
    wait_for_required_services,
    call_trigger,
    check_ack,
    read_phase,
    wait_for_phase,
    request_safety_hold,
    TERMINAL_PHASES,
    ERROR_PHASES
)


DEFAULT_PARAMS = [1.0, 1.0, 1.0, 1.0, 0.5]

TEST_PARAMS = [1.5, 2.0, 2.0, 3.0, 0.1]


# ========================================================
# Verify new successful solver results
# ========================================================

def wait_for_solver_success(
    bridge,
    required_count=3,
    timeout_sec=5.0
):

    initial_count = bridge.solver_success_count

    start = time.monotonic()

    while time.monotonic() - start < timeout_sec:

        new_count = (
            bridge.solver_success_count - initial_count
        )

        if new_count >= required_count:

            print(
                f'[PASS] {new_count} new successful '
                'MPC solver results',
                flush=True
            )

            return

        time.sleep(0.05)

    raise TimeoutError(
        'No sufficient new successful MPC solver results'
    )


# ========================================================
# Execute one episode with online MPC updates
# ========================================================

def run_test(bridge):

    # ----------------------------------------------------
    # 1. Reset control pipeline
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # 2. Reset localization
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # 3. Start episode
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # 4. Monitor episode and update parameters
    # ----------------------------------------------------

    print('\n[4] Monitor and update MPC')

    start_sim_ns = (
        bridge.get_clock().now().nanoseconds
    )

    if start_sim_ns <= 0:
        raise RuntimeError('Simulation clock unavailable')

    start_wall = time.monotonic()

    action_sent = False
    restored = False

    while time.monotonic() - start_wall < 240.0:

        status = read_phase(
            bridge,
            bridge.episode_status_client,
            'episode_status'
        )

        phase = status['phase']

        # --------------------------------------------
        # Terminal event
        # --------------------------------------------

        if phase in TERMINAL_PHASES:

            print('\n[INFO] Episode terminal received')
            print(status)

            if not action_sent or not restored:
                raise RuntimeError(
                    'Episode ended before both parameter '
                    'updates were completed'
                )

            print('\n[PASS] Online adaptive MPC test completed')
            return status

        if phase in ERROR_PHASES:

            raise RuntimeError(
                f'Episode Manager error: {status}'
            )

        if phase == 'EP_WAIT_GOAL_HOLD':

            time.sleep(0.2)
            continue

        if phase != 'EPISODE_RUNNING':

            raise RuntimeError(
                f'Unexpected phase: {phase}'
            )

        # --------------------------------------------
        # Simulation elapsed time
        # --------------------------------------------

        now_sim_ns = (
            bridge.get_clock().now().nanoseconds
        )

        sim_elapsed = (
            now_sim_ns - start_sim_ns
        ) / 1e9

        # --------------------------------------------
        # Update after 4 simulation seconds
        # --------------------------------------------

        if sim_elapsed >= 4.0 and not action_sent:

            print(
                '\n[5] Update adaptive MPC parameters'
            )

            print('Sent:', TEST_PARAMS)

            # Mark the action as possibly applied before
            # publishing, so cleanup also runs on ACK timeout.
            action_sent = True

            feedback = bridge.send_adaptive_params_wait(
                TEST_PARAMS,
                timeout_sec=5.0
            )

            print('[PASS] Adaptive MPC ACK received')
            print('Feedback:', feedback)

            wait_for_solver_success(bridge)

        # --------------------------------------------
        # Restore after 9 simulation seconds
        # --------------------------------------------

        if (
            sim_elapsed >= 9.0
            and action_sent
            and not restored
        ):

            print(
                '\n[6] Restore default MPC parameters'
            )

            print('Sent:', DEFAULT_PARAMS)

            feedback = bridge.send_adaptive_params_wait(
                DEFAULT_PARAMS,
                timeout_sec=5.0
            )

            restored = True

            print('[PASS] Default parameter ACK received')
            print('Feedback:', feedback)

            wait_for_solver_success(bridge)

        time.sleep(0.2)

    raise TimeoutError(
        'Episode monitoring exceeded 240 seconds'
    )


# ========================================================
# Main
# ========================================================

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

    sequence_started = False

    try:

        print('\nOnline Adaptive MPC Test')

        wait_for_required_services(
            bridge,
            timeout_sec=45.0
        )

        initial_status = read_phase(
            bridge,
            bridge.episode_status_client,
            'episode_status'
        )

        print('Initial status:', initial_status)

        initial_phase = initial_status['phase']

        if initial_phase not in (
            'IDLE',
            *TERMINAL_PHASES
        ):

            raise RuntimeError(
                f'Cannot start test from phase={initial_phase}'
            )

        answer = input(
            '\nType START to execute test: '
        )

        if answer.strip() != 'START':

            print('Test cancelled')
            return

        sequence_started = True

        result = run_test(bridge)

        print('\nTEST COMPLETED')
        print('Final phase:', result['phase'])

    except (Exception, KeyboardInterrupt):

        if sequence_started:
            request_safety_hold(bridge)

        raise

    finally:

        # Restore defaults even if a parameter ACK or the
        # mission fails after the test begins.
        if sequence_started:

            try:

                feedback = bridge.send_adaptive_params_wait(
                    DEFAULT_PARAMS,
                    timeout_sec=5.0
                )

                print(
                    '[CLEANUP] Default parameters confirmed:',
                    feedback
                )

            except Exception as exc:

                print(
                    '[WARNING] Default parameters not confirmed:',
                    exc
                )

                print(
                    '[WARNING] Verify MPC state before '
                    'starting another episode.'
                )

        executor.shutdown()

        spin_thread.join(timeout=2.0)

        bridge.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
