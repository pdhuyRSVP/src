#!/usr/bin/env python3

"""Step 4.2.3b - Test action, MPC ACK and new observation."""

import time
import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)

from rl_mpc_training.training.test_adaptive_running import (
    wait_for_solver_success
)


TEST_ACTION = [-0.9, -0.9, -0.5, -0.5, 0.0]

DEFAULT_ACTION = [-1.0, -1.0, -1.0, -1.0, 1.0]

DEFAULT_PARAMS = [1.0, 1.0, 1.0, 1.0, 0.5]


# ========================================================
# Check episode state
# ========================================================

def check_running(env):

    status = env._get_phase(
        env.bridge.episode_status_client
    )

    if status['phase'] != 'EPISODE_RUNNING':

        raise RuntimeError(
            f'Episode is not running: {status}'
        )

    return status


# ========================================================
# Wait for a simulation-time interval
# ========================================================

def wait_sim_interval(env, duration_sec, timeout_sec=15.0):

    bridge = env.bridge

    start_ns = bridge.get_clock().now().nanoseconds

    if start_ns <= 0:
        raise RuntimeError('Simulation clock unavailable')

    target_ns = (
        start_ns + int(duration_sec * 1e9)
    )

    deadline = time.monotonic() + timeout_sec

    previous_ns = start_ns

    while time.monotonic() < deadline:

        current_ns = bridge.get_clock().now().nanoseconds

        if current_ns < previous_ns:

            raise RuntimeError(
                'Simulation clock jumped backwards'
            )

        if current_ns >= target_ns:
            return

        previous_ns = current_ns

        time.sleep(0.05)

    raise TimeoutError(
        'Simulation-time wait exceeded wall timeout'
    )


# ========================================================
# Apply action and collect new observation
# ========================================================

def apply_action_and_observe(
    env,
    action,
    episode_path_stamp_ns
):

    bridge = env.bridge

    check_running(env)

    # ----------------------------------------------------
    # 1. Map SAC action to MPC parameters
    # ----------------------------------------------------

    params = map_action_to_mpc(action)

    print('\nAction:', action)
    print('Mapped MPC parameters:', params)

    # ----------------------------------------------------
    # 2. Send parameters and wait for ACK
    # ----------------------------------------------------

    feedback = bridge.send_adaptive_params_wait(
        params,
        timeout_sec=5.0
    )

    np.testing.assert_allclose(
        feedback,
        params,
        rtol=0.0,
        atol=1e-8
    )

    print('[PASS] MPC ACK received')

    # ----------------------------------------------------
    # 3. Record snapshot after ACK
    # ----------------------------------------------------

    baseline = bridge.get_observation_snapshot()

    ack_seq = baseline['adaptive_ack_seq']

    if ack_seq <= 0:
        raise RuntimeError('Invalid MPC ACK sequence')

    # ----------------------------------------------------
    # 4. Wait for new MPC solver results
    # ----------------------------------------------------

    wait_for_solver_success(
        bridge,
        required_count=3,
        timeout_sec=6.0
    )

    # Allow MPC and robot motion to respond.
    wait_sim_interval(
        env,
        duration_sec=0.5
    )

    check_running(env)

    # ----------------------------------------------------
    # 5. Collect new observation
    # ----------------------------------------------------

    observation, raw, metadata = (
        env._initial_observation(
            baseline=baseline,
            ack_baseline=ack_seq - 1,
            episode_start_ns=episode_path_stamp_ns,
            timeout_sec=5.0,
            expected_params=params,
            require_new_path=False
        )
    )

    # ----------------------------------------------------
    # 6. Validate observation
    # ----------------------------------------------------

    if not env.observation_space.contains(observation):

        raise RuntimeError(
            'Observation outside observation_space'
        )

    np.testing.assert_allclose(
        raw[25:30],
        params,
        rtol=0.0,
        atol=1e-5
    )

    # Check that logarithmic normalization
    # is consistent with SAC action mapping.
    expected_log = (
        np.asarray(action[2:4], dtype=np.float64)
        + 1.0
    ) / 2.0

    np.testing.assert_allclose(
        observation[27:29],
        expected_log,
        rtol=0.0,
        atol=1e-6
    )

    # ----------------------------------------------------
    # 7. Check episode again
    # ----------------------------------------------------

    check_running(env)

    print('[PASS] New observation received')

    print('Observation shape:', observation.shape)

    print(
        'Current MPC parameters:',
        raw[25:30].tolist()
    )

    print(
        'Normalized alpha_Sv, alpha_Sw:',
        observation[27:29].tolist()
    )

    print(
        'Robot velocity:',
        float(raw[22]),
        'm/s'
    )

    print(
        'Goal distance:',
        float(raw[4]),
        'm'
    )

    print(
        'Sensor skew:',
        metadata['sensor_skew_s'],
        's'
    )

    return observation, raw, metadata


# ========================================================
# Main
# ========================================================

def main():

    env = RLMPCEnv()

    reset_completed = False

    try:

        print('\nSAC Action-Observation Integration Test')

        answer = input(
            '\nType START to execute test: '
        )

        if answer.strip() != 'START':

            print('Test cancelled')
            return

        # ------------------------------------------------
        # 1. Reset environment
        # ------------------------------------------------

        print('\n[1] Reset Environment')

        observation, info = env.reset(seed=123)

        reset_completed = True

        print('[PASS] Initial observation received')

        episode_path_stamp_ns = info['path_stamp_ns']

        # ------------------------------------------------
        # 2. Allow robot to start moving
        # ------------------------------------------------

        print('\n[2] Wait for robot motion')

        wait_sim_interval(
            env,
            duration_sec=2.0
        )

        check_running(env)

        # ------------------------------------------------
        # 3. Apply test action
        # ------------------------------------------------

        print('\n[3] Apply test SAC action')

        _, raw_test, _ = apply_action_and_observe(
            env,
            TEST_ACTION,
            episode_path_stamp_ns
        )

        print('[PASS] Test action reflected in observation')

        # ------------------------------------------------
        # 4. Restore default MPC action
        # ------------------------------------------------

        print('\n[4] Restore default MPC action')

        _, raw_default, _ = apply_action_and_observe(
            env,
            DEFAULT_ACTION,
            episode_path_stamp_ns
        )

        np.testing.assert_allclose(
            raw_default[25:30],
            DEFAULT_PARAMS,
            rtol=0.0,
            atol=1e-6
        )

        print('[PASS] Default MPC parameters restored')

        # ------------------------------------------------
        # 5. Finish
        # ------------------------------------------------

        print('\n[PASS] Action-observation integration test completed')

    finally:

        # Restore defaults if test parameters were
        # already applied but an intermediate check failed.
        if reset_completed:

            try:

                feedback = env.bridge.send_adaptive_params_wait(
                    DEFAULT_PARAMS,
                    timeout_sec=5.0
                )

                print(
                    '[CLEANUP] Default MPC parameters:',
                    feedback
                )

            except Exception as exc:

                print(
                    '[WARNING] MPC default restore unconfirmed:',
                    exc
                )

        print('\nClosing environment')

        env.close()


if __name__ == '__main__':
    main()
