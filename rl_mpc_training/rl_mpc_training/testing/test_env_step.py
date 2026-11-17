#!/usr/bin/env python3

"""Step 4.2.3c.2 - Test Gymnasium step integration."""

import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)


TEST_ACTION = np.array(
    [-0.9, -0.9, -0.5, -0.5, 0.0],
    dtype=np.float32
)

DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0],
    dtype=np.float32
)


def validate_step(
    env,
    result,
    action,
    label
):

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = result

    print(f'\n[{label}]')

    print('Observation shape:', observation.shape)
    print('Reward:', reward)

    print('Terminated:', terminated)
    print('Truncated:', truncated)

    print('MPC parameters:', info.get('mpc_params'))

    print('Actual dt:', info.get('actual_dt_s'))
    print('Skipped ticks:', info.get('skipped_ticks'))

    assert env.observation_space.contains(observation)

    assert observation.shape == (30,)
    assert observation.dtype == np.float32

    assert isinstance(reward, float)

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert not terminated
    assert not truncated

    assert info['action_applied'] is True

    assert info['diagnostic_reward_only'] is False

    assert np.isfinite(reward)

    assert info['reward_mode'] == 'full'

    components = info['reward_components']

    np.testing.assert_allclose(
        reward,
        sum(components.values()),
        rtol=0.0,
        atol=1e-6
    )

    assert components['terminal'] == 0.0

    print('[PASS] Reward calculation')

    assert info['terminal_observation_reused'] is False

    expected_params = map_action_to_mpc(action)

    np.testing.assert_allclose(
        info['raw_observation'][25:30],
        expected_params,
        rtol=0.0,
        atol=1e-5
    )

    assert info['actual_dt_s'] > 0.0

    assert info['skipped_ticks'] >= 0

    print('[PASS] Gymnasium step interface')
    print('[PASS] MPC action reflected in observation')


def main():

    env = RLMPCEnv()

    try:

        print('\nGymnasium Step Integration Test')

        answer = input(
            '\nType START to execute test: '
        )

        if answer.strip() != 'START':

            print('Test cancelled')
            return

        # --------------------------------------------
        # 1. Reset
        # --------------------------------------------

        print('\n[1] Reset Environment')

        observation, info = env.reset(seed=123)

        assert env.observation_space.contains(
            observation
        )

        print('[PASS] Gymnasium reset')

        # --------------------------------------------
        # 2. Test action
        # --------------------------------------------

        print('\n[2] Apply test action')

        result = env.step(TEST_ACTION)

        validate_step(
            env,
            result,
            TEST_ACTION,
            'TEST ACTION'
        )

        # --------------------------------------------
        # 3. Restore default
        # --------------------------------------------

        print('\n[3] Restore default action')

        result = env.step(DEFAULT_ACTION)

        validate_step(
            env,
            result,
            DEFAULT_ACTION,
            'DEFAULT ACTION'
        )

        print(
            '\n[PASS] Gymnasium step integration test completed'
        )

    finally:

        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
