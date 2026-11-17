#!/usr/bin/env python3

"""Test Gymnasium terminal cases in ROS2/Gazebo."""

import time
import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0],
    dtype=np.float32
)


EXPECTED = {
    'TIMEOUT': {
        'phase': 'EPISODE_TIMEOUT',
        'event': 'TIMEOUT',
        'terminated': False,
        'truncated': True,
        'terminal_reward': -30.0
    },

    'PLANNING_FAILURE': {
        'phase': 'EPISODE_PLANNING_FAILURE',
        'event': 'PLANNING_FAILURE',
        'terminated': False,
        'truncated': True,
        'terminal_reward': -30.0
    },

    'COLLISION': {
        'phase': 'EPISODE_COLLISION',
        'event': 'COLLISION',
        'terminated': True,
        'truncated': False,
        'terminal_reward': -100.0
    }
}


def main():

    env = RLMPCEnv()

    try:

        print('\nGymnasium Terminal Integration Test')

        scenario = input(
            '\nType TIMEOUT, PLANNING_FAILURE or COLLISION: '
        ).strip().upper()

        if scenario not in EXPECTED:
            print('Test cancelled')
            return

        expected = EXPECTED[scenario]

        # ----------------------------------------
        # Reset
        # ----------------------------------------

        print('\n[1] Reset Environment')

        observation, info = env.reset(seed=123)

        assert env.observation_space.contains(observation)

        print('[PASS] Environment reset')

        # ----------------------------------------
        # Execute episode
        # ----------------------------------------

        print('\n[2] Execute Gymnasium steps')

        episode_return = 0.0
        terminal_received = False

        wall_start = time.monotonic()

        for step_index in range(1, 161):

            if time.monotonic() - wall_start > 180.0:
                raise TimeoutError(
                    'Terminal test exceeded wall-time limit'
                )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(DEFAULT_ACTION)

            assert env.observation_space.contains(
                observation
            )

            assert np.isfinite(reward)

            components = info['reward_components']

            np.testing.assert_allclose(
                reward,
                sum(components.values()),
                atol=1e-6,
                rtol=0
            )

            episode_return += reward

            if terminated or truncated:

                terminal_received = True

                print('\n[3] Terminal received')

                print('Phase:', info['phase'])
                print('Event:', info['event'])

                print('Terminated:', terminated)
                print('Truncated:', truncated)

                print('Terminal reward:', reward)
                print('Reward components:', components)

                print(
                    'Observation reused:',
                    info['terminal_observation_reused']
                )

                print(
                    'Observation reason:',
                    info['terminal_observation_reason']
                )

                print(
                    'Reward mode:',
                    info['reward_mode']
                )

                print(
                    'Action applied:',
                    info['action_applied']
                )

                print(
                    'Replay eligible:',
                    info['replay_eligible']
                )

                print(
                    'Bootstrap allowed:',
                    info['bootstrap_allowed']
                )

                print(
                    'Episode status:',
                    info['episode_status']
                )

                break

            assert info['phase'] == 'EPISODE_RUNNING'
            assert components['terminal'] == 0.0

            if step_index % 5 == 0:

                print(
                    f'[STEP {step_index:03d}] '
                    f'reward={reward:.3f} | '
                    f'd_goal='
                    f'{info["raw_observation"][4]:.3f}m'
                )

        # ----------------------------------------
        # Validate terminal
        # ----------------------------------------

        if not terminal_received:

            raise RuntimeError(
                'No terminal event received'
            )

        assert info['phase'] == expected['phase']

        assert info['event'] == expected['event']

        assert terminated == expected['terminated']

        assert truncated == expected['truncated']

        assert (
            components['terminal']
            == expected['terminal_reward']
        )

        print('[PASS] Terminal phase and flags')
        print('[PASS] Terminal reward component')

        # ----------------------------------------
        # Validate observation/replay policy
        # ----------------------------------------

        reused = bool(
            info['terminal_observation_reused']
        )

        applied = bool(
            info['action_applied']
        )

        expected_mode = (
            'full'
            if not reused and applied
            else 'event_only'
        )

        assert info['reward_mode'] == expected_mode

        expected_replay = (
            applied and (terminated or not reused)
        )

        assert (
            info['replay_eligible']
            == expected_replay
        )

        expected_bootstrap = (
            truncated and not reused
        )

        assert (
            info['bootstrap_allowed']
            == expected_bootstrap
        )

        if expected_mode == 'event_only':

            np.testing.assert_allclose(
                reward,
                expected['terminal_reward'],
                atol=1e-6,
                rtol=0
            )

        print('[PASS] Terminal reward mode')
        print('[PASS] Replay eligibility policy')
        print('[PASS] Bootstrap policy')

        # ----------------------------------------
        # Check episode termination
        # ----------------------------------------

        try:

            env.step(DEFAULT_ACTION)

        except RuntimeError as exc:

            if 'Episode is not active' not in str(exc):
                raise

            print('[PASS] Step after terminal rejected')

        else:

            raise AssertionError(
                'Step accepted after terminal'
            )

        print('\n[4] Test summary')

        print('Scenario:', scenario)
        print('Total steps:', step_index)
        print('Episode return:', episode_return)

        print('\n[PASS] Terminal integration test completed')

    finally:

        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
