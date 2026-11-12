#!/usr/bin/env python3

"""Step 4.2.5d.3 - Real failure transition collection test."""

import time
import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

from rl_mpc_training.utils.sac_transition import (
    build_sac_transition
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer
)

from rl_mpc_training.training.test_env_terminal_cases import (
    DEFAULT_ACTION,
    EXPECTED
)

from rl_mpc_training.training.test_sac_collection_episode import (
    validate_observation
)


BUFFER_CAPACITY = 256
MAX_STEPS = 160
MAX_WALL_TIME_SEC = 240.0

PLANNING_FAILURE_MODE = 'task_failure'


def verify_buffer(
    buffer,
    step_count,
    episode_return,
    scenario,
    terminal_info
):

    print('\n[3] Verify Replay Buffer')

    assert len(buffer) == step_count
    assert buffer.total_added == step_count

    # --------------------------------------------
    # Minibatch
    # --------------------------------------------

    batch_size = min(8, len(buffer))

    batch = buffer.sample(batch_size)

    assert batch['observations'].shape == (
        batch_size, 30
    )

    assert batch['actions'].shape == (
        batch_size, 5
    )

    assert batch['rewards'].shape == (
        batch_size, 1
    )

    assert batch['bootstrap_masks'].shape == (
        batch_size, 1
    )

    print('[PASS] Minibatch sampling')

    # --------------------------------------------
    # Read all stored transitions
    # --------------------------------------------

    data = buffer.sample(len(buffer))

    order = np.argsort(
        data['step_indices']
    )

    steps = data['step_indices'][order]

    obs = data['observations'][order]
    next_obs = data['next_observations'][order]

    rewards = data['rewards'][order, 0]

    masks = data['bootstrap_masks'][order, 0]

    components = data['reward_components'][order]

    phases = [
        data['phases'][int(i)]
        for i in order
    ]

    # --------------------------------------------
    # Indices and observation continuity
    # --------------------------------------------

    np.testing.assert_array_equal(
        steps,
        np.arange(1, step_count + 1)
    )

    np.testing.assert_array_equal(
        data['episode_indices'][order],
        np.zeros(step_count, dtype=np.int64)
    )

    np.testing.assert_array_equal(
        obs[1:],
        next_obs[:-1]
    )

    print('[PASS] Transition indices')
    print('[PASS] Observation continuity')

    # --------------------------------------------
    # Reward consistency
    # --------------------------------------------

    np.testing.assert_allclose(
        rewards,
        components.sum(axis=1),
        atol=2e-5,
        rtol=1e-6
    )

    np.testing.assert_allclose(
        rewards.sum(dtype=np.float64),
        episode_return,
        atol=1e-3,
        rtol=1e-6
    )

    print('[PASS] Stored reward consistency')

    # --------------------------------------------
    # Running and terminal transitions
    # --------------------------------------------

    assert all(
        phase == 'EPISODE_RUNNING'
        for phase in phases[:-1]
    )

    assert np.all(masks[:-1] == 1.0)

    assert phases[-1] == EXPECTED[scenario]['phase']

    terminal_row = int(order[-1])

    expected_mask = (
        1.0 if scenario == 'TIMEOUT'
        else 0.0
    )

    assert masks[-1] == expected_mask

    assert bool(
        data['terminated'][terminal_row]
    ) == EXPECTED[scenario]['terminated']

    assert bool(
        data['truncated'][terminal_row]
    ) == EXPECTED[scenario]['truncated']

    # --------------------------------------------
    # Planning failure special policy
    # --------------------------------------------

    if scenario == 'PLANNING_FAILURE':

        assert (
            terminal_info['replay_eligible']
            is False
        )

        assert data['policy_overrides'][
            terminal_row
        ]

        assert data['learning_terminated'][
            terminal_row
        ]

        assert masks[-1] == 0.0

        # Reused terminal observation.
        np.testing.assert_array_equal(
            obs[-1],
            next_obs[-1]
        )

        print(
            '[PASS] PLANNING_FAILURE '
            'policy override stored'
        )

    elif scenario == 'TIMEOUT':

        assert data['learning_truncated'][
            terminal_row
        ]

        assert masks[-1] == 1.0

        print(
            '[PASS] TIMEOUT bootstrap preserved'
        )

    elif scenario == 'COLLISION':

        assert data['learning_terminated'][
            terminal_row
        ]

        assert masks[-1] == 0.0

        print(
            '[PASS] COLLISION terminal stored'
        )

    print('[PASS] Terminal transition verified')

    # --------------------------------------------
    # Summary
    # --------------------------------------------

    print('\n[4] Test summary')

    print('Scenario:', scenario)

    print('Gymnasium steps:', step_count)

    print('Stored transitions:', len(buffer))

    print('Episode return:', episode_return)

    print(
        'Stored reward sum:',
        float(rewards.sum(dtype=np.float64))
    )

    print(
        'Terminal reward:',
        float(rewards[-1])
    )

    print(
        'Terminal bootstrap mask:',
        float(masks[-1])
    )

    print('[PASS] Failure Replay Buffer verified')


def main():

    print(
        '\nSAC Collection - Failure Integration Test'
    )

    scenario = input(
        '\nType TIMEOUT, PLANNING_FAILURE or COLLISION: '
    ).strip().upper()

    if scenario not in EXPECTED:
        print('Test cancelled')
        return

    env = RLMPCEnv()

    buffer = SACReplayBuffer(
        capacity=BUFFER_CAPACITY,
        seed=123
    )

    try:

        # --------------------------------------------
        # Reset
        # --------------------------------------------

        observation, reset_info = env.reset(
            seed=123
        )

        validate_observation(env, observation)

        assert reset_info['phase'] == 'EPISODE_RUNNING'

        print('[PASS] Environment reset')

        # --------------------------------------------
        # Collect transitions
        # --------------------------------------------

        episode_return = 0.0

        terminal_received = False

        start_wall = time.monotonic()

        for step_index in range(1, MAX_STEPS + 1):

            if (
                time.monotonic() - start_wall
                > MAX_WALL_TIME_SEC
            ):
                raise TimeoutError(
                    'Collection wall-time limit exceeded'
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(DEFAULT_ACTION)

            validate_observation(
                env,
                next_observation
            )

            assert np.isfinite(reward)

            np.testing.assert_allclose(
                reward,
                sum(info['reward_components'].values()),
                atol=1e-6,
                rtol=0.0
            )

            # ----------------------------------------
            # SAC Transition
            # ----------------------------------------

            transition, policy = build_sac_transition(
                observation=observation,
                action=DEFAULT_ACTION,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
                episode_index=0,
                step_index=step_index,
                planning_failure_mode=PLANNING_FAILURE_MODE
            )

            # ----------------------------------------
            # Replay Buffer
            # ----------------------------------------

            if transition is not None:

                buffer.add(transition)

            else:

                print(
                    '[WARNING] Transition skipped:',
                    policy['reason']
                )

            episode_return += float(reward)

            observation = next_observation.copy()

            if step_index % 5 == 0 or terminated or truncated:

                print(
                    f'[STEP {step_index:03d}] '
                    f'phase={info["phase"]} | '
                    f'reward={reward:.3f} | '
                    f'buffer={len(buffer)} | '
                    f'mask={policy["bootstrap_mask"]:.1f}'
                )

            # ----------------------------------------
            # Terminal
            # ----------------------------------------

            if terminated or truncated:

                terminal_received = True

                print('\n[2] Terminal received')

                print('Phase:', info['phase'])
                print('Event:', info['event'])

                print(
                    'Terminal reward:',
                    reward
                )

                print(
                    'Terminal component:',
                    info['reward_components']['terminal']
                )

                print(
                    'Observation reused:',
                    info['terminal_observation_reused']
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
                    'Environment replay eligible:',
                    info['replay_eligible']
                )

                print(
                    'Policy store transition:',
                    policy['store_transition']
                )

                print(
                    'Policy bootstrap mask:',
                    policy['bootstrap_mask']
                )

                print(
                    'Policy override:',
                    policy['policy_override']
                )

                print(
                    'Policy reason:',
                    policy['reason']
                )

                break

        # --------------------------------------------
        # Validate terminal
        # --------------------------------------------

        assert terminal_received

        expected = EXPECTED[scenario]

        assert info['phase'] == expected['phase']

        assert info['event'] == expected['event']

        assert terminated == expected['terminated']

        assert truncated == expected['truncated']

        assert (
            info['reward_components']['terminal']
            == expected['terminal_reward']
        )

        # The test is intended to verify actual
        # insertion of the terminal transition.
        assert info['action_applied'], (
            'Terminal occurred before action was applied. '
            'Repeat this scenario after control is active.'
        )

        if scenario == 'TIMEOUT':

            assert not info['terminal_observation_reused'], (
                'Timeout observation was reused. '
                'Replay Policy correctly skips this case, '
                'but this test requires a fresh observation.'
            )

        if scenario == 'PLANNING_FAILURE':

            assert info['terminal_observation_reused'], (
                'This run did not exercise the reused '
                'planning-failure observation.'
            )

        assert policy['store_transition']

        assert transition is not None

        assert len(buffer) == step_index

        print('[PASS] Terminal reward and flags')
        print('[PASS] Terminal transition stored')

        # --------------------------------------------
        # Verify replay data
        # --------------------------------------------

        verify_buffer(
            buffer=buffer,
            step_count=step_index,
            episode_return=episode_return,
            scenario=scenario,
            terminal_info=info
        )

        # --------------------------------------------
        # Reject steps after terminal
        # --------------------------------------------

        try:

            env.step(DEFAULT_ACTION)

        except RuntimeError as exc:

            assert 'Episode is not active' in str(exc)

            print('[PASS] Step after terminal rejected')

        else:

            raise AssertionError(
                'Step accepted after terminal'
            )

        print(
            '\n[PASS] SAC failure collection '
            'integration test completed'
        )

    finally:

        print('\nClosing environment')

        env.close()


if __name__ == '__main__':
    main()
