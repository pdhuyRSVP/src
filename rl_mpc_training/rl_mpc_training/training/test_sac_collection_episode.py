#!/usr/bin/env python3

"""Step 4.2.5d.1 - Collect one real Gymnasium episode."""

import time

import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

from rl_mpc_training.utils.sac_transition import (
    build_sac_transition
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer
)

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)


# ======================================================
# Test configuration
# ======================================================

DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0],
    dtype=np.float32
)

TEST_ACTION = np.array(
    [-0.9, -0.9, -0.5, -0.5, 0.0],
    dtype=np.float32
)

MAX_STEPS = 160
MAX_WALL_TIME_SEC = 240.0

BUFFER_CAPACITY = 256
MINIBATCH_SIZE = 16


def select_action(step_index):

    if 6 <= step_index <= 10:
        return TEST_ACTION.copy(), 'TEST'

    return DEFAULT_ACTION.copy(), 'DEFAULT'


def validate_observation(env, observation):

    assert observation.shape == (30,)
    assert observation.dtype == np.float32

    assert env.observation_space.contains(
        observation
    )


# ======================================================
# Verify collected replay data
# ======================================================

def verify_replay_buffer(
    buffer,
    expected_steps,
    episode_return
):

    print('\n[4] Verify Replay Buffer')

    assert len(buffer) == expected_steps

    assert buffer.total_added == expected_steps

    assert buffer.can_sample(
        MINIBATCH_SIZE
    )

    # --------------------------------------------------
    # 1. Sample a regular minibatch
    # --------------------------------------------------

    batch = buffer.sample(
        MINIBATCH_SIZE
    )

    assert batch['observations'].shape == (
        MINIBATCH_SIZE, 30
    )

    assert batch['actions'].shape == (
        MINIBATCH_SIZE, 5
    )

    assert batch['rewards'].shape == (
        MINIBATCH_SIZE, 1
    )

    assert batch['next_observations'].shape == (
        MINIBATCH_SIZE, 30
    )

    assert batch['bootstrap_masks'].shape == (
        MINIBATCH_SIZE, 1
    )

    assert len(set(
        batch['indices'].tolist()
    )) == MINIBATCH_SIZE

    assert np.isfinite(
        batch['rewards']
    ).all()

    print('[PASS] Minibatch sampling')

    # --------------------------------------------------
    # 2. Read every stored transition
    # --------------------------------------------------

    full_batch = buffer.sample(
        len(buffer)
    )

    order = np.argsort(
        full_batch['step_indices']
    )

    step_indices = full_batch[
        'step_indices'
    ][order]

    observations = full_batch[
        'observations'
    ][order]

    actions = full_batch[
        'actions'
    ][order]

    next_observations = full_batch[
        'next_observations'
    ][order]

    rewards = full_batch[
        'rewards'
    ][order, 0]

    masks = full_batch[
        'bootstrap_masks'
    ][order, 0]

    components = full_batch[
        'reward_components'
    ][order]

    phases = [
        full_batch['phases'][int(i)]
        for i in order
    ]

    terminated = full_batch[
        'terminated'
    ][order]

    truncated = full_batch[
        'truncated'
    ][order]

    learning_terminated = full_batch[
        'learning_terminated'
    ][order]

    episode_indices = full_batch[
        'episode_indices'
    ][order]

    # --------------------------------------------------
    # 3. Check step indexing
    # --------------------------------------------------

    np.testing.assert_array_equal(
        step_indices,
        np.arange(
            1, expected_steps + 1
        )
    )

    np.testing.assert_array_equal(
        episode_indices,
        np.zeros(
            expected_steps,
            dtype=np.int64
        )
    )

    print('[PASS] Episode and step indices')

    # --------------------------------------------------
    # 4. Check observation continuity
    # --------------------------------------------------

    np.testing.assert_allclose(
        observations[1:],
        next_observations[:-1],
        atol=0.0,
        rtol=0.0
    )

    print('[PASS] Observation continuity')

    # --------------------------------------------------
    # 5. Check applied actions
    # --------------------------------------------------

    for i, step_index in enumerate(step_indices):

        expected_action, _ = select_action(
            int(step_index)
        )

        np.testing.assert_allclose(
            actions[i],
            expected_action,
            atol=0.0,
            rtol=0.0
        )

    print('[PASS] Stored actions')

    # --------------------------------------------------
    # 6. Check rewards
    # --------------------------------------------------

    np.testing.assert_allclose(
        rewards,
        components.sum(axis=1),
        atol=1e-5,
        rtol=1e-6
    )

    np.testing.assert_allclose(
        rewards.sum(dtype=np.float64),
        episode_return,
        atol=1e-4,
        rtol=1e-6
    )

    print('[PASS] Stored rewards')

    # --------------------------------------------------
    # 7. Check bootstrap masks
    # --------------------------------------------------

    assert np.all(
        masks[:-1] == 1.0
    )

    assert masks[-1] == 0.0

    assert all(
        phase == 'EPISODE_RUNNING'
        for phase in phases[:-1]
    )

    assert phases[-1] == 'EPISODE_SUCCESS'

    assert not np.any(
        terminated[:-1]
    )

    assert not np.any(
        truncated[:-1]
    )

    assert terminated[-1]
    assert not truncated[-1]

    assert learning_terminated[-1]

    print('[PASS] Bootstrap masks and terminal flags')

    # --------------------------------------------------
    # 8. Report
    # --------------------------------------------------

    print('\n[5] Replay Buffer statistics')

    print('Stored transitions:', len(buffer))

    print('Total added:', buffer.total_added)

    print('Buffer capacity:', buffer.capacity)

    print('Episode return:', episode_return)

    print(
        'Stored reward sum:',
        float(rewards.sum(dtype=np.float64))
    )

    print(
        'Mean stored reward:',
        float(np.mean(rewards))
    )

    print(
        'Terminal reward:',
        float(rewards[-1])
    )

    print(
        'Terminal bootstrap mask:',
        float(masks[-1])
    )

    print(
        'Terminal phase:',
        phases[-1]
    )

    print('[PASS] Replay Buffer verified')


# ======================================================
# Main integration test
# ======================================================

def main():

    env = RLMPCEnv()

    buffer = SACReplayBuffer(
        capacity=BUFFER_CAPACITY,
        seed=123
    )

    try:

        print('\nSAC Collection - Single Episode Test')

        scenario = input(
            '\nType CLEAR or OBSTACLE: '
        ).strip().upper()

        if scenario not in ('CLEAR', 'OBSTACLE'):
            print('Test cancelled')
            return

        # ----------------------------------------------
        # 1. Reset Gymnasium
        # ----------------------------------------------

        print('\n[1] Reset Environment')

        observation, reset_info = env.reset(
            seed=123
        )

        validate_observation(
            env,
            observation
        )

        print('[PASS] Gymnasium reset')

        last_path_count = int(
            reset_info['observation_counts']['accepted_path_count']
        )

        path_changes = 0

        # ----------------------------------------------
        # 2. Collect transitions
        # ----------------------------------------------

        print('\n[2] Collect transitions')

        episode_return = 0.0

        stored_count = 0
        skipped_count = 0

        terminal_received = False

        wall_start = time.monotonic()

        for step_index in range(
            1, MAX_STEPS + 1
        ):

            if (
                time.monotonic() - wall_start
                > MAX_WALL_TIME_SEC
            ):
                raise TimeoutError(
                    'Collection exceeded wall-time limit'
                )

            action, action_label = select_action(
                step_index
            )

            # ------------------------------------------
            # Gymnasium step
            # ------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            validate_observation(
                env,
                next_observation
            )

            assert np.isfinite(reward)

            # ------------------------------------------
            # Build SAC transition
            # ------------------------------------------

            transition, policy = (
                build_sac_transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    episode_index=0,
                    step_index=step_index,
                    planning_failure_mode='task_failure'
                )
            )

            # ------------------------------------------
            # Store transition
            # ------------------------------------------

            if transition is not None:

                buffer.add(transition)

                stored_count += 1

                if not (terminated or truncated):

                    assert (
                        transition.bootstrap_mask
                        == 1.0
                    )

            else:

                skipped_count += 1

                print(
                    '[WARNING] Transition skipped:',
                    policy['reason']
                )

            episode_return += float(reward)

            # ------------------------------------------
            # Check MPC action feedback
            # ------------------------------------------

            if not (terminated or truncated):

                expected_params = (
                    map_action_to_mpc(action)
                )

                np.testing.assert_allclose(
                    info['raw_observation'][25:30],
                    expected_params,
                    atol=1e-5,
                    rtol=0.0
                )

                current_path_count = int(
                    info['observation_counts']['accepted_path_count']
                )

                if current_path_count != last_path_count:

                    print(
                        '[INFO] Accepted Path count changed:',
                        last_path_count,
                        '->',
                        current_path_count
                    )

                    path_changes += 1

                    last_path_count = current_path_count

            # ------------------------------------------
            # Log
            # ------------------------------------------

            if (
                step_index % 5 == 0
                or terminated
                or truncated
            ):

                distance = float(
                    info['raw_observation'][4]
                )

                print(
                    f'[STEP {step_index:03d}] '
                    f'action={action_label} | '
                    f'd_goal={distance:.3f}m | '
                    f'r={reward:.3f} | '
                    f'buffer={len(buffer)} | '
                    f'mask={policy["bootstrap_mask"]:.1f}'
                )

            # ------------------------------------------
            # Prepare next transition
            # ------------------------------------------

            observation = (
                next_observation.copy()
            )

            # ------------------------------------------
            # Terminal handling
            # ------------------------------------------

            if terminated or truncated:

                terminal_received = True

                print('\n[3] Terminal received')

                print(
                    'Phase:',
                    info['phase']
                )

                print(
                    'Terminated:',
                    terminated
                )

                print(
                    'Truncated:',
                    truncated
                )

                print(
                    'Terminal reward:',
                    reward
                )

                print(
                    'Terminal observation reused:',
                    info['terminal_observation_reused']
                )

                print(
                    'Replay policy:',
                    policy['reason']
                )

                print(
                    'Store transition:',
                    policy['store_transition']
                )

                print(
                    'Bootstrap mask:',
                    policy['bootstrap_mask']
                )

                break

        # ----------------------------------------------
        # 3. Validate episode
        # ----------------------------------------------

        assert terminal_received, (
            'Episode did not reach terminal'
        )

        assert info['phase'] == 'EPISODE_SUCCESS'

        assert terminated is True
        assert truncated is False

        assert (
            info['reward_components']['terminal']
            == 25.0
        )

        assert skipped_count == 0, (
            f'Unexpected skipped transitions: '
            f'{skipped_count}'
        )

        assert stored_count == step_index

        assert len(buffer) == stored_count

        print(
            'Accepted Path changes:',
            path_changes
        )

        if scenario == 'OBSTACLE':

            assert path_changes > 0, (
                'No Accepted Path update observed '
                'in OBSTACLE scenario'
            )

            print('[PASS] Replanning observed')

        print('[PASS] Episode transitions collected')

        # ----------------------------------------------
        # 4. Verify replay data
        # ----------------------------------------------

        verify_replay_buffer(
            buffer,
            expected_steps=stored_count,
            episode_return=episode_return
        )

        print(
            '\n[PASS] Single-episode '
            'SAC collection test completed'
        )

    finally:

        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
