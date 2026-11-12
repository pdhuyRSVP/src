#!/usr/bin/env python3

"""Step 4.2.5c - SAC Replay Buffer unit tests."""

from dataclasses import replace

import numpy as np

from rl_mpc_training.utils.sac_transition import (
    build_sac_transition
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer
)

from rl_mpc_training.training.test_sac_transition import (
    make_case
)


def make_transition(
    step_index,
    phase='EPISODE_RUNNING',
    episode_index=0,
    reused=False,
    mode='task_failure'
):

    inputs = make_case(
        phase=phase,
        reused=reused
    )

    inputs['episode_index'] = episode_index
    inputs['step_index'] = step_index

    transition, policy = build_sac_transition(
        **inputs,
        planning_failure_mode=mode
    )

    return transition, policy


def expect_rejection(name, operation, exception):

    try:
        operation()

    except exception:
        print(f'[PASS] {name} rejected')

    else:
        raise AssertionError(
            f'Expected rejection: {name}'
        )


def main():

    # ==================================================
    # 1. Basic storage and minibatch sampling
    # ==================================================

    print('\n[1] Basic storage and minibatch')

    buffer = SACReplayBuffer(
        capacity=8,
        seed=123
    )

    assert len(buffer) == 0
    assert buffer.can_sample(1) is False

    for step in range(1, 6):

        transition, policy = make_transition(
            step_index=step
        )

        assert policy['store_transition']

        buffer.add(transition)

    assert len(buffer) == 5
    assert buffer.total_added == 5
    assert buffer.position == 5

    assert buffer.can_sample(4)

    batch = buffer.sample(4)

    assert batch['observations'].shape == (4, 30)
    assert batch['actions'].shape == (4, 5)
    assert batch['rewards'].shape == (4, 1)

    assert batch['next_observations'].shape == (4, 30)
    assert batch['bootstrap_masks'].shape == (4, 1)

    assert batch['reward_components'].shape == (4, 8)

    assert batch['observations'].dtype == np.float32
    assert batch['actions'].dtype == np.float32
    assert batch['rewards'].dtype == np.float32

    assert len(set(batch['indices'].tolist())) == 4

    np.testing.assert_allclose(
        batch['rewards'][:, 0],
        batch['reward_components'].sum(axis=1),
        atol=1e-5,
        rtol=1e-6
    )

    np.testing.assert_array_equal(
        batch['bootstrap_masks'],
        np.ones((4, 1), dtype=np.float32)
    )

    print('[PASS] Transition storage')
    print('[PASS] Minibatch shapes and dtypes')
    print('[PASS] Uniform sampling without replacement')
    print('[PASS] Reward component consistency')

    # ==================================================
    # 2. Circular overwrite behavior
    # ==================================================

    print('\n[2] Circular overwrite')

    small = SACReplayBuffer(
        capacity=3,
        seed=42
    )

    for step in range(1, 6):

        transition, _ = make_transition(
            step_index=step
        )

        small.add(transition)

    assert len(small) == 3
    assert small.total_added == 5
    assert small.position == 2

    batch = small.sample(3)

    assert sorted(
        batch['step_indices'].tolist()
    ) == [3, 4, 5]

    print('[PASS] Oldest transitions overwritten')

    transition, _ = make_transition(
        step_index=6
    )

    small.add(transition)

    batch = small.sample(3)

    assert sorted(
        batch['step_indices'].tolist()
    ) == [4, 5, 6]

    print('[PASS] Circular overwrite continues correctly')

    # ==================================================
    # 3. Terminal states and bootstrap masks
    # ==================================================

    print('\n[3] Terminal transition storage')

    terminal_buffer = SACReplayBuffer(
        capacity=8,
        seed=7
    )

    cases = (
        ('EPISODE_SUCCESS', False, 0.0),
        ('EPISODE_COLLISION', False, 0.0),
        ('EPISODE_TIMEOUT', False, 1.0),
        ('EPISODE_PLANNING_FAILURE', True, 0.0)
    )

    for step, (phase, reused, expected_mask) in enumerate(
        cases, start=1
    ):

        transition, policy = make_transition(
            step_index=step,
            phase=phase,
            reused=reused
        )

        assert transition is not None

        assert (
            transition.bootstrap_mask
            == expected_mask
        )

        terminal_buffer.add(transition)

    batch = terminal_buffer.sample(4)

    expected_masks = {
        phase: mask
        for phase, reused, mask in cases
    }

    for i, phase in enumerate(batch['phases']):

        actual_mask = float(
            batch['bootstrap_masks'][i, 0]
        )

        assert (
            actual_mask == expected_masks[phase]
        )

    assert (
        'EPISODE_PLANNING_FAILURE'
        in batch['phases']
    )

    planning_index = batch['phases'].index(
        'EPISODE_PLANNING_FAILURE'
    )

    assert batch['learning_terminated'][
        planning_index
    ]

    assert batch['truncated'][
        planning_index
    ]

    assert batch['policy_overrides'][
        planning_index
    ]

    print('[PASS] SUCCESS and COLLISION masks')
    print('[PASS] TIMEOUT bootstrap preserved')
    print('[PASS] PLANNING_FAILURE policy override preserved')

    # ==================================================
    # 4. Ineligible transitions
    # ==================================================

    print('\n[4] Ineligible transitions')

    transition, policy = make_transition(
        step_index=7,
        phase='EPISODE_TIMEOUT',
        reused=True
    )

    assert transition is None
    assert not policy['store_transition']

    transition, policy = make_transition(
        step_index=8,
        phase='EPISODE_PLANNING_FAILURE',
        reused=True,
        mode='external_interruption'
    )

    assert transition is None
    assert not policy['store_transition']

    print('[PASS] TIMEOUT reused observation skipped')
    print('[PASS] External interruption skipped')

    # ==================================================
    # 5. Independent samples and reproducibility
    # ==================================================

    print('\n[5] Sample independence and reproducibility')

    single = SACReplayBuffer(
        capacity=1,
        seed=11
    )

    transition, _ = make_transition(step_index=1)

    single.add(transition)

    batch = single.sample(1)

    original = float(
        batch['observations'][0, 4]
    )

    batch['observations'][0, 4] = 0.99
    batch['actions'][0, 0] = 0.99

    new_batch = single.sample(1)

    assert new_batch['observations'][0, 4] == original
    assert new_batch['actions'][0, 0] == -1.0

    print('[PASS] Sample arrays are independent copies')

    a = SACReplayBuffer(capacity=10, seed=77)
    b = SACReplayBuffer(capacity=10, seed=77)

    for step in range(1, 9):

        transition, _ = make_transition(
            step_index=step
        )

        a.add(transition)
        b.add(transition)

    sample_a = a.sample(5)
    sample_b = b.sample(5)

    np.testing.assert_array_equal(
        sample_a['indices'],
        sample_b['indices']
    )

    print('[PASS] Sampling reproducible with same seed')

    # ==================================================
    # 6. Invalid operations
    # ==================================================

    print('\n[6] Invalid operations')

    expect_rejection(
        'Invalid capacity',
        lambda: SACReplayBuffer(capacity=0),
        ValueError
    )

    expect_rejection(
        'Invalid seed',
        lambda: SACReplayBuffer(capacity=4, seed=-1),
        ValueError
    )

    expect_rejection(
        'Oversized batch',
        lambda: small.sample(4),
        ValueError
    )

    expect_rejection(
        'Zero batch size',
        lambda: small.sample(0),
        ValueError
    )

    expect_rejection(
        'Invalid transition type',
        lambda: small.add(None),
        TypeError
    )

    valid, _ = make_transition(step_index=9)

    bad_mask = replace(
        valid,
        bootstrap_mask=0.5
    )

    before = small.total_added

    expect_rejection(
        'Invalid bootstrap mask',
        lambda: small.add(bad_mask),
        ValueError
    )

    assert small.total_added == before

    bad_obs = replace(
        valid,
        observation=np.zeros(29, dtype=np.float32)
    )

    expect_rejection(
        'Invalid observation dimension',
        lambda: small.add(bad_obs),
        ValueError
    )

    bad_reward = replace(
        valid,
        reward=float('nan')
    )

    expect_rejection(
        'Non-finite reward',
        lambda: small.add(bad_reward),
        ValueError
    )

    print('[PASS] Invalid insertion leaves buffer unchanged')

    # ==================================================
    # 7. Clear
    # ==================================================

    print('\n[7] Clear buffer')

    small.clear()

    assert len(small) == 0
    assert small.position == 0
    assert small.total_added == 0
    assert not small.can_sample(1)

    expect_rejection(
        'Sample from empty buffer',
        lambda: small.sample(1),
        ValueError
    )

    print('[PASS] Buffer cleared successfully')

    print(
        '\n[PASS] SAC Replay Buffer tests completed'
    )


if __name__ == '__main__':
    main()
