#!/usr/bin/env python3

"""Step 4.3d.2a - SAC Action Guard unit tests."""

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import (
    SACAgent
)

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION,
    TEST_ACTION,
    MAX_DELTA,
    SACActionGuard
)


def expect_error(name, callback, exception):

    try:

        callback()

    except exception:

        print(f'[PASS] {name}')

    else:

        raise AssertionError(
            f'Expected {exception.__name__}: {name}'
        )


def main():

    print(
        '\nStep 4.3d.2a - '
        'SAC Action Guard Unit Tests'
    )

    # ==================================================
    # 1. Warmup
    # ==================================================

    print('\n[1] Warmup behavior')

    guard = SACActionGuard(
        warmup_steps=5
    )

    for step in range(1, 6):

        raw = np.zeros(
            5,
            dtype=np.float32
        )

        action = guard.propose(
            raw,
            step
        )

        np.testing.assert_array_equal(
            action,
            DEFAULT_ACTION
        )

        # Not committed before ACK.
        np.testing.assert_array_equal(
            guard.last_applied,
            DEFAULT_ACTION
        )

        guard.confirm(
            action,
            action_applied=True
        )

    assert guard.steps_applied == 5

    print(
        '[PASS] First five actions remain DEFAULT'
    )

    print(
        '[PASS] Action committed only after confirmation'
    )

    # ==================================================
    # 2. Safe envelope and rate limits
    # ==================================================

    print('\n[2] Action clipping and rate limits')

    previous = guard.last_applied.copy()

    for step in range(6, 17):

        raw = np.zeros(
            5,
            dtype=np.float32
        )

        original_raw = raw.copy()

        action = guard.propose(
            raw,
            step
        )

        # Proposed SAC action must not be modified.
        np.testing.assert_array_equal(
            raw,
            original_raw
        )

        assert action.shape == (5,)

        assert action.dtype == np.float32

        assert np.isfinite(action).all()

        assert np.all(
            action >= guard.lower - 1e-7
        )

        assert np.all(
            action <= guard.upper + 1e-7
        )

        delta = np.abs(
            action - previous
        )

        assert np.all(
            delta <= MAX_DELTA + 1e-7
        )

        params = map_action_to_mpc(
            action
        )

        assert len(params) == 5

        assert np.isfinite(
            params
        ).all()

        guard.confirm(
            action,
            action_applied=True
        )

        np.testing.assert_array_equal(
            guard.last_applied,
            action
        )

        previous = action.copy()

    assert not np.array_equal(
        previous,
        DEFAULT_ACTION
    )

    print(
        '[PASS] Actions remain inside guarded envelope'
    )

    print(
        '[PASS] Per-step action changes limited'
    )

    print(
        '[PASS] MPC parameter mapping valid'
    )

    # ==================================================
    # 3. Unapplied action and reset
    # ==================================================

    print('\n[3] Confirmation and reset')

    previous_step = guard.steps_applied

    previous_action = guard.last_applied.copy()

    action = guard.propose(
        TEST_ACTION,
        previous_step + 1
    )

    guard.confirm(
        action,
        action_applied=False
    )

    assert guard.steps_applied == previous_step

    np.testing.assert_array_equal(
        guard.last_applied,
        previous_action
    )

    print(
        '[PASS] Unapplied action does not update Guard'
    )

    # Re-propose at the same step.
    action = guard.propose(
        TEST_ACTION,
        previous_step + 1
    )

    guard.confirm(
        action,
        action_applied=True
    )

    assert guard.steps_applied == previous_step + 1

    guard.reset()

    assert guard.steps_applied == 0

    np.testing.assert_array_equal(
        guard.last_applied,
        DEFAULT_ACTION
    )

    print(
        '[PASS] Reset restores initial Guard state'
    )

    # ==================================================
    # 4. Invalid action handling
    # ==================================================

    print('\n[4] Invalid actions')

    expect_error(
        'Wrong action dimension rejected',
        lambda: guard.propose(
            np.zeros(4),
            1
        ),
        ValueError
    )

    expect_error(
        'Non-finite action rejected',
        lambda: guard.propose(
            np.array([np.nan, 0, 0, 0, 0]),
            1
        ),
        ValueError
    )

    expect_error(
        'Out-of-range action rejected',
        lambda: guard.propose(
            np.array([1.1, 0, 0, 0, 0]),
            1
        ),
        ValueError
    )

    expect_error(
        'Invalid step index rejected',
        lambda: guard.propose(
            np.zeros(5),
            2
        ),
        ValueError
    )

    action = guard.propose(
        np.zeros(5),
        1
    )

    expect_error(
        'Unconfirmed proposal blocked',
        lambda: guard.propose(
            np.zeros(5),
            1
        ),
        RuntimeError
    )

    expect_error(
        'Mismatched ACK action rejected',
        lambda: guard.confirm(
            TEST_ACTION,
            action_applied=True
        ),
        ValueError
    )

    expect_error(
        'Nonboolean ACK rejected',
        lambda: guard.confirm(
            action,
            action_applied=1
        ),
        ValueError
    )

    guard.confirm(
        action,
        action_applied=True
    )

    expect_error(
        'Double confirmation rejected',
        lambda: guard.confirm(
            action,
            action_applied=True
        ),
        RuntimeError
    )

    print(
        '[PASS] Invalid inputs cannot be committed'
    )

    # ==================================================
    # 5. SAC Agent integration
    # ==================================================

    print('\n[5] SAC Agent integration')

    torch.manual_seed(431)
    torch.set_num_threads(1)

    agent = SACAgent(
        obs_dim=30,
        action_dim=5,
        device='cpu'
    )

    guard.reset()

    observation = np.zeros(
        30,
        dtype=np.float32
    )

    for step in range(1, 16):

        raw_action = agent.act(
            observation,
            deterministic=False
        )

        assert raw_action.shape == (5,)

        assert np.isfinite(
            raw_action
        ).all()

        applied_action = guard.propose(
            raw_action,
            step
        )

        assert applied_action.shape == (5,)

        assert np.all(
            applied_action >= guard.lower - 1e-7
        )

        assert np.all(
            applied_action <= guard.upper + 1e-7
        )

        guard.confirm(
            applied_action,
            action_applied=True
        )

    assert guard.steps_applied == 15

    assert agent.update_count == 0

    print(
        '[PASS] SAC actions processed by Action Guard'
    )

    print(
        '[PASS] No SAC training performed'
    )

    print(
        '\n[PASS] SAC Action Guard tests completed'
    )


if __name__ == '__main__':
    main()
