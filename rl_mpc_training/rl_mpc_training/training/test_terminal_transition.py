#!/usr/bin/env python3

"""Step 4.2.4b.1 - Test terminal transition policy."""

import numpy as np

from rl_mpc_training.utils.terminal_transition import (
    build_terminal_transition
)


TERMINAL_CASES = (
    ('EPISODE_SUCCESS', True, False),
    ('EPISODE_COLLISION', True, False),
    ('EPISODE_TIMEOUT', False, True),
    ('EPISODE_PLANNING_FAILURE', False, True),
)


def make_status(phase, terminated, truncated):

    return {
        'phase': phase,
        'attempt': '1',
        'terminated': str(terminated),
        'truncated': str(truncated),
    }


def make_observation(value=0.0):

    normalized = np.zeros(
        30,
        dtype=np.float32
    )

    raw = np.zeros(
        30,
        dtype=np.float32
    )

    normalized[4] = value
    raw[4] = value * 20.0

    return normalized, raw


def main():

    print('\n[1] Reused terminal observations')

    last_obs, last_raw = make_observation(0.25)

    for phase, terminated, truncated in TERMINAL_CASES:

        result = build_terminal_transition(
            status=make_status(
                phase,
                terminated,
                truncated
            ),
            last_observation=last_obs,
            last_raw_observation=last_raw,
            last_sample_ns=1_000_000_000
        )

        assert result['terminated'] == terminated
        assert result['truncated'] == truncated

        assert result[
            'terminal_observation_reused'
        ] is True

        assert result['reward_mode'] == 'event_only'

        assert result['bootstrap_allowed'] is False

        # A reused terminal observation is only
        # replay-eligible for true terminations.
        assert (
            result['replay_eligible'] == terminated
        )

        print(f'[PASS] Reused {phase}')

    # ----------------------------------------------------
    # 2. Verified terminal observation
    # ----------------------------------------------------

    print('\n[2] Verified terminal observations')

    terminal_obs, terminal_raw = (
        make_observation(0.10)
    )

    for phase, terminated, truncated in TERMINAL_CASES:

        result = build_terminal_transition(
            status=make_status(
                phase,
                terminated,
                truncated
            ),
            last_observation=last_obs,
            last_raw_observation=last_raw,
            last_sample_ns=1_000_000_000,

            terminal_observation=terminal_obs,
            terminal_raw_observation=terminal_raw,

            terminal_sample_ns=1_500_000_000,
            terminal_verified=True
        )

        assert result['terminated'] == terminated
        assert result['truncated'] == truncated

        assert result[
            'terminal_observation_reused'
        ] is False

        assert result['replay_eligible'] is True

        assert result[
            'bootstrap_allowed'
        ] == truncated

        assert result['reward_mode'] == 'full'

        np.testing.assert_array_equal(
            result['observation'],
            terminal_obs
        )

        print(f'[PASS] Verified {phase}')

    # ----------------------------------------------------
    # 3. Reject unverified terminal data
    # ----------------------------------------------------

    print('\n[3] Invalid terminal data')

    try:

        build_terminal_transition(
            status=make_status(
                'EPISODE_TIMEOUT',
                False,
                True
            ),
            last_observation=last_obs,
            last_raw_observation=last_raw,
            last_sample_ns=1_000_000_000,

            terminal_observation=terminal_obs,
            terminal_raw_observation=terminal_raw,

            terminal_sample_ns=900_000_000,
            terminal_verified=True
        )

    except ValueError:

        print('[PASS] Stale terminal timestamp rejected')

    else:

        raise AssertionError(
            'Stale terminal timestamp accepted'
        )

    try:

        build_terminal_transition(
            status={
                'phase': 'EPISODE_RUNNING'
            },
            last_observation=last_obs,
            last_raw_observation=last_raw,
            last_sample_ns=1_000_000_000
        )

    except ValueError:

        print('[PASS] Non-terminal phase rejected')

    else:

        raise AssertionError(
            'Non-terminal phase accepted'
        )

    try:

        build_terminal_transition(
            status=make_status(
                'EPISODE_SUCCESS',
                True,
                False
            ),
            last_observation=last_obs,
            last_raw_observation=last_raw,
            last_sample_ns=1_000_000_000,

            terminal_observation=terminal_obs,
            terminal_raw_observation=terminal_raw,

            terminal_sample_ns=1_500_000_000,
            terminal_verified=False
        )

    except ValueError:

        print('[PASS] Unverified terminal data rejected')

    else:

        raise AssertionError(
            'Unverified terminal data accepted'
        )

    # ----------------------------------------------------
    # 4. Verify arrays are independent
    # ----------------------------------------------------

    result = build_terminal_transition(
        status=make_status(
            'EPISODE_SUCCESS',
            True,
            False
        ),
        last_observation=last_obs,
        last_raw_observation=last_raw,
        last_sample_ns=1_000_000_000
    )

    result['observation'][0] = 0.75
    result['raw_observation'][0] = 100.0

    assert last_obs[0] == 0.0
    assert last_raw[0] == 0.0

    print('[PASS] Independent terminal observation arrays')

    print(
        '\n[PASS] Terminal transition policy tests completed'
    )


if __name__ == '__main__':
    main()
