#!/usr/bin/env python3

"""Step 4.2.3c.1 - Test episode-state and timing utilities."""

from rl_mpc_training.utils.step_utils import (
    RL_PERIOD_NS,
    classify_episode_status,
    schedule_next_tick_ns
)


def test_episode_states():

    print('\n[1] Testing episode states')

    running = classify_episode_status({
        'phase': 'EPISODE_RUNNING'
    })

    assert running['stage'] == 'running'
    assert running['terminated'] is False
    assert running['truncated'] is False

    print('[PASS] Running episode')

    finalizing = classify_episode_status({
        'phase': 'EP_WAIT_GOAL_HOLD',
        'event': 'GOAL_REACHED',
        'terminated': 'True',
        'truncated': 'False'
    })

    assert finalizing['stage'] == 'finalizing'
    assert finalizing['terminated'] is False
    assert finalizing['truncated'] is False

    print('[PASS] Waiting for final HOLD')

    cases = (
        ('EPISODE_SUCCESS', True, False),
        ('EPISODE_COLLISION', True, False),
        ('EPISODE_TIMEOUT', False, True),
        ('EPISODE_PLANNING_FAILURE', False, True),
    )

    for phase, terminated, truncated in cases:

        result = classify_episode_status({
            'phase': phase,
            'terminated': str(terminated),
            'truncated': str(truncated)
        })

        assert result['stage'] == 'terminal'
        assert result['terminated'] == terminated
        assert result['truncated'] == truncated

        print(f'[PASS] {phase}')

    try:

        classify_episode_status({
            'phase': 'EPISODE_END_ERROR'
        })

    except RuntimeError:

        print('[PASS] Episode error rejected')

    else:

        raise AssertionError(
            'Episode error was not rejected'
        )

    try:

        classify_episode_status({
            'phase': 'EPISODE_SUCCESS',
            'terminated': 'False',
            'truncated': 'True'
        })

    except ValueError:

        print('[PASS] Inconsistent terminal flags rejected')

    else:

        raise AssertionError(
            'Inconsistent terminal flags accepted'
        )


def test_timing():

    print('\n[2] Testing simulation-time scheduling')

    assert RL_PERIOD_NS == 500_000_000

    # Previous target: 1.0 s
    # Current time: 1.2 s
    # Next target: 1.5 s

    next_tick, skipped = schedule_next_tick_ns(
        1_000_000_000,
        1_200_000_000
    )

    assert next_tick == 1_500_000_000
    assert skipped == 0

    print('[PASS] Normal scheduling')

    # Previous target: 1.0 s
    # Current time: 1.8 s
    # The 1.5 s target has been missed.
    # Next target: 2.0 s

    next_tick, skipped = schedule_next_tick_ns(
        1_000_000_000,
        1_800_000_000
    )

    assert next_tick == 2_000_000_000
    assert skipped == 1

    print('[PASS] Missed tick detected')

    # More than one missed target.

    next_tick, skipped = schedule_next_tick_ns(
        1_000_000_000,
        2_800_000_000
    )

    assert next_tick == 3_000_000_000
    assert skipped == 3

    print('[PASS] Multiple missed ticks detected')

    try:

        schedule_next_tick_ns(
            2_000_000_000,
            1_000_000_000
        )

    except RuntimeError:

        print('[PASS] Clock rollback detected')

    else:

        raise AssertionError(
            'Clock rollback was not detected'
        )


def main():

    test_episode_states()

    test_timing()

    print('\n[PASS] Step utilities test completed')


if __name__ == '__main__':
    main()
