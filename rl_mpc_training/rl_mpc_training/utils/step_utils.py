#!/usr/bin/env python3

"""Gymnasium step timing and episode-state utilities."""


RL_PERIOD_NS = 500_000_000


TERMINAL_FLAGS = {
    'EPISODE_SUCCESS': (True, False),
    'EPISODE_COLLISION': (True, False),
    'EPISODE_TIMEOUT': (False, True),
    'EPISODE_PLANNING_FAILURE': (False, True),
}


ERROR_PHASES = {
    'RESET_ERROR',
    'LOCALIZATION_ERROR',
    'EPISODE_ERROR',
    'EPISODE_END_ERROR',
}


def _parse_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        if value.lower() == 'true':
            return True

        if value.lower() == 'false':
            return False

    raise ValueError(
        f'Invalid episode boolean value: {value}'
    )


def classify_episode_status(status):
    """
    Interpret an Episode Manager status response.

    Returns:
        {
            'phase': str,
            'stage': 'running' | 'finalizing' | 'terminal',
            'terminated': bool,
            'truncated': bool
        }
    """

    phase = status.get('phase')

    if phase in ERROR_PHASES:

        raise RuntimeError(
            f'Episode Manager error: {status}'
        )

    if phase == 'EPISODE_RUNNING':

        return {
            'phase': phase,
            'stage': 'running',
            'terminated': False,
            'truncated': False
        }

    if phase == 'EP_WAIT_GOAL_HOLD':

        return {
            'phase': phase,
            'stage': 'finalizing',
            'terminated': False,
            'truncated': False
        }

    if phase not in TERMINAL_FLAGS:

        raise ValueError(
            f'Unexpected episode phase: {phase}'
        )

    if (
        'terminated' not in status
        or 'truncated' not in status
    ):

        raise ValueError(
            'Terminal status missing Gymnasium flags'
        )

    terminated = _parse_bool(status['terminated'])
    truncated = _parse_bool(status['truncated'])

    expected = TERMINAL_FLAGS[phase]

    if (terminated, truncated) != expected:

        raise ValueError(
            f'Inconsistent terminal flags for {phase}: '
            f'terminated={terminated}, truncated={truncated}'
        )

    return {
        'phase': phase,
        'stage': 'terminal',
        'terminated': terminated,
        'truncated': truncated
    }


def schedule_next_tick_ns(
    previous_tick_ns,
    current_ns,
    period_ns=RL_PERIOD_NS
):
    """
    Return the next future simulation-time tick.

    The schedule is anchored to the previous tick,
    rather than the completion time of the current step.

    Returns:
        next_tick_ns, skipped_ticks
    """

    previous_tick_ns = int(previous_tick_ns)
    current_ns = int(current_ns)
    period_ns = int(period_ns)

    if period_ns <= 0:
        raise ValueError('RL period must be positive')

    if previous_tick_ns < 0 or current_ns < 0:
        raise ValueError('Invalid simulation timestamp')

    if current_ns < previous_tick_ns:
        raise RuntimeError(
            'Simulation clock moved backwards'
        )

    next_tick_ns = previous_tick_ns + period_ns

    skipped_ticks = 0

    if current_ns >= next_tick_ns:

        skipped_ticks = (
            (current_ns - next_tick_ns)
            // period_ns
        ) + 1

        next_tick_ns += skipped_ticks * period_ns

    return int(next_tick_ns), int(skipped_ticks)
