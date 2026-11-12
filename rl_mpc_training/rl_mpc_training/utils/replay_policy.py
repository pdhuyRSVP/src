#!/usr/bin/env python3

"""Replay storage and bootstrap policy for SAC."""

import math

from rl_mpc_training.utils.reward_function import (
    TERMINAL_REWARDS
)

from rl_mpc_training.utils.step_utils import (
    TERMINAL_FLAGS
)


PLANNING_FAILURE_MODES = (
    'task_failure',
    'external_interruption'
)


def decide_replay_policy(
    *,
    terminated,
    truncated,
    info,
    planning_failure_mode='task_failure'
):
    """
    Decide how one Gymnasium transition is used by SAC.

    Does not modify Gymnasium flags or ROS2 state.

    Returns:
        store_transition
        bootstrap_mask
        learning_terminated
        learning_truncated
        episode_done
        policy_override
        reason
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if (
        type(terminated) is not bool
        or type(truncated) is not bool
    ):
        raise ValueError(
            'terminated and truncated must be bool'
        )

    if not isinstance(info, dict):
        raise ValueError('info must be a dictionary')

    if planning_failure_mode not in PLANNING_FAILURE_MODES:
        raise ValueError('Unknown planning failure mode')

    required = (
        'phase',
        'action_applied',
        'terminal_observation_reused',
        'reward_mode',
        'reward_components'
    )

    if any(name not in info for name in required):
        raise ValueError(
            'Missing replay-policy metadata in info'
        )

    phase = info['phase']
    applied = info['action_applied']
    reused = info['terminal_observation_reused']

    reward_mode = info['reward_mode']
    components = info['reward_components']

    if (
        type(applied) is not bool
        or type(reused) is not bool
    ):
        raise ValueError(
            'Invalid action/observation validity flags'
        )

    if phase not in TERMINAL_REWARDS:
        raise ValueError(
            f'Unknown episode phase: {phase}'
        )

    if (
        not isinstance(components, dict)
        or 'terminal' not in components
    ):
        raise ValueError(
            'Missing terminal reward component'
        )

    try:
        terminal_component = float(
            components['terminal']
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'Invalid terminal reward component'
        ) from exc

    expected_terminal_reward = TERMINAL_REWARDS[phase]

    if (
        not math.isfinite(terminal_component)
        or abs(
            terminal_component - expected_terminal_reward
        ) > 1e-6
    ):
        raise ValueError(
            'Terminal reward component disagrees with phase'
        )

    # --------------------------------------------------
    # 2. Running transition
    # --------------------------------------------------

    if phase == 'EPISODE_RUNNING':

        if terminated or truncated:
            raise ValueError(
                'RUNNING must have both episode flags False'
            )

        if (
            not applied
            or reused
            or reward_mode != 'full'
        ):
            raise ValueError(
                'Invalid RUNNING transition'
            )

        return {
            'store_transition': True,

            'bootstrap_mask': 1.0,

            'learning_terminated': False,
            'learning_truncated': False,

            'episode_done': False,

            'policy_override': False,

            'reason': 'running_valid'
        }

    # --------------------------------------------------
    # 3. Validate terminal flags
    # --------------------------------------------------

    expected_flags = TERMINAL_FLAGS[phase]

    if (terminated, truncated) != expected_flags:
        raise ValueError(
            'Terminal flags disagree with episode phase'
        )

    expected_reward_mode = (
        'full'
        if applied and not reused
        else 'event_only'
    )

    if reward_mode != expected_reward_mode:
        raise ValueError(
            'Terminal reward_mode inconsistent '
            'with observation/action'
        )

    # Check the original environment replay flag.
    # It may intentionally differ from the new policy.
    env_replay_eligible = info.get(
        'replay_eligible'
    )

    if env_replay_eligible is not None:

        if type(env_replay_eligible) is not bool:
            raise ValueError(
                'Invalid environment replay_eligible flag'
            )

        expected_env_eligible = (
            applied and (terminated or not reused)
        )

        if env_replay_eligible != expected_env_eligible:
            raise ValueError(
                'Environment replay eligibility inconsistent'
            )

    # --------------------------------------------------
    # 4. Action was not applied
    # --------------------------------------------------

    if not applied:

        store = False
        bootstrap = False

        learned_terminated = terminated
        learned_truncated = truncated

        reason = 'action_not_applied'

    # --------------------------------------------------
    # 5. Planning failure
    # --------------------------------------------------

    elif phase == 'EPISODE_PLANNING_FAILURE':

        if planning_failure_mode == 'task_failure':

            # Treat as terminal task failure for SAC.
            # No bootstrap from reused observation.
            store = True
            bootstrap = False

            learned_terminated = True
            learned_truncated = False

            reason = (
                'planning_failure_as_terminal_task_failure'
            )

        else:

            # Exogenous environment interruption.
            # Do not attribute this failure to the policy.
            store = False
            bootstrap = False

            learned_terminated = False
            learned_truncated = True

            reason = (
                'planning_failure_external_interruption'
            )

    # --------------------------------------------------
    # 6. Episode timeout
    # --------------------------------------------------

    elif phase == 'EPISODE_TIMEOUT':

        # A truncation needs a valid final observation
        # for bootstrapping.
        store = not reused
        bootstrap = store

        learned_terminated = False
        learned_truncated = True

        reason = (
            'timeout_fresh'
            if store
            else 'timeout_stale_dropped'
        )

    # --------------------------------------------------
    # 7. Success or collision
    # --------------------------------------------------

    else:

        # Both are genuine terminal states.
        store = True
        bootstrap = False

        learned_terminated = True
        learned_truncated = False

        reason = (
            'terminal_with_fresh_obs'
            if not reused
            else 'terminal_event_only'
        )

    # --------------------------------------------------
    # 8. Final consistency
    # --------------------------------------------------

    if not store:
        bootstrap = False

    policy_override = (
        (learned_terminated, learned_truncated)
        != (terminated, truncated)
        or (
            env_replay_eligible is not None
            and store != env_replay_eligible
        )
    )

    return {
        'store_transition': bool(store),

        'bootstrap_mask': (
            1.0 if bootstrap else 0.0
        ),

        'learning_terminated':
            bool(learned_terminated),

        'learning_truncated':
            bool(learned_truncated),

        'episode_done': True,

        'policy_override':
            bool(policy_override),

        'reason': reason
    }
