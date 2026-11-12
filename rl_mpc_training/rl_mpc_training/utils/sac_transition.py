#!/usr/bin/env python3

"""Build validated SAC transitions from Gymnasium steps."""

from dataclasses import dataclass

import numpy as np

from rl_mpc_training.utils.replay_policy import (
    decide_replay_policy
)


OBS_DIM = 30
ACTION_DIM = 5


REWARD_KEYS = (
    'progress',
    'path',
    'heading',
    'clearance',
    'unknown_lidar',
    'action_change',
    'time',
    'terminal',
)


@dataclass(frozen=True)
class SACTransition:
    """
    One validated transition for SAC training.

    observation and next_observation are normalized.
    action is normalized to [-1, 1].
    """

    observation: np.ndarray
    action: np.ndarray

    reward: float

    next_observation: np.ndarray
    bootstrap_mask: float

    episode_index: int
    step_index: int

    phase: str

    terminated: bool
    truncated: bool

    learning_terminated: bool
    learning_truncated: bool

    terminal_observation_reused: bool

    reward_components: dict

    policy_reason: str
    policy_override: bool


def _validate_vector(
    value,
    expected_dim,
    name
):
    """Validate and copy a normalized vector."""

    array = np.asarray(
        value,
        dtype=np.float32
    )

    if array.shape != (expected_dim,):
        raise ValueError(
            f'{name} must have shape ({expected_dim},)'
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f'{name} contains non-finite values'
        )

    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError(
            f'{name} outside [-1, 1]'
        )

    # Independent storage.
    result = array.copy()

    # Prevent accidental modification.
    result.setflags(write=False)

    return result


def build_sac_transition(
    *,
    observation,
    action,
    reward,
    next_observation,
    terminated,
    truncated,
    info,
    episode_index,
    step_index,
    planning_failure_mode='task_failure'
):
    """
    Convert one Gymnasium step to a SAC transition.

    Returns:
        transition:
            SACTransition or None

        policy:
            Replay Policy decision dictionary
    """

    # --------------------------------------------------
    # 1. Validate episode/step identifiers
    # --------------------------------------------------

    if (
        type(episode_index) is not int
        or episode_index < 0
    ):
        raise ValueError(
            'episode_index must be a nonnegative integer'
        )

    if (
        type(step_index) is not int
        or step_index < 1
    ):
        raise ValueError(
            'step_index must be a positive integer'
        )

    # --------------------------------------------------
    # 2. Validate transition vectors
    # --------------------------------------------------

    obs = _validate_vector(
        observation,
        OBS_DIM,
        'observation'
    )

    act = _validate_vector(
        action,
        ACTION_DIM,
        'action'
    )

    next_obs = _validate_vector(
        next_observation,
        OBS_DIM,
        'next_observation'
    )

    # --------------------------------------------------
    # 3. Validate reward
    # --------------------------------------------------

    reward = float(reward)

    if not np.isfinite(reward):
        raise ValueError('Non-finite reward')

    if not isinstance(info, dict):
        raise ValueError('info must be a dictionary')

    components = info.get('reward_components')

    if not isinstance(components, dict):
        raise ValueError(
            'Missing reward components'
        )

    if set(components.keys()) != set(REWARD_KEYS):
        raise ValueError(
            'Unexpected reward component keys'
        )

    components_copy = {}

    for name in REWARD_KEYS:

        value = float(components[name])

        if not np.isfinite(value):
            raise ValueError(
                f'Non-finite reward component: {name}'
            )

        components_copy[name] = value

    component_total = sum(
        components_copy.values()
    )

    if not np.isclose(
        reward,
        component_total,
        rtol=1e-6,
        atol=1e-5
    ):
        raise ValueError(
            'Reward does not match reward components'
        )

    # --------------------------------------------------
    # 4. Apply Replay Policy
    # --------------------------------------------------

    policy = decide_replay_policy(
        terminated=terminated,
        truncated=truncated,
        info=info,
        planning_failure_mode=planning_failure_mode
    )

    # --------------------------------------------------
    # 5. Skip ineligible transitions
    # --------------------------------------------------

    if not policy['store_transition']:

        return None, policy

    # --------------------------------------------------
    # 6. Build SAC transition
    # --------------------------------------------------

    transition = SACTransition(
        observation=obs,

        action=act,

        reward=reward,

        next_observation=next_obs,

        bootstrap_mask=float(
            policy['bootstrap_mask']
        ),

        episode_index=episode_index,
        step_index=step_index,

        phase=str(info['phase']),

        terminated=bool(terminated),
        truncated=bool(truncated),

        learning_terminated=bool(
            policy['learning_terminated']
        ),

        learning_truncated=bool(
            policy['learning_truncated']
        ),

        terminal_observation_reused=bool(
            info['terminal_observation_reused']
        ),

        reward_components=components_copy,

        policy_reason=str(
            policy['reason']
        ),

        policy_override=bool(
            policy['policy_override']
        )
    )

    return transition, policy
