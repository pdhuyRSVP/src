#!/usr/bin/env python3

"""Terminal transition validation for RL-MPC Gymnasium."""

import numpy as np

from rl_mpc_training.utils.step_utils import (
    classify_episode_status
)


OBS_DIM = 30


def _validate_observation(observation, raw):

    observation = np.asarray(
        observation,
        dtype=np.float32
    )

    raw = np.asarray(
        raw,
        dtype=np.float32
    )

    if observation.shape != (OBS_DIM,):
        raise ValueError(
            'Terminal observation must be 30D'
        )

    if raw.shape != (OBS_DIM,):
        raise ValueError(
            'Terminal raw observation must be 30D'
        )

    if not (
        np.isfinite(observation).all()
        and np.isfinite(raw).all()
    ):
        raise ValueError(
            'Non-finite terminal observation'
        )

    if np.any(observation < -1.0) or np.any(
        observation > 1.0
    ):
        raise ValueError(
            'Normalized terminal observation outside [-1,1]'
        )

    return observation.copy(), raw.copy()


def build_terminal_transition(
    status,
    last_observation,
    last_raw_observation,
    last_sample_ns,
    terminal_observation=None,
    terminal_raw_observation=None,
    terminal_sample_ns=None,
    terminal_verified=False
):
    """
    Prepare terminal observation and replay metadata.

    terminal_verified=True is allowed only when the
    caller has already validated the terminal sensor
    snapshot, coordinate frames, path consistency,
    and timestamps.

    This function does not acquire ROS data.
    """

    # ----------------------------------------------------
    # 1. Validate Episode Manager status
    # ----------------------------------------------------

    state = classify_episode_status(status)

    if state['stage'] != 'terminal':

        raise ValueError(
            'Terminal transition requires terminal phase'
        )

    # ----------------------------------------------------
    # 2. Validate previous observation
    # ----------------------------------------------------

    last_obs, last_raw = _validate_observation(
        last_observation,
        last_raw_observation
    )

    last_sample_ns = int(last_sample_ns)

    if last_sample_ns <= 0:
        raise ValueError(
            'Invalid previous sample timestamp'
        )

    # ----------------------------------------------------
    # 3. Select terminal observation
    # ----------------------------------------------------

    if terminal_verified:

        if (
            terminal_observation is None
            or terminal_raw_observation is None
            or terminal_sample_ns is None
        ):
            raise ValueError(
                'Verified terminal requires observation, '
                'raw observation and timestamp'
            )

        terminal_sample_ns = int(
            terminal_sample_ns
        )

        if terminal_sample_ns <= last_sample_ns:

            raise ValueError(
                'Terminal observation is not newer '
                'than previous observation'
            )

        observation, raw = _validate_observation(
            terminal_observation,
            terminal_raw_observation
        )

        source = 'fresh'

    else:

        if (
            terminal_observation is not None
            or terminal_raw_observation is not None
            or terminal_sample_ns is not None
        ):
            raise ValueError(
                'Unverified terminal data must not '
                'be supplied'
            )

        observation = last_obs
        raw = last_raw

        source = 'reused'

    # ----------------------------------------------------
    # 4. Replay and bootstrap policy
    # ----------------------------------------------------

    terminated = bool(state['terminated'])
    truncated = bool(state['truncated'])

    replay_eligible = (
        terminated or terminal_verified
    )

    bootstrap_allowed = (
        truncated and terminal_verified
    )

    # Without a fresh terminal state, the geometric
    # reward components cannot be updated reliably.
    reward_mode = (
        'full'
        if terminal_verified
        else 'event_only'
    )

    # ----------------------------------------------------
    # 5. Return terminal transition data
    # ----------------------------------------------------

    return {
        'observation': observation,
        'raw_observation': raw,

        'phase': state['phase'],

        'terminated': terminated,
        'truncated': truncated,

        'terminal_observation_reused':
            not terminal_verified,

        'terminal_observation_source': source,

        'replay_eligible': bool(
            replay_eligible
        ),

        'bootstrap_allowed': bool(
            bootstrap_allowed
        ),

        'reward_mode': reward_mode,
    }
