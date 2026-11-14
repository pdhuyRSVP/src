#!/usr/bin/env python3

"""Stateless state-conditioned action mapping for trainable SAC-MPC.

The SAC policy acts in command coordinates u in [-1,1]^5.

The environment applies:
    a = T(observation, u)

The mapping respects the tested action envelope and
per-step action change bounds.

The Replay Buffer must store u, the action of the
wrapped training environment.

The actual MPC action a must be checked against MPC
feedback and recorded separately for diagnostics.

No ROS2 interfaces or optimizer modifications.
"""

import numpy as np

from rl_mpc_training.utils.action_mapping import (
    MPC_PARAM_MIN,
    MPC_PARAM_MAX,
    map_action_to_mpc,
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION,
    TEST_ACTION,
    MAX_DELTA,
)


# ======================================================
# Configuration
# ======================================================

SAFE_LOWER = np.minimum(
    DEFAULT_ACTION,
    TEST_ACTION
).astype(np.float64)

SAFE_UPPER = np.maximum(
    DEFAULT_ACTION,
    TEST_ACTION
).astype(np.float64)

SAFE_LOWER[4] = -1.0
SAFE_UPPER[4] = 1.0

DELTA = MAX_DELTA.astype(np.float64)

EPS = 2e-4


# ======================================================
# 1. Input validation
# ======================================================

def _validate(
    value,
    shape,
    label,
    normalized=True
):

    try:

        arr = np.asarray(
            value,
            dtype=np.float64
        )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            f'{label} must be numeric'
        ) from exc

    if (
        arr.shape != shape
        or not np.isfinite(arr).all()
    ):

        raise ValueError(
            f'{label} must be finite '
            f'with shape {shape}'
        )

    if normalized and np.any(
        np.abs(arr) > 1.0
    ):

        raise ValueError(
            f'{label} outside normalized [-1,1]'
        )

    return arr.copy()


# ======================================================
# 2. Reconstruct previous MPC action
# ======================================================

def previous_action_from_observation(observation):

    """Recover previously applied normalized MPC action."""

    obs = _validate(
        observation,
        (30,),
        'observation'
    )

    if (
        np.any(obs[25:30] < -EPS)
        or np.any(obs[25:30] > 1.0 + EPS)
    ):

        raise ValueError(
            'MPC observation features '
            'must be in [0,1]'
        )

    previous = np.empty(
        5,
        dtype=np.float64
    )

    # --------------------------------------------------
    # First four parameters
    #
    # Qxy/Qyaw use linear normalization.
    # Sv/Sw use logarithmic normalization.
    #
    # Both correspond to (action + 1) / 2.
    # --------------------------------------------------

    previous[:4] = (
        2.0 * obs[25:29] - 1.0
    )

    # --------------------------------------------------
    # Reference speed
    #
    # observation[29] = v_ref / 1.0
    # --------------------------------------------------

    v_ref = (
        obs[29] * MPC_PARAM_MAX[4]
    )

    previous[4] = (
        2.0
        * (
            (v_ref - MPC_PARAM_MIN[4])
            / (
                MPC_PARAM_MAX[4]
                - MPC_PARAM_MIN[4]
            )
        )
        - 1.0
    )

    # --------------------------------------------------
    # Check tested action envelope
    # --------------------------------------------------

    if (
        np.any(previous < SAFE_LOWER - EPS)
        or np.any(previous > SAFE_UPPER + EPS)
    ):

        raise ValueError(
            'Previous MPC action outside '
            'tested safe envelope'
        )

    return np.clip(
        previous,
        SAFE_LOWER,
        SAFE_UPPER
    )


# ======================================================
# 3. Admissible action interval
# ======================================================

def admissible_action_bounds(observation):

    """Calculate state-dependent local action bounds."""

    previous = previous_action_from_observation(
        observation
    )

    low = np.maximum(
        SAFE_LOWER,
        previous - DELTA
    )

    high = np.minimum(
        SAFE_UPPER,
        previous + DELTA
    )

    if not np.all(high > low):

        raise ValueError(
            'Degenerate local action interval'
        )

    return low, high


# ======================================================
# 4. SAC command -> MPC action
# ======================================================

def command_to_mpc_action(
    observation,
    command
):

    """Map normalized SAC command to bounded MPC action."""

    u = _validate(
        command,
        (5,),
        'SAC command'
    )

    low, high = admissible_action_bounds(
        observation
    )

    # --------------------------------------------------
    # State-dependent affine mapping
    # --------------------------------------------------

    applied = (
        low
        + 0.5
        * (u + 1.0)
        * (high - low)
    )

    applied = np.clip(
        applied,
        low,
        high
    ).astype(np.float32)

    # --------------------------------------------------
    # Check MPC parameter mapping
    # --------------------------------------------------

    params = np.asarray(
        map_action_to_mpc(applied),
        dtype=np.float64
    )

    if (
        np.any(params < MPC_PARAM_MIN - 1e-7)
        or np.any(params > MPC_PARAM_MAX + 1e-7)
    ):

        raise ValueError(
            'Mapped MPC parameters '
            'exceed global limits'
        )

    return applied


# ======================================================
# 5. MPC action -> SAC command
# ======================================================

def mpc_action_to_command(
    observation,
    applied_action
):

    """Invert T(observation, command) for feasible actions."""

    applied = _validate(
        applied_action,
        (5,),
        'applied MPC action'
    )

    low, high = admissible_action_bounds(
        observation
    )

    if (
        np.any(applied < low - EPS)
        or np.any(applied > high + EPS)
    ):

        raise ValueError(
            'Applied action violates local bounds'
        )

    u = (
        2.0
        * (
            np.clip(applied, low, high)
            - low
        )
        / (high - low)
        - 1.0
    )

    return np.clip(
        u,
        -1.0,
        1.0
    ).astype(np.float32)
