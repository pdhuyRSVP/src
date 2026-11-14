#!/usr/bin/env python3

"""Map normalized SAC actions to adaptive MPC parameters."""

import numpy as np


# ========================================================
# Adaptive MPC parameter bounds
# ========================================================

MPC_PARAM_MIN = np.array(
    [1.0, 1.0, 1.0, 1.0, 0.2],
    dtype=np.float64
)

MPC_PARAM_MAX = np.array(
    [10.0, 20.0, 100.0, 100.0, 1.0],
    dtype=np.float64
)


# ========================================================
# Normalized action -> MPC parameters
# ========================================================

def map_action_to_mpc(action):
    """
    Convert a normalized SAC action to MPC parameters.

    Input:
        action = [a1, a2, a3, a4, a5]
        each component in [-1, 1]

    Output:
        [alpha_Qxy, alpha_Qyaw, alpha_Sv, alpha_Sw, v_ref]
    """

    # ----------------------------------------------------
    # 1. Convert input
    # ----------------------------------------------------

    action = np.asarray(
        action,
        dtype=np.float64
    )

    # ----------------------------------------------------
    # 2. Validate dimensions
    # ----------------------------------------------------

    if action.shape != (5,):

        raise ValueError(
            'SAC action must contain exactly 5 values'
        )

    # ----------------------------------------------------
    # 3. Validate finite values
    # ----------------------------------------------------

    if not np.all(np.isfinite(action)):

        raise ValueError(
            'SAC action contains non-finite values'
        )

    # ----------------------------------------------------
    # 4. Validate normalized action bounds
    # ----------------------------------------------------

    if np.any(action < -1.0) or np.any(action > 1.0):

        raise ValueError(
            'SAC action must be within [-1, 1]'
        )

    # ----------------------------------------------------
    # 5. Hybrid linear-logarithmic mapping
    # ----------------------------------------------------

    normalized = (action + 1.0) / 2.0

    # Linear mapping for all parameters initially
    parameters = (
        MPC_PARAM_MIN
        + normalized * (MPC_PARAM_MAX - MPC_PARAM_MIN)
    )

    # Logarithmic mapping for alpha_Sv and alpha_Sw
    parameters[2:4] = (
        MPC_PARAM_MIN[2:4]
        * np.power(
            MPC_PARAM_MAX[2:4] / MPC_PARAM_MIN[2:4],
            normalized[2:4]
        )
    )

    # ----------------------------------------------------
    # 6. Return Python list
    # ----------------------------------------------------

    return [float(x) for x in parameters]
