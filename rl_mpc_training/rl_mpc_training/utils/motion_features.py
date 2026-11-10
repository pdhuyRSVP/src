#!/usr/bin/env python3

"""Motion and adaptive MPC features for RL observations."""

import numpy as np


def compute_motion_features(
    v_mps,
    omega_radps,
    adaptive_params
):
    """
    Compute motion and adaptive MPC features.

    Parameters
    ----------
    v_mps:
        Robot longitudinal velocity in m/s.

    omega_radps:
        Robot yaw rate in rad/s.

    adaptive_params:
        [alpha_Qxy, alpha_Qyaw, alpha_Sv,
         alpha_Sw, v_ref]

    Returns
    -------
    Dictionary containing individual features and
    an 8-dimensional float32 feature vector.
    """

    # ----------------------------------------------------
    # 1. Validate robot velocity
    # ----------------------------------------------------

    velocity = np.asarray(
        [v_mps, omega_radps],
        dtype=np.float64
    )

    if not np.all(np.isfinite(velocity)):
        raise ValueError(
            'Robot velocity contains non-finite values'
        )

    v = float(velocity[0])
    omega = float(velocity[1])

    # ----------------------------------------------------
    # 2. Validate adaptive MPC parameters
    # ----------------------------------------------------

    params = np.asarray(
        adaptive_params,
        dtype=np.float64
    )

    if params.shape != (5,):
        raise ValueError(
            'Adaptive MPC parameters must contain 5 values'
        )

    if not np.all(np.isfinite(params)):
        raise ValueError(
            'Adaptive MPC parameters contain non-finite values'
        )

    if np.any(params[:4] <= 0.0):
        raise ValueError(
            'Adaptive MPC weight multipliers must be positive'
        )

    if params[4] <= 0.0:
        raise ValueError(
            'MPC reference velocity must be positive'
        )

    # ----------------------------------------------------
    # 3. Extract adaptive parameters
    # ----------------------------------------------------

    alpha_Qxy = float(params[0])
    alpha_Qyaw = float(params[1])
    alpha_Sv = float(params[2])
    alpha_Sw = float(params[3])

    v_ref = float(params[4])

    # ----------------------------------------------------
    # 4. Reference speed difference
    # ----------------------------------------------------

    speed_gap = v_ref - v

    # ----------------------------------------------------
    # 5. Build feature vector
    # ----------------------------------------------------

    vector = np.array(
        [
            v,
            omega,
            speed_gap,
            alpha_Qxy,
            alpha_Qyaw,
            alpha_Sv,
            alpha_Sw,
            v_ref
        ],
        dtype=np.float32
    )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            'Motion feature vector contains invalid values'
        )

    # ----------------------------------------------------
    # 6. Return features
    # ----------------------------------------------------

    return {
        'v_mps': v,
        'omega_radps': omega,
        'speed_gap_mps': speed_gap,
        'alpha_Qxy': alpha_Qxy,
        'alpha_Qyaw': alpha_Qyaw,
        'alpha_Sv': alpha_Sv,
        'alpha_Sw': alpha_Sw,
        'v_ref_mps': v_ref,
        'vector': vector
    }
