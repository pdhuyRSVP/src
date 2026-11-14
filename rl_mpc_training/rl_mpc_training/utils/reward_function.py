#!/usr/bin/env python3

"""Reward function for SAC adaptive MPC navigation."""

import numpy as np


# ========================================================
# Reward configuration
# ========================================================

MAX_SPEED_MPS = 1.0
NOMINAL_DT_S = 0.5

PATH_SCALE_M = 0.5
HEADING_SCALE_RAD = 0.8

SAFE_DISTANCE_M = 0.75

W_PROGRESS = 1.5
W_PATH = 0.6
W_HEADING = 0.3
W_CLEARANCE = 0.8
W_ACTION_CHANGE = 0.1
W_UNKNOWN_LIDAR = 0.25
W_TIME = 0.02


TERMINAL_REWARDS = {
    'EPISODE_RUNNING': 0.0,
    'EPISODE_SUCCESS': 25.0,
    'EPISODE_COLLISION': -100.0,
    'EPISODE_TIMEOUT': -30.0,
    'EPISODE_PLANNING_FAILURE': -30.0,
}


def compute_reward(
    previous_raw,
    current_raw,
    previous_action,
    current_action,
    dt_s,
    phase='EPISODE_RUNNING'
):
    """
    Compute SAC reward from two consecutive observations.

    Observation indexing follows observation_builder.py.

    Returns:
        {
            'total': float,
            'components': dict,
            'metrics': dict
        }
    """

    previous = np.asarray(
        previous_raw,
        dtype=np.float64
    )

    current = np.asarray(
        current_raw,
        dtype=np.float64
    )

    action_prev = np.asarray(
        previous_action,
        dtype=np.float64
    )

    action_curr = np.asarray(
        current_action,
        dtype=np.float64
    )

    # ----------------------------------------------------
    # 1. Validate inputs
    # ----------------------------------------------------

    if previous.shape != (30,) or current.shape != (30,):
        raise ValueError('Reward requires 30D observations')

    if action_prev.shape != (5,) or action_curr.shape != (5,):
        raise ValueError('Reward requires 5D actions')

    if not (
        np.isfinite(previous).all()
        and np.isfinite(current).all()
        and np.isfinite(action_prev).all()
        and np.isfinite(action_curr).all()
    ):
        raise ValueError('Non-finite reward input')

    if (
        np.any(np.abs(action_prev) > 1.0)
        or np.any(np.abs(action_curr) > 1.0)
    ):
        raise ValueError('Action outside [-1, 1]')

    dt_s = float(dt_s)

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError('Invalid reward time interval')

    if phase not in TERMINAL_REWARDS:
        raise ValueError(f'Invalid reward phase: {phase}')

    # ----------------------------------------------------
    # 2. Progress reward
    # ----------------------------------------------------

    previous_distance = float(previous[4])
    current_distance = float(current[4])

    if previous_distance < 0.0 or current_distance < 0.0:
        raise ValueError('Invalid goal distance')

    progress_m = (
        previous_distance - current_distance
    )

    progress_normalized = np.clip(
        progress_m / (MAX_SPEED_MPS * dt_s),
        -1.0,
        1.0
    )

    r_progress = (
        W_PROGRESS * progress_normalized
    )

    # ----------------------------------------------------
    # 3. Path tracking penalty
    # ----------------------------------------------------

    e_path = float(current[3])

    path_error_normalized = min(
        abs(e_path) / PATH_SCALE_M,
        2.0
    )

    r_path = (
        -W_PATH * path_error_normalized ** 2
    )

    # ----------------------------------------------------
    # 4. Heading tracking penalty
    # ----------------------------------------------------

    e_heading = float(current[2])

    heading_error_normalized = min(
        abs(e_heading) / HEADING_SCALE_RAD,
        2.0
    )

    r_heading = (
        -W_HEADING * heading_error_normalized ** 2
    )

    # ----------------------------------------------------
    # 5. LiDAR clearance penalty
    # ----------------------------------------------------

    distances = current[7:12]

    valid_mask = current[12:17]
    hit_mask = current[17:22]

    if not (
        np.isin(valid_mask, [0.0, 1.0]).all()
        and np.isin(hit_mask, [0.0, 1.0]).all()
    ):
        raise ValueError('Invalid LiDAR masks')

    if np.any(
        (hit_mask == 1.0) & (valid_mask == 0.0)
    ):
        raise ValueError('LiDAR hit without valid measurement')

    hit_sectors = (
        (valid_mask == 1.0)
        & (hit_mask == 1.0)
    )

    if np.any(hit_sectors):

        min_distance = float(
            np.min(distances[hit_sectors])
        )

        if min_distance < 0.0:
            raise ValueError('Negative LiDAR distance')

        proximity = np.clip(
            (SAFE_DISTANCE_M - min_distance)
            / SAFE_DISTANCE_M,
            0.0,
            1.0
        )

    else:

        min_distance = None
        proximity = 0.0

    r_clearance = (
        -W_CLEARANCE * proximity ** 2
    )

    # ----------------------------------------------------
    # 6. Unknown LiDAR sectors
    # ----------------------------------------------------

    unknown_fraction = float(
        np.mean(valid_mask == 0.0)
    )

    r_unknown = (
        -W_UNKNOWN_LIDAR * unknown_fraction
    )

    # ----------------------------------------------------
    # 7. Action variation
    # ----------------------------------------------------

    action_change = float(
        np.mean((action_curr - action_prev) ** 2)
    )

    r_action = (
        -W_ACTION_CHANGE * action_change
    )

    # ----------------------------------------------------
    # 8. Time penalty
    # ----------------------------------------------------

    r_time = (
        -W_TIME * dt_s / NOMINAL_DT_S
    )

    # ----------------------------------------------------
    # 9. Terminal reward
    # ----------------------------------------------------

    r_terminal = TERMINAL_REWARDS[phase]

    # ----------------------------------------------------
    # 10. Total reward
    # ----------------------------------------------------

    components = {
        'progress': float(r_progress),
        'path': float(r_path),
        'heading': float(r_heading),
        'clearance': float(r_clearance),
        'unknown_lidar': float(r_unknown),
        'action_change': float(r_action),
        'time': float(r_time),
        'terminal': float(r_terminal),
    }

    total = float(sum(components.values()))

    return {
        'total': total,
        'components': components,
        'metrics': {
            'progress_m': float(progress_m),
            'e_path_m': float(e_path),
            'e_heading_rad': float(e_heading),
            'min_hit_distance_m': min_distance,
            'unknown_sector_fraction': unknown_fraction,
            'action_change': action_change,
            'dt_s': dt_s,
        }
    }
