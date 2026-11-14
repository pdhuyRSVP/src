#!/usr/bin/env python3

"""Build normalized RL-MPC observations from validated features."""

import math
import numpy as np


GEOMETRY_KEYS = (
    'e_long_m',
    'e_lateral_m',
    'e_heading_rad',
    'e_path_m',
    'd_goal_m',
    'remaining_path_m',
    'curvature_ahead_1pm',
)

OBSERVATION_DIM = 30

GEOMETRY_SCALES = np.array(
    [2.0, 2.0, math.pi, 2.0, 20.0, 30.0, 2.0],
    dtype=np.float64
)

ALPHA_MIN = np.array(
    [1.0, 1.0, 1.0, 1.0],
    dtype=np.float64
)

ALPHA_MAX = np.array(
    [10.0, 20.0, 100.0, 100.0],
    dtype=np.float64
)


def build_observation(geometry, lidar, motion, lidar_max_m=12.0):
    """
    Build 30-dimensional normalized observation.

    Input:
        geometry: output of compute_path_geometry()
        lidar:    output of compute_lidar_features()
        motion:   output of compute_motion_features()

    Returns:
        raw:        unnormalized 30-dimensional vector
        normalized: normalized 30-dimensional vector
    """

    # ----------------------------------------------------
    # 1. Geometry
    # ----------------------------------------------------

    geometry_values = np.asarray(
        [geometry[key] for key in GEOMETRY_KEYS],
        dtype=np.float64
    )

    # ----------------------------------------------------
    # 2. LiDAR
    # ----------------------------------------------------

    distances = np.asarray(
        lidar['distances_m'],
        dtype=np.float64
    )

    valid_mask = np.asarray(
        lidar['valid_mask'],
        dtype=bool
    )

    hit_mask = np.asarray(
        lidar['hit_mask'],
        dtype=bool
    )

    if any(
        value.shape != (5,)
        for value in (distances, valid_mask, hit_mask)
    ):
        raise ValueError('Invalid LiDAR feature dimensions')

    if np.any(hit_mask & ~valid_mask):
        raise ValueError('LiDAR hit without valid measurement')

    if not math.isfinite(lidar_max_m) or lidar_max_m <= 0:
        raise ValueError('Invalid LiDAR maximum range')

    if (
        not np.isfinite(distances).all()
        or np.any(distances < 0)
        or np.any(distances > lidar_max_m)
    ):
        raise ValueError('Invalid LiDAR distances')

    # ----------------------------------------------------
    # 3. Motion and MPC
    # ----------------------------------------------------

    motion_values = np.asarray(
        motion['vector'],
        dtype=np.float64
    )

    if motion_values.shape != (8,):
        raise ValueError('Invalid motion feature dimensions')

    # ----------------------------------------------------
    # 4. Raw observation
    # ----------------------------------------------------

    raw = np.concatenate(
        (
            geometry_values,
            distances,
            valid_mask.astype(np.float64),
            hit_mask.astype(np.float64),
            motion_values
        )
    )

    if raw.shape != (OBSERVATION_DIM,):
        raise ValueError('Unexpected observation dimension')

    if not np.isfinite(raw).all():
        raise ValueError('Non-finite observation')

    # ----------------------------------------------------
    # 5. Normalize
    # ----------------------------------------------------

    normalized = np.empty(
        OBSERVATION_DIM,
        dtype=np.float64
    )

    # Geometry: indices 0..6
    normalized[0:7] = np.clip(
        raw[0:7] / GEOMETRY_SCALES,
        -1.0,
        1.0
    )

    # LiDAR distances: indices 7..11
    normalized[7:12] = np.clip(
        raw[7:12] / lidar_max_m,
        0.0,
        1.0
    )

    # LiDAR masks: indices 12..21
    normalized[12:22] = raw[12:22]

    # Motion: indices 22..24
    normalized[22] = np.clip(raw[22] / 1.0, -1.0, 1.0)
    normalized[23] = np.clip(raw[23] / 1.0, -1.0, 1.0)
    normalized[24] = np.clip(raw[24] / 1.0, -1.0, 1.0)

    # MPC multipliers: indices 25..28
    # Linear normalization: alpha_Qxy, alpha_Qyaw
    normalized[25:27] = np.clip(
        (raw[25:27] - ALPHA_MIN[0:2])
        / (ALPHA_MAX[0:2] - ALPHA_MIN[0:2]),
        0.0,
        1.0
    )
    # Logarithmic normalization: alpha_Sv, alpha_Sw
    normalized[27:29] = np.clip(
        np.log(raw[27:29] / ALPHA_MIN[2:4])
        / np.log(ALPHA_MAX[2:4] / ALPHA_MIN[2:4]),
        0.0,
        1.0
    )

    # Reference speed: index 29
    normalized[29] = np.clip(
        raw[29] / 1.0,
        0.0,
        1.0
    )

    normalized = normalized.astype(np.float32)

    if not np.isfinite(normalized).all():
        raise ValueError('Invalid normalized observation')

    return {
        'raw': raw.astype(np.float32),
        'normalized': normalized
    }
