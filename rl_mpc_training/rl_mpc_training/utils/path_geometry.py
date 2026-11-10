#!/usr/bin/env python3

"""Geometric features for RL-MPC observations."""

import math
import numpy as np


EPS = 1e-9


def compute_path_geometry(
    robot_pose,
    reference_pose,
    path_xy,
    lookahead_m=1.0
):
    """
    robot_pose:     [x, y, yaw] in map frame
    reference_pose: [x_ref, y_ref, yaw_ref] in map frame
    path_xy:        Nx2 accepted path coordinates in map frame
    """

    robot = np.asarray(robot_pose, dtype=np.float64)
    ref = np.asarray(reference_pose, dtype=np.float64)
    path = np.asarray(path_xy, dtype=np.float64)

    # ----------------------------------------------------
    # 1. Validate inputs
    # ----------------------------------------------------

    if robot.shape != (3,) or ref.shape != (3,):
        raise ValueError(
            'Robot and reference pose must be [x, y, yaw]'
        )

    if (
        path.ndim != 2
        or path.shape[1] != 2
        or path.shape[0] < 2
    ):
        raise ValueError(
            'Accepted path must contain at least 2 XY points'
        )

    if not (
        np.all(np.isfinite(robot))
        and np.all(np.isfinite(ref))
        and np.all(np.isfinite(path))
    ):
        raise ValueError('Non-finite geometry data')

    if not math.isfinite(lookahead_m) or lookahead_m <= 0:
        raise ValueError('Invalid lookahead distance')

    # ----------------------------------------------------
    # 2. Reference tracking errors
    # ----------------------------------------------------

    dx = robot[0] - ref[0]
    dy = robot[1] - ref[1]

    c = math.cos(ref[2])
    s = math.sin(ref[2])

    e_long = c * dx + s * dy

    e_lateral = -s * dx + c * dy

    dyaw = robot[2] - ref[2]

    e_heading = math.atan2(
        math.sin(dyaw),
        math.cos(dyaw)
    )

    # ----------------------------------------------------
    # 3. Path segments
    # ----------------------------------------------------

    segments = np.diff(path, axis=0)

    seg_lengths = np.linalg.norm(
        segments,
        axis=1
    )

    valid_idx = np.flatnonzero(seg_lengths > EPS)

    if valid_idx.size == 0:
        raise ValueError(
            'Accepted path contains no valid segment'
        )

    # ----------------------------------------------------
    # 4. Nearest projection onto the path
    # ----------------------------------------------------

    starts = path[valid_idx]
    vectors = segments[valid_idx]
    lengths = seg_lengths[valid_idx]

    relative = robot[:2] - starts

    ratios = np.sum(
        relative * vectors,
        axis=1
    ) / (lengths * lengths)

    ratios = np.clip(ratios, 0.0, 1.0)

    projected = (
        starts + ratios[:, None] * vectors
    )

    distances_sq = np.sum(
        (projected - robot[:2]) ** 2,
        axis=1
    )

    nearest = int(np.argmin(distances_sq))

    segment_index = int(valid_idx[nearest])

    closest_point = projected[nearest]

    tangent = (
        vectors[nearest] / lengths[nearest]
    )

    # Signed cross-track error relative to path
    error_vector = robot[:2] - closest_point

    e_path = (
        -tangent[1] * error_vector[0]
        + tangent[0] * error_vector[1]
    )

    # ----------------------------------------------------
    # 5. Cumulative path arc length
    # ----------------------------------------------------

    arc_length = np.concatenate((
        [0.0],
        np.cumsum(seg_lengths)
    ))

    s_nearest = (
        arc_length[segment_index]
        + ratios[nearest] * seg_lengths[segment_index]
    )

    remaining_path = max(
        0.0,
        float(arc_length[-1] - s_nearest)
    )

    # ----------------------------------------------------
    # 6. Euclidean distance to goal
    # ----------------------------------------------------

    d_goal = float(
        np.linalg.norm(robot[:2] - path[-1])
    )

    # ----------------------------------------------------
    # 7. Interpolate a point at path arc length
    # ----------------------------------------------------

    def point_at_arc(s_query):

        target = float(np.clip(
            s_query,
            0.0,
            arc_length[-1]
        ))

        if target >= arc_length[-1]:
            return path[-1].copy()

        index = int(
            np.searchsorted(
                arc_length,
                target,
                side='right'
            ) - 1
        )

        index = min(
            max(index, 0),
            len(seg_lengths) - 1
        )

        if seg_lengths[index] <= EPS:
            return path[index].copy()

        ratio = (
            (target - arc_length[index])
            / seg_lengths[index]
        )

        return (
            path[index] + ratio * segments[index]
        )

    # ----------------------------------------------------
    # 8. Curvature ahead
    # ----------------------------------------------------

    p0 = point_at_arc(s_nearest)

    p1 = point_at_arc(
        s_nearest + lookahead_m / 2.0
    )

    p2 = point_at_arc(
        s_nearest + lookahead_m
    )

    ab = p1 - p0
    bc = p2 - p1
    ac = p2 - p0

    denominator = float(
        np.linalg.norm(ab)
        * np.linalg.norm(bc)
        * np.linalg.norm(ac)
    )

    if denominator <= EPS:

        curvature = 0.0

    else:

        cross_z = (
            ab[0] * ac[1]
            - ab[1] * ac[0]
        )

        curvature = (
            2.0 * cross_z / denominator
        )

    # ----------------------------------------------------
    # 9. Return geometric features
    # ----------------------------------------------------

    return {
        'e_long_m': float(e_long),
        'e_lateral_m': float(e_lateral),
        'e_heading_rad': float(e_heading),
        'e_path_m': float(e_path),
        'd_goal_m': d_goal,
        'remaining_path_m': remaining_path,
        'curvature_ahead_1pm': float(curvature),
    }
