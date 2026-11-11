#!/usr/bin/env python3

"""Build a terminal observation from validated ROS snapshots."""

import math

import numpy as np

from rl_mpc_training.utils.path_geometry import (
    compute_path_geometry
)
from rl_mpc_training.utils.lidar_features import (
    compute_lidar_features
)
from rl_mpc_training.utils.motion_features import (
    compute_motion_features
)
from rl_mpc_training.utils.observation_builder import (
    build_observation
)
from rl_mpc_training.utils.action_mapping import (
    MPC_PARAM_MIN,
    MPC_PARAM_MAX
)


def stamp_ns(stamp):
    return (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
    )


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def build_terminal_observation_candidate(
    snapshot,
    transform,
    now_ns,
    last_sample_ns,
    episode_start_ns,
    robot_frame='base_footprint'
):
    """
    Build a candidate terminal observation.

    Requirements:
      - Accepted path remains valid.
      - MPC reference belongs to current path.
      - Odometry and LiDAR are newer than last observation.
      - Sensor timestamps are within configured tolerances.
      - MPC parameters are within the declared action bounds.

    The MPC reference may be latched briefly after HOLD.

    Returns:
        {
            'observation': normalized 30D vector,
            'raw_observation': raw 30D vector,
            'sample_ns': int,
            'metadata': dict
        }
    """

    now_ns = int(now_ns)
    last_sample_ns = int(last_sample_ns)
    episode_start_ns = int(episode_start_ns)

    if now_ns <= last_sample_ns:
        raise ValueError('No terminal sampling time available')

    # --------------------------------------------------
    # 1. Obtain sources
    # --------------------------------------------------

    path = snapshot['accepted_path']
    reference = snapshot['reference']
    odom = snapshot['odom']
    scan = snapshot['scan']
    params = snapshot['adaptive_params']

    if snapshot['accepted_path_invalidated']:
        raise ValueError('Terminal accepted path invalidated')

    if any(
        value is None
        for value in (path, reference, odom, scan, params)
    ):
        raise ValueError('Missing terminal observation source')

    if len(path.poses) < 2:
        raise ValueError('Terminal accepted path too short')

    if snapshot['adaptive_ack_seq'] <= 0:
        raise ValueError('MPC parameter feedback unavailable')

    # --------------------------------------------------
    # 2. Coordinate frames
    # --------------------------------------------------

    if (
        path.header.frame_id != 'map'
        or reference.header.frame_id != 'map'
        or odom.child_frame_id != robot_frame
        or transform.header.frame_id != 'map'
        or transform.child_frame_id != robot_frame
    ):
        raise ValueError('Terminal coordinate frame mismatch')

    # --------------------------------------------------
    # 3. Timestamps
    # --------------------------------------------------

    path_ns = stamp_ns(path.header.stamp)
    reference_ns = stamp_ns(reference.header.stamp)
    odom_ns = stamp_ns(odom.header.stamp)
    scan_ns = stamp_ns(scan.header.stamp)
    tf_ns = stamp_ns(transform.header.stamp)

    if min(
        path_ns,
        reference_ns,
        odom_ns,
        scan_ns,
        tf_ns
    ) <= 0:
        raise ValueError('Invalid terminal timestamp')

    if min(
        path_ns,
        reference_ns,
        odom_ns,
        scan_ns
    ) < episode_start_ns:
        raise ValueError('Terminal data predates episode')

    if reference_ns < path_ns:
        raise ValueError('Terminal reference older than path')

    # Both sensors must be newer than the previous
    # observation returned by Gymnasium.
    sample_ns = min(odom_ns, scan_ns)

    if sample_ns <= last_sample_ns:
        raise ValueError('Waiting for new terminal sensor data')

    ages = {
        'odom': (now_ns - odom_ns) / 1e9,
        'scan': (now_ns - scan_ns) / 1e9,
        'reference': (now_ns - reference_ns) / 1e9,
        'tf': (now_ns - tf_ns) / 1e9,
    }

    if any(
        ages[name] < -0.1 or ages[name] > 0.5
        for name in ('odom', 'scan')
    ):
        raise ValueError('Stale terminal sensor data')

    # MPC reference can stop updating after HOLD.
    # Permit a short, bounded period of latching.
    if not -0.1 <= ages['reference'] <= 1.0:
        raise ValueError('Terminal reference too old')

    if not -1.5 <= ages['tf'] <= 1.5:
        raise ValueError('Terminal TF outside tolerance')

    sensor_skew_s = abs(
        odom_ns - scan_ns
    ) / 1e9

    if sensor_skew_s > 0.3:
        raise ValueError('Terminal sensor skew too large')

    # --------------------------------------------------
    # 4. MPC parameters
    # --------------------------------------------------

    params = np.asarray(params, dtype=np.float64)

    if params.shape != (5,) or not np.isfinite(params).all():
        raise ValueError('Invalid terminal MPC parameters')

    if (
        np.any(params < MPC_PARAM_MIN - 1e-6)
        or np.any(params > MPC_PARAM_MAX + 1e-6)
    ):
        raise ValueError('Terminal MPC parameters out of bounds')

    # --------------------------------------------------
    # 5. Robot pose from TF
    # --------------------------------------------------

    p = transform.transform.translation
    q = transform.transform.rotation

    robot_pose = [
        float(p.x),
        float(p.y),
        yaw_from_quaternion(q)
    ]

    # --------------------------------------------------
    # 6. MPC reference
    # --------------------------------------------------

    rp = reference.pose.position
    rq = reference.pose.orientation

    reference_pose = [
        float(rp.x),
        float(rp.y),
        yaw_from_quaternion(rq)
    ]

    # --------------------------------------------------
    # 7. Accepted path
    # --------------------------------------------------

    path_xy = np.array(
        [
            [
                pose.pose.position.x,
                pose.pose.position.y
            ]
            for pose in path.poses
        ],
        dtype=np.float64
    )

    # --------------------------------------------------
    # 8. Rebuild 30D observation
    # --------------------------------------------------

    geometry = compute_path_geometry(
        robot_pose,
        reference_pose,
        path_xy,
        lookahead_m=1.0
    )

    lidar = compute_lidar_features(scan)

    motion = compute_motion_features(
        float(odom.twist.twist.linear.x),
        float(odom.twist.twist.angular.z),
        params
    )

    result = build_observation(
        geometry,
        lidar,
        motion,
        lidar_max_m=float(scan.range_max)
    )

    observation = result['normalized']
    raw = result['raw']

    if (
        observation.shape != (30,)
        or raw.shape != (30,)
        or not np.isfinite(observation).all()
        or not np.isfinite(raw).all()
        or np.any(observation < -1.0)
        or np.any(observation > 1.0)
    ):
        raise ValueError('Invalid terminal observation vector')

    return {
        'observation': observation.copy(),
        'raw_observation': raw.copy(),
        'sample_ns': int(sample_ns),

        'metadata': {
            'terminal_sensor_ages_s': {
                'odom': ages['odom'],
                'scan': ages['scan'],
            },
            'terminal_reference_age_s': ages['reference'],
            'terminal_tf_age_s': ages['tf'],
            'terminal_sensor_skew_s': sensor_skew_s,
            'terminal_path_stamp_ns': path_ns,
        }
    }
