#!/usr/bin/env python3

"""Step 4.2.2b - Verify RL-MPC path geometry."""

import math
import time

import numpy as np

from rclpy.time import Time
from tf2_ros import TransformException

from rl_mpc_training.utils.path_geometry import (
    compute_path_geometry
)

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


def yaw_from_quaternion(q):

    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def stamp_ns(stamp):

    return (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
    )


def test_synthetic():

    print('\n[1] Testing geometry with known values')

    path = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0]
    ])

    robot = [0.8, 0.2, 0.1]
    reference = [1.0, 0.0, 0.0]

    data = compute_path_geometry(
        robot,
        reference,
        path
    )

    assert math.isclose(
        data['e_long_m'], -0.2, abs_tol=1e-8
    )

    assert math.isclose(
        data['e_lateral_m'], 0.2, abs_tol=1e-8
    )

    assert math.isclose(
        data['e_heading_rad'], 0.1, abs_tol=1e-8
    )

    assert math.isclose(
        data['e_path_m'], 0.2, abs_tol=1e-8
    )

    assert math.isclose(
        data['remaining_path_m'], 2.2, abs_tol=1e-8
    )

    assert math.isclose(
        data['curvature_ahead_1pm'], 0.0, abs_tol=1e-8
    )

    print('[PASS] Straight path geometry')

    # Left-turning path: curvature should be positive.
    turning_path = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.5, 0.25],
        [2.0, 1.0]
    ])

    turning_data = compute_path_geometry(
        [0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        turning_path,
        lookahead_m=2.0
    )

    assert turning_data['curvature_ahead_1pm'] > 0.0

    print('[PASS] Left-turn curvature')

    return True


def test_ros():

    env = RLMPCEnv()

    try:

        print('\n[2] ROS2 geometry integration test')

        answer = input(
            '\nType TEST to start one episode: '
        )

        if answer.strip() != 'TEST':
            print('Test cancelled')
            return

        bridge = env.bridge

        # Freshness baselines before starting a new episode.
        path_baseline = bridge.accepted_path_count
        reference_baseline = bridge.reference_count

        observation, info = env.reset(seed=123)

        print('[PASS] Gymnasium reset completed')
        print('Initial observation:', observation)

        # --------------------------------------------
        # Wait for current-episode path and reference
        # --------------------------------------------

        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:

            path = bridge.accepted_path
            reference = bridge.reference

            fresh_path = (
                bridge.accepted_path_count > path_baseline
            )

            fresh_reference = (
                bridge.reference_count > reference_baseline
            )

            if (
                not fresh_path
                or not fresh_reference
                or path is None
                or reference is None
                or len(path.poses) < 2
                or bridge.accepted_path_invalidated
            ):

                time.sleep(0.05)
                continue

            if (
                path.header.frame_id != 'map'
                or reference.header.frame_id != 'map'
            ):
                raise RuntimeError(
                    'Path and reference must use map frame'
                )

            path_time = stamp_ns(path.header.stamp)
            reference_time = stamp_ns(
                reference.header.stamp
            )

            now_ns = bridge.get_clock().now().nanoseconds

            reference_age = (
                now_ns - reference_time
            ) / 1e9

            # Do not combine a new path with an older
            # MPC reference.
            if (
                path_time <= 0
                or reference_time < path_time
                or not (0.0 <= reference_age <= 1.0)
            ):

                time.sleep(0.05)
                continue

            try:

                transform = bridge.tf_buffer.lookup_transform(
                    'map',
                    'base_footprint',
                    Time()
                )

            except TransformException:

                time.sleep(0.05)
                continue

            # ----------------------------------------
            # Robot pose
            # ----------------------------------------

            p = transform.transform.translation
            q = transform.transform.rotation

            robot_pose = [
                p.x,
                p.y,
                yaw_from_quaternion(q)
            ]

            # ----------------------------------------
            # MPC reference pose
            # ----------------------------------------

            rp = reference.pose.position
            rq = reference.pose.orientation

            reference_pose = [
                rp.x,
                rp.y,
                yaw_from_quaternion(rq)
            ]

            # ----------------------------------------
            # Accepted path
            # ----------------------------------------

            path_xy = np.array([
                [
                    pose.pose.position.x,
                    pose.pose.position.y
                ]
                for pose in path.poses
            ], dtype=np.float64)

            # ----------------------------------------
            # Compute geometry
            # ----------------------------------------

            features = compute_path_geometry(
                robot_pose,
                reference_pose,
                path_xy,
                lookahead_m=1.0
            )

            print('\n[3] Robot pose:', robot_pose)
            print('[4] MPC reference:', reference_pose)
            print('[5] Accepted path poses:', len(path_xy))
            print('[6] Reference age:', reference_age, 's')

            print('\n[7] Geometric features')

            for key, value in features.items():
                print(f'{key}: {value:.6f}')

            assert all(
                math.isfinite(v)
                for v in features.values()
            )

            print('\n[PASS] Path geometry integration test')

            break

        else:

            raise TimeoutError(
                'Fresh accepted path, reference or TF unavailable'
            )

    finally:

        print('\nClosing environment')
        env.close()


def main():

    test_synthetic()

    print('\n[PASS] Synthetic geometry tests completed')

    test_ros()


if __name__ == '__main__':
    main()
