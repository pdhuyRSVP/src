#!/usr/bin/env python3

"""Step 4.2.2d - Test motion and adaptive MPC features."""

import time

import numpy as np

from rl_mpc_training.utils.motion_features import (
    compute_motion_features
)

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


# ========================================================
# 1. Synthetic test
# ========================================================

def test_synthetic():

    print('\n[1] Testing synthetic motion features')

    v = 0.3
    omega = -0.2

    params = [
        1.5,
        2.0,
        3.0,
        4.0,
        0.5
    ]

    result = compute_motion_features(
        v,
        omega,
        params
    )

    expected = np.array(
        [
            0.3,
            -0.2,
            0.2,
            1.5,
            2.0,
            3.0,
            4.0,
            0.5
        ],
        dtype=np.float32
    )

    np.testing.assert_allclose(
        result['vector'],
        expected,
        rtol=0.0,
        atol=1e-6
    )

    assert result['vector'].shape == (8,)

    assert result['vector'].dtype == np.float32

    print('[PASS] Motion feature vector')

    # ----------------------------------------------------
    # Invalid velocity test
    # ----------------------------------------------------

    try:

        compute_motion_features(
            np.nan,
            0.0,
            params
        )

    except ValueError:

        print('[PASS] Invalid velocity rejected')

    else:

        raise AssertionError(
            'Invalid velocity was not rejected'
        )

    # ----------------------------------------------------
    # Invalid parameter test
    # ----------------------------------------------------

    try:

        compute_motion_features(
            0.2,
            0.0,
            [1.0, 1.0, -1.0, 1.0, 0.5]
        )

    except ValueError:

        print('[PASS] Invalid MPC parameter rejected')

    else:

        raise AssertionError(
            'Invalid MPC parameter was not rejected'
        )


# ========================================================
# 2. ROS2/Gazebo integration test
# ========================================================

def test_ros():

    env = RLMPCEnv()

    try:

        print('\n[2] ROS2 motion integration test')

        answer = input(
            '\nType TEST to start one episode: '
        )

        if answer.strip() != 'TEST':

            print('Test cancelled')
            return

        bridge = env.bridge

        # --------------------------------------------
        # Reset and start episode
        # --------------------------------------------

        observation, info = env.reset(seed=123)

        print('\n[PASS] Gymnasium reset completed')

        print('Initial observation:', observation)
        print('Episode:', info['attempt'])

        # Wait for an odometry message newer than
        # the observation returned by reset().
        odom_baseline = bridge.odom_count

        deadline = time.monotonic() + 5.0

        odom = None
        params = None

        # --------------------------------------------
        # Wait for motion and MPC data
        # --------------------------------------------

        while time.monotonic() < deadline:

            current_odom = bridge.odom
            current_params = bridge.adaptive_params

            if (
                current_odom is None
                or current_params is None
                or bridge.odom_count <= odom_baseline
            ):

                time.sleep(0.05)
                continue

            if len(current_params) != 5:

                time.sleep(0.05)
                continue

            # Check odometry timestamp in simulation time.
            stamp = current_odom.header.stamp

            stamp_ns = (
                int(stamp.sec) * 1_000_000_000
                + int(stamp.nanosec)
            )

            now_ns = bridge.get_clock().now().nanoseconds

            age_sec = (now_ns - stamp_ns) / 1e9

            if not (0.0 <= age_sec <= 0.5):

                time.sleep(0.05)
                continue

            odom = current_odom
            params = list(current_params)

            break

        if odom is None or params is None:

            raise TimeoutError(
                'Fresh odometry or adaptive MPC parameters unavailable'
            )

        # --------------------------------------------
        # Check coordinate frames
        # --------------------------------------------

        print('\n[3] Odometry information')

        print('Header frame:', odom.header.frame_id)
        print('Child frame:', odom.child_frame_id)
        print('Message age:', age_sec, 's')

        if odom.child_frame_id != 'base_footprint':

            raise RuntimeError(
                'Unexpected odometry child frame: '
                f'{odom.child_frame_id}'
            )

        # --------------------------------------------
        # Extract motion data
        # --------------------------------------------

        v = float(
            odom.twist.twist.linear.x
        )

        omega = float(
            odom.twist.twist.angular.z
        )

        print('\n[4] Robot motion')

        print(f'Linear velocity: {v:.4f} m/s')
        print(f'Angular velocity: {omega:.4f} rad/s')

        # --------------------------------------------
        # Current adaptive MPC parameters
        # --------------------------------------------

        print('\n[5] Adaptive MPC parameters')

        print('Parameters:', params)

        expected_defaults = [
            1.0, 1.0, 1.0, 1.0, 0.5
        ]

        if not np.allclose(
            params,
            expected_defaults,
            rtol=0.0,
            atol=1e-8
        ):

            raise RuntimeError(
                'MPC parameters differ from reset defaults'
            )

        print('[PASS] Default MPC parameters confirmed')

        # --------------------------------------------
        # Compute motion features
        # --------------------------------------------

        result = compute_motion_features(
            v,
            omega,
            params
        )

        print('\n[6] Motion features')

        for key, value in result.items():

            if key == 'vector':
                continue

            print(f'{key}: {value:.6f}')

        vector = result['vector']

        print('\nMotion feature vector:')
        print(vector)

        # --------------------------------------------
        # Validate output
        # --------------------------------------------

        assert vector.shape == (8,)

        assert vector.dtype == np.float32

        assert np.all(np.isfinite(vector))

        assert np.allclose(
            vector[3:],
            expected_defaults,
            atol=1e-6
        )

        print('\n[PASS] Motion features integration test')

    finally:

        print('\nClosing environment')

        env.close()


# ========================================================
# Main
# ========================================================

def main():

    test_synthetic()

    print('\n[PASS] Synthetic motion tests completed')

    test_ros()


if __name__ == '__main__':
    main()
