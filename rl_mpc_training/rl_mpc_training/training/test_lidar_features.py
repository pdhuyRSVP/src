#!/usr/bin/env python3

"""Step 4.2.2c - Test LiDAR sector features."""

import math
import time

import numpy as np

from types import SimpleNamespace

from rl_mpc_training.utils.lidar_features import (
    compute_lidar_features
)

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


# ========================================================
# 1. Synthetic test
# ========================================================

def test_synthetic():

    print('\n[1] Testing synthetic LiDAR')

    n = 281

    ranges = np.full(
        n,
        np.inf,
        dtype=np.float64
    )

    angle_min = -1.22
    angle_increment = 2.44 / (n - 1)

    angles = (
        angle_min
        + np.arange(n) * angle_increment
    )

    # Five known obstacle returns.
    test_returns = [
        (-1.0, 1.5),
        (-0.5, 2.0),
        (0.0, 0.8),
        (0.5, 3.0),
        (1.0, 2.5),
    ]

    for angle, distance in test_returns:

        idx = int(
            np.argmin(np.abs(angles - angle))
        )

        ranges[idx] = distance

    scan = SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.15,
        range_max=12.0
    )

    result = compute_lidar_features(scan)

    expected = np.array(
        [1.5, 2.0, 0.8, 3.0, 2.5]
    )

    assert np.allclose(
        result['distances_m'],
        expected,
        atol=1e-6
    )

    assert np.all(result['valid_mask'])
    assert np.all(result['hit_mask'])

    print('[PASS] Five sector distances')

    # ----------------------------------------------------
    # Test invalid measurements
    # ----------------------------------------------------

    scan.ranges = np.full(n, np.nan)

    result = compute_lidar_features(scan)

    assert not np.any(result['valid_mask'])
    assert not np.any(result['hit_mask'])

    assert np.allclose(
        result['distances_m'],
        12.0
    )

    print('[PASS] Invalid measurements handled')

    # ----------------------------------------------------
    # Test no-return measurements
    # ----------------------------------------------------

    scan.ranges = np.full(n, np.inf)

    result = compute_lidar_features(scan)

    assert np.all(result['valid_mask'])
    assert not np.any(result['hit_mask'])

    assert np.allclose(
        result['distances_m'],
        12.0
    )

    print('[PASS] No-return measurements handled')


# ========================================================
# 2. ROS2/Gazebo test
# ========================================================

def test_ros():

    env = RLMPCEnv()

    try:

        print('\n[2] ROS2 LiDAR integration test')

        bridge = env.bridge

        baseline = bridge.scan_count

        deadline = time.monotonic() + 10.0

        scan = None

        while time.monotonic() < deadline:

            if (
                bridge.scan is not None
                and bridge.scan_count > baseline
            ):

                scan = bridge.scan
                break

            time.sleep(0.05)

        if scan is None:
            raise TimeoutError(
                'Fresh /scan_filtered message unavailable'
            )

        # ------------------------------------------------
        # Sensor metadata
        # ------------------------------------------------

        angle_last = (
            scan.angle_min
            + (len(scan.ranges) - 1)
            * scan.angle_increment
        )

        print('\n[3] Sensor information')

        print('Frame:', scan.header.frame_id)
        print('Number of rays:', len(scan.ranges))

        print(
            'Angle min:',
            math.degrees(scan.angle_min),
            'deg'
        )

        print(
            'Angle max:',
            math.degrees(angle_last),
            'deg'
        )

        print('Range min:', scan.range_min)
        print('Range max:', scan.range_max)

        # ------------------------------------------------
        # Sector features
        # ------------------------------------------------

        result = compute_lidar_features(scan)

        distances = result['distances_m']
        valid = result['valid_mask']
        hits = result['hit_mask']

        print('\n[4] LiDAR sector features')

        for i in range(5):

            print(
                f'Sector {i + 1}: '
                f'd={distances[i]:.3f} m | '
                f'valid={bool(valid[i])} | '
                f'hit={bool(hits[i])} | '
                f'usable_rays={result["valid_counts"][i]} | '
                f'total_rays={result["sample_counts"][i]}'
            )

        # ------------------------------------------------
        # Validate results
        # ------------------------------------------------

        assert distances.shape == (5,)
        assert distances.dtype == np.float32

        assert np.isfinite(distances).all()

        assert np.all(
            distances >= scan.range_min
        )

        assert np.all(
            distances <= scan.range_max
        )

        assert np.all(
            result['sample_counts'] > 0
        ), 'One or more sectors are outside scan coverage'

        print('\n[PASS] LiDAR sector integration test')

    finally:

        print('\nClosing environment')
        env.close()


def main():

    test_synthetic()

    print('\n[PASS] Synthetic LiDAR tests completed')

    test_ros()


if __name__ == '__main__':
    main()
