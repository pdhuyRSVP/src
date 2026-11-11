#!/usr/bin/env python3

"""Step 4.2.2e.1 - Validate observation construction."""

import numpy as np

from types import SimpleNamespace

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
    build_observation,
    OBSERVATION_DIM
)


def main():

    print('\n[1] Prepare synthetic input data')

    # --------------------------------------------
    # Geometry
    # --------------------------------------------

    path = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0]
    ])

    geometry = compute_path_geometry(
        [0.8, 0.2, 0.1],
        [1.0, 0.0, 0.0],
        path
    )

    # --------------------------------------------
    # LiDAR
    # --------------------------------------------

    n = 281

    ranges = np.full(n, np.inf)

    angles = np.linspace(
        -1.22,
        1.22,
        n
    )

    known_returns = [
        (-1.0, 1.5),
        (-0.5, 2.0),
        (0.0, 0.8),
        (0.5, 3.0),
        (1.0, 2.5)
    ]

    for angle, distance in known_returns:

        index = np.argmin(
            np.abs(angles - angle)
        )

        ranges[index] = distance

    scan = SimpleNamespace(
        ranges=ranges,
        angle_min=-1.22,
        angle_increment=2.44 / (n - 1),
        range_min=0.15,
        range_max=12.0
    )

    lidar = compute_lidar_features(scan)

    # --------------------------------------------
    # Motion and MPC
    # --------------------------------------------

    motion = compute_motion_features(
        0.3,
        -0.2,
        [1.0, 1.0, 1.0, 1.0, 0.5]
    )

    # --------------------------------------------
    # Build observation
    # --------------------------------------------

    print('\n[2] Build normalized observation')

    result = build_observation(
        geometry,
        lidar,
        motion,
        lidar_max_m=12.0
    )

    raw = result['raw']
    obs = result['normalized']

    print('Raw observation:', raw)
    print('Normalized observation:', obs)

    # --------------------------------------------
    # Validate
    # --------------------------------------------

    print('\n[3] Validate observation')

    assert raw.shape == (OBSERVATION_DIM,)
    assert obs.shape == (OBSERVATION_DIM,)

    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)

    print('[PASS] Observation shape and dtype')
    print('[PASS] Observation normalization range')

    # Confirm LiDAR distance order.
    np.testing.assert_allclose(
        raw[7:12],
        [1.5, 2.0, 0.8, 3.0, 2.5],
        atol=1e-6
    )

    print('[PASS] LiDAR feature positions')

    # Confirm motion positions.
    np.testing.assert_allclose(
        raw[22:25],
        [0.3, -0.2, 0.2],
        atol=1e-6
    )

    np.testing.assert_allclose(
        raw[25:30],
        [1.0, 1.0, 1.0, 1.0, 0.5],
        atol=1e-6
    )

    print('[PASS] Motion and MPC feature positions')

    # Test independent copies / no accidental mutation.
    assert not np.shares_memory(raw, obs)

    print('[PASS] Raw and normalized vectors independent')

    log_motion = compute_motion_features(
        0.3,
        -0.2,
        [1.0, 1.0, 10.0, 10.0, 0.5]
    )

    log_result = build_observation(
        geometry,
        lidar,
        log_motion,
        lidar_max_m=12.0
    )

    np.testing.assert_allclose(
        log_result['normalized'][27:29],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-6
    )

    print('[PASS] Logarithmic MPC normalization')

    print('\n[PASS] Observation builder test completed')


if __name__ == '__main__':
    main()
