#!/usr/bin/env python3

"""Step 4.2.3a - Test SAC action mapping."""

import numpy as np

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc,
    MPC_PARAM_MIN,
    MPC_PARAM_MAX
)


def check_mapping(action, expected, name):

    result = map_action_to_mpc(action)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=0.0,
        atol=1e-8
    )

    print(f'[PASS] {name}')
    print('  Action:', action)
    print('  MPC parameters:', result)


def check_invalid(action, name):

    try:

        map_action_to_mpc(action)

    except ValueError:

        print(f'[PASS] {name}')

    else:

        raise AssertionError(
            f'Invalid action accepted: {action}'
        )


def main():

    print('\nSAC Action Mapping Test')

    # ----------------------------------------------------
    # 1. Minimum action
    # ----------------------------------------------------

    check_mapping(
        [-1.0] * 5,
        MPC_PARAM_MIN,
        'Minimum action'
    )

    # ----------------------------------------------------
    # 2. Maximum action
    # ----------------------------------------------------

    check_mapping(
        [1.0] * 5,
        MPC_PARAM_MAX,
        'Maximum action'
    )

    # ----------------------------------------------------
    # 3. Center action
    # ----------------------------------------------------

    check_mapping(
        [0.0] * 5,
        [5.5, 10.5, 50.5, 50.5, 0.35],
        'Center action'
    )

    # ----------------------------------------------------
    # 4. MPC default parameters
    # ----------------------------------------------------

    check_mapping(
        [-1.0, -1.0, -1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.5],
        'Default MPC parameters'
    )

    # ----------------------------------------------------
    # 5. Mixed action
    # ----------------------------------------------------

    check_mapping(
        [-1.0, 1.0, 0.0, -1.0, 0.0],
        [1.0, 20.0, 50.5, 1.0, 0.35],
        'Mixed action'
    )

    # ----------------------------------------------------
    # 6. Invalid inputs
    # ----------------------------------------------------

    print('\nTesting invalid actions')

    check_invalid(
        [0.0, 0.0],
        'Incorrect action dimension'
    )

    check_invalid(
        [0.0, np.nan, 0.0, 0.0, 0.0],
        'NaN action'
    )

    check_invalid(
        [0.0, np.inf, 0.0, 0.0, 0.0],
        'Infinite action'
    )

    check_invalid(
        [0.0, 0.0, 1.2, 0.0, 0.0],
        'Action above upper bound'
    )

    check_invalid(
        [0.0, -1.1, 0.0, 0.0, 0.0],
        'Action below lower bound'
    )

    print('\n[PASS] SAC action mapping test completed')


if __name__ == '__main__':
    main()
