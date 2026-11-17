#!/usr/bin/env python3

"""Step 4.2.4a - Synthetic reward tests."""

import numpy as np

from rl_mpc_training.utils.reward_function import (
    compute_reward
)


DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0]
)


def make_observation(distance=10.0):

    raw = np.zeros(30, dtype=np.float32)

    # Goal distance
    raw[4] = distance

    # LiDAR distances: all sectors have valid hits.
    raw[7:12] = 3.0

    raw[12:17] = 1.0
    raw[17:22] = 1.0

    # MPC parameters
    raw[25:30] = [
        1.0, 1.0, 1.0, 1.0, 0.5
    ]

    return raw


def reward(prev, curr, phase='EPISODE_RUNNING',
           prev_action=None, curr_action=None):

    if prev_action is None:
        prev_action = DEFAULT_ACTION

    if curr_action is None:
        curr_action = DEFAULT_ACTION

    return compute_reward(
        prev,
        curr,
        prev_action,
        curr_action,
        dt_s=0.5,
        phase=phase
    )


def main():

    print('\n[1] Testing reward components')

    previous = make_observation(10.0)

    current = make_observation(9.75)

    # --------------------------------------------
    # Forward progress
    # --------------------------------------------

    forward = reward(previous, current)

    assert forward['total'] > 0.0

    print('[PASS] Forward progress rewarded')

    # --------------------------------------------
    # No progress
    # --------------------------------------------

    stationary = reward(
        previous,
        make_observation(10.0)
    )

    assert forward['total'] > stationary['total']

    print('[PASS] Stationary behavior less rewarded')

    # --------------------------------------------
    # Moving away from goal
    # --------------------------------------------

    backward = reward(
        previous,
        make_observation(10.1)
    )

    assert backward['total'] < stationary['total']

    print('[PASS] Moving away penalized')

    # --------------------------------------------
    # Path error
    # --------------------------------------------

    path_error = current.copy()
    path_error[3] = 0.5

    with_path_error = reward(
        previous,
        path_error
    )

    assert with_path_error['total'] < forward['total']

    print('[PASS] Path tracking error penalized')

    # --------------------------------------------
    # Heading error
    # --------------------------------------------

    heading_error = current.copy()
    heading_error[2] = 0.5

    with_heading_error = reward(
        previous,
        heading_error
    )

    assert with_heading_error['total'] < forward['total']

    print('[PASS] Heading error penalized')

    # --------------------------------------------
    # Obstacle proximity
    # --------------------------------------------

    near_obstacle = current.copy()
    near_obstacle[7] = 0.3

    with_obstacle = reward(
        previous,
        near_obstacle
    )

    assert with_obstacle['total'] < forward['total']

    print('[PASS] Obstacle proximity penalized')

    # --------------------------------------------
    # Invalid LiDAR sector
    # --------------------------------------------

    unknown_scan = current.copy()

    unknown_scan[7] = 12.0
    unknown_scan[12] = 0.0
    unknown_scan[17] = 0.0

    with_unknown = reward(
        previous,
        unknown_scan
    )

    assert with_unknown['total'] < forward['total']

    print('[PASS] Unknown LiDAR sector penalized')

    # --------------------------------------------
    # Action variation
    # --------------------------------------------

    different_action = np.array(
        [-0.9, -0.9, -0.5, -0.5, 0.0]
    )

    changed = reward(
        previous,
        current,
        curr_action=different_action
    )

    assert changed['total'] < forward['total']

    print('[PASS] Action variation penalized')

    # --------------------------------------------
    # Terminal rewards
    # --------------------------------------------

    print('\n[2] Testing terminal rewards')

    success = reward(
        previous,
        current,
        phase='EPISODE_SUCCESS'
    )

    collision = reward(
        previous,
        current,
        phase='EPISODE_COLLISION'
    )

    timeout = reward(
        previous,
        current,
        phase='EPISODE_TIMEOUT'
    )

    planning_failure = reward(
        previous,
        current,
        phase='EPISODE_PLANNING_FAILURE'
    )

    assert success['total'] > forward['total']

    assert collision['total'] < timeout['total']

    assert timeout['total'] < forward['total']

    assert planning_failure['total'] < forward['total']

    print('[PASS] Success rewarded')
    print('[PASS] Collision penalized')
    print('[PASS] Timeout penalized')
    print('[PASS] Planning failure penalized')

    # --------------------------------------------
    # Invalid input
    # --------------------------------------------

    print('\n[3] Testing invalid inputs')

    try:

        compute_reward(
            previous,
            current,
            DEFAULT_ACTION,
            DEFAULT_ACTION,
            dt_s=0.0
        )

    except ValueError:

        print('[PASS] Invalid time interval rejected')

    else:

        raise AssertionError('Invalid dt accepted')

    try:

        compute_reward(
            previous,
            current,
            DEFAULT_ACTION,
            DEFAULT_ACTION,
            dt_s=0.5,
            phase='EP_WAIT_GOAL_HOLD'
        )

    except ValueError:

        print('[PASS] Non-terminal HOLD rejected')

    else:

        raise AssertionError('Finalizing phase accepted')

    print('\n[4] Example reward breakdown')

    print('Total reward:', forward['total'])

    for name, value in forward['components'].items():

        print(f'{name}: {value:.6f}')

    print('\n[PASS] Reward function tests completed')


if __name__ == '__main__':
    main()
