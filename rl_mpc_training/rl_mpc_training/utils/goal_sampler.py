#!/usr/bin/env python3

"""Deterministic goal selection for SAC-MPC training."""

import math
import random


def select_episode_goal(
    goal_candidates,
    episode_index,
    seed=431,
    home=(0.0, 0.0),
    min_distance_m=2.0
):
    """
    Select a goal for a global training episode.

    Parameters
    ----------
    goal_candidates:
        Sequence of (x, y, yaw) in the map frame.

    episode_index:
        Zero-based global episode index.

    seed:
        Reproducible random seed.

    home:
        Fixed robot start position (x, y).

    min_distance_m:
        Minimum Euclidean distance from Home.

    Returns
    -------
    tuple:
        (goal_x, goal_y, goal_yaw)
    """

    # ================================================
    # 1. Validate configuration
    # ================================================

    if type(episode_index) is not int or episode_index < 0:
        raise ValueError(
            'episode_index must be a nonnegative integer'
        )

    if type(seed) is not int or seed < 0:
        raise ValueError(
            'seed must be a nonnegative integer'
        )

    if not math.isfinite(min_distance_m) or min_distance_m <= 0:
        raise ValueError(
            'min_distance_m must be positive and finite'
        )

    if len(home) != 2:
        raise ValueError(
            'home must contain (x, y)'
        )

    home_x, home_y = map(float, home)

    if not all(math.isfinite(v) for v in (home_x, home_y)):
        raise ValueError(
            'Home coordinates must be finite'
        )

    # ================================================
    # 2. Validate goal candidates
    # ================================================

    if not goal_candidates:
        raise ValueError(
            'Goal candidate list is empty'
        )

    validated_goals = []

    for index, goal in enumerate(goal_candidates):

        if len(goal) != 3:
            raise ValueError(
                f'Goal {index} must contain (x, y, yaw)'
            )

        x, y, yaw = map(float, goal)

        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError(
                f'Goal {index} contains non-finite values'
            )

        distance = math.hypot(
            x - home_x,
            y - home_y
        )

        if distance < min_distance_m:
            raise ValueError(
                f'Goal {index} is too close to Home'
            )

        validated_goals.append(
            (x, y, yaw)
        )

    if len(set(validated_goals)) != len(validated_goals):
        raise ValueError(
            'Duplicate goals are not allowed'
        )

    # ================================================
    # 3. Deterministic random permutation
    # ================================================

    number_of_goals = len(validated_goals)

    cycle_index = (
        episode_index // number_of_goals
    )

    position_in_cycle = (
        episode_index % number_of_goals
    )

    # Independent deterministic permutation per cycle.
    cycle_seed = (
        seed + cycle_index * 1_000_003
    )

    rng = random.Random(cycle_seed)

    goal_indices = list(
        range(number_of_goals)
    )

    rng.shuffle(goal_indices)

    selected_index = goal_indices[
        position_in_cycle
    ]

    # ================================================
    # 4. Return selected goal
    # ================================================

    return validated_goals[selected_index]
