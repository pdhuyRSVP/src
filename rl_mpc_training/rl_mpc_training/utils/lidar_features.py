#!/usr/bin/env python3

"""LiDAR sector features for RL-MPC observations."""

import math
import numpy as np


# Five fixed sectors within the filtered front scan.
SECTOR_EDGES = np.array(
    [-1.22, -0.732, -0.244, 0.244, 0.732, 1.22],
    dtype=np.float64
)


def compute_lidar_features(scan):
    """
    Convert a LaserScan message into five sector distances.

    Distances are measured from the LiDAR origin.

    Returns:
        distances_m: 5 minimum distances in meters
        valid_mask:  5 booleans indicating usable measurements
        hit_mask:    5 booleans indicating finite returns
        valid_counts: Number of usable rays per sector
        sample_counts: Number of rays per sector
    """

    ranges = np.asarray(scan.ranges, dtype=np.float64)

    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError('Empty or invalid LaserScan')

    angle_min = float(scan.angle_min)
    angle_increment = float(scan.angle_increment)

    range_min = float(scan.range_min)
    range_max = float(scan.range_max)

    if not all(math.isfinite(x) for x in (
        angle_min,
        angle_increment,
        range_min,
        range_max
    )):
        raise ValueError('Invalid LaserScan metadata')

    if (
        angle_increment <= 0.0
        or range_min < 0.0
        or range_max <= range_min
    ):
        raise ValueError('Invalid LaserScan limits')

    # ----------------------------------------------------
    # 1. Compute actual angle of each ray
    # ----------------------------------------------------

    angles = (
        angle_min
        + np.arange(ranges.size) * angle_increment
    )

    # ----------------------------------------------------
    # 2. Classify scan values
    # ----------------------------------------------------

    finite_returns = (
        np.isfinite(ranges)
        & (ranges >= range_min)
        & (ranges <= range_max)
    )

    # +Inf is conventionally a no-return measurement.
    # Its meaning must still be checked against the
    # filtering pipeline used by the physical sensor.
    no_returns = np.isposinf(ranges)

    usable = finite_returns | no_returns

    # ----------------------------------------------------
    # 3. Allocate sector features
    # ----------------------------------------------------

    num_sectors = len(SECTOR_EDGES) - 1

    distances = np.full(
        num_sectors,
        range_max,
        dtype=np.float32
    )

    valid_mask = np.zeros(
        num_sectors,
        dtype=bool
    )

    hit_mask = np.zeros(
        num_sectors,
        dtype=bool
    )

    valid_counts = np.zeros(
        num_sectors,
        dtype=np.int32
    )

    sample_counts = np.zeros(
        num_sectors,
        dtype=np.int32
    )

    # ----------------------------------------------------
    # 4. Process each sector
    # ----------------------------------------------------

    for i in range(num_sectors):

        lower = SECTOR_EDGES[i]
        upper = SECTOR_EDGES[i + 1]

        if i == num_sectors - 1:
            sector = (
                (angles >= lower)
                & (angles <= upper)
            )
        else:
            sector = (
                (angles >= lower)
                & (angles < upper)
            )

        sample_counts[i] = np.count_nonzero(sector)

        sector_usable = sector & usable
        sector_hits = sector & finite_returns

        valid_counts[i] = np.count_nonzero(
            sector_usable
        )

        valid_mask[i] = (
            valid_counts[i] > 0
        )

        hit_mask[i] = bool(
            np.any(sector_hits)
        )

        if hit_mask[i]:

            distances[i] = float(
                np.min(ranges[sector_hits])
            )

        # Otherwise distance remains range_max.
        # valid_mask distinguishes no-return from
        # a sector without usable measurements.

    return {
        'distances_m': distances,
        'valid_mask': valid_mask,
        'hit_mask': hit_mask,
        'valid_counts': valid_counts,
        'sample_counts': sample_counts,
    }
