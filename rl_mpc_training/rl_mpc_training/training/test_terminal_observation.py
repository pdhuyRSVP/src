#!/usr/bin/env python3

"""Step 4.2.4b.2 - Synthetic terminal observation test."""

from types import SimpleNamespace as NS

import numpy as np

from rl_mpc_training.utils.terminal_observation import (
    build_terminal_observation_candidate
)


NOW_NS = 100_000_000_000
LAST_NS = 99_500_000_000
EPISODE_START_NS = 70_000_000_000


def make_stamp(ns):
    return NS(
        sec=int(ns // 1_000_000_000),
        nanosec=int(ns % 1_000_000_000)
    )


def make_header(ns, frame='map'):
    return NS(
        stamp=make_stamp(ns),
        frame_id=frame
    )


def make_pose(x, y):
    return NS(
        position=NS(x=x, y=y, z=0.0),
        orientation=NS(
            x=0.0, y=0.0, z=0.0, w=1.0
        )
    )


def make_inputs():

    path = NS(
        header=make_header(80_000_000_000),
        poses=[
            NS(pose=make_pose(0.0, 0.0)),
            NS(pose=make_pose(5.0, 0.0)),
            NS(pose=make_pose(10.0, 0.0))
        ]
    )

    reference = NS(
        header=make_header(99_800_000_000),
        pose=make_pose(9.9, 0.0)
    )

    odom = NS(
        header=make_header(99_950_000_000),
        child_frame_id='base_footprint',
        twist=NS(
            twist=NS(
                linear=NS(x=0.0),
                angular=NS(z=0.0)
            )
        )
    )

    scan = NS(
        header=make_header(
            99_940_000_000,
            frame='lidar_link'
        ),
        angle_min=-1.22,
        angle_increment=2.44 / 279.0,
        range_min=0.05,
        range_max=12.0,
        ranges=[3.0] * 280
    )

    transform = NS(
        header=make_header(99_960_000_000),
        child_frame_id='base_footprint',
        transform=NS(
            translation=NS(
                x=9.9, y=0.02, z=0.0
            ),
            rotation=NS(
                x=0.0, y=0.0, z=0.0, w=1.0
            )
        )
    )

    snapshot = {
        'accepted_path': path,
        'accepted_path_count': 1,
        'accepted_path_invalidated': False,
        'reference': reference,
        'odom': odom,
        'scan': scan,
        'adaptive_params': [
            1.0, 1.0, 1.0, 1.0, 0.5
        ],
        'adaptive_ack_seq': 1,
    }

    return snapshot, transform


def build(snapshot, transform):

    return build_terminal_observation_candidate(
        snapshot=snapshot,
        transform=transform,
        now_ns=NOW_NS,
        last_sample_ns=LAST_NS,
        episode_start_ns=EPISODE_START_NS
    )


def expect_rejection(snapshot, transform, name):

    try:
        build(snapshot, transform)

    except ValueError:
        print(f'[PASS] {name}')

    else:
        raise AssertionError(
            f'Invalid terminal candidate accepted: {name}'
        )


def main():

    print('\n[1] Valid terminal observation')

    snapshot, transform = make_inputs()

    result = build(snapshot, transform)

    obs = result['observation']
    raw = result['raw_observation']

    assert obs.shape == (30,)
    assert obs.dtype == np.float32

    assert raw.shape == (30,)
    assert raw.dtype == np.float32

    assert np.isfinite(obs).all()
    assert np.isfinite(raw).all()

    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)

    assert result['sample_ns'] > LAST_NS

    np.testing.assert_allclose(
        raw[25:30],
        [1.0, 1.0, 1.0, 1.0, 0.5],
        atol=1e-6
    )

    assert 0.0 < float(raw[4]) < 0.2

    print('[PASS] Fresh terminal observation built')
    print('Goal distance:', float(raw[4]))
    print('Sample timestamp:', result['sample_ns'])
    print('Metadata:', result['metadata'])

    print('\n[2] Invalid terminal observations')

    snapshot, transform = make_inputs()
    snapshot['accepted_path_invalidated'] = True

    expect_rejection(
        snapshot,
        transform,
        'Invalidated accepted path'
    )

    snapshot, transform = make_inputs()
    snapshot['reference'].header.stamp = make_stamp(
        98_000_000_000
    )

    expect_rejection(
        snapshot,
        transform,
        'Stale MPC reference'
    )

    snapshot, transform = make_inputs()
    snapshot['scan'].header.stamp = make_stamp(
        99_400_000_000
    )

    expect_rejection(
        snapshot,
        transform,
        'Sensor older than previous observation'
    )

    snapshot, transform = make_inputs()
    snapshot['adaptive_params'][2] = 150.0

    expect_rejection(
        snapshot,
        transform,
        'Out-of-range MPC parameters'
    )

    snapshot, transform = make_inputs()
    transform.child_frame_id = 'unexpected_frame'

    expect_rejection(
        snapshot,
        transform,
        'Incorrect TF frame'
    )

    print(
        '\n[PASS] Terminal observation builder tests completed'
    )


if __name__ == '__main__':
    main()
