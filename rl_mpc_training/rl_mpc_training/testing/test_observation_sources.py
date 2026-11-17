#!/usr/bin/env python3

"""Step 4.2.2a - Verify observation data sources."""

import math
import time

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


def main():

    env = RLMPCEnv()

    try:

        print('\nObservation Sources Test')

        answer = input(
            '\nType TEST to start one episode: '
        )

        if answer.strip() != 'TEST':

            print('Test cancelled')
            return

        observation, info = env.reset(seed=123)

        print('\n[1] Gymnasium reset completed')
        print('Observation:', observation)
        print('Info:', info)

        # ----------------------------------------
        # Wait for observation sources
        # ----------------------------------------

        print('\n[2] Waiting for observation sources')

        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:

            bridge = env.bridge

            path = bridge.accepted_path
            reference = bridge.reference
            scan = bridge.scan

            if (
                path is not None
                and len(path.poses) > 0
                and reference is not None
                and scan is not None
            ):
                break

            time.sleep(0.05)

        else:
            raise TimeoutError(
                'Accepted path, reference or LiDAR unavailable'
            )

        # ----------------------------------------
        # Verify Accepted Path
        # ----------------------------------------

        print('\n[3] Accepted Path')

        print('Frame:', path.header.frame_id)
        print('Number of poses:', len(path.poses))

        goal = path.poses[-1].pose.position

        print(
            f'Path endpoint: x={goal.x:.3f}, '
            f'y={goal.y:.3f}'
        )

        assert path.header.frame_id == 'map'
        assert len(path.poses) >= 2

        print('[PASS] Accepted path received')

        # ----------------------------------------
        # Verify MPC Reference
        # ----------------------------------------

        print('\n[4] MPC Reference')

        ref = reference.pose.position

        print('Frame:', reference.header.frame_id)

        print(
            f'Reference: x={ref.x:.3f}, '
            f'y={ref.y:.3f}'
        )

        assert reference.header.frame_id == 'map'

        print('[PASS] MPC reference received')

        # ----------------------------------------
        # Verify LiDAR
        # ----------------------------------------

        print('\n[5] LiDAR')

        valid_ranges = [
            float(r)
            for r in scan.ranges
            if (
                math.isfinite(r)
                and scan.range_min <= r <= scan.range_max
            )
        ]

        print('Number of scan points:', len(scan.ranges))
        print('Number of valid returns:', len(valid_ranges))
        print('Range min:', scan.range_min)
        print('Range max:', scan.range_max)

        if valid_ranges:
            print('Nearest valid return:', min(valid_ranges))
        else:
            print('No finite returns in valid range')

        assert len(scan.ranges) > 0

        print('[PASS] LiDAR received')

        # ----------------------------------------
        # Verify counters
        # ----------------------------------------

        print('\n[6] Message counters')

        print('Odometry:', bridge.odom_count)
        print('Accepted path:', bridge.accepted_path_count)
        print('Reference:', bridge.reference_count)
        print('LiDAR:', bridge.scan_count)

        assert bridge.odom_count > 0
        assert bridge.accepted_path_count > 0
        assert bridge.reference_count > 0
        assert bridge.scan_count > 0

        print('\n[PASS] Observation sources test completed')

    finally:

        print('\nClosing environment')

        env.close()


if __name__ == '__main__':
    main()
