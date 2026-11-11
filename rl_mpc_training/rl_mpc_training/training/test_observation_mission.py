#!/usr/bin/env python3

"""Step 4.2.2f - Monitor observations during a complete mission."""

import time

import numpy as np

from rl_mpc_training.env.rl_mpc_env import (
    RLMPCEnv,
    TERMINAL_PHASES,
    ERROR_PHASES
)


SAMPLE_PERIOD_SEC = 0.5
MAX_WALL_TIME_SEC = 180.0

DEFAULT_PARAMS = [1.0, 1.0, 1.0, 1.0, 0.5]


def collect_running_observation(env, episode_path_stamp_ns):
    """
    Collect a new observation while the episode is running.

    Reuse the existing observation collector for this
    fixed-parameter integration test.
    """

    bridge = env.bridge

    snapshot = bridge.get_observation_snapshot()

    if (
        snapshot['accepted_path'] is None
        or snapshot['accepted_path_invalidated']
    ):
        raise RuntimeError('Accepted path unavailable')

    ack_seq = snapshot['adaptive_ack_seq']

    if ack_seq <= 0:
        raise RuntimeError('No MPC parameter ACK available')

    # Require new odometry, reference and LiDAR.
    # The accepted path may remain unchanged.
    baseline = {
        'accepted_path_count': max(
            0,
            snapshot['accepted_path_count'] - 1
        ),

        'reference_count': snapshot['reference_count'],

        'odom_count': snapshot['odom_count'],

        'scan_count': snapshot['scan_count'],
    }

    # MPC parameters remain unchanged in this test.
    ack_baseline = ack_seq - 1

    return env._initial_observation(
        baseline=baseline,
        ack_baseline=ack_baseline,
        episode_start_ns=episode_path_stamp_ns,
        timeout_sec=3.0
    )


def validate_observation(env, obs, raw, metadata, last_counts):

    if not env.observation_space.contains(obs):
        raise RuntimeError('Observation outside declared space')

    if obs.shape != (30,) or obs.dtype != np.float32:
        raise RuntimeError('Invalid observation shape or dtype')

    if raw.shape != (30,) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid raw observation')

    if not np.allclose(
        raw[25:30],
        DEFAULT_PARAMS,
        rtol=0.0,
        atol=1e-6
    ):
        raise RuntimeError('Unexpected MPC parameters')

    if metadata['sensor_skew_s'] > 0.3:
        raise RuntimeError('Sensor timestamp skew exceeded')

    if any(
        age < -0.1 or age > 0.5
        for age in metadata['sensor_ages_s'].values()
    ):
        raise RuntimeError('Stale sensor data')

    counts = metadata['observation_counts']

    # Require new sensor data for each observation.
    for name in (
        'reference_count',
        'odom_count',
        'scan_count'
    ):
        if counts[name] <= last_counts[name]:
            raise RuntimeError(
                f'No new {name} since previous observation'
            )

    # Path may remain the same, but must not regress.
    if (
        counts['accepted_path_count']
        < last_counts['accepted_path_count']
    ):
        raise RuntimeError('Accepted path count regressed')


def main():

    env = RLMPCEnv()

    try:

        print('\nFull Mission Observation Test')

        answer = input(
            '\nType START to execute one mission: '
        )

        if answer.strip() != 'START':
            print('Test cancelled')
            return

        # ------------------------------------------------
        # 1. Reset environment
        # ------------------------------------------------

        print('\n[1] Reset Environment')

        observation, info = env.reset(seed=123)

        print('[PASS] Initial observation received')

        raw_initial = info['raw_observation']

        initial_goal_distance = float(raw_initial[4])
        min_goal_distance = initial_goal_distance

        last_counts = dict(info['observation_counts'])

        # The first accepted path belongs to this episode.
        episode_path_stamp_ns = info['path_stamp_ns']

        bridge = env.bridge

        initial_solver_count = bridge.solver_success_count

        # ------------------------------------------------
        # 2. Initialize monitoring
        # ------------------------------------------------

        print('\n[2] Monitor observation during mission')

        start_wall = time.monotonic()

        start_sim_ns = bridge.get_clock().now().nanoseconds

        sample_period_ns = int(
            SAMPLE_PERIOD_SEC * 1e9
        )

        next_sample_ns = start_sim_ns + sample_period_ns

        observation_count = 0

        max_skew = float(info['sensor_skew_s'])

        max_velocity = float(raw_initial[22])

        min_lidar_distance = float(
            np.min(raw_initial[7:12])
        )

        final_status = None

        # ------------------------------------------------
        # 3. Monitor until episode terminal
        # ------------------------------------------------

        while (
            time.monotonic() - start_wall
            < MAX_WALL_TIME_SEC
        ):

            status = env._get_phase(
                bridge.episode_status_client
            )

            phase = status['phase']

            # ----------------------------------------
            # Terminal event
            # ----------------------------------------

            if phase in TERMINAL_PHASES:

                final_status = status
                break

            if phase in ERROR_PHASES:

                raise RuntimeError(
                    f'Episode Manager error: {status}'
                )

            # Episode Manager may wait for MPC HOLD
            # after receiving the goal event.
            if phase == 'EP_WAIT_GOAL_HOLD':

                time.sleep(0.1)
                continue

            if phase != 'EPISODE_RUNNING':

                raise RuntimeError(
                    f'Unexpected episode phase: {phase}'
                )

            # ----------------------------------------
            # Avoid collecting after robot is very
            # close to the goal. Wait for terminal.
            # ----------------------------------------

            if min_goal_distance <= 0.5:

                time.sleep(0.1)
                continue

            # ----------------------------------------
            # Sampling based on simulation time
            # ----------------------------------------

            now_sim_ns = (
                bridge.get_clock().now().nanoseconds
            )

            if now_sim_ns < next_sample_ns:

                time.sleep(0.05)
                continue

            # ----------------------------------------
            # Collect synchronized observation
            # ----------------------------------------

            try:

                obs, raw, metadata = (
                    collect_running_observation(
                        env,
                        episode_path_stamp_ns
                    )
                )

            except TimeoutError as exc:

                # A terminal transition can happen
                # while waiting for sensor updates.
                latest_status = env._get_phase(
                    bridge.episode_status_client
                )

                latest_phase = latest_status['phase']

                if latest_phase in TERMINAL_PHASES:

                    final_status = latest_status
                    break

                if latest_phase == 'EP_WAIT_GOAL_HOLD':

                    time.sleep(0.1)
                    continue

                raise RuntimeError(
                    f'Observation timeout during mission: {exc}'
                ) from exc

            # ----------------------------------------
            # Confirm episode is still running
            # ----------------------------------------

            latest_status = env._get_phase(
                bridge.episode_status_client
            )

            latest_phase = latest_status['phase']

            if latest_phase in TERMINAL_PHASES:

                final_status = latest_status
                break

            if latest_phase == 'EP_WAIT_GOAL_HOLD':

                time.sleep(0.1)
                continue

            if latest_phase != 'EPISODE_RUNNING':

                raise RuntimeError(
                    f'Episode changed phase: {latest_status}'
                )

            # ----------------------------------------
            # Validate observation
            # ----------------------------------------

            validate_observation(
                env,
                obs,
                raw,
                metadata,
                last_counts
            )

            # ----------------------------------------
            # Extract diagnostic values
            # ----------------------------------------

            distance = float(raw[4])

            e_lateral = float(raw[1])

            velocity = float(raw[22])

            lidar_min = float(
                np.min(raw[7:12])
            )

            skew = float(
                metadata['sensor_skew_s']
            )

            counts = metadata['observation_counts']

            # ----------------------------------------
            # Update statistics
            # ----------------------------------------

            observation_count += 1

            min_goal_distance = min(
                min_goal_distance,
                distance
            )

            max_velocity = max(
                max_velocity,
                velocity
            )

            min_lidar_distance = min(
                min_lidar_distance,
                lidar_min
            )

            max_skew = max(
                max_skew,
                skew
            )

            if (
                counts['accepted_path_count']
                > last_counts['accepted_path_count']
            ):
                print(
                    '[INFO] New accepted path detected'
                )

            last_counts = dict(counts)

            sim_elapsed = (
                bridge.get_clock().now().nanoseconds
                - start_sim_ns
            ) / 1e9

            print(
                f'[OBS {observation_count:03d}] '
                f't={sim_elapsed:.2f}s | '
                f'd_goal={distance:.3f}m | '
                f'e_lat={e_lateral:.3f}m | '
                f'v={velocity:.3f}m/s | '
                f'lidar_min={lidar_min:.3f}m | '
                f'skew={skew:.3f}s'
            )

            # Schedule next sample in simulation time.
            next_sample_ns = (
                bridge.get_clock().now().nanoseconds
                + sample_period_ns
            )

        # ------------------------------------------------
        # 4. Check terminal
        # ------------------------------------------------

        if final_status is None:

            raise TimeoutError(
                'Mission monitoring timeout'
            )

        print('\n[3] Episode terminal received')
        print(final_status)

        # ------------------------------------------------
        # 5. Final statistics
        # ------------------------------------------------

        progress = (
            initial_goal_distance - min_goal_distance
        )

        new_solver_results = (
            bridge.solver_success_count
            - initial_solver_count
        )

        print('\n[4] Observation statistics')

        print('Observation samples:', observation_count)

        print(
            'Initial goal distance:',
            initial_goal_distance
        )

        print(
            'Minimum observed goal distance:',
            min_goal_distance
        )

        print('Observed progress:', progress)

        print('Maximum observed velocity:', max_velocity)

        print(
            'Minimum LiDAR distance:',
            min_lidar_distance
        )

        print('Maximum sensor skew:', max_skew)

        print(
            'New successful MPC solver results:',
            new_solver_results
        )

        # ------------------------------------------------
        # 6. PASS criteria
        # ------------------------------------------------

        if observation_count < 8:

            raise RuntimeError(
                'Insufficient observation samples'
            )

        if progress < 2.0:

            raise RuntimeError(
                'Insufficient observed mission progress'
            )

        if new_solver_results < 5:

            raise RuntimeError(
                'Insufficient successful MPC solver results'
            )

        if final_status['phase'] != 'EPISODE_SUCCESS':

            raise RuntimeError(
                f'Mission did not succeed: {final_status}'
            )

        print('\n[PASS] Full mission observation test completed')

    finally:

        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
