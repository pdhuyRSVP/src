#!/usr/bin/env python3

"""Step 4.2.3d - Full Gymnasium step mission test."""

import time
import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv
from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)


DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0],
    dtype=np.float32
)

TEST_ACTION = np.array(
    [-0.9, -0.9, -0.5, -0.5, 0.0],
    dtype=np.float32
)

MAX_STEPS = 160
MAX_WALL_TIME_SEC = 240.0


def select_action(step_index):

    # Use a moderate action for five steps.
    if 6 <= step_index <= 10:
        return TEST_ACTION, 'TEST'

    return DEFAULT_ACTION, 'DEFAULT'


def check_observation(env, observation):

    if observation.shape != (30,):
        raise RuntimeError(
            'Invalid observation dimension'
        )

    if observation.dtype != np.float32:
        raise RuntimeError(
            'Invalid observation dtype'
        )

    if not env.observation_space.contains(observation):
        raise RuntimeError(
            'Observation outside observation_space'
        )


def main():

    env = RLMPCEnv()

    try:

        print('\nFull Gymnasium Step Mission Test')

        scenario = input(
            '\nType CLEAR or OBSTACLE to start: '
        ).strip().upper()

        if scenario not in ('CLEAR', 'OBSTACLE'):
            print('Test cancelled')
            return

        # --------------------------------------------
        # 1. Reset
        # --------------------------------------------

        print('\n[1] Reset Environment')

        observation, reset_info = env.reset(seed=123)

        check_observation(env, observation)

        initial_distance = float(
            reset_info['raw_observation'][4]
        )

        min_distance = initial_distance

        initial_solver_count = (
            env.bridge.solver_success_count
        )

        initial_path_count = (
            reset_info['observation_counts'][
                'accepted_path_count'
            ]
        )

        last_path_count = initial_path_count

        start_sim_ns = (
            env.bridge.get_clock().now().nanoseconds
        )

        start_wall = time.monotonic()

        # --------------------------------------------
        # 2. Statistics
        # --------------------------------------------

        actual_dt_values = []
        skew_values = []

        total_skipped_ticks = 0

        path_message_changes = 0
        action_updates = 0

        terminal_info = None

        episode_return = 0.0

        reward_values = []

        print('\n[2] Execute Gymnasium steps')

        # --------------------------------------------
        # 3. Full mission
        # --------------------------------------------

        for step_index in range(1, MAX_STEPS + 1):

            if (
                time.monotonic() - start_wall
                > MAX_WALL_TIME_SEC
            ):
                raise TimeoutError(
                    'Mission exceeded wall-time limit'
                )

            action, action_label = select_action(
                step_index
            )

            expected_params = map_action_to_mpc(
                action
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            # ----------------------------------------
            # Validate Gymnasium API
            # ----------------------------------------

            check_observation(env, observation)

            if not isinstance(reward, float):
                raise RuntimeError(
                    'Reward must be a float'
                )

            if not np.isfinite(reward):

                raise RuntimeError(
                    'Non-finite reward'
                )

            if not isinstance(terminated, bool):
                raise RuntimeError(
                    'Invalid terminated flag'
                )

            if not isinstance(truncated, bool):
                raise RuntimeError(
                    'Invalid truncated flag'
                )

            if info.get('diagnostic_reward_only', True):

                raise RuntimeError(
                    'Reward is still diagnostic'
                )


            # Verify the reward breakdown.
            components = info['reward_components']

            np.testing.assert_allclose(
                reward,
                sum(components.values()),
                rtol=0.0,
                atol=1e-6
            )

            episode_return += reward

            reward_values.append(reward)

            # ----------------------------------------
            # Check accepted path changes
            # ----------------------------------------

            path_snapshot = (
                env.bridge.get_observation_snapshot()
            )

            current_path_count = (
                path_snapshot['accepted_path_count']
            )

            if current_path_count < last_path_count:
                raise RuntimeError(
                    'Accepted path counter regressed'
                )

            if current_path_count > last_path_count:

                delta = (
                    current_path_count
                    - last_path_count
                )

                path_message_changes += delta

                print(
                    '[INFO] Accepted path message count '
                    f'changed: {last_path_count} -> '
                    f'{current_path_count}'
                )

                last_path_count = current_path_count

            # ----------------------------------------
            # Track applied actions
            # ----------------------------------------

            if info.get('action_applied', False):
                action_updates += 1

            # ----------------------------------------
            # Terminal handling
            # ----------------------------------------

            if terminated or truncated:

                terminal_info = info

                print('\n[3] Terminal received')

                print('Phase:', info.get('phase'))
                print('Event:', info.get('event'))

                print('Terminated:', terminated)
                print('Truncated:', truncated)

                print(
                    'Terminal observation reused:',
                    info.get(
                        'terminal_observation_reused'
                    )
                )

                print(
                    'Terminal observation source:',
                    info.get('terminal_observation_source')
                )

                print(
                    'Terminal sample timestamp:',
                    info.get('terminal_sample_ns')
                )

                print(
                    'Terminal observation reason:',
                    info.get('terminal_observation_reason')
                )

                print(
                    'Replay eligible:',
                    info.get('replay_eligible')
                )

                print(
                    'Bootstrap allowed:',
                    info.get('bootstrap_allowed')
                )

                print(
                    'Reward mode:',
                    info.get('reward_mode')
                )

                print(
                    'Terminal metadata:',
                    info.get('terminal_metadata')
                )

                print(
                    'Action applied:',
                    info.get('action_applied')
                )

                print(
                    'Episode status:',
                    info.get('episode_status')
                )

                print(
                    'Terminal reward:',
                    reward
                )

                print(
                    'Terminal reward components:',
                    info['reward_components']
                )

                print(
                    'Terminal reward mode:',
                    info['reward_mode']
                )

                break

            # ----------------------------------------
            # Validate running observation
            # ----------------------------------------

            if info['phase'] != 'EPISODE_RUNNING':
                raise RuntimeError(
                    'Unexpected running phase'
                )

            if info['terminal_observation_reused']:
                raise RuntimeError(
                    'Running observation was reused'
                )

            raw = info['raw_observation']

            np.testing.assert_allclose(
                raw[25:30],
                expected_params,
                rtol=0.0,
                atol=1e-5
            )

            # ----------------------------------------
            # Timing statistics
            # ----------------------------------------

            actual_dt = float(
                info['actual_dt_s']
            )

            sensor_skew = float(
                info['sensor_skew_s']
            )

            skipped_ticks = int(
                info['skipped_ticks']
            )

            if actual_dt <= 0.0:
                raise RuntimeError(
                    'Invalid step duration'
                )

            if not 0.0 <= sensor_skew <= 0.3:
                raise RuntimeError(
                    'Sensor timestamp skew exceeded'
                )

            actual_dt_values.append(actual_dt)
            skew_values.append(sensor_skew)

            total_skipped_ticks += skipped_ticks

            # ----------------------------------------
            # Observation statistics
            # ----------------------------------------

            distance = float(raw[4])
            e_lateral = float(raw[1])
            velocity = float(raw[22])

            min_distance = min(
                min_distance,
                distance
            )

            lidar_min = float(
                np.min(raw[7:12])
            )

            sim_elapsed = (
                env.bridge.get_clock().now().nanoseconds
                - start_sim_ns
            ) / 1e9

            print(
                f'[STEP {step_index:03d}] '
                f't={sim_elapsed:.2f}s | '
                f'action={action_label} | '
                f'd_goal={distance:.3f}m | '
                f'e_lat={e_lateral:.3f}m | '
                f'v={velocity:.3f}m/s | '
                f'lidar={lidar_min:.3f}m | '
                f'dt={actual_dt:.3f}s | '
                f'skipped={skipped_ticks} | '
                f'reward={reward:.3f}'
            )

        # --------------------------------------------
        # 4. Verify terminal
        # --------------------------------------------

        if terminal_info is None:
            raise RuntimeError(
                'Maximum steps reached without terminal'
            )

        if (
            terminal_info['phase']
            != 'EPISODE_SUCCESS'
        ):
            raise RuntimeError(
                'Mission did not succeed: '
                f'{terminal_info}'
            )

        # --------------------------------------------
        # 5. Summary
        # --------------------------------------------

        print('\n[4] Mission statistics')

        print(
            'Scenario:',
            scenario
        )

        print(
            'Completed running steps:',
            len(actual_dt_values)
        )

        print(
            'Action updates:',
            action_updates
        )

        print(
            'Initial goal distance:',
            initial_distance
        )

        print(
            'Minimum observed goal distance:',
            min_distance
        )

        print(
            'Observed progress:',
            initial_distance - min_distance
        )

        print(
            'Accepted path message changes:',
            path_message_changes
        )

        print(
            'Total skipped ticks:',
            total_skipped_ticks
        )

        print(
            'Episode return:',
            episode_return
        )

        print(
            'Reward samples:',
            len(reward_values)
        )

        print(
            'Mean reward:',
            float(np.mean(reward_values))
        )

        print(
            'Minimum reward:',
            float(np.min(reward_values))
        )

        print(
            'Maximum reward:',
            float(np.max(reward_values))
        )

        if actual_dt_values:

            dt_array = np.asarray(
                actual_dt_values
            )

            print(
                'Mean step dt:',
                float(np.mean(dt_array))
            )

            print(
                'Minimum step dt:',
                float(np.min(dt_array))
            )

            print(
                'Maximum step dt:',
                float(np.max(dt_array))
            )

            print(
                'Step dt standard deviation:',
                float(np.std(dt_array))
            )

        if skew_values:

            print(
                'Maximum sensor skew:',
                float(np.max(skew_values))
            )

        solver_delta = (
            env.bridge.solver_success_count
            - initial_solver_count
        )

        print(
            'New successful MPC solver results:',
            solver_delta
        )

        # --------------------------------------------
        # 6. Validate full mission
        # --------------------------------------------

        if len(actual_dt_values) < 8:
            raise RuntimeError(
                'Insufficient running step samples'
            )

        if initial_distance - min_distance < 2.0:
            raise RuntimeError(
                'Insufficient observed progress'
            )

        if solver_delta < 5:
            raise RuntimeError(
                'Insufficient MPC solver results'
            )

        # ----------------------------------------------------
        # Validate terminal observation policy
        # ----------------------------------------------------

        reused = terminal_info[
            'terminal_observation_reused'
        ]

        source = terminal_info[
            'terminal_observation_source'
        ]

        if reused:

            assert source == 'reused'

            assert (
                terminal_info['reward_mode']
                == 'event_only'
            )

            print(
                '[WARNING] Terminal observation reused: '
                f"{terminal_info['terminal_observation_reason']}"
            )

        else:

            assert source == 'fresh'

            assert (
                terminal_info['reward_mode']
                == 'full'
            )

            assert (
                terminal_info['terminal_sample_ns']
                is not None
            )

            print('[PASS] Fresh terminal observation received')


        # For this test, the episode must end with SUCCESS.
        assert terminal_info['phase'] == 'EPISODE_SUCCESS'

        assert terminal_info['replay_eligible'] is True

        assert terminal_info['bootstrap_allowed'] is False

        assert terminal_info['reward_components']['terminal'] == 25.0

        assert terminal_info['reward_mode'] in ('full', 'event_only')

        assert np.isfinite(episode_return)

        print('[PASS] Terminal reward calculation')

        print(
            '\n[PASS] Full Gymnasium step mission'
        )

        # --------------------------------------------
        # 7. Check that step after terminal is rejected
        # --------------------------------------------

        try:

            env.step(DEFAULT_ACTION)

        except RuntimeError as exc:

            if 'Episode is not active' not in str(exc):
                raise

            print(
                '[PASS] Step after terminal rejected'
            )

        else:

            raise AssertionError(
                'Step accepted after episode terminal'
            )

        # --------------------------------------------
        # 8. Replanning diagnostic
        # --------------------------------------------

        if scenario == 'OBSTACLE':

            if path_message_changes > 0:

                print(
                    '[PASS] Accepted path updates observed'
                )

            else:

                print(
                    '[WARNING] No accepted path update '
                    'observed. Replanning was not verified.'
                )

        print(
            '\n[PASS] Step mission test completed'
        )

    finally:

        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
