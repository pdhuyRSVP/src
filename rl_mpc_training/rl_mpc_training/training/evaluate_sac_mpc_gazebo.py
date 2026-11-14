#!/usr/bin/env python3

"""Step 4.3e.6b.1 - Frozen SAC-MPC policy evaluation in Gazebo."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rl_mpc_training.training.export_sac_actor import (
    load_exported_actor
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION
)

from rl_mpc_training.utils.sac_training_action_adapter import (
    command_to_mpc_action,
    previous_action_from_observation
)

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc
)


POLICY_PATH = (
    Path.home()
    / 'agv_ws'
    / 'sac_runs'
    / 'clear_e5c'
    / 'policies'
    / 'actor_ep030.pt'
)

MAX_EPISODE_STEPS = 160
MAX_EPISODE_WALL_SEC = 240.0
WARMUP_STEPS = 5


def run_evaluation(policy_path, evaluation_mode='frozen'):

    # ================================================
    # 1. Configure evaluation mode
    # ================================================

    if evaluation_mode not in ('frozen', 'fixed'):
        raise ValueError(
            f'Unsupported evaluation mode: {evaluation_mode}'
        )

    torch.set_num_threads(1)

    actor = None

    if evaluation_mode == 'frozen':

        actor, metadata = load_exported_actor(
            policy_path
        )

        actor.eval()
        actor.requires_grad_(False)

        print('\n[MODE] Frozen SAC-MPC')
        print('[POLICY] Frozen Actor loaded')
        print(f'[METADATA] {metadata}')

    else:

        print('\n[MODE] Fixed MPC baseline')

        print(
            '[FIXED PARAMS] '
            '[1.0, 1.0, 1.0, 1.0, 0.5]'
        )

    # ================================================
    # 2. Create evaluation environment
    # ================================================

    # Import lazily: CPU policy-loading code remains
    # independent of the ROS2/Gazebo environment.
    from rl_mpc_training.env.rl_mpc_env import (
        RLMPCEnv
    )

    env = RLMPCEnv()

    episode_return = 0.0
    applied_steps = 0

    phase = None
    completed = False

    start_wall = time.monotonic()

    try:

        # ============================================
        # 3. Reset environment
        # ============================================

        observation, reset_info = env.reset(
            seed=431
        )

        if reset_info['phase'] != 'EPISODE_RUNNING':
            raise RuntimeError(
                'Evaluation reset did not start episode'
            )

        print('\n[PASS] Evaluation reset')
        print(
            f'[EVALUATION] '
            f'{evaluation_mode.upper()} MPC started'
        )

        # ============================================
        # 4. Evaluation loop
        # ============================================

        for step_index in range(
            1,
            MAX_EPISODE_STEPS + 1
        ):

            elapsed_wall = (
                time.monotonic() - start_wall
            )

            if elapsed_wall > MAX_EPISODE_WALL_SEC:
                raise TimeoutError(
                    'Evaluation wall-time limit exceeded'
                )

            # ----------------------------------------
            # Validate observation
            # ----------------------------------------

            obs = np.asarray(
                observation,
                dtype=np.float32
            )

            if not env.observation_space.contains(obs):
                raise ValueError(
                    'Invalid evaluation observation'
                )

            # ----------------------------------------
            # Select MPC parameter source
            # ----------------------------------------

            if evaluation_mode == 'fixed':

                # Fixed MPC for the entire episode.
                command = DEFAULT_ACTION.copy()

                mode = 'fixed'

            elif step_index <= WARMUP_STEPS:

                # Same warmup as SAC training.
                command = DEFAULT_ACTION.copy()

                mode = 'warmup'

            else:

                # Deterministic frozen SAC Actor.
                command = actor.act(
                    obs,
                    deterministic=True
                )

                mode = 'frozen_actor'

            # ----------------------------------------
            # Convert policy command to MPC action
            # ----------------------------------------

            mpc_action = command_to_mpc_action(
                obs,
                command
            )

            expected_params = np.asarray(
                map_action_to_mpc(mpc_action),
                dtype=np.float64
            )

            # ----------------------------------------
            # Fixed MPC parameter verification
            # ----------------------------------------

            if evaluation_mode == 'fixed':

                fixed_params = np.array(
                    [1.0, 1.0, 1.0, 1.0, 0.5],
                    dtype=np.float64
                )

                np.testing.assert_allclose(
                    expected_params,
                    fixed_params,
                    rtol=0.0,
                    atol=1e-8
                )

            # ----------------------------------------
            # Apply action through existing environment
            # ----------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(mpc_action)

            if not np.isfinite(reward):
                raise ValueError(
                    'Non-finite evaluation reward'
                )

            episode_return += float(reward)

            phase = info['phase']

            # ----------------------------------------
            # Validate applied MPC action
            # ----------------------------------------

            if not (terminated or truncated):

                if (
                    phase != 'EPISODE_RUNNING'
                    or not info['action_applied']
                ):
                    raise RuntimeError(
                        'Invalid evaluation transition'
                    )

                np.testing.assert_allclose(
                    info['raw_observation'][25:30],
                    expected_params,
                    rtol=0.0,
                    atol=1e-5
                )

                np.testing.assert_allclose(
                    previous_action_from_observation(
                        next_observation
                    ),
                    mpc_action,
                    rtol=0.0,
                    atol=3e-5
                )

            if info['action_applied']:
                applied_steps += 1

            # ----------------------------------------
            # Print results
            # ----------------------------------------

            if (
                step_index % 10 == 0
                or terminated
                or truncated
            ):

                print(
                    f'[EVAL STEP {step_index:03d}] '
                    f'mode={mode} '
                    f'phase={phase} '
                    f'ack={info["action_applied"]} '
                    f'mpc={np.round(expected_params, 3).tolist()} '
                    f'r={reward:.3f}',
                    flush=True
                )

            observation = next_observation

            # ----------------------------------------
            # Terminal
            # ----------------------------------------

            if terminated or truncated:

                completed = True

                print('\n[EVALUATION SUMMARY]')

                print(f'Phase: {phase}')

                print(
                    f'Steps: {step_index}'
                )

                print(
                    f'Applied actions: {applied_steps}'
                )

                print(
                    f'Return: {episode_return:.3f}'
                )

                print(
                    f'Wall time: '
                    f'{time.monotonic() - start_wall:.2f} s'
                )

                break

        if not completed:
            raise RuntimeError(
                'Evaluation exceeded maximum steps'
            )

        if phase != 'EPISODE_SUCCESS':
            raise RuntimeError(
                f'Evaluation did not succeed: {phase}'
            )

        print(
            f'\n[PASS] '
            f'{evaluation_mode.upper()} MPC evaluation'
        )

    finally:

        # Always request MPC HOLD on exit.
        env.close()

        print('[CLOSE] Evaluation environment closed')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        choices=('frozen', 'fixed'),
        default='frozen',
        help='MPC evaluation mode'
    )

    parser.add_argument(
        '--policy',
        type=Path,
        default=POLICY_PATH
    )

    # Leave ROS2 arguments for RLMPCEnv/rclpy.
    args, unknown = parser.parse_known_args()

    if unknown and unknown[0] != '--ros-args':
        parser.error(
            f'Unknown arguments: {unknown}'
        )

    run_evaluation(
        args.policy,
        evaluation_mode=args.mode
    )


if __name__ == '__main__':
    main()
