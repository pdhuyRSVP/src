#!/usr/bin/env python3
"""Step 4.3e.2: one live Gazebo episode using SAC command coordinates.

No optimization. Actor commands u are saved in Replay Buffer, while the
state-dependent adapter outputs the actual MPC actions a=T(obs,u).
Gazebo only. Do NOT run on physical hardware.
"""

import time

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent
from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

from rl_mpc_training.training.test_sac_collection_episode import (
    validate_observation,
)

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc,
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION,
    MAX_DELTA,
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer,
)

from rl_mpc_training.utils.sac_transition import (
    build_sac_transition,
)

from rl_mpc_training.utils.sac_training_action_adapter import (
    admissible_action_bounds,
    command_to_mpc_action,
    mpc_action_to_command,
    previous_action_from_observation,
)


# ======================================================
# Configuration
# ======================================================

SEED = 431
WARMUP_STEPS = 5

MAX_STEPS = 160
MAX_WALL_SEC = 240.0

BUFFER_CAPACITY = 256
BATCH_SIZE = 16


# ======================================================
# 1. Verify Replay Buffer
# ======================================================

def verify_buffer(
    buffer,
    commands,
    mpc_actions,
    episode_return
):
    """
    Replay action must be SAC command,
    never physical MPC action.
    """

    print(
        '\n[3] Verify Replay Buffer action semantics'
    )

    n = len(commands)

    assert n >= BATCH_SIZE

    assert len(buffer) == buffer.total_added == n

    assert buffer.can_sample(BATCH_SIZE)

    # --------------------------------------------------
    # Minibatch
    # --------------------------------------------------

    batch = buffer.sample(BATCH_SIZE)

    assert batch['observations'].shape == (
        BATCH_SIZE, 30
    )

    assert batch['actions'].shape == (
        BATCH_SIZE, 5
    )

    assert batch['rewards'].shape == (
        BATCH_SIZE, 1
    )

    assert batch['bootstrap_masks'].shape == (
        BATCH_SIZE, 1
    )

    print('[PASS] SAC minibatch dimensions')

    # --------------------------------------------------
    # Read all transitions
    # --------------------------------------------------

    full = buffer.sample(n)

    order = np.argsort(
        full['step_indices']
    )

    # Step indices.
    np.testing.assert_array_equal(
        full['step_indices'][order],
        np.arange(1, n + 1)
    )

    # Episode indices.
    np.testing.assert_array_equal(
        full['episode_indices'][order],
        np.zeros(n, dtype=np.int64)
    )

    # Observation continuity.
    np.testing.assert_array_equal(
        full['observations'][order][1:],
        full['next_observations'][order][:-1],
    )

    # --------------------------------------------------
    # Verify SAC commands
    # --------------------------------------------------

    stored_commands = full['actions'][order]

    np.testing.assert_array_equal(
        stored_commands,
        np.stack(commands)
    )

    # Stored commands must differ from physical MPC actions.
    assert not np.array_equal(
        stored_commands,
        np.stack(mpc_actions)
    )

    print(
        '[PASS] Replay Buffer stores SAC commands, '
        'NOT applied MPC actions'
    )

    # --------------------------------------------------
    # Reconstruct applied MPC actions
    # --------------------------------------------------

    for row, actual_mpc in zip(
        order,
        mpc_actions
    ):

        calculated = command_to_mpc_action(
            full['observations'][row],
            full['actions'][row]
        )

        np.testing.assert_array_equal(
            calculated,
            actual_mpc
        )

    print(
        '[PASS] Every stored command reproduces '
        'the applied MPC action'
    )

    # --------------------------------------------------
    # Reward consistency
    # --------------------------------------------------

    rewards = full['rewards'][order, 0]

    np.testing.assert_allclose(
        rewards,
        full['reward_components'][order].sum(axis=1),
        atol=2e-5,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        rewards.sum(dtype=np.float64),
        episode_return,
        atol=1e-3,
        rtol=1e-6,
    )

    # --------------------------------------------------
    # Terminal bootstrap masks
    # --------------------------------------------------

    masks = full['bootstrap_masks'][order, 0]

    assert np.all(masks[:-1] == 1.0)

    assert masks[-1] == 0.0

    phases = [
        full['phases'][int(row)]
        for row in order
    ]

    assert all(
        p == 'EPISODE_RUNNING'
        for p in phases[:-1]
    )

    assert phases[-1] == 'EPISODE_SUCCESS'

    assert bool(
        full['terminated'][order[-1]]
    )

    assert not bool(
        full['truncated'][order[-1]]
    )

    print(
        '[PASS] Indices, observation continuity, '
        'rewards and terminal mask'
    )


# ======================================================
# 2. Main
# ======================================================

def main():

    print(
        '\nStep 4.3e.2 - '
        'Live SAC Command -> Adapter -> MPC Test'
    )

    scenario = input(
        'Type CLEAR to start: '
    ).strip().upper()

    if scenario != 'CLEAR':

        print(
            'Test cancelled (only CLEAR is enabled)'
        )

        return

    # --------------------------------------------------
    # Initialize SAC
    # --------------------------------------------------

    torch.manual_seed(SEED)

    torch.set_num_threads(1)

    agent = SACAgent(
        obs_dim=30,
        action_dim=5,
        device='cpu'
    )

    # Save Actor weights.
    actor_before = [
        p.detach().clone()
        for p in agent.actor.parameters()
    ]

    # --------------------------------------------------
    # Replay Buffer
    # --------------------------------------------------

    buffer = SACReplayBuffer(
        capacity=BUFFER_CAPACITY,
        seed=SEED
    )

    # --------------------------------------------------
    # Gymnasium
    # --------------------------------------------------

    env = RLMPCEnv()

    try:

        # ==================================================
        # 1. Reset
        # ==================================================

        print(
            '\n[1] Reset environment and '
            'verify initial MPC parameters'
        )

        observation, reset_info = env.reset(
            seed=123
        )

        validate_observation(
            env,
            observation
        )

        assert (
            reset_info['phase']
            == 'EPISODE_RUNNING'
        )

        np.testing.assert_allclose(
            reset_info['raw_observation'][25:30],
            map_action_to_mpc(DEFAULT_ACTION),
            atol=1e-5,
            rtol=0.0,
        )

        # Verify action reconstruction.
        np.testing.assert_allclose(
            previous_action_from_observation(
                observation
            ),
            DEFAULT_ACTION,
            atol=3e-5,
            rtol=0.0,
        )

        print(
            '[PASS] Initial observation '
            'decodes DEFAULT action'
        )

        # ==================================================
        # 2. Episode variables
        # ==================================================

        previous_applied = DEFAULT_ACTION.copy()

        commands = []
        mpc_actions = []

        episode_return = 0.0

        modified_steps = 0
        terminal_seen = False

        start_wall = time.monotonic()

        print(
            '\n[2] One live CLEAR episode '
            '(no SAC training)'
        )

        # ==================================================
        # 3. Episode loop
        # ==================================================

        for step_index in range(
            1,
            MAX_STEPS + 1
        ):

            if (
                time.monotonic() - start_wall
                > MAX_WALL_SEC
            ):

                raise TimeoutError(
                    'Episode wall-time limit exceeded'
                )

            # ----------------------------------------------
            # Verify observation encodes previous action
            # ----------------------------------------------

            np.testing.assert_allclose(
                previous_action_from_observation(
                    observation
                ),
                previous_applied,
                atol=3e-5,
                rtol=0.0,
            )

            # ----------------------------------------------
            # SAC command selection
            # ----------------------------------------------

            if step_index <= WARMUP_STEPS:

                # DEFAULT command still passes
                # through the Adapter.
                command = DEFAULT_ACTION.copy()

            else:

                command = agent.act(
                    observation,
                    deterministic=True
                )

            assert command.shape == (5,)

            assert command.dtype == np.float32

            assert np.isfinite(command).all()

            assert env.action_space.contains(
                command
            )

            # ----------------------------------------------
            # Training Action Adapter
            # ----------------------------------------------

            low, high = admissible_action_bounds(
                observation
            )

            mpc_action = command_to_mpc_action(
                observation,
                command
            )

            assert env.action_space.contains(
                mpc_action
            )

            assert np.all(
                mpc_action >= low - 1e-6
            )

            assert np.all(
                mpc_action <= high + 1e-6
            )

            # ----------------------------------------------
            # Rate limit
            # ----------------------------------------------

            assert np.all(
                np.abs(
                    mpc_action - previous_applied
                ) <= MAX_DELTA + 1e-6
            )

            # ----------------------------------------------
            # Inverse mapping
            # ----------------------------------------------

            reconstructed = mpc_action_to_command(
                observation,
                mpc_action
            )

            np.testing.assert_allclose(
                reconstructed,
                command,
                atol=3e-5,
                rtol=0.0,
            )

            # ----------------------------------------------
            # Warmup
            # ----------------------------------------------

            if step_index <= WARMUP_STEPS:

                np.testing.assert_array_equal(
                    mpc_action,
                    DEFAULT_ACTION
                )

            if not np.array_equal(
                mpc_action,
                DEFAULT_ACTION
            ):

                modified_steps += 1

            # ==============================================
            # 4. Apply action to MPC
            # ==============================================

            # IMPORTANT:
            # Only the physical MPC action is sent.
            # The SAC command is never sent directly.

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(mpc_action)

            validate_observation(
                env,
                next_observation
            )

            assert np.isfinite(reward)

            assert not info.get(
                'diagnostic_reward_only',
                True
            )

            assert info['action_applied'], (
                f'Action not applied at step '
                f'{step_index}; '
                f'terminal={info["phase"]}'
            )

            # ----------------------------------------------
            # MPC feedback validation
            # ----------------------------------------------

            expected_params = np.asarray(
                map_action_to_mpc(
                    mpc_action
                )
            )

            if not (terminated or truncated):

                assert (
                    info['phase']
                    == 'EPISODE_RUNNING'
                )

                np.testing.assert_allclose(
                    info['mpc_params'],
                    expected_params,
                    atol=1e-6,
                    rtol=0.0,
                )

                np.testing.assert_allclose(
                    info['raw_observation'][25:30],
                    expected_params,
                    atol=1e-5,
                    rtol=0.0,
                )

                # The next observation must contain
                # the MPC action just applied.
                np.testing.assert_allclose(
                    previous_action_from_observation(
                        next_observation
                    ),
                    mpc_action,
                    atol=3e-5,
                    rtol=0.0,
                )

            elif not info['terminal_observation_reused']:

                np.testing.assert_allclose(
                    info['raw_observation'][25:30],
                    expected_params,
                    atol=1e-5,
                    rtol=0.0,
                )

            # ==============================================
            # 5. Build SAC Transition
            # ==============================================

            # IMPORTANT:
            # The training-environment action is COMMAND u.
            # It is not the mapped MPC action.

            transition, policy = build_sac_transition(
                observation=observation,

                action=command,

                reward=reward,

                next_observation=next_observation,

                terminated=terminated,
                truncated=truncated,

                info=info,

                episode_index=0,
                step_index=step_index,

                planning_failure_mode='external_interruption',
            )

            assert policy['store_transition'], (
                policy['reason']
            )

            assert transition is not None

            buffer.add(
                transition
            )

            # ----------------------------------------------
            # Store separate diagnostics
            # ----------------------------------------------

            commands.append(
                command.copy()
            )

            mpc_actions.append(
                mpc_action.copy()
            )

            episode_return += float(
                reward
            )

            # ==============================================
            # 6. Progress log
            # ==============================================

            if (
                step_index % 5 == 0
                or terminated
                or truncated
            ):

                print(
                    f'[STEP {step_index:03d}] '
                    f'command='
                    f'{np.round(command, 3).tolist()} | '
                    f'mpc_action='
                    f'{np.round(mpc_action, 3).tolist()} | '
                    f'params='
                    f'{np.round(expected_params, 3).tolist()} | '
                    f'r={reward:.3f} | '
                    f'buffer={len(buffer)} | '
                    f'mask={policy["bootstrap_mask"]:.1f}'
                )

            # ----------------------------------------------
            # Next observation
            # ----------------------------------------------

            previous_applied = (
                mpc_action.copy()
            )

            observation = (
                next_observation.copy()
            )

            # ==============================================
            # 7. Terminal handling
            # ==============================================

            if terminated or truncated:

                terminal_seen = True

                print(
                    f'\n[TERMINAL] {info["phase"]}'
                )

                break

        # ==================================================
        # 8. Verify episode
        # ==================================================

        assert terminal_seen, (
            'No terminal event received'
        )

        assert terminated and not truncated

        assert (
            info['phase']
            == 'EPISODE_SUCCESS'
        )

        assert not info['terminal_observation_reused']

        assert (
            policy['bootstrap_mask']
            == 0.0
        )

        assert len(buffer) == step_index

        assert modified_steps > 0

        print(
            '[PASS] Adapter changed '
            'applied MPC parameters'
        )

        # ==================================================
        # 9. Replay Buffer validation
        # ==================================================

        verify_buffer(
            buffer,
            commands,
            mpc_actions,
            episode_return
        )

        # ==================================================
        # 10. Verify no SAC training
        # ==================================================

        assert agent.update_count == 0

        for before, after in zip(
            actor_before,
            agent.actor.parameters()
        ):

            torch.testing.assert_close(
                before,
                after.detach(),
                rtol=0,
                atol=0
            )

        print(
            '[PASS] SAC Actor unchanged; '
            'no training performed'
        )

        # ==================================================
        # 11. Summary
        # ==================================================

        print(
            f'[SUMMARY] '
            f'steps={step_index}, '
            f'return={episode_return:.4f}, '
            f'nondefault_steps={modified_steps}, '
            f'transitions={len(buffer)}'
        )

        print(
            '\n[PASS] SAC training-action Adapter '
            'Gazebo test completed'
        )

    finally:

        print('\nClosing environment')

        env.close()


if __name__ == '__main__':
    main()
