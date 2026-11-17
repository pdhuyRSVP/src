#!/usr/bin/env python3

"""Step 4.3d.2b: Controlled SAC -> Action Guard -> MPC in Gazebo.

One CLEAR episode, deterministic untrained Actor, no training updates.
Do not use this test on physical hardware.
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
    map_action_to_mpc
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION,
    MAX_DELTA,
    SACActionGuard,
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer
)

from rl_mpc_training.utils.sac_transition import (
    build_sac_transition
)


# ======================================================
# Configuration
# ======================================================

MAX_STEPS = 160
MAX_WALL_TIME_SEC = 240.0

BUFFER_CAPACITY = 256
BATCH_SIZE = 16

WARMUP_STEPS = 5


# ======================================================
# 1. Validate guarded action
# ======================================================

def check_action(guard, action, previous):

    """Check guarded proposal before sending to MPC."""

    assert action.shape == (5,)
    assert action.dtype == np.float32

    assert np.isfinite(action).all()

    # Guarded action envelope.
    assert np.all(
        action >= guard.lower - 1e-7
    )

    assert np.all(
        action <= guard.upper + 1e-7
    )

    # Maximum allowed action change.
    delta = np.abs(
        action - previous
    )

    assert np.all(
        delta <= MAX_DELTA + 1e-7
    )

    # MPC parameter mapping.
    params = np.asarray(
        map_action_to_mpc(action),
        dtype=np.float64
    )

    assert params.shape == (5,)
    assert np.isfinite(params).all()

    return params


# ======================================================
# 2. Verify Replay Buffer
# ======================================================

def verify_buffer(
    buffer,
    actions_sent,
    episode_return
):

    print('\n[3] Verify SAC Replay Buffer')

    count = len(actions_sent)

    assert count >= BATCH_SIZE

    assert len(buffer) == count
    assert buffer.total_added == count

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

    assert batch['next_observations'].shape == (
        BATCH_SIZE, 30
    )

    assert batch['bootstrap_masks'].shape == (
        BATCH_SIZE, 1
    )

    print('[PASS] SAC minibatch dimensions')

    # --------------------------------------------------
    # Read all transitions
    # --------------------------------------------------

    data = buffer.sample(len(buffer))

    rows = np.argsort(
        data['step_indices']
    )

    # Step indices.
    np.testing.assert_array_equal(
        data['step_indices'][rows],
        np.arange(1, count + 1)
    )

    # Episode indices.
    np.testing.assert_array_equal(
        data['episode_indices'][rows],
        np.zeros(count, dtype=np.int64)
    )

    # Observation continuity.
    np.testing.assert_array_equal(
        data['observations'][rows[1:]],
        data['next_observations'][rows[:-1]]
    )

    print(
        '[PASS] Episode indices and '
        'observation continuity'
    )

    # --------------------------------------------------
    # Applied action consistency
    # --------------------------------------------------

    np.testing.assert_array_equal(
        data['actions'][rows],
        np.stack(
            actions_sent,
            axis=0
        )
    )

    print(
        '[PASS] Stored actions are guarded '
        'actions actually sent to MPC'
    )

    # --------------------------------------------------
    # Reward consistency
    # --------------------------------------------------

    rewards = data['rewards'][rows, 0]

    components = data[
        'reward_components'
    ][rows]

    np.testing.assert_allclose(
        rewards,
        components.sum(axis=1),
        atol=2e-5,
        rtol=1e-6
    )

    np.testing.assert_allclose(
        rewards.sum(dtype=np.float64),
        episode_return,
        atol=1e-3,
        rtol=1e-6
    )

    # --------------------------------------------------
    # Terminal validation
    # --------------------------------------------------

    masks = data[
        'bootstrap_masks'
    ][rows, 0]

    assert np.all(
        masks[:-1] == 1.0
    )

    assert masks[-1] == 0.0

    phases = [
        data['phases'][int(i)]
        for i in rows
    ]

    assert all(
        phase == 'EPISODE_RUNNING'
        for phase in phases[:-1]
    )

    assert phases[-1] == 'EPISODE_SUCCESS'

    assert bool(
        data['terminated'][rows[-1]]
    )

    assert not bool(
        data['truncated'][rows[-1]]
    )

    assert bool(
        data['learning_terminated'][rows[-1]]
    )

    print(
        '[PASS] Reward totals and '
        'terminal bootstrap masks'
    )


# ======================================================
# 3. Main integration test
# ======================================================

def main():

    print(
        '\nStep 4.3d.2b - '
        'SAC Guarded MPC Gazebo Test'
    )

    scenario = input(
        'Type CLEAR to start: '
    ).strip().upper()

    if scenario != 'CLEAR':

        print(
            'Test cancelled; '
            'only CLEAR is enabled'
        )

        return

    # --------------------------------------------------
    # Initialize SAC Agent
    # --------------------------------------------------

    torch.manual_seed(431)
    torch.set_num_threads(1)

    agent = SACAgent(
        obs_dim=30,
        action_dim=5,
        device='cpu'
    )

    # Save Actor parameters for later verification.
    initial_actor = [
        p.detach().clone()
        for p in agent.actor.parameters()
    ]

    # --------------------------------------------------
    # Initialize Action Guard
    # --------------------------------------------------

    guard = SACActionGuard(
        warmup_steps=WARMUP_STEPS
    )

    # --------------------------------------------------
    # Initialize Replay Buffer
    # --------------------------------------------------

    buffer = SACReplayBuffer(
        capacity=BUFFER_CAPACITY,
        seed=431
    )

    # --------------------------------------------------
    # Initialize Gymnasium
    # --------------------------------------------------

    env = RLMPCEnv()

    try:

        # ==============================================
        # 1. Reset
        # ==============================================

        print(
            '\n[1] Reset Gymnasium and Action Guard'
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

        # Verify MPC reset parameters.
        np.testing.assert_allclose(
            reset_info['raw_observation'][25:30],
            map_action_to_mpc(DEFAULT_ACTION),
            rtol=0.0,
            atol=1e-5
        )

        guard.reset()

        print(
            '[PASS] Reset ready; '
            'Guard starts at DEFAULT'
        )

        # ==============================================
        # 2. Episode statistics
        # ==============================================

        start_wall = time.monotonic()

        actions_sent = []

        episode_return = 0.0

        changed_steps = 0
        clip_steps = 0

        terminal_received = False

        print(
            '\n[2] Execute guarded '
            'SAC actions in Gazebo'
        )

        # ==============================================
        # 3. Episode loop
        # ==============================================

        for step_index in range(
            1, MAX_STEPS + 1
        ):

            if (
                time.monotonic() - start_wall
                > MAX_WALL_TIME_SEC
            ):

                raise TimeoutError(
                    'Episode exceeded '
                    'wall-clock deadline'
                )

            # ------------------------------------------
            # SAC proposes deterministic action
            # ------------------------------------------

            raw_action = agent.act(
                observation,
                deterministic=True
            )

            # Save previously applied action.
            previous = (
                guard.last_applied.copy()
            )

            # ------------------------------------------
            # Action Guard
            # ------------------------------------------

            applied_action = guard.propose(
                raw_action,
                step_index
            )

            expected_params = check_action(
                guard,
                applied_action,
                previous
            )

            assert env.action_space.contains(
                applied_action
            )

            # Warmup must use DEFAULT.
            if step_index <= WARMUP_STEPS:

                np.testing.assert_array_equal(
                    applied_action,
                    DEFAULT_ACTION
                )

            # Count steps with non-default action.
            if not np.array_equal(
                applied_action,
                DEFAULT_ACTION
            ):

                changed_steps += 1

            # Count modified SAC proposals.
            if np.any(
                np.abs(raw_action - applied_action)
                > 1e-6
            ):

                clip_steps += 1

            # ------------------------------------------
            # Apply guarded action to MPC
            #
            # Raw SAC action is NEVER sent directly.
            # ------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(applied_action)

            validate_observation(
                env,
                next_observation
            )

            assert np.isfinite(reward)

            assert not info.get(
                'diagnostic_reward_only',
                True
            )

            # ------------------------------------------
            # Confirm MPC action application
            # ------------------------------------------

            acked = bool(
                info['action_applied']
            )

            guard.confirm(
                applied_action,
                action_applied=acked
            )

            if acked:

                np.testing.assert_array_equal(
                    guard.last_applied,
                    applied_action
                )

                assert (
                    guard.steps_applied
                    == step_index
                )

            # ------------------------------------------
            # Check MPC feedback on running steps
            # ------------------------------------------

            if not (terminated or truncated):

                assert acked, (
                    'Running step was not applied'
                )

                assert (
                    info['phase']
                    == 'EPISODE_RUNNING'
                )

                np.testing.assert_allclose(
                    info['raw_observation'][25:30],
                    expected_params,
                    rtol=0.0,
                    atol=1e-5
                )

            # ------------------------------------------
            # Build SAC Transition
            #
            # Store the action ACTUALLY sent to MPC.
            # ------------------------------------------

            transition, policy = (
                build_sac_transition(
                    observation=observation,

                    action=applied_action,

                    reward=reward,

                    next_observation=next_observation,

                    terminated=terminated,
                    truncated=truncated,

                    info=info,

                    episode_index=0,
                    step_index=step_index,

                    planning_failure_mode=(
                        'external_interruption'
                    )
                )
            )

            # ------------------------------------------
            # Store transition
            # ------------------------------------------

            if transition is not None:

                buffer.add(transition)

                actions_sent.append(
                    applied_action.copy()
                )

            else:

                print(
                    f'[SKIP] Step {step_index}: '
                    f'{policy["reason"]}'
                )

            episode_return += float(reward)

            # ------------------------------------------
            # Progress log
            # ------------------------------------------

            if (
                step_index % 5 == 0
                or terminated
                or truncated
            ):

                print(
                    f'[STEP {step_index:03d}] '
                    f'raw='
                    f'{np.round(raw_action, 2).tolist()} | '
                    f'applied='
                    f'{np.round(applied_action, 3).tolist()} | '
                    f'mpc='
                    f'{np.round(expected_params, 3).tolist()} | '
                    f'r={reward:.3f} | '
                    f'buffer={len(buffer)} | '
                    f'mask={policy["bootstrap_mask"]:.1f}'
                )

            # ------------------------------------------
            # Update observation
            # ------------------------------------------

            observation = (
                next_observation.copy()
            )

            # ------------------------------------------
            # Terminal handling
            # ------------------------------------------

            if terminated or truncated:

                terminal_received = True

                print(
                    f'\n[TERMINAL] '
                    f'{info["phase"]} | '
                    f'action_applied={acked}'
                )

                break

        # ==============================================
        # 4. Verify episode
        # ==============================================

        assert terminal_received, (
            'No terminal status reached'
        )

        assert (
            info['phase']
            == 'EPISODE_SUCCESS'
        ), info['phase']

        assert terminated and not truncated

        assert info['action_applied'], (
            'Terminal action was not applied; '
            'terminal transition is ineligible. '
            'Repeat with normal mission timing.'
        )

        assert policy['store_transition']

        assert transition is not None

        assert (
            len(actions_sent)
            == step_index
        )

        assert changed_steps > 0, (
            'Guard never applied '
            'a non-default action'
        )

        assert (
            guard.steps_applied
            == step_index
        )

        print(
            '[PASS] SAC-controlled MPC parameters '
            'changed within Guard'
        )

        # ==============================================
        # 5. Verify Replay Buffer
        # ==============================================

        verify_buffer(
            buffer,
            actions_sent,
            episode_return
        )

        # ==============================================
        # 6. Verify no SAC training
        # ==============================================

        assert agent.update_count == 0

        for before, after in zip(
            initial_actor,
            agent.actor.parameters()
        ):

            torch.testing.assert_close(
                before,
                after.detach(),
                rtol=0,
                atol=0
            )

        print(
            '[PASS] No SAC training; '
            'Actor weights unchanged'
        )

        # ==============================================
        # 7. Summary
        # ==============================================

        print(
            f'[SUMMARY] '
            f'steps={step_index} | '
            f'return={episode_return:.4f} | '
            f'nondefault_steps={changed_steps} | '
            f'guarded_proposals={clip_steps} | '
            f'stored={len(buffer)}'
        )

        print(
            '\n[PASS] SAC guarded Gazebo '
            'integration test completed'
        )

    finally:

        print('\nClosing environment')

        # Existing RLMPCEnv requests MPC HOLD.
        env.close()


if __name__ == '__main__':
    main()
