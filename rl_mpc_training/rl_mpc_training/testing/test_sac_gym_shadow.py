#!/usr/bin/env python3

"""Step 4.3d.1 - Live SAC Shadow Policy Integration Test.

The SAC policy observes the real Gymnasium state and proposes
actions, but its actions are NEVER sent to MPC.

Only previously validated DEFAULT/TEST actions control the robot.
No SAC learning updates are performed.
"""

import time

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import (
    SACAgent
)

from rl_mpc_training.env.rl_mpc_env import (
    RLMPCEnv
)

from rl_mpc_training.training.test_sac_collection_episode import (
    select_action,
    validate_observation
)

from rl_mpc_training.utils.action_mapping import (
    MPC_PARAM_MIN,
    MPC_PARAM_MAX,
    map_action_to_mpc
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

MINIBATCH_SIZE = 16


# ======================================================
# 1. Validate shadow action
# ======================================================

def check_shadow_action(
    agent,
    observation,
    action_space
):
    """
    Generate SAC actions from real Gymnasium observation.

    Actions are inspected only.
    They are NOT sent to MPC.
    """

    deterministic = agent.act(
        observation,
        deterministic=True
    )

    deterministic_again = agent.act(
        observation,
        deterministic=True
    )

    np.testing.assert_array_equal(
        deterministic,
        deterministic_again
    )

    stochastic = agent.act(
        observation,
        deterministic=False
    )

    for name, action in (
        ('deterministic', deterministic),
        ('stochastic', stochastic)
    ):

        assert action.shape == (5,), name

        assert action.dtype == np.float32, name

        assert np.isfinite(action).all(), name

        assert action_space.contains(action), name

        # Convert the proposed action to MPC parameters.
        # This only performs mathematical mapping.
        params = np.asarray(
            map_action_to_mpc(action),
            dtype=np.float64
        )

        assert params.shape == (5,), name

        assert np.isfinite(params).all(), name

        assert np.all(
            params >= MPC_PARAM_MIN - 1e-8
        ), name

        assert np.all(
            params <= MPC_PARAM_MAX + 1e-8
        ), name

    return deterministic, stochastic


# ======================================================
# 2. Verify Replay Buffer
# ======================================================

def verify_buffer(
    buffer,
    step_count,
    episode_return
):

    print('\n[3] Verify Replay Buffer')

    assert len(buffer) == step_count

    assert buffer.total_added == step_count

    assert buffer.can_sample(
        MINIBATCH_SIZE
    )

    # --------------------------------------------------
    # Sample minibatch
    # --------------------------------------------------

    batch = buffer.sample(
        MINIBATCH_SIZE
    )

    assert batch['observations'].shape == (
        MINIBATCH_SIZE, 30
    )

    assert batch['actions'].shape == (
        MINIBATCH_SIZE, 5
    )

    assert batch['rewards'].shape == (
        MINIBATCH_SIZE, 1
    )

    assert batch['bootstrap_masks'].shape == (
        MINIBATCH_SIZE, 1
    )

    print('[PASS] SAC-compatible minibatch')

    # --------------------------------------------------
    # Inspect all stored transitions
    # --------------------------------------------------

    data = buffer.sample(
        len(buffer)
    )

    order = np.argsort(
        data['step_indices']
    )

    steps = data['step_indices'][order]

    obs = data['observations'][order]

    next_obs = data['next_observations'][order]

    actions = data['actions'][order]

    rewards = data['rewards'][order, 0]

    masks = data['bootstrap_masks'][order, 0]

    phases = [
        data['phases'][int(i)]
        for i in order
    ]

    # --------------------------------------------------
    # Step indices
    # --------------------------------------------------

    np.testing.assert_array_equal(
        steps,
        np.arange(1, step_count + 1)
    )

    np.testing.assert_array_equal(
        data['episode_indices'][order],
        np.zeros(step_count, dtype=np.int64)
    )

    # --------------------------------------------------
    # Observation continuity
    # --------------------------------------------------

    np.testing.assert_array_equal(
        obs[1:],
        next_obs[:-1]
    )

    # --------------------------------------------------
    # Stored actions must be MPC-applied actions
    # --------------------------------------------------

    for i, step in enumerate(steps):

        actual_action, _ = select_action(
            int(step)
        )

        np.testing.assert_array_equal(
            actions[i],
            actual_action
        )

    # --------------------------------------------------
    # Reward consistency
    # --------------------------------------------------

    np.testing.assert_allclose(
        rewards,
        data['reward_components'][order].sum(axis=1),
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

    assert all(
        phase == 'EPISODE_RUNNING'
        for phase in phases[:-1]
    )

    assert phases[-1] == 'EPISODE_SUCCESS'

    assert np.all(
        masks[:-1] == 1.0
    )

    assert masks[-1] == 0.0

    assert bool(
        data['terminated'][order[-1]]
    )

    assert not bool(
        data['truncated'][order[-1]]
    )

    print(
        '[PASS] Observation continuity, '
        'reward, terminal mask'
    )

    print(
        '[PASS] Stored actions equal MPC-applied actions, '
        'not shadow actions'
    )


# ======================================================
# 3. Main test
# ======================================================

def main():

    print(
        '\nStep 4.3d.1 - '
        'Live Gymnasium SAC Shadow Policy Test'
    )

    scenario = input(
        '\nType CLEAR to start: '
    ).strip().upper()

    if scenario != 'CLEAR':

        print(
            'Test cancelled '
            '(only CLEAR is enabled in 4.3d.1)'
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
        gamma=0.99,
        tau=0.005,
        init_alpha=0.2,
        automatic_entropy_tuning=True,
        device='cpu'
    )

    # Save initial Actor parameters.
    # This test must not train the Actor.
    actor_before = [
        p.detach().clone()
        for p in agent.actor.parameters()
    ]

    # --------------------------------------------------
    # Replay Buffer
    # --------------------------------------------------

    buffer = SACReplayBuffer(
        capacity=BUFFER_CAPACITY,
        seed=431
    )

    # --------------------------------------------------
    # Gymnasium Environment
    # --------------------------------------------------

    env = RLMPCEnv()

    try:

        print('\n[1] Reset Gymnasium')

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

        print('[PASS] Gymnasium observation 30D')

        # --------------------------------------------------
        # Collection statistics
        # --------------------------------------------------

        episode_return = 0.0

        proposals = 0

        distinct_proposals = 0

        terminal_received = False

        start_wall = time.monotonic()

        print(
            '\n[2] Run shadow policy '
            'with validated MPC actions'
        )

        # ==================================================
        # Episode loop
        # ==================================================

        for step_index in range(
            1, MAX_STEPS + 1
        ):

            if (
                time.monotonic() - start_wall
                > MAX_WALL_TIME_SEC
            ):

                raise TimeoutError(
                    'Shadow episode exceeded wall-time limit'
                )

            # ----------------------------------------------
            # SAC proposes actions
            # ----------------------------------------------

            shadow_det, shadow_stoch = (
                check_shadow_action(
                    agent,
                    observation,
                    env.action_space
                )
            )

            proposals += 1

            # ----------------------------------------------
            # Select validated action for actual MPC
            # ----------------------------------------------

            applied_action, label = select_action(
                step_index
            )

            if not np.array_equal(
                shadow_stoch,
                applied_action
            ):

                distinct_proposals += 1

            # ----------------------------------------------
            # Gymnasium step
            #
            # IMPORTANT:
            # The SAC shadow actions are NOT applied.
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Build SAC Transition
            #
            # Store the action actually applied to MPC.
            # ----------------------------------------------

            transition, policy = build_sac_transition(
                observation=observation,

                action=applied_action,

                reward=reward,

                next_observation=next_observation,

                terminated=terminated,
                truncated=truncated,

                info=info,

                episode_index=0,
                step_index=step_index,

                planning_failure_mode='external_interruption'
            )

            if transition is None:

                raise AssertionError(
                    f'Ineligible transition at step '
                    f'{step_index}: {policy["reason"]}'
                )

            buffer.add(transition)

            episode_return += float(reward)

            # ----------------------------------------------
            # Confirm actually applied MPC parameters
            # ----------------------------------------------

            if not (terminated or truncated):

                assert (
                    info['phase']
                    == 'EPISODE_RUNNING'
                )

                np.testing.assert_allclose(
                    info['raw_observation'][25:30],
                    map_action_to_mpc(applied_action),
                    atol=1e-5,
                    rtol=0.0
                )

            # ----------------------------------------------
            # Progress log
            # ----------------------------------------------

            if (
                step_index % 5 == 0
                or terminated
                or truncated
            ):

                print(
                    f'[STEP {step_index:03d}] '
                    f'applied={label} | '
                    f'shadow_det='
                    f'{np.round(shadow_det, 2).tolist()} | '
                    f'shadow_stoch='
                    f'{np.round(shadow_stoch, 2).tolist()} | '
                    f'reward={reward:.3f} | '
                    f'buffer={len(buffer)} | '
                    f'mask={policy["bootstrap_mask"]:.1f}'
                )

            # ----------------------------------------------
            # Next observation
            # ----------------------------------------------

            observation = (
                next_observation.copy()
            )

            # ----------------------------------------------
            # Terminal
            # ----------------------------------------------

            if terminated or truncated:

                terminal_received = True

                print(
                    f'\n[TERMINAL] {info["phase"]}'
                )

                break

        # ==================================================
        # Verify episode
        # ==================================================

        print(
            '\n[3] Verify live shadow-mode collection'
        )

        assert terminal_received, (
            'No terminal event received'
        )

        assert (
            info['phase']
            == 'EPISODE_SUCCESS'
        ), info['phase']

        assert terminated and not truncated

        assert info['action_applied']

        assert policy['store_transition']

        assert (
            policy['bootstrap_mask']
            == 0.0
        )

        assert len(buffer) == step_index

        assert proposals == step_index

        assert distinct_proposals > 0

        # ----------------------------------------------
        # Replay Buffer verification
        # ----------------------------------------------

        verify_buffer(
            buffer,
            step_index,
            episode_return
        )

        # ----------------------------------------------
        # SAC weights must remain unchanged
        # ----------------------------------------------

        assert agent.update_count == 0

        for before, after in zip(
            actor_before,
            agent.actor.parameters()
        ):

            torch.testing.assert_close(
                before,
                after.detach(),
                atol=0,
                rtol=0
            )

        print(
            '[PASS] SAC policy generated '
            'real-observation proposals'
        )

        print(
            '[PASS] No SAC action sent to MPC; '
            'no learning occurred'
        )

        print(
            f'[SUMMARY] Steps={step_index}, '
            f'Return={episode_return:.4f}, '
            f'Shadow proposals={proposals}, '
            f'buffer={len(buffer)}'
        )

        print(
            '\n[PASS] SAC Gymnasium '
            'shadow-policy test completed'
        )

    finally:

        print('\nClosing environment')

        env.close()


if __name__ == '__main__':
    main()
