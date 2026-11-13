#!/usr/bin/env python3
"""Step 4.3e.3: CPU-only SAC training-loop integration test.

Uses a lightweight fake Gymnasium environment with realistic
ACK, observation, reward and terminal metadata.

No ROS2 or Gazebo. Does not evaluate navigation performance.
"""

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent

from rl_mpc_training.training.sac_training_loop import (
    SACTrainingLoop,
    TrainingConfig,
)

from rl_mpc_training.utils.action_mapping import (
    MPC_PARAM_MIN,
    MPC_PARAM_MAX,
    map_action_to_mpc,
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION,
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer,
)

from rl_mpc_training.utils.reward_function import (
    TERMINAL_REWARDS,
)

from rl_mpc_training.utils.sac_transition import (
    REWARD_KEYS,
)

from rl_mpc_training.utils.sac_training_action_adapter import (
    command_to_mpc_action,
)


# ======================================================
# 1. Minimal normalized observation/action space
# ======================================================

class NormalizedSpace:

    def __init__(self, dim):
        self.dim = dim

    def contains(self, x):

        value = np.asarray(x)

        return (
            value.shape == (self.dim,)
            and value.dtype == np.float32
            and np.isfinite(value).all()
            and np.all(value >= -1.0)
            and np.all(value <= 1.0)
        )


# ======================================================
# 2. Fake RL-MPC environment
# ======================================================

class FakeRLMPCEnv:

    PHASES = (
        'EPISODE_SUCCESS',
        'EPISODE_COLLISION',
        'EPISODE_TIMEOUT',
        'EPISODE_PLANNING_FAILURE',
    )

    def __init__(self, episode_length=12):

        self.observation_space = NormalizedSpace(30)
        self.action_space = NormalizedSpace(5)

        self.episode_length = episode_length

        self.episode_index = -1
        self.done = True

        self.applied_history = {}

    # --------------------------------------------------
    # Build normalized observation
    # --------------------------------------------------

    def _observe(self, action):

        params = np.asarray(
            map_action_to_mpc(action),
            dtype=np.float64
        )

        obs = np.zeros(
            30,
            dtype=np.float32
        )

        # Synthetic geometry features.
        obs[0] = np.float32(
            np.clip(self.progress / 10.0, -1, 1)
        )

        obs[4] = np.float32(
            np.clip(
                (10.0 - self.progress) / 20.0,
                0.0,
                1.0
            )
        )

        # Synthetic LiDAR.
        obs[7:12] = 0.8
        obs[12:17] = 1.0

        # Synthetic velocity feature.
        obs[22] = np.float32(
            np.clip(params[4] / 0.5, 0, 1)
        )

        # Linear MPC parameter normalization.
        obs[25:27] = (
            (params[:2] - MPC_PARAM_MIN[:2])
            /
            (MPC_PARAM_MAX[:2] - MPC_PARAM_MIN[:2])
        )

        # Logarithmic MPC parameter normalization.
        obs[27:29] = (
            np.log(
                params[2:4] / MPC_PARAM_MIN[2:4]
            )
            /
            np.log(
                MPC_PARAM_MAX[2:4] / MPC_PARAM_MIN[2:4]
            )
        )

        # Reference velocity normalization.
        obs[29] = (
            params[4] / MPC_PARAM_MAX[4]
        )

        return obs

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self, seed=None):

        self.episode_index += 1

        self.done = False

        self.steps = 0
        self.progress = 0.0

        self.last_mpc = DEFAULT_ACTION.copy()

        self.obs = self._observe(
            self.last_mpc
        )

        return (
            self.obs.copy(),
            {'phase': 'EPISODE_RUNNING'}
        )

    # --------------------------------------------------
    # Step
    # --------------------------------------------------

    def step(self, action):

        if self.done:
            raise RuntimeError(
                'Episode is not active'
            )

        assert self.action_space.contains(action)

        self.steps += 1

        terminal = (
            self.steps == self.episode_length
        )

        phase = (
            self.PHASES[self.episode_index % 4]
            if terminal
            else 'EPISODE_RUNNING'
        )

        # Final episode simulates a terminal event
        # occurring before MPC applies the new action.
        action_applied = not (
            terminal and self.episode_index == 7
        )

        reused = (
            terminal
            and phase == 'EPISODE_PLANNING_FAILURE'
        )

        reward_mode = (
            'event_only'
            if reused or not action_applied
            else 'full'
        )

        # ----------------------------------------------
        # Apply action
        # ----------------------------------------------

        if action_applied:

            self.applied_history[
                (self.episode_index, self.steps)
            ] = np.asarray(
                action,
                dtype=np.float32
            ).copy()

            self.last_mpc = np.asarray(
                action,
                dtype=np.float32
            ).copy()

            self.progress += 0.2

        # ----------------------------------------------
        # Next observation
        # ----------------------------------------------

        next_obs = (
            self.obs.copy()
            if reused
            else self._observe(self.last_mpc)
        )

        # ----------------------------------------------
        # Reward
        # ----------------------------------------------

        components = {
            key: 0.0
            for key in REWARD_KEYS
        }

        if reward_mode == 'full':

            components['progress'] = 0.9
            components['path'] = -0.02
            components['time'] = -0.01

        components['terminal'] = float(
            TERMINAL_REWARDS[phase]
        )

        reward = float(
            sum(components.values())
        )

        # ----------------------------------------------
        # Gymnasium info
        # ----------------------------------------------

        info = {
            'phase': phase,

            'action_applied': bool(action_applied),

            'terminal_observation_reused': bool(
                reused
            ),

            'reward_mode': reward_mode,

            'reward_components': components,

            'diagnostic_reward_only': False,

            'raw_observation': np.concatenate(
                (
                    np.zeros(25),
                    map_action_to_mpc(self.last_mpc)
                )
            )
        }

        if terminal:

            info['replay_eligible'] = bool(
                action_applied
                and (
                    phase in (
                        'EPISODE_SUCCESS',
                        'EPISODE_COLLISION'
                    )
                    or not reused
                )
            )

        else:

            info['mpc_params'] = (
                map_action_to_mpc(action)
            )

        # ----------------------------------------------
        # Episode state
        # ----------------------------------------------

        self.done = terminal

        self.obs = next_obs.copy()

        terminated = bool(
            terminal
            and phase in (
                'EPISODE_SUCCESS',
                'EPISODE_COLLISION'
            )
        )

        truncated = bool(
            terminal
            and phase in (
                'EPISODE_TIMEOUT',
                'EPISODE_PLANNING_FAILURE'
            )
        )

        return (
            next_obs,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):

        self.done = True


# ======================================================
# 3. Main integration test
# ======================================================

def main():

    print(
        '\nStep 4.3e.3 - '
        'SAC Training Loop (CPU only)'
    )

    torch.manual_seed(431)
    torch.set_num_threads(1)

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    config = TrainingConfig(
        episodes=8,

        max_episode_steps=12,
        warmup_steps=5,

        initial_random_steps=48,
        learning_starts=32,

        batch_size=16,
        updates_per_step=1,

        planning_failure_mode='task_failure'
    )

    # --------------------------------------------------
    # Initialize components
    # --------------------------------------------------

    agent = SACAgent(
        device='cpu'
    )

    buffer = SACReplayBuffer(
        capacity=128,
        seed=431
    )

    env = FakeRLMPCEnv(
        episode_length=12
    )

    # Store initial neural-network parameters.
    actor_before = [
        p.detach().clone()
        for p in agent.actor.parameters()
    ]

    critic_before = [
        p.detach().clone()
        for p in agent.critic.parameters()
    ]

    trainer = SACTrainingLoop(
        env=env,
        agent=agent,
        buffer=buffer,
        config=config,
        seed=431
    )

    try:

        # ==============================================
        # 1. Execute training loop
        # ==============================================

        print(
            '\n[1] Multiple episodes, '
            'command mapping, replay policy'
        )

        rows = trainer.run()

        assert len(rows) == 8

        assert trainer.total_env_steps == 96

        assert trainer.total_stored == 95

        assert trainer.total_skipped == 1

        assert len(buffer) == 95

        assert [
            episode['phase']
            for episode in rows
        ] == list(env.PHASES) * 2

        assert all(
            episode['steps'] == 12
            for episode in rows
        )

        # Last episode has one skipped terminal.
        assert rows[-1]['skipped'] == 1

        assert rows[-1]['stored'] == 11

        print(
            '[PASS] Eight 12-step episodes; '
            'one non-applied terminal skipped'
        )

        print(
            '[PASS] Reset/episode accounting '
            'and valid replay insertion'
        )

        # ==============================================
        # 2. Verify Replay Buffer
        # ==============================================

        print(
            '\n[2] Verify stored command '
            'and bootstrap semantics'
        )

        data = buffer.sample(
            len(buffer)
        )

        phases = np.asarray(
            data['phases'],
            dtype=object
        )

        masks = (
            data['bootstrap_masks'][:, 0]
        )

        # ----------------------------------------------
        # Validate bootstrap masks
        # ----------------------------------------------

        for phase, expected_mask in (
            ('EPISODE_RUNNING', 1.0),
            ('EPISODE_TIMEOUT', 1.0),
            ('EPISODE_SUCCESS', 0.0),
            ('EPISODE_COLLISION', 0.0),
            ('EPISODE_PLANNING_FAILURE', 0.0),
        ):

            assert np.all(
                masks[phases == phase]
                == expected_mask
            )

        # One planning failure stored;
        # the other planning failure was skipped.
        assert sum(
            phases == 'EPISODE_PLANNING_FAILURE'
        ) == 1

        assert data['policy_overrides'][
            phases == 'EPISODE_PLANNING_FAILURE'
        ].all()

        # ----------------------------------------------
        # Replay commands must reproduce actual MPC actions
        # ----------------------------------------------

        for i, (obs, command) in enumerate(
            zip(
                data['observations'],
                data['actions']
            )
        ):

            applied = command_to_mpc_action(
                obs,
                command
            )

            key = (
                int(data['episode_indices'][i]),
                int(data['step_indices'][i])
            )

            np.testing.assert_array_equal(
                applied,
                env.applied_history[key]
            )

        print(
            '[PASS] Replay contains command-space '
            'actions and correct terminal masks'
        )

        # ==============================================
        # 3. Verify SAC training
        # ==============================================

        print(
            '\n[3] Warmup, exploration, '
            'stochastic actor and gradient updates'
        )

        # Every action-selection mode must be exercised.
        assert all(
            trainer.mode_counts[name] > 0
            for name in (
                'warmup',
                'random',
                'actor'
            )
        )

        # Update accounting.
        assert (
            len(trainer.updates_log)
            == agent.update_count
        )

        assert agent.update_count > 0

        # No updates before the configured threshold.
        assert all(
            entry['env_step']
            >= config.initial_random_steps
            for entry in trainer.updates_log
        )

        # ----------------------------------------------
        # Actor parameters changed
        # ----------------------------------------------

        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                actor_before,
                agent.actor.parameters()
            )
        )

        # ----------------------------------------------
        # Critic parameters changed
        # ----------------------------------------------

        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                critic_before,
                agent.critic.parameters()
            )
        )

        # ----------------------------------------------
        # Training metrics finite
        # ----------------------------------------------

        for record in trainer.updates_log:

            assert np.isfinite(
                record['critic_loss']
            )

            assert np.isfinite(
                record['actor_loss']
            )

            assert record['alpha'] > 0

        print(
            '[PASS] All action modes exercised; '
            'updates gated by replay readiness'
        )

        print(
            '[PASS] Actor/Critic changed; '
            'learning metrics finite'
        )

        # ==============================================
        # 4. Episode summaries
        # ==============================================

        print(
            '\n[4] Episode summaries'
        )

        for row in rows:

            print(
                f'[EP {row["episode_index"]:02d}] '
                f'phase={row["phase"]} '
                f'steps={row["steps"]} '
                f'stored={row["stored"]} '
                f'skipped={row["skipped"]} '
                f'updates={row["updates"]} '
                f'return={row["return"]:.3f}'
            )

        print(
            f'[SUMMARY] '
            f'steps={trainer.total_env_steps} '
            f'stored={trainer.total_stored} '
            f'skipped={trainer.total_skipped} '
            f'updates={agent.update_count} '
            f'alpha={agent.alpha:.4f}'
        )

        print(
            '\n[PASS] SAC Training Loop '
            'tests completed'
        )

    finally:

        env.close()


if __name__ == '__main__':
    main()
