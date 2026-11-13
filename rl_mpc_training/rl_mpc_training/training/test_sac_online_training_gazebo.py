#!/usr/bin/env python3
"""Step 4.3e.4: three-episode SAC-MPC online training smoke test.

Gazebo only. Uses the validated Training Action Adapter through
SACTrainingLoop. No checkpoints or permanent training artifacts.
"""

import time

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent
from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

from rl_mpc_training.training.sac_training_loop import (
    SACTrainingLoop,
    TrainingConfig,
)

from rl_mpc_training.utils.action_mapping import (
    map_action_to_mpc,
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer,
)

from rl_mpc_training.utils.sac_training_action_adapter import (
    command_to_mpc_action,
)


# ======================================================
# Configuration
# ======================================================

SEED = 431

EPISODES = 3

MAX_EPISODE_STEPS = 160

MAX_EPISODE_WALL_SEC = 240.0


# ======================================================
# 1. Timed SAC Agent
# ======================================================

class TimedSACAgent(SACAgent):
    """Measure update cost without changing SAC optimizer behavior."""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.update_times_ms = []

    def update(self, batch):

        start = time.perf_counter()

        metrics = super().update(batch)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        self.update_times_ms.append(
            elapsed_ms
        )

        if (
            self.update_count == 1
            or self.update_count % 10 == 0
        ):

            print(
                f'[LEARN {self.update_count:04d}] '
                f'critic={metrics["critic_loss"]:.3f} '
                f'actor={metrics["actor_loss"]:.3f} '
                f'alpha={metrics["alpha"]:.4f} '
                f'update_ms={elapsed_ms:.1f}',
                flush=True,
            )

        return metrics


# ======================================================
# 2. Monitored Gymnasium Environment
# ======================================================

class MonitoredRLMPCEnv:
    """Monitor MPC actions and stop after failed terminal events."""

    def __init__(self, base):

        self.base = base

        self.observation_space = (
            base.observation_space
        )

        self.action_space = (
            base.action_space
        )

        self.episode_index = -1

        self.episode_steps = 0

        self.episode_start = None

        self.terminal_phase = None

        self.confirmed_actions = {}

        self.step_wall_ms = []

        self.actual_dt_s = []

    # ==================================================
    # Reset
    # ==================================================

    def reset(self, seed=None):

        # Do not restart after collision, timeout,
        # or planning failure in this smoke test.
        if self.episode_index >= 0:

            if self.terminal_phase != 'EPISODE_SUCCESS':

                raise RuntimeError(
                    'Refusing another episode after '
                    f'{self.terminal_phase}'
                )

        observation, info = self.base.reset(
            seed=seed
        )

        self.episode_index += 1

        self.episode_steps = 0

        self.terminal_phase = None

        self.episode_start = time.monotonic()

        print(
            f'\n[EP {self.episode_index:02d}] '
            f'RESET phase={info["phase"]}',
            flush=True,
        )

        return observation, info

    # ==================================================
    # Step
    # ==================================================

    def step(self, mpc_action):

        if self.episode_start is None:

            raise RuntimeError(
                'Call reset before step'
            )

        if (
            time.monotonic() - self.episode_start
            > MAX_EPISODE_WALL_SEC
        ):

            raise TimeoutError(
                'Episode exceeded wall-clock safety limit'
            )

        # Expected MPC parameters.
        expected_params = np.asarray(
            map_action_to_mpc(mpc_action),
            dtype=np.float64
        )

        start = time.perf_counter()

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = self.base.step(mpc_action)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        self.step_wall_ms.append(
            elapsed_ms
        )

        self.episode_steps += 1

        applied = info['action_applied']

        # ----------------------------------------------
        # Record confirmed MPC actions
        # ----------------------------------------------

        if applied:

            key = (
                self.episode_index,
                self.episode_steps
            )

            self.confirmed_actions[key] = (
                np.asarray(
                    mpc_action,
                    dtype=np.float32
                ).copy()
            )

        # ----------------------------------------------
        # Verify running MPC feedback
        # ----------------------------------------------

        if not (terminated or truncated):

            if not applied:

                raise RuntimeError(
                    'Running MPC action not acknowledged'
                )

            np.testing.assert_allclose(
                info['raw_observation'][25:30],
                expected_params,
                rtol=0.0,
                atol=1e-5,
            )

        # ----------------------------------------------
        # Simulation time diagnostics
        # ----------------------------------------------

        dt_s = info.get(
            'actual_dt_s'
        )

        if dt_s is not None:

            self.actual_dt_s.append(
                float(dt_s)
            )

        # ----------------------------------------------
        # Progress logging
        # ----------------------------------------------

        if (
            self.episode_steps % 5 == 0
            or terminated
            or truncated
        ):

            print(
                f'[EP {self.episode_index:02d} '
                f'STEP {self.episode_steps:03d}] '
                f'phase={info["phase"]} '
                f'ack={applied} '
                f'params='
                f'{np.round(expected_params, 3).tolist()} '
                f'r={float(reward):.3f} '
                f'step_wall_ms={elapsed_ms:.0f}',
                flush=True,
            )

        # ----------------------------------------------
        # Terminal handling
        # ----------------------------------------------

        if terminated or truncated:

            self.terminal_phase = info['phase']

            print(
                f'[EP {self.episode_index:02d}] '
                f'TERMINAL {self.terminal_phase}',
                flush=True,
            )

        return (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ==================================================
    # Close
    # ==================================================

    def close(self):

        self.base.close()


# ======================================================
# 3. Verify training
# ======================================================

def verify_training(
    trainer,
    agent,
    buffer,
    monitor,
    actor_before
):

    print(
        '\n[VERIFY] Episode, Replay Buffer '
        'and gradient consistency'
    )

    # --------------------------------------------------
    # Episode verification
    # --------------------------------------------------

    assert len(
        trainer.episodes_log
    ) == EPISODES

    assert all(
        row['phase'] == 'EPISODE_SUCCESS'
        for row in trainer.episodes_log
    )

    assert agent.update_count > 0

    assert (
        len(trainer.updates_log)
        == agent.update_count
    )

    assert (
        len(agent.update_times_ms)
        == agent.update_count
    )

    assert (
        trainer.total_stored
        == len(buffer)
    )

    assert (
        trainer.total_env_steps
        == sum(
            row['steps']
            for row in trainer.episodes_log
        )
    )

    # All action-selection modes must execute.
    assert all(
        trainer.mode_counts[name] > 0
        for name in (
            'warmup',
            'random',
            'actor'
        )
    )

    # Actor weights must have changed.
    assert any(
        not torch.equal(
            old,
            new.detach()
        )
        for old, new in zip(
            actor_before,
            agent.actor.parameters()
        )
    )

    print(
        '[PASS] Three successful episodes, '
        'all selection modes, SAC updates'
    )

    # ==================================================
    # 4. Verify Replay Buffer
    # ==================================================

    full = buffer.sample(
        len(buffer)
    )

    for i in range(len(buffer)):

        key = (
            int(
                full['episode_indices'][i]
            ),
            int(
                full['step_indices'][i]
            )
        )

        assert key in (
            monitor.confirmed_actions
        ), key

        # Reconstruct applied MPC action
        # from stored SAC command.
        reconstructed = command_to_mpc_action(
            full['observations'][i],
            full['actions'][i]
        )

        np.testing.assert_array_equal(
            reconstructed,
            monitor.confirmed_actions[key]
        )

    print(
        '[PASS] Stored SAC commands reproduce '
        'all applied MPC actions'
    )

    # ==================================================
    # 5. Verify SAC update metrics
    # ==================================================

    for record in trainer.updates_log:

        assert record['env_step'] >= 64

        for name in (
            'critic_loss',
            'actor_loss',
            'alpha',
            'target_q_mean',
            'bootstrap_fraction'
        ):

            assert np.isfinite(
                record[name]
            ), name

        assert record['alpha'] > 0.0

    print(
        '[PASS] Training metrics finite and '
        'learning threshold respected'
    )

    # ==================================================
    # 6. Episode summaries
    # ==================================================

    for row in trainer.episodes_log:

        print(
            f'[SUMMARY EP {row["episode_index"]:02d}] '
            f'phase={row["phase"]} '
            f'steps={row["steps"]} '
            f'return={row["return"]:.3f} '
            f'stored={row["stored"]} '
            f'skipped={row["skipped"]} '
            f'updates={row["updates"]}'
        )

    # ==================================================
    # 7. Timing diagnostics
    # ==================================================

    times = np.asarray(
        agent.update_times_ms
    )

    print(
        f'[TRAIN] '
        f'env_steps={trainer.total_env_steps}, '
        f'stored={trainer.total_stored}, '
        f'skipped={trainer.total_skipped}, '
        f'updates={agent.update_count}, '
        f'alpha={agent.alpha:.4f}'
    )

    print(
        f'[TIMING] update_ms: '
        f'mean={times.mean():.1f}, '
        f'p95={np.percentile(times, 95):.1f}, '
        f'max={times.max():.1f}'
    )

    if monitor.actual_dt_s:

        dt = np.asarray(
            monitor.actual_dt_s
        )

        print(
            f'[TIMING] actual_dt_s: '
            f'mean={dt.mean():.3f}, '
            f'max={dt.max():.3f}'
        )

    if (
        np.percentile(times, 95)
        > 500.0
    ):

        print(
            '[WARNING] SAC updates may disrupt '
            'the nominal 0.5 s step cadence'
        )

    print(
        '\n[PASS] SAC Gazebo online-training '
        'smoke test completed'
    )


# ======================================================
# 8. Main
# ======================================================

def main():

    print(
        '\nStep 4.3e.4 - '
        'SAC-MPC Online Training Smoke Test'
    )

    scenario = input(
        'Type TRAIN_CLEAR to start: '
    ).strip().upper()

    if scenario != 'TRAIN_CLEAR':

        print('Cancelled')
        return

    # --------------------------------------------------
    # Random seeds
    # --------------------------------------------------

    torch.manual_seed(SEED)

    torch.set_num_threads(1)

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    config = TrainingConfig(
        episodes=EPISODES,

        max_episode_steps=MAX_EPISODE_STEPS,

        warmup_steps=5,

        initial_random_steps=64,

        learning_starts=64,

        batch_size=32,

        updates_per_step=1,

        planning_failure_mode='external_interruption'
    )

    # --------------------------------------------------
    # SAC Agent
    # --------------------------------------------------

    agent = TimedSACAgent(
        obs_dim=30,
        action_dim=5,

        device='cpu',

        gamma=0.99,
        tau=0.005,

        init_alpha=0.2,

        automatic_entropy_tuning=True
    )

    # --------------------------------------------------
    # Replay Buffer
    # --------------------------------------------------

    buffer = SACReplayBuffer(
        capacity=1024,
        seed=SEED
    )

    # Initial Actor parameters for comparison.
    actor_before = [
        p.detach().clone()
        for p in agent.actor.parameters()
    ]

    # --------------------------------------------------
    # Gymnasium Environment
    # --------------------------------------------------

    # RLMPCEnv.close() requests MPC HOLD on
    # both successful completion and error.
    monitor = MonitoredRLMPCEnv(
        RLMPCEnv()
    )

    try:

        # ----------------------------------------------
        # Training Loop
        # ----------------------------------------------

        trainer = SACTrainingLoop(
            env=monitor,
            agent=agent,
            buffer=buffer,
            config=config,
            seed=SEED
        )

        # ----------------------------------------------
        # Execute three episodes
        # ----------------------------------------------

        trainer.run()

        # ----------------------------------------------
        # Verify results
        # ----------------------------------------------

        verify_training(
            trainer,
            agent,
            buffer,
            monitor,
            actor_before
        )

    finally:

        print(
            '\nClosing environment; '
            'requesting MPC HOLD',
            flush=True
        )

        monitor.close()


if __name__ == '__main__':
    main()
