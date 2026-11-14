#!/usr/bin/env python3
"""Step 4.3e.5c - Persistent SAC-MPC training in Gazebo.

Features:
- Multiple episodes per session.
- Checkpoint and CSV after each completed episode.
- Automatic resume at the next global episode.
- Verify replay commands against acknowledged MPC actions.
- Stop after unsuccessful terminal events.
- No physical-robot support.

Only load checkpoints created by this project.
"""

import shutil
import time
from pathlib import Path

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent

from rl_mpc_training.training.sac_training_loop import (
    SACTrainingLoop,
    TrainingConfig,
)

from rl_mpc_training.training.sac_training_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    export_training_csv,
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

from rl_mpc_training.utils.goal_sampler import (
    select_episode_goal,
)

# ======================================================
# Configuration
# ======================================================

SEED = 431

GOAL_CANDIDATES = (
    (5.0, 0.0, 0.0),
    (7.5, 0.0, 0.0),
    (10.0, 0.0, 0.0),
)

HOME_POSITION = (0.0, 0.0)

TOTAL_EPISODES = 30

EPISODES_PER_SESSION = 3

BUFFER_CAPACITY = 4096

MAX_EPISODE_WALL_SEC = 240.0

MIN_FREE_DISK_MB = 80

RUN_DIR = (
    Path.home()
    / 'agv_ws'
    / 'sac_runs'
    / 'random_goal_44'
)


def make_config():

    return TrainingConfig(
        episodes=TOTAL_EPISODES,
        max_episode_steps=160,
        warmup_steps=5,
        initial_random_steps=128,
        learning_starts=128,
        batch_size=64,
        updates_per_step=1,
        planning_failure_mode='external_interruption',
    )


# ======================================================
# 1. Timed SAC Agent
# ======================================================

class TimedSACAgent(SACAgent):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.update_times_ms = []

    def update(self, batch):

        start = time.perf_counter()

        metrics = super().update(batch)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        self.update_times_ms.append(elapsed_ms)

        if (
            self.update_count == 1
            or self.update_count % 20 == 0
        ):

            print(
                f'[LEARN {self.update_count:05d}] '
                f'critic={metrics["critic_loss"]:.3f} '
                f'actor={metrics["actor_loss"]:.3f} '
                f'alpha={metrics["alpha"]:.4f} '
                f'update_ms={elapsed_ms:.1f}',
                flush=True,
            )

        return metrics


# ======================================================
# 2. Gazebo Environment Monitor
# ======================================================

class TrainingMonitor:

    def __init__(self,base,seed=SEED,goal_candidates=None,):

        self.base = base
        self.seed = seed

        self.goal_candidates = (
            tuple(
                tuple(goal)
                for goal in goal_candidates
            )
            if goal_candidates is not None
            else None
        )

        self.selected_goal = None

        self.observation_space = (
            base.observation_space
        )

        self.action_space = (
            base.action_space
        )

        self.episode_index = None
        self.episode_steps = 0
        self.episode_start = None

        self.confirmed_actions = {}

        self.step_times_ms = []
        self.actual_dt_s = []

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self, seed=None):

        if seed is None:
            raise ValueError(
                'Global episode seed is required'
            )

        # ============================================
        # 1. Global episode index
        # ============================================

        self.episode_index = (
            int(seed) - self.seed
        )

        if self.episode_index < 0:
            raise ValueError(
                'Invalid global episode index'
            )

        self.selected_goal = None

        # ============================================
        # 2. Select episode goal
        # ============================================

        if self.goal_candidates is not None:

            self.selected_goal = select_episode_goal(
                goal_candidates=self.goal_candidates,
                episode_index=self.episode_index,
                seed=self.seed,
                home=HOME_POSITION,
            )

            goal_x, goal_y, goal_yaw = (
                self.selected_goal
            )

            print(
                f'\n[EP {self.episode_index:03d}] '
                f'Random Goal: '
                f'x={goal_x:.3f}, '
                f'y={goal_y:.3f}, '
                f'yaw={goal_yaw:.3f}',
                flush=True,
            )

            # Pass selected goal to RLMPCEnv.
            obs, info = self.base.reset(
                seed=seed,
                options={
                    'goal': self.selected_goal
                }
            )

            info = dict(info)

            info['selected_goal'] = (
                self.selected_goal
            )

        else:

            # Backward-compatible fixed-goal reset.
            obs, info = self.base.reset(
                seed=seed
            )

        self.episode_steps = 0
        self.episode_start = time.monotonic()

        if info['phase'] != 'EPISODE_RUNNING':
            raise RuntimeError(
                'Episode reset did not reach RUNNING'
            )

        print(
            f'\n[EP {self.episode_index:03d}] '
            'RESET',
            flush=True,
        )

        return obs, info

    # --------------------------------------------------
    # Step
    # --------------------------------------------------

    def step(self, mpc_action):

        if self.episode_start is None:
            raise RuntimeError(
                'Call reset() before step()'
            )

        if (
            time.monotonic() - self.episode_start
            > MAX_EPISODE_WALL_SEC
        ):
            raise TimeoutError(
                'Episode wall-time limit exceeded'
            )

        expected_params = np.asarray(
            map_action_to_mpc(mpc_action),
            dtype=np.float64,
        )

        start = time.perf_counter()

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.base.step(mpc_action)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        self.step_times_ms.append(
            elapsed_ms
        )

        self.episode_steps += 1

        # --------------------------------------------------
        # Record only ACK-confirmed MPC actions
        # --------------------------------------------------

        applied = bool(
            info['action_applied']
        )

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

        # --------------------------------------------------
        # Verify feedback during running transitions
        # --------------------------------------------------

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

        dt = info.get('actual_dt_s')

        if dt is not None:

            dt = float(dt)

            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(
                    'Invalid actual_dt_s'
                )

            self.actual_dt_s.append(dt)

        if (
            self.episode_steps % 10 == 0
            or terminated
            or truncated
        ):

            print(
                f'[EP {self.episode_index:03d} '
                f'STEP {self.episode_steps:03d}] '
                f'phase={info["phase"]} '
                f'ack={applied} '
                f'mpc={np.round(expected_params, 3).tolist()} '
                f'r={float(reward):.3f}',
                flush=True,
            )

        return (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):

        self.base.close()


# ======================================================
# 3. Verify stored commands for one completed episode
# ======================================================

def verify_episode_replay(
    buffer,
    monitor,
    episode_row
):

    ep = int(
        episode_row['episode_index']
    )

    count = 0

    # Read active physical slots without sampling.
    # This preserves Replay Buffer RNG state.

    for slot in range(len(buffer)):

        if int(buffer.episode_indices[slot]) != ep:
            continue

        step = int(
            buffer.step_indices[slot]
        )

        key = (ep, step)

        if key not in monitor.confirmed_actions:
            raise AssertionError(
                f'Missing MPC ACK for {key}'
            )

        reconstructed = command_to_mpc_action(
            buffer.observations[slot],
            buffer.actions[slot],
        )

        np.testing.assert_array_equal(
            reconstructed,
            monitor.confirmed_actions[key],
        )

        count += 1

    if count != episode_row['stored']:
        raise AssertionError(
            'Episode Replay Buffer count mismatch'
        )

    print(
        f'[PASS] Episode {ep}: '
        f'{count} SAC commands reproduce ACK actions'
    )


# ======================================================
# 4. Build, resume and run a training session
# ======================================================

def run_training_session(
    *,
    run_dir,
    config,
    env_factory,
    episodes_this_session,
    capacity=BUFFER_CAPACITY,
    seed=SEED,
    goal_candidates=None,
):

    if (
        type(episodes_this_session) is not int
        or episodes_this_session < 1
    ):
        raise ValueError(
            'episodes_this_session must be positive'
        )

    run_dir = Path(
        run_dir
    ).expanduser()

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    free_mb = (
        shutil.disk_usage(run_dir).free
        / (1024 * 1024)
    )

    if free_mb < MIN_FREE_DISK_MB:
        raise RuntimeError(
            f'Insufficient free disk space: '
            f'{free_mb:.1f} MB'
        )

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    agent = TimedSACAgent(
        obs_dim=30,
        action_dim=5,
        device='cpu',
        gamma=0.99,
        tau=0.005,
        init_alpha=0.2,
        automatic_entropy_tuning=True,
    )

    buffer = SACReplayBuffer(
        capacity=capacity,
        seed=seed,
    )

    # Environment will be attached after restoring state.
    trainer = SACTrainingLoop(
        env=None,
        agent=agent,
        buffer=buffer,
        config=config,
        seed=seed,
    )

    checkpoint = (
        run_dir / 'latest.pt'
    )

    # ==================================================
    # Restore existing session
    # ==================================================

    if checkpoint.exists():

        print(
            f'[RESUME] {checkpoint}',
            flush=True,
        )

        metadata = load_checkpoint(
            checkpoint,
            agent,
            buffer,
            trainer,
        )

        print(
            '[RESUME] '
            f'episodes={metadata["episodes"]} '
            f'env_steps={metadata["env_steps"]} '
            f'updates={metadata["updates"]} '
            f'replay={metadata["replay_size"]}',
            flush=True,
        )

        # Regenerate logs from the authoritative checkpoint.
        export_training_csv(
            run_dir / 'logs',
            trainer,
        )

    else:

        print(
            '[NEW RUN] Starting from initialized SAC',
            flush=True,
        )

    completed = len(
        trainer.episodes_log
    )

    # ==================================================
    # Do not continue after a failed terminal
    # ==================================================

    if completed:

        last_phase = (
            trainer.episodes_log[-1]['phase']
        )

        if last_phase != 'EPISODE_SUCCESS':

            raise RuntimeError(
                'Previous completed episode ended with '
                f'{last_phase}. '
                'Inspect the failure before resuming.'
            )

    remaining = (
        config.episodes - completed
    )

    if remaining < 0:
        raise RuntimeError(
            'Checkpoint exceeds configured episode target'
        )

    if remaining == 0:

        print(
            '[COMPLETE] Training target already reached'
        )

        return agent, buffer, trainer

    count = min(
        remaining,
        episodes_this_session
    )

    print(
        f'[SESSION] completed={completed} '
        f'new_episodes={count} '
        f'target={config.episodes}',
        flush=True,
    )

    # ==================================================
    # Initialize environment only when work remains
    # ==================================================

    monitor = None

    try:

        monitor = TrainingMonitor(
            env_factory(),
            seed=seed,
            goal_candidates=goal_candidates,
        )

        trainer.env = monitor

        # ==============================================
        # Execute exactly one episode per trainer call
        # ==============================================

        for _ in range(count):

            updates_before = (
                agent.update_count
            )

            trainer.run(
                max_new_episodes=1
            )

            row = trainer.episodes_log[-1]

            # Verify action semantics before checkpointing.
            verify_episode_replay(
                buffer,
                monitor,
                row,
            )

            # ==========================================
            # Save completed episode
            # ==========================================

            save_checkpoint(
                checkpoint,
                agent,
                buffer,
                trainer,
            )

            outputs = export_training_csv(
                run_dir / 'logs',
                trainer,
            )

            new_updates = (
                agent.update_count - updates_before
            )

            print(
                f'[SAVED EP {row["episode_index"]:03d}] '
                f'phase={row["phase"]} '
                f'steps={row["steps"]} '
                f'return={row["return"]:.3f} '
                f'stored={row["stored"]} '
                f'updates={new_updates} '
                f'buffer={len(buffer)} '
                f'alpha={agent.alpha:.4f}',
                flush=True,
            )

            print(
                f'[CHECKPOINT] {checkpoint}',
                flush=True,
            )

            print(
                f'[CSV] {outputs["episodes.csv"]}',
                flush=True,
            )

            # ==========================================
            # Stop after failed terminal
            # ==========================================

            if row['phase'] != 'EPISODE_SUCCESS':

                print(
                    '[STOP] Unsuccessful terminal. '
                    'Checkpoint saved; no further '
                    'episodes will be started.',
                    flush=True,
                )

                break

        # ==============================================
        # End-of-session statistics
        # ==============================================

        print(
            '\n[SESSION SUMMARY] '
            f'completed={len(trainer.episodes_log)}/'
            f'{config.episodes} '
            f'env_steps={trainer.total_env_steps} '
            f'stored={trainer.total_stored} '
            f'skipped={trainer.total_skipped} '
            f'updates={agent.update_count} '
            f'alpha={agent.alpha:.4f}',
            flush=True,
        )

        if agent.update_times_ms:

            times = np.asarray(
                agent.update_times_ms
            )

            print(
                '[TIMING] SAC update: '
                f'mean={times.mean():.1f} ms, '
                f'p95={np.percentile(times, 95):.1f} ms, '
                f'max={times.max():.1f} ms',
                flush=True,
            )

        if monitor.actual_dt_s:

            dt = np.asarray(
                monitor.actual_dt_s
            )

            print(
                '[TIMING] RL actual_dt: '
                f'mean={dt.mean():.3f} s, '
                f'max={dt.max():.3f} s',
                flush=True,
            )

        return agent, buffer, trainer

    finally:

        if monitor is not None:

            print(
                '[CLOSE] Requesting environment shutdown',
                flush=True,
            )

            monitor.close()


# ======================================================
# 5. Main - Gazebo only
# ======================================================

def main():

    print(
        '\nStep 4.3e.5c - '
        'Persistent SAC-MPC Gazebo Training'
    )

    print(
        f'Target episodes: {TOTAL_EPISODES}\n'
        f'Episodes per session: {EPISODES_PER_SESSION}\n'
        f'Output: {RUN_DIR}'
    )

    answer = input(
        'Type TRAIN_CLEAR to continue: '
    ).strip().upper()

    if answer != 'TRAIN_CLEAR':

        print('Cancelled')
        return

    # Lazy import: CPU tests do not require ROS2.
    from rl_mpc_training.env.rl_mpc_env import RLMPCEnv

    run_training_session(
        run_dir=RUN_DIR,
        config=make_config(),
        env_factory=RLMPCEnv,
        episodes_this_session=EPISODES_PER_SESSION,
        capacity=BUFFER_CAPACITY,
        seed=SEED,
        goal_candidates=GOAL_CANDIDATES,
    )


if __name__ == '__main__':
    main()
