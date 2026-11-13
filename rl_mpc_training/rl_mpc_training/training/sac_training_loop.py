#!/usr/bin/env python3
"""Step 4.3e.3: reusable SAC-MPC training loop (Gymnasium-like API).

The SAC action is a normalized COMMAND u. The adapter sends T(obs,u) to
MPC; replay stores u only when Replay Policy accepts the transition.
No ROS imports. Environment lifecycle is owned by the caller.
"""

from dataclasses import dataclass

import numpy as np

from rl_mpc_training.utils.action_mapping import map_action_to_mpc
from rl_mpc_training.utils.sac_action_guard import DEFAULT_ACTION
from rl_mpc_training.utils.sac_training_action_adapter import (
    command_to_mpc_action,
    previous_action_from_observation,
)
from rl_mpc_training.utils.sac_transition import build_sac_transition


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 10
    max_episode_steps: int = 160
    warmup_steps: int = 5
    initial_random_steps: int = 256
    learning_starts: int = 256
    batch_size: int = 64
    updates_per_step: int = 1
    planning_failure_mode: str = 'task_failure'

    def __post_init__(self):
        for name in (
            'episodes', 'max_episode_steps', 'warmup_steps',
            'initial_random_steps', 'learning_starts',
            'batch_size', 'updates_per_step',
        ):
            value = getattr(self, name)

            minimum = (
                0 if name in (
                    'warmup_steps',
                    'initial_random_steps'
                )
                else 1
            )

            if type(value) is not int or value < minimum:
                raise ValueError(
                    f'{name} must be an integer >= {minimum}'
                )

        if self.learning_starts < self.batch_size:
            raise ValueError(
                'learning_starts must be >= batch_size'
            )

        if self.planning_failure_mode not in (
            'task_failure',
            'external_interruption'
        ):
            raise ValueError(
                'Invalid planning_failure_mode'
            )


class SACTrainingLoop:
    """Runs episodes and SAC updates; does not create/close environment."""

    def __init__(
        self,
        env,
        agent,
        buffer,
        config=None,
        seed=431
    ):
        if type(seed) is not int or seed < 0:
            raise ValueError(
                'seed must be a nonnegative integer'
            )

        self.env = env
        self.agent = agent
        self.buffer = buffer

        self.config = (
            config
            if config is not None
            else TrainingConfig()
        )

        if not isinstance(self.config, TrainingConfig):
            raise TypeError(
                'config must be TrainingConfig'
            )

        if buffer.capacity < self.config.learning_starts:
            raise ValueError(
                'Replay capacity is smaller than learning_starts'
            )

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.total_env_steps = 0
        self.total_stored = 0
        self.total_skipped = 0

        self.mode_counts = {
            'warmup': 0,
            'random': 0,
            'actor': 0
        }

        self.episodes_log = []
        self.updates_log = []

    # ==================================================
    # 1. Validate observation
    # ==================================================

    @staticmethod
    def _observation(value, space):

        obs = np.asarray(
            value,
            dtype=np.float32
        )

        if (
            obs.shape != (30,)
            or not np.isfinite(obs).all()
        ):
            raise ValueError(
                'Expected finite observation shape (30,)'
            )

        if not space.contains(obs):
            raise ValueError(
                'Observation outside Gymnasium space'
            )

        return obs.copy()

    # ==================================================
    # 2. Select SAC command
    # ==================================================

    def _select_command(self, obs, step_index):

        # Warmup at the beginning of every episode.
        if step_index <= self.config.warmup_steps:
            return DEFAULT_ACTION.copy(), 'warmup'

        # Initial random exploration.
        if self.total_env_steps < self.config.initial_random_steps:

            command = self.rng.uniform(
                -1.0,
                1.0,
                5
            ).astype(np.float32)

            return command, 'random'

        # Stochastic SAC policy.
        return (
            self.agent.act(
                obs,
                deterministic=False
            ),
            'actor'
        )

    # ==================================================
    # 3. Run training episodes
    # ==================================================

    def run(self, max_new_episodes=None):
        """
        Run training until the configured total episode count.

        max_new_episodes:
            None: run all remaining episodes.
            int : run at most this many new episodes.

        Safe to call again after a completed episode or
        after restoring a compatible checkpoint.
        """

        cfg = self.config

        # ==============================================
        # 1. Validate requested episode count
        # ==============================================

        if max_new_episodes is not None:

            if (
                type(max_new_episodes) is not int
                or max_new_episodes < 1
            ):
                raise ValueError(
                    'max_new_episodes must be '
                    'a positive integer or None'
                )

        # ==============================================
        # 2. Determine next global episode index
        # ==============================================

        start_episode = len(self.episodes_log)

        if start_episode > cfg.episodes:
            raise RuntimeError(
                'Completed episodes exceed training target'
            )

        # Confirm global episode indices are contiguous.
        for index, row in enumerate(self.episodes_log):

            if row['episode_index'] != index:
                raise RuntimeError(
                    'Episode history contains '
                    'duplicate or missing indices'
                )

        # ==============================================
        # 3. Validate checkpoint boundary
        # ==============================================

        completed_steps = sum(
            row['steps']
            for row in self.episodes_log
        )

        if completed_steps != self.total_env_steps:
            raise RuntimeError(
                'Training state is not at '
                'a completed episode boundary'
            )

        if (
            self.total_stored + self.total_skipped
            != self.total_env_steps
        ):
            raise RuntimeError(
                'Interaction counters are inconsistent'
            )

        if self.total_stored != self.buffer.total_added:
            raise RuntimeError(
                'Replay Buffer counters are inconsistent'
            )

        if len(self.updates_log) != self.agent.update_count:
            raise RuntimeError(
                'SAC update counters are inconsistent'
            )

        # ==============================================
        # 4. Determine final episode of this call
        # ==============================================

        if max_new_episodes is None:

            stop_episode = cfg.episodes

        else:

            stop_episode = min(
                cfg.episodes,
                start_episode + max_new_episodes
            )

        print(
            f'[TRAINING LOOP] '
            f'completed={start_episode} | '
            f'run_until={stop_episode} | '
            f'target={cfg.episodes}'
        )

        # ==============================================
        # 5. Continue from next global episode
        # ==============================================

        for ep in range(start_episode, stop_episode):

            # ------------------------------------------
            # Reset environment
            # ------------------------------------------

            obs_raw, reset_info = self.env.reset(
                seed=self.seed + ep
            )

            obs = self._observation(
                obs_raw,
                self.env.observation_space
            )

            if reset_info['phase'] != 'EPISODE_RUNNING':
                raise RuntimeError(
                    'Reset did not start an episode'
                )

            # MPC must start each episode from DEFAULT.
            np.testing.assert_allclose(
                previous_action_from_observation(obs),
                DEFAULT_ACTION,
                rtol=0.0,
                atol=3e-5
            )

            episode_return = 0.0
            episode_stored = 0
            episode_skipped = 0
            episode_updates = 0

            # ==========================================
            # Episode step loop
            # ==========================================

            for step_index in range(
                1,
                cfg.max_episode_steps + 1
            ):

                # --------------------------------------
                # Select command
                # --------------------------------------

                command, mode = self._select_command(
                    obs,
                    step_index
                )

                command = np.asarray(
                    command,
                    dtype=np.float32
                )

                if not self.env.action_space.contains(command):
                    raise ValueError(
                        'Invalid SAC command'
                    )

                # --------------------------------------
                # Convert command to physical MPC action
                # --------------------------------------

                mpc_action = command_to_mpc_action(
                    obs,
                    command
                )

                expected_params = np.asarray(
                    map_action_to_mpc(mpc_action)
                )

                # --------------------------------------
                # Environment step
                # --------------------------------------

                (
                    next_raw,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = self.env.step(mpc_action)

                next_obs = self._observation(
                    next_raw,
                    self.env.observation_space
                )

                if not np.isfinite(reward):
                    raise ValueError(
                        'Non-finite step reward'
                    )

                if info.get(
                    'diagnostic_reward_only',
                    False
                ):
                    raise RuntimeError(
                        'Diagnostic-only reward'
                    )

                if type(info['action_applied']) is not bool:
                    raise ValueError(
                        'Invalid action_applied flag'
                    )

                # --------------------------------------
                # Check MPC feedback on running steps
                # --------------------------------------

                if not (terminated or truncated):

                    if (
                        info['phase'] != 'EPISODE_RUNNING'
                        or not info['action_applied']
                    ):
                        raise RuntimeError(
                            'Invalid running transition'
                        )

                    np.testing.assert_allclose(
                        info['raw_observation'][25:30],
                        expected_params,
                        atol=1e-5,
                        rtol=0.0
                    )

                    # Next observation must encode
                    # the MPC action just applied.
                    np.testing.assert_allclose(
                        previous_action_from_observation(
                            next_obs
                        ),
                        mpc_action,
                        atol=3e-5,
                        rtol=0.0
                    )

                # --------------------------------------
                # Build SAC transition
                # --------------------------------------

                transition, policy = build_sac_transition(
                    observation=obs,

                    # Store COMMAND, not MPC action.
                    action=command,

                    reward=reward,
                    next_observation=next_obs,

                    terminated=bool(terminated),
                    truncated=bool(truncated),

                    info=info,

                    episode_index=ep,
                    step_index=step_index,

                    planning_failure_mode=(
                        cfg.planning_failure_mode
                    )
                )

                # --------------------------------------
                # Update interaction statistics
                # --------------------------------------

                self.total_env_steps += 1
                self.mode_counts[mode] += 1

                episode_return += float(reward)

                # --------------------------------------
                # Replay Buffer
                # --------------------------------------

                if transition is not None:

                    self.buffer.add(transition)

                    self.total_stored += 1
                    episode_stored += 1

                else:

                    assert not policy['store_transition']

                    self.total_skipped += 1
                    episode_skipped += 1

                # ======================================
                # SAC gradient updates
                # ======================================

                # A skipped transition must not directly
                # trigger an update.

                if (
                    transition is not None
                    and self.total_env_steps >= cfg.initial_random_steps
                    and len(self.buffer) >= cfg.learning_starts
                ):

                    for _ in range(
                        cfg.updates_per_step
                    ):

                        batch = self.buffer.sample(
                            cfg.batch_size
                        )

                        metrics = self.agent.update(
                            batch
                        )

                        # Verify training metrics.
                        if not all(
                            np.isfinite(value)
                            for value in metrics.values()
                            if value is not None
                        ):
                            raise FloatingPointError(
                                'Non-finite SAC metrics'
                            )

                        self.updates_log.append({
                            'env_step': self.total_env_steps,
                            **metrics
                        })

                        episode_updates += 1

                # --------------------------------------
                # Move to next observation
                # --------------------------------------

                obs = next_obs

                # ======================================
                # Terminal handling
                # ======================================

                if terminated or truncated:

                    summary = {
                        'episode_index': ep,
                        'steps': step_index,
                        'return': episode_return,
                        'phase': info['phase'],
                        'stored': episode_stored,
                        'skipped': episode_skipped,
                        'updates': episode_updates,
                        'buffer_size': len(self.buffer),
                        'alpha': self.agent.alpha
                    }

                    self.episodes_log.append(
                        summary
                    )

                    break

            else:
                raise RuntimeError(
                    f'Episode {ep} reached '
                    'max_episode_steps without terminal'
                )

        return self.episodes_log
