#!/usr/bin/env python3

"""Fixed-capacity NumPy replay buffer for SAC."""

import numpy as np

from rl_mpc_training.utils.sac_transition import (
    ACTION_DIM,
    OBS_DIM,
    REWARD_KEYS,
    SACTransition
)

from rl_mpc_training.utils.step_utils import (
    TERMINAL_FLAGS
)

from rl_mpc_training.utils.reward_function import (
    TERMINAL_REWARDS
)


class SACReplayBuffer:
    """
    Circular replay buffer with uniform sampling.

    Designed for single-threaded training access.
    """

    def __init__(self, capacity, seed=None):

        if type(capacity) is not int or capacity < 1:
            raise ValueError(
                'capacity must be a positive integer'
            )

        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise ValueError(
                    'seed must be nonnegative or None'
                )

        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        # Main SAC arrays
        self.observations = np.empty(
            (capacity, OBS_DIM), dtype=np.float32
        )

        self.actions = np.empty(
            (capacity, ACTION_DIM), dtype=np.float32
        )

        self.rewards = np.empty(
            (capacity, 1), dtype=np.float32
        )

        self.next_observations = np.empty(
            (capacity, OBS_DIM), dtype=np.float32
        )

        self.bootstrap_masks = np.empty(
            (capacity, 1), dtype=np.float32
        )

        # Reward diagnostics
        self.reward_components = np.empty(
            (capacity, len(REWARD_KEYS)),
            dtype=np.float32
        )

        # Episode metadata
        self.episode_indices = np.empty(
            capacity, dtype=np.int64
        )

        self.step_indices = np.empty(
            capacity, dtype=np.int64
        )

        self.terminated = np.empty(
            capacity, dtype=np.bool_
        )

        self.truncated = np.empty(
            capacity, dtype=np.bool_
        )

        self.learning_terminated = np.empty(
            capacity, dtype=np.bool_
        )

        self.learning_truncated = np.empty(
            capacity, dtype=np.bool_
        )

        self.terminal_observation_reused = np.empty(
            capacity, dtype=np.bool_
        )

        self.policy_overrides = np.empty(
            capacity, dtype=np.bool_
        )

        self.phases = [None] * capacity
        self.policy_reasons = [None] * capacity

        # Circular buffer state
        self._size = 0
        self._position = 0
        self.total_added = 0

    # ==================================================
    # 1. Buffer status
    # ==================================================

    def __len__(self):
        return self._size

    @property
    def position(self):
        """Next physical slot to write."""
        return self._position

    def can_sample(self, batch_size):

        return (
            type(batch_size) is int
            and batch_size > 0
            and batch_size <= self._size
        )

    # ==================================================
    # 2. Input validation
    # ==================================================

    @staticmethod
    def _valid_vector(value, dim, name):

        array = np.asarray(
            value, dtype=np.float32
        )

        if array.shape != (dim,):
            raise ValueError(
                f'{name} must have shape ({dim},)'
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f'{name} contains non-finite values'
            )

        if (
            np.any(array < -1.0)
            or np.any(array > 1.0)
        ):
            raise ValueError(
                f'{name} outside [-1, 1]'
            )

        return array

    # ==================================================
    # 3. Add transition
    # ==================================================

    def add(self, transition):
        """
        Store one validated SACTransition.

        All checks occur before modifying buffer data.

        Returns:
            Physical storage index.
        """

        if not isinstance(transition, SACTransition):
            raise TypeError(
                'add() requires SACTransition'
            )

        obs = self._valid_vector(
            transition.observation,
            OBS_DIM,
            'observation'
        )

        action = self._valid_vector(
            transition.action,
            ACTION_DIM,
            'action'
        )

        next_obs = self._valid_vector(
            transition.next_observation,
            OBS_DIM,
            'next_observation'
        )

        reward = float(transition.reward)
        mask = float(transition.bootstrap_mask)

        if (
            not np.isfinite(reward)
            or abs(reward) > np.finfo(np.float32).max
        ):
            raise ValueError('Invalid float32 reward')

        if mask not in (0.0, 1.0):
            raise ValueError(
                'bootstrap_mask must be 0.0 or 1.0'
            )

        if (
            type(transition.episode_index) is not int
            or transition.episode_index < 0
        ):
            raise ValueError('Invalid episode_index')

        if (
            type(transition.step_index) is not int
            or transition.step_index < 1
        ):
            raise ValueError('Invalid step_index')

        flags = (
            transition.terminated,
            transition.truncated,
            transition.learning_terminated,
            transition.learning_truncated,
            transition.terminal_observation_reused,
            transition.policy_override
        )

        if any(type(flag) is not bool for flag in flags):
            raise ValueError(
                'Transition flags must be bool'
            )

        if (
            transition.learning_terminated
            and transition.learning_truncated
        ):
            raise ValueError(
                'Learning terminal flags conflict'
            )

        if (
            transition.learning_terminated
            and mask != 0.0
        ):
            raise ValueError(
                'Cannot bootstrap a learning termination'
            )

        if (
            transition.terminal_observation_reused
            and mask != 0.0
        ):
            raise ValueError(
                'Cannot bootstrap reused observation'
            )

        phase = transition.phase

        if phase not in TERMINAL_REWARDS:
            raise ValueError(
                'Unknown transition phase'
            )

        if phase == 'EPISODE_RUNNING':

            if (
                transition.terminated
                or transition.truncated
                or transition.learning_terminated
                or transition.learning_truncated
                or mask != 1.0
            ):
                raise ValueError(
                    'Inconsistent RUNNING transition'
                )

        elif (
            transition.terminated,
            transition.truncated
        ) != TERMINAL_FLAGS[phase]:

            raise ValueError(
                'Inconsistent terminal flags'
            )

        if (
            not isinstance(transition.policy_reason, str)
            or not transition.policy_reason
        ):
            raise ValueError(
                'Invalid policy_reason'
            )

        components_dict = transition.reward_components

        if not isinstance(components_dict, dict):
            raise ValueError(
                'reward_components must be a dictionary'
            )

        if set(components_dict) != set(REWARD_KEYS):
            raise ValueError(
                'Unexpected reward component keys'
            )

        components = np.array(
            [
                float(components_dict[name])
                for name in REWARD_KEYS
            ],
            dtype=np.float64
        )

        if (
            not np.isfinite(components).all()
            or np.any(
                np.abs(components)
                > np.finfo(np.float32).max
            )
        ):
            raise ValueError(
                'Invalid reward components'
            )

        if not np.isclose(
            components.sum(),
            reward,
            atol=1e-5,
            rtol=1e-6
        ):
            raise ValueError(
                'Reward and component total disagree'
            )

        # ----------------------------------------------
        # Store transition after successful validation
        # ----------------------------------------------

        index = self._position

        self.observations[index] = obs
        self.actions[index] = action
        self.rewards[index, 0] = reward

        self.next_observations[index] = next_obs
        self.bootstrap_masks[index, 0] = mask

        self.reward_components[index] = components

        self.episode_indices[index] = (
            transition.episode_index
        )

        self.step_indices[index] = (
            transition.step_index
        )

        self.terminated[index] = (
            transition.terminated
        )

        self.truncated[index] = (
            transition.truncated
        )

        self.learning_terminated[index] = (
            transition.learning_terminated
        )

        self.learning_truncated[index] = (
            transition.learning_truncated
        )

        self.terminal_observation_reused[index] = (
            transition.terminal_observation_reused
        )

        self.policy_overrides[index] = (
            transition.policy_override
        )

        self.phases[index] = phase

        self.policy_reasons[index] = (
            transition.policy_reason
        )

        # ----------------------------------------------
        # Update circular buffer
        # ----------------------------------------------

        self._position = (
            index + 1
        ) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity
        )

        self.total_added += 1

        return index

    # ==================================================
    # 4. Uniform minibatch sampling
    # ==================================================

    def sample(self, batch_size):
        """
        Sample without replacement.

        Returned arrays are independent copies.
        """

        if not self.can_sample(batch_size):
            raise ValueError(
                'batch_size must be in [1, buffer size]'
            )

        indices = self.rng.choice(
            self._size,
            size=batch_size,
            replace=False
        )

        return {
            'observations':
                self.observations[indices].copy(),

            'actions':
                self.actions[indices].copy(),

            'rewards':
                self.rewards[indices].copy(),

            'next_observations':
                self.next_observations[indices].copy(),

            'bootstrap_masks':
                self.bootstrap_masks[indices].copy(),

            'reward_components':
                self.reward_components[indices].copy(),

            'episode_indices':
                self.episode_indices[indices].copy(),

            'step_indices':
                self.step_indices[indices].copy(),

            'terminated':
                self.terminated[indices].copy(),

            'truncated':
                self.truncated[indices].copy(),

            'learning_terminated':
                self.learning_terminated[indices].copy(),

            'learning_truncated':
                self.learning_truncated[indices].copy(),

            'terminal_observation_reused':
                self.terminal_observation_reused[
                    indices
                ].copy(),

            'policy_overrides':
                self.policy_overrides[indices].copy(),

            'phases': [
                self.phases[int(i)]
                for i in indices
            ],

            'policy_reasons': [
                self.policy_reasons[int(i)]
                for i in indices
            ],

            'indices': indices.copy()
        }

    # ==================================================
    # 5. Clear buffer
    # ==================================================

    def clear(self):
        """Clear entries while keeping allocated arrays."""

        self._size = 0
        self._position = 0
        self.total_added = 0

        self.phases[:] = [None] * self.capacity
        self.policy_reasons[:] = [None] * self.capacity
