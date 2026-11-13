#!/usr/bin/env python3
"""Step 4.3b: Soft Actor-Critic updates for normalized 30D/5D RL-MPC.

No ROS2/Gazebo interfaces. Expects NumPy minibatches returned by
SACReplayBuffer.sample(). All critic targets use bootstrap_masks, NOT done.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_mpc_training.agents.sac_networks import (
    ACTION_DIM,
    HIDDEN_SIZES,
    OBS_DIM,
    SquashedGaussianActor,
    TwinQCritic,
)


class SACAgent:
    """One-step SAC updates with twin Q targets and automatic entropy tuning."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=HIDDEN_SIZES,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        init_alpha=0.2,
        automatic_entropy_tuning=True,
        target_entropy=None,
        device='cpu',
    ):
        if type(obs_dim) is not int or obs_dim < 1:
            raise ValueError('obs_dim must be a positive integer')
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError('action_dim must be a positive integer')

        for name, value, lower, upper in (
            ('gamma', gamma, 0.0, 1.0),
            ('tau', tau, 0.0, 1.0),
        ):
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name} outside [{lower}, {upper}]')
        if tau <= 0.0:
            raise ValueError('tau must be > 0')

        for name, value in (
            ('actor_lr', actor_lr), ('critic_lr', critic_lr),
            ('alpha_lr', alpha_lr), ('init_alpha', init_alpha),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and > 0')
        if type(automatic_entropy_tuning) is not bool:
            raise ValueError('automatic_entropy_tuning must be bool')
        if target_entropy is not None and not np.isfinite(target_entropy):
            raise ValueError('target_entropy must be finite or None')

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(device)
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = float(
            -action_dim if target_entropy is None else target_entropy
        )
        self.update_count = 0

        self.actor = SquashedGaussianActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes
        ).to(self.device)
        self.critic = TwinQCritic(
            obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)
        self.critic_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Optimize log(alpha) to keep alpha positive, matching modern SAC.
        self.log_alpha = nn.Parameter(torch.tensor(
            math.log(init_alpha), dtype=torch.float32, device=self.device
        ))
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            if automatic_entropy_tuning else None
        )
        if not automatic_entropy_tuning:
            self.log_alpha.requires_grad_(False)

    @property
    def alpha(self):
        """Current scalar entropy coefficient (diagnostic; no autograd)."""
        return float(self.log_alpha.detach().exp().item())

    def act(self, observation, deterministic=False):
        """Normalized NumPy observation -> normalized NumPy 5D action."""
        return self.actor.act(observation, deterministic=deterministic)

    def _prepare_batch(self, batch):
        """Validate the five SAC arrays from SACReplayBuffer.sample()."""
        if not isinstance(batch, dict):
            raise ValueError('batch must be a dictionary')
        spec = {
            'observations': self.obs_dim,
            'actions': self.action_dim,
            'rewards': 1,
            'next_observations': self.obs_dim,
            'bootstrap_masks': 1,
        }
        result = {}
        batch_size = None
        for name, width in spec.items():
            if name not in batch:
                raise ValueError(f'missing minibatch field: {name}')
            try:
                arr = np.asarray(batch[name], dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{name} is not numeric') from exc
            if arr.ndim != 2 or arr.shape[1] != width or arr.shape[0] == 0:
                raise ValueError(f'{name} must have shape (B, {width}), B >= 1')
            if batch_size is None:
                batch_size = arr.shape[0]
            elif arr.shape[0] != batch_size:
                raise ValueError('inconsistent minibatch batch sizes')
            if not np.isfinite(arr).all():
                raise ValueError(f'non-finite minibatch values: {name}')
            if name in ('observations', 'actions', 'next_observations'):
                if np.any(arr < -1.0) or np.any(arr > 1.0):
                    raise ValueError(f'{name} outside normalized [-1, 1]')
            if name == 'bootstrap_masks' and not np.isin(arr, (0.0, 1.0)).all():
                raise ValueError('bootstrap_masks must contain only 0 or 1')
            result[name] = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        return result

    @torch.no_grad()
    def _critic_targets_from_tensors(self, b):
        next_action, next_logp = self.actor(b['next_observations'])
        q1_next, q2_next = self.critic_target(b['next_observations'], next_action)
        soft_next_q = torch.minimum(q1_next, q2_next) - (
            self.log_alpha.detach().exp() * next_logp
        )
        # A zero mask makes the target exactly reward, even for reused state.
        return b['rewards'] + self.gamma * b['bootstrap_masks'] * soft_next_q

    def compute_critic_targets(self, batch):
        """Detached (B,1) target for inspection and unit tests."""
        b = self._prepare_batch(batch)
        return self._critic_targets_from_tensors(b)

    @torch.no_grad()
    def soft_update_targets(self):
        """Polyak: target = (1-tau)*target + tau*online."""
        for target_param, source_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.lerp_(source_param, self.tau)

    def update(self, batch):
        """Perform one critic, actor, alpha, and target update; return metrics."""
        b = self._prepare_batch(batch)
        obs = b['observations']
        action = b['actions']

        # 1. Critic update. Never backpropagate through Bellman targets.
        target_q = self._critic_targets_from_tensors(b)
        q1, q2 = self.critic(obs, action)
        critic_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('non-finite critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor update. Freeze critic weights but preserve dQ/da gradient.
        self.critic.requires_grad_(False)
        try:
            pi_action, log_pi = self.actor(obs)
            q1_pi, q2_pi = self.critic(obs, pi_action)
            actor_loss = (
                self.log_alpha.detach().exp() * log_pi
                - torch.minimum(q1_pi, q2_pi)
            ).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError('non-finite actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        # 3. Alpha update; detached log-probability prevents actor gradients.
        alpha_loss_value = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi.detach() + self.target_entropy)
            ).mean()
            if not torch.isfinite(alpha_loss):
                raise FloatingPointError('non-finite alpha loss')
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().item())

        # 4. Target Q update.
        self.soft_update_targets()
        self.update_count += 1

        return {
            'update_count': self.update_count,
            'batch_size': int(obs.shape[0]),
            'critic_loss': float(critic_loss.detach().item()),
            'actor_loss': float(actor_loss.detach().item()),
            'alpha_loss': alpha_loss_value,
            'alpha': self.alpha,
            'target_q_mean': float(target_q.mean().detach().item()),
            'q1_mean': float(q1.mean().detach().item()),
            'q2_mean': float(q2.mean().detach().item()),
            'policy_entropy_estimate': float(-log_pi.detach().mean().item()),
            'bootstrap_fraction': float(b['bootstrap_masks'].mean().item()),
        }
