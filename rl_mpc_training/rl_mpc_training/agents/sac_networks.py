#!/usr/bin/env python3
"""Step 4.3a: Actor and twin Q-critic networks for normalized SAC actions.

The action produced by the actor is *normalized* to [-1, 1] in each of
five dimensions. MPC parameter conversion remains in RLMPCEnv.step().
"""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


OBS_DIM = 30
ACTION_DIM = 5
HIDDEN_SIZES = (256, 256)
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _make_mlp(input_dim, hidden_sizes):
    layers = []
    last_dim = input_dim
    for width in hidden_sizes:
        if type(width) is not int or width <= 0:
            raise ValueError('hidden layer widths must be positive integers')
        layers.extend((nn.Linear(last_dim, width), nn.ReLU()))
        last_dim = width
    return nn.Sequential(*layers), last_dim


class SquashedGaussianActor(nn.Module):
    """Reparameterized, tanh-squashed diagonal Gaussian policy."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=HIDDEN_SIZES,
    ):
        super().__init__()
        if type(obs_dim) is not int or obs_dim < 1:
            raise ValueError('obs_dim must be a positive integer')
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError('action_dim must be a positive integer')

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trunk, last_dim = _make_mlp(obs_dim, hidden_sizes)
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """Input (B, obs_dim); return actions (B, act_dim) and logp (B, 1) or None."""
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f'obs must have shape (batch, {self.obs_dim})')

        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features), min=LOG_STD_MIN, max=LOG_STD_MAX
        )
        std = log_std.exp()
        distribution = Normal(mean, std)

        pre_tanh = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = None
        if with_logprob:
            # Stable log|d tanh(z)/dz| for a Gaussian pre-squash sample z.
            # This is equivalent to log(1 - tanh(z)^2), but is numerically
            # stable for large |z|. Sum across independent action components.
            log_det = 2.0 * (
                math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
            )
            log_prob = (
                distribution.log_prob(pre_tanh) - log_det
            ).sum(dim=-1, keepdim=True)

        return action, log_prob

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        """One 30D NumPy observation -> one float32 normalized 5D action."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f'observation must have shape ({self.obs_dim},)')
        if not np.isfinite(obs).all():
            raise ValueError('observation contains non-finite values')
        if np.any(obs < -1.0) or np.any(obs > 1.0):
            raise ValueError('normalized observation outside [-1, 1]')

        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = self.forward(obs_t, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy().astype(np.float32, copy=True)


class QNetwork(nn.Module):
    """One scalar Q(o,a) estimator."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=HIDDEN_SIZES,
    ):
        super().__init__()
        if type(obs_dim) is not int or obs_dim < 1:
            raise ValueError('obs_dim must be a positive integer')
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError('action_dim must be a positive integer')

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trunk, last_dim = _make_mlp(obs_dim + action_dim, hidden_sizes)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, obs, action):
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f'obs must have shape (batch, {self.obs_dim})')
        if action.ndim != 2 or action.shape != (obs.shape[0], self.action_dim):
            raise ValueError(f'action must have shape (batch, {self.action_dim})')
        features = self.trunk(torch.cat((obs, action), dim=-1))
        return self.value_head(features)


class TwinQCritic(nn.Module):
    """Two independent Q estimators for clipped double Q-learning."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=HIDDEN_SIZES,
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_sizes)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
