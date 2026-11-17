#!/usr/bin/env python3
"""Step 4.3a: CPU-only SAC Actor/Twin-Critic unit tests (no ROS)."""

import copy

import numpy as np
import torch

from rl_mpc_training.agents.sac_networks import (
    ACTION_DIM,
    OBS_DIM,
    SquashedGaussianActor,
    TwinQCritic,
)


def expect_value_error(label, callback):
    try:
        callback()
    except ValueError:
        print(f'[PASS] {label} rejected')
    else:
        raise AssertionError(f'Expected ValueError for: {label}')


def main():
    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)

    actor = SquashedGaussianActor()
    critic = TwinQCritic()

    print('\n[1] Action and Q-value tensor dimensions')
    obs = torch.rand(16, OBS_DIM) * 2.0 - 1.0
    action, logp = actor(obs)
    assert action.shape == (16, ACTION_DIM)
    assert logp.shape == (16, 1)
    assert torch.isfinite(action).all() and torch.isfinite(logp).all()
    assert (action >= -1.0).all() and (action <= 1.0).all()
    q1, q2 = critic(obs, action)
    assert q1.shape == q2.shape == (16, 1)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    print('[PASS] Actor: 30D -> 5D bounded action + log probability')
    print('[PASS] Twin Critic: (30D, 5D) -> two scalar Q values')

    print('\n[2] Deterministic and stochastic action')
    det1, det_logp = actor(obs, deterministic=True)
    det2, _ = actor(obs, deterministic=True)
    assert det_logp.shape == (16, 1)
    torch.testing.assert_close(det1, det2, rtol=0, atol=0)
    sample1, _ = actor(obs)
    sample2, _ = actor(obs)
    assert not torch.equal(sample1, sample2)
    numpy_action = actor.act(np.zeros(OBS_DIM, dtype=np.float32), deterministic=True)
    assert numpy_action.shape == (ACTION_DIM,)
    assert numpy_action.dtype == np.float32
    assert np.isfinite(numpy_action).all()
    assert np.all((-1 <= numpy_action) & (numpy_action <= 1))
    print('[PASS] Deterministic action repeatable')
    print('[PASS] Stochastic action samples vary')
    print('[PASS] NumPy 30D -> normalized 5D action')

    print('\n[3] Squashed-Gaussian log probability')
    # Check the log-Jacobian correction numerically away from saturation.
    with torch.no_grad():
        normal_mean = actor.mean_head(actor.trunk(obs))
        raw_log_std = actor.log_std_head(actor.trunk(obs))
        normal_std = raw_log_std.clamp(-20.0, 2.0).exp()
        distribution = torch.distributions.Normal(normal_mean, normal_std)
        unsquashed_logp = distribution.log_prob(normal_mean).sum(-1, keepdim=True)
        expected_logp = unsquashed_logp - torch.log(
            1.0 - torch.tanh(normal_mean).pow(2)
        ).sum(-1, keepdim=True)
    torch.testing.assert_close(det_logp, expected_logp, rtol=1e-4, atol=1e-4)
    print('[PASS] Tanh log-probability correction')

    large_obs = torch.full((8, OBS_DIM), 1.0)
    extreme_action, extreme_logp = actor(large_obs)
    assert torch.isfinite(extreme_action).all()
    assert torch.isfinite(extreme_logp).all()
    print('[PASS] Log probability finite at observation bounds')

    print('\n[4] Gradient flow')
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    a, log_prob = actor(obs)
    qa, qb = critic(obs, a)
    actor_loss = (0.1 * log_prob - torch.minimum(qa, qb)).mean()
    actor_loss.backward()
    actor_grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert actor_grads and all(torch.isfinite(g).all() for g in actor_grads)
    assert any(g.abs().sum() > 0 for g in actor_grads)
    print('[PASS] Reparameterized action supports actor gradients')

    critic.zero_grad(set_to_none=True)
    detached_a = a.detach()
    qc1, qc2 = critic(obs, detached_a)
    target = torch.ones_like(qc1)
    loss = ((qc1-target)**2 + (qc2-target)**2).mean()
    loss.backward()
    for net in (critic.q1, critic.q2):
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    assert not ({id(p) for p in critic.q1.parameters()} & {id(p) for p in critic.q2.parameters()})
    print('[PASS] Twin Q networks independent and differentiable')

    print('\n[5] Target-network copy readiness')
    target_critic = copy.deepcopy(critic)
    for p, pt in zip(critic.parameters(), target_critic.parameters()):
        torch.testing.assert_close(p, pt)
        assert id(p) != id(pt)
    print('[PASS] Target Critic is a separate parameter copy')

    print('\n[6] Invalid input shapes')
    expect_value_error('Actor observation shape', lambda: actor(torch.zeros(OBS_DIM)))
    expect_value_error('Critic action shape', lambda: critic(obs, torch.zeros(16, 4)))
    expect_value_error('NumPy observation shape', lambda: actor.act(np.zeros(29)))
    expect_value_error('Non-finite NumPy observation', lambda: actor.act(np.full(30, np.nan)))
    print('\n[PASS] SAC Actor-Critic network tests completed')


if __name__ == '__main__':
    main()
