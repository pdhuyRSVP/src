#!/usr/bin/env python3
"""Step 4.3b: CPU-only SAC update tests (no ROS2/Gazebo)."""

import copy

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent
from rl_mpc_training.agents.sac_networks import ACTION_DIM, OBS_DIM


def make_batch(batch_size=32, seed=123):
    """Generate a synthetic replay batch with SACReplayBuffer.sample keys."""
    rng = np.random.default_rng(seed)
    masks = np.ones((batch_size, 1), dtype=np.float32)
    if batch_size > 0:
        masks[0] = 0.0      # SUCCESS/COLLISION/task failure -> no bootstrap
    if batch_size > 1:
        masks[1] = 1.0      # TIMEOUT with fresh next state -> bootstrap
    return {
        'observations': rng.uniform(-1, 1, (batch_size, OBS_DIM)).astype(np.float32),
        'actions': rng.uniform(-1, 1, (batch_size, ACTION_DIM)).astype(np.float32),
        'rewards': rng.normal(0.1, 1.0, (batch_size, 1)).astype(np.float32),
        'next_observations': rng.uniform(-1, 1, (batch_size, OBS_DIM)).astype(np.float32),
        'bootstrap_masks': masks,
        # Extra fields from ReplayBuffer.sample() are intentionally permitted.
        'episode_indices': np.zeros(batch_size, dtype=np.int64),
    }


def expect_value_error(name, callable_):
    try:
        callable_()
    except ValueError:
        print(f'[PASS] {name} rejected')
    else:
        raise AssertionError(f'Expected ValueError: {name}')


def copied_params(model):
    return [p.detach().clone() for p in model.parameters()]


def changed(before, model):
    return any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters()))


def main():
    torch.manual_seed(123)
    torch.set_num_threads(1)

    agent = SACAgent(device='cpu', tau=0.05, init_alpha=0.2)
    batch = make_batch()

    print('\n[1] Initialization and target critic')
    assert agent.obs_dim == 30 and agent.action_dim == 5
    assert np.isclose(agent.alpha, 0.2, atol=1e-7)
    assert agent.target_entropy == -5.0
    for source, target in zip(agent.critic.parameters(), agent.critic_target.parameters()):
        torch.testing.assert_close(source, target, atol=0, rtol=0)
        assert source is not target
        assert target.requires_grad is False
    print('[PASS] Networks, independent targets, initial alpha')

    print('\n[2] Bellman target and bootstrap masks')
    torch.manual_seed(202)
    computed = agent.compute_critic_targets(batch)
    assert computed.shape == (32, 1)
    assert not computed.requires_grad
    assert torch.isfinite(computed).all()
    np.testing.assert_allclose(computed[0].numpy(), batch['rewards'][0], atol=1e-6)

    # Independently reconstruct the SAC target with the same sampled action.
    torch.manual_seed(202)
    with torch.no_grad():
        obs_next = torch.tensor(batch['next_observations'])
        act_next, logp_next = agent.actor(obs_next)
        q1, q2 = agent.critic_target(obs_next, act_next)
        soft_value = torch.minimum(q1, q2) - agent.alpha * logp_next
        expected = torch.tensor(batch['rewards']) + (
            agent.gamma * torch.tensor(batch['bootstrap_masks']) * soft_value
        )
    torch.testing.assert_close(computed, expected, atol=1e-6, rtol=1e-6)
    print('[PASS] Correct SAC target with target critics and entropy')

    all_terminal = copy.deepcopy(batch)
    all_terminal['bootstrap_masks'] = np.zeros((32, 1), dtype=np.float32)
    target_terminal = agent.compute_critic_targets(all_terminal)
    torch.testing.assert_close(target_terminal, torch.tensor(batch['rewards']), atol=0, rtol=0)
    print('[PASS] Mask 0: target equals reward, no bootstrap')

    print('\n[3] One full SAC gradient update')
    actor_before = copied_params(agent.actor)
    critic_before = copied_params(agent.critic)
    target_before = copied_params(agent.critic_target)
    alpha_before = agent.alpha
    original_obs = batch['observations'].copy()
    stats = agent.update(batch)
    assert stats['update_count'] == 1 and stats['batch_size'] == 32
    assert agent.update_count == 1
    assert changed(actor_before, agent.actor)
    assert changed(critic_before, agent.critic)
    assert changed(target_before, agent.critic_target)
    assert stats['alpha_loss'] is not None
    assert agent.alpha > 0 and agent.alpha != alpha_before
    assert np.array_equal(batch['observations'], original_obs)
    for k in ('critic_loss', 'actor_loss', 'alpha_loss', 'alpha',
              'target_q_mean', 'policy_entropy_estimate', 'q1_mean', 'q2_mean'):
        assert np.isfinite(stats[k]), k
    np.testing.assert_allclose(stats['bootstrap_fraction'], 31 / 32, atol=1e-7)
    print('[PASS] Critic, actor, alpha updated; metrics finite')
    print('[PASS] Input minibatch remains unchanged')

    # Targets must be interpolated from the *pre-update* target towards
    # the *post-critic-update* online network, not replaced wholesale.
    for old, online, now in zip(target_before, agent.critic.parameters(), agent.critic_target.parameters()):
        expected_polyak = old * (1.0 - agent.tau) + online.detach() * agent.tau
        torch.testing.assert_close(now, expected_polyak, atol=3e-7, rtol=1e-5)
        assert now.grad is None
    print('[PASS] Polyak target update and no target gradients')

    print('\n[4] Multiple updates and action sampling')
    for i in range(12):
        log = agent.update(make_batch(seed=1000 + i))
        assert log['update_count'] == i + 2
        assert np.isfinite(log['critic_loss']) and np.isfinite(log['actor_loss'])
        assert np.isfinite(log['alpha']) and log['alpha'] > 0
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    action = agent.act(observation, deterministic=True)
    assert action.shape == (ACTION_DIM,) and action.dtype == np.float32
    assert np.isfinite(action).all() and np.all((-1 <= action) & (action <= 1))
    print('[PASS] Repeated updates finite, deterministic 5D action valid')

    print('\n[5] Fixed entropy coefficient')
    fixed = SACAgent(automatic_entropy_tuning=False, init_alpha=0.15, tau=0.1)
    fixed_before = fixed.alpha
    result = fixed.update(make_batch(batch_size=16, seed=300))
    assert result['alpha_loss'] is None
    assert fixed.alpha == fixed_before
    assert fixed.log_alpha.requires_grad is False
    print('[PASS] Fixed alpha remains constant')

    print('\n[6] Invalid batches and hyperparameters')
    expect_value_error('Invalid gamma', lambda: SACAgent(gamma=1.1))
    expect_value_error('Invalid tau', lambda: SACAgent(tau=0))
    expect_value_error('Invalid initial alpha', lambda: SACAgent(init_alpha=0))
    expect_value_error('Invalid auto-alpha flag', lambda: SACAgent(automatic_entropy_tuning='yes'))
    expect_value_error('Empty batch', lambda: agent.update(make_batch(batch_size=0)))
    wrong = make_batch()
    wrong['actions'] = np.zeros((32, 4), dtype=np.float32)
    expect_value_error('Wrong action dimension', lambda: agent.update(wrong))
    wrong = make_batch()
    wrong['bootstrap_masks'][3, 0] = 0.5
    expect_value_error('Invalid mask', lambda: agent.update(wrong))
    wrong = make_batch()
    wrong['rewards'][3, 0] = np.nan
    expect_value_error('NaN reward', lambda: agent.update(wrong))
    wrong = make_batch()
    wrong['observations'][0, 0] = 2.0
    expect_value_error('Out-of-range observation', lambda: agent.update(wrong))
    wrong = make_batch()
    del wrong['next_observations']
    expect_value_error('Missing next observation', lambda: agent.update(wrong))
    wrong = make_batch()
    wrong['rewards'] = wrong['rewards'].reshape(-1)
    expect_value_error('Incorrect reward shape', lambda: agent.update(wrong))
    print('\n[PASS] SAC agent update tests completed')


if __name__ == '__main__':
    main()
