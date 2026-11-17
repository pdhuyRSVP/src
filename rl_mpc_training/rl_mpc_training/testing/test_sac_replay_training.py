#!/usr/bin/env python3
"""Step 4.3c: SACAgent updates on actual SACReplayBuffer minibatches.

This is a CPU-only integration test using *synthetic, policy-valid*
SACTransitions. It does not control ROS2/Gazebo and does not prove
that the learned policy improves navigation.
"""

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent
from rl_mpc_training.utils.reward_function import TERMINAL_REWARDS
from rl_mpc_training.utils.sac_replay_buffer import SACReplayBuffer
from rl_mpc_training.utils.sac_transition import build_sac_transition


SEED = 431
EPISODES = 16
STEPS_PER_EPISODE = 25
BUFFER_CAPACITY = 512
BATCH_SIZE = 64
UPDATE_STEPS = 50

TERMINAL_PHASES = (
    'EPISODE_SUCCESS',
    'EPISODE_COLLISION',
    'EPISODE_TIMEOUT',
    'EPISODE_PLANNING_FAILURE',
)

REWARD_KEYS = (
    'progress', 'path', 'heading', 'clearance',
    'unknown_lidar', 'action_change', 'time', 'terminal',
)


def make_observation(rng):
    """Create valid 30D normalized features for an offline test."""

    obs = rng.uniform(
        -0.7, 0.7, size=30
    ).astype(np.float32)

    obs[7:12] = rng.uniform(
        0.1, 0.95, size=5
    )

    valid = rng.integers(
        0, 2, size=5
    ).astype(np.float32)

    obs[12:17] = valid

    obs[17:22] = (
        valid * rng.integers(0, 2, size=5)
    )

    obs[25:29] = rng.uniform(
        0.0, 1.0, size=4
    )

    obs[29] = rng.uniform(0.2, 1.0)

    return obs


def make_transition(
    rng,
    observation,
    episode_index,
    step_index,
    phase,
    planning_failure_mode='task_failure'
):
    """Use the existing Replay Policy and SACTransition builder."""

    terminal = phase != 'EPISODE_RUNNING'

    terminated = phase in (
        'EPISODE_SUCCESS',
        'EPISODE_COLLISION'
    )

    truncated = phase in (
        'EPISODE_TIMEOUT',
        'EPISODE_PLANNING_FAILURE'
    )

    reused = (
        phase == 'EPISODE_PLANNING_FAILURE'
    )

    action = rng.uniform(
        -0.9, 0.9, size=5
    ).astype(np.float32)

    next_observation = (
        observation.copy()
        if reused
        else make_observation(rng)
    )

    components = {
        key: 0.0
        for key in REWARD_KEYS
    }

    components['terminal'] = float(
        TERMINAL_REWARDS[phase]
    )

    if not reused:

        components['progress'] = float(
            rng.uniform(-0.3, 1.4)
        )

        components['path'] = -float(
            rng.uniform(0.0, 0.2)
        )

        components['heading'] = -float(
            rng.uniform(0.0, 0.1)
        )

        components['clearance'] = -float(
            rng.uniform(0.0, 0.25)
        )

        components['time'] = -0.02

    reward = float(
        sum(components.values())
    )

    info = {
        'phase': phase,

        'action_applied': True,

        'terminal_observation_reused': reused,

        'reward_mode': (
            'event_only' if reused else 'full'
        ),

        'reward_components': components,
    }

    if terminal:

        info['replay_eligible'] = bool(
            terminated or not reused
        )

    transition, policy = build_sac_transition(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,

        terminated=terminated,
        truncated=truncated,

        info=info,

        episode_index=episode_index,
        step_index=step_index,

        planning_failure_mode=planning_failure_mode,
    )

    return transition, policy, next_observation


def fill_buffer(buffer):
    """Fill a single replay buffer with multiple valid episode types."""

    rng = np.random.default_rng(SEED)

    expected_terminal_counts = {
        phase: 0
        for phase in TERMINAL_PHASES
    }

    for episode_index in range(EPISODES):

        observation = make_observation(rng)

        terminal_phase = TERMINAL_PHASES[
            episode_index % len(TERMINAL_PHASES)
        ]

        for step_index in range(
            1, STEPS_PER_EPISODE + 1
        ):

            phase = (
                terminal_phase
                if step_index == STEPS_PER_EPISODE
                else 'EPISODE_RUNNING'
            )

            (
                transition,
                policy,
                next_observation
            ) = make_transition(
                rng,
                observation,
                episode_index,
                step_index,
                phase
            )

            assert transition is not None
            assert policy['store_transition']

            expected_mask = (
                1.0
                if phase in (
                    'EPISODE_RUNNING',
                    'EPISODE_TIMEOUT'
                )
                else 0.0
            )

            assert (
                transition.bootstrap_mask
                == expected_mask
            )

            if phase == 'EPISODE_PLANNING_FAILURE':

                assert transition.truncated
                assert transition.learning_terminated

                assert (
                    transition.terminal_observation_reused
                )

                assert transition.policy_override

                np.testing.assert_array_equal(
                    transition.observation,
                    transition.next_observation,
                )

            if phase == 'EPISODE_TIMEOUT':

                assert transition.truncated

                assert not transition.learning_terminated

                assert (
                    transition.bootstrap_mask == 1.0
                )

            buffer.add(transition)

            observation = next_observation.copy()

        expected_terminal_counts[terminal_phase] += 1

    assert (
        len(buffer)
        == EPISODES * STEPS_PER_EPISODE
    )

    assert buffer.total_added == len(buffer)

    assert len(buffer) < buffer.capacity

    # External interruption must not be stored.
    rng = np.random.default_rng(SEED + 1)

    before = len(buffer)

    observation = make_observation(rng)

    dropped, policy, _ = make_transition(
        rng,
        observation,
        EPISODES,
        1,
        'EPISODE_PLANNING_FAILURE',
        planning_failure_mode='external_interruption',
    )

    assert dropped is None

    assert (
        policy['reason']
        == 'planning_failure_external_interruption'
    )

    assert not policy['store_transition']

    assert len(buffer) == before

    return expected_terminal_counts


def inspect_buffer(buffer, expected_terminal_counts):
    """Inspect every stored sample."""

    all_data = buffer.sample(
        len(buffer)
    )

    phases = np.array(
        all_data['phases'],
        dtype=object
    )

    masks = all_data[
        'bootstrap_masks'
    ][:, 0]

    assert all_data['observations'].shape == (
        len(buffer), 30
    )

    assert all_data['actions'].shape == (
        len(buffer), 5
    )

    assert all_data['rewards'].shape == (
        len(buffer), 1
    )

    assert all_data['next_observations'].shape == (
        len(buffer), 30
    )

    np.testing.assert_allclose(
        all_data['rewards'][:, 0],
        all_data['reward_components'].sum(axis=1),
        rtol=1e-6,
        atol=2e-5,
    )

    for phase, count in expected_terminal_counts.items():

        rows = np.flatnonzero(
            phases == phase
        )

        assert len(rows) == count, (
            phase, len(rows), count
        )

        target_mask = (
            1.0
            if phase == 'EPISODE_TIMEOUT'
            else 0.0
        )

        assert np.all(
            masks[rows] == target_mask
        )

        if phase == 'EPISODE_PLANNING_FAILURE':

            assert np.all(
                all_data['truncated'][rows]
            )

            assert np.all(
                all_data['learning_terminated'][rows]
            )

            assert np.all(
                all_data[
                    'terminal_observation_reused'
                ][rows]
            )

            assert np.all(
                all_data['policy_overrides'][rows]
            )

            np.testing.assert_array_equal(
                all_data['observations'][rows],
                all_data['next_observations'][rows],
            )

    running = np.flatnonzero(
        phases == 'EPISODE_RUNNING'
    )

    assert np.all(
        masks[running] == 1.0
    )

    for episode_index in range(EPISODES):

        rows = np.flatnonzero(
            all_data['episode_indices']
            == episode_index
        )

        rows = rows[
            np.argsort(
                all_data['step_indices'][rows]
            )
        ]

        np.testing.assert_array_equal(
            all_data['step_indices'][rows],
            np.arange(
                1, STEPS_PER_EPISODE + 1
            ),
        )

        np.testing.assert_array_equal(
            all_data['observations'][rows[1:]],
            all_data['next_observations'][rows[:-1]],
        )

    print(
        '[PASS] Buffer stored all episodes '
        'without mixing data'
    )

    print(
        '[PASS] Reward components and '
        'per-phase bootstrap masks'
    )

    print(
        '[PASS] Reused PLANNING_FAILURE '
        'and skipped external interruption'
    )

    return all_data, phases


def make_mixed_target_batch(data, phases):
    """One RUNNING and one of each terminal type."""

    names = (
        'EPISODE_RUNNING',
    ) + TERMINAL_PHASES

    rows = np.array([
        np.flatnonzero(
            phases == name
        )[0]
        for name in names
    ])

    keys = (
        'observations',
        'actions',
        'rewards',
        'next_observations',
        'bootstrap_masks',
    )

    batch = {
        name: data[name][rows].copy()
        for name in keys
    }

    return batch, names


def parameters(model):

    return [
        p.detach().cpu().clone()
        for p in model.parameters()
    ]


def changed(before, model):

    return any(
        not torch.equal(
            old,
            new.detach().cpu()
        )
        for old, new in zip(
            before,
            model.parameters()
        )
    )


def test_agent_updates(buffer, all_data, phases):

    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    agent = SACAgent(
        obs_dim=30,
        action_dim=5,

        gamma=0.99,
        tau=0.005,

        init_alpha=0.2,

        automatic_entropy_tuning=True,

        device='cpu',
    )

    # ==================================================
    # 1. Bellman targets
    # ==================================================

    print(
        '\n[3] Verify Bellman targets '
        'from Replay Buffer'
    )

    target_batch, phase_names = (
        make_mixed_target_batch(
            all_data,
            phases
        )
    )

    targets = agent.compute_critic_targets(
        target_batch
    )

    assert targets.shape == (
        len(phase_names), 1
    )

    assert not targets.requires_grad

    assert torch.isfinite(targets).all()

    targets_np = (
        targets.detach().cpu().numpy()
    )

    zero_mask = (
        target_batch['bootstrap_masks'][:, 0]
        == 0
    )

    one_mask = ~zero_mask

    assert zero_mask.sum() == 3
    assert one_mask.sum() == 2

    np.testing.assert_array_equal(
        targets_np[zero_mask],
        target_batch['rewards'][zero_mask]
    )

    # TIMEOUT bootstraps; PLANNING_FAILURE does not.
    timeout_idx = phase_names.index(
        'EPISODE_TIMEOUT'
    )

    planning_idx = phase_names.index(
        'EPISODE_PLANNING_FAILURE'
    )

    assert (
        target_batch['bootstrap_masks'][
            timeout_idx, 0
        ] == 1.0
    )

    assert (
        target_batch['bootstrap_masks'][
            planning_idx, 0
        ] == 0.0
    )

    print(
        '[PASS] Mask=0 implies target Q '
        'equals reward exactly'
    )

    print(
        '[PASS] TIMEOUT bootstraps, '
        'PLANNING_FAILURE does not'
    )

    # ==================================================
    # 2. Repeated SAC updates
    # ==================================================

    print(
        '\n[4] Repeated SAC updates '
        'using sampled minibatches'
    )

    actor_before = parameters(
        agent.actor
    )

    critic_before = parameters(
        agent.critic
    )

    targets_before = parameters(
        agent.critic_target
    )

    alpha_before = agent.alpha

    losses = []
    bootstrap_fractions = []

    for i in range(UPDATE_STEPS):

        batch = buffer.sample(
            BATCH_SIZE
        )

        originals = {
            key: batch[key].copy()
            for key in (
                'observations',
                'actions',
                'rewards',
                'next_observations',
                'bootstrap_masks',
            )
        }

        metrics = agent.update(batch)

        assert (
            metrics['update_count']
            == i + 1
        )

        assert (
            metrics['batch_size']
            == BATCH_SIZE
        )

        for key in (
            'critic_loss',
            'actor_loss',
            'alpha_loss',
            'alpha',
            'target_q_mean',
            'q1_mean',
            'q2_mean',
            'policy_entropy_estimate',
            'bootstrap_fraction'
        ):

            assert metrics[key] is not None

            assert np.isfinite(
                metrics[key]
            ), key

        assert metrics['alpha'] > 0.0

        np.testing.assert_allclose(
            metrics['bootstrap_fraction'],
            float(
                batch['bootstrap_masks'].mean()
            ),
            atol=1e-7,
            rtol=0.0,
        )

        for key in originals:

            np.testing.assert_array_equal(
                batch[key],
                originals[key]
            )

        assert all(
            torch.isfinite(p).all()
            for p in agent.actor.parameters()
        )

        assert all(
            torch.isfinite(p).all()
            for p in agent.critic.parameters()
        )

        assert all(
            torch.isfinite(p).all()
            for p in agent.critic_target.parameters()
        )

        losses.append(
            (
                metrics['critic_loss'],
                metrics['actor_loss'],
                metrics['alpha']
            )
        )

        bootstrap_fractions.append(
            metrics['bootstrap_fraction']
        )

        if i in (
            0, 9, 24, UPDATE_STEPS - 1
        ):

            print(
                f'[UPDATE {i + 1:03d}] '
                f'critic_loss={metrics["critic_loss"]:.4f} | '
                f'actor_loss={metrics["actor_loss"]:.4f} | '
                f'alpha={metrics["alpha"]:.4f} | '
                f'bootstrap_fraction='
                f'{metrics["bootstrap_fraction"]:.3f}'
            )

    # ==================================================
    # 3. Validate learned parameters
    # ==================================================

    assert changed(
        actor_before,
        agent.actor
    )

    assert changed(
        critic_before,
        agent.critic
    )

    assert changed(
        targets_before,
        agent.critic_target
    )

    assert not np.isclose(
        agent.alpha,
        alpha_before,
        rtol=0.0,
        atol=1e-7
    )

    assert all(
        p.grad is None
        for p in agent.critic_target.parameters()
    )

    for name, model in (
        ('Actor', agent.actor),
        ('Critic', agent.critic)
    ):

        grads = [
            p.grad
            for p in model.parameters()
            if p.grad is not None
        ]

        assert grads, name

        assert all(
            torch.isfinite(g).all()
            for g in grads
        ), name

        assert any(
            torch.any(g.abs() > 0)
            for g in grads
        ), name

    assert len(losses) == UPDATE_STEPS

    assert np.isfinite(
        np.asarray(losses)
    ).all()

    assert all(
        0.0 <= fraction <= 1.0
        for fraction in bootstrap_fractions
    )

    # ==================================================
    # 4. Validate actor actions
    # ==================================================

    sample_obs = all_data[
        'observations'
    ][0]

    for deterministic in (
        True, False
    ):

        act = agent.act(
            sample_obs,
            deterministic=deterministic
        )

        assert act.shape == (5,)

        assert act.dtype == np.float32

        assert np.isfinite(
            act
        ).all()

        assert np.all(
            (-1 <= act) & (act <= 1)
        )

    print(
        '[PASS] Actor, Critic, '
        'Target Critic, alpha changed'
    )

    print(
        '[PASS] Gradients and parameters '
        'remain finite'
    )

    print(
        '[PASS] Sampled batch arrays '
        'were not modified'
    )

    print(
        '[PASS] Both deterministic/stochastic '
        'actions remain valid'
    )


def main():

    print(
        '\nStep 4.3c - SAC x '
        'SACReplayBuffer (CPU only)'
    )

    print(
        '[1] Populate Replay Buffer with '
        'synthetic validated transitions'
    )

    buffer = SACReplayBuffer(
        capacity=BUFFER_CAPACITY,
        seed=SEED
    )

    counts = fill_buffer(
        buffer
    )

    assert (
        len(buffer)
        == EPISODES * STEPS_PER_EPISODE
    )

    print(
        f'[PASS] Stored {len(buffer)} '
        f'transitions across {EPISODES} episodes'
    )

    print(
        '[2] Validate Replay Buffer contents'
    )

    all_data, phases = inspect_buffer(
        buffer,
        counts
    )

    test_agent_updates(
        buffer,
        all_data,
        phases
    )

    print(
        '\n[PASS] SAC Replay Buffer '
        'integration tests completed'
    )


if __name__ == '__main__':
    main()
