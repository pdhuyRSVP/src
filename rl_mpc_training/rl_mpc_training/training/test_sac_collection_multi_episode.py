#!/usr/bin/env python3
"""Step 4.2.5d.2: Real multi-episode Gymnasium -> SAC Replay Buffer test."""

import time
import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv
from rl_mpc_training.utils.action_mapping import map_action_to_mpc
from rl_mpc_training.utils.sac_transition import build_sac_transition
from rl_mpc_training.utils.sac_replay_buffer import SACReplayBuffer
from rl_mpc_training.training.test_sac_collection_episode import (
    DEFAULT_ACTION, select_action, validate_observation
)

EPISODES = 3
BUFFER_CAPACITY = 512
BATCH_SIZE = 16
MAX_STEPS = 160
MAX_EPISODE_WALL_SEC = 240.0


def collect_episode(env, buffer, episode_index):
    """Reset the SAME env and append transitions to the SAME buffer."""
    before = len(buffer)
    observation, reset_info = env.reset(seed=123 + episode_index)
    validate_observation(env, observation)
    assert reset_info['phase'] == 'EPISODE_RUNNING'
    initial_observation = observation.copy()
    initial_distance = float(reset_info['raw_observation'][4])
    assert 9.0 <= initial_distance <= 10.5, (
        f'Episode {episode_index}: reset distance {initial_distance:.3f} m'
    )
    np.testing.assert_allclose(
        reset_info['raw_observation'][25:30],
        map_action_to_mpc(DEFAULT_ACTION), atol=1e-5, rtol=0.0
    )
    assert len(buffer) == before  # reset() must not clear replay data

    path_count = int(reset_info['observation_counts']['accepted_path_count'])
    changes = 0
    episode_return = 0.0
    stored = 0
    skipped = 0
    terminal = False
    start_wall = time.monotonic()

    for step_index in range(1, MAX_STEPS + 1):
        if time.monotonic() - start_wall > MAX_EPISODE_WALL_SEC:
            raise TimeoutError(f'Episode {episode_index}: wall-time limit')

        action, label = select_action(step_index)
        next_observation, reward, terminated, truncated, info = env.step(action)
        validate_observation(env, next_observation)
        assert np.isfinite(reward)
        transition, policy = build_sac_transition(
            observation=observation, action=action, reward=reward,
            next_observation=next_observation, terminated=terminated,
            truncated=truncated, info=info,
            episode_index=episode_index, step_index=step_index,
            planning_failure_mode='task_failure'
        )
        if transition is None:
            skipped += 1
            print(f'  [SKIP {step_index:03d}] {policy["reason"]}')
        else:
            buffer.add(transition)
            stored += 1
            assert transition.bootstrap_mask == (
                0.0 if (terminated or truncated) else 1.0
            )
        episode_return += float(reward)

        if not (terminated or truncated):
            assert info['phase'] == 'EPISODE_RUNNING'
            assert int(info['step_count']) == step_index
            np.testing.assert_allclose(
                info['raw_observation'][25:30],
                map_action_to_mpc(action), atol=1e-5, rtol=0.0
            )
            count = int(info['observation_counts']['accepted_path_count'])
            if count != path_count:
                changes += 1
                print(f'  [REPLAN] Accepted path: {path_count} -> {count}')
                path_count = count

        if step_index % 10 == 0 or terminated or truncated:
            print(
                f'  [STEP {step_index:03d}] action={label} '
                f'd_goal={float(info["raw_observation"][4]):.3f} m '
                f'r={reward:.3f} buffer={len(buffer)} '
                f'mask={policy["bootstrap_mask"]:.1f}'
            )

        observation = next_observation.copy()
        if terminated or truncated:
            terminal = True
            assert info['phase'] == 'EPISODE_SUCCESS', info['phase']
            assert terminated and not truncated
            assert info['reward_components']['terminal'] == 25.0
            assert transition is not None
            assert not info['terminal_observation_reused']
            assert policy['store_transition']
            assert policy['bootstrap_mask'] == 0.0
            break

    assert terminal, f'Episode {episode_index} did not reach terminal'
    assert skipped == 0 and stored == step_index
    assert len(buffer) == before + stored

    # No new action may be submitted until the next reset().
    try:
        env.step(DEFAULT_ACTION)
    except RuntimeError as exc:
        assert 'Episode is not active' in str(exc), str(exc)
    else:
        raise AssertionError('step() accepted after terminal')

    print(f'[PASS] Episode {episode_index}: stored={stored}, '
          f'return={episode_return:.3f}, replans={changes}')
    return dict(episode_index=episode_index, steps=step_index,
                stored=stored, episode_return=episode_return,
                path_changes=changes,
                initial_observation=initial_observation)


def verify_buffer(buffer, reports, scenario):
    """Validate all stored episodes without crossing reset boundaries."""
    print('\n[VERIFY] Multi-episode Replay Buffer')
    count = sum(report['stored'] for report in reports)
    assert count < buffer.capacity, 'Test capacity must avoid overwriting'
    assert len(buffer) == buffer.total_added == count
    assert buffer.can_sample(BATCH_SIZE)

    sample = buffer.sample(BATCH_SIZE)
    assert sample['observations'].shape == (BATCH_SIZE, 30)
    assert sample['actions'].shape == (BATCH_SIZE, 5)
    assert sample['rewards'].shape == (BATCH_SIZE, 1)
    assert sample['next_observations'].shape == (BATCH_SIZE, 30)
    assert sample['bootstrap_masks'].shape == (BATCH_SIZE, 1)
    assert len(set(sample['indices'].tolist())) == BATCH_SIZE
    print('[PASS] Minibatch shapes and unique samples')

    # Sampling the full buffer without replacement allows inspecting all episodes.
    all_data = buffer.sample(len(buffer))
    episode_ids = all_data['episode_indices']
    steps = all_data['step_indices']
    assert set(episode_ids.tolist()) == set(range(EPISODES))
    np.testing.assert_allclose(
        all_data['rewards'][:, 0],
        all_data['reward_components'].sum(axis=1),
        atol=2e-5, rtol=1e-6
    )

    for report in reports:
        ep = report['episode_index']
        rows = np.flatnonzero(episode_ids == ep)
        rows = rows[np.argsort(steps[rows])]
        assert len(rows) == report['stored']
        np.testing.assert_array_equal(
            steps[rows], np.arange(1, report['steps'] + 1)
        )
        obs = all_data['observations'][rows]
        next_obs = all_data['next_observations'][rows]
        actions = all_data['actions'][rows]
        rewards = all_data['rewards'][rows, 0]
        masks = all_data['bootstrap_masks'][rows, 0]
        phases = [all_data['phases'][int(i)] for i in rows]

        np.testing.assert_array_equal(obs[0], report['initial_observation'])
        np.testing.assert_array_equal(obs[1:], next_obs[:-1])
        for j, step in enumerate(steps[rows]):
            expected_action, _ = select_action(int(step))
            np.testing.assert_array_equal(actions[j], expected_action)
        assert all(p == 'EPISODE_RUNNING' for p in phases[:-1])
        assert phases[-1] == 'EPISODE_SUCCESS'
        assert np.all(masks[:-1] == 1.0) and masks[-1] == 0.0
        assert not np.any(all_data['terminated'][rows[:-1]])
        assert bool(all_data['terminated'][rows[-1]])
        assert not np.any(all_data['truncated'][rows])
        assert bool(all_data['learning_terminated'][rows[-1]])

        replay_return = float(rewards.sum(dtype=np.float64))
        np.testing.assert_allclose(
            replay_return, report['episode_return'], atol=1e-3, rtol=1e-6
        )
        print(f'[PASS] Episode {ep}: indices, continuity, action, '
              f'reward, terminal mask (n={len(rows)})')

    total_return = sum(report['episode_return'] for report in reports)
    np.testing.assert_allclose(
        float(all_data['rewards'].sum(dtype=np.float64)),
        total_return, atol=1e-3, rtol=1e-6
    )
    print('[PASS] Buffer persists across resets without episode mixing')

    path_changes = sum(report['path_changes'] for report in reports)
    if scenario == 'OBSTACLE':
        assert path_changes > 0, 'No Accepted Path changes observed'
        print('[PASS] Replanning observed')

    print('\n[SUMMARY]')
    print('Scenario:', scenario)
    print('Episodes:', EPISODES)
    print('Stored transitions:', len(buffer))
    print('Total added:', buffer.total_added)
    print('Total episode return:', total_return)
    print('Accepted Path changes:', path_changes)
    print('[PASS] Multi-episode collection verified')


def main():
    env = RLMPCEnv()
    buffer = SACReplayBuffer(capacity=BUFFER_CAPACITY, seed=123)
    try:
        print('\nSAC Collection - Multi-Episode Integration Test')
        scenario = input('\nType CLEAR or OBSTACLE: ').strip().upper()
        if scenario not in ('CLEAR', 'OBSTACLE'):
            print('Test cancelled')
            return

        reports = []
        for episode_index in range(EPISODES):
            print(f'\n========== EPISODE {episode_index} ==========')
            reports.append(collect_episode(env, buffer, episode_index))
        verify_buffer(buffer, reports, scenario)
        print('\n[PASS] SAC multi-episode collection test completed')
    finally:
        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
