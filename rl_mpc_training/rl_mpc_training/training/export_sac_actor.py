#!/usr/bin/env python3
"""Export a frozen SAC actor from a trusted training checkpoint.

CPU-only. Does not import ROS 2, access Gazebo, or modify the source checkpoint.
The actor emits policy COMMANDS, not directly applied MPC parameters.
"""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from rl_mpc_training.agents.sac_networks import (
    ACTION_DIM,
    HIDDEN_SIZES,
    OBS_DIM,
    SquashedGaussianActor,
)

POLICY_FORMAT = 'sac_mpc_actor_v1'

EXPORTABLE_TERMINAL_PHASES = frozenset({
    'EPISODE_SUCCESS',
    'EPISODE_COLLISION',
    'EPISODE_TIMEOUT',
    'EPISODE_PLANNING_FAILURE',
})


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_actor(state_dict):
    actor = SquashedGaussianActor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=HIDDEN_SIZES,
    ).cpu()
    actor.load_state_dict(state_dict, strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor


def load_exported_actor(policy_path):
    """Safe loading path for a policy produced by this exporter."""
    policy_path = Path(policy_path).expanduser()
    payload = torch.load(policy_path, map_location='cpu', weights_only=True)

    if not isinstance(payload, dict) or payload.get('format') != POLICY_FORMAT:
        raise ValueError('Unsupported actor policy format')
    if payload.get('observation_dim') != OBS_DIM:
        raise ValueError('Policy observation dimension mismatch')
    if payload.get('action_dim') != ACTION_DIM:
        raise ValueError('Policy action dimension mismatch')
    if tuple(payload.get('hidden_sizes', ())) != HIDDEN_SIZES:
        raise ValueError('Policy actor architecture mismatch')
    if payload.get('action_semantics') != 'sac_command_before_adapter':
        raise ValueError('Policy action semantics mismatch')

    actor = build_actor(payload['actor_state_dict'])
    return actor, payload['metadata']


def export_actor(checkpoint_path, output_path, expected_episodes=None):
    """Export only from a locally trusted, completed training checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Training checkpoint not found: {checkpoint_path}')

    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing policy: {output_path}')

    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError('Policy output must not replace training checkpoint')

    if expected_episodes is not None:

        if (
            type(expected_episodes) is not int
            or expected_episodes < 1
        ):

            raise ValueError(
                'expected_episodes must be a positive integer'
            )

    # Trusted local project checkpoint only. It may contain NumPy objects.
    # Never use weights_only=False for an unknown/untrusted file.
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    if not isinstance(checkpoint, dict) or checkpoint.get('version') != 1:
        raise ValueError('Unsupported training checkpoint version')

    agent_state = checkpoint['agent']
    trainer_state = checkpoint['trainer']
    episodes_log = trainer_state['episodes_log']
    updates_log = trainer_state['updates_log']

    if agent_state['obs_dim'] != OBS_DIM or agent_state['action_dim'] != ACTION_DIM:
        raise ValueError('Training actor dimensions do not match current model')

    completed_episodes = len(episodes_log)

    if completed_episodes < 1:

        raise ValueError(
            'Checkpoint contains no completed episodes'
        )

    if (
        expected_episodes is not None
        and completed_episodes != expected_episodes
    ):

        raise ValueError(
            f'Expected {expected_episodes} completed episodes, '
            f'found {completed_episodes}'
        )

    # Episode indices must be contiguous.
    if any(
        row['episode_index'] != i
        for i, row in enumerate(episodes_log)
    ):

        raise ValueError(
            'Episode indices are not contiguous'
        )

    # ======================================================
    # 2. Validate terminal phases
    # ======================================================

    phase_counts = {
        phase: 0
        for phase in sorted(EXPORTABLE_TERMINAL_PHASES)
    }

    for row in episodes_log:

        phase = row['phase']

        if phase not in EXPORTABLE_TERMINAL_PHASES:

            raise ValueError(
                f'Unsupported terminal phase: {phase}'
            )

        phase_counts[phase] += 1

    if trainer_state['env_steps'] != sum(row['steps'] for row in episodes_log):
        raise ValueError('Training step counters are inconsistent')

    if agent_state['update_count'] != len(updates_log):
        raise ValueError('SAC update counters are inconsistent')

    source_actor = build_actor(agent_state['actor'])

    metadata = {
        'completed_episodes': completed_episodes,

        'training_env_steps': int(trainer_state['env_steps']),

        'sac_updates': int(agent_state['update_count']),

        'terminal_phase_counts': phase_counts,

        'source_checkpoint_sha256': sha256_file(checkpoint_path),

        'inference_mode': 'deterministic',
    }

    policy = {
        'format': POLICY_FORMAT,
        'observation_dim': OBS_DIM,
        'action_dim': ACTION_DIM,
        'hidden_sizes': list(HIDDEN_SIZES),
        'action_semantics': 'sac_command_before_adapter',
        'actor_state_dict': {
            key: value.detach().cpu().clone()
            for key, value in source_actor.state_dict().items()
        },
        'metadata': metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', prefix='.actor_tmp_', suffix='.pt',
            dir=output_path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            torch.save(policy, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Verify the saved artifact before publishing it.
        loaded_actor, loaded_metadata = load_exported_actor(temp_path)
        probe = np.zeros(OBS_DIM, dtype=np.float32)
        probe[-1] = 0.5  # Default MPC speed feature.
        before = source_actor.act(probe, deterministic=True)
        after = loaded_actor.act(probe, deterministic=True)
        np.testing.assert_array_equal(before, after)
        assert metadata == loaded_metadata
        if not np.isfinite(after).all() or np.any(np.abs(after) > 1.0):
            raise ValueError('Invalid exported actor output')
        # This operation never modifies the original training checkpoint.
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return output_path, metadata, after


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Source SAC training checkpoint'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output frozen actor policy'
    )

    parser.add_argument(
        '--expected-episodes',
        type=int,
        default=None,
        help='Optional expected completed episode count'
    )

    args = parser.parse_args()

    path, metadata, test_action = export_actor(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        expected_episodes=args.expected_episodes,
    )

    print(
        '[PASS] Frozen actor exported and reloaded (CPU)'
    )

    print(f'[POLICY] {path}')

    print(f'[METADATA] {metadata}')

    print(
        '[TEST] deterministic SAC command = '
        f'{test_action.tolist()}'
    )


if __name__ == '__main__':
    main()
