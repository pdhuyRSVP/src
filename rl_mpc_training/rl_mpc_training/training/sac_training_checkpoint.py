#!/usr/bin/env python3
"""4.3e.5a: episode-boundary SAC checkpoint and CSV export.

CPU-only. Only load checkpoint files created by this project.
"""

import copy
import csv
import os
import tempfile

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


VERSION = 1

BUFFER_ARRAYS = (
    'observations',
    'actions',
    'rewards',
    'next_observations',
    'bootstrap_masks',
    'reward_components',
    'episode_indices',
    'step_indices',
    'terminated',
    'truncated',
    'learning_terminated',
    'learning_truncated',
    'terminal_observation_reused',
    'policy_overrides',
)

EPISODE_COLUMNS = (
    'episode_index',
    'steps',
    'return',
    'phase',
    'stored',
    'skipped',
    'updates',
    'buffer_size',
    'alpha',
    'goal_x',
    'goal_y',
    'goal_yaw',
)

UPDATE_COLUMNS = (
    'env_step',
    'update_count',
    'batch_size',
    'critic_loss',
    'actor_loss',
    'alpha_loss',
    'alpha',
    'target_q_mean',
    'q1_mean',
    'q2_mean',
    'policy_entropy_estimate',
    'bootstrap_fraction',
)


# ======================================================
# 1. Validate checkpoint boundary
# ======================================================

def _completed_boundary(agent, buffer, trainer):

    if not trainer.episodes_log:
        raise RuntimeError(
            'No completed episode to checkpoint'
        )

    completed_steps = sum(
        row['steps']
        for row in trainer.episodes_log
    )

    if trainer.total_env_steps != completed_steps:
        raise RuntimeError(
            'Checkpoint requires completed episode boundary'
        )

    if trainer.total_stored != buffer.total_added:
        raise RuntimeError(
            'Replay and trainer counters disagree'
        )

    if len(trainer.updates_log) != agent.update_count:
        raise RuntimeError(
            'SAC update counters disagree'
        )

    if buffer.position != (
        buffer.total_added % buffer.capacity
    ):
        raise RuntimeError(
            'Replay ring position inconsistent'
        )

    if len(buffer) != min(
        buffer.capacity,
        buffer.total_added
    ):
        raise RuntimeError(
            'Replay size inconsistent'
        )

    if agent.device.type != 'cpu':
        raise ValueError(
            'Checkpoint currently supports CPU agents only'
        )


# ======================================================
# 2. Replay Buffer state
# ======================================================

def _buffer_state(buffer):

    n = len(buffer)

    return {
        'capacity': buffer.capacity,
        'size': n,
        'position': buffer.position,
        'total_added': buffer.total_added,

        'arrays': {
            name: getattr(buffer, name)[:n].copy()
            for name in BUFFER_ARRAYS
        },

        'phases': list(buffer.phases[:n]),

        'policy_reasons': list(
            buffer.policy_reasons[:n]
        ),

        'rng_state': copy.deepcopy(
            buffer.rng.bit_generator.state
        ),
    }


# ======================================================
# 3. SAC Agent state
# ======================================================

def _agent_state(agent):

    return {
        'obs_dim': agent.obs_dim,
        'action_dim': agent.action_dim,

        'gamma': agent.gamma,
        'tau': agent.tau,

        'auto_alpha':
            agent.automatic_entropy_tuning,

        'target_entropy':
            agent.target_entropy,

        'actor':
            agent.actor.state_dict(),

        'critic':
            agent.critic.state_dict(),

        'critic_target':
            agent.critic_target.state_dict(),

        'actor_optimizer':
            agent.actor_optimizer.state_dict(),

        'critic_optimizer':
            agent.critic_optimizer.state_dict(),

        'log_alpha':
            agent.log_alpha.detach().cpu().clone(),

        'alpha_optimizer': (
            agent.alpha_optimizer.state_dict()
            if agent.alpha_optimizer is not None
            else None
        ),

        'update_count':
            agent.update_count,
    }


# ======================================================
# 4. Training Loop state
# ======================================================

def _trainer_state(trainer):

    return {
        'seed': trainer.seed,

        'config': asdict(
            trainer.config
        ),

        'env_steps': trainer.total_env_steps,

        'stored': trainer.total_stored,

        'skipped': trainer.total_skipped,

        'mode_counts': copy.deepcopy(
            trainer.mode_counts
        ),

        'episodes_log': copy.deepcopy(
            trainer.episodes_log
        ),

        'updates_log': copy.deepcopy(
            trainer.updates_log
        ),

        'rng_state': copy.deepcopy(
            trainer.rng.bit_generator.state
        ),
    }


# ======================================================
# 5. Save checkpoint
# ======================================================

def save_checkpoint(
    path,
    agent,
    buffer,
    trainer
):
    """Save atomically after a completed episode."""

    _completed_boundary(
        agent,
        buffer,
        trainer
    )

    path = Path(path).expanduser()

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {
        'version': VERSION,

        'agent': _agent_state(agent),

        'buffer': _buffer_state(buffer),

        'trainer': _trainer_state(trainer),

        'torch_cpu_rng':
            torch.get_rng_state().clone(),
    }

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode='wb',
            prefix='.sac_tmp_',
            suffix='.pt',
            dir=path.parent,
            delete=False
        ) as stream:

            temp_path = stream.name

            torch.save(
                payload,
                stream
            )

            stream.flush()

            os.fsync(
                stream.fileno()
            )

        # Atomic replacement.
        os.replace(
            temp_path,
            path
        )

    finally:

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):
            os.unlink(temp_path)

    return path


# ======================================================
# 6. Load checkpoint
# ======================================================

def load_checkpoint(
    path,
    agent,
    buffer,
    trainer
):
    """
    Restore a trusted checkpoint into fresh objects.

    SECURITY:
    weights_only=False permits Python unpickling.
    Never load checkpoint files from untrusted sources.
    """

    if agent.device.type != 'cpu':
        raise ValueError(
            'Checkpoint currently supports CPU agents only'
        )

    path = Path(path).expanduser()

    payload = torch.load(
        path,
        map_location='cpu',
        weights_only=False
    )

    if (
        not isinstance(payload, dict)
        or payload.get('version') != VERSION
    ):
        raise ValueError(
            'Unsupported SAC checkpoint format'
        )

    a = payload['agent']
    b = payload['buffer']
    t = payload['trainer']

    # --------------------------------------------------
    # Validate agent configuration
    # --------------------------------------------------

    for name, current in (
        ('obs_dim', agent.obs_dim),
        ('action_dim', agent.action_dim),
        ('gamma', agent.gamma),
        ('tau', agent.tau),
        (
            'auto_alpha',
            agent.automatic_entropy_tuning
        ),
        (
            'target_entropy',
            agent.target_entropy
        ),
    ):

        if a[name] != current:
            raise ValueError(
                f'Agent mismatch: {name}'
            )

    # --------------------------------------------------
    # Validate buffer configuration
    # --------------------------------------------------

    if b['capacity'] != buffer.capacity:
        raise ValueError(
            'Replay capacity mismatch'
        )

    # Allow extending the training episode target

    saved_config = t['config']

    current_config = asdict(
        trainer.config
    )

    # ------------------------------------------------------------
    # 1. Validate configuration structure
    # ------------------------------------------------------------

    if not isinstance(saved_config, dict):

        raise ValueError(
            'Invalid saved trainer configuration'
        )

    if set(saved_config) != set(current_config):

        raise ValueError(
            'Trainer configuration fields mismatch'
        )

    # ------------------------------------------------------------
    # 2. Seed must remain unchanged
    # ------------------------------------------------------------

    if t['seed'] != trainer.seed:

        raise ValueError(
            'Trainer seed mismatch'
        )

    # ------------------------------------------------------------
    # 3. Compare all training parameters except episodes
    # ------------------------------------------------------------

    for name in current_config:

        if name == 'episodes':
            continue

        if saved_config[name] != current_config[name]:

            raise ValueError(
                'Trainer configuration mismatch: '
                f'{name}'
            )

    # ------------------------------------------------------------
    # 4. Validate episode targets
    # ------------------------------------------------------------

    saved_target = saved_config['episodes']

    requested_target = current_config['episodes']

    completed = len(
        t['episodes_log']
    )

    if (
        type(saved_target) is not int
        or saved_target < 1
    ):

        raise ValueError(
            'Invalid saved training episode target'
        )

    if completed > saved_target:

        raise ValueError(
            'Completed episodes exceed saved target'
        )

    if requested_target < completed:

        raise ValueError(
            'Requested episode target is smaller '
            'than completed episodes'
        )

    if requested_target < saved_target:

        raise ValueError(
            'Reducing the training episode target '
            'is not supported'
        )

    # ------------------------------------------------------------
    # 5. Configuration is compatible
    #
    # The newly constructed trainer already contains
    # the requested target in trainer.config.
    #
    # Do not overwrite trainer.config with saved_config.
    # ------------------------------------------------------------

    n = b['size']

    if (
        not isinstance(n, int)
        or not 0 <= n <= buffer.capacity
        or b['total_added'] < n
        or b['position'] != (
            b['total_added'] % buffer.capacity
        )
        or n != min(
            buffer.capacity,
            b['total_added']
        )
    ):
        raise ValueError(
            'Invalid Replay Buffer counters'
        )

    if (
        len(b['phases']) != n
        or len(b['policy_reasons']) != n
    ):
        raise ValueError(
            'Invalid Replay Buffer metadata'
        )

    # Validate arrays before restoration.
    for name in BUFFER_ARRAYS:

        value = b['arrays'][name]

        target = getattr(
            buffer,
            name
        )

        expected_shape = (
            n,
            *target.shape[1:]
        )

        if (
            value.shape != expected_shape
            or value.dtype != target.dtype
        ):
            raise ValueError(
                f'Invalid Replay Buffer array: {name}'
            )

    # --------------------------------------------------
    # Validate training counters
    # --------------------------------------------------

    completed_steps = sum(
        row['steps']
        for row in t['episodes_log']
    )

    if t['env_steps'] != completed_steps:
        raise ValueError(
            'Checkpoint is not at episode boundary'
        )

    if t['stored'] != b['total_added']:
        raise ValueError(
            'Trainer and Replay counters mismatch'
        )

    if len(t['updates_log']) != a['update_count']:
        raise ValueError(
            'Trainer and SAC update counts mismatch'
        )

    if (
        a['auto_alpha']
        and a['alpha_optimizer'] is None
    ):
        raise ValueError(
            'Missing entropy optimizer'
        )

    # ==================================================
    # Restore SAC Agent
    # ==================================================

    agent.actor.load_state_dict(
        a['actor']
    )

    agent.critic.load_state_dict(
        a['critic']
    )

    agent.critic_target.load_state_dict(
        a['critic_target']
    )

    agent.critic_target.requires_grad_(
        False
    )

    agent.critic_target.eval()

    agent.log_alpha.data.copy_(
        a['log_alpha']
    )

    agent.actor_optimizer.load_state_dict(
        a['actor_optimizer']
    )

    agent.critic_optimizer.load_state_dict(
        a['critic_optimizer']
    )

    if agent.alpha_optimizer is not None:

        agent.alpha_optimizer.load_state_dict(
            a['alpha_optimizer']
        )

    agent.update_count = int(
        a['update_count']
    )

    # ==================================================
    # Restore Replay Buffer
    # ==================================================

    for name in BUFFER_ARRAYS:

        getattr(buffer, name)[:n] = (
            b['arrays'][name]
        )

    buffer.phases[:] = (
        list(b['phases'])
        + [None] * (buffer.capacity - n)
    )

    buffer.policy_reasons[:] = (
        list(b['policy_reasons'])
        + [None] * (buffer.capacity - n)
    )

    buffer._size = n

    buffer._position = int(
        b['position']
    )

    buffer.total_added = int(
        b['total_added']
    )

    buffer.rng.bit_generator.state = (
        copy.deepcopy(
            b['rng_state']
        )
    )

    # ==================================================
    # Restore Training Loop
    # ==================================================

    trainer.total_env_steps = int(
        t['env_steps']
    )

    trainer.total_stored = int(
        t['stored']
    )

    trainer.total_skipped = int(
        t['skipped']
    )

    trainer.mode_counts = copy.deepcopy(
        t['mode_counts']
    )

    trainer.episodes_log = copy.deepcopy(
        t['episodes_log']
    )

    trainer.updates_log = copy.deepcopy(
        t['updates_log']
    )

    trainer.rng.bit_generator.state = (
        copy.deepcopy(
            t['rng_state']
        )
    )

    # Restore PyTorch stochastic sampling state.
    torch.set_rng_state(
        payload['torch_cpu_rng']
    )

    return {
        'episodes': len(
            trainer.episodes_log
        ),

        'env_steps':
            trainer.total_env_steps,

        'updates':
            agent.update_count,

        'replay_size':
            len(buffer)
    }


# ======================================================
# 7. Export CSV logs
# ======================================================

def export_training_csv(
    folder,
    trainer
):
    """Export current training logs using atomic replacement."""

    folder = Path(folder).expanduser()

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {}

    datasets = (
        (
            'episodes.csv',
            EPISODE_COLUMNS,
            trainer.episodes_log
        ),
        (
            'updates.csv',
            UPDATE_COLUMNS,
            trainer.updates_log
        ),
    )

    for filename, columns, records in datasets:

        target = folder / filename

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                mode='w',
                newline='',
                encoding='utf-8',
                dir=folder,
                prefix='.csv_tmp_',
                delete=False
            ) as stream:

                temp_path = stream.name

                writer = csv.DictWriter(
                    stream,
                    fieldnames=columns,
                    extrasaction='ignore'
                )

                writer.writeheader()

                writer.writerows(
                    records
                )

                stream.flush()

                os.fsync(
                    stream.fileno()
                )

            os.replace(
                temp_path,
                target
            )

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):
                os.unlink(temp_path)

        outputs[filename] = target

    return outputs
