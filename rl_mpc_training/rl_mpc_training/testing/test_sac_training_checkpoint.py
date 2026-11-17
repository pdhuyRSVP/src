#!/usr/bin/env python3

"""Step 4.3e.5a - SAC Checkpoint and CSV tests.

CPU only. Reuses the FakeRLMPCEnv from Step 4.3e.3.
"""

import csv
import tempfile

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent

from rl_mpc_training.training.sac_training_loop import (
    SACTrainingLoop,
    TrainingConfig,
)

from rl_mpc_training.training.test_sac_training_loop import (
    FakeRLMPCEnv,
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer,
)

from rl_mpc_training.training.sac_training_checkpoint import (
    BUFFER_ARRAYS,
    save_checkpoint,
    load_checkpoint,
    export_training_csv,
)


SEED = 431


# ======================================================
# 1. Build fresh objects
# ======================================================

def build_objects(config):

    agent = SACAgent(
        obs_dim=30,
        action_dim=5,
        device='cpu'
    )

    buffer = SACReplayBuffer(
        capacity=128,
        seed=SEED
    )

    env = FakeRLMPCEnv(
        episode_length=12
    )

    trainer = SACTrainingLoop(
        env=env,
        agent=agent,
        buffer=buffer,
        config=config,
        seed=SEED
    )

    return agent, buffer, env, trainer


def expect_value_error(callback, label):

    try:
        callback()

    except ValueError:
        print(f'[PASS] {label}')

    else:
        raise AssertionError(label)


# ======================================================
# 2. Main
# ======================================================

def main():

    print(
        '\nStep 4.3e.5a - '
        'SAC Checkpoint and CSV (CPU only)'
    )

    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    config = TrainingConfig(
        episodes=4,

        max_episode_steps=12,

        warmup_steps=3,

        initial_random_steps=24,

        learning_starts=16,

        batch_size=8,

        updates_per_step=1,

        planning_failure_mode='task_failure'
    )

    original, buf, env, trainer = (
        build_objects(config)
    )

    try:

        # ==============================================
        # 1. Train
        # ==============================================

        print(
            '\n[1] Train a short reproducible CPU run'
        )

        rows = trainer.run()

        assert len(rows) == 4

        assert trainer.total_env_steps == 48

        assert trainer.total_stored == 48

        assert original.update_count > 0

        assert (
            len(trainer.updates_log)
            == original.update_count
        )

        print(
            '[PASS] Four completed episodes '
            'and finite SAC updates'
        )

        # ==============================================
        # 2. Temporary checkpoint directory
        # ==============================================

        with tempfile.TemporaryDirectory(
            prefix='sac_e5a_'
        ) as directory:

            checkpoint = (
                Path(directory) / 'training.pt'
            )

            print(
                '\n[2] Save checkpoint and export logs'
            )

            save_checkpoint(
                checkpoint,
                original,
                buf,
                trainer
            )

            assert checkpoint.is_file()

            assert checkpoint.stat().st_size > 0

            outputs = export_training_csv(
                Path(directory) / 'logs',
                trainer
            )

            for filename, expected_count in (
                (
                    'episodes.csv',
                    len(rows)
                ),
                (
                    'updates.csv',
                    original.update_count
                ),
            ):

                with open(
                    outputs[filename],
                    newline='',
                    encoding='utf-8'
                ) as stream:

                    records = list(
                        csv.DictReader(stream)
                    )

                assert len(records) == expected_count

            print(
                '[PASS] Versioned checkpoint '
                'and episode/update CSV'
            )

            # ==========================================
            # 3. Restore
            # ==========================================

            print(
                '\n[3] Restore into independent '
                'fresh objects'
            )

            restored, buf2, env2, trainer2 = (
                build_objects(config)
            )

            try:

                metadata = load_checkpoint(
                    checkpoint,
                    restored,
                    buf2,
                    trainer2
                )

                assert metadata['episodes'] == 4

                assert metadata['env_steps'] == 48

                assert (
                    metadata['updates']
                    == original.update_count
                )

                assert (
                    metadata['replay_size']
                    == len(buf)
                )

                # --------------------------------------
                # Training state
                # --------------------------------------

                assert (
                    trainer2.episodes_log
                    == trainer.episodes_log
                )

                assert (
                    trainer2.updates_log
                    == trainer.updates_log
                )

                assert (
                    trainer2.mode_counts
                    == trainer.mode_counts
                )

                assert (
                    trainer2.total_stored
                    == trainer.total_stored
                )

                assert (
                    trainer2.total_skipped
                    == trainer.total_skipped
                )

                # --------------------------------------
                # Replay Buffer
                # --------------------------------------

                assert buf2.position == buf.position

                assert (
                    buf2.total_added
                    == buf.total_added
                )

                assert buf2.phases == buf.phases

                assert (
                    buf2.policy_reasons
                    == buf.policy_reasons
                )

                for name in BUFFER_ARRAYS:

                    np.testing.assert_array_equal(
                        getattr(buf2, name)[:len(buf2)],
                        getattr(buf, name)[:len(buf)]
                    )

                # --------------------------------------
                # Neural networks
                # --------------------------------------

                for model_name in (
                    'actor',
                    'critic',
                    'critic_target'
                ):

                    p1 = getattr(
                        original,
                        model_name
                    ).state_dict()

                    p2 = getattr(
                        restored,
                        model_name
                    ).state_dict()

                    for name in p1:

                        torch.testing.assert_close(
                            p1[name],
                            p2[name],
                            atol=0,
                            rtol=0
                        )

                assert original.alpha == restored.alpha

                print(
                    '[PASS] Actor, critics, alpha, '
                    'replay ring, counters and logs restored'
                )

                # ======================================
                # 4. RNG and optimizer continuation
                # ======================================

                print(
                    '\n[4] Verify random-generator '
                    'and optimizer continuation'
                )

                cmd1 = trainer.rng.uniform(
                    -1, 1, size=5
                )

                cmd2 = trainer2.rng.uniform(
                    -1, 1, size=5
                )

                np.testing.assert_array_equal(
                    cmd1,
                    cmd2
                )

                # Identical minibatch sampling.
                sample1 = buf.sample(8)
                sample2 = buf2.sample(8)

                for name in (
                    'indices',
                    'observations',
                    'actions',
                    'rewards',
                    'bootstrap_masks',
                    'next_observations'
                ):

                    np.testing.assert_array_equal(
                        sample1[name],
                        sample2[name]
                    )

                # --------------------------------------
                # Reproduce the next SAC update
                # --------------------------------------

                rng_state = (
                    torch.get_rng_state().clone()
                )

                metrics1 = original.update(
                    sample1
                )

                torch.set_rng_state(
                    rng_state
                )

                metrics2 = restored.update(
                    sample2
                )

                for name in metrics1:

                    if metrics1[name] is None:

                        assert metrics2[name] is None

                    else:

                        np.testing.assert_allclose(
                            metrics1[name],
                            metrics2[name],
                            rtol=1e-6,
                            atol=1e-7
                        )

                # --------------------------------------
                # Compare network weights
                # --------------------------------------

                for model_name in (
                    'actor',
                    'critic',
                    'critic_target'
                ):

                    model1 = getattr(
                        original,
                        model_name
                    )

                    model2 = getattr(
                        restored,
                        model_name
                    )

                    for p1, p2 in zip(
                        model1.parameters(),
                        model2.parameters()
                    ):

                        torch.testing.assert_close(
                            p1,
                            p2,
                            rtol=0,
                            atol=0
                        )

                print(
                    '[PASS] Identical next minibatch '
                    'and resumed SAC gradient update'
                )

                # ======================================
                # 5. Invalid checkpoint configuration
                # ======================================

                print(
                    '\n[5] Reject incompatible configurations'
                )

                mismatch_config = replace(
                    config,
                    warmup_steps=4
                )

                (
                    mismatch,
                    mismatch_buffer,
                    mismatch_env,
                    mismatch_trainer
                ) = build_objects(
                    mismatch_config
                )

                try:

                    expect_value_error(
                        lambda: load_checkpoint(
                            checkpoint,
                            mismatch,
                            mismatch_buffer,
                            mismatch_trainer
                        ),
                        'Mismatched training config rejected'
                    )

                finally:

                    mismatch_env.close()

            finally:

                env2.close()

    finally:

        env.close()

    print(
        '\n[PASS] SAC checkpoint and CSV '
        'tests completed'
    )


if __name__ == '__main__':
    main()
