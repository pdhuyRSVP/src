#!/usr/bin/env python3
"""Step 4.3e.5b - SAC training resume integration test.

CPU only. Compare:
    A) 8 uninterrupted episodes
    B) 3 episodes -> checkpoint -> restore -> 5 episodes

This tests episode indices, replay state, RNG and SAC updates.
No ROS2 or Gazebo.
"""

import csv
import tempfile

from pathlib import Path

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import SACAgent

from rl_mpc_training.training.sac_training_loop import (
    SACTrainingLoop,
    TrainingConfig,
)

from rl_mpc_training.training.sac_training_checkpoint import (
    BUFFER_ARRAYS,
    save_checkpoint,
    load_checkpoint,
    export_training_csv,
)

from rl_mpc_training.training.test_sac_training_loop import (
    FakeRLMPCEnv,
)

from rl_mpc_training.utils.sac_replay_buffer import (
    SACReplayBuffer,
)


SEED = 431
TOTAL_EPISODES = 8
SPLIT_EPISODES = 3


# ======================================================
# 1. Fake environment with global episode identity
# ======================================================

class IndexedFakeRLMPCEnv(FakeRLMPCEnv):
    """
    Fake environment uses reset seed to recover
    the global episode identity.

    This is only for deterministic comparison
    of uninterrupted and resumed training.
    """

    def reset(self, seed=None):

        if seed is None:
            raise ValueError(
                'This test requires an episode seed'
            )

        global_episode = int(seed) - SEED

        if (
            global_episode < 0
            or global_episode >= TOTAL_EPISODES
        ):
            raise ValueError(
                'Invalid global episode index'
            )

        # FakeRLMPCEnv.reset increments episode_index.
        self.episode_index = global_episode - 1

        return super().reset(seed=seed)


# ======================================================
# 2. Build fresh independent training objects
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

    env = IndexedFakeRLMPCEnv(
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


# ======================================================
# 3. Compare neural networks
# ======================================================

def compare_networks(agent_a, agent_b):

    for model_name in (
        'actor',
        'critic',
        'critic_target'
    ):

        model_a = getattr(
            agent_a,
            model_name
        )

        model_b = getattr(
            agent_b,
            model_name
        )

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()

        assert state_a.keys() == state_b.keys()

        for key in state_a:

            torch.testing.assert_close(
                state_a[key],
                state_b[key],
                rtol=0,
                atol=0
            )

    torch.testing.assert_close(
        agent_a.log_alpha,
        agent_b.log_alpha,
        rtol=0,
        atol=0
    )

    assert (
        agent_a.update_count
        == agent_b.update_count
    )


# ======================================================
# 4. Compare Replay Buffers
# ======================================================

def compare_buffers(buffer_a, buffer_b):

    assert (
        buffer_a.capacity
        == buffer_b.capacity
    )

    assert len(buffer_a) == len(buffer_b)

    assert (
        buffer_a.position
        == buffer_b.position
    )

    assert (
        buffer_a.total_added
        == buffer_b.total_added
    )

    n = len(buffer_a)

    for name in BUFFER_ARRAYS:

        np.testing.assert_array_equal(
            getattr(buffer_a, name)[:n],
            getattr(buffer_b, name)[:n]
        )

    assert (
        buffer_a.phases
        == buffer_b.phases
    )

    assert (
        buffer_a.policy_reasons
        == buffer_b.policy_reasons
    )

    # Next random minibatch must also be identical.
    batch_a = buffer_a.sample(16)
    batch_b = buffer_b.sample(16)

    np.testing.assert_array_equal(
        batch_a['indices'],
        batch_b['indices']
    )

    for key in (
        'observations',
        'actions',
        'rewards',
        'next_observations',
        'bootstrap_masks'
    ):

        np.testing.assert_array_equal(
            batch_a[key],
            batch_b[key]
        )


# ======================================================
# 5. Compare training statistics
# ======================================================

def compare_trainers(trainer_a, trainer_b):

    assert (
        trainer_a.total_env_steps
        == trainer_b.total_env_steps
    )

    assert (
        trainer_a.total_stored
        == trainer_b.total_stored
    )

    assert (
        trainer_a.total_skipped
        == trainer_b.total_skipped
    )

    assert (
        trainer_a.mode_counts
        == trainer_b.mode_counts
    )

    assert (
        trainer_a.episodes_log
        == trainer_b.episodes_log
    )

    assert len(
        trainer_a.updates_log
    ) == len(
        trainer_b.updates_log
    )

    for row_a, row_b in zip(
        trainer_a.updates_log,
        trainer_b.updates_log
    ):

        assert row_a.keys() == row_b.keys()

        for key in row_a:

            value_a = row_a[key]
            value_b = row_b[key]

            if value_a is None:
                assert value_b is None

            elif isinstance(
                value_a,
                (float, np.floating)
            ):

                np.testing.assert_allclose(
                    value_a,
                    value_b,
                    rtol=1e-7,
                    atol=1e-8
                )

            else:

                assert value_a == value_b

    # Next training RNG output must match.
    np.testing.assert_array_equal(
        trainer_a.rng.random(8),
        trainer_b.rng.random(8)
    )


# ======================================================
# 6. Main test
# ======================================================

def main():

    print(
        '\nStep 4.3e.5b - '
        'SAC Training Resume (CPU only)'
    )

    torch.set_num_threads(1)

    config = TrainingConfig(
        episodes=TOTAL_EPISODES,

        max_episode_steps=12,
        warmup_steps=5,

        initial_random_steps=48,
        learning_starts=32,

        batch_size=16,
        updates_per_step=1,

        planning_failure_mode='task_failure'
    )

    # ==================================================
    # TEST A: Uninterrupted training
    # ==================================================

    print(
        '\n[1] Reference: 8 uninterrupted episodes'
    )

    torch.manual_seed(SEED)

    agent_a, buffer_a, env_a, trainer_a = (
        build_objects(config)
    )

    try:

        rows_a = trainer_a.run()

        assert len(rows_a) == TOTAL_EPISODES

        assert trainer_a.total_env_steps == 96

        assert trainer_a.total_stored == 95

        assert trainer_a.total_skipped == 1

        assert agent_a.update_count == 48

        reference_torch_rng = (
            torch.get_rng_state().clone()
        )

        print(
            '[PASS] Reference run: '
            '96 interactions, 95 stored, 48 updates'
        )

        # ==================================================
        # TEST B: Split training
        # ==================================================

        print(
            '\n[2] Run only first 3 episodes'
        )

        torch.manual_seed(SEED)

        agent_b, buffer_b, env_b, trainer_b = (
            build_objects(config)
        )

        try:

            partial_rows = trainer_b.run(
                max_new_episodes=SPLIT_EPISODES
            )

            assert len(partial_rows) == 3

            assert (
                trainer_b.total_env_steps
                == 36
            )

            assert (
                trainer_b.total_stored
                == 36
            )

            assert (
                trainer_b.total_skipped
                == 0
            )

            np.testing.assert_array_equal(
                [
                    row['episode_index']
                    for row in partial_rows
                ],
                [0, 1, 2]
            )

            print(
                '[PASS] First session stopped '
                'at completed episode 2'
            )

            # ==================================================
            # 3. Save checkpoint
            # ==================================================

            with tempfile.TemporaryDirectory(
                prefix='sac_e5b_'
            ) as directory:

                folder = Path(directory)

                checkpoint = (
                    folder / 'training.pt'
                )

                print(
                    '\n[3] Save checkpoint '
                    'after episode 2'
                )

                save_checkpoint(
                    checkpoint,
                    agent_b,
                    buffer_b,
                    trainer_b
                )

                assert checkpoint.is_file()

                print(
                    '[PASS] Checkpoint saved '
                    'at episode boundary'
                )

                # ==============================================
                # 4. Create a new training session
                # ==============================================

                print(
                    '\n[4] Recreate Agent, Buffer '
                    'and Training Loop'
                )

                # Deliberately disturb global RNG.
                # Checkpoint load must restore it.
                torch.manual_seed(999)

                (
                    agent_c,
                    buffer_c,
                    env_c,
                    trainer_c
                ) = build_objects(config)

                try:

                    metadata = load_checkpoint(
                        checkpoint,
                        agent_c,
                        buffer_c,
                        trainer_c
                    )

                    assert metadata['episodes'] == 3

                    assert metadata['env_steps'] == 36

                    assert metadata['replay_size'] == 36

                    assert (
                        len(trainer_c.episodes_log)
                        == 3
                    )

                    print(
                        '[PASS] Training state restored; '
                        'next episode index is 3'
                    )

                    # ==========================================
                    # 5. Resume remaining episodes
                    # ==========================================

                    print(
                        '\n[5] Resume episodes 3 through 7'
                    )

                    rows_c = trainer_c.run()

                    assert len(rows_c) == 8

                    np.testing.assert_array_equal(
                        [
                            row['episode_index']
                            for row in rows_c
                        ],
                        np.arange(8)
                    )

                    assert (
                        trainer_c.total_env_steps
                        == 96
                    )

                    assert (
                        trainer_c.total_stored
                        == 95
                    )

                    assert (
                        trainer_c.total_skipped
                        == 1
                    )

                    assert agent_c.update_count == 48

                    resumed_torch_rng = (
                        torch.get_rng_state().clone()
                    )

                    print(
                        '[PASS] Resumed global '
                        'episode indices 3..7'
                    )

                    # ==========================================
                    # 6. Compare with uninterrupted run
                    # ==========================================

                    print(
                        '\n[6] Compare final '
                        'training states'
                    )

                    compare_networks(
                        agent_a,
                        agent_c
                    )

                    compare_trainers(
                        trainer_a,
                        trainer_c
                    )

                    compare_buffers(
                        buffer_a,
                        buffer_c
                    )

                    torch.testing.assert_close(
                        reference_torch_rng,
                        resumed_torch_rng,
                        rtol=0,
                        atol=0
                    )

                    print(
                        '[PASS] Actor/Critic/Alpha '
                        'match uninterrupted training'
                    )

                    print(
                        '[PASS] Replay Buffer, metrics '
                        'and random states match'
                    )

                    # ==========================================
                    # 7. Verify CSV logs
                    # ==========================================

                    print(
                        '\n[7] Export resumed training logs'
                    )

                    outputs = export_training_csv(
                        folder / 'logs',
                        trainer_c
                    )

                    for filename, expected in (
                        ('episodes.csv', 8),
                        ('updates.csv', 48),
                    ):

                        with open(
                            outputs[filename],
                            newline='',
                            encoding='utf-8'
                        ) as stream:

                            records = list(
                                csv.DictReader(stream)
                            )

                        assert len(records) == expected

                    print(
                        '[PASS] Complete episode '
                        'and SAC update CSV histories'
                    )

                    # ==========================================
                    # 8. Repeated run must do nothing
                    # ==========================================

                    print(
                        '\n[8] Verify completed training '
                        'does not restart'
                    )

                    steps_before = (
                        trainer_c.total_env_steps
                    )

                    updates_before = (
                        agent_c.update_count
                    )

                    rows_again = trainer_c.run()

                    assert len(rows_again) == 8

                    assert (
                        trainer_c.total_env_steps
                        == steps_before
                    )

                    assert (
                        agent_c.update_count
                        == updates_before
                    )

                    print(
                        '[PASS] No duplicated episodes '
                        'after target reached'
                    )

                finally:

                    env_c.close()

        finally:

            env_b.close()

    finally:

        env_a.close()

    print(
        '\n[PASS] SAC Training Resume '
        'tests completed'
    )


if __name__ == '__main__':
    main()
