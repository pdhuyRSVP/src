#!/usr/bin/env python3
"""Step 4.3e.5c - CPU test for persistent training driver.

No ROS2/Gazebo.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from rl_mpc_training.training.sac_training_loop import (
    TrainingConfig,
)

from rl_mpc_training.training.train_sac_mpc_gazebo import (
    run_training_session,
)

from rl_mpc_training.training.test_sac_training_loop import (
    FakeRLMPCEnv,
)

from rl_mpc_training.training.sac_training_checkpoint import (
    BUFFER_ARRAYS,
)


SEED = 431


# ======================================================
# 1. Fake environments
# ======================================================

class SuccessFakeEnv(FakeRLMPCEnv):

    def reset(self, seed=None):

        # Force SUCCESS at the terminal of each episode.
        self.episode_index = -1

        return super().reset(seed=seed)


class ScheduledFailureEnv(FakeRLMPCEnv):

    def reset(self, seed=None):

        # Episode 0 -> SUCCESS
        # Episode 1 -> COLLISION
        self.episode_index = int(seed) - SEED - 1

        return super().reset(seed=seed)


# ======================================================
# 2. Build test configuration
# ======================================================

def make_test_config():

    return TrainingConfig(
        episodes=5,
        max_episode_steps=12,
        warmup_steps=3,
        initial_random_steps=20,
        learning_starts=20,
        batch_size=8,
        updates_per_step=1,
        planning_failure_mode='external_interruption',
    )


# ======================================================
# 3. Compare final training states
# ======================================================

def compare_agents(a, b):

    assert a.update_count == b.update_count

    assert a.alpha == b.alpha

    for name in (
        'actor',
        'critic',
        'critic_target',
    ):

        state_a = getattr(a, name).state_dict()
        state_b = getattr(b, name).state_dict()

        assert state_a.keys() == state_b.keys()

        for key in state_a:

            torch.testing.assert_close(
                state_a[key],
                state_b[key],
                rtol=0,
                atol=0,
            )


def compare_buffers(a, b):

    assert len(a) == len(b)

    assert a.position == b.position

    assert a.total_added == b.total_added

    for name in BUFFER_ARRAYS:

        np.testing.assert_array_equal(
            getattr(a, name)[:len(a)],
            getattr(b, name)[:len(b)],
        )

    assert a.phases == b.phases

    assert a.policy_reasons == b.policy_reasons

    # Verify future replay sampling is identical.
    sample_a = a.sample(8)
    sample_b = b.sample(8)

    np.testing.assert_array_equal(
        sample_a['indices'],
        sample_b['indices'],
    )


def compare_trainers(a, b):

    assert a.total_env_steps == b.total_env_steps

    assert a.total_stored == b.total_stored

    assert a.total_skipped == b.total_skipped

    assert a.mode_counts == b.mode_counts

    assert a.episodes_log == b.episodes_log

    assert len(a.updates_log) == len(b.updates_log)

    for x, y in zip(a.updates_log, b.updates_log):

        assert x.keys() == y.keys()

        for key in x:

            if x[key] is None:
                assert y[key] is None

            else:
                np.testing.assert_allclose(
                    x[key],
                    y[key],
                    rtol=0,
                    atol=0,
                )


# ======================================================
# 4. Main test
# ======================================================

def main():

    print(
        '\nStep 4.3e.5c - '
        'Persistent SAC Training CPU Test'
    )

    torch.set_num_threads(1)

    config = make_test_config()

    with tempfile.TemporaryDirectory(
        prefix='sac_e5c_'
    ) as directory:

        root = Path(directory)

        # ==============================================
        # A. Continuous training
        # ==============================================

        print(
            '\n[1] Reference: five episodes in one session'
        )

        agent_a, buffer_a, trainer_a = (
            run_training_session(
                run_dir=root / 'reference',
                config=config,
                env_factory=SuccessFakeEnv,
                episodes_this_session=5,
                capacity=128,
                seed=SEED,
            )
        )

        assert len(trainer_a.episodes_log) == 5

        assert trainer_a.total_env_steps == 60

        assert agent_a.update_count > 0

        assert all(
            row['phase'] == 'EPISODE_SUCCESS'
            for row in trainer_a.episodes_log
        )

        print(
            '[PASS] Five episodes, checkpoint '
            'and CSV'
        )

        # ==============================================
        # B. Split training: 2 + 3
        # ==============================================

        print(
            '\n[2] First session: two episodes'
        )

        _, _, first_trainer = (
            run_training_session(
                run_dir=root / 'resumed',
                config=config,
                env_factory=SuccessFakeEnv,
                episodes_this_session=2,
                capacity=128,
                seed=SEED,
            )
        )

        assert len(first_trainer.episodes_log) == 2

        assert (
            root / 'resumed' / 'latest.pt'
        ).is_file()

        print(
            '[PASS] Episode 0 and 1 saved'
        )

        # ==============================================
        # C. Resume
        # ==============================================

        print(
            '\n[3] Resume remaining three episodes'
        )

        agent_b, buffer_b, trainer_b = (
            run_training_session(
                run_dir=root / 'resumed',
                config=config,
                env_factory=SuccessFakeEnv,
                episodes_this_session=3,
                capacity=128,
                seed=SEED,
            )
        )

        np.testing.assert_array_equal(
            [
                row['episode_index']
                for row in trainer_b.episodes_log
            ],
            np.arange(5),
        )

        assert len(trainer_b.episodes_log) == 5

        print(
            '[PASS] Global indices 0..4 '
            'without duplication'
        )

        # ==============================================
        # D. Compare final states
        # ==============================================

        print(
            '\n[4] Compare continuous and resumed results'
        )

        compare_agents(
            agent_a,
            agent_b,
        )

        compare_trainers(
            trainer_a,
            trainer_b,
        )

        compare_buffers(
            buffer_a,
            buffer_b,
        )

        print(
            '[PASS] Networks, counters, '
            'metrics and Replay Buffer match'
        )

        # ==============================================
        # E. CSV persistence
        # ==============================================

        print(
            '\n[5] Verify persistent CSV'
        )

        folder = root / 'resumed' / 'logs'

        for name, expected in (
            ('episodes.csv', 5),
            ('updates.csv', agent_b.update_count),
        ):

            with open(
                folder / name,
                newline='',
                encoding='utf-8',
            ) as stream:

                rows = list(
                    csv.DictReader(stream)
                )

            assert len(rows) == expected

        print(
            '[PASS] Episode and update histories '
            'survive resume'
        )

        # ==============================================
        # F. Already completed
        # ==============================================

        print(
            '\n[6] Completed training must not restart'
        )

        _, _, trainer_done = (
            run_training_session(
                run_dir=root / 'resumed',
                config=config,
                env_factory=SuccessFakeEnv,
                episodes_this_session=2,
                capacity=128,
                seed=SEED,
            )
        )

        assert len(trainer_done.episodes_log) == 5

        print(
            '[PASS] No duplicate episodes'
        )

        # ==============================================
        # G. Failed terminal
        # ==============================================

        print(
            '\n[7] Stop after terminal failure'
        )

        fail_config = TrainingConfig(
            episodes=3,
            max_episode_steps=12,
            warmup_steps=3,
            initial_random_steps=20,
            learning_starts=20,
            batch_size=8,
            updates_per_step=1,
            planning_failure_mode='external_interruption',
        )

        _, _, failure_trainer = (
            run_training_session(
                run_dir=root / 'failure',
                config=fail_config,
                env_factory=ScheduledFailureEnv,
                episodes_this_session=3,
                capacity=128,
                seed=SEED,
            )
        )

        assert len(failure_trainer.episodes_log) == 2

        assert (
            failure_trainer.episodes_log[-1]['phase']
            == 'EPISODE_COLLISION'
        )

        assert (
            root / 'failure' / 'latest.pt'
        ).is_file()

        print(
            '[PASS] Failure checkpoint saved; '
            'third episode not started'
        )

        # ==============================================
        # H. Reject automatic resume after failure
        # ==============================================

        print(
            '\n[8] Block resume after failed terminal'
        )

        try:

            run_training_session(
                run_dir=root / 'failure',
                config=fail_config,
                env_factory=ScheduledFailureEnv,
                episodes_this_session=1,
                capacity=128,
                seed=SEED,
            )

        except RuntimeError as exc:

            assert 'Previous completed episode' in str(exc)

        else:

            raise AssertionError(
                'Failed terminal was resumed automatically'
            )

        print(
            '[PASS] Unsafe automatic resume blocked'
        )

    print(
        '\n[PASS] Persistent SAC Training '
        'CPU tests completed'
    )


if __name__ == '__main__':
    main()
