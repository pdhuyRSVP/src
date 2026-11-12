#!/usr/bin/env python3

"""Step 4.2.5a - Replay policy unit tests."""

from rl_mpc_training.utils.replay_policy import (
    decide_replay_policy
)

from rl_mpc_training.utils.reward_function import (
    TERMINAL_REWARDS
)


def make_info(
    phase,
    applied=True,
    reused=False
):

    terminated = phase in (
        'EPISODE_SUCCESS',
        'EPISODE_COLLISION'
    )

    info = {
        'phase': phase,

        'action_applied': applied,

        'terminal_observation_reused': reused,

        'reward_mode': (
            'full'
            if applied and not reused
            else 'event_only'
        ),

        'reward_components': {
            'terminal': TERMINAL_REWARDS[phase]
        }
    }

    if phase != 'EPISODE_RUNNING':

        info['replay_eligible'] = (
            applied and (terminated or not reused)
        )

    return info


def check(
    name,
    phase,
    flags,
    expected,
    applied=True,
    reused=False,
    mode='task_failure'
):

    policy = decide_replay_policy(
        terminated=flags[0],
        truncated=flags[1],

        info=make_info(
            phase,
            applied=applied,
            reused=reused
        ),

        planning_failure_mode=mode
    )

    actual = (
        policy['store_transition'],
        policy['bootstrap_mask'],
        policy['learning_terminated'],
        policy['learning_truncated']
    )

    assert actual == expected, (
        f'{name}: {actual} != {expected}'
    )

    expected_done = (
        phase != 'EPISODE_RUNNING'
    )

    assert (
        policy['episode_done']
        == expected_done
    )

    if not policy['store_transition']:

        assert (
            policy['bootstrap_mask']
            == 0.0
        )

    print(
        f'[PASS] {name}: '
        f'{policy["reason"]}'
    )


def expect_rejection(name, **kwargs):

    try:

        decide_replay_policy(**kwargs)

    except ValueError:

        print(f'[PASS] {name} rejected')

    else:

        raise AssertionError(
            f'Expected rejection: {name}'
        )


def main():

    print('\n[1] Valid replay transitions')

    # --------------------------------------------------
    # RUNNING
    # --------------------------------------------------

    check(
        'RUNNING',
        'EPISODE_RUNNING',
        (False, False),
        (True, 1.0, False, False)
    )

    # --------------------------------------------------
    # SUCCESS
    # --------------------------------------------------

    check(
        'SUCCESS fresh',
        'EPISODE_SUCCESS',
        (True, False),
        (True, 0.0, True, False)
    )

    check(
        'SUCCESS reused',
        'EPISODE_SUCCESS',
        (True, False),
        (True, 0.0, True, False),
        reused=True
    )

    # --------------------------------------------------
    # COLLISION
    # --------------------------------------------------

    check(
        'COLLISION fresh',
        'EPISODE_COLLISION',
        (True, False),
        (True, 0.0, True, False)
    )

    check(
        'COLLISION reused',
        'EPISODE_COLLISION',
        (True, False),
        (True, 0.0, True, False),
        reused=True
    )

    # --------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------

    check(
        'TIMEOUT fresh',
        'EPISODE_TIMEOUT',
        (False, True),
        (True, 1.0, False, True)
    )

    check(
        'TIMEOUT reused',
        'EPISODE_TIMEOUT',
        (False, True),
        (False, 0.0, False, True),
        reused=True
    )

    # --------------------------------------------------
    # PLANNING_FAILURE
    # --------------------------------------------------

    check(
        'PLANNING_FAILURE reused as task failure',
        'EPISODE_PLANNING_FAILURE',
        (False, True),
        (True, 0.0, True, False),
        reused=True
    )

    check(
        'PLANNING_FAILURE fresh as task failure',
        'EPISODE_PLANNING_FAILURE',
        (False, True),
        (True, 0.0, True, False)
    )

    check(
        'PLANNING_FAILURE external interruption',
        'EPISODE_PLANNING_FAILURE',
        (False, True),
        (False, 0.0, False, True),
        reused=True,
        mode='external_interruption'
    )

    # --------------------------------------------------
    # Action not applied
    # --------------------------------------------------

    check(
        'SUCCESS action not applied',
        'EPISODE_SUCCESS',
        (True, False),
        (False, 0.0, True, False),
        applied=False
    )

    check(
        'PLANNING_FAILURE action not applied',
        'EPISODE_PLANNING_FAILURE',
        (False, True),
        (False, 0.0, False, True),
        applied=False,
        reused=True
    )

    # --------------------------------------------------
    # Invalid inputs
    # --------------------------------------------------

    print('\n[2] Invalid replay metadata')

    expect_rejection(
        'Inconsistent terminal flags',
        terminated=False,
        truncated=True,
        info=make_info('EPISODE_SUCCESS')
    )

    expect_rejection(
        'Unknown planning failure mode',
        terminated=True,
        truncated=False,
        info=make_info('EPISODE_SUCCESS'),
        planning_failure_mode='invalid'
    )

    bad_info = make_info('EPISODE_SUCCESS')

    bad_info['reward_components']['terminal'] = -100.0

    expect_rejection(
        'Wrong terminal penalty',
        terminated=True,
        truncated=False,
        info=bad_info
    )

    bad_info = make_info(
        'EPISODE_TIMEOUT',
        reused=True
    )

    bad_info['reward_mode'] = 'full'

    expect_rejection(
        'Reused observation with full reward',
        terminated=False,
        truncated=True,
        info=bad_info
    )

    bad_info = make_info('EPISODE_SUCCESS')

    bad_info['replay_eligible'] = False

    expect_rejection(
        'Inconsistent environment replay flag',
        terminated=True,
        truncated=False,
        info=bad_info
    )

    bad_info = make_info('EPISODE_RUNNING')

    bad_info['action_applied'] = False

    expect_rejection(
        'RUNNING without applied action',
        terminated=False,
        truncated=False,
        info=bad_info
    )

    print(
        '\n[PASS] Replay policy tests completed'
    )


if __name__ == '__main__':
    main()
