#!/usr/bin/env python3

"""Step 4.2.5b - SAC transition unit tests."""

import numpy as np

from rl_mpc_training.utils.sac_transition import (
    SACTransition,
    build_sac_transition
)

from rl_mpc_training.utils.reward_function import (
    TERMINAL_REWARDS
)


DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0],
    dtype=np.float32
)


def make_case(
    phase='EPISODE_RUNNING',
    reused=False,
    applied=True
):

    observation = np.zeros(
        30,
        dtype=np.float32
    )

    next_observation = np.zeros(
        30,
        dtype=np.float32
    )

    observation[4] = 0.5
    next_observation[4] = 0.4

    terminated = phase in (
        'EPISODE_SUCCESS',
        'EPISODE_COLLISION'
    )

    truncated = phase in (
        'EPISODE_TIMEOUT',
        'EPISODE_PLANNING_FAILURE'
    )

    components = {
        'progress': 0.0,
        'path': 0.0,
        'heading': 0.0,
        'clearance': 0.0,
        'unknown_lidar': 0.0,
        'action_change': 0.0,
        'time': 0.0,
        'terminal': TERMINAL_REWARDS[phase]
    }

    if applied and not reused:

        components['progress'] = 1.0
        components['time'] = -0.02

    reward = sum(
        components.values()
    )

    info = {
        'phase': phase,

        'action_applied': bool(applied),

        'terminal_observation_reused': bool(reused),

        'reward_mode': (
            'full'
            if applied and not reused
            else 'event_only'
        ),

        'reward_components': components
    }

    if phase != 'EPISODE_RUNNING':

        info['replay_eligible'] = bool(
            applied and (terminated or not reused)
        )

    return {
        'observation': observation,

        'action': DEFAULT_ACTION.copy(),

        'reward': reward,

        'next_observation': next_observation,

        'terminated': terminated,
        'truncated': truncated,

        'info': info,

        'episode_index': 3,
        'step_index': 5
    }


def test_case(
    name,
    phase,
    expected_store,
    expected_mask,
    reused=False,
    applied=True,
    mode='task_failure'
):

    inputs = make_case(
        phase,
        reused=reused,
        applied=applied
    )

    transition, policy = build_sac_transition(
        **inputs,
        planning_failure_mode=mode
    )

    assert (
        policy['store_transition']
        == expected_store
    )

    assert (
        policy['bootstrap_mask']
        == expected_mask
    )

    if expected_store:

        assert isinstance(
            transition,
            SACTransition
        )

        assert transition.observation.shape == (30,)
        assert transition.action.shape == (5,)

        assert transition.next_observation.shape == (30,)

        assert transition.observation.dtype == np.float32
        assert transition.action.dtype == np.float32

        assert (
            transition.next_observation.dtype
            == np.float32
        )

        assert (
            transition.bootstrap_mask
            == expected_mask
        )

        assert (
            transition.reward
            == inputs['reward']
        )

        assert (
            transition.phase
            == phase
        )

    else:

        assert transition is None

    print(
        f'[PASS] {name}: '
        f'{policy["reason"]}'
    )


def expect_rejection(name, inputs):

    try:

        build_sac_transition(**inputs)

    except ValueError:

        print(f'[PASS] {name} rejected')

    else:

        raise AssertionError(
            f'Invalid transition accepted: {name}'
        )


def main():

    print('\n[1] Valid SAC transitions')

    test_case(
        'RUNNING',
        'EPISODE_RUNNING',
        True,
        1.0
    )

    test_case(
        'SUCCESS',
        'EPISODE_SUCCESS',
        True,
        0.0
    )

    test_case(
        'COLLISION',
        'EPISODE_COLLISION',
        True,
        0.0
    )

    test_case(
        'TIMEOUT fresh',
        'EPISODE_TIMEOUT',
        True,
        1.0
    )

    test_case(
        'TIMEOUT reused',
        'EPISODE_TIMEOUT',
        False,
        0.0,
        reused=True
    )

    test_case(
        'PLANNING_FAILURE task failure',
        'EPISODE_PLANNING_FAILURE',
        True,
        0.0,
        reused=True
    )

    test_case(
        'PLANNING_FAILURE external interruption',
        'EPISODE_PLANNING_FAILURE',
        False,
        0.0,
        reused=True,
        mode='external_interruption'
    )

    test_case(
        'SUCCESS action not applied',
        'EPISODE_SUCCESS',
        False,
        0.0,
        applied=False
    )

    # --------------------------------------------------
    # 2. Data independence
    # --------------------------------------------------

    print('\n[2] Data independence')

    inputs = make_case()

    transition, policy = build_sac_transition(
        **inputs
    )

    original_obs_value = float(
        transition.observation[4]
    )

    original_action_value = float(
        transition.action[0]
    )

    inputs['observation'][4] = 0.9
    inputs['action'][0] = 0.5

    assert (
        transition.observation[4]
        == original_obs_value
    )

    assert (
        transition.action[0]
        == original_action_value
    )

    assert not transition.observation.flags.writeable
    assert not transition.action.flags.writeable

    print('[PASS] Independent observation/action copies')
    print('[PASS] Stored vectors are read-only')

    # --------------------------------------------------
    # 3. Invalid inputs
    # --------------------------------------------------

    print('\n[3] Invalid transitions')

    inputs = make_case()
    inputs['observation'] = np.zeros(29)

    expect_rejection(
        'Incorrect observation dimension',
        inputs
    )

    inputs = make_case()
    inputs['action'] = np.zeros(4)

    expect_rejection(
        'Incorrect action dimension',
        inputs
    )

    inputs = make_case()
    inputs['action'][0] = 2.0

    expect_rejection(
        'Action outside [-1, 1]',
        inputs
    )

    inputs = make_case()
    inputs['next_observation'][4] = np.nan

    expect_rejection(
        'Non-finite next observation',
        inputs
    )

    inputs = make_case()
    inputs['reward'] = np.inf

    expect_rejection(
        'Non-finite reward',
        inputs
    )

    inputs = make_case()
    inputs['reward'] += 0.5

    expect_rejection(
        'Reward component mismatch',
        inputs
    )

    inputs = make_case()
    inputs['step_index'] = 0

    expect_rejection(
        'Invalid step index',
        inputs
    )

    inputs = make_case()
    inputs['episode_index'] = -1

    expect_rejection(
        'Invalid episode index',
        inputs
    )

    inputs = make_case()
    inputs['info']['reward_components'].pop('time')

    expect_rejection(
        'Missing reward component',
        inputs
    )

    inputs = make_case('EPISODE_COLLISION')
    inputs['truncated'] = True

    expect_rejection(
        'Inconsistent terminal flags',
        inputs
    )

    print(
        '\n[PASS] SAC transition tests completed'
    )


if __name__ == '__main__':
    main()
