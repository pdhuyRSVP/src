#!/usr/bin/env python3
"""Integration test: recover from an initial-path startup timeout.

Run with Gazebo, Nav2, Goal Manager, MPC Controller and Episode Manager active.
Do not start the SAC training loop or manually reset Episode Manager between tests.
"""

import argparse

from rl_mpc_training.env.rl_mpc_env import (
    EpisodeStartupRetryableError,
    RLMPCEnv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test automatic startup recovery across two Gymnasium resets"
    )
    parser.add_argument(
        '--bad-goal', nargs=3, type=float, metavar=('X', 'Y', 'YAW'),
        default=(10.0, 5.0, 0.0),
        help='Goal known to time out in EP_WAIT_PATH (default: 10 5 0)',
    )
    parser.add_argument(
        '--good-goal', nargs=3, type=float, metavar=('X', 'Y', 'YAW'),
        default=(10.0, 0.0, 0.0),
        help='Goal with a previously VERIFIED accepted path',
    )
    parser.add_argument('--seed', type=int, default=431)
    args = parser.parse_args()

    env = RLMPCEnv()
    try:
        initial_status = env._get_phase(env.bridge.episode_status_client)
        print(f'\n[INIT] Episode Manager status: {initial_status}', flush=True)
        if initial_status.get('phase') != 'IDLE':
            raise RuntimeError(
                'Start the test with a freshly started Episode Manager '
                '(phase=IDLE). Do not manually reset the pipeline first.'
            )

        print(f'\n[TEST 1] Bad goal: {args.bad_goal}', flush=True)
        try:
            env.reset(seed=args.seed, options={'goal': tuple(args.bad_goal)})
        except EpisodeStartupRetryableError as exc:
            first = exc.status
            assert first.get('phase') == 'EPISODE_ERROR', first
            assert first.get('startup_recoverable') == 'true', first
            assert first.get('startup_failure_phase') == 'EP_WAIT_PATH', first
            assert first.get('resume_attempts') == '0', first
            print('[PASS] First reset raised EpisodeStartupRetryableError', flush=True)
            print('[PASS] Recovery eligibility confirmed by Episode Manager', flush=True)
            print('[PASS] RLMPCEnv confirmed MPC HOLD before raising', flush=True)
            print(f'[INFO] Failed startup attempt={first.get("attempt")}', flush=True)
        else:
            raise AssertionError(
                'The bad goal unexpectedly started successfully. '
                'Use --bad-goal with one that reliably causes EP_WAIT_PATH timeout.'
            )

        after_failure = env._get_phase(env.bridge.episode_status_client)
        if after_failure.get('phase') != 'EPISODE_ERROR':
            raise AssertionError(f'Unexpected phase after failure: {after_failure}')

        print(f'\n[TEST 2] Verified-good goal: {args.good_goal}', flush=True)
        print('[INFO] No manual Episode Manager reset between tests', flush=True)
        try:
            observation, info = env.reset(
                seed=args.seed,
                options={'goal': tuple(args.good_goal)},
            )
        except EpisodeStartupRetryableError as exc:
            raise RuntimeError(
                'Second goal also failed path startup. Choose a goal with '
                'verified accepted-path availability; this alone does not '
                'prove that the recovery mechanism is broken.'
            ) from exc

        assert info['phase'] == 'EPISODE_RUNNING', info
        assert env.observation_space.contains(observation), observation
        assert int(info['attempt']) > int(first['attempt']), (first, info)
        print('[PASS] Full reset pipeline completed after startup failure', flush=True)
        print('[PASS] New episode reached EPISODE_RUNNING', flush=True)
        print(f'[PASS] Observation shape={observation.shape}', flush=True)
        print(f'[PASS] New startup attempt={info["attempt"]}', flush=True)
        print('\n[PASS] STARTUP RECOVERY INTEGRATION TEST', flush=True)

    finally:
        print('\n[CLEANUP] Closing RLMPCEnv and requesting MPC HOLD', flush=True)
        env.close()


if __name__ == '__main__':
    main()
