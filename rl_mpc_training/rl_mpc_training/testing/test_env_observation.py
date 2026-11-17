#!/usr/bin/env python3

"""Step 4.2.2e.2 - Test synchronized Gymnasium observation."""

import numpy as np

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


def main():

    env = RLMPCEnv()

    try:

        print('\nSynchronized Gymnasium Observation Test')

        answer = input(
            '\nType TEST to start one episode: '
        )

        if answer.strip() != 'TEST':

            print('Test cancelled')
            return

        # --------------------------------------------
        # 1. Reset environment
        # --------------------------------------------

        observation, info = env.reset(seed=123)

        print('\n[1] Gymnasium reset completed')

        # --------------------------------------------
        # 2. Observation
        # --------------------------------------------

        print('\n[2] Normalized observation')

        print(observation)

        print('Shape:', observation.shape)
        print('Dtype:', observation.dtype)

        assert observation.shape == (30,)
        assert observation.dtype == np.float32

        assert env.observation_space.contains(observation)

        print('[PASS] Observation space')

        # --------------------------------------------
        # 3. Raw observation
        # --------------------------------------------

        raw = info['raw_observation']

        print('\n[3] Raw observation')

        print(raw)

        assert raw.shape == (30,)

        assert np.isfinite(raw).all()

        # --------------------------------------------
        # 4. Verify MPC defaults
        # --------------------------------------------

        np.testing.assert_allclose(
            raw[25:30],
            [1.0, 1.0, 1.0, 1.0, 0.5],
            rtol=0.0,
            atol=1e-6
        )

        print('[PASS] Current MPC parameters')

        # --------------------------------------------
        # 5. Verify LiDAR masks
        # --------------------------------------------

        masks = raw[12:22]

        assert np.all(
            np.isin(masks, [0.0, 1.0])
        )

        print('[PASS] LiDAR masks')

        # --------------------------------------------
        # 6. Verify synchronization metadata
        # --------------------------------------------

        print('\n[4] Synchronization')

        print(
            'Sensor ages:',
            info['sensor_ages_s']
        )

        print(
            'Sensor skew:',
            info['sensor_skew_s']
        )

        print(
            'TF age:',
            info['tf_age_s']
        )

        print(
            'Message counters:',
            info['observation_counts']
        )

        assert info['sensor_skew_s'] <= 0.3

        assert all(
            -0.1 <= age <= 0.5
            for age in info['sensor_ages_s'].values()
        )

        print('[PASS] Sensor timestamp checks')

        # --------------------------------------------
        # 7. Verify episode
        # --------------------------------------------

        print('\n[5] Episode information')

        print('Phase:', info['phase'])
        print('Attempt:', info['attempt'])

        assert info['phase'] == 'EPISODE_RUNNING'

        print('[PASS] Episode state')

        print(
            '\n[PASS] Synchronized observation test completed'
        )

    finally:

        print('\nClosing environment')
        env.close()


if __name__ == '__main__':
    main()
