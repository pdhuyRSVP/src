#!/usr/bin/env python3

"""Step 4.2.1 - Gymnasium reset integration test."""

import time

from rl_mpc_training.env.rl_mpc_env import RLMPCEnv


def main():

    env = RLMPCEnv()

    try:

        print('\nGymnasium Environment Reset Test')

        answer = input(
            '\nType RESET to start one episode: '
        )

        if answer.strip() != 'RESET':

            print('Test cancelled')
            return

        observation, info = env.reset(seed=123)

        print('\n[PASS] reset() returned')

        print('Observation:', observation)

        print('Observation shape:', observation.shape)

        print('Info:', info)

        assert env.observation_space.contains(observation)

        print('[PASS] Observation matches declared space')

        # Briefly allow the running state to be observed.
        time.sleep(1.0)

        print('\n[PASS] Gymnasium reset integration test')

    finally:

        print('\nClosing environment')

        env.close()


if __name__ == '__main__':
    main()
