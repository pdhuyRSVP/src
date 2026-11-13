#!/usr/bin/env python3

"""Step 4.3e.1 - SAC Training Action Adapter tests.

CPU-only test.
No ROS2, Gazebo or MPC commands.
"""

import numpy as np
import torch

from rl_mpc_training.agents.sac_agent import (
    SACAgent
)

from rl_mpc_training.utils.action_mapping import (
    MPC_PARAM_MIN,
    MPC_PARAM_MAX,
    map_action_to_mpc
)

from rl_mpc_training.utils.sac_action_guard import (
    DEFAULT_ACTION,
    TEST_ACTION,
    MAX_DELTA
)

from rl_mpc_training.utils.sac_training_action_adapter import (
    admissible_action_bounds,
    previous_action_from_observation,
    command_to_mpc_action,
    mpc_action_to_command
)


# ======================================================
# 1. Construct test observation
# ======================================================

def mock_observation(previous_action, old=None):

    """
    Emulate the MPC normalization used by
    observation_builder.py.
    """

    obs = (
        np.zeros(30, dtype=np.float32)
        if old is None
        else old.copy()
    )

    params = np.asarray(
        map_action_to_mpc(previous_action)
    )

    # Linear normalization for Qxy, Qyaw.
    obs[25:27] = (
        (params[:2] - MPC_PARAM_MIN[:2])
        / (
            MPC_PARAM_MAX[:2]
            - MPC_PARAM_MIN[:2]
        )
    )

    # Log normalization for Sv, Sw.
    obs[27:29] = (
        np.log(
            params[2:4] / MPC_PARAM_MIN[2:4]
        )
        / np.log(
            MPC_PARAM_MAX[2:4]
            / MPC_PARAM_MIN[2:4]
        )
    )

    # Reference velocity normalization.
    obs[29] = (
        params[4] / MPC_PARAM_MAX[4]
    )

    assert np.all(
        (obs >= -1.0) & (obs <= 1.0)
    )

    return obs


# ======================================================
# 2. Main test
# ======================================================

def main():

    print(
        '\nStep 4.3e.1 - '
        'SAC Training Action Adapter'
    )

    torch.manual_seed(431)
    torch.set_num_threads(1)

    rng = np.random.default_rng(431)

    # ==================================================
    # Test 1: Previous action reconstruction
    # ==================================================

    print(
        '\n[1] Decode previous MPC action '
        'from 30D observation'
    )

    for action in (
        DEFAULT_ACTION,
        TEST_ACTION
    ):

        observation = mock_observation(
            action
        )

        recovered = previous_action_from_observation(
            observation
        )

        np.testing.assert_allclose(
            recovered,
            action,
            atol=2e-6,
            rtol=0.0
        )

    default_obs = mock_observation(
        DEFAULT_ACTION
    )

    np.testing.assert_allclose(
        default_obs[29],
        1.0,
        atol=1e-7
    )

    test_obs = mock_observation(
        TEST_ACTION
    )

    np.testing.assert_allclose(
        test_obs[29],
        0.7,
        atol=1e-7
    )

    print(
        '[PASS] Previous action reconstructed, '
        'including v_ref normalization'
    )

    # ==================================================
    # Test 2: Action mapping and bounds
    # ==================================================

    print(
        '\n[2] Map SAC commands to '
        'locally feasible MPC actions'
    )

    observation = mock_observation(
        DEFAULT_ACTION
    )

    previous = DEFAULT_ACTION.copy()

    modified_count = 0

    for _ in range(120):

        command = rng.uniform(
            -1.0,
            1.0,
            size=5
        ).astype(np.float32)

        low, high = admissible_action_bounds(
            observation
        )

        applied = command_to_mpc_action(
            observation,
            command
        )

        assert applied.shape == (5,)

        assert applied.dtype == np.float32

        assert np.isfinite(
            applied
        ).all()

        assert np.all(
            applied >= low - 1e-6
        )

        assert np.all(
            applied <= high + 1e-6
        )

        # ----------------------------------------------
        # Rate limit verification
        # ----------------------------------------------

        delta = np.abs(
            applied - previous
        )

        assert np.all(
            delta <= MAX_DELTA + 2e-6
        )

        # ----------------------------------------------
        # Inverse mapping verification
        # ----------------------------------------------

        reconstructed_command = (
            mpc_action_to_command(
                observation,
                applied
            )
        )

        np.testing.assert_allclose(
            reconstructed_command,
            command,
            atol=3e-5,
            rtol=0.0
        )

        if not np.array_equal(
            applied,
            command
        ):

            modified_count += 1

        # ----------------------------------------------
        # Simulate next observation
        # ----------------------------------------------

        observation = mock_observation(
            applied,
            observation
        )

        recovered_previous = (
            previous_action_from_observation(
                observation
            )
        )

        np.testing.assert_allclose(
            recovered_previous,
            applied,
            atol=3e-6,
            rtol=0.0
        )

        previous = applied.copy()

    assert modified_count > 0

    print(
        '[PASS] Bounds, rate limits, '
        'action inverse and state continuity'
    )

    # ==================================================
    # Test 3: SAC Agent integration
    # ==================================================

    print(
        '\n[3] SAC Agent interface'
    )

    agent = SACAgent(
        device='cpu',
        obs_dim=30,
        action_dim=5
    )

    observation = mock_observation(
        DEFAULT_ACTION
    )

    commands = []

    for _ in range(32):

        # ----------------------------------------------
        # SAC command
        # ----------------------------------------------

        command = agent.act(
            observation,
            deterministic=False
        )

        # ----------------------------------------------
        # MPC action after Adapter
        # ----------------------------------------------

        applied = command_to_mpc_action(
            observation,
            command
        )

        assert np.all(
            (command >= -1.0)
            & (command <= 1.0)
        )

        # ----------------------------------------------
        # Verify command can be reconstructed
        # ----------------------------------------------

        recovered_command = (
            mpc_action_to_command(
                observation,
                applied
            )
        )

        np.testing.assert_allclose(
            recovered_command,
            command,
            atol=3e-5,
            rtol=0.0
        )

        commands.append(
            command.copy()
        )

        # Simulate next observation.
        observation = mock_observation(
            applied,
            observation
        )

    command_array = np.stack(
        commands,
        axis=0
    )

    assert (
        np.std(
            command_array,
            axis=0
        ).max() > 0.05
    )

    assert agent.update_count == 0

    print(
        '[PASS] Actor stochastic commands '
        'produce feasible applied actions'
    )

    print(
        '[PASS] No SAC training performed'
    )

    # ==================================================
    # Test 4: Invalid inputs
    # ==================================================

    print(
        '\n[4] Invalid observation/command handling'
    )

    invalid_cases = (
        (
            np.zeros(29),
            np.zeros(5)
        ),

        (
            np.full(30, np.nan),
            np.zeros(5)
        ),

        (
            mock_observation(DEFAULT_ACTION),
            np.zeros(4)
        ),

        (
            mock_observation(DEFAULT_ACTION),
            np.full(5, np.nan)
        ),

        (
            mock_observation(DEFAULT_ACTION),
            np.full(5, 1.1)
        ),
    )

    for observation, command in invalid_cases:

        try:

            command_to_mpc_action(
                observation,
                command
            )

        except ValueError:

            pass

        else:

            raise AssertionError(
                'Invalid input accepted'
            )

    print(
        '[PASS] Invalid inputs rejected'
    )

    print(
        '\n[PASS] SAC training action '
        'adapter tests completed'
    )


if __name__ == '__main__':
    main()
