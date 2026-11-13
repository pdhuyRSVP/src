#!/usr/bin/env python3

"""Conservative SAC action guard for initial Gazebo tests.

This module restricts SAC proposals to a narrow action envelope
and limits changes between consecutive applied actions.

It does NOT guarantee closed-loop stability or collision avoidance.
"""

import numpy as np

from rl_mpc_training.utils.action_mapping import (
    MPC_PARAM_MIN,
    MPC_PARAM_MAX,
    map_action_to_mpc,
)


# ======================================================
# Previously tested normalized actions
# ======================================================

DEFAULT_ACTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, 1.0],
    dtype=np.float32
)

TEST_ACTION = np.array(
    [-0.9, -0.9, -0.5, -0.5, 0.0],
    dtype=np.float32
)


# Maximum normalized action change per applied step
MAX_DELTA = np.array(
    [0.02, 0.02, 0.06, 0.06, 0.10],
    dtype=np.float32
)


class SACActionGuard:
    """Validate, clip and rate-limit SAC actions."""

    def __init__(self, warmup_steps=5):

        if (
            type(warmup_steps) is not int
            or warmup_steps < 0
        ):
            raise ValueError(
                'warmup_steps must be a nonnegative int'
            )

        self.warmup_steps = warmup_steps

        self.lower = np.minimum(
            DEFAULT_ACTION,
            TEST_ACTION
        )

        self.upper = np.maximum(
            DEFAULT_ACTION,
            TEST_ACTION
        )

        self.reset()

    # ==================================================
    # 1. Reset
    # ==================================================

    def reset(self):
        """Reset after Gymnasium reset completes."""

        self.last_applied = DEFAULT_ACTION.copy()

        self.steps_applied = 0

        self._pending = None

    # ==================================================
    # 2. Validate normalized action
    # ==================================================

    @staticmethod
    def _validate_action(action, name):

        try:

            raw = np.asarray(
                action,
                dtype=np.float64
            )

        except (ValueError, TypeError) as exc:

            raise ValueError(
                f'{name} must be numeric'
            ) from exc

        if raw.shape != (5,):

            raise ValueError(
                f'{name} must have shape (5,)'
            )

        if not np.isfinite(raw).all():

            raise ValueError(
                f'{name} must be finite'
            )

        if (
            np.any(raw < -1.0)
            or np.any(raw > 1.0)
        ):

            raise ValueError(
                f'{name} must be normalized to [-1, 1]'
            )

        return raw.astype(np.float32)

    # ==================================================
    # 3. Generate guarded action
    # ==================================================

    def propose(self, raw_action, step_index):
        """Prepare action without changing last_applied."""

        if self._pending is not None:

            raise RuntimeError(
                'confirm() the pending action first'
            )

        if (
            type(step_index) is not int
            or step_index != self.steps_applied + 1
        ):

            raise ValueError(
                'step_index must be the next unconfirmed step'
            )

        raw = self._validate_action(
            raw_action,
            'raw_action'
        )

        # ----------------------------------------------
        # Warmup
        # ----------------------------------------------

        if step_index <= self.warmup_steps:

            applied = DEFAULT_ACTION.copy()

        else:

            # ------------------------------------------
            # Clip SAC proposal to test envelope
            # ------------------------------------------

            confined = np.clip(
                raw,
                self.lower,
                self.upper
            )

            # ------------------------------------------
            # Limit action changes
            # ------------------------------------------

            delta = np.clip(
                confined - self.last_applied,
                -MAX_DELTA,
                MAX_DELTA
            )

            applied = np.clip(
                self.last_applied + delta,
                self.lower,
                self.upper
            ).astype(np.float32)

        # ----------------------------------------------
        # Verify resulting MPC parameter bounds
        # ----------------------------------------------

        params = np.asarray(
            map_action_to_mpc(applied),
            dtype=np.float64
        )

        if not np.isfinite(params).all():

            raise ValueError(
                'mapped MPC parameters are not finite'
            )

        if (
            np.any(params < MPC_PARAM_MIN - 1e-8)
            or np.any(params > MPC_PARAM_MAX + 1e-8)
        ):

            raise ValueError(
                'mapped MPC parameters outside global bounds'
            )

        # Do not update last_applied yet.
        self._pending = applied.copy()

        return applied.copy()

    # ==================================================
    # 4. Confirm actual application
    # ==================================================

    def confirm(self, applied_action, *, action_applied):
        """Commit only after MPC reports action applied."""

        if self._pending is None:

            raise RuntimeError(
                'no pending action to confirm'
            )

        if type(action_applied) is not bool:

            raise ValueError(
                'action_applied must be bool'
            )

        action = self._validate_action(
            applied_action,
            'applied_action'
        )

        if not np.array_equal(
            action,
            self._pending
        ):

            raise ValueError(
                'confirmed action differs from proposed action'
            )

        if action_applied:

            self.last_applied = (
                self._pending.copy()
            )

            self.steps_applied += 1

        self._pending = None
