import math
import time
import json

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_srvs.srv import SetBool, Trigger

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64, Bool, Float64MultiArray, String

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException
)

class MPCController(Node):

    def __init__(self):

        super().__init__('mpc_controller')

        # ============================================================
        # MPC parameters
        # ============================================================

        self.N = 20
        self.dt = 0.1

        # Reference speed used to time-parameterize global path
        self.v_ref = 0.5

        # Input limits
        self.v_min = -0.20
        self.v_max = 1.0

        self.w_min = -1.0
        self.w_max = 1.0

        # Goal condition
        self.goal_tolerance = 0.15

        # Final yaw condition
        self.goal_yaw_tolerance = 0.10

        # Final in-place rotation
        self.final_yaw_kp = 1.5
        self.final_w_max = 0.60          # rad/s

        # ============================================================
        # MPC weights
        # ============================================================

        # Stage state tracking
        self.Q_values = np.array([
            10.0,    # x
            10.0,    # y
            1.0       # yaw
        ], dtype=float)

        # Control effort
        self.R_values = np.array([
            0.50,     # v
            0.20      # w
        ], dtype=float)

        # Control variation
        self.S_values = np.array([
            0.1,      # delta v
            0.1       # delta w
        ], dtype=float)

        # Terminal cost
        self.Qf_values = np.array([
            120.0,    # terminal x
            120.0,    # terminal y
            20.0      # terminal yaw
        ], dtype=float)

        # ============================================================
        # Online adaptive MPC parameters
        # ============================================================
        self.alpha_Qxy = 1.0
        self.alpha_Qyaw = 1.0
        self.alpha_Sv = 1.0
        self.alpha_Sw = 1.0

        # ============================================================
        # RL Training - Default adaptive parameters
        # ============================================================

        self.training_default_params = (
            float(self.alpha_Qxy),
            float(self.alpha_Qyaw),
            float(self.alpha_Sv),
            float(self.alpha_Sw),
            float(self.v_ref)
        )

        # ============================================================
        # Subscriber
        # ============================================================

        self.path_sub = self.create_subscription(
            Path,
            '/plan_smoothed',
            self.path_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )

        # Test update parameter MPC online
        self.adaptive_param_sub = self.create_subscription(
            Float64MultiArray,
            '/mpc/adaptive_params',
            self.adaptive_param_callback,
            10
        )

        # RL Training - Accepted path from Goal Manager
        self.training_path_sub = self.create_subscription(
            Path,
            '/rl_mpc/accepted_plan',
            self.accepted_plan_callback,
            10
        )

        # ============================================================
        # Publisher
        # ============================================================

        self.cmd_pub = self.create_publisher(
            Twist,
            '/diff_drive_controller/cmd_vel_unstamped',
            10
        )

        self.reference_pub = self.create_publisher(
            PoseStamped,
            '/mpc/reference_pose',
            10
        )

        self.solver_time_pub = self.create_publisher(
            Float64,
            '/mpc/solver_time',
            10
        )

        self.solver_success_pub = self.create_publisher(
            Bool,
            '/mpc/solver_success',
            10
        )

        self.adaptive_param_pub = self.create_publisher(
            Float64MultiArray,
            '/mpc/current_adaptive_params',
            10
        )

        # RL Training - Episode event publisher
        self.training_event_pub = self.create_publisher(
            String,
            '/mpc/training_event',
            10
        )

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ============================================================
        # States
        # ============================================================

        self.path = None

        self.current_state = np.zeros(3)

        self.state_available = False

        self.goal_reached = False

        self.active_goal = None

        self.goal_start_time = None

        self.commanded_goal = None

        # Previous applied control
        self.u_prev = np.zeros(2)

        # Warm start
        self.u0 = np.zeros((2 * self.N, 1))

        # ============================================================
        # RL Training - Internal state
        # ============================================================

        self.training_hold = False

        # True after a training reset.
        # MPC must wait for a new episode before resuming.
        self.training_reset_pending = False

        self.training_episode_phase = 'NORMAL'

        self.training_episode_id = 0

        self.training_expected_goal = None

        self.training_goal_receive_ns = None

        # False: normal MPC or episode preparation
        # True : MPC is executing a training episode
        self.training_episode_active = False

        # Automatically resume only after a hold caused by
        # an invalid global path.
        self.training_auto_resume_after_replan = False

        # Timestamp of the latest path invalidation
        self.training_last_invalid_ns = 0

        # Timestamp of the last accepted path, used to reject stale messages
        self.training_last_accepted_ns = 0

        self.training_goal_event_sent = False

        self.training_goal_stamp_ns = 0

        # ============================================================
        # MPC solver
        # ============================================================

        self.setup_mpc()

        # ============================================================
        # RL Training - Services
        # ============================================================

        # HOLD / RESUME Service
        self.training_hold_srv = self.create_service(
            SetBool,
            '/mpc/set_training_hold',
            self.training_hold_callback
        )

        # Reset MPC state
        self.training_reset_srv = self.create_service(
            Trigger,
            '/mpc/reset_training_state',
            self.reset_training_state_callback
        )

        # Prepare new episode
        self.training_prepare_srv = self.create_service(
            Trigger,
            '/mpc/prepare_training_episode',
            self.prepare_training_episode_callback
        )

        # ============================================================
        # Control timer
        # ============================================================

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('MPC Controller Started')

    # ================================================================
    # Adaptive parameters callback
    # ================================================================
    def adaptive_param_callback(self, msg):

        if len(msg.data) != 5:
            self.get_logger().warn('Adaptive MPC parameter message must contain 5 values')
            return

        alpha_Qxy = float(msg.data[0])
        alpha_Qyaw = float(msg.data[1])
        alpha_Sv = float(msg.data[2])
        alpha_Sw = float(msg.data[3])
        v_ref = float(msg.data[4])

        # --------------------------------------------------------
        # Basic safety validation
        # --------------------------------------------------------

        if not np.all(np.isfinite([
            alpha_Qxy, alpha_Qyaw, alpha_Sv, alpha_Sw, v_ref
        ])):
            self.get_logger().warn('Rejected non-finite adaptive MPC parameter')
            return

        if(alpha_Qxy <= 0.0 or alpha_Qyaw <= 0.0 or alpha_Sv <= 0.0 or alpha_Sw <= 0.0):
            self.get_logger().warn(
                'Rejected adaptive MPC parameters: '
                'all weight multipliers must be > 0'
            )
            return

        if(v_ref <= 0.0 or v_ref > self.v_max):
            self.get_logger().warn(
                f'Rejected v_ref={v_ref:.3f}; '
                f'valid range is (0, {self.v_max:.3f}]'
            )
            return

        # --------------------------------------------------------
        # Apply new values
        # --------------------------------------------------------

        self.alpha_Qxy = alpha_Qxy
        self.alpha_Qyaw = alpha_Qyaw
        self.alpha_Sv = alpha_Sv
        self.alpha_Sw = alpha_Sw
        self.v_ref = v_ref

        self.get_logger().info(
            'Update adaptive MPC parameters: '
            f'Qxy={self.alpha_Qxy:.3f}, '
            f'Qyaw={self.alpha_Qyaw:.3f}, '
            f'Sv={self.alpha_Sv:.3f}, '
            f'Sw={self.alpha_Sw:.3f}, '
            f'v_ref={self.v_ref:.3f}'
        )

        self.publish_adaptive_params()

    # ================================================================
    # Publish adaptive parameter
    # ================================================================

    def publish_adaptive_params(self):

        msg = Float64MultiArray()

        msg.data = [
            float(self.alpha_Qxy),
            float(self.alpha_Qyaw),
            float(self.alpha_Sv),
            float(self.alpha_Sw),
            float(self.v_ref)
        ]

        self.adaptive_param_pub.publish(msg)

    # ================================================================
    # Goal pose callback
    # ================================================================

    def goal_pose_callback(self, msg):

        # ------------------------------------------------------------
        # Training episode initialization
        # ------------------------------------------------------------

        if self.training_reset_pending:

            # Only accept a new goal after PREPARE
            if self.training_episode_phase != 'WAIT_GOAL':
                return

            # Verify coordinate frame
            goal_frame = msg.header.frame_id or 'map'

            if goal_frame != 'map':

                self.get_logger().warn(
                    f'RL goal rejected: '
                    f'unsupported frame {goal_frame}'
                )
                return

        # ------------------------------------------------------------
        # Extract goal
        # ------------------------------------------------------------

        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)
        gyaw = self.quaternion_to_yaw(msg.pose.orientation)

        self.commanded_goal = np.array([gx,gy,gyaw])

        # ------------------------------------------------------------
        # Store new training goal
        # ------------------------------------------------------------

        if self.training_reset_pending:

            self.training_expected_goal = (
                self.commanded_goal.copy()
            )

            self.training_goal_receive_ns = (
                self.get_clock().now().nanoseconds
            )

            self.training_goal_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )

            self.training_episode_phase = 'WAIT_PATH'

            self.get_logger().info(
                f'RL Episode {self.training_episode_id}: '
                f'new goal accepted '
                f'x={gx:.2f}, '
                f'y={gy:.2f}, '
                f'yaw={gyaw:.2f}. '
                'Waiting for accepted path.'
            )

            return

        # ------------------------------------------------------------
        # Normal MPC operation
        # ------------------------------------------------------------

        self.get_logger().info(
            f'Received desired waypoint: '
            f'x={gx:.2f}, '
            f'y={gy:.2f}, '
            f'yaw={gyaw:.2f} rad'
        )

    # ================================================================
    # Path callback
    # ================================================================

    def path_callback(self, msg):

        # During episode preparation or execution,
        # paths must come from /rl_mpc/accepted_plan.
        if (
            self.training_reset_pending
            or self.training_episode_active
        ):
            return

        # Existing path handling continues below

        if len(msg.poses) == 0:
            self.path = None
            self.publish_stop()
            return

        final_pose = msg.poses[-1].pose

        new_goal = np.array([final_pose.position.x, final_pose.position.y])

        # Detect whether this is a new waypoint or just replanning
        if (self.active_goal is None
            or np.linalg.norm(new_goal - self.active_goal) > 0.05):

            self.active_goal = new_goal

            self.goal_reached = False

            self.goal_start_time = (self.get_clock().now())

            # Reset optimizer initialization for a new goal
            self.u0[:] = 0.0
            self.u_prev[:] = 0.0

            self.get_logger().info(
                f'New MPC goal: '
                f'x={new_goal[0]:.2f}, '
                f'y={new_goal[1]:.2f}'
            )

        self.path = msg

    # ================================================================
    # RL Training - Accepted path callback
    # ================================================================

    def accepted_plan_callback(self, msg):

        # ------------------------------------------------------------
        # 1. Check training state
        # ------------------------------------------------------------

        preparing = (
            self.training_reset_pending
            and self.training_episode_phase in (
                'WAIT_PATH',
                'READY'
            )
        )

        running = (
            self.training_episode_active
            and self.training_episode_phase in (
                'RUNNING',
                'PAUSED'
            )
        )

        if not (preparing or running):
            return

        # ------------------------------------------------------------
        # 2. Check expected goal
        # ------------------------------------------------------------

        if self.training_expected_goal is None:
            return

        if self.training_goal_receive_ns is None:
            return

        # ------------------------------------------------------------
        # 3. Check path validity
        # ------------------------------------------------------------

        if len(msg.poses) == 0:

            # Reject old invalidation messages
            invalid_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )

            if (
                invalid_stamp_ns == 0
                or invalid_stamp_ns <= self.training_goal_receive_ns
                or invalid_stamp_ns <= self.training_last_accepted_ns
                or invalid_stamp_ns <= self.training_last_invalid_ns
            ):
                return

            # Clear old path
            self.path = None

            # Reset optimizer states
            self.u0[:] = 0.0
            self.u_prev[:] = 0.0

            # --------------------------------------------------------
            # Invalidation while preparing episode
            # --------------------------------------------------------

            if preparing:

                self.training_episode_phase = 'WAIT_PATH'

                self.publish_stop()

                self.get_logger().warn(
                    f'RL Episode {self.training_episode_id}: '
                    'path invalidated during preparation. '
                    'Waiting for a new accepted path.'
                )

                return

            # --------------------------------------------------------
            # Invalidation while running episode
            # --------------------------------------------------------

            if running:

                # Only enable auto-resume if MPC was actively running.
                # Do not overwrite a previous automatic pause.
                if (
                    self.training_episode_phase == 'RUNNING'
                    and not self.training_hold
                ):
                    self.training_auto_resume_after_replan = True

                self.training_last_invalid_ns = invalid_stamp_ns

                self.training_hold = True
                self.training_episode_phase = 'PAUSED'

                self.publish_stop()

                self.get_logger().warn(
                    f'RL Episode {self.training_episode_id}: '
                    'PATH INVALID - Automatic HOLD. '
                    'Waiting for accepted replanned path.'
                )

                return

        path_frame = msg.header.frame_id or 'map'

        if path_frame != 'map':

            self.get_logger().warn(
                f'RL accepted path rejected: '
                f'invalid frame {path_frame}'
            )
            return

        # ------------------------------------------------------------
        # 4. Check path timestamp
        # ------------------------------------------------------------

        path_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

        if (
            path_stamp_ns == 0
            or path_stamp_ns <= self.training_goal_receive_ns
        ):

            self.get_logger().warn(
                'RL accepted path rejected: '
                'path timestamp is older than current goal'
            )
            return

        # ------------------------------------------------------------
        # 5. Check path endpoint against expected goal
        # ------------------------------------------------------------

        final_pose = msg.poses[-1].pose

        goal_x = float(self.training_expected_goal[0])
        goal_y = float(self.training_expected_goal[1])

        endpoint_error = math.hypot(
            final_pose.position.x - goal_x,
            final_pose.position.y - goal_y
        )

        endpoint_tolerance = 0.35

        if endpoint_error > endpoint_tolerance:

            self.get_logger().warn(
                f'RL accepted path rejected: '
                f'endpoint_error={endpoint_error:.3f} m'
            )
            return

        # ------------------------------------------------------------
        # 6. Reject path messages older than the last accepted path
        # ------------------------------------------------------------

        if path_stamp_ns <= self.training_last_accepted_ns:
            self.get_logger().warn(
                'RL accepted path rejected: stale or duplicate path'
            )
            return

        # ------------------------------------------------------------
        # 7. Initialize new episode
        # ------------------------------------------------------------

        if preparing:

            self.path = msg
            self.training_last_accepted_ns = path_stamp_ns

            self.active_goal = np.array([
                goal_x,
                goal_y
            ], dtype=float)

            self.goal_reached = False

            self.goal_start_time = None

            self.u0[:] = 0.0

            self.u_prev[:] = 0.0

            self.training_episode_phase = 'READY'

            self.get_logger().info(
                f'RL Episode {self.training_episode_id}: '
                f'accepted path READY, '
                f'poses={len(msg.poses)}, '
                f'endpoint_error={endpoint_error:.3f} m'
            )

            return

        # ------------------------------------------------------------
        # 8. Update path during a running episode
        # ------------------------------------------------------------

        if running:

            if self.training_episode_phase == 'PAUSED':

                if self.training_auto_resume_after_replan:

                    # Require a path accepted after invalidation
                    if path_stamp_ns <= self.training_last_invalid_ns:

                        self.get_logger().warn(
                            'Auto-Resume rejected: stale accepted path'
                        )
                        return

                    # Check current robot TF
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            'map',
                            'base_footprint',
                            Time()
                        )

                    except TransformException:

                        self.get_logger().warn(
                            'Auto-Resume rejected: TF unavailable'
                        )
                        return

                    robot_x = transform.transform.translation.x
                    robot_y = transform.transform.translation.y

                    first_pose = msg.poses[0].pose

                    start_error = math.hypot(
                        robot_x - first_pose.position.x,
                        robot_y - first_pose.position.y
                    )

                    if start_error > 1.0:

                        self.get_logger().warn(
                            f'Auto-Resume rejected: '
                            f'robot/path start mismatch '
                            f'({start_error:.3f} m)'
                        )
                        return

                    # Commit path ONLY after validation succeeds
                    self.path = msg
                    self.training_last_accepted_ns = path_stamp_ns

                    # Reset optimizer states
                    self.u0[:] = 0.0
                    self.u_prev[:] = 0.0

                    # Resume the current training episode
                    self.training_hold = False
                    self.training_episode_phase = 'RUNNING'

                    self.training_auto_resume_after_replan = False
                    self.training_last_invalid_ns = 0

                    self.get_logger().info(
                        f'RL Episode {self.training_episode_id}: '
                        f'AUTO-RESUMED after replanning, '
                        f'poses={len(msg.poses)}, '
                        f'start_error={start_error:.3f} m'
                    )

                else:

                    # A manually paused episode stays stopped. Reject a
                    # path older than the most recent invalidation.
                    if path_stamp_ns <= self.training_last_invalid_ns:
                        self.get_logger().warn(
                            'Manual HOLD: rejected stale accepted path'
                        )
                        return

                    self.path = msg
                    self.training_last_accepted_ns = path_stamp_ns

                    self.get_logger().info(
                        'New accepted path available. '
                        'Manual HOLD remains active.'
                    )

            else:

                # The episode is running: update only with a validated path.
                self.path = msg
                self.training_last_accepted_ns = path_stamp_ns

                self.get_logger().info(
                    f'RL Episode {self.training_episode_id}: '
                    f'accepted replanned path, '
                    f'poses={len(msg.poses)}'
                )

    # ================================================================
    # Robot pose from TF
    # ================================================================

    def update_current_state(self):

        if self.path is None:
            return False

        target_frame = (
            self.path.header.frame_id
            if self.path.header.frame_id
            else 'map'
        )

        try:
            transform = (
                self.tf_buffer.lookup_transform(
                    target_frame,
                    'base_footprint',
                    Time()
                ))
        except TransformException:
            self.state_available = False
            return False

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = self.quaternion_to_yaw(q)

        self.current_state = np.array([x, y, yaw])

        self.state_available = True

        return True

    # ================================================================
    # Quaternion
    # ================================================================

    def quaternion_to_yaw(self, q):

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        return math.atan2(siny_cosp, cosy_cosp)

    # ================================================================
    # Angle utility
    # ================================================================

    def unwrap_angle_near(self, angle, reference):

        while angle - reference > math.pi:
            angle -= 2.0 * math.pi

        while angle - reference < -math.pi:
            angle += 2.0 * math.pi

        return angle

    # ================================================================
    # MPC setup
    # ================================================================

    def setup_mpc(self):

        N = self.N
        dt = self.dt

        # ============================================================
        # State
        # ============================================================

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')

        states = ca.vertcat(x, y, theta)

        # ============================================================
        # Control
        # ============================================================

        v = ca.SX.sym('v')
        w = ca.SX.sym('w')

        controls = ca.vertcat(v, w)

        # ============================================================
        # Continuous kinematic model
        # ============================================================

        rhs = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            w
        )

        self.f = ca.Function('f', [states, controls], [rhs])

        # ============================================================
        # Optimization variables
        # ============================================================

        U = ca.SX.sym('U', 2, N)

        # ============================================================
        # Parameters
        #
        # P =
        # [
        #   current_state            -> 3
        #   reference trajectory     -> 3*(N+1)
        #   previous control         -> 2
        #   alpha_Qxy                -> 1
        #   alpha_Qyaw               -> 1
        #   alpha_Sv                 -> 1
        #   alpha_Sw                 -> 1
        # ]
        # ============================================================

        n_ref = 3 * (N + 1)

        ref_offset = 3

        previous_u_offset = (ref_offset + n_ref)

        adaptive_offset = (previous_u_offset + 2)

        P = ca.SX.sym(
            'P',
            3 + n_ref + 2 + 4
        )

        previous_u = P[
            previous_u_offset:
            previous_u_offset + 2
        ]

        alpha_Qxy = P[adaptive_offset]

        alpha_Qyaw = P[adaptive_offset + 1]

        alpha_Sv = P[adaptive_offset + 2]

        alpha_Sw = P[adaptive_offset + 3]

        # ============================================================
        # State prediction
        # ============================================================

        X = []

        X.append(P[0:3])

        for k in range(N):
            st = X[k]
            con = U[:, k]
            # Forward Euler
            st_next = (st + dt * self.f(st, con))
            X.append(st_next)

        # ============================================================
        # ============================================================
        # Cost matrices
        # ============================================================

        Q = ca.diag(
            ca.vertcat(
                self.Q_values[0] * alpha_Qxy,
                self.Q_values[1] * alpha_Qxy,
                self.Q_values[2] * alpha_Qyaw
            ))

        R = ca.DM(np.diag(self.R_values))   # fixed parameter

        S = ca.diag(
            ca.vertcat(
                self.S_values[0] * alpha_Sv,
                self.S_values[1] * alpha_Sw
            ))

        Qf = ca.DM(np.diag(self.Qf_values)) # fixed parameter

        # ============================================================
        # Objective function
        # ============================================================

        obj = 0

        for k in range(N):

            st = X[k]
            con = U[:, k]

            ref_start = (ref_offset + 3 * k)

            ref = P[ref_start:ref_start + 3]

            # --------------------------------------------------------
            # State tracking error
            # --------------------------------------------------------

            yaw_error = ca.atan2(
                ca.sin(st[2] - ref[2]),
                ca.cos(st[2] - ref[2]))

            state_error = ca.vertcat(
                st[0] - ref[0],
                st[1] - ref[1],
                yaw_error
            )

            # --------------------------------------------------------
            # Delta control
            # --------------------------------------------------------

            if k == 0:
                delta_u = (con - previous_u)
            else:
                delta_u = (con - U[:, k - 1])

            # --------------------------------------------------------
            # Stage cost
            # --------------------------------------------------------

            obj += ca.mtimes([state_error.T, Q, state_error])

            obj += ca.mtimes([con.T, R, con])

            obj += ca.mtimes([delta_u.T, S, delta_u])

        # ============================================================
        # TERMINAL COST
        # ============================================================

        terminal_state = X[N]

        terminal_ref_start = (ref_offset + 3 * N)

        terminal_ref = P[terminal_ref_start:terminal_ref_start + 3]

        terminal_yaw_error = ca.atan2(
            ca.sin(terminal_state[2] - terminal_ref[2]),
            ca.cos(terminal_state[2] - terminal_ref[2]))

        terminal_error = ca.vertcat(
            terminal_state[0] - terminal_ref[0],
            terminal_state[1] - terminal_ref[1],
            terminal_yaw_error)

        obj += ca.mtimes([terminal_error.T, Qf, terminal_error])

        # ============================================================
        # NLP
        # ============================================================

        OPT_variables = ca.reshape(U, -1, 1)

        nlp_prob = {
            'f': obj,
            'x': OPT_variables,
            'p': P
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,

            'ipopt.max_iter': 60,

            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-5
        }

        self.solver = ca.nlpsol(
            'solver',
            'ipopt',
            nlp_prob,
            opts
        )

        # ============================================================
        # Control bounds
        # ============================================================

        lbx = []
        ubx = []

        for _ in range(N):

            lbx.append(self.v_min)

            ubx.append(self.v_max)

            lbx.append(self.w_min)

            ubx.append(self.w_max)

        self.lbx = lbx
        self.ubx = ubx

    # ================================================================
    # Reference trajectory
    # ================================================================

    def get_reference_trajectory(self):

        if self.path is None:
            return None

        poses = self.path.poses

        if len(poses) == 0:
            return None

        # ============================================================
        # Find nearest path point
        # ============================================================

        nearest_idx = 0
        min_distance = float('inf')

        for i, pose in enumerate(poses):

            px = pose.pose.position.x
            py = pose.pose.position.y

            distance = math.hypot(
                px - self.current_state[0],
                py - self.current_state[1])

            if distance < min_distance:
                min_distance = distance
                nearest_idx = i

        # ============================================================
        # Remaining path coordinates
        # ============================================================

        path_points = []

        for pose in poses[nearest_idx:]:
            path_points.append([pose.pose.position.x, pose.pose.position.y])

        path_points = np.array(path_points)

        # Path contains only one remaining point
        if len(path_points) == 1:

            final_pose = poses[-1]

            final_yaw = (self.quaternion_to_yaw(final_pose.pose.orientation))

            ref = []

            for _ in range(self.N + 1):
                ref.extend([
                    path_points[0, 0],
                    path_points[0, 1],
                    final_yaw])

            return np.array(ref)

        # ============================================================
        # Arc length
        # ============================================================

        s = np.zeros(len(path_points))

        for i in range(1, len(path_points)):

            ds = math.hypot(
                path_points[i, 0] - path_points[i - 1, 0],
                path_points[i, 1] - path_points[i - 1, 1])

            s[i] = (s[i - 1] + ds)

        ref = []

        previous_yaw = (self.current_state[2])

        # ============================================================
        # Generate N+1 reference states
        # ============================================================

        for k in range(self.N + 1):

            target_s = (k * self.v_ref * self.dt)

            # --------------------------------------------------------
            # Horizon exceeds remaining path
            # --------------------------------------------------------

            if target_s >= s[-1]:
                final_pose = poses[-1]
                x_ref = (final_pose.pose.position.x)
                y_ref = (final_pose.pose.position.y)
                yaw_ref = (self.quaternion_to_yaw(final_pose.pose.orientation))
            else:
                index = np.searchsorted(s, target_s, side='right') - 1
                index = max(0, min(index, len(path_points) - 2))

                ds = (s[index + 1] - s[index])

                if ds < 1e-8:
                    ratio = 0.0
                else:
                    ratio = (target_s - s[index]) / ds

                x_ref = (
                    path_points[index, 0]
                    +
                    ratio
                    *
                    (path_points[index + 1, 0] - path_points[index, 0]))

                y_ref = (
                    path_points[index, 1]
                    +
                    ratio
                    *
                    (path_points[index + 1, 1]- path_points[index, 1]))

                dx = (path_points[index + 1, 0] - path_points[index, 0])

                dy = (path_points[index + 1, 1] - path_points[index, 1])

                yaw_ref = math.atan2(dy, dx)

            # Avoid +/- pi reference discontinuity
            yaw_ref = (self.unwrap_angle_near(yaw_ref, previous_yaw))

            previous_yaw = yaw_ref

            ref.extend([x_ref, y_ref, yaw_ref])

        return np.array(ref)

    # ================================================================
    # Stop
    # ================================================================

    def publish_stop(self):

        cmd = Twist()

        self.cmd_pub.publish(cmd)

        self.u_prev[:] = 0.0

    # ================================================================
    # RL Training - Hold/Resume callback
    # ================================================================

    def training_hold_callback(self, request, response):

        requested_hold = bool(request.data)

        # ------------------------------------------------------------
        # 1. Activate HOLD
        # ------------------------------------------------------------

        if requested_hold:

            # Explicit HOLD disables automatic resuming
            self.training_auto_resume_after_replan = False

            self.training_hold = True

            self.publish_stop()

            self.u0[:] = 0.0

            # Pause an active training episode
            if (
                self.training_episode_active
                and self.training_episode_phase == 'RUNNING'
            ):
                self.training_episode_phase = 'PAUSED'

            self.get_logger().info(
                'RL training HOLD activated - MPC stopped'
            )

            response.success = True
            response.message = 'MPC HOLD activated'

            return response

        # ------------------------------------------------------------
        # 2. Resume after training reset
        # ------------------------------------------------------------

        if self.training_reset_pending:

            if self.training_episode_phase != 'READY':

                response.success = False
                response.message = (
                    'MPC RESUME rejected: '
                    'training episode is not READY'
                )

                self.get_logger().warn(response.message)

                return response

            if self.path is None or len(self.path.poses) == 0:

                response.success = False
                response.message = (
                    'MPC RESUME rejected: '
                    'no accepted path available'
                )

                return response

            # --------------------------------------------------------
            # Check robot TF before resuming
            # --------------------------------------------------------

            try:

                transform = self.tf_buffer.lookup_transform(
                    'map',
                    'base_footprint',
                    Time()
                )

            except TransformException:

                response.success = False
                response.message = (
                    'MPC RESUME rejected: '
                    'robot TF unavailable'
                )

                return response

            # --------------------------------------------------------
            # Check robot position against path start
            # --------------------------------------------------------

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

            first_pose = self.path.poses[0].pose

            start_error = math.hypot(
                robot_x - first_pose.position.x,
                robot_y - first_pose.position.y
            )

            if start_error > 1.0:

                response.success = False

                response.message = (
                    'MPC RESUME rejected: '
                    f'robot/path start mismatch '
                    f'({start_error:.3f} m)'
                )

                self.get_logger().warn(response.message)

                return response

            # --------------------------------------------------------
            # Initialize episode execution
            # --------------------------------------------------------

            self.u0[:] = 0.0
            self.u_prev[:] = 0.0

            self.goal_reached = False

            self.goal_start_time = (
                self.get_clock().now()
            )

            self.training_reset_pending = False

            self.training_episode_active = True

            self.training_episode_phase = 'RUNNING'

            self.training_hold = False

            self.get_logger().info(
                f'RL Episode {self.training_episode_id} '
                f'RUNNING - MPC resumed, '
                f'path_start_error={start_error:.3f} m'
            )

            response.success = True

            response.message = (
                f'RL Episode {self.training_episode_id} '
                'resumed successfully'
            )

            return response

        # ------------------------------------------------------------
        # 3. Resume a paused training episode
        # ------------------------------------------------------------

        if self.training_episode_active:

            if self.training_episode_phase not in (
                'PAUSED',
                'RUNNING'
            ):

                response.success = False
                response.message = (
                    'MPC RESUME rejected: invalid training phase'
                )

                return response

            if self.path is None or len(self.path.poses) == 0:

                response.success = False
                response.message = (
                    'MPC RESUME rejected: no path'
                )

                return response

            self.u0[:] = 0.0
            self.u_prev[:] = 0.0

            self.training_episode_phase = 'RUNNING'

            self.training_hold = False

            response.success = True
            response.message = 'RL training episode resumed'

            return response

        # ------------------------------------------------------------
        # 4. Normal MPC Resume
        # ------------------------------------------------------------

        if self.path is None or len(self.path.poses) == 0:

            response.success = False
            response.message = (
                'MPC RESUME rejected: no valid path available'
            )

            return response

        self.u0[:] = 0.0
        self.u_prev[:] = 0.0

        self.training_hold = False

        self.get_logger().info(
            'RL training HOLD released - MPC resumed'
        )

        response.success = True
        response.message = 'MPC resumed'

        return response

    # ================================================================
    # RL Training - Reset MPC internal state
    # ================================================================

    def reset_training_state_callback(self, request, response):

        # ------------------------------------------------------------
        # 1. Reset is only allowed while MPC is in HOLD mode
        # ------------------------------------------------------------

        if not self.training_hold:

            response.success = False
            response.message = (
                'MPC reset rejected: activate HOLD first'
            )

            self.get_logger().warn(response.message)

            return response

        # ------------------------------------------------------------
        # 2. Publish zero velocity
        # ------------------------------------------------------------

        self.publish_stop()

        # ------------------------------------------------------------
        # 3. Reset optimizer states
        # ------------------------------------------------------------

        self.u0[:] = 0.0

        self.u_prev[:] = 0.0

        # ------------------------------------------------------------
        # 4. Reset mission states
        # ------------------------------------------------------------

        self.path = None

        self.active_goal = None

        self.commanded_goal = None

        self.goal_reached = False

        self.goal_start_time = None

        # Require a new TF update in the next episode
        self.state_available = False

        # ------------------------------------------------------------
        # 5. Reset adaptive MPC parameters
        # ------------------------------------------------------------

        (
            self.alpha_Qxy,
            self.alpha_Qyaw,
            self.alpha_Sv,
            self.alpha_Sw,
            self.v_ref
        ) = self.training_default_params

        # Publish the restored adaptive parameters
        self.publish_adaptive_params()

        # ------------------------------------------------------------
        # 6. Keep MPC in HOLD and wait for a new episode
        # ------------------------------------------------------------

        self.training_hold = True

        self.training_reset_pending = True

        self.training_episode_active = False

        # Reset episode initialization state
        self.training_episode_phase = 'WAIT_PREPARE'

        self.training_expected_goal = None

        self.training_goal_receive_ns = None

        self.training_auto_resume_after_replan = False

        self.training_last_invalid_ns = 0
        self.training_last_accepted_ns = 0

        self.training_goal_event_sent = False
        self.training_goal_stamp_ns = 0

        # ------------------------------------------------------------
        # 7. Report result
        # ------------------------------------------------------------

        self.get_logger().info(
            'RL Training MPC state reset completed: '
            'optimizer, mission and adaptive parameters cleared. '
            'Waiting for new episode.'
        )

        response.success = True

        response.message = (
            'MPC state reset completed. '
            'MPC remains in HOLD mode.'
        )

        return response

    # ================================================================
    # RL Training - Prepare new episode
    # ================================================================

    def prepare_training_episode_callback(self, request, response):

        # ------------------------------------------------------------
        # 1. MPC must already be in HOLD
        # ------------------------------------------------------------

        if not self.training_hold:

            response.success = False
            response.message = (
                'Prepare rejected: MPC is not in HOLD'
            )
            return response

        # ------------------------------------------------------------
        # 2. A training reset must have been completed
        # ------------------------------------------------------------

        if not self.training_reset_pending:

            response.success = False
            response.message = (
                'Prepare rejected: reset MPC first'
            )
            return response

        # ------------------------------------------------------------
        # 3. Check episode initialization phase
        # ------------------------------------------------------------

        if self.training_episode_phase != 'WAIT_PREPARE':

            response.success = False
            response.message = (
                'Prepare rejected: invalid episode phase '
                f'{self.training_episode_phase}'
            )
            return response

        # ------------------------------------------------------------
        # 4. Initialize new episode
        # ------------------------------------------------------------

        self.training_episode_id += 1

        self.training_expected_goal = None

        self.training_goal_receive_ns = None

        self.training_episode_phase = 'WAIT_GOAL'

        # MPC must remain stopped
        self.training_hold = True

        self.publish_stop()

        # ------------------------------------------------------------
        # 5. Response
        # ------------------------------------------------------------

        self.get_logger().info(
            f'RL Episode {self.training_episode_id} prepared. '
            'Waiting for new goal.'
        )

        response.success = True

        response.message = (
            f'Episode {self.training_episode_id} prepared. '
            'Waiting for new goal.'
        )

        return response

    # ================================================================
    # Publish reference trajectory of MPC
    # ================================================================

    def publish_reference_pose(self, reference):

        x_ref = float(reference[0])
        y_ref = float(reference[1])
        yaw_ref = float(reference[2])

        msg = PoseStamped()

        msg.header.stamp = (self.get_clock().now().to_msg())

        if (self.path is not None and self.path.header.frame_id):
            msg.header.frame_id = self.path.header.frame_id
        else:
            msg.header.frame_id = 'map'

        msg.pose.position.x = x_ref
        msg.pose.position.y = y_ref
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0

        msg.pose.orientation.z = math.sin(yaw_ref / 2.0)

        msg.pose.orientation.w = math.cos(yaw_ref / 2.0)

        self.reference_pub.publish(msg)

    # ================================================================
    # Control loop
    # ================================================================

    def control_loop(self):

        # ============================================================
        # RL Training - HOLD
        # ============================================================

        if self.training_hold:

            self.publish_stop()

            return

        # ============================================================
        # Normal MPC operation
        # ============================================================

        if self.path is None:
            return

        if len(self.path.poses) == 0:
            return

        # ============================================================
        # Robot state
        # ============================================================

        if not self.update_current_state():
            return

        # ============================================================
        # ============================================================
        # Goal check position + final yaw
        # ============================================================

        final_goal = self.path.poses[-1]

        gx = final_goal.pose.position.x
        gy = final_goal.pose.position.y

        dist_to_goal = math.hypot(
            gx - self.current_state[0],
            gy - self.current_state[1]
        )

        # ------------------------------------------------------------
        # Desired final yaw
        #
        # Prefer yaw received directly from /goal_pose because this is
        # the yaw defined in waypoint_manager.yaml.
        # ------------------------------------------------------------

        use_commanded_goal = False

        if self.commanded_goal is not None:
            goal_match_distance = math.hypot(
                gx - self.commanded_goal[0],
                gy - self.commanded_goal[1])
            # Make sure /goal_pose corresponds to the current path goal
            if goal_match_distance < 0.50:
                use_commanded_goal = True

        if use_commanded_goal:
            final_yaw = float(self.commanded_goal[2])
        else:
            # Fallback to orientation stored in final path pose
            final_yaw = self.quaternion_to_yaw(final_goal.pose.orientation)

        # ------------------------------------------------------------
        # Wrapped yaw error [-pi, pi]
        # ------------------------------------------------------------

        yaw_error = math.atan2(
            math.sin(final_yaw - self.current_state[2]),
            math.cos(final_yaw - self.current_state[2]))

        # ============================================================
        # POSITION REACHED
        # ============================================================

        if dist_to_goal < self.goal_tolerance:

            # Always publish the true final waypoint reference
            self.publish_reference_pose(np.array([gx, gy, final_yaw]))

            # --------------------------------------------------------
            # Position reached, but yaw is NOT reached
            # -> rotate in place
            # --------------------------------------------------------

            if abs(yaw_error) >= self.goal_yaw_tolerance:

                self.goal_reached = False

                w_final = (self.final_yaw_kp * yaw_error)

                w_final = float(np.clip(w_final, -self.final_w_max, self.final_w_max))

                cmd = Twist()

                # No translation during final orientation correction
                cmd.linear.x = 0.0
                cmd.angular.z = w_final

                self.cmd_pub.publish(cmd)

                # Store actual applied command
                self.u_prev = np.array([0.0, w_final])

                return

            # --------------------------------------------------------
            # Position AND yaw reached
            # --------------------------------------------------------

            self.publish_stop()

            self.u0[:] = 0.0

            if not self.goal_reached:

                self.goal_reached = True

                if self.goal_start_time is not None:
                    total_time = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
                    self.get_logger().info(
                        f'GOAL REACHED: '
                        f'd={dist_to_goal:.3f} m, '
                        f'yaw_error={yaw_error:.3f} rad, '
                        f'time={total_time:.2f} s'
                    )
                else:
                    self.get_logger().info(
                        f'GOAL REACHED: '
                        f'd={dist_to_goal:.3f} m, '
                        f'yaw_error={yaw_error:.3f} rad'
                    )
                # ============================================================
                # RL Training - Publish GOAL_REACHED event once
                # ============================================================

                if (
                    self.training_episode_active
                    and self.training_episode_phase == 'RUNNING'
                    and not self.training_goal_event_sent
                ):

                    elapsed_sec = None

                    if self.goal_start_time is not None:

                        elapsed_sec = (
                            self.get_clock().now()
                            - self.goal_start_time
                        ).nanoseconds / 1e9

                    event_data = {
                        'event': 'GOAL_REACHED',
                        'mpc_episode_id': int(self.training_episode_id),
                        'goal_stamp_ns': int(self.training_goal_stamp_ns),
                        'stamp_ns': int(self.get_clock().now().nanoseconds),
                        'distance_m': float(dist_to_goal),
                        'yaw_error_rad': float(yaw_error),
                        'elapsed_sec': elapsed_sec
                    }

                    event_msg = String()

                    event_msg.data = json.dumps(event_data)

                    self.training_event_pub.publish(event_msg)

                    self.training_goal_event_sent = True

                    self.get_logger().info(
                        'RL GOAL_REACHED event published'
                    )
                
            return

        # Robot is still outside goal position tolerance
        self.goal_reached = False

        # ============================================================
        # ============================================================
        # Reference trajectory
        # ============================================================

        ref_traj = (self.get_reference_trajectory())

        if ref_traj is None:
            return

        reference_matrix = ref_traj.reshape((self.N + 1, 3))

        current_reference = reference_matrix[0]

        self.publish_reference_pose(current_reference)

        # ============================================================
        # Solver parameter vector
        # ============================================================

        adaptive_params = np.array([
            self.alpha_Qxy,
            self.alpha_Qyaw,
            self.alpha_Sv,
            self.alpha_Sw
        ], dtype=float)

        p = np.concatenate((
            self.current_state,
            ref_traj,
            self.u_prev,
            adaptive_params))

        # ============================================================
        # Solve MPC
        # ============================================================

        start_solver = (time.perf_counter())

        try:
            sol = self.solver(
                x0=self.u0,
                p=p,
                lbx=self.lbx,
                ubx=self.ubx
            )
        except Exception as exc:
            solve_time = ( time.perf_counter() - start_solver )

            time_msg = Float64()
            time_msg.data = float(solve_time)
            self.solver_time_pub.publish(time_msg)

            success_msg = Bool()
            success_msg.data = False
            self.solver_success_pub.publish(success_msg)

            self.get_logger().error(
                f'MPC solver exception: '
                f'{exc}'
            )

            self.publish_stop()

            return

        solve_time = (time.perf_counter() - start_solver)

        time_msg = Float64()
        time_msg.data = float(solve_time)
        self.solver_time_pub.publish(time_msg)

        stats = self.solver.stats()

        return_status = str(stats.get('return_status', 'unknown'))

        success = bool(stats.get('success',False))
        success_msg = Bool()
        success_msg.data = success
        self.solver_success_pub.publish(success_msg)

        # ============================================================
        # Solver status
        # ============================================================
        if not success:
            self.get_logger().warn(
                f'MPC solver failed: status={return_status}'
            )
            self.publish_stop()
            return

        # ============================================================
        # Extract optimized control sequence
        # ============================================================

        u_vector = (sol['x'].full().flatten())

        U_opt = (u_vector.reshape((2, self.N), order='F'))

        v = float(np.clip(U_opt[0, 0], self.v_min, self.v_max))
        w = float(np.clip(U_opt[1, 0], self.w_min, self.w_max))

        # ============================================================
        # Publish
        # ============================================================

        cmd = Twist()

        cmd.linear.x = v
        cmd.angular.z = w

        self.cmd_pub.publish(cmd)

        # ============================================================
        # Store previous applied control
        # ============================================================

        self.u_prev = np.array([v, w])

        # ============================================================
        # Shift warm start
        # ============================================================

        U_shift = np.hstack((U_opt[:, 1:], U_opt[:, -1:]))

        self.u0 = U_shift.reshape((-1, 1), order='F')

        # ============================================================
        # Optional debug output
        # ============================================================
        self.get_logger().debug(
            f'v={v:.3f}, '
            f'w={w:.3f}, '
            f'd_goal={dist_to_goal:.3f}, '
            f'status={return_status}, '
            f'solve={solve_time * 1000:.1f} ms'
        )

    # ================================================================
    # Shutdown
    # ================================================================

    def destroy_node(self):

        self.publish_stop()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = MPCController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()