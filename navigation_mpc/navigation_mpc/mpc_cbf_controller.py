import math
import time

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64, Bool, Float64MultiArray

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
        self.v_ref = 0.50

        # Input limits
        self.v_min = -0.20
        self.v_max = 0.50

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
            1.0,     # x
            1.0,     # y
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
        # CBF parameters
        # ============================================================

        # Discrete-time CBF parameter
        self.cbf_gamma = 0.5

        # Numerical tolerance used only for CBF diagnostics
        self.cbf_constraint_tol = 1e-6

        # Conservative circular approximation of robot footprint.
        # Replace this value using the actual robot footprint.
        self.robot_radius = 0.40

        # Additional safety margin
        self.cbf_margin = 0.10

        # ------------------------------------------------------------
        # Static circular obstacle used ONLY for the first CBF test.
        #
        # Replace x/y/r with values appropriate for your test world.
        # ------------------------------------------------------------
        self.obstacle_x = 5.0
        self.obstacle_y = 0.0
        self.obstacle_radius = 0.20

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

        # Current barrier value h(x_k)
        self.cbf_h_current_pub = self.create_publisher(
            Float64,
            '/mpc/cbf_h_current',
            10
        )

        # Center-to-center distance robot <-> obstacle
        self.cbf_distance_pub = self.create_publisher(
            Float64,
            '/mpc/cbf_obstacle_distance',
            10
        )

        # Distance to CBF safety boundary
        self.cbf_clearance_pub = self.create_publisher(
            Float64,
            '/mpc/cbf_clearance',
            10
        )

        # Minimum predicted h over MPC horizon
        self.cbf_h_min_pred_pub = self.create_publisher(
            Float64,
            '/mpc/cbf_h_min_pred',
            10
        )

        # Minimum value of actual CBF constraint g returned by solver
        self.cbf_g_min_pub = self.create_publisher(
            Float64,
            '/mpc/cbf_g_min',
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
        # MPC solver
        # ============================================================

        self.setup_mpc()

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

        gx = msg.pose.position.x
        gy = msg.pose.position.y

        gyaw = self.quaternion_to_yaw(msg.pose.orientation)

        self.commanded_goal = np.array([gx, gy, gyaw])

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

        obstacle_offset = (adaptive_offset + 4)

        P = ca.SX.sym(
            'P',
            3 + n_ref + 2 + 4 + 3
        )

        previous_u = P[
            previous_u_offset:
            previous_u_offset + 2
        ]

        alpha_Qxy = P[adaptive_offset]

        alpha_Qyaw = P[adaptive_offset + 1]

        alpha_Sv = P[adaptive_offset + 2]

        alpha_Sw = P[adaptive_offset + 3]

        obstacle_x = P[obstacle_offset]

        obstacle_y = P[obstacle_offset + 1]

        obstacle_radius = P[obstacle_offset + 2]

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
        # CBF CONSTRAINTS
        # ============================================================

        cbf_constraints = []

        # Total minimum center-to-center distance
        d_safe = (self.robot_radius + obstacle_radius + self.cbf_margin)

        for k in range(N):

            state_k = X[k]
            state_next = X[k + 1]
            # --------------------------------------------------------
            # Barrier at current predicted state
            # --------------------------------------------------------
            h_k = (
                (state_k[0] - obstacle_x) ** 2
                +
                (state_k[1] - obstacle_y) ** 2
                -
                d_safe ** 2
            )

            # --------------------------------------------------------
            # Barrier at next predicted state
            # --------------------------------------------------------
            h_next = (
                (state_next[0] - obstacle_x) ** 2
                +
                (state_next[1] - obstacle_y) ** 2
                -
                d_safe ** 2
            )

            # --------------------------------------------------------
            # Discrete Control Barrier Function
            #
            # h(k+1) - (1-gamma) h(k) >= 0
            # --------------------------------------------------------

            cbf_k = (h_next - (1.0 - self.cbf_gamma) * h_k)

            cbf_constraints.append(cbf_k)

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

        g = ca.vertcat(*cbf_constraints)

        OPT_variables = ca.reshape(U, -1, 1)

        nlp_prob = {
            'f': obj,
            'x': OPT_variables,
            'p': P,
            'g': g
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

        self.lbg = np.zeros(N)
        self.ubg = np.full(N,np.inf)

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
        # Current CBF Diagnostics
        # ============================================================

        dx_obs = (self.current_state[0] - self.obstacle_x)
        dy_obs = (self.current_state[1] - self.obstacle_y)
        # center-to-center distance
        obstacle_distance = math.hypot(dx_obs, dy_obs)
        # CBF safe distance
        d_safe = (self.robot_radius + self.obstacle_radius + self.cbf_margin)
        # Current barrier function
        h_current = (obstacle_distance**2 - d_safe**2)
        # Distance to the CBF safety boundary [m]
        cbf_clearance = (obstacle_distance - d_safe)

        h_msg = Float64()
        h_msg.data = float(h_current)
        self.cbf_h_current_pub.publish(h_msg)

        distance_msg = Float64()
        distance_msg.data = float(obstacle_distance)
        self.cbf_distance_pub.publish(distance_msg)

        clearance_msg = Float64()
        clearance_msg.data = float(cbf_clearance)
        self.cbf_clearance_pub.publish(clearance_msg)

        if h_current < -self.cbf_constraint_tol:
            self.get_logger().warn(
                f'Current state is outside CBF safe set: '
                f'h={h_current:.6f}, '
                f'clearance={cbf_clearance:.4f} m'
            )

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
            return

        # Robot is still outside goal position tolerance
        self.goal_reached = False

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

        obstacle_params = np.array([
            self.obstacle_x,
            self.obstacle_y,
            self.obstacle_radius
        ], dtype=float)

        p = np.concatenate((
            self.current_state,
            ref_traj,
            self.u_prev,
            adaptive_params,
            obstacle_params))

        # ============================================================
        # Solve MPC
        # ============================================================

        start_solver = (time.perf_counter())

        try:
            sol = self.solver(
                x0=self.u0,
                p=p,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg
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

            nan_msg = Float64()
            nan_msg.data = float('nan')
            self.cbf_g_min_pub.publish(nan_msg)
            self.cbf_h_min_pred_pub.publish(nan_msg)
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
        # CBF constraint residual returned by optimizer
        # ============================================================
        try:
            g_opt = (sol['g'].full().flatten())
            if g_opt.size > 0:
                min_cbf_g = float(np.min(g_opt))
            else:
                min_cbf_g = float('nan')
        except Exception:
            min_cbf_g = float('nan')

        g_msg = Float64()
        g_msg.data = float(min_cbf_g)
        self.cbf_g_min_pub.publish(g_msg)

        if (math.isfinite(min_cbf_g) and min_cbf_g < -self.cbf_constraint_tol):
            self.get_logger().warn(
                f'CBF constraint violation in solver solution: '
                f'g_min={min_cbf_g:.6e}'
            )

        if not stats.get('success',False):
            self.get_logger().warn(
                f'MPC solver failed: '
                f'status={return_status}, '
                f'g_min={min_cbf_g:.6e}'
            )
            # No valid predicted CBF trajectory is available on failure.
            h_nan = Float64()
            h_nan.data = float('nan')
            self.cbf_h_min_pred_pub.publish(h_nan)
            self.publish_stop()
            return

        # ============================================================
        # Extract optimized control sequence
        # ============================================================

        u_vector = (sol['x'].full().flatten())

        U_opt = (u_vector.reshape((2, self.N), order='F'))

        # ============================================================
        # Predicted CBF Diagnostics
        # ============================================================

        predicted_state = (self.current_state.copy())

        predicted_h_values = []

        for k in range(self.N):

            vk = float(U_opt[0, k])
            wk = float(U_opt[1, k])
            xk = float(predicted_state[0])
            yk = float(predicted_state[1])
            thetak = float(predicted_state[2])

            # Same Forward Euler model used inside MPC
            x_next = (xk + self.dt * vk * math.cos(thetak))
            y_next = (yk + self.dt * vk * math.sin(thetak))
            theta_next = (thetak + self.dt * wk)

            predicted_state = np.array([x_next, y_next, theta_next])

            # Barrier at predicted state
            dx_pred = (x_next - self.obstacle_x)
            dy_pred = (y_next - self.obstacle_y)
            h_pred = (dx_pred**2 + dy_pred**2 - d_safe**2)
            predicted_h_values.append(h_pred)

        if len(predicted_h_values) > 0:
            h_min_pred = float(min(predicted_h_values))
        else:
            h_min_pred = float('nan')

        h_min_msg = Float64()
        h_min_msg.data = float(h_min_pred)
        self.cbf_h_min_pred_pub.publish(h_min_msg)

        if (math.isfinite(h_min_pred) and h_min_pred < -self.cbf_constraint_tol):
            self.get_logger().warn(
                f'Predicted trajectory enters unsafe set: '
                f'h_min_pred={h_min_pred:.6e}'
            )

        v = float(U_opt[0, 0])
        w = float(U_opt[1, 0])
        # Safety saturation
        v = float(np.clip(v, self.v_min, self.v_max))
        w = float(np.clip(w, self.w_min, self.w_max))

        # ============================================================
        # Validate solved CBF residuals BEFORE publishing cmd_vel
        # ============================================================
        if h_current >= -self.cbf_constraint_tol:
            h_pred_lower_limit = -self.cbf_constraint_tol
        else:
            h_pred_lower_limit = ((1.0 - self.cbf_gamma) * h_current - self.cbf_constraint_tol)

        if (
            not math.isfinite(min_cbf_g)
            or min_cbf_g < -self.cbf_constraint_tol
            or not math.isfinite(h_min_pred)
            or h_min_pred < h_pred_lower_limit
        ):
            self.get_logger().error(
                f'Rejected MPC output due to CBF validation: '
                f'h_current={h_current:.6e}, '
                f'g_min={min_cbf_g:.6e}, '
                f'h_min_pred={h_min_pred:.6e}, '
                f'h_pred_lower_limit={h_pred_lower_limit:.6e}'
            )
            self.publish_stop()
            return

        if h_current < -self.cbf_constraint_tol:
            self.get_logger().warn(
                'CBF recovery mode (already outside modeled safe set): '
                f'h_current={h_current:.6e}, '
                f'h_min_pred={h_min_pred:.6e}, '
                f'g_min={min_cbf_g:.6e}'
            )

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
            f'obs_dist={obstacle_distance:.3f}, '
            f'clearance={cbf_clearance:.3f}, '
            f'h={h_current:.4f}, '
            f'h_min_pred={h_min_pred:.4f}, '
            f'g_min={min_cbf_g:.4e}, '
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