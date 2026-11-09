#!/usr/bin/env python3
"""RL–MPC Episode Manager: checkpoint 3.5.5.

Stages:
  /rl_mpc/reset_control_pipeline : MPC HOLD -> Goal Manager reset -> MPC reset
  /rl_mpc/reset_localization     : Gazebo -> EKF -> AMCL -> verification
  /rl_mpc/start_episode          : Prepare -> publish goal -> accepted path -> Resume
The start procedure is explicit, never launched automatically at node startup.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_srvs.srv import SetBool, Trigger, Empty
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from robot_localization.srv import SetPose
from nav2_msgs.srv import SetInitialPose, ClearEntireCostmap
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path, Odometry
from tf2_ros import Buffer, TransformListener, TransformException


class EpisodeManager(Node):
    def __init__(self):
        super().__init__('rl_episode_manager')

        self.declare_parameter('required_goal_subscribers', 2)
        self.declare_parameter('reset_timeout_sec', 10.0)
        self.declare_parameter('localization_timeout_sec', 20.0)
        self.declare_parameter('entity_name', 'agv')
        self.declare_parameter('reset_x', 0.0)
        self.declare_parameter('reset_y', 0.0)
        self.declare_parameter('reset_yaw', 0.0)
        self.declare_parameter('reset_z', 0.001)
        self.declare_parameter('gazebo_xy_tolerance', 0.02)
        self.declare_parameter('ekf_xy_tolerance', 0.05)
        self.declare_parameter('localization_xy_tolerance', 0.15)
        self.declare_parameter('yaw_tolerance', 0.10)
        self.declare_parameter('velocity_tolerance', 0.02)

        # 3.5.5: fixed goal for the first end-to-end episode test.
        self.declare_parameter('goal_x', 50.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('episode_start_timeout_sec', 30.0)
        self.declare_parameter('resume_retry_sec', 0.75)
        self.declare_parameter('resume_settle_sec', 0.30)
        self.declare_parameter('max_resume_attempts', 8)
        self.declare_parameter('costmap_settle_sec', 2.0)

        self.required_goal_subscribers = int(
            self.get_parameter('required_goal_subscribers').value)
        self.reset_timeout_sec = float(
            self.get_parameter('reset_timeout_sec').value)
        self.localization_timeout_sec = float(
            self.get_parameter('localization_timeout_sec').value)
        self.entity_name = str(self.get_parameter('entity_name').value)
        self.reset_x = float(self.get_parameter('reset_x').value)
        self.reset_y = float(self.get_parameter('reset_y').value)
        self.reset_yaw = float(self.get_parameter('reset_yaw').value)
        self.reset_z = float(self.get_parameter('reset_z').value)
        self.gazebo_xy_tolerance = float(
            self.get_parameter('gazebo_xy_tolerance').value)
        self.ekf_xy_tolerance = float(
            self.get_parameter('ekf_xy_tolerance').value)
        self.localization_xy_tolerance = float(
            self.get_parameter('localization_xy_tolerance').value)
        self.yaw_tolerance = float(self.get_parameter('yaw_tolerance').value)
        self.velocity_tolerance = float(
            self.get_parameter('velocity_tolerance').value)
        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_yaw = float(self.get_parameter('goal_yaw').value)
        self.episode_start_timeout_sec = float(
            self.get_parameter('episode_start_timeout_sec').value)
        self.resume_retry_sec = float(self.get_parameter('resume_retry_sec').value)
        self.resume_settle_sec = float(self.get_parameter('resume_settle_sec').value)
        self.max_resume_attempts = int(
            self.get_parameter('max_resume_attempts').value)
        self.costmap_settle_sec = float(
            self.get_parameter('costmap_settle_sec').value)

        values = (self.reset_x, self.reset_y, self.reset_yaw, self.reset_z,
                  self.gazebo_xy_tolerance, self.ekf_xy_tolerance,
                  self.localization_xy_tolerance, self.yaw_tolerance,
                  self.velocity_tolerance)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('All reset values and tolerances must be finite')
        if any(v <= 0.0 for v in (self.reset_timeout_sec,
                                  self.localization_timeout_sec,
                                  self.gazebo_xy_tolerance,
                                  self.ekf_xy_tolerance,
                                  self.localization_xy_tolerance,
                                  self.yaw_tolerance,
                                  self.velocity_tolerance)):
            raise ValueError('Timeouts and tolerances must be positive')
        if self.required_goal_subscribers < 1:
            raise ValueError('required_goal_subscribers must be >= 1')
        if not all(math.isfinite(v) for v in (
                self.goal_x, self.goal_y, self.goal_yaw,
                self.episode_start_timeout_sec, self.resume_retry_sec,
                self.resume_settle_sec)):
            raise ValueError('Episode goal and start parameters must be finite')
        if any(v <= 0 for v in (
                self.episode_start_timeout_sec, self.resume_retry_sec)):
            raise ValueError('Episode timeout/retry must be positive')
        if self.resume_settle_sec < 0 or self.max_resume_attempts < 1:
            raise ValueError('Invalid resume delay/attempt count')
        if (not math.isfinite(self.costmap_settle_sec)
                or self.costmap_settle_sec < 0.0):
            raise ValueError(
                'costmap_settle_sec must be finite and nonnegative')

        # Existing service clients from step 3.5.2.
        self.mpc_hold_client = self.create_client(
            SetBool, '/mpc/set_training_hold')
        self.mpc_reset_client = self.create_client(
            Trigger, '/mpc/reset_training_state')
        self.mpc_prepare_client = self.create_client(
            Trigger, '/mpc/prepare_training_episode')
        self.goal_reset_client = self.create_client(
            Trigger, '/goal_manager/reset_training_state')
        self.gazebo_get_client = self.create_client(
            GetEntityState, '/get_entity_state')
        self.gazebo_set_client = self.create_client(
            SetEntityState, '/set_entity_state')
        self.ekf_set_client = self.create_client(SetPose, '/set_pose')
        self.amcl_set_client = self.create_client(
            SetInitialPose, '/set_initial_pose')
        self.amcl_nomotion_client = self.create_client(
            Empty, '/request_nomotion_update')
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap,
            '/global_costmap/clear_entirely_global_costmap')

        self.goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10)
        self.accepted_path_sub = self.create_subscription(
            Path, '/rl_mpc/accepted_plan', self.accepted_path_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.episode_id = 0
        self.episode_phase = 'IDLE'
        self.latest_odom = None
        self.latest_amcl = None
        self.latest_accepted_path = None
        self.odom_count = 0
        self.amcl_count = 0
        self.interfaces_ready = False
        self.last_status = None

        self.service_clients = {
            'mpc_hold': self.mpc_hold_client,
            'mpc_reset': self.mpc_reset_client,
            'mpc_prepare': self.mpc_prepare_client,
            'goal_reset': self.goal_reset_client,
            'gazebo_get': self.gazebo_get_client,
            'gazebo_set': self.gazebo_set_client,
            'ekf_set': self.ekf_set_client,
            'amcl_set': self.amcl_set_client,
            'amcl_nomotion': self.amcl_nomotion_client,
            'global_costmap_clear': self.clear_global_costmap_client,
        }

        # 3.5.3: previously validated control pipeline reset.
        self.reset_steps = [
            ('HOLD_MPC', self.mpc_hold_client),
            ('RESET_GOAL_MANAGER', self.goal_reset_client),
            ('RESET_MPC', self.mpc_reset_client),
        ]
        self.reset_step_index = -1
        self.reset_future = None
        self.reset_future_client = None
        self.reset_start_time = None
        self.reset_last_error = ''

        self.reset_sequence_srv = self.create_service(
            Trigger, '/rl_mpc/reset_control_pipeline',
            self.start_reset_callback)
        self.reset_status_srv = self.create_service(
            Trigger, '/rl_mpc/reset_status', self.reset_status_callback)

        # 3.5.4: independent Gazebo -> EKF -> AMCL reset.
        self.loc_future = None
        self.loc_future_client = None
        self.loc_stage_started = 0.0
        self.loc_stage = None
        self.loc_odom_baseline = 0
        self.loc_amcl_baseline = 0
        self.loc_last_get_sent = 0.0
        self.loc_last_error = ''

        self.localization_srv = self.create_service(
            Trigger, '/rl_mpc/reset_localization',
            self.start_localization_callback)

        # 3.5.5: episode startup is explicit; no startup side effects.
        self.episode_future = None
        self.episode_future_client = None
        self.episode_started_monotonic = 0.0
        self.episode_goal_stamp_ns = 0
        self.episode_accepted_path = None
        self.episode_accepted_stamp_ns = 0
        self.episode_last_invalid_stamp_ns = 0
        self.episode_accepted_monotonic = 0.0
        self.episode_next_resume_at = 0.0
        self.episode_resume_attempts = 0
        self.episode_last_error = ''
        self.episode_costmap_clear_done = 0.0

        self.start_episode_srv = self.create_service(
            Trigger, '/rl_mpc/start_episode', self.start_episode_callback)
        self.episode_status_srv = self.create_service(
            Trigger, '/rl_mpc/episode_status', self.episode_status_callback)

        self.status_timer = self.create_timer(2.0, self.check_interfaces)
        self.reset_timer = self.create_timer(0.1, self.process_reset_sequence)
        self.localization_timer = self.create_timer(
            0.1, self.process_localization)
        self.episode_timer = self.create_timer(0.1, self.process_episode_start)

        self.get_logger().info('RL Episode Manager started')
        self.get_logger().info(
            'Mode: MANUAL reset + EXPLICIT start_episode; NO AUTO START')

    # ------------------------------------------------------------
    # Subscriptions and interface diagnostics
    # ------------------------------------------------------------
    def accepted_path_callback(self, msg):
        self.latest_accepted_path = msg

        # Only use paths after this episode's goal has been published.
        # Paths of previous missions and messages arriving during reset
        # cannot start an episode.
        if self.episode_phase not in ('EP_WAIT_PATH', 'EP_WAIT_RESUME'):
            return
        stamp_ns = self.stamp_to_ns(msg.header.stamp)
        if stamp_ns <= self.episode_goal_stamp_ns or stamp_ns == 0:
            return
        if (msg.header.frame_id or 'map') != 'map':
            self.get_logger().warn('Episode path rejected: frame is not map')
            return

        if not msg.poses:
            # An empty accepted_plan is Goal Manager's invalidation signal.
            # Do not resume based on a path invalidated later in time.
            if stamp_ns >= self.episode_accepted_stamp_ns:
                self.episode_last_invalid_stamp_ns = max(
                    self.episode_last_invalid_stamp_ns, stamp_ns)
                self.episode_accepted_path = None
                self.get_logger().warn(
                    'Episode startup: path invalidated; waiting for new path')
            return

        endpoint = msg.poses[-1].pose.position
        endpoint_error = math.hypot(
            endpoint.x - self.goal_x, endpoint.y - self.goal_y)
        if endpoint_error > 0.35:
            self.get_logger().warn(
                f'Episode path rejected: endpoint error={endpoint_error:.3f} m')
            return
        if stamp_ns <= self.episode_last_invalid_stamp_ns:
            return
        if stamp_ns < self.episode_accepted_stamp_ns:
            return

        # This is evidence that Goal Manager accepted the path; it is NOT
        # proof that the MPC has processed it. Resume's service reply is
        # the final readiness handshake.
        self.episode_accepted_path = msg
        self.episode_accepted_stamp_ns = stamp_ns
        self.episode_accepted_monotonic = time.monotonic()
        self.episode_next_resume_at = max(
            self.episode_next_resume_at,
            self.episode_accepted_monotonic + self.resume_settle_sec)
        self.get_logger().info(
            f'Episode accepted path: poses={len(msg.poses)}, '
            f'endpoint_error={endpoint_error:.3f} m; waiting for MPC READY')

    def odom_callback(self, msg):
        self.latest_odom = msg
        self.odom_count += 1

    def amcl_callback(self, msg):
        self.latest_amcl = msg
        self.amcl_count += 1

    def check_interfaces(self):
        missing_services = [name for name, client in
                            self.service_clients.items()
                            if not client.service_is_ready()]
        services_ready = not missing_services
        goal_subscribers = self.goal_pub.get_subscription_count()
        goal_ready = goal_subscribers >= self.required_goal_subscribers
        tf_ready = self.tf_buffer.can_transform(
            'map', 'base_footprint', Time())
        odom_ready = self.latest_odom is not None
        amcl_received = self.latest_amcl is not None
        clock_ready = self.get_clock().now().nanoseconds > 0
        self.interfaces_ready = all((services_ready, goal_ready,
                                     tf_ready, odom_ready, clock_ready))
        status = (services_ready, goal_subscribers, tf_ready,
                  odom_ready, amcl_received, clock_ready)
        if status != self.last_status:
            self.get_logger().info(
                f'Services: {len(self.service_clients)-len(missing_services)}'
                f'/{len(self.service_clients)} | '
                f'Goal subscribers: {goal_subscribers} | '
                f'TF: {tf_ready} | EKF: {odom_ready} | '
                f'AMCL received: {amcl_received} | Clock: {clock_ready}')
            if missing_services:
                self.get_logger().warn(
                    f'Missing services: {missing_services}')
            if self.interfaces_ready:
                self.get_logger().info('ALL REQUIRED INTERFACES READY')
            self.last_status = status

    # ------------------------------------------------------------
    # Step 3.5.3: unchanged call sequence.
    # ------------------------------------------------------------
    def start_reset_callback(self, request, response):
        if self.episode_phase not in (
                'IDLE', 'RESET_READY', 'LOCALIZATION_READY',
                'EPISODE_RUNNING'):
            response.success = False
            response.message = (
                f'Reset rejected: phase={self.episode_phase}')
            return response

        missing = [name for name, client in self.reset_steps
                   if not client.service_is_ready()]
        if missing:
            response.success = False
            response.message = f'Reset services unavailable: {missing}'
            return response

        self.reset_step_index = 0
        self.reset_future = None
        self.reset_future_client = None
        self.reset_last_error = ''
        self.loc_last_error = ''
        self.latest_accepted_path = None
        # Reset may also be explicitly requested to stop an active episode.
        self.episode_accepted_path = None
        self.episode_future = None
        self.episode_future_client = None
        self.get_logger().info('Control pipeline reset requested')
        self.send_reset_step()
        response.success = self.episode_phase != 'RESET_ERROR'
        response.message = (
            'Reset accepted. Wait for RESET_READY confirmation.'
            if response.success else self.reset_last_error)
        return response

    def reset_status_callback(self, request, response):
        response.success = self.episode_phase in (
            'RESET_READY', 'LOCALIZATION_READY')
        error = self.loc_last_error or self.reset_last_error or 'none'
        response.message = f'phase={self.episode_phase}; error={error}'
        return response

    def send_reset_step(self):
        name, client = self.reset_steps[self.reset_step_index]
        if not client.service_is_ready():
            self.fail_reset(f'Service unavailable: {name}')
            return
        request = SetBool.Request() if name == 'HOLD_MPC' \
            else Trigger.Request()
        if name == 'HOLD_MPC':
            request.data = True
        try:
            self.reset_future = client.call_async(request)
        except Exception as exc:
            self.fail_reset(f'{name} request failed: {exc}')
            return
        self.reset_future_client = client
        self.reset_start_time = time.monotonic()
        self.episode_phase = f'WAIT_{name}'
        self.get_logger().info(
            f'[{self.reset_step_index + 1}/3] {name}: request sent')

    def process_reset_sequence(self):
        if self.reset_future is None:
            return
        name, _ = self.reset_steps[self.reset_step_index]
        if not self.reset_future.done():
            elapsed = time.monotonic() - self.reset_start_time
            if elapsed > self.reset_timeout_sec:
                try:
                    self.reset_future_client.remove_pending_request(
                        self.reset_future)
                except Exception:
                    pass
                self.fail_reset(
                    f'{name} timeout after {elapsed:.1f} s. '
                    'Remote service may still execute.')
            return
        try:
            result = self.reset_future.result()
            if result is None or not result.success:
                message = getattr(result, 'message', 'No response')
                self.fail_reset(f'{name} failed: {message}')
                return
        except Exception as exc:
            self.fail_reset(f'{name} exception: {exc}')
            return
        self.get_logger().info(
            f'[{self.reset_step_index + 1}/3] {name}: SUCCESS')
        self.reset_future = None
        self.reset_future_client = None
        self.reset_step_index += 1
        if self.reset_step_index >= len(self.reset_steps):
            self.episode_phase = 'RESET_READY'
            self.get_logger().info(
                'RESET_READY: MPC HOLD active, '
                'Goal Manager reset completed, MPC state reset completed.')
            return
        self.send_reset_step()

    def fail_reset(self, reason):
        self.reset_last_error = reason
        self.episode_phase = 'RESET_ERROR'
        self.reset_future = None
        self.reset_future_client = None
        self.get_logger().error(
            f'CONTROL PIPELINE RESET FAILED: {reason}')
        self.get_logger().error(
            'Sequence stopped. Verify MPC HOLD before restarting '
            'Episode Manager.')

    # ------------------------------------------------------------
    # Step 3.5.4: localization reset requested by service only.
    # ------------------------------------------------------------
    @staticmethod
    def yaw_of(q):
        return math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z))

    @staticmethod
    def yaw_error(actual, desired):
        return math.atan2(math.sin(actual-desired),
                          math.cos(actual-desired))

    @staticmethod
    def fill_pose(pose, x, y, z, yaw):
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)

    @staticmethod
    def initial_covariance():
        cov = [0.0] * 36
        for idx in (0, 7, 14, 21, 28, 35):
            cov[idx] = 0.01
        return cov

    def start_localization_callback(self, request, response):
        # This state was reached only after HOLD+reset succeeded.
        if self.episode_phase != 'RESET_READY':
            response.success = False
            response.message = (
                f'Localization reset rejected: phase={self.episode_phase}; '
                'call /rl_mpc/reset_control_pipeline first')
            return response

        needed = {
            'gazebo_set': self.gazebo_set_client,
            'gazebo_get': self.gazebo_get_client,
            'ekf_set': self.ekf_set_client,
            'amcl_set': self.amcl_set_client,
            'amcl_nomotion': self.amcl_nomotion_client,
        }
        missing = [name for name, client in needed.items()
                   if not client.service_is_ready()]
        if missing or self.get_clock().now().nanoseconds == 0:
            response.success = False
            response.message = (
                f'Localization services missing: {missing}; '
                'or simulation clock unavailable')
            return response

        self.loc_last_error = ''
        self.loc_future = None
        self.loc_future_client = None
        self.get_logger().info(
            'Localization reset requested: '
            f'Gazebo(world), EKF(odom), AMCL(map) -> '
            f'({self.reset_x:.3f}, {self.reset_y:.3f}, '
            f'{self.reset_yaw:.3f} rad). MPC must remain HOLD.')
        self.enter_loc_stage('SET_GAZEBO')
        response.success = self.episode_phase != 'LOCALIZATION_ERROR'
        response.message = (
            'Localization reset accepted; poll /rl_mpc/reset_status'
            if response.success else self.loc_last_error)
        return response

    def enter_loc_stage(self, stage):
        
        self.loc_stage = stage
        self.loc_stage_started = time.monotonic()

        self.loc_future = None
        self.loc_future_client = None

        self.episode_phase = f'LOC_{stage}'

        self.get_logger().info(
            f'Localization: {stage}'
        )

        # Capture AMCL message count BEFORE resetting AMCL
        if stage == 'SET_AMCL':

            self.loc_amcl_baseline = self.amcl_count

            self.get_logger().info(
                f'AMCL baseline recorded: {self.loc_amcl_baseline}'
            )

        if stage == 'VERIFY_GAZEBO':
            self.loc_last_get_sent = 0.0

        if stage == 'VERIFY_EKF':
            self.loc_odom_baseline = self.odom_count

        # Send service request
        if stage in (
            'SET_GAZEBO',
            'SET_EKF',
            'SET_AMCL',
            'AMCL_NOMOTION'
        ):
            self.send_loc_service(stage)

    def send_loc_service(self, stage):
        if stage == 'SET_GAZEBO':
            client = self.gazebo_set_client
            request = SetEntityState.Request()
            state = EntityState()
            state.name = self.entity_name
            state.reference_frame = 'world'
            self.fill_pose(
                state.pose, self.reset_x, self.reset_y,
                self.reset_z, self.reset_yaw)
            request.state = state  # Twist defaults to zero.
        elif stage == 'VERIFY_GAZEBO':
            client = self.gazebo_get_client
            request = GetEntityState.Request()
            request.name = self.entity_name
            request.reference_frame = 'world'
        elif stage == 'SET_EKF':
            client = self.ekf_set_client
            request = SetPose.Request()
            request.pose.header.stamp = self.get_clock().now().to_msg()
            request.pose.header.frame_id = 'odom'
            self.fill_pose(
                request.pose.pose.pose, self.reset_x, self.reset_y,
                0.0, self.reset_yaw)
            request.pose.pose.covariance = self.initial_covariance()
        elif stage == 'SET_AMCL':
            client = self.amcl_set_client
            request = SetInitialPose.Request()
            request.pose.header.stamp = self.get_clock().now().to_msg()
            request.pose.header.frame_id = 'map'
            self.fill_pose(
                request.pose.pose.pose, self.reset_x, self.reset_y,
                0.0, self.reset_yaw)
            request.pose.pose.covariance = self.initial_covariance()
        elif stage == 'AMCL_NOMOTION':
            client = self.amcl_nomotion_client
            request = Empty.Request()
        else:
            self.fail_localization(f'Unknown service stage: {stage}')
            return

        if not client.service_is_ready():
            self.fail_localization(f'{stage}: service unavailable')
            return
        try:
            self.loc_future = client.call_async(request)
            self.loc_future_client = client
        except Exception as exc:
            self.fail_localization(f'{stage}: {exc}')

    def process_localization(self):
        if not self.episode_phase.startswith('LOC_'):
            return
        if (time.monotonic() - self.loc_stage_started >
                self.localization_timeout_sec):
            self.fail_localization(
                f'{self.loc_stage}: timeout; '
                'remote request may still complete')
            return

        stage = self.loc_stage
        if self.loc_future is not None:
            if not self.loc_future.done():
                return
            try:
                result = self.loc_future.result()
                if result is None:
                    raise RuntimeError('empty service response')
                # SetEntityState/GetEntityState have success flags;
                # SetPose/SetInitialPose/Empty have empty responses.
                if hasattr(result, 'success') and not result.success:
                    raise RuntimeError('service success=False')
            except Exception as exc:
                self.fail_localization(f'{stage}: {exc}')
                return
            self.loc_future = None
            self.loc_future_client = None

            if stage == 'SET_GAZEBO':
                self.enter_loc_stage('VERIFY_GAZEBO')
                return
            if stage == 'VERIFY_GAZEBO':
                if self.gazebo_state_is_valid(result.state):
                    self.get_logger().info('Gazebo reset VERIFIED')
                    self.enter_loc_stage('SET_EKF')
                return
            if stage == 'SET_EKF':
                self.enter_loc_stage('VERIFY_EKF')
                return
            if stage == 'SET_AMCL':
                self.enter_loc_stage('AMCL_NOMOTION')
                return
            if stage == 'AMCL_NOMOTION':
                self.enter_loc_stage('VERIFY_LOCALIZATION')
                return
            return

        if stage == 'VERIFY_GAZEBO':
            # Re-query Gazebo until its state is consistent.
            if time.monotonic() - self.loc_last_get_sent >= 0.3:
                self.loc_last_get_sent = time.monotonic()
                self.send_loc_service('VERIFY_GAZEBO')
        elif stage == 'VERIFY_EKF':
            # Require a message received AFTER the SetPose response.
            if (self.odom_count > self.loc_odom_baseline
                    and self.odom_state_is_valid()):
                self.get_logger().info('EKF reset VERIFIED')
                self.enter_loc_stage('SET_AMCL')
        elif stage == 'VERIFY_LOCALIZATION':
            # Require a new /amcl_pose after nomotion-update response.
            if (self.amcl_count > self.loc_amcl_baseline
                    and self.amcl_and_tf_are_valid()):
                self.episode_phase = 'LOCALIZATION_READY'
                self.loc_stage = None
                self.get_logger().info(
                    'LOCALIZATION_READY: Gazebo, EKF, AMCL, TF '
                    'consistent. MPC remains HOLD.')

    def gazebo_state_is_valid(self, state):
        pose = state.pose
        twist = state.twist
        xy = math.hypot(pose.position.x - self.reset_x,
                        pose.position.y - self.reset_y)
        yaw = abs(self.yaw_error(
            self.yaw_of(pose.orientation), self.reset_yaw))
        vxy = math.hypot(twist.linear.x, twist.linear.y)
        wz = abs(twist.angular.z)
        return (xy < self.gazebo_xy_tolerance and
                yaw < self.yaw_tolerance and
                vxy < self.velocity_tolerance and
                wz < self.velocity_tolerance)

    def odom_state_is_valid(self):
        if self.latest_odom is None:
            return False
        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        xy = math.hypot(pose.position.x - self.reset_x,
                        pose.position.y - self.reset_y)
        yaw = abs(self.yaw_error(
            self.yaw_of(pose.orientation), self.reset_yaw))
        vxy = math.hypot(twist.linear.x, twist.linear.y)
        wz = abs(twist.angular.z)
        return (xy < self.ekf_xy_tolerance and
                yaw < self.yaw_tolerance and
                vxy < self.velocity_tolerance and
                wz < self.velocity_tolerance)

    def amcl_and_tf_are_valid(self):
        if self.latest_amcl is None or not self.odom_state_is_valid():
            return False
        pose = self.latest_amcl.pose.pose
        amcl_xy = math.hypot(pose.position.x - self.reset_x,
                             pose.position.y - self.reset_y)
        amcl_yaw = abs(self.yaw_error(
            self.yaw_of(pose.orientation), self.reset_yaw))
        if (amcl_xy >= self.localization_xy_tolerance or
                amcl_yaw >= self.yaw_tolerance):
            return False
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', Time())
        except TransformException:
            return False
        trans = tf.transform.translation
        tf_xy = math.hypot(trans.x - self.reset_x,
                           trans.y - self.reset_y)
        tf_yaw = abs(self.yaw_error(
            self.yaw_of(tf.transform.rotation), self.reset_yaw))
        return (tf_xy < self.localization_xy_tolerance and
                tf_yaw < self.yaw_tolerance)

    # ------------------------------------------------------------
    # Step 3.5.5: explicit Prepare -> Goal -> Accepted Path -> Resume.
    # ------------------------------------------------------------
    @staticmethod
    def stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def start_episode_callback(self, request, response):
        if self.episode_phase != 'LOCALIZATION_READY':
            response.success = False
            response.message = (
                f'Start rejected: phase={self.episode_phase}; '
                'complete control and localization reset first')
            return response

        if not self.mpc_prepare_client.service_is_ready() \
                or not self.mpc_hold_client.service_is_ready():
            response.success = False
            response.message = 'Start rejected: MPC services unavailable'
            return response
        if not self.clear_global_costmap_client.service_is_ready():
            response.success = False
            response.message = (
                'Start rejected: Global Costmap clear service unavailable'
            )
            return response

        count = self.goal_pub.get_subscription_count()
        if count < self.required_goal_subscribers:
            response.success = False
            response.message = (
                f'Start rejected: goal subscribers={count}, '
                f'need {self.required_goal_subscribers}')
            return response
        if self.get_clock().now().nanoseconds == 0:
            response.success = False
            response.message = 'Start rejected: simulation clock unavailable'
            return response
        # The reset may be old: require current EKF, AMCL, TF near start.
        if not self.amcl_and_tf_are_valid():
            response.success = False
            response.message = 'Start rejected: localization no longer valid'
            return response

        self.episode_id += 1
        self.episode_started_monotonic = time.monotonic()
        self.episode_goal_stamp_ns = 0
        self.episode_accepted_path = None
        self.episode_accepted_stamp_ns = 0
        self.episode_last_invalid_stamp_ns = 0
        self.episode_accepted_monotonic = 0.0
        self.episode_next_resume_at = 0.0
        self.episode_resume_attempts = 0
        self.episode_last_error = ''
        self.episode_costmap_clear_done = 0.0
        self.latest_accepted_path = None

        self.episode_phase = 'EP_WAIT_CLEAR_COSTMAP'
        self.get_logger().info(
            f'Episode start requested (attempt={self.episode_id}), '
            f'goal=({self.goal_x:.3f}, {self.goal_y:.3f}, '
            f'{self.goal_yaw:.3f} rad)')
        self.send_episode_service('CLEAR_COSTMAP')
        response.success = self.episode_phase != 'EPISODE_ERROR'
        response.message = (
            'Episode start accepted; poll /rl_mpc/episode_status'
            if response.success else self.episode_last_error)
        return response

    def episode_status_callback(self, request, response):
        response.success = self.episode_phase == 'EPISODE_RUNNING'
        count = (len(self.episode_accepted_path.poses)
                 if self.episode_accepted_path is not None else 0)
        response.message = (
            f'phase={self.episode_phase}; attempt={self.episode_id}; '
            f'accepted_poses={count}; resume_attempts='
            f'{self.episode_resume_attempts}; '
            f'error={self.episode_last_error or "none"}')
        return response

    def send_episode_service(self, action):
        if action == 'CLEAR_COSTMAP':
            client = self.clear_global_costmap_client
            request = ClearEntireCostmap.Request()
            self.episode_phase = 'EP_WAIT_CLEAR_COSTMAP'
        elif action == 'PREPARE':
            client = self.mpc_prepare_client
            request = Trigger.Request()
            self.episode_phase = 'EP_WAIT_PREPARE'
        elif action == 'RESUME':
            client = self.mpc_hold_client
            request = SetBool.Request()
            request.data = False
            self.episode_phase = 'EP_WAIT_RESUME'
            self.episode_resume_attempts += 1
        else:
            self.fail_episode(f'Unknown episode action: {action}')
            return

        if not client.service_is_ready():
            self.fail_episode(f'{action}: service unavailable')
            return
        try:
            self.episode_future = client.call_async(request)
            self.episode_future_client = client
        except Exception as exc:
            self.fail_episode(f'{action}: service call failed: {exc}')
            return
        self.get_logger().info(
            f'{action}: request sent'
            + (f' (attempt {self.episode_resume_attempts})'
               if action == 'RESUME' else ''))

    def publish_episode_goal(self):
        if (self.goal_pub.get_subscription_count() <
                self.required_goal_subscribers):
            self.fail_episode('Goal subscribers disconnected before publish')
            return
        # Take ROS simulation time immediately before publishing.
        stamp_ns = self.get_clock().now().nanoseconds
        if stamp_ns == 0:
            self.fail_episode('Simulation clock unavailable at goal publish')
            return

        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        self.episode_goal_stamp_ns = self.stamp_to_ns(msg.header.stamp)
        self.fill_pose(msg.pose, self.goal_x, self.goal_y,
                       0.0, self.goal_yaw)
        self.episode_phase = 'EP_WAIT_PATH'
        self.episode_next_resume_at = 0.0
        self.goal_pub.publish(msg)
        self.get_logger().info(
            f'Goal published once to /goal_pose: '
            f'x={self.goal_x:.2f}, y={self.goal_y:.2f}, '
            f'yaw={self.goal_yaw:.2f}; waiting for accepted path')

    def process_episode_start(self):
        if self.episode_phase not in (
                'EP_WAIT_CLEAR_COSTMAP', 'EP_WAIT_COSTMAP_SETTLE',
                'EP_WAIT_PREPARE', 'EP_WAIT_PATH', 'EP_WAIT_RESUME'):
            return

        elapsed = time.monotonic() - self.episode_started_monotonic
        if elapsed > self.episode_start_timeout_sec:
            self.fail_episode(
                f'Episode startup timed out after {elapsed:.1f} s '
                f'(phase={self.episode_phase})')
            return

        if self.episode_future is not None:
            if not self.episode_future.done():
                return
            try:
                result = self.episode_future.result()
                if result is None:
                    raise RuntimeError('empty service response')
            except Exception as exc:
                self.fail_episode(f'MPC service exception: {exc}')
                return
            self.episode_future = None
            self.episode_future_client = None

            if self.episode_phase == 'EP_WAIT_CLEAR_COSTMAP':
                if hasattr(result, 'success') and not result.success:
                    self.fail_episode(
                        'CLEAR_COSTMAP returned success=False'
                    )
                    return

                self.episode_costmap_clear_done = time.monotonic()
                self.episode_phase = 'EP_WAIT_COSTMAP_SETTLE'

                self.get_logger().info(
                    'CLEAR_COSTMAP: SUCCESS; waiting '
                    f'{self.costmap_settle_sec:.2f} s before planning'
                )
                return

            if self.episode_phase == 'EP_WAIT_PREPARE':
                if not result.success:
                    self.fail_episode(
                        f'PREPARE rejected: {result.message}')
                    return
                self.get_logger().info('PREPARE: SUCCESS')
                self.publish_episode_goal()
                return

            if self.episode_phase == 'EP_WAIT_RESUME':
                if result.success:
                    if self.episode_accepted_path is None:
                        # Path was invalidated while RESUME was in flight.
                        self.fail_episode(
                            'RESUME acknowledged but path invalidated; '
                            'HOLD requested')
                        return
                    self.episode_phase = 'EPISODE_RUNNING'
                    self.get_logger().info(
                        f'EPISODE_RUNNING: MPC RESUME confirmed; '
                        f'attempt={self.episode_id}, '
                        f'goal=({self.goal_x:.2f},{self.goal_y:.2f}), '
                        f'resume_attempts={self.episode_resume_attempts}')
                    return

                reason = str(result.message)
                # At this point Goal Manager has produced a new accepted
                # path, but MPC might not have processed it yet.
                retryable = any(term in reason.lower() for term in (
                    'not ready', 'no accepted path', 'tf unavailable'))
                if retryable and (self.episode_resume_attempts <
                                  self.max_resume_attempts):
                    self.episode_phase = 'EP_WAIT_PATH'
                    self.episode_next_resume_at = (
                        time.monotonic() + self.resume_retry_sec)
                    self.get_logger().warn(
                        f'RESUME deferred ({reason}); '
                        'waiting for MPC to become READY')
                else:
                    self.fail_episode(f'RESUME rejected: {reason}')
                return

        if self.episode_phase == 'EP_WAIT_COSTMAP_SETTLE':
            if (time.monotonic() - self.episode_costmap_clear_done
                    >= self.costmap_settle_sec):

                self.get_logger().info(
                    'Costmap settle period completed; '
                    'starting MPC PREPARE'
                )
                self.send_episode_service('PREPARE')

            return

        if self.episode_phase == 'EP_WAIT_PATH':
            if self.episode_accepted_path is None:
                return
            if (time.monotonic() < self.episode_next_resume_at or
                    time.monotonic() - self.episode_accepted_monotonic <
                    self.resume_settle_sec):
                return
            if self.episode_resume_attempts >= self.max_resume_attempts:
                self.fail_episode('Maximum RESUME attempts reached')
                return
            self.send_episode_service('RESUME')

    def fail_episode(self, reason):
        old_phase = self.episode_phase
        if self.episode_future is not None and self.episode_future_client:
            try:
                if not self.episode_future.done():
                    self.episode_future_client.remove_pending_request(
                        self.episode_future)
            except Exception:
                pass
        self.episode_future = None
        self.episode_future_client = None
        self.episode_last_error = reason
        self.episode_phase = 'EPISODE_ERROR'
        self.get_logger().error(
            f'EPISODE START FAILED: {reason}. '
            'Do not resume MPC manually until checked.')
        # MPC normally remains in HOLD until RESUME succeeds. If a RESUME
        # request was in flight, it might still run remotely after timeout;
        # issue a best-effort explicit HOLD (not an execution guarantee).
        if self.episode_resume_attempts:
            if self.mpc_hold_client.service_is_ready():
                try:
                    request = SetBool.Request()
                    request.data = True
                    fut = self.mpc_hold_client.call_async(request)
                    fut.add_done_callback(self.episode_failure_hold_callback)
                except Exception as exc:
                    self.get_logger().error(
                        f'Failed to request safety HOLD: {exc}')
        if old_phase == 'EP_WAIT_RESUME':
            self.get_logger().warn(
                'A timed-out RESUME may finish remotely after HOLD; '
                'check MPC state before recovering.')

    def episode_failure_hold_callback(self, future):
        try:
            result = future.result()
            if result is not None and result.success:
                self.get_logger().warn(
                    'Safety HOLD acknowledged after episode-start failure')
            else:
                self.get_logger().error(
                    'Safety HOLD not acknowledged; manual check required')
        except Exception as exc:
            self.get_logger().error(
                f'Safety HOLD response error: {exc}')

    def fail_localization(self, reason):
        # Removing a pending future only discards the response;
        # it cannot cancel an already running server request.
        if self.loc_future is not None and self.loc_future_client is not None:
            try:
                if not self.loc_future.done():
                    self.loc_future_client.remove_pending_request(
                        self.loc_future)
            except Exception:
                pass
        self.loc_future = None
        self.loc_future_client = None
        self.loc_last_error = reason
        self.episode_phase = 'LOCALIZATION_ERROR'
        self.get_logger().error(
            f'LOCALIZATION RESET FAILED: {reason}. '
            'MPC must remain HOLD; do not RESUME.')


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()