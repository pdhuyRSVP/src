#!/usr/bin/env python3

import time
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Float64MultiArray
from std_srvs.srv import Trigger, SetBool

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException
)


class RosBridge(Node):

    def __init__(self):

        super().__init__('rl_mpc_ros_bridge')

        # ====================================================
        # Parameters
        # ====================================================

        self.declare_parameter(
            'path_topic',
            '/plan_smoothed'
        )

        self.declare_parameter(
            'odom_topic',
            '/odometry/filtered'
        )

        self.declare_parameter(
            'scan_topic',
            '/scan_filtered'
        )

        self.declare_parameter(
            'global_frame',
            'map'
        )

        self.declare_parameter(
            'robot_frame',
            'base_footprint'
        )

        self.declare_parameter(
            'status_probe_enabled',
            False
        )

        self.status_probe_enabled = bool(
            self.get_parameter('status_probe_enabled').value)

        path_topic = self.get_parameter(
            'path_topic').value

        odom_topic = self.get_parameter(
            'odom_topic').value

        scan_topic = self.get_parameter(
            'scan_topic').value

        self.global_frame = self.get_parameter(
            'global_frame').value

        self.robot_frame = self.get_parameter(
            'robot_frame').value

        # ====================================================
        # Data storage
        # ====================================================

        self.path = None

        self.odom = None

        self.scan = None

        self.reference = None

        self.solver_success = None

        self.solver_time = None

        self.adaptive_params = None

        # ====================================================
        # Subscribers
        # ====================================================

        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.reference_sub = self.create_subscription(
            PoseStamped,
            '/mpc/reference_pose',
            self.reference_callback,
            10
        )

        self.solver_success_sub = self.create_subscription(
            Bool,
            '/mpc/solver_success',
            self.solver_success_callback,
            10
        )

        self.solver_time_sub = self.create_subscription(
            Float64,
            '/mpc/solver_time',
            self.solver_time_callback,
            10
        )

        self.adaptive_params_sub = self.create_subscription(
            Float64MultiArray,
            '/mpc/current_adaptive_params',
            self.adaptive_params_callback,
            10
        )

        # ====================================================
        # RL Training - Episode Manager service clients
        # ====================================================

        self.reset_control_client = self.create_client(
            Trigger,
            '/rl_mpc/reset_control_pipeline'
        )

        self.reset_status_client = self.create_client(
            Trigger,
            '/rl_mpc/reset_status'
        )

        self.reset_localization_client = self.create_client(
            Trigger,
            '/rl_mpc/reset_localization'
        )

        self.start_episode_client = self.create_client(
            Trigger,
            '/rl_mpc/start_episode'
        )

        self.episode_status_client = self.create_client(
            Trigger,
            '/rl_mpc/episode_status'
        )

        # MPC safety HOLD
        self.mpc_hold_client = self.create_client(
            SetBool,
            '/mpc/set_training_hold'
        )

        # ====================================================
        # Required services for Gymnasium
        # ====================================================

        self.required_clients = {
            'reset_control': self.reset_control_client,
            'reset_status': self.reset_status_client,
            'reset_localization': self.reset_localization_client,
            'start_episode': self.start_episode_client,
            'episode_status': self.episode_status_client,
            'mpc_hold': self.mpc_hold_client,
        }

        # ====================================================
        # RL Training - Episode status request tracking
        # ====================================================

        self.episode_status_future = None

        self.episode_status_started = 0.0

        self.episode_status_timeout_sec = 5.0

        self.last_episode_status = None

        self.last_episode_phase = None

        # ====================================================
        # RL Training - Adaptive MPC publisher
        # ====================================================

        self.adaptive_param_pub = self.create_publisher(
            Float64MultiArray,
            '/mpc/adaptive_params',
            10
        )

        # ====================================================
        # TF
        # ====================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ====================================================
        # Diagnostic timer
        # ====================================================

        self.timer = self.create_timer(
            2.0,
            self.report_status
        )

        self.get_logger().info(
            'RL-MPC ROS Bridge Started'
        )

        self.get_logger().info(
            f'Path topic: {path_topic}'
        )

        self.get_logger().info(
            f'Odometry topic: {odom_topic}'
        )

        self.get_logger().info(
            f'LiDAR topic: {scan_topic}'
        )

    # ========================================================
    # Callbacks
    # ========================================================

    def path_callback(self, msg):

        self.path = msg

    def odom_callback(self, msg):

        self.odom = msg

    def scan_callback(self, msg):

        self.scan = msg

    def reference_callback(self, msg):

        self.reference = msg

    def solver_success_callback(self, msg):

        self.solver_success = bool(msg.data)

    def solver_time_callback(self, msg):

        self.solver_time = float(msg.data)

    def adaptive_params_callback(self, msg):

        if len(msg.data) != 5:

            self.get_logger().warn(
                'Expected 5 adaptive MPC parameters'
            )

            return

        self.adaptive_params = list(msg.data)

    # ========================================================
    # RL Training - Parse Episode Manager service response
    # ========================================================

    @staticmethod
    def parse_service_message(message):

        fields = {}

        for part in str(message).split(';'):

            if '=' not in part:
                continue

            key, value = part.strip().split('=', 1)

            fields[key.strip()] = value.strip()

        return fields

    # ========================================================
    # RL Training - Non-blocking Episode Manager status poll
    # ========================================================

    def poll_episode_status(self):

        if not self.status_probe_enabled:
            return

        # ----------------------------------------------------
        # 1. Process existing request
        # ----------------------------------------------------

        if self.episode_status_future is not None:

            future = self.episode_status_future

            if future.done():

                try:

                    response = future.result()

                    if response is None:
                        raise RuntimeError(
                            'Empty Episode Manager response'
                        )

                    status = self.parse_service_message(
                        response.message
                    )

                    phase = status.get('phase')

                    if not phase:

                        self.get_logger().warn(
                            'Episode status response has no phase'
                        )

                    else:

                        self.last_episode_status = status

                        if phase != self.last_episode_phase:

                            self.last_episode_phase = phase

                            self.get_logger().info(
                                f'Episode Manager phase={phase}'
                            )

                except Exception as exc:

                    self.get_logger().warn(
                        f'Episode status request failed: {exc}'
                    )

                self.episode_status_future = None

            else:

                elapsed = (
                    time.monotonic()
                    - self.episode_status_started
                )

                if elapsed > self.episode_status_timeout_sec:

                    try:

                        self.episode_status_client.remove_pending_request(
                            future
                        )

                    except Exception:
                        pass

                    self.episode_status_future = None

                    self.get_logger().warn(
                        'Episode status request timed out'
                    )

                return

        # ----------------------------------------------------
        # 2. Check service availability
        # ----------------------------------------------------

        if not self.episode_status_client.service_is_ready():
            return

        # ----------------------------------------------------
        # 3. Send new asynchronous request
        # ----------------------------------------------------

        try:

            self.episode_status_future = (
                self.episode_status_client.call_async(
                    Trigger.Request()
                )
            )

            self.episode_status_started = time.monotonic()

        except Exception as exc:

            self.get_logger().warn(
                f'Failed to send episode status request: {exc}'
            )

    # ========================================================
    # Robot pose from TF
    # ========================================================

    def get_robot_pose(self):

        target_frame = self.global_frame

        if (
            self.path is not None
            and self.path.header.frame_id
        ):

            target_frame = self.path.header.frame_id

        try:

            transform = self.tf_buffer.lookup_transform(
                target_frame,
                self.robot_frame,
                Time()
            )

        except TransformException:

            return None

        x = transform.transform.translation.x

        y = transform.transform.translation.y

        return x, y

    # ========================================================
    # Diagnostic report
    # ========================================================

    def report_status(self):

        # ====================================================
        # RL Training - Service availability
        # ====================================================

        missing_services = [
            name
            for name, client in self.required_clients.items()
            if not client.service_is_ready()
        ]

        ready_services = (
            len(self.required_clients) - len(missing_services)
        )

        total_services = len(self.required_clients)

        # Check whether MPC subscribes to adaptive parameters
        adaptive_subscribers = (
            self.adaptive_param_pub.get_subscription_count()
        )

        robot_pose = self.get_robot_pose()

        if robot_pose is None:

            tf_status = 'WAIT'

        else:

            tf_status = (
                f'OK x={robot_pose[0]:.2f}, '
                f'y={robot_pose[1]:.2f}'
            )

        if self.path is None:

            path_status = 'WAIT'

        else:

            path_status = str(len(self.path.poses))

        if self.odom is None:

            odom_status = 'WAIT'

        else:

            v = self.odom.twist.twist.linear.x

            w = self.odom.twist.twist.angular.z

            odom_status = (
                f'OK v={v:.2f}, w={w:.2f}'
            )

        if self.scan is None:

            scan_status = 'WAIT'

        else:

            scan_status = str(len(self.scan.ranges))

        if self.solver_success is None:

            solver_status = 'WAIT'

        else:

            solver_status = str(self.solver_success)

        if self.solver_time is None:

            solver_time_status = 'WAIT'

        else:

            solver_time_status = (
                f'{self.solver_time * 1000.0:.1f} ms'
            )

        if self.adaptive_params is None:

            adaptive_status = 'WAIT'

        else:

            adaptive_status = str(
                [round(x, 3) for x in self.adaptive_params]
            )

        self.get_logger().info(
            f'TF: {tf_status} | '
            f'Path: {path_status} | '
            f'Odom: {odom_status} | '
            f'LiDAR: {scan_status} | '
            f'Solver: {solver_status} | '
            f'Solve time: {solver_time_status} | '
            f'Adaptive: {adaptive_status} | '
            f'RL Services: {ready_services}/{total_services} | '
            f'Adaptive subscribers: {adaptive_subscribers}'
        )
        
        if missing_services:
            self.get_logger().debug(
                f'Missing RL services: {missing_services}'
            )

        # Non-blocking status probe
        self.poll_episode_status()


def main(args=None):

    rclpy.init(args=args)

    node = RosBridge()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
