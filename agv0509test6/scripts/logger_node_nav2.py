#!/usr/bin/env python3

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import (
    Twist,
    PoseStamped
)

from nav_msgs.msg import (
    Odometry,
    Path
)

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException
)


class Nav2Logger(Node):

    def __init__(self):

        super().__init__('nav2_logger')

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter(
            'log_frequency',
            10.0
        )

        self.declare_parameter(
            'output_directory',
            os.path.expanduser(
                '~/agv_ws/results'
            )
        )

        # Used only for file naming:
        # dwb / mppi
        self.declare_parameter(
            'controller_name',
            'nav2'
        )

        # Start recording when robot actually starts moving
        self.declare_parameter(
            'linear_start_threshold',
            0.01
        )

        self.declare_parameter(
            'angular_start_threshold',
            0.02
        )

        # Stop condition
        self.declare_parameter(
            'linear_stop_threshold',
            0.01
        )

        self.declare_parameter(
            'angular_stop_threshold',
            0.02
        )

        self.declare_parameter(
            'stop_hold_time',
            0.5
        )

        # ============================================================
        # Read parameters
        # ============================================================

        self.log_frequency = float(
            self.get_parameter(
                'log_frequency'
            ).value
        )

        self.output_directory = str(
            self.get_parameter(
                'output_directory'
            ).value
        )

        self.controller_name = str(
            self.get_parameter(
                'controller_name'
            ).value
        )

        self.linear_start_threshold = float(
            self.get_parameter(
                'linear_start_threshold'
            ).value
        )

        self.angular_start_threshold = float(
            self.get_parameter(
                'angular_start_threshold'
            ).value
        )

        self.linear_stop_threshold = float(
            self.get_parameter(
                'linear_stop_threshold'
            ).value
        )

        self.angular_stop_threshold = float(
            self.get_parameter(
                'angular_stop_threshold'
            ).value
        )

        self.stop_hold_time = float(
            self.get_parameter(
                'stop_hold_time'
            ).value
        )

        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

        # ============================================================
        # Logger state
        # ============================================================

        self.first_goal_received = False

        self.recording = False

        self.finished = False

        self.mission_complete = False

        self.start_time = None

        self.stop_candidate_time = None

        # ============================================================
        # Files
        # ============================================================

        self.csv_file = None
        self.writer = None

        self.path_csv_file = None
        self.path_writer = None

        self.data_file_path = None
        self.path_file_path = None

        self.rows_since_flush = 0

        # ============================================================
        # Latest data
        # ============================================================

        self.v_cmd = 0.0
        self.w_cmd = 0.0

        self.v_actual = 0.0
        self.w_actual = 0.0

        self.min_laser_range = float('nan')

        self.goal_x = float('nan')
        self.goal_y = float('nan')
        self.goal_yaw = float('nan')

        self.path_id = 0
        self.num_path_points = 0

        self.latest_path = None
        self.latest_path_id = 0

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # Subscribers
        # ============================================================

        # Diagnostic topic published by waypoint_manager_nav2
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # Final mission status
        self.mission_complete_sub = (
            self.create_subscription(
                Bool,
                '/navigation_nav2/mission_complete',
                self.mission_complete_callback,
                10
            )
        )

        # Controller command
        self.cmd_sub = self.create_subscription(
            Twist,
            '/diff_drive_controller/cmd_vel_unstamped',
            self.cmd_callback,
            10
        )

        # Actual velocity from EKF
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10
        )

        # LiDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.scan_callback,
            10
        )

        # Global path
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # ============================================================
        # Logger timer
        #
        # Default: 10 Hz
        # ============================================================

        self.timer = self.create_timer(
            1.0 / self.log_frequency,
            self.logger_loop
        )

        self.get_logger().info(
            f'Nav2 logger started '
            f'[{self.controller_name}] - '
            f'waiting for first waypoint'
        )

    # ================================================================
    # Quaternion -> yaw
    # ================================================================

    def quaternion_to_yaw(
        self,
        q
    ):

        siny_cosp = 2.0 * (
            q.w * q.z
            +
            q.x * q.y
        )

        cosy_cosp = (
            1.0
            -
            2.0
            *
            (
                q.y * q.y
                +
                q.z * q.z
            )
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    # ================================================================
    # Goal callback
    # ================================================================

    def goal_callback(
        self,
        msg
    ):

        self.goal_x = (
            msg.pose.position.x
        )

        self.goal_y = (
            msg.pose.position.y
        )

        self.goal_yaw = (
            self.quaternion_to_yaw(
                msg.pose.orientation
            )
        )

        if not self.first_goal_received:

            self.first_goal_received = True

            self.get_logger().info(
                'First waypoint received - '
                'waiting for robot motion'
            )

    # ================================================================
    # Mission complete
    # ================================================================

    def mission_complete_callback(
        self,
        msg
    ):

        if msg.data:

            self.mission_complete = True

            self.get_logger().info(
                'Final waypoint completed - '
                'waiting for robot to stop'
            )

    # ================================================================
    # Command callback
    # ================================================================

    def cmd_callback(
        self,
        msg
    ):

        self.v_cmd = (
            msg.linear.x
        )

        self.w_cmd = (
            msg.angular.z
        )

    # ================================================================
    # Odometry callback
    #
    # IMPORTANT:
    # Pose from this message is NOT used because it is in odom frame.
    # Only velocity is used here.
    # ================================================================

    def odom_callback(
        self,
        msg
    ):

        self.v_actual = (
            msg.twist.twist.linear.x
        )

        self.w_actual = (
            msg.twist.twist.angular.z
        )

    # ================================================================
    # Laser callback
    # ================================================================

    def scan_callback(
        self,
        msg
    ):

        valid_ranges = [
            r
            for r in msg.ranges
            if (
                math.isfinite(r)
                and
                msg.range_min <= r <= msg.range_max
            )
        ]

        if valid_ranges:

            self.min_laser_range = min(
                valid_ranges
            )

        else:

            self.min_laser_range = (
                float('nan')
            )

    # ================================================================
    # Global path callback
    # ================================================================

    def path_callback(
        self,
        msg
    ):

        if len(msg.poses) == 0:
            return

        self.path_id += 1

        self.latest_path_id = (
            self.path_id
        )

        self.latest_path = msg

        self.num_path_points = len(
            msg.poses
        )

        # If experiment is already running,
        # save every new/replanned path immediately.
        if self.recording:

            self.write_path(
                msg,
                self.path_id
            )

    # ================================================================
    # Start condition
    # ================================================================

    def robot_is_moving(self):

        return (
            abs(self.v_actual)
            >
            self.linear_start_threshold

            or

            abs(self.w_actual)
            >
            self.angular_start_threshold
        )

    # ================================================================
    # Stop condition
    # ================================================================

    def robot_is_stopped(self):

        actual_stopped = (
            abs(self.v_actual)
            <
            self.linear_stop_threshold

            and

            abs(self.w_actual)
            <
            self.angular_stop_threshold
        )

        command_stopped = (
            abs(self.v_cmd)
            <
            self.linear_stop_threshold

            and

            abs(self.w_cmd)
            <
            self.angular_stop_threshold
        )

        return (
            actual_stopped
            and
            command_stopped
        )

    # ================================================================
    # Start recording
    # ================================================================

    def start_recording(self):

        timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        # ------------------------------------------------------------
        # Main data file
        # ------------------------------------------------------------

        self.data_file_path = os.path.join(
            self.output_directory,
            f'{self.controller_name}_data_{timestamp}.csv'
        )

        # ------------------------------------------------------------
        # Global path file
        # ------------------------------------------------------------

        self.path_file_path = os.path.join(
            self.output_directory,
            f'{self.controller_name}_path_{timestamp}.csv'
        )

        # ============================================================
        # Main CSV
        # ============================================================

        self.csv_file = open(
            self.data_file_path,
            'w',
            newline=''
        )

        self.writer = csv.writer(
            self.csv_file
        )

        self.writer.writerow([
            'time',

            'x',
            'y',
            'yaw',

            'v_cmd',
            'w_cmd',

            'v_actual',
            'w_actual',

            'min_laser_range',

            'goal_x',
            'goal_y',
            'goal_yaw',

            'path_id',
            'num_path_points'
        ])

        # ============================================================
        # Path CSV
        # ============================================================

        self.path_csv_file = open(
            self.path_file_path,
            'w',
            newline=''
        )

        self.path_writer = csv.writer(
            self.path_csv_file
        )

        self.path_writer.writerow([
            'path_id',
            'point_index',
            'x',
            'y',
            'yaw'
        ])

        # ------------------------------------------------------------
        # Save path which existed before robot started moving
        # ------------------------------------------------------------

        if self.latest_path is not None:

            self.write_path(
                self.latest_path,
                self.latest_path_id
            )

        # ------------------------------------------------------------
        # Experiment starts at t = 0
        # ------------------------------------------------------------

        self.start_time = (
            self.get_clock().now()
        )

        self.recording = True

        self.get_logger().info(
            'Recording started'
        )

        self.get_logger().info(
            f'Data file: '
            f'{self.data_file_path}'
        )

        self.get_logger().info(
            f'Path file: '
            f'{self.path_file_path}'
        )

    # ================================================================
    # Write complete global path
    # ================================================================

    def write_path(
        self,
        msg,
        path_id
    ):

        if self.path_writer is None:
            return

        for i, pose_stamped in enumerate(
            msg.poses
        ):

            pose = (
                pose_stamped.pose
            )

            yaw = (
                self.quaternion_to_yaw(
                    pose.orientation
                )
            )

            self.path_writer.writerow([
                path_id,
                i,
                pose.position.x,
                pose.position.y,
                yaw
            ])

        self.path_csv_file.flush()

    # ================================================================
    # Logger loop
    # ================================================================

    def logger_loop(self):

        # ------------------------------------------------------------
        # Experiment already finished
        # ------------------------------------------------------------

        if self.finished:
            return

        # ------------------------------------------------------------
        # Wait until first waypoint is assigned
        # ------------------------------------------------------------

        if not self.first_goal_received:
            return

        # ------------------------------------------------------------
        # Wait until robot actually starts moving
        # ------------------------------------------------------------

        if not self.recording:

            if (
                self.robot_is_moving()
                and
                self.latest_path is not None
            ):

                self.start_recording()

            else:

                return

        # ============================================================
        # Robot pose in MAP frame
        # ============================================================

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    'map',
                    'base_footprint',
                    Time()
                )
            )

        except TransformException:
            return

        x = (
            transform.transform.translation.x
        )

        y = (
            transform.transform.translation.y
        )

        yaw = (
            self.quaternion_to_yaw(
                transform.transform.rotation
            )
        )

        # ============================================================
        # Relative ROS simulation time
        # ============================================================

        now = (
            self.get_clock().now()
        )

        elapsed_time = (
            now
            -
            self.start_time
        ).nanoseconds / 1e9

        # ============================================================
        # Write RAW data only
        # ============================================================

        self.writer.writerow([
            elapsed_time,

            x,
            y,
            yaw,

            self.v_cmd,
            self.w_cmd,

            self.v_actual,
            self.w_actual,

            self.min_laser_range,

            self.goal_x,
            self.goal_y,
            self.goal_yaw,

            self.path_id,
            self.num_path_points
        ])

        # ============================================================
        # Buffered flush
        # ============================================================

        self.rows_since_flush += 1

        if (
            self.rows_since_flush
            >=
            50
        ):

            self.csv_file.flush()

            self.rows_since_flush = 0

        # ============================================================
        # Automatic finish condition
        # ============================================================

        if self.mission_complete:

            if self.robot_is_stopped():

                if (
                    self.stop_candidate_time
                    is None
                ):

                    self.stop_candidate_time = (
                        now
                    )

                stopped_duration = (
                    now
                    -
                    self.stop_candidate_time
                ).nanoseconds / 1e9

                if (
                    stopped_duration
                    >=
                    self.stop_hold_time
                ):

                    self.finish_recording()

            else:

                # Robot moved again
                self.stop_candidate_time = None

    # ================================================================
    # Finish recording
    # ================================================================

    def finish_recording(self):

        if self.finished:
            return

        self.recording = False
        self.finished = True

        if self.csv_file is not None:

            self.csv_file.flush()
            self.csv_file.close()

            self.csv_file = None

        if self.path_csv_file is not None:

            self.path_csv_file.flush()
            self.path_csv_file.close()

            self.path_csv_file = None

        self.get_logger().info(
            'Recording completed automatically'
        )

        self.get_logger().info(
            f'Data saved: '
            f'{self.data_file_path}'
        )

        self.get_logger().info(
            f'Path saved: '
            f'{self.path_file_path}'
        )

        self.timer.cancel()

    # ================================================================
    # Shutdown fallback
    # ================================================================

    def destroy_node(self):

        if self.csv_file is not None:

            self.csv_file.flush()
            self.csv_file.close()

            self.csv_file = None

        if self.path_csv_file is not None:

            self.path_csv_file.flush()
            self.path_csv_file.close()

            self.path_csv_file = None

        super().destroy_node()


def main(args=None):

    rclpy.init(
        args=args
    )

    node = Nav2Logger()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()