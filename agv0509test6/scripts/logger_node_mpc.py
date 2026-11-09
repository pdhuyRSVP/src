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

from std_msgs.msg import (
    Float64,
    Bool
)

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException
)


class MPCDataLogger(Node):

    def __init__(self):

        super().__init__('mpc_data_logger')

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

        # Robot considered moving
        self.declare_parameter(
            'linear_start_threshold',
            0.01
        )

        self.declare_parameter(
            'angular_start_threshold',
            0.02
        )

        # Robot considered stopped
        self.declare_parameter(
            'linear_stop_threshold',
            0.01
        )

        self.declare_parameter(
            'angular_stop_threshold',
            0.02
        )

        # Robot must remain stopped this long
        self.declare_parameter(
            'stop_hold_time',
            0.5
        )

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
        # Logger state machine
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
        # TF
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # Latest raw data
        # ============================================================

        self.reference_pose = None

        self.v_cmd = 0.0
        self.w_cmd = 0.0

        self.v_actual = 0.0
        self.w_actual = 0.0

        self.solver_time = float('nan')
        self.solver_success = False
        self.solver_id = 0

        self.min_laser_range = float('nan')

        self.goal_x = float('nan')
        self.goal_y = float('nan')

        self.path_id = 0
        self.num_path_points = 0

        self.latest_path = None
        self.latest_path_id = 0

        # ============================================================
        # Subscribers
        # ============================================================

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.mission_complete_sub = (
            self.create_subscription(
                Bool,
                '/navigation_mpc/mission_complete',
                self.mission_complete_callback,
                10
            )
        )

        self.reference_sub = (
            self.create_subscription(
                PoseStamped,
                '/mpc/reference_pose',
                self.reference_callback,
                10
            )
        )

        # MPC output command
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

        self.solver_time_sub = (
            self.create_subscription(
                Float64,
                '/mpc/solver_time',
                self.solver_time_callback,
                10
            )
        )

        self.solver_success_sub = (
            self.create_subscription(
                Bool,
                '/mpc/solver_success',
                self.solver_success_callback,
                10
            )
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.scan_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # ============================================================
        # Logger timer
        # ============================================================

        self.timer = self.create_timer(
            1.0 / self.log_frequency,
            self.logger_loop
        )

        self.get_logger().info(
            'MPC logger started - waiting for first waypoint'
        )

    # ================================================================
    # Goal
    # ================================================================

    def goal_callback(self, msg):

        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

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
                'Final waypoint reached - '
                'waiting for robot to stop'
            )

    # ================================================================
    # MPC reference
    # ================================================================

    def reference_callback(
        self,
        msg
    ):

        self.reference_pose = msg

    # ================================================================
    # MPC command
    # ================================================================

    def cmd_callback(
        self,
        msg
    ):

        self.v_cmd = msg.linear.x
        self.w_cmd = msg.angular.z

    # ================================================================
    # Actual velocity
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
    # Solver
    # ================================================================

    def solver_time_callback(
        self,
        msg
    ):

        self.solver_time = msg.data

    def solver_success_callback(
        self,
        msg
    ):

        self.solver_success = msg.data

        # One ID per completed solver call
        self.solver_id += 1

    # ================================================================
    # Laser
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
    # Global path
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

        # Save new/replanned path while recording
        if self.recording:

            self.write_path(
                msg,
                self.path_id
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
    # Create files and begin experiment
    # ================================================================

    def start_recording(self):

        timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        self.data_file_path = os.path.join(
            self.output_directory,
            f'mpc_data_{timestamp}.csv'
        )

        self.path_file_path = os.path.join(
            self.output_directory,
            f'mpc_path_{timestamp}.csv'
        )

        # ------------------------------------------------------------
        # Main data file
        # ------------------------------------------------------------

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

            'x_ref',
            'y_ref',
            'yaw_ref',

            'v_cmd',
            'w_cmd',

            'v_actual',
            'w_actual',

            'solver_id',
            'solver_time',
            'solver_success',

            'min_laser_range',

            'goal_x',
            'goal_y',

            'path_id',
            'num_path_points'
        ])

        # ------------------------------------------------------------
        # Global path file
        # ------------------------------------------------------------

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

        # Save current global path immediately
        if self.latest_path is not None:

            self.write_path(
                self.latest_path,
                self.latest_path_id
            )

        # ------------------------------------------------------------
        # Start experiment time
        # ------------------------------------------------------------

        self.start_time = (
            self.get_clock().now()
        )

        self.recording = True

        self.get_logger().info(
            'Recording started'
        )

        self.get_logger().info(
            f'Data file: {self.data_file_path}'
        )

        self.get_logger().info(
            f'Path file: {self.path_file_path}'
        )

    # ================================================================
    # Save complete global path
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

            pose = pose_stamped.pose

            yaw = self.quaternion_to_yaw(
                pose.orientation
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
    # Main logger loop
    # ================================================================

    def logger_loop(self):

        if self.finished:
            return

        # ============================================================
        # WAIT FOR FIRST WAYPOINT
        # ============================================================

        if not self.first_goal_received:
            return

        # ============================================================
        # WAIT FOR ROBOT TO ACTUALLY MOVE
        # ============================================================

        if not self.recording:

            if (
                self.robot_is_moving()
                and
                self.reference_pose is not None
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

        yaw = self.quaternion_to_yaw(
            transform.transform.rotation
        )

        # ============================================================
        # MPC reference
        # ============================================================

        if self.reference_pose is None:
            return

        x_ref = (
            self.reference_pose
            .pose
            .position
            .x
        )

        y_ref = (
            self.reference_pose
            .pose
            .position
            .y
        )

        yaw_ref = self.quaternion_to_yaw(
            self.reference_pose
            .pose
            .orientation
        )

        # ============================================================
        # Relative experiment time
        # ============================================================

        now = self.get_clock().now()

        elapsed_time = (
            now - self.start_time
        ).nanoseconds / 1e9

        # ============================================================
        # Write RAW data only
        # ============================================================

        self.writer.writerow([
            elapsed_time,

            x,
            y,
            yaw,

            x_ref,
            y_ref,
            yaw_ref,

            self.v_cmd,
            self.w_cmd,

            self.v_actual,
            self.w_actual,

            self.solver_id,
            self.solver_time,
            int(
                self.solver_success
            ),

            self.min_laser_range,

            self.goal_x,
            self.goal_y,

            self.path_id,
            self.num_path_points
        ])

        # ============================================================
        # Buffered flush
        # ============================================================

        self.rows_since_flush += 1

        if self.rows_since_flush >= 50:

            self.csv_file.flush()

            self.rows_since_flush = 0

        # ============================================================
        # AUTOMATIC END CONDITION
        # ============================================================

        if self.mission_complete:

            if self.robot_is_stopped():

                if self.stop_candidate_time is None:

                    self.stop_candidate_time = now

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

                # Robot moved again -> reset stop timer
                self.stop_candidate_time = None

    # ================================================================
    # Finish experiment automatically
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
            f'Data saved: {self.data_file_path}'
        )

        self.get_logger().info(
            f'Path saved: {self.path_file_path}'
        )

        # No more periodic logging
        self.timer.cancel()

    # ================================================================
    # Shutdown fallback
    # ================================================================

    def destroy_node(self):

        # Only needed if ROS is closed before normal mission completion
        if self.csv_file is not None:

            self.csv_file.flush()
            self.csv_file.close()

        if self.path_csv_file is not None:

            self.path_csv_file.flush()
            self.path_csv_file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = MPCDataLogger()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()