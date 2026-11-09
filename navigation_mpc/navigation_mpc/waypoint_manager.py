import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException
)


class WaypointManager(Node):

    def __init__(self):

        super().__init__('waypoint_manager')

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter(
            'position_tolerance',
            0.25
        )

        self.declare_parameter(
            'arrival_hold_time',
            1.0
        )

        self.declare_parameter(
            'check_yaw',
            False
        )

        self.declare_parameter(
            'yaw_tolerance',
            0.20
        )

        # ============================================================
        # Waypoint parameters
        #
        # Default values are used only if YAML is not loaded.
        # ============================================================

        self.declare_parameter(
            'waypoint_x',
            [10.0, 0.0]
        )

        self.declare_parameter(
            'waypoint_y',
            [0.0, 5.0]
        )

        self.declare_parameter(
            'waypoint_yaw',
            [0.0, 0.0]
        )

        # ============================================================
        # Read parameters
        # ============================================================

        self.position_tolerance = float(
            self.get_parameter(
                'position_tolerance'
            ).value
        )

        self.arrival_hold_time = float(
            self.get_parameter(
                'arrival_hold_time'
            ).value
        )

        self.check_yaw = bool(
            self.get_parameter(
                'check_yaw'
            ).value
        )

        self.yaw_tolerance = float(
            self.get_parameter(
                'yaw_tolerance'
            ).value
        )

        waypoint_x = list(
            self.get_parameter(
                'waypoint_x'
            ).value
        )

        waypoint_y = list(
            self.get_parameter(
                'waypoint_y'
            ).value
        )

        waypoint_yaw = list(
            self.get_parameter(
                'waypoint_yaw'
            ).value
        )

        # ============================================================
        # Validate waypoint configuration
        # ============================================================

        if not (
            len(waypoint_x)
            ==
            len(waypoint_y)
            ==
            len(waypoint_yaw)
        ):

            raise RuntimeError(
                'Waypoint configuration error: '
                'waypoint_x, waypoint_y and waypoint_yaw '
                'must have the same number of elements.'
            )

        if len(waypoint_x) == 0:

            raise RuntimeError(
                'Waypoint configuration error: '
                'no waypoints defined.'
            )

        # ============================================================
        # Build waypoint list
        #
        # [
        #   (x1, y1, yaw1),
        #   (x2, y2, yaw2),
        #   ...
        # ]
        # ============================================================

        self.waypoints = []

        for x, y, yaw in zip(
            waypoint_x,
            waypoint_y,
            waypoint_yaw
        ):

            self.waypoints.append(
                (
                    float(x),
                    float(y),
                    float(yaw)
                )
            )

        # ============================================================
        # Publishers
        # ============================================================

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.mission_complete_pub = (
            self.create_publisher(
                Bool,
                '/navigation_mpc/mission_complete',
                10
            )
        )

        self.mission_complete_sent = False

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # States
        # ============================================================

        self.current_waypoint = 0

        self.goal_sent = False

        self.arrival_start_time = None

        self.mission_completed = False

        self.robot_pose_available = False

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # ============================================================
        # Timer
        #
        # 0.2 s = 5 Hz
        # ============================================================

        self.timer = self.create_timer(
            0.2,
            self.control_loop
        )

        # ============================================================
        # Startup information
        # ============================================================

        self.get_logger().info(
            f'Waypoint Manager started: '
            f'{len(self.waypoints)} waypoint(s)'
        )

        for i, waypoint in enumerate(
            self.waypoints
        ):

            x, y, yaw = waypoint

            self.get_logger().info(
                f'Waypoint {i + 1}: '
                f'x={x:.2f}, '
                f'y={y:.2f}, '
                f'yaw={yaw:.2f}'
            )

    # ================================================================
    # TF robot pose
    # ================================================================

    def update_robot_pose(self):

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    'map',
                    'base_footprint',
                    Time()
                )
            )

        except TransformException:

            self.robot_pose_available = False

            return False

        self.robot_x = (
            transform.transform.translation.x
        )

        self.robot_y = (
            transform.transform.translation.y
        )

        q = transform.transform.rotation

        self.robot_yaw = (
            self.quaternion_to_yaw(
                q.x,
                q.y,
                q.z,
                q.w
            )
        )

        self.robot_pose_available = True

        return True

    # ================================================================
    # Angle utilities
    # ================================================================

    def yaw_to_quaternion(
        self,
        yaw
    ):

        qz = math.sin(
            yaw / 2.0
        )

        qw = math.cos(
            yaw / 2.0
        )

        return qz, qw

    def quaternion_to_yaw(
        self,
        qx,
        qy,
        qz,
        qw
    ):

        siny_cosp = 2.0 * (
            qw * qz
            +
            qx * qy
        )

        cosy_cosp = (
            1.0
            -
            2.0
            *
            (
                qy * qy
                +
                qz * qz
            )
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    def normalize_angle(
        self,
        angle
    ):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    # ================================================================
    # Publish waypoint
    # ================================================================

    def publish_goal(
        self,
        x,
        y,
        yaw
    ):

        goal = PoseStamped()

        goal.header.frame_id = 'map'

        goal.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        qz, qw = (
            self.yaw_to_quaternion(
                yaw
            )
        )

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(
            goal
        )

        self.get_logger().info(
            f'Publish waypoint '
            f'{self.current_waypoint + 1}/'
            f'{len(self.waypoints)}: '
            f'x={x:.2f}, '
            f'y={y:.2f}, '
            f'yaw={yaw:.2f}'
        )

    # ================================================================
    # Publish mission complete
    # ================================================================

    def publish_mission_complete(self):

        if self.mission_complete_sent:
            return

        msg = Bool()

        msg.data = True

        self.mission_complete_pub.publish(
            msg
        )

        self.mission_complete_sent = True

        self.get_logger().info(
            'MISSION COMPLETE signal published'
        )

    # ================================================================
    # Waypoint condition
    # ================================================================

    def waypoint_reached(
        self,
        x,
        y,
        yaw
    ):

        distance = math.hypot(
            x - self.robot_x,
            y - self.robot_y
        )

        # ------------------------------------------------------------
        # Position condition
        # ------------------------------------------------------------

        if (
            distance
            >
            self.position_tolerance
        ):

            return False

        # ------------------------------------------------------------
        # Optional yaw condition
        # ------------------------------------------------------------

        if self.check_yaw:

            yaw_error = abs(
                self.normalize_angle(
                    yaw
                    -
                    self.robot_yaw
                )
            )

            if (
                yaw_error
                >
                self.yaw_tolerance
            ):

                return False

        return True

    # ================================================================
    # Control loop
    # ================================================================

    def control_loop(self):

        # ------------------------------------------------------------
        # Mission already completed
        # ------------------------------------------------------------

        if self.mission_completed:
            return

        # ------------------------------------------------------------
        # Need valid robot localization
        # ------------------------------------------------------------

        if not self.update_robot_pose():
            return

        # ------------------------------------------------------------
        # Safety check: no more waypoints
        # ------------------------------------------------------------

        if (
            self.current_waypoint
            >=
            len(self.waypoints)
        ):

            self.mission_completed = True

            self.publish_mission_complete()

            self.get_logger().info(
                'ALL WAYPOINTS COMPLETED'
            )

            return

        # ============================================================
        # Current waypoint
        # ============================================================

        x, y, yaw = (
            self.waypoints[
                self.current_waypoint
            ]
        )

        # ------------------------------------------------------------
        # Send waypoint
        # ------------------------------------------------------------

        if not self.goal_sent:

            self.publish_goal(
                x,
                y,
                yaw
            )

            self.goal_sent = True

            self.arrival_start_time = None

            return

        # ------------------------------------------------------------
        # Check waypoint
        # ------------------------------------------------------------

        if not self.waypoint_reached(
            x,
            y,
            yaw
        ):

            self.arrival_start_time = None

            return

        # ============================================================
        # Robot is inside waypoint tolerance
        # ============================================================

        now = self.get_clock().now()

        # ------------------------------------------------------------
        # Start hold timer
        # ------------------------------------------------------------

        if self.arrival_start_time is None:

            self.arrival_start_time = now

            return

        # ------------------------------------------------------------
        # Check hold duration
        # ------------------------------------------------------------

        elapsed = (
            now
            -
            self.arrival_start_time
        ).nanoseconds / 1e9

        if (
            elapsed
            <
            self.arrival_hold_time
        ):

            return

        # ============================================================
        # Waypoint successfully reached
        # ============================================================

        self.get_logger().info(
            f'Reached waypoint '
            f'{self.current_waypoint + 1}/'
            f'{len(self.waypoints)}'
        )

        # Move to next waypoint
        self.current_waypoint += 1

        # Reset waypoint state
        self.goal_sent = False
        self.arrival_start_time = None

        # ============================================================
        # Was this the final waypoint?
        # ============================================================

        if (
            self.current_waypoint
            >=
            len(self.waypoints)
        ):

            self.mission_completed = True

            self.publish_mission_complete()

            self.get_logger().info(
                'ALL WAYPOINTS COMPLETED'
            )

            return


def main(args=None):

    rclpy.init(
        args=args
    )

    node = WaypointManager()

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