#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class WaypointManagerNav2(Node):

    def __init__(self):

        # ============================================================
        # Keep the same node name as MPC waypoint_manager
        # so both can use the SAME waypoint_manager.yaml
        # ============================================================

        super().__init__('waypoint_manager')

        # ============================================================
        # Parameters
        #
        # waypoint_manager.yaml:
        #
        # waypoint_x:   [...]
        # waypoint_y:   [...]
        # waypoint_yaw: [...]
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
        # State
        # ============================================================

        self.current_waypoint = 0

        self.goal_in_progress = False

        self.mission_completed = False

        self.mission_complete_sent = False

        # ============================================================
        # Goal topic
        #
        # This topic is mainly used by the logger so that MPC,
        # DWB and MPPI experiments have the same mission interface.
        #
        # Navigation itself is performed by NavigateToPose ACTION.
        # ============================================================

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # ============================================================
        # Mission complete topic
        # ============================================================

        self.mission_complete_pub = (
            self.create_publisher(
                Bool,
                '/navigation_nav2/mission_complete',
                10
            )
        )

        # ============================================================
        # NavigateToPose action client
        #
        # Same interface for DWB and MPPI.
        # The controller plugin is selected by nav2_params_*.yaml.
        # ============================================================

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        # ============================================================
        # Startup information
        # ============================================================

        self.get_logger().info(
            f'Nav2 Waypoint Manager started: '
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

        self.get_logger().info(
            'Waiting for NavigateToPose action server...'
        )

        # ============================================================
        # Wait until BT Navigator becomes active
        # ============================================================

        self.nav_client.wait_for_server()

        self.get_logger().info(
            'Connected to NavigateToPose action server'
        )

        # ============================================================
        # Start mission
        # ============================================================

        self.send_next_goal()

    # ================================================================
    # Yaw -> quaternion
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

    # ================================================================
    # Create PoseStamped
    # ================================================================

    def create_goal_pose(
        self,
        x,
        y,
        yaw
    ):

        goal_pose = PoseStamped()

        goal_pose.header.frame_id = 'map'

        goal_pose.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0

        qz, qw = self.yaw_to_quaternion(
            yaw
        )

        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        return goal_pose

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
    # Send next waypoint
    # ================================================================

    def send_next_goal(self):

        # ------------------------------------------------------------
        # Mission already completed
        # ------------------------------------------------------------

        if self.mission_completed:
            return

        # ------------------------------------------------------------
        # End of waypoint list
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

        # ------------------------------------------------------------
        # Prevent duplicate goals
        # ------------------------------------------------------------

        if self.goal_in_progress:
            return

        # ============================================================
        # Current waypoint
        # ============================================================

        x, y, yaw = (
            self.waypoints[
                self.current_waypoint
            ]
        )

        # ============================================================
        # Create goal pose
        # ============================================================

        goal_pose = self.create_goal_pose(
            x,
            y,
            yaw
        )

        # ============================================================
        # Publish /goal_pose
        #
        # Diagnostic / logger interface.
        # This does NOT control Nav2 directly.
        # ============================================================

        self.goal_pub.publish(
            goal_pose
        )

        # ============================================================
        # NavigateToPose goal
        # ============================================================

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'Sending waypoint '
            f'{self.current_waypoint + 1}/'
            f'{len(self.waypoints)}: '
            f'x={x:.2f}, '
            f'y={y:.2f}, '
            f'yaw={yaw:.2f}'
        )

        self.goal_in_progress = True

        future = (
            self.nav_client.send_goal_async(
                goal_msg
            )
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    # ================================================================
    # Goal response
    # ================================================================

    def goal_response_callback(
        self,
        future
    ):

        try:

            goal_handle = (
                future.result()
            )

        except Exception as exc:

            self.goal_in_progress = False

            self.get_logger().error(
                f'Failed to send NavigateToPose goal: '
                f'{exc}'
            )

            return

        # ------------------------------------------------------------
        # Goal rejected
        # ------------------------------------------------------------

        if not goal_handle.accepted:

            self.goal_in_progress = False

            self.get_logger().error(
                f'Waypoint '
                f'{self.current_waypoint + 1} '
                f'was rejected'
            )

            return

        self.get_logger().info(
            f'Waypoint '
            f'{self.current_waypoint + 1} '
            f'accepted'
        )

        # ============================================================
        # Wait asynchronously for navigation result
        # ============================================================

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.goal_result_callback
        )

    # ================================================================
    # Goal result
    # ================================================================

    def goal_result_callback(
        self,
        future
    ):

        try:

            wrapped_result = (
                future.result()
            )

        except Exception as exc:

            self.goal_in_progress = False

            self.get_logger().error(
                f'NavigateToPose result exception: '
                f'{exc}'
            )

            return

        status = (
            wrapped_result.status
        )

        self.goal_in_progress = False

        # ============================================================
        # Goal succeeded
        # ============================================================

        if (
            status
            ==
            GoalStatus.STATUS_SUCCEEDED
        ):

            self.get_logger().info(
                f'Reached waypoint '
                f'{self.current_waypoint + 1}/'
                f'{len(self.waypoints)}'
            )

            # --------------------------------------------------------
            # Advance waypoint
            # --------------------------------------------------------

            self.current_waypoint += 1

            # --------------------------------------------------------
            # Final waypoint?
            # --------------------------------------------------------

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

            # --------------------------------------------------------
            # Send next waypoint
            # --------------------------------------------------------

            self.send_next_goal()

            return

        # ============================================================
        # Goal canceled
        # ============================================================

        if (
            status
            ==
            GoalStatus.STATUS_CANCELED
        ):

            self.get_logger().warn(
                f'Waypoint '
                f'{self.current_waypoint + 1} '
                f'was canceled'
            )

            return

        # ============================================================
        # Goal failed / aborted
        # ============================================================

        if (
            status
            ==
            GoalStatus.STATUS_ABORTED
        ):

            self.get_logger().error(
                f'Waypoint '
                f'{self.current_waypoint + 1} '
                f'was aborted'
            )

            return

        # ============================================================
        # Unknown status
        # ============================================================

        self.get_logger().error(
            f'Navigation failed at waypoint '
            f'{self.current_waypoint + 1}: '
            f'status={status}'
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = WaypointManagerNav2()

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