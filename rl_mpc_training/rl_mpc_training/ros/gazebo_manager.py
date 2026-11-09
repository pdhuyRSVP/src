#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import (
    GetEntityState,
    SetEntityState
)

from std_srvs.srv import Empty

from gazebo_msgs.msg import EntityState


class GazeboManager(Node):

    def __init__(self):

        super().__init__('rl_gazebo_manager')

        # ====================================================
        # Parameters
        # ====================================================

        self.declare_parameter(
            'entity_name',
            'agv'
        )

        self.entity_name = self.get_parameter(
            'entity_name'
        ).value

        # ====================================================
        # Gazebo service clients
        # ====================================================

        self.get_state_client = self.create_client(
            GetEntityState,
            '/get_entity_state'
        )

        self.set_state_client = self.create_client(
            SetEntityState,
            '/set_entity_state'
        )

        self.pause_client = self.create_client(
            Empty,
            '/pause_physics'
        )

        self.unpause_client = self.create_client(
            Empty,
            '/unpause_physics'
        )

        # ====================================================
        # Internal states
        # ====================================================

        self.pending_get_state = None

        self.services_reported = False

        # Store the latest entity state returned by Gazebo
        self.last_entity_state = None

        # Track pending SetEntityState request
        self.pending_set_state = None

        # ====================================================
        # Diagnostic timer
        # ====================================================

        self.timer = self.create_timer(
            2.0,
            self.check_gazebo
        )

        self.get_logger().info(
            'Gazebo Manager Started'
        )

        self.get_logger().info(
            f'Target entity: {self.entity_name}'
        )

    # ========================================================
    # Check Gazebo services
    # ========================================================

    def check_services(self):

        services = {
            '/get_entity_state': self.get_state_client,
            '/set_entity_state': self.set_state_client,
            '/pause_physics': self.pause_client,
            '/unpause_physics': self.unpause_client
        }

        all_ready = True

        for name, client in services.items():

            if not client.service_is_ready():

                all_ready = False

        if all_ready and not self.services_reported:

            self.get_logger().info(
                'All required Gazebo services are available'
            )

            self.services_reported = True

        return all_ready

    # ========================================================
    # Request robot state
    # ========================================================

    def request_robot_state(self):

        if not self.get_state_client.service_is_ready():

            self.get_logger().warn(
                'GetEntityState service is not available'
            )

            return

        # Avoid overlapping requests
        if self.pending_get_state is not None:

            if not self.pending_get_state.done():
                return

        request = GetEntityState.Request()

        request.name = self.entity_name

        request.reference_frame = 'world'

        self.pending_get_state = (
            self.get_state_client.call_async(request)
        )

        self.pending_get_state.add_done_callback(
            self.handle_robot_state
        )

    # ========================================================
    # Process returned robot state
    # ========================================================

    def handle_robot_state(self, future):

        self.pending_get_state = None

        try:

            response = future.result()

        except Exception as exc:

            self.get_logger().error(
                f'GetEntityState exception: {exc}'
            )

            return

        if not response.success:

            self.get_logger().warn(
                'Failed to get Gazebo entity state'
            )

            return

        # Save the latest valid Gazebo entity state
        self.last_entity_state = response.state

        state = response.state

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        x = float(state.pose.position.x)
        y = float(state.pose.position.y)
        z = float(state.pose.position.z)

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        q = state.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z + q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y + q.z * q.z
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        # ----------------------------------------------------
        # World-frame velocity
        # ----------------------------------------------------

        vx = float(state.twist.linear.x)
        vy = float(state.twist.linear.y)

        wz = float(state.twist.angular.z)

        speed_xy = math.hypot(vx, vy)

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        self.get_logger().info(
            f'Gazebo entity={self.entity_name} | '
            f'x={x:.3f}, '
            f'y={y:.3f}, '
            f'z={z:.3f}, '
            f'yaw={yaw:.3f} rad | '
            f'speed_xy={speed_xy:.3f} m/s, '
            f'wz={wz:.3f} rad/s'
        )

    # ================================================================
    # Request Gazebo robot state reset
    # ================================================================

    def request_set_robot_state(self, x, y, yaw, z=None):

        # ------------------------------------------------------------
        # 1. Check service availability
        # ------------------------------------------------------------

        if not self.set_state_client.service_is_ready():

            self.get_logger().error(
                'SetEntityState service is not available'
            )

            return False

        # ------------------------------------------------------------
        # 2. Avoid overlapping requests
        # ------------------------------------------------------------

        if self.pending_set_state is not None:

            if not self.pending_set_state.done():

                self.get_logger().warn(
                    'Previous SetEntityState request is pending'
                )

                return False

        # ------------------------------------------------------------
        # 3. Validate target position
        # ------------------------------------------------------------

        x = float(x)
        y = float(y)
        yaw = float(yaw)

        if not all(math.isfinite(v) for v in (x, y, yaw)):

            self.get_logger().error(
                'Invalid target robot position or yaw'
            )

            return False

        # ------------------------------------------------------------
        # 4. Determine target height
        # ------------------------------------------------------------

        if z is None:

            if self.last_entity_state is None:

                self.get_logger().error(
                    'Robot state unavailable: cannot determine z'
                )

                return False

            z = float(self.last_entity_state.pose.position.z)

        else:

            z = float(z)

        if not math.isfinite(z):

            self.get_logger().error(
                'Invalid robot height'
            )

            return False

        # ------------------------------------------------------------
        # 5. Create SetEntityState request
        # ------------------------------------------------------------

        request = SetEntityState.Request()

        state = EntityState()

        state.name = self.entity_name

        state.reference_frame = 'world'

        # Position
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z

        # Orientation: yaw -> quaternion
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0

        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)

        # Reset linear velocity
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0

        # Reset angular velocity
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        request.state = state

        # ------------------------------------------------------------
        # 6. Send asynchronous request
        # ------------------------------------------------------------

        self.pending_set_state = (
            self.set_state_client.call_async(request)
        )

        self.pending_set_state.add_done_callback(
            self.handle_set_robot_state
        )

        self.get_logger().info(
            f'SetEntityState requested: '
            f'x={x:.3f}, '
            f'y={y:.3f}, '
            f'z={z:.3f}, '
            f'yaw={yaw:.3f}'
        )

        return True

    # ================================================================
    # Handle Gazebo SetEntityState response
    # ================================================================

    def handle_set_robot_state(self, future):

        self.pending_set_state = None

        try:

            response = future.result()

        except Exception as exc:

            self.get_logger().error(
                f'SetEntityState exception: {exc}'
            )

            return

        if not response.success:

            self.get_logger().error(
                'Gazebo SetEntityState failed'
            )

            return

        self.get_logger().info(
            'Gazebo SetEntityState succeeded'
        )

        # Request the actual state after setting
        self.request_robot_state()

    # ========================================================
    # Periodic diagnostic
    # ========================================================

    def check_gazebo(self):

        self.check_services()

        self.request_robot_state()


def main(args=None):

    rclpy.init(args=args)

    node = GazeboManager()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
