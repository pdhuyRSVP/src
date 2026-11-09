import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus

from nav2_msgs.action import ComputePathToPose, SmoothPath
from nav2_msgs.srv import IsPathValid


class GoalManager(Node):

    def __init__(self):

        super().__init__('goal_manager')

        # ============================================================
        # Parameters
        # ============================================================

        # Chỉ dùng để GIÁM SÁT điều kiện replan.
        # Không replan theo chu kỳ.
        self.declare_parameter('monitor_period', 0.5)

        # Nếu robot lệch global path quá xa -> replan
        self.declare_parameter('path_deviation_threshold', 0.45)

        # Tránh replan liên tục khi obstacle/costmap dao động
        self.declare_parameter('replan_cooldown', 1.0)

        # Path phải invalid liên tiếp N lần mới replan
        self.declare_parameter('invalid_checks_required', 2)

        # Smoother plugin đã cấu hình trong smoother_server
        self.declare_parameter('smoother_id', 'simple_smoother')

        # Thời gian tối đa cho mỗi lần smoothing
        self.declare_parameter('smoothing_max_duration', 2.0)

        # Kiểm tra collision sau khi smoothing
        self.declare_parameter('check_smoother_collisions', True)

        self.monitor_period = (
            self.get_parameter('monitor_period')
            .get_parameter_value()
            .double_value
        )

        self.path_deviation_threshold = (
            self.get_parameter('path_deviation_threshold')
            .get_parameter_value()
            .double_value
        )

        self.replan_cooldown = (
            self.get_parameter('replan_cooldown')
            .get_parameter_value()
            .double_value
        )

        self.invalid_checks_required = (
            self.get_parameter('invalid_checks_required')
            .get_parameter_value()
            .integer_value
        )

        self.smoother_id = (
            self.get_parameter('smoother_id')
            .get_parameter_value()
            .string_value
        )

        self.smoothing_max_duration = (
            self.get_parameter('smoothing_max_duration')
            .get_parameter_value()
            .double_value
        )

        self.check_smoother_collisions = (
            self.get_parameter('check_smoother_collisions')
            .get_parameter_value()
            .bool_value
        )

        # ============================================================
        # Subscribers
        # ============================================================

        # Goal từ RViz
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # Pose robot trong frame map từ AMCL
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        # Raw path từ Planner Server.
        # Chỉ giữ lại để debug/fallback, KHÔNG dùng trực tiếp cho MPC.
        self.raw_path_sub = self.create_subscription(
            Path,
            '/plan',
            self.raw_path_callback,
            10
        )

        # ============================================================
        # Nav2 interfaces
        # ============================================================

        self.planner_client = ActionClient(
            self,
            ComputePathToPose,
            '/compute_path_to_pose'
        )

        self.smoother_client = ActionClient(
            self,
            SmoothPath,
            '/smooth_path'
        )

        # Chỉ dùng khi Smoother Server lỗi / từ chối.
        # Bình thường Smoother Server tự publish kết quả trên /plan_smoothed.
        self.fallback_path_pub = self.create_publisher(
            Path,
            '/plan_smoothed',
            10
        )

        self.path_valid_client = self.create_client(
            IsPathValid,
            '/is_path_valid'
        )

        # ============================================================
        # Internal states
        # ============================================================

        self.current_goal = None

        # Raw NavFn path và smoothed path được tách riêng.
        self.raw_path = None
        self.current_path = None

        self.current_pose = None

        # planning_in_progress bao phủ toàn bộ pipeline:
        # ComputePathToPose -> SmoothPath.
        self.planning_in_progress = False
        self.smoothing_in_progress = False

        self.path_check_in_progress = False

        self.smoother_goal_handle = None

        self.invalid_path_count = 0

        self.last_replan_time = -float('inf')

        # ID để bỏ qua kết quả cũ nếu có goal mới
        self.request_id = 0
        self.active_request_id = 0

        # Nếu đang planning mà xuất hiện trigger mới
        self.pending_replan_reason = None

        # ============================================================
        # Monitor
        # ============================================================

        # Timer này KHÔNG replan theo chu kỳ.
        # Nó chỉ kiểm tra các điều kiện kích hoạt replan.
        self.monitor_timer = self.create_timer(
            self.monitor_period,
            self.monitor_callback
        )

        self.get_logger().info(
            'Goal Manager started - NavFn -> SimpleSmoother -> MPC pipeline enabled'
        )

    # ================================================================
    # Goal
    # ================================================================

    def goal_callback(self, msg):

        self.get_logger().info(
            f'New goal received: '
            f'x={msg.pose.position.x:.2f}, '
            f'y={msg.pose.position.y:.2f}'
        )

        self.current_goal = msg

        # Path cũ không còn liên quan tới goal mới
        self.raw_path = None
        self.current_path = None

        self.invalid_path_count = 0

        # Nếu smoother của goal cũ còn chạy thì yêu cầu hủy.
        if self.smoother_goal_handle is not None:

            try:
                self.smoother_goal_handle.cancel_goal_async()
            except Exception:
                pass

            self.smoother_goal_handle = None

        self.smoothing_in_progress = False

        # Goal mới phải được ưu tiên ngay lập tức
        self.send_planning_request(
            reason='new goal',
            force=True
        )

    # ================================================================
    # AMCL pose
    # ================================================================

    def pose_callback(self, msg):

        self.current_pose = msg.pose.pose

    # ================================================================
    # Raw global path from Planner Server
    # ================================================================

    def raw_path_callback(self, msg):

        if len(msg.poses) == 0:
            return

        # /plan chỉ là đường raw để debug/fallback.
        # current_path chỉ được cập nhật sau bước smoothing.
        self.raw_path = msg

    # ================================================================
    # Monitor conditions
    # ================================================================

    def monitor_callback(self):

        if self.current_goal is None:
            return

        # ------------------------------------------------------------
        # Case 1:
        # Có goal nhưng chưa có path.
        # Ví dụ goal đến ngay lúc Planner Server chưa ready.
        # ------------------------------------------------------------

        if self.current_path is None:

            if not self.planning_in_progress:
                self.request_replan('no valid global path')

            return

        # ------------------------------------------------------------
        # Case 2:
        # Robot lệch khỏi global path quá xa
        # ------------------------------------------------------------

        if self.current_pose is not None:

            deviation = self.distance_to_path()

            if (
                deviation is not None
                and deviation > self.path_deviation_threshold
            ):

                self.request_replan(
                    f'robot deviated from path '
                    f'({deviation:.2f} m)'
                )

                return

        # ------------------------------------------------------------
        # Case 3:
        # Kiểm tra path có bị obstacle chặn hay không
        # ------------------------------------------------------------

        self.check_path_validity()

    # ================================================================
    # Distance robot -> current path
    # ================================================================

    def distance_to_path(self):

        if self.current_path is None:
            return None

        if self.current_pose is None:
            return None

        if len(self.current_path.poses) == 0:
            return None

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        min_distance = float('inf')

        for pose_stamped in self.current_path.poses:

            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y

            distance = math.hypot(
                px - robot_x,
                py - robot_y
            )

            if distance < min_distance:
                min_distance = distance

        return min_distance

    # ================================================================
    # Path validity
    # ================================================================

    def check_path_validity(self):

        if self.current_path is None:
            return

        if self.path_check_in_progress:
            return

        if not self.path_valid_client.service_is_ready():
            return

        request = IsPathValid.Request()

        request.path = self.current_path

        self.path_check_in_progress = True

        future = self.path_valid_client.call_async(request)

        future.add_done_callback(
            self.path_validity_callback
        )

    def path_validity_callback(self, future):

        self.path_check_in_progress = False

        try:

            response = future.result()

        except Exception as exc:

            self.get_logger().warn(
                f'Failed to check path validity: {exc}'
            )

            return

        if response.is_valid:

            self.invalid_path_count = 0
            return

        self.invalid_path_count += 1

        self.get_logger().warn(
            f'Global path invalid '
            f'({self.invalid_path_count}/'
            f'{self.invalid_checks_required})'
        )

        if (
            self.invalid_path_count
            >= self.invalid_checks_required
        ):

            self.invalid_path_count = 0

            self.request_replan(
                'global path blocked or invalid'
            )

    # ================================================================
    # Replanning decision
    # ================================================================

    def request_replan(self, reason):

        if self.current_goal is None:
            return

        # Nếu planner đang chạy thì không gửi chồng request
        if self.planning_in_progress:

            self.pending_replan_reason = reason
            return

        now = time.monotonic()

        if (
            now - self.last_replan_time
            < self.replan_cooldown
        ):
            return

        self.send_planning_request(
            reason=reason,
            force=False
        )

    # ================================================================
    # Planning request
    # ================================================================

    def send_planning_request(
        self,
        reason='unknown',
        force=False
    ):

        if self.current_goal is None:
            return

        if (
            self.planning_in_progress
            and not force
        ):
            return

        if not self.planner_client.server_is_ready():

            self.get_logger().warn(
                'Planner Server is not ready'
            )

            return

        self.request_id += 1

        request_id = self.request_id

        self.active_request_id = request_id

        goal_msg = ComputePathToPose.Goal()

        goal_msg.goal = self.current_goal

        # Planner Server tự lấy pose robot hiện tại
        goal_msg.use_start = False

        self.planning_in_progress = True

        self.last_replan_time = time.monotonic()

        self.get_logger().info(
            f'Planning request #{request_id}: {reason}'
        )

        future = self.planner_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            lambda f:
            self.goal_response_callback(
                f,
                request_id,
                reason
            )
        )

    # ================================================================
    # Planner goal response
    # ================================================================

    def goal_response_callback(
        self,
        future,
        request_id,
        reason
    ):

        try:

            goal_handle = future.result()

        except Exception as exc:

            if request_id == self.active_request_id:
                self.planning_in_progress = False

            self.get_logger().error(
                f'Planning request failed: {exc}'
            )

            return

        # Request cũ sau khi goal mới đã đến
        if request_id != self.active_request_id:

            if goal_handle.accepted:
                goal_handle.cancel_goal_async()

            return

        if not goal_handle.accepted:

            self.planning_in_progress = False

            self.get_logger().error(
                f'Planning request #{request_id} rejected'
            )

            return

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            lambda f:
            self.get_result_callback(
                f,
                request_id,
                reason
            )
        )

    # ================================================================
    # Planner result
    # ================================================================

    def get_result_callback(
        self,
        future,
        request_id,
        reason
    ):

        # Ignore result from old request
        if request_id != self.active_request_id:
            return

        try:

            wrapped_result = future.result()

            result = wrapped_result.result

            raw_path = result.path

        except Exception as exc:

            self.planning_in_progress = False
            self.smoothing_in_progress = False

            self.get_logger().error(
                f'Failed to receive planner result: {exc}'
            )

            return

        if len(raw_path.poses) == 0:

            self.planning_in_progress = False
            self.smoothing_in_progress = False

            self.get_logger().error(
                f'Planner returned empty path '
                f'for request #{request_id}'
            )

            return

        self.raw_path = raw_path

        self.get_logger().info(
            f'Raw global path received: '
            f'{len(raw_path.poses)} poses '
            f'[{reason}]'
        )

        # Không đưa raw path trực tiếp cho MPC.
        # Chuyển sang bước SmoothPath.
        self.send_smoothing_request(
            raw_path,
            request_id,
            reason
        )

    # ================================================================
    # Smoothing request
    # ================================================================

    def send_smoothing_request(
        self,
        raw_path,
        request_id,
        reason
    ):

        # Goal mới đã xuất hiện trong lúc chờ planner
        if request_id != self.active_request_id:
            return

        if not self.smoother_client.server_is_ready():

            self.get_logger().warn(
                'Smoother Server is not ready - '
                'using raw path as fallback'
            )

            self.use_raw_path_fallback(
                raw_path,
                request_id,
                reason,
                'smoother server not ready'
            )

            return

        goal_msg = SmoothPath.Goal()

        goal_msg.path = raw_path

        goal_msg.smoother_id = self.smoother_id

        goal_msg.max_smoothing_duration = (
            Duration(
                seconds=self.smoothing_max_duration
            ).to_msg()
        )

        goal_msg.check_for_collisions = (
            self.check_smoother_collisions
        )

        self.smoothing_in_progress = True

        self.get_logger().info(
            f'Smoothing request #{request_id}: '
            f'{len(raw_path.poses)} raw poses, '
            f'plugin={self.smoother_id}'
        )

        future = self.smoother_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            lambda f:
            self.smoother_goal_response_callback(
                f,
                request_id,
                reason,
                raw_path
            )
        )

    # ================================================================
    # Smoother goal response
    # ================================================================

    def smoother_goal_response_callback(
        self,
        future,
        request_id,
        reason,
        raw_path
    ):

        try:

            goal_handle = future.result()

        except Exception as exc:

            if request_id == self.active_request_id:

                self.get_logger().warn(
                    f'Smoothing request failed: {exc}. '
                    f'Using raw path as fallback.'
                )

                self.use_raw_path_fallback(
                    raw_path,
                    request_id,
                    reason,
                    'smoothing request exception'
                )

            return

        # Smoothing request thuộc goal cũ
        if request_id != self.active_request_id:

            if goal_handle.accepted:
                goal_handle.cancel_goal_async()

            return

        if not goal_handle.accepted:

            self.get_logger().warn(
                f'Smoothing request #{request_id} rejected. '
                f'Using raw path as fallback.'
            )

            self.use_raw_path_fallback(
                raw_path,
                request_id,
                reason,
                'smoothing request rejected'
            )

            return

        self.smoother_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            lambda f:
            self.smoother_result_callback(
                f,
                request_id,
                reason,
                raw_path
            )
        )

    # ================================================================
    # Smoother result
    # ================================================================

    def smoother_result_callback(
        self,
        future,
        request_id,
        reason,
        raw_path
    ):

        # Result cũ không được phép cập nhật current_path.
        if request_id != self.active_request_id:
            return

        self.smoother_goal_handle = None

        try:

            wrapped_result = future.result()

            result = wrapped_result.result

            # Chỉ chấp nhận path nếu SmoothPath action thực sự SUCCEEDED.
            # Nếu collision checking làm action abort thì không dùng
            # smoothed path đó cho current_path.
            if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:

                error_code = getattr(
                    result,
                    'error_code',
                    'unknown'
                )

                self.get_logger().warn(
                    f'Smoothing action did not succeed '
                    f'(status={wrapped_result.status}, '
                    f'error_code={error_code}). '
                    f'Using raw path as fallback.'
                )

                self.use_raw_path_fallback(
                    raw_path,
                    request_id,
                    reason,
                    'smoothing action not successful'
                )

                return

            smoothed_path = result.path

        except Exception as exc:

            self.get_logger().warn(
                f'Failed to receive smoother result: {exc}. '
                f'Using raw path as fallback.'
            )

            self.use_raw_path_fallback(
                raw_path,
                request_id,
                reason,
                'failed to receive smoother result'
            )

            return

        if len(smoothed_path.poses) == 0:

            self.get_logger().warn(
                f'Smoother returned empty path for '
                f'request #{request_id}. '
                f'Using raw path as fallback.'
            )

            self.use_raw_path_fallback(
                raw_path,
                request_id,
                reason,
                'empty smoothed path'
            )

            return

        # ------------------------------------------------------------
        # Đây mới là path chính thức dùng cho:
        #   1) MPC tracking
        #   2) path deviation monitoring
        #   3) IsPathValid
        #
        # Smoother Server tự publish cùng path này trên /plan_smoothed.
        # ------------------------------------------------------------

        self.current_path = smoothed_path

        self.invalid_path_count = 0

        self.smoothing_in_progress = False
        self.planning_in_progress = False

        was_completed = bool(
            getattr(
                result,
                'was_completed',
                True
            )
        )

        if was_completed:

            self.get_logger().info(
                f'Smoothed global path ready: '
                f'{len(smoothed_path.poses)} poses '
                f'[{reason}]'
            )

        else:

            self.get_logger().warn(
                f'Smoother reached its time limit but returned '
                f'a usable path with '
                f'{len(smoothed_path.poses)} poses '
                f'[{reason}]'
            )

        self.handle_pending_replan()

    # ================================================================
    # Raw-path fallback
    # ================================================================

    def use_raw_path_fallback(
        self,
        raw_path,
        request_id,
        reason,
        fallback_reason
    ):

        if request_id != self.active_request_id:
            return

        self.current_path = raw_path

        self.invalid_path_count = 0

        self.smoothing_in_progress = False
        self.planning_in_progress = False

        # MPC đang subscribe /plan_smoothed.
        # Khi smoother lỗi, GoalManager publish raw path lên cùng topic
        # để navigation vẫn tiếp tục thay vì treo mission.
        self.fallback_path_pub.publish(
            raw_path
        )

        self.get_logger().warn(
            f'Using RAW path fallback: '
            f'{len(raw_path.poses)} poses '
            f'[{reason}; {fallback_reason}]'
        )

        self.handle_pending_replan()

    # ================================================================
    # Pending trigger after planning/smoothing pipeline
    # ================================================================

    def handle_pending_replan(self):

        if self.pending_replan_reason is None:
            return

        pending_reason = self.pending_replan_reason

        self.pending_replan_reason = None

        # Không replan ngay lập tức vì path vừa được cập nhật.
        # Monitor cycle kế tiếp sẽ đánh giá lại deviation/path validity
        # trên chính smoothed path mới.
        self.get_logger().debug(
            f'Pending replan trigger will be re-evaluated: '
            f'{pending_reason}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = GoalManager()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()