import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    ExecuteProcess
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():

    pkg_path = get_package_share_directory("agv0509test6")
    gazebo_ros_path = get_package_share_directory("gazebo_ros")

    # ============================================================
    # Paths
    # ============================================================

    # world = os.path.join(
    #     pkg_path,
    #     "worlds",
    #     "maze"
    # )

    world = os.path.join(
        '/home/pdhuy/agv_ws/src/agv0509test6/worlds/maze_no_obs'
    )

    urdf_file = os.path.join(
        pkg_path,
        "urdf",
        "agv0509test6.urdf"
    )

    gazebo_params = os.path.join(
        pkg_path,
        "config",
        "gazebo_params.yaml"
    )

    scan_filter_file = os.path.join(
        pkg_path,
        "config",
        "scan_filter.yaml"
    )

    ekf_file = os.path.join(
        pkg_path,
        "config",
        "ekf.yaml"
    )

    # ============================================================
    # Robot description
    # ============================================================

    with open(urdf_file, "r", encoding="utf-8") as f:
        robot_description = f.read()

    if robot_description.lstrip().startswith("<?xml"):
        robot_description = robot_description.split("?>", 1)[1].lstrip()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True
            }
        ]
    )

    # ============================================================
    # Gazebo server
    # ============================================================

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros_path,
                "launch",
                "gzserver.launch.py"
            )
        ),
        launch_arguments={
            "world": world,
            "verbose": "true",
            "params_file": gazebo_params
        }.items()
    )

    # ============================================================
    # Gazebo GUI
    # ============================================================

    gzclient = ExecuteProcess(
        cmd=["gzclient"],
        output="screen"
    )

    # ============================================================
    # Spawn robot
    # ============================================================

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "agv",
            "-topic", "robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.001"
        ],
        output="screen"
    )

    # ============================================================
    # ROS2 controllers
    # ============================================================

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ],
        output="screen"
    )

    diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--controller-manager",
            "/controller_manager"
        ],
        output="screen"
    )

    # ============================================================
    # Laser filter
    # ============================================================

    laser_filter = Node(
        package="laser_filters",
        executable="scan_to_scan_filter_chain",
        name="laser_filter",
        output="screen",
        parameters=[
            scan_filter_file,
            {"use_sim_time": True}
        ],
        remappings=[
            ("scan", "/scan"),
            ("scan_filtered", "/scan_filtered")
        ]
    )

    # ============================================================
    # EKF
    # ============================================================

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            ekf_file,
            {'use_sim_time': True}
        ]
    )

    # ============================================================
    # Startup sequence
    # ============================================================

    start_joint_state_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster]
        )
    )

    start_diff_drive_after_joint_state = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[diff_drive_controller]
        )
    )

    start_ekf_after_diff_drive = RegisterEventHandler(
        OnProcessExit(
            target_action=diff_drive_controller,
            on_exit=[ekf_node]
        )
    )

    # ============================================================
    # Launch
    # ============================================================

    return LaunchDescription([
        gzserver,
        gzclient,

        robot_state_publisher,

        laser_filter,

        spawn_robot,

        start_joint_state_after_spawn,
        start_diff_drive_after_joint_state,
        start_ekf_after_diff_drive
    ])