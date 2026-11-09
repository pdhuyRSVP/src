import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    pkg_path = get_package_share_directory("agv0509test6")

    params = os.path.join(
        pkg_path,
        "config",
        "nav2_params_mppi.yaml"
    )

    map_yaml = os.path.join(
        pkg_path,
        "maps",
        "map.yaml"
    )

    ###################################################
    # Map Server
    ###################################################

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            params,
            {
                "yaml_filename": map_yaml
            }
        ]
    )

    ###################################################
    # AMCL
    ###################################################

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[params]
    )

    ###################################################
    # Planner
    ###################################################

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[params]
    )

    ###################################################
    # Controller
    ###################################################

    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[params],
        remappings=[
            ("cmd_vel",
             "/diff_drive_controller/cmd_vel_unstamped")
        ]
    )
    ###################################################
    # Controller
    ###################################################

    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[params]
    )

    ###################################################
    # BT Navigator
    ###################################################

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[params]
    )

    ###################################################
    # Lifecycle
    ###################################################

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "autostart": True,
                "bond_timeout": 4.0,
                "node_names": [
                    "map_server",
                    "amcl",
                    "planner_server",
                    "controller_server",
                    "behavior_server",
                    "bt_navigator"
                ]
            }
        ]
    )

    return LaunchDescription([

        map_server,

        amcl,

        planner_server,

        controller_server,

        behavior_server,

        bt_navigator,

        lifecycle_manager

    ])