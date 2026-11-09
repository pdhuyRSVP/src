import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():

    pkg_path = get_package_share_directory("agv0509test6")

    # ============================================================
    # Paths
    # ============================================================

    params_file = os.path.join(
        pkg_path,
        "config",
        "nav2_params_mpc.yaml"
    )

    map_yaml = os.path.join(
        pkg_path,
        "maps",
        "map.yaml"
    )

    # ============================================================
    # Map Server
    # ============================================================

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            params_file,
            {
                "yaml_filename": map_yaml
            }
        ]
    )

    # ============================================================
    # AMCL
    # ============================================================

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            params_file,
            {'use_sim_time': True}
        ]
    )

    # ============================================================
    # Planner Server
    # ============================================================

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            params_file
        ]
    )

    # ============================================================
    # Smoother Server
    # ============================================================
    
    smoother_server = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        output='screen',
        parameters=[params_file]
    )


    # ============================================================
    # Lifecycle Manager
    # ============================================================

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_mpc_navigation",
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
                    "smoother_server"
                ]
            }
        ]
    )

    # ============================================================
    # Launch
    # ============================================================

    return LaunchDescription([
        map_server,
        amcl,
        planner_server,
        smoother_server,
        lifecycle_manager
    ])