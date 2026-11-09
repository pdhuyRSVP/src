import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():

    pkg_path = get_package_share_directory("agv0509test6")

    urdf_file = os.path.join(
        pkg_path,
        "urdf",
        "agv0509test6.urdf"
    )

    with open(urdf_file, "r", encoding="utf-8") as f:
        robot_description = f.read()
    if robot_description.lstrip().startswith("<?xml"):
        robot_description = robot_description.split("?>", 1)[1].lstrip()

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen"
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen"
    )

    return LaunchDescription([

        joint_state_publisher_gui,

        rviz

    ])