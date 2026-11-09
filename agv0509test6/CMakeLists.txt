cmake_minimum_required(VERSION 3.8)
project(agv0509test6)

find_package(ament_cmake REQUIRED)

# Cài đặt các tài nguyên chia sẻ (không cần tồn tại hết; nếu thiếu CMake vẫn cấu hình OK)
install(
  DIRECTORY
    config
    launch
    meshes
    urdf
    worlds
    maps
  DESTINATION share/${PROJECT_NAME}
)

# Nếu sau này có script Python, bỏ comment và thêm vào đây:
install(PROGRAMS
  scripts/waypoint_manager_nav2.py
  scripts/logger_node_nav2.py
  scripts/logger_node_mpc.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
