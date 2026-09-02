cmake_minimum_required(VERSION 3.8)
project(gz_optitrack_ros2_emulator)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# --- change the version of the following based on the version of ROS and ubuntu --- #
set(GZ_TRANSPORT_VERSION 13)
set(GZ_MSG_VERSION 10)
# --- #
find_package(gz-transport${GZ_TRANSPORT_VERSION} REQUIRED)
find_package(gz-msgs${GZ_MSG_VERSION} REQUIRED)

# System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

# fsc custom message
find_package(fsc_autopilot_ros2_msgs REQUIRED)

set(_ob_deps
    ament_cmake
    rclcpp
    std_msgs
    geometry_msgs
    tf2_ros
    tf2_eigen
    nav_msgs
    fsc_autopilot_ros2_msgs
    gz-msgs${GZ_MSG_VERSION}
    rosidl_default_generators)

# export
# ament_export_dependencies(ament_cmake rclcpp rosidl_default_runtime)

# export
ament_export_dependencies(${_ob_deps})

# include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add source files
add_subdirectory(src)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(FILES package.xml DESTINATION share/${PROJECT_NAME})

# install launch file
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# install parameter file
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
