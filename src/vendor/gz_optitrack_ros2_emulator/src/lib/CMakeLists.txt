set(GZ_LIB_NAME gz_optitrack_lib)

add_library(${GZ_LIB_NAME} STATIC gz_optitrack_emulator.cpp)

target_include_directories(${GZ_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${GZ_LIB_NAME}
  gz-transport${GZ_TRANSPORT_VERSION}::gz-transport${GZ_TRANSPORT_VERSION}
  gz-msgs${GZ_MSG_VERSION}::gz-msgs${GZ_MSG_VERSION}
  ${cpp_typesupport_target}
  Eigen3::Eigen
)

ament_target_dependencies(${GZ_LIB_NAME}
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  fsc_autopilot_ros2_msgs
)