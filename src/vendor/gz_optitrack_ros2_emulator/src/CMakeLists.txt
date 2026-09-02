add_subdirectory(lib)

add_executable(gz_optitrack_emulator_node
               gz_optitrack_emulator_node.cpp)
target_link_libraries(gz_optitrack_emulator_node
  gz_optitrack_lib
)

ament_target_dependencies(gz_optitrack_emulator_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
)

install(TARGETS gz_optitrack_emulator_node DESTINATION lib/${PROJECT_NAME})