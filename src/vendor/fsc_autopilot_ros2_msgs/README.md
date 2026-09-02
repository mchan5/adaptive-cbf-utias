## ROS2 Fsc_Autopilot Message Definition 
 This package contains all the ros2 message definitions
 
### Use the following command to compile the message.
- Remove if previously built by the standard procedure:

```
rm -rf build/fsc_autopilot_ros2_msgs/ament_cmake_python/fsc_autopilot_ros2_msgs/fsc_autopilot_ros2_msgs
```

- Use the following command to build the message:

```
colcon build --packages-select fsc_autopilot_ros2_msgs --symlink-install
```

- Source the workspace (If the workspace is not sourced in ~/.bashrc):

```
source ~/source/fsc_autopilot2_ws/install/setup.bash
```

- Check message visibility:

```
ros2 interface list | grep fsc_autopilot_ros2_msgs
```

If the messages are built correctly, you should able to see the following:

```
<your_ubuntu_name>@your_device_name:~/source/fsc_autopilot2_ws$ ros2 interface list | grep fsc_autopilot_ros2_msgs
    fsc_autopilot_ros2_msgs/msg/AttitudeControllerState
    fsc_autopilot_ros2_msgs/msg/PositionControllerReference
    fsc_autopilot_ros2_msgs/msg/PositionControllerState
    fsc_autopilot_ros2_msgs/msg/UDEState
```
