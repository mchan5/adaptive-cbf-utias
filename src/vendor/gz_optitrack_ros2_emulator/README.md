# ROS2 PX4 SITL Gazebo OptiTrack Emulator
## A ROS2 package served as the OptiTrack Emulator
- Convert gz odometry message to ROS2 mocap message.
- Use gz_transport to facilitate Gazebo message subscription.

## Add plugin to the gazebo .sdf file to enable custom odometry publishing
### Step 1: Obtain the groundtruth from PX4 SITL ROS2 Gazebo simulion
- Read the following [instruction](docs/px4_sitl_groundtruth.md) to learn how to publish simulation groundtruth.

## Install and use the package
### Step 1: pull the source code to the src directory of your workspace
```
$ cd <path_to_workspace>/src

$ git clone https://github.com/FSC-Lab/gz_optitrack_ros2_emulator.git
```
### Step 2: Determine the gz-transport and gz-msg version
```
$ dpkg -l | grep gz-transport

$ dpkg -l | grep gz-msgs
```
- (you could consult Chatgpt for the command)
- Change the version in the root CMakeLists.txt accordingly. For example, for gz-transport13 and gz-msg10, the value in the CMakeLists.txt should look follows:
```
# --- change the version of the following based on the version of ROS and ubuntu --- #
set(GZ_TRANSPORT_VERSION 13)
set(GZ_MSG_VERSION 10)
```

### Step 3: Build the node
```
$ colcon build --packages-select gz_optitrack_ros2_emulator
```
- Source ``<workspace_name>`` by adding the following to the ``~/.bashrc`` or ``~/.zhrc``: 

```
source ~/path_to_workspace/<workspace_name>/install/local_setup.bash
```

### Step 4: Set the model name
- The model names are listed in the ``config/params.yaml`` under the name of ``gz_model_list``. Read the following [instruction](docs/px4_sitl_groundtruth.md), make sure that the model name match the gz topic of the odometry plugin.
- The convention of the package is to subscribe to the gz topic:
``model/<model_name>/groundtruth_odometry``
 and publish to the ros topic: ``/mocap/<model_name>``
- IMPORTANT: everytime the parameter file is changed, you need to build the node so that the updated parameter file takes effect.

### Step 5: Launch the node:

```
$ ros2 launch gz_optitrack_ros2_emulator emulator_for_gazebo_launch.py
```


## The off-board autopilot code:
- The repository of the fsc-autopilot is [https://github.com/FSC-Lab/fsc_autopilot](https://github.com/FSC-Lab/fsc_autopilot)