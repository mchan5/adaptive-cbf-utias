# Vendored third-party packages

Pinned working-tree copies — not submodules. Never edit these in place.
To update: replace the tree from upstream at a new commit and note it here.

| Package | Upstream | Commit at vendor time |
|---|---|---|
| fsc_autopilot_ros2_msgs | https://github.com/FSC-Lab/fsc_autopilot_ros2_msgs.git | `35db1f6a59847cda9b08c2a623444a016d7f2571` |
| livox_ros_driver2 | https://github.com/Livox-SDK/livox_ros_driver2.git | `4a1def929e5b59c7a8122d19fce6efba581ce9f7` |
| natnet_ros2 | https://github.com/L2S-lab/natnet_ros2.git | `883b09518196c2f7cdd6aecd552d3c85c78f37e7` |
| gz_optitrack_ros2_emulator | https://github.com/FSC-Lab/gz_optitrack_ros2_emulator.git | `150bef0b9f8d630f32777db89bc85c24f8fb0dda` |
| px4_msgs | https://github.com/PX4/px4_msgs (branch `release/1.15`) | `a1045ec4feb6d709bdecaf3895f1d5b43a5dabb8` |
| px4_ros_com | https://github.com/PX4/px4_ros_com | (vendored plain, pre-existing) |

`px4_msgs` **must** track the firmware: PX4 v1.15.x → branch `release/1.15`.
`main` carries the post-1.15 message-versioning scheme (264 msgs with
`MESSAGE_VERSION`) whose structs differ from v1.15.4's — the uXRCE-DDS bridge
then fails to deserialize (`VehicleStatus` 88 B from PX4 vs 87 B in ROS,
`RTPS_READER_HISTORY` errors) and the vehicle never sees arming state.
