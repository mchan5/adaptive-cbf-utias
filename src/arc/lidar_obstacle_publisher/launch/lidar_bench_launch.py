"""Bench bring-up for LiDAR -> /arc/obstacles perception -- no mocap, no autopilot.

Starts (each toggleable):
  * livox_ros_driver2 Mid-360 driver with xfer_format=0, i.e. a real
    sensor_msgs/PointCloud2 on /livox/lidar (the stock livox launch uses the
    custom format, which lidar_obstacle_publisher cannot read)
  * lidar_obstacle_publisher, with the world-frame height filter opened up
    (z_min/z_max args) so a sensor sitting on a bench still sees obstacles
  * a static map->livox_frame identity TF so RViz can overlay cloud + markers
  * an identity Odometry publisher on <uav_prefix>/state_estimator/local_position/odom
    -- the node drops every cloud until it has a pose; with identity odom the
    world frame IS the sensor frame (x fwd, y left, z up), which is all a
    perception check needs
  * RViz2 with rviz/lidar_bench.rviz

This is a perception sanity rig only. Real flight uses
single_vehicle_cbf_rate_arc/launch/cbf_rate_arc_hardware_launch.py with
obstacle_source:=lidar, which wires in the real odom source, the measured
mount offset, and the production sensing envelope.

    ros2 launch lidar_obstacle_publisher lidar_bench_launch.py
    ros2 launch lidar_obstacle_publisher lidar_bench_launch.py driver:=false   # driver already up
    ros2 launch lidar_obstacle_publisher lidar_bench_launch.py rviz:=false fake_odom:=false
    ros2 launch lidar_obstacle_publisher lidar_bench_launch.py \
        self_filter:=0.35,0.35,0.20 roi:=-1.5,-0.5,1.5,6.5   # cull the airframe + clip to arena
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_TRUE = ("1", "true", "yes")


def _setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("lidar_obstacle_publisher")

    uav_prefix = LaunchConfiguration("uav_prefix").perform(context)
    pointcloud_topic = LaunchConfiguration("pointcloud_topic").perform(context)
    z_min = LaunchConfiguration("z_min").perform(context)
    z_max = LaunchConfiguration("z_max").perform(context)
    marker_lifetime = LaunchConfiguration("marker_lifetime").perform(context)
    marker_shape = LaunchConfiguration("marker_shape").perform(context).strip()
    self_filter = LaunchConfiguration("self_filter").perform(context).strip()
    roi = LaunchConfiguration("roi").perform(context).strip()
    run_driver = LaunchConfiguration("driver").perform(context).lower() in _TRUE
    run_fake_odom = LaunchConfiguration("fake_odom").perform(context).lower() in _TRUE

    odom_topic = f"/{uav_prefix}/state_estimator/local_position/odom"
    actions = []

    if run_driver:
        config_file = LaunchConfiguration("config_file").perform(context)
        if not config_file:
            config_file = os.path.join(
                get_package_share_directory("livox_ros_driver2"),
                "config", "MID360_config.json")
        actions.append(Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=[{
                "xfer_format": 0,       # 0 = sensor_msgs/PointCloud2 (not the custom Livox msg)
                "multi_topic": 0,
                "data_src": 0,
                "publish_freq": 10.0,
                "output_data_type": 0,
                "frame_id": "livox_frame",
                "user_config_path": config_file,
            }],
        ))

    node_params = {
        "pointcloud_topic": pointcloud_topic,
        "frame_id": "map",
        # Bench default: open the world-frame height band right up. Production
        # values (0.1 .. 2.5) live in
        # single_vehicle_cbf_rate_arc/config/obstacle_sensing_envelope.yaml.
        "z_min_m": float(z_min),
        "z_max_m": float(z_max),
        # Stale markers self-expire in RViz instead of piling up (node only emits ADD).
        "marker_lifetime_s": float(marker_lifetime),
        "marker_shape": marker_shape,
    }
    if self_filter:
        hx, hy, hz = (float(v) for v in self_filter.split(","))
        node_params["self_filter_half_extents_xyz"] = [hx, hy, hz]
    if roi:
        xmin, ymin, xmax, ymax = (float(v) for v in roi.split(","))
        node_params["roi_enabled"] = True
        node_params["roi_xy_min"] = [xmin, ymin]
        node_params["roi_xy_max"] = [xmax, ymax]

    actions.append(Node(
        package="lidar_obstacle_publisher",
        executable="lidar_obstacle_publisher",
        name="lidar_obstacle_publisher_node",
        namespace=uav_prefix,
        output="screen",
        parameters=[node_params],
    ))

    actions.append(Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_livox_frame",
        arguments=[
            "--frame-id", "map", "--child-frame-id", "livox_frame",
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
        ],
    ))

    if run_fake_odom:
        odom_msg = (
            "{header: {frame_id: map}, child_frame_id: base_link, "
            "pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, "
            "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
        )
        actions.append(ExecuteProcess(
            cmd=["ros2", "topic", "pub", "-r", "30", odom_topic,
                 "nav_msgs/msg/Odometry", odom_msg],
            output="screen",
        ))

    actions.append(Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_share, "rviz", "lidar_bench.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("uav_prefix", default_value="uav_0"),
        DeclareLaunchArgument(
            "driver", default_value="true",
            description="Run the livox_ros_driver2 Mid-360 driver (false if it's already up)"),
        DeclareLaunchArgument(
            "config_file", default_value="",
            description="Livox JSON config path; '' = packaged livox_ros_driver2 MID360_config.json"),
        DeclareLaunchArgument("pointcloud_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument(
            "fake_odom", default_value="true",
            description="Publish an identity Odometry so the node runs without mocap"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("z_min", default_value="-2.0"),
        DeclareLaunchArgument("z_max", default_value="3.0"),
        DeclareLaunchArgument(
            "marker_lifetime", default_value="0.3",
            description="Seconds each marker lives in RViz before it self-expires; 0 = forever"),
        DeclareLaunchArgument(
            "marker_shape", default_value="sphere",
            description="'sphere' or 'cylinder' -- match the CBF obstacle model in use"),
        DeclareLaunchArgument(
            "self_filter", default_value="",
            description="Body-frame self-filter half-extents 'hx,hy,hz' in metres; "
                        "'' disables. Drops points inside the box (the airframe)."),
        DeclareLaunchArgument(
            "roi", default_value="",
            description="World-frame arena ROI 'xmin,ymin,xmax,ymax' in metres; "
                        "'' disables. Drops points outside the box before clustering."),
        OpaqueFunction(function=_setup),
    ])
