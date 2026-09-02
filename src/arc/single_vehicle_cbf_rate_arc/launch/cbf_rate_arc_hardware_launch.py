import json
import os

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

# Real-hardware analog of cbf_rate_arc_sitl_test_launch.py.


def _sensing_envelope():
    """Shared obstacle sensing envelope (max_range_m / z_min_m / z_max_m /
    max_obstacle_radius_m)."""
    pkg_share = get_package_share_directory("single_vehicle_cbf_rate_arc")
    with open(os.path.join(pkg_share, "config", "obstacle_sensing_envelope.yaml")) as f:
        return yaml.safe_load(f)


def _launch_obstacle_publisher(context, *args, **kwargs):
    source = LaunchConfiguration("obstacle_source").perform(context)

    envelope = _sensing_envelope()

    if source == "lidar":
        return [Node(
            package="lidar_obstacle_publisher",
            executable="lidar_obstacle_publisher",
            name="lidar_obstacle_publisher",
            namespace=LaunchConfiguration("uav_prefix"),
            parameters=[{
                "pointcloud_topic": LaunchConfiguration("lidar_pointcloud_topic"),
                "frame_id": "map",
                **envelope,
            }],
        )]

    if source != "manual":
        raise RuntimeError(f"Unknown obstacle_source '{source}', expected 'manual' or 'lidar'")

    obstacle_file = LaunchConfiguration("obstacle_file").perform(context)
    if not obstacle_file:
        return []
    with open(obstacle_file) as f:
        obstacles = json.load(f)
    if not obstacles:
        return []
    return [Node(
        package="synthetic_obstacle_publisher",
        executable="synthetic_obstacle_publisher",
        name="synthetic_obstacle_publisher",
        # Must be namespaced under uav_prefix, exactly like the lidar branch above and the SITL
        # launch: synthetic_obstacle_publisher_node subscribes to the RELATIVE …
        namespace=LaunchConfiguration("uav_prefix"),
        parameters=[{
            "obstacles": obstacles,
            "frame_id": "map",
            "publish_rate_hz": 10.0,
            # synthetic has no clustering step -- max_obstacle_radius_m N/A
            "max_range_m": envelope["max_range_m"],
            "z_min_m": envelope["z_min_m"],
            "z_max_m": envelope["z_max_m"],
        }],
    )]


def _launch_arm_node(context, *args, **kwargs):
    """Select the arming driver."""
    arm_mode = LaunchConfiguration("arm_mode").perform(context)
    uav_prefix = LaunchConfiguration("uav_prefix")

    if arm_mode == "operator":
        return [Node(
            package="hardware_test_support",
            executable="operator_arm_node",
            name="operator_arm_node",
            namespace=uav_prefix,
            parameters=[{"vehicle_status_topic": LaunchConfiguration("vehicle_status_topic")}],
        )]

    if arm_mode != "auto":
        raise RuntimeError(f"Unknown arm_mode '{arm_mode}', expected 'operator' or 'auto'")

    return [TimerAction(
        period=5.0,
        actions=[Node(
            package="sitl_test_support",
            executable="ground_station_stub_node",
            name="ground_station_stub_node",
            namespace=uav_prefix,
            parameters=[{
                "waypoint_x": ParameterValue(
                    LaunchConfiguration("waypoint_x"), value_type=float),
                "waypoint_y": ParameterValue(
                    LaunchConfiguration("waypoint_y"), value_type=float),
                "waypoint_z": ParameterValue(
                    LaunchConfiguration("waypoint_z"), value_type=float),
                "arm_delay_sec": 5.0,
                "publish_goal": ParameterValue(
                    LaunchConfiguration("publish_goal"), value_type=bool),
            }],
        )],
    )]


def generate_launch_description():
    uav_prefix = LaunchConfiguration("uav_prefix")
    pkg_share = get_package_share_directory("single_vehicle_cbf_rate_arc")

    cbf_rate_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "single_vehicle_cbf_rate_arc_hardware_launch.py")
        ),
        launch_arguments={
            "uav_prefix": uav_prefix,
            "cbf_cylinder_barrier": LaunchConfiguration("cbf_cylinder_barrier"),
            "vehicle_status_topic": LaunchConfiguration("vehicle_status_topic"),
        }.items(),
    )

    mocap_bridge = Node(
        package="hardware_test_support",
        executable="mocap_odom_bridge_node",
        name="mocap_odom_bridge_node",
        namespace=uav_prefix,
        parameters=[{"model_name": uav_prefix}],
    )

    geofence_monitor = Node(
        package="hardware_test_support",
        executable="geofence_monitor_node",
        name="geofence_monitor_node",
        namespace=uav_prefix,
        parameters=[{
            "enabled": LaunchConfiguration("geofence_enabled"),
            "x_min": LaunchConfiguration("geofence_x_min"),
            "x_max": LaunchConfiguration("geofence_x_max"),
            "y_min": LaunchConfiguration("geofence_y_min"),
            "y_max": LaunchConfiguration("geofence_y_max"),
            "z_min": LaunchConfiguration("geofence_z_min"),
            "z_max": LaunchConfiguration("geofence_z_max"),
        }],
    )

    # /arc/obstacles is a MarkerArray only -- same as sim, RViz is the only
    # way to see what the CBF node is actually reasoning about.
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_share, "config", "cbf_sitl_test.rviz")],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "uav_prefix",
            default_value="uav_0",
            description="Namespace for the UAV -- must match the OptiTrack driver's model name",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz2 with a display for /arc/obstacles and drone odometry",
        ),
        DeclareLaunchArgument(
            "obstacle_source",
            default_value="manual",
            description="Obstacle source: 'manual' (hand-measured obstacle_file) or "
                         "'lidar' (real-time detection from lidar_pointcloud_topic)",
        ),
        DeclareLaunchArgument(
            "obstacle_file",
            default_value=os.path.join(pkg_share, "config", "obstacles.json"),
            description="[obstacle_source:=manual] Path to a JSON file of measured real "
                         "obstacle positions ([x,y,z,diameter,...], flat list). Defaults to "
                         "the packaged config/obstacles.json (the two-cylinder head-on "
                         "avoidance course); pass obstacle_file:='' for a hover-only test.",
        ),
        DeclareLaunchArgument(
            "lidar_pointcloud_topic",
            default_value="/livox/lidar",
            description="[obstacle_source:=lidar] sensor_msgs/PointCloud2 topic published by "
                         "livox_ros_driver2 (Mid-360). Absolute path by default -- verify "
                         "against your driver's actual topic, it likely isn't namespaced "
                         "under uav_prefix.",
        ),
        # Sensing envelope (range / z-band / max cluster radius) is not a launch arg -- it lives in
        # config/obstacle_sensing_envelope.yaml so the LiDAR and synthetic sources can't diverge.
        DeclareLaunchArgument(
            "geofence_enabled",
            default_value="true",
            description="Enable the companion-computer-side geofence backup "
                         "(the primary geofence must still be set via PX4 GF_* params)",
        ),
        DeclareLaunchArgument("geofence_x_min", default_value="-5.0"),
        DeclareLaunchArgument("geofence_x_max", default_value="5.0"),
        DeclareLaunchArgument("geofence_y_min", default_value="-5.0"),
        DeclareLaunchArgument("geofence_y_max", default_value="5.0"),
        DeclareLaunchArgument("geofence_z_min", default_value="0.0"),
        # Hard height cap for the head-on avoidance test: obstacle cylinders are 0.75m tall,
        # waypoint is at their mid-height (z=0.38), so 0.75 caps the vehicle at the top of the …
        DeclareLaunchArgument("geofence_z_max", default_value="0.75"),
        DeclareLaunchArgument(
            "arm_mode",
            default_value="operator",
            description="'operator' (default) = operator_arm_node, arms on a "
                         "rising edge of operator/arm_confirm. 'auto' = "
                         "ground_station_stub_node auto-arms ~5 s after launch "
                         "-- FOOT-GUN on a real vehicle, bench / props-off only.",
        ),
        DeclareLaunchArgument(
            "publish_goal",
            default_value="true",
            description="[arm_mode:=auto] Let ground_station_stub_node publish "
                         "goal_waypoint. Set false when the GUI owns the goal.",
        ),
        # Only consumed when arm_mode:=auto (the stub's hold/goal target).
        DeclareLaunchArgument("waypoint_x", default_value="0.0"),
        DeclareLaunchArgument("waypoint_y", default_value="5.0"),
        DeclareLaunchArgument("waypoint_z", default_value="0.38"),
        DeclareLaunchArgument(
            "cbf_cylinder_barrier",
            default_value="",
            description="Obstacle model override passed to the CBF node: "
                        "'' keeps the hardware YAML value, 'true' = vertical-"
                        "cylinder barrier, 'false' = sphere barrier. NOT yet "
                        "flown -- SITL-rehearse before setting true on a real "
                        "vehicle, and only with genuinely floor-to-ceiling "
                        "obstacles.",
        ),
        DeclareLaunchArgument(
            "vehicle_status_topic",
            default_value=os.environ.get("ARC_VEHICLE_STATUS_TOPIC", "fmu/out/vehicle_status"),
            description="Versioned suffix differs between SITL and a real board's firmware -- "
                        "confirm with config/px4/verify_vehicle_status_topic.sh before flying. "
                        "Defaults from $ARC_VEHICLE_STATUS_TOPIC if set, else unversioned (SITL). "
                        "Passed through to both the CBF node and operator_arm_node.",
        ),

        cbf_rate_stack,
        mocap_bridge,
        OpaqueFunction(function=_launch_arm_node),
        geofence_monitor,
        OpaqueFunction(function=_launch_obstacle_publisher),
        rviz,
    ])
