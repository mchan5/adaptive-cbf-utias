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

# Real-hardware analog of cbf_rate_arc_sitl_test_launch.py. Deliberately does
# NOT include:
#   - single_drone_sitl_launch.py (Gazebo + PX4 SITL + MicroXRCEAgent-over-UDP)
#     -- on hardware the Pixhawk is already running PX4 firmware, and
#     MicroXRCEAgent is started manually over serial before this launch file
#     runs (see fsc_autopilot_ros2/docs/indoor_experiments.md:
#     `MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600`), not launched
#     from ROS2.
#   - sitl_test_support's px4_odom_bridge_node / ground_station_stub_node
#     -- replaced below by hardware_test_support's real analogs.
# single_vehicle_cbf_rate_arc_hardware_launch.py (the CBF node itself) is
# untouched code, just pointed at the hardware params file.
#
# Two mutually exclusive obstacle sources, selected via obstacle_source --
# both publish the identical MarkerArray contract on /arc/obstacles, so the
# CBF node behaves the same regardless of which is running:
#   - "manual" (default): synthetic_obstacle_publisher (reused as-is from
#     the sim stack) fed a JSON file of hand-measured real obstacle
#     positions via obstacle_file. There's no obstacle_scene here -- sim's
#     canned courses aren't meaningful for a real room. Empty obstacle_file
#     means no obstacles, the right default for early hover-only tests.
#   - "lidar": lidar_obstacle_publisher, clustering a live
#     sensor_msgs/PointCloud2 feed into obstacle markers in real time.
#     Target unit is a Livox Mid-360 via Livox-SDK/livox_ros_driver2 (see
#     that package's node docstring for the required xfer_format:=0 driver
#     config and the topic-namespacing caveat) -- the driver itself is not
#     launched from here, and a measured sensor mount offset is required
#     before obstacle positions mean anything.
# Both sources take their sensing envelope (max_range_m, z-band, and -- lidar
# only -- max_obstacle_radius_m) from config/obstacle_sensing_envelope.yaml,
# so SITL-validated behavior can't silently diverge from hardware.


def _sensing_envelope():
    """Shared obstacle sensing envelope (max_range_m / z_min_m / z_max_m /
    max_obstacle_radius_m). Single source of truth for both obstacle sources --
    see config/obstacle_sensing_envelope.yaml."""
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
        # Must be namespaced under uav_prefix, exactly like the lidar branch
        # above and the SITL launch: synthetic_obstacle_publisher_node
        # subscribes to the RELATIVE state_estimator/local_position/odom, so
        # without this it resolves to /state_estimator/... while the mocap
        # bridge publishes /<uav_prefix>/state_estimator/... -- _odom_valid
        # never gets set, and the node publishes an empty MarkerArray (i.e.
        # the CBF node flies seeing zero obstacles). This also puts the node
        # at /<uav_prefix>/synthetic_obstacle_publisher, where the GUI's
        # scene-push SetParameters client expects it (ros_node.py).
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
    """Select the arming driver. Kept byte-for-byte in sync with
    cbf_rate_arc_sitl_test_launch.py's copy so a SITL rehearsal and a
    hardware flight arm through the same code path.

      arm_mode:=operator (default) -> operator_arm_node: arms only on a
                                      rising edge of operator/arm_confirm.
      arm_mode:=auto               -> ground_station_stub_node: auto-arms
                                      ~5 s after launch. FOOT-GUN on a real
                                      vehicle -- bench / props-off only.
    """
    arm_mode = LaunchConfiguration("arm_mode").perform(context)
    uav_prefix = LaunchConfiguration("uav_prefix")

    if arm_mode == "operator":
        return [Node(
            package="hardware_test_support",
            executable="operator_arm_node",
            name="operator_arm_node",
            namespace=uav_prefix,
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
        # Sensing envelope (range / z-band / max cluster radius) is not a launch
        # arg -- it lives in config/obstacle_sensing_envelope.yaml so the LiDAR
        # and synthetic sources can't diverge. Edit that file to retune.
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
        # Hard height cap for the head-on avoidance test: obstacle cylinders are
        # 0.75m tall, waypoint is at their mid-height (z=0.38), so 0.75 caps the
        # vehicle at the top of the obstacles with ~0.37m of climb margin before
        # the companion-side geofence fires. This is only the BACKUP layer (and
        # it reacts with RTL, not a clamp) -- set PX4 GF_MAX_VER_DIST / GF_ACTION
        # on the Pixhawk to match before flying.
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
        # Only consumed when arm_mode:=auto (the stub's hold/goal target). The
        # CBF node itself reads waypoint_x/y/z from the hardware params YAML,
        # not from here. Defaults match that YAML's course endpoint.
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

        cbf_rate_stack,
        mocap_bridge,
        OpaqueFunction(function=_launch_arm_node),
        geofence_monitor,
        OpaqueFunction(function=_launch_obstacle_publisher),
        rviz,
    ])
