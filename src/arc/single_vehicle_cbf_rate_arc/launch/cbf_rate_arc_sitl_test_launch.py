import json
import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

# (x, y, z, diameter) groups.
OBSTACLE_SCENES = {
    "none": [],
    "one": [0.0, 2.5, 1.5, 1.0],
    "two": [-1.2, 1.0, 1.5, 0.8, 1.2, 1.0, 1.5, 0.8],
    "slalom": [
        -1.2, 1.0, 1.5, 0.8,
         1.2, 1.0, 1.5, 0.8,
         0.0, 2.5, 1.5, 1.0,
        -1.2, 4.0, 1.3, 0.9,
         1.2, 4.0, 1.7, 0.9,
    ],
    "seven": [
        -1.3, 1.0, 1.4, 0.8,
         1.3, 1.0, 1.6, 0.8,
        -1.3, 2.5, 1.5, 0.9,
         1.3, 2.5, 1.4, 0.9,
         0.0, 4.2, 1.6, 1.0,
        -1.3, 6.0, 1.3, 0.8,
         1.3, 6.0, 1.7, 0.8,
    ],
}

# No hardcoded machine-specific defaults here -- leave micro_xrce_agent_src_dir and px4_src_dir
# undeclared-default so single_drone_sitl_launch.py's own MICRO_XRCE_AGENT_SRC_DIR/PX4_SRC_DIR …
_default_base = os.environ.get("HOME", os.curdir)
_MICRO_XRCE_AGENT_SRC_DIR_DEFAULT = os.environ.get(
    "MICRO_XRCE_AGENT_SRC_DIR", os.path.join(_default_base, "src", "Micro-XRCE-DDS-Agent"))
_PX4_SRC_DIR_DEFAULT = os.environ.get(
    "PX4_SRC_DIR", os.path.join(_default_base, "src", "PX4-Autopilot"))


def _sensing_envelope():
    """Shared obstacle sensing envelope -- single source of truth for both
    obstacle sources. See config/obstacle_sensing_envelope.yaml."""
    pkg_share = get_package_share_directory("single_vehicle_cbf_rate_arc")
    with open(os.path.join(pkg_share, "config", "obstacle_sensing_envelope.yaml")) as f:
        return yaml.safe_load(f)


def _launch_obstacle_publisher(context, *args, **kwargs):
    # obstacle_source is accepted for argument-parity with cbf_rate_arc_hardware_launch.py so the
    # operator GUI can pass the same vocabulary to either regime.
    source = LaunchConfiguration("obstacle_source").perform(context)
    if source == "lidar":
        raise RuntimeError(
            "obstacle_source:=lidar is not supported in SITL (no simulated "
            "LiDAR feed); use 'manual' with obstacle_file / obstacle_scene.")
    if source != "manual":
        raise RuntimeError(f"Unknown obstacle_source '{source}', expected 'manual' or 'lidar'")

    # obstacle_file (a JSON list of floats) overrides obstacle_scene when set -- lets
    # automated/randomized trials supply obstacles without editing this file's OBSTACLE_SCENES per …
    obstacle_file = LaunchConfiguration("obstacle_file").perform(context)
    if obstacle_file:
        with open(obstacle_file) as f:
            obstacles = json.load(f)
    else:
        scene = LaunchConfiguration("obstacle_scene").perform(context)
        if scene not in OBSTACLE_SCENES:
            raise RuntimeError(
                f"Unknown obstacle_scene '{scene}', expected one of {list(OBSTACLE_SCENES)}")
        obstacles = OBSTACLE_SCENES[scene]
    if not obstacles:
        return []
    envelope = _sensing_envelope()
    uav_prefix = LaunchConfiguration("uav_prefix").perform(context)
    return [Node(
        package="synthetic_obstacle_publisher",
        executable="synthetic_obstacle_publisher",
        name="synthetic_obstacle_publisher",
        namespace=uav_prefix,
        parameters=[{
            "obstacles": obstacles,
            "frame_id": "map",
            "publish_rate_hz": 10.0,
            "max_range_m": envelope["max_range_m"],
            "z_min_m": envelope["z_min_m"],
            "z_max_m": envelope["z_max_m"],
        }],
    )]


def _launch_arm_node(context, *args, **kwargs):
    """Select the arming driver, matching cbf_rate_arc_hardware_launch.py."""
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
        raise RuntimeError(f"Unknown arm_mode '{arm_mode}', expected 'auto' or 'operator'")

    # Delayed so PX4/EKF and the CBF stack are up before we start arming.
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
                # publish_goal:=false lets an external driver (the GUI) own
                # goal_waypoint without the stub's 10 Hz publish stomping it.
                "publish_goal": ParameterValue(
                    LaunchConfiguration("publish_goal"), value_type=bool),
            }],
        )],
    )]


def generate_launch_description():
    uav_prefix = LaunchConfiguration("uav_prefix")
    micro_xrce_agent_src_dir = LaunchConfiguration("micro_xrce_agent_src_dir")
    px4_src_dir = LaunchConfiguration("px4_src_dir")
    pkg_share = get_package_share_directory("single_vehicle_cbf_rate_arc")

    sitl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "single_drone_sitl_launch.py")
        ),
        launch_arguments={
            "uav_namespace": uav_prefix,
            "micro_xrce_agent_src_dir": micro_xrce_agent_src_dir,
            "px4_src_dir": px4_src_dir,
        }.items(),
    )

    cbf_rate_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "single_vehicle_cbf_rate_arc_launch.py")
        ),
        launch_arguments={
            "uav_prefix": uav_prefix,
            "cbf_cylinder_barrier": LaunchConfiguration("cbf_cylinder_barrier"),
        }.items(),
    )

    odom_bridge = Node(
        package="sitl_test_support",
        executable="px4_odom_bridge_node",
        name="px4_odom_bridge_node",
        namespace=uav_prefix,
    )

    # /arc/obstacles is a MarkerArray only -- synthetic_obstacle_publisher never spawns anything
    # into the Gazebo world itself, so RViz is the only way to see what the CBF node is actually …
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_share, "config", "cbf_sitl_test.rviz")],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    geofence_monitor = Node(
        package="hardware_test_support",
        executable="geofence_monitor_node",
        name="geofence_monitor_node",
        namespace=uav_prefix,
        parameters=[{
            "enabled": ParameterValue(
                LaunchConfiguration("geofence_enabled"), value_type=bool),
            "x_min": ParameterValue(LaunchConfiguration("geofence_x_min"), value_type=float),
            "x_max": ParameterValue(LaunchConfiguration("geofence_x_max"), value_type=float),
            "y_min": ParameterValue(LaunchConfiguration("geofence_y_min"), value_type=float),
            "y_max": ParameterValue(LaunchConfiguration("geofence_y_max"), value_type=float),
            "z_min": ParameterValue(LaunchConfiguration("geofence_z_min"), value_type=float),
            "z_max": ParameterValue(LaunchConfiguration("geofence_z_max"), value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "uav_prefix",
            default_value="uav_0",
            description="Namespace for the UAV",
        ),
        DeclareLaunchArgument(
            "micro_xrce_agent_src_dir",
            default_value=_MICRO_XRCE_AGENT_SRC_DIR_DEFAULT,
            description="Path to the MicroXRCEAgent source/build directory "
                         "(defaults to $MICRO_XRCE_AGENT_SRC_DIR, else $HOME/src/Micro-XRCE-DDS-Agent)",
        ),
        DeclareLaunchArgument(
            "px4_src_dir",
            default_value=_PX4_SRC_DIR_DEFAULT,
            description="Path to the PX4-Autopilot directory "
                         "(defaults to $PX4_SRC_DIR, else $HOME/src/PX4-Autopilot)",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Launch RViz2 with a display for /arc/obstacles and drone "
                        "odometry (opt-in; SITL is headless by default)",
        ),
        DeclareLaunchArgument(
            "obstacle_scene",
            default_value="one",
            description=f"Obstacle scene: one of {list(OBSTACLE_SCENES)}",
        ),
        DeclareLaunchArgument(
            "obstacle_file",
            default_value="",
            description="Path to a JSON file containing a flat [x,y,z,diameter,...] "
                         "obstacle list. Overrides obstacle_scene when non-empty.",
        ),
        # Goal waypoint.
        DeclareLaunchArgument(
            "cbf_cylinder_barrier",
            default_value="",
            description="Obstacle model override passed to the CBF node: "
                        "'' keeps the YAML value, 'true' = vertical-cylinder "
                        "barrier, 'false' = sphere barrier.",
        ),
        DeclareLaunchArgument("waypoint_x", default_value="0.0"),
        DeclareLaunchArgument("waypoint_y", default_value="5.5"),
        DeclareLaunchArgument("waypoint_z", default_value="1.5"),
        DeclareLaunchArgument(
            "obstacle_source",
            default_value="manual",
            description="Obstacle source: 'manual' (obstacle_file / obstacle_scene). "
                         "'lidar' is hardware-only and raises here -- accepted only "
                         "for parity with the hardware launch.",
        ),
        DeclareLaunchArgument(
            "arm_mode",
            default_value="auto",
            description="'auto' = ground_station_stub_node auto-arms ~5 s after "
                         "launch (unattended-batch default). 'operator' = "
                         "operator_arm_node, arms only on a rising edge of "
                         "operator/arm_confirm -- the path the GUI drives, same "
                         "node as the hardware launch.",
        ),
        DeclareLaunchArgument(
            "publish_goal",
            default_value="true",
            description="[arm_mode:=auto] Let ground_station_stub_node publish "
                         "goal_waypoint at 10 Hz. Set false when an external "
                         "driver (the GUI) owns the goal.",
        ),
        DeclareLaunchArgument(
            "geofence_enabled",
            default_value="false",
            description="Companion-side geofence backup (geofence_monitor_node). "
                         "Off by default in SITL; turn on to exercise the RTL path "
                         "before trusting it on hardware.",
        ),
        DeclareLaunchArgument("geofence_x_min", default_value="-5.0"),
        DeclareLaunchArgument("geofence_x_max", default_value="5.0"),
        DeclareLaunchArgument("geofence_y_min", default_value="-5.0"),
        DeclareLaunchArgument("geofence_y_max", default_value="5.0"),
        DeclareLaunchArgument("geofence_z_min", default_value="0.0"),
        DeclareLaunchArgument("geofence_z_max", default_value="3.0"),
        sitl,
        cbf_rate_stack,
        OpaqueFunction(function=_launch_obstacle_publisher),
        odom_bridge,
        OpaqueFunction(function=_launch_arm_node),
        geofence_monitor,
        rviz,
    ])
