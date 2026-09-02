"""Bring up just the shadow-HITL echo node.

The two ends are started separately and on purpose:

  1. SITL:   ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py
             launch_rviz:=false            # brings up uav_0 + gz_x500 + CBF node
  2. Bench:  on the real companion computer, start the serial agent on a
             SEPARATE namespace so it cannot collide with the SITL agent:
               MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600
             with the Pixhawk's uxrce_dds_client configured `-n uav_bench`.
             Confirm the bridge:  px4_config/verify_vehicle_status_topic.sh
  3. This:   ros2 launch hitl_bench_support hitl_bench_launch.py
  4. Put the bench board in OFFBOARD and arm it from the RC (this node only
     forwards setpoints -- it never arms), then release the gate:
       ros2 topic pub -1 /uav_bench/bench_enable std_msgs/msg/Bool "{data: true}"

PROPS OFF. Airframe strapped down. RC kill switch mapped and tested first.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sim_prefix", default_value="uav_0"),
        DeclareLaunchArgument("bench_prefix", default_value="uav_bench"),
        DeclareLaunchArgument(
            "thrust_clamp_norm", default_value="0.15",
            description="Normalised thrust magnitude cap on the forwarded command."),
        DeclareLaunchArgument("deadman_timeout_sec", default_value="0.1"),
        DeclareLaunchArgument(
            "require_bench_enable", default_value="true",
            description="Wait for a rising edge on <bench_prefix>/bench_enable "
                        "before forwarding anything."),
        Node(
            package="hitl_bench_support",
            executable="hitl_rate_echo_node",
            name="hitl_rate_echo_node",
            output="screen",
            parameters=[{
                "sim_prefix": LaunchConfiguration("sim_prefix"),
                "bench_prefix": LaunchConfiguration("bench_prefix"),
                "thrust_clamp_norm": ParameterValue(
                    LaunchConfiguration("thrust_clamp_norm"), value_type=float),
                "deadman_timeout_sec": ParameterValue(
                    LaunchConfiguration("deadman_timeout_sec"), value_type=float),
                "require_bench_enable": ParameterValue(
                    LaunchConfiguration("require_bench_enable"), value_type=bool),
            }],
        ),
    ])
