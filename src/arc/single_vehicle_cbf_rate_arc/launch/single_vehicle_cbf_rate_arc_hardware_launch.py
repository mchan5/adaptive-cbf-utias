import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml

# Real-hardware analog of single_vehicle_cbf_rate_arc_launch.py -- identical
# node, identical single-process architecture (no separate obstacle
# controller process, no venv PYTHONPATH -- libtorch is linked in at build
# time). The only difference is the parameter file: this one loads
# params_single_vehicle_cbf_rate_arc_hardware.yaml instead of the sim
# params, which carries real-vehicle mass/thrust-curve values instead of
# copied Gazebo x500 SITL constants. single_vehicle_cbf_rate_client.cpp
# itself is untouched -- it doesn't know or care whether its odometry and
# obstacle inputs came from Gazebo or a real OptiTrack rig.


def _make_node(context, *args, **kwargs):
    uav_prefix = LaunchConfiguration("uav_prefix")

    param_file_path = os.path.join(
        get_package_share_directory("single_vehicle_cbf_rate_arc"),
        "config",
        "params_single_vehicle_cbf_rate_arc_hardware.yaml"
    )

    with open(param_file_path, "r") as f:
        ros_parameters = yaml.safe_load(f)

    autopilot_params = ros_parameters.get("/**/autopilot_sv_cbf_rate_node", {}).get("ros__parameters", {})

    node_parameters = [autopilot_params, {"uav_prefix": uav_prefix}]

    # Optional runtime override for the obstacle model, mirroring the sim
    # launch. Empty (default) keeps the hardware YAML's cbf_cylinder_barrier
    # value; "true"/"false" forces it so a single flight can A/B sphere vs
    # vertical-cylinder without editing YAML.
    raw = LaunchConfiguration("cbf_cylinder_barrier").perform(context).strip().lower()
    if raw in ("true", "1", "false", "0"):
        node_parameters.append({"cbf_cylinder_barrier": raw in ("true", "1")})

    return [
        Node(
            package="single_vehicle_cbf_rate_arc",
            executable="autopilot_sv_cbf_rate_node",
            name="autopilot_sv_cbf_rate_node",
            namespace=uav_prefix,
            parameters=node_parameters,
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "uav_prefix",
            default_value="uav_0",
            description="Namespace for UAV"
        ),
        DeclareLaunchArgument(
            "cbf_cylinder_barrier",
            default_value="",
            description="Override the obstacle model: '' keeps the YAML value, "
                        "'true' = vertical-cylinder barrier, 'false' = sphere barrier.",
        ),
        OpaqueFunction(function=_make_node),
    ])
