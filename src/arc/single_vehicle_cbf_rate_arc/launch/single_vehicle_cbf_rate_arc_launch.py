import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml

# single_vehicle_cbf_rate_arc is a single merged C++ node (bridge + CBF-QP + PENN/GAT, all in one
# process/one timer -- see the port plan).


def _make_node(context, *args, **kwargs):
    uav_prefix = LaunchConfiguration("uav_prefix")

    param_file_path = os.path.join(
        get_package_share_directory("single_vehicle_cbf_rate_arc"),
        "config",
        "params_single_vehicle_cbf_rate_arc.yaml"
    )

    with open(param_file_path, "r") as f:
        ros_parameters = yaml.safe_load(f)

    autopilot_params = ros_parameters.get("/**/autopilot_sv_cbf_rate_node", {}).get("ros__parameters", {})

    node_parameters = [autopilot_params, {"uav_prefix": uav_prefix}]

    # Optional runtime override for the obstacle model.
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
