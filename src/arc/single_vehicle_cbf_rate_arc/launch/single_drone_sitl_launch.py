import os

from launch import LaunchDescription, actions, substitutions

MICRO_XRCE_AGENT_EXECUTABLE = "MicroXRCEAgent"


def generate_launch_description():
    default_base = os.environ.get("HOME", os.curdir)

    micro_xrce_agent_src_dir_default = os.environ.get(
        "MICRO_XRCE_AGENT_SRC_DIR",
        os.path.join(default_base, "src", "Micro-XRCE-DDS-Agent"),
    )

    px4_src_dir_default = os.environ.get(
        "PX4_SRC_DIR",
        os.path.join(default_base, "src", "PX4-Autopilot"),
    )
    _ = {
        "micro_xrce": actions.DeclareLaunchArgument(
            "micro_xrce_agent_src_dir",
            default_value=micro_xrce_agent_src_dir_default,
            description="Path to the MicroXRCEAgent executable",
        ),
        "px4_src_dir": actions.DeclareLaunchArgument(
            "px4_src_dir",
            default_value=px4_src_dir_default,
            description="Path to the PX4-Autopilot directory",
        ),
        "model": actions.DeclareLaunchArgument(
            "uav_model",
            default_value="gz_x500",
            description="Model of the vehicle",
        ),
        "namespace": actions.DeclareLaunchArgument(
            "uav_namespace",
            default_value="",
            description="Namespace for the UAV",
        ),
    }

    micro_xrce_agent_dir = substitutions.PathJoinSubstitution(
        [
            substitutions.LaunchConfiguration(
                "micro_xrce_agent_src_dir", default=micro_xrce_agent_src_dir_default
            ),
            "build",
        ]
    )
    micro_xrce_agent_executable = substitutions.PathJoinSubstitution(
        [micro_xrce_agent_dir, MICRO_XRCE_AGENT_EXECUTABLE]
    )
    px4_dir = substitutions.PathJoinSubstitution(
        [
            substitutions.LaunchConfiguration(
                "px4_src_dir", default=px4_src_dir_default
            ),
            "build",
            "px4_sitl_default",
            "bin",
        ]
    )
    px4_executable = substitutions.PathJoinSubstitution([px4_dir, "px4"])

    # Only forward vars that are actually set in the parent shell -- these are opportunistic
    # passthroughs, not requirements.
    passthrough_env = {
        it: os.environ[it]
        for it in [
            "DISPLAY",
            "HOME",
            "GZ_CONFIG_PATH",
            "GZ_SIM_RESOURCE_PATH",
            "LD_LIBRARY_PATH",
            # HEADLESS=1 tells PX4's gz launch to start `gz sim` with no GUI -- required for
            # unattended batteries (authority sweep).
            "HEADLESS",
        ]
        if it in os.environ
    }

    return LaunchDescription(
        [
            actions.ExecuteProcess(
                cmd=[micro_xrce_agent_executable, "udp4", "-p", "8888"],
                name="microxrce_agent",
            ),
            actions.ExecuteProcess(
                cmd=[px4_executable],
                name="px4_sitl",
                env={
                    **passthrough_env,
                    "PATH": [
                        substitutions.EnvironmentVariable("PATH"),
                        substitutions.TextSubstitution(text=os.pathsep),
                        px4_dir,
                    ],
                    "PX4_SYS_AUTOSTART": "4001",
                    "PX4_SIM_MODEL": substitutions.LaunchConfiguration(
                        "uav_model", default="gz_x500"
                    ),
                    "PX4_UXRCE_DDS_NS": substitutions.LaunchConfiguration(
                        "uav_namespace", default=""
                    ),
                    # This harness has neither RC nor a GCS connected, so disable the "no data link"
                    # arming block (see rcAndDataLinkCheck.cpp) -- rcS applies any …
                    "PX4_PARAM_NAV_DLL_ACT": "0",
                },
                output="screen",
            ),
        ]
    )
