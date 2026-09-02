from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('cbf_alpha',     default_value='0.5'),
        DeclareLaunchArgument('cbf_alpha_obs', default_value='1.0'),
        DeclareLaunchArgument('k_p',           default_value='0.5'),
        DeclareLaunchArgument('v_max',         default_value='0.3'),
        DeclareLaunchArgument('yaw_deg',       default_value='0.0'),
        DeclareLaunchArgument('serial_port',   default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baud',          default_value='921600'),
    ]

    dds_agent = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'serial',
            '--dev', LaunchConfiguration('serial_port'),
            '-b',   LaunchConfiguration('baud'),
        ],
        output='screen',
    )

    obstacle_cbf = Node(
        package='arc_cbf_node',
        executable='obstacle_cbf',
        name='obstacle_cbf_node',
        output='screen',
        parameters=[{
            'cbf_alpha':     LaunchConfiguration('cbf_alpha'),
            'cbf_alpha_obs': LaunchConfiguration('cbf_alpha_obs'),
            'k_p':           LaunchConfiguration('k_p'),
            'v_max':         LaunchConfiguration('v_max'),
            'yaw_deg':       LaunchConfiguration('yaw_deg'),
        }],
    )

    return LaunchDescription(args + [dds_agent, obstacle_cbf])
