import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


SCENES = {
    'none': {
        'obstacles':  [],
        'init_wp_x':  3.0,
        'init_wp_y':  0.0,
        'init_wp_z':  1.5,
        'init_max_x': 5.0,
    },
    'single': {
        'obstacles':  [1.5, 0.0, 1.5, 0.5],
        'init_wp_x':  3.0,
        'init_wp_y':  0.0,
        'init_wp_z':  1.5,
        'init_max_x': 5.0,
    },
    'gauntlet': {
        'obstacles':  [1.0, 1.0, 1.5, 0.4,  2.0, -1.0, 1.5, 0.4,  3.0, 1.0, 1.5, 0.4,  4.0, -1.0, 1.5, 0.4],
        'init_wp_x':  5.5,
        'init_wp_y':  0.0,
        'init_wp_z':  1.5,
        'init_max_x': 7.0,
    },
}


def launch_setup(context, *args, **kwargs):
    scene = LaunchConfiguration('scene').perform(context)
    preset = SCENES.get(scene, SCENES['single'])

    boundary_manager = Node(
        package='arc_cbf_node',
        executable='boundary_manager',
        name='boundary_manager_node',
        output='screen',
        parameters=[{
            'init_min_x': -1.0,
            'init_min_y': -3.0,
            'init_min_z':  0.5,
            'init_max_x': float(preset['init_max_x']),
            'init_max_y':  3.0,
            'init_max_z':  3.0,
            'init_wp_x':  float(preset['init_wp_x']),
            'init_wp_y':  float(preset['init_wp_y']),
            'init_wp_z':  float(preset['init_wp_z']),
        }],
    )

    # An empty Python list has no inferable ROS parameter type, so skip the
    # node entirely for a no-obstacle scene rather than passing 'obstacles': [].
    obstacle_pub = [Node(
        package='arc_perception',
        executable='obstacle_pub',
        name='obstacle_publisher',
        output='screen',
        parameters=[{
            'obstacles': preset['obstacles'],
        }],
    )] if preset['obstacles'] else []

    obstacle_cbf = Node(
        package='arc_cbf_node',
        executable='obstacle_cbf',
        name='obstacle_cbf_node',
        output='screen',
        parameters=[{
            'cbf_alpha':          float(LaunchConfiguration('cbf_alpha').perform(context)),
            'cbf_alpha_obs':      float(LaunchConfiguration('cbf_alpha_obs').perform(context)),
            'obs_safety_margin':  float(LaunchConfiguration('obs_safety_margin').perform(context)),
            'k_p':                float(LaunchConfiguration('k_p').perform(context)),
            'v_max':              float(LaunchConfiguration('v_max').perform(context)),
            'yaw_deg':            float(LaunchConfiguration('yaw_deg').perform(context)),
            'penn_enabled':       LaunchConfiguration('penn_enabled').perform(context).lower() == 'true',
        }],
    )

    return [boundary_manager, *obstacle_pub, obstacle_cbf]


def generate_launch_description():
    worlds_dir = os.path.join(
        get_package_share_directory('arc_cbf_node'), 'worlds')

    gz_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        worlds_dir + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''))

    dds_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'world', '--child-frame-id', 'map'],
    )

    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument('scene',              default_value='none',
                              description='Gazebo world preset: none | single | gauntlet'),
        DeclareLaunchArgument('cbf_alpha',          default_value='0.5'),
        DeclareLaunchArgument('cbf_alpha_obs',      default_value='1.0'),
        DeclareLaunchArgument('obs_safety_margin',  default_value='0.5'),
        DeclareLaunchArgument('k_p',                default_value='0.5'),
        DeclareLaunchArgument('v_max',              default_value='0.5'),
        DeclareLaunchArgument('yaw_deg',            default_value='0.0'),
        DeclareLaunchArgument('penn_enabled',       default_value='true'),
        dds_agent,
        static_tf,
        OpaqueFunction(function=launch_setup),
    ])
