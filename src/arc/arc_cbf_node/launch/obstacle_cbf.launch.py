from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        # boundary
        DeclareLaunchArgument('init_min_x', default_value='-3.0'),
        DeclareLaunchArgument('init_min_y', default_value='-3.0'),
        DeclareLaunchArgument('init_min_z', default_value='0.5'),
        DeclareLaunchArgument('init_max_x', default_value='3.0'),
        DeclareLaunchArgument('init_max_y', default_value='3.0'),
        DeclareLaunchArgument('init_max_z', default_value='2.5'),
        # waypoint
        DeclareLaunchArgument('init_wp_x', default_value='0.0'),
        DeclareLaunchArgument('init_wp_y', default_value='0.0'),
        DeclareLaunchArgument('init_wp_z', default_value='1.5'),
        # controller gains
        DeclareLaunchArgument('cbf_alpha',     default_value='0.5'),
        DeclareLaunchArgument('cbf_alpha_obs', default_value='1.0'),
        DeclareLaunchArgument('k_p',           default_value='0.5'),
        DeclareLaunchArgument('v_max',         default_value='0.5'),
        # obstacles: flat list [x, y, z, r, x, y, z, r, ...]
        # default: one 0.5 m-radius sphere at (1.5, 0.0, 1.5)
        DeclareLaunchArgument('obstacles', default_value='[1.5, 0.0, 1.5, 0.5]'),
        DeclareLaunchArgument('yaw_deg',   default_value='0.0'),
    ]

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'world', '--child-frame-id', 'map'],
    )

    boundary_manager = Node(
        package='arc_cbf_node',
        executable='boundary_manager',
        name='boundary_manager_node',
        output='screen',
        parameters=[{
            'init_min_x': LaunchConfiguration('init_min_x'),
            'init_min_y': LaunchConfiguration('init_min_y'),
            'init_min_z': LaunchConfiguration('init_min_z'),
            'init_max_x': LaunchConfiguration('init_max_x'),
            'init_max_y': LaunchConfiguration('init_max_y'),
            'init_max_z': LaunchConfiguration('init_max_z'),
            'init_wp_x':  LaunchConfiguration('init_wp_x'),
            'init_wp_y':  LaunchConfiguration('init_wp_y'),
            'init_wp_z':  LaunchConfiguration('init_wp_z'),
        }],
    )

    obstacle_pub = Node(
        package='arc_perception',
        executable='obstacle_pub',
        name='obstacle_publisher',
        output='screen',
        parameters=[{
            'obstacles': LaunchConfiguration('obstacles'),
        }],
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

    return LaunchDescription(args + [static_tf, boundary_manager, obstacle_pub, obstacle_cbf])
