"""
MIT License

Copyright (c) 2025 FSC Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context, *args, **kwargs):
    objects_arg = LaunchConfiguration('objects').perform(context)
    object_list = objects_arg.split()

    # Get absolute path to the parameter file
    package_dir = get_package_share_directory('gz_optitrack_ros2_emulator')
    param_file = os.path.join(package_dir, 'config', 'params.yaml')

    return [
        Node(
            package='gz_optitrack_ros2_emulator',
            executable='gz_optitrack_emulator_node',
            name='mocap_emulator',
            output='screen',
            parameters=[param_file],  # ✅ now it's a full path
            arguments=object_list
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'objects',
            default_value='x500_0',
            description='List of objects to track'
        ),
        OpaqueFunction(function=launch_setup)
    ])