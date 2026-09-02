import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'arc_cbf_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mchan5',
    maintainer_email='matthew.chan0607@gmail.com',
    description='Boundary CBF node for PX4 quadrotor via ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'boundary_cbf     = arc_cbf_node.boundary_cbf_node:main',
            'boundary_manager = arc_cbf_node.boundary_manager_node:main',
            'obstacle_cbf     = arc_cbf_node.obstacle_cbf_node:main',
        ],
    },
)
