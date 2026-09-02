import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sitl_test_support'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mchan5',
    maintainer_email='matthew.chan0607@gmail.com',
    description='SITL-only stand-ins for the missing state estimator and ground-station GUI, for testing single_vehicle_cbf in Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'px4_odom_bridge_node = sitl_test_support.px4_odom_bridge_node:main',
            'ground_station_stub_node = sitl_test_support.ground_station_stub_node:main',
        ],
    },
)
