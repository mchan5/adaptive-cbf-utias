import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hardware_test_support'

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
    description='Real-hardware bridge/arming/geofence nodes for single_vehicle_cbf_rate_arc, separate from the SITL-only sitl_test_support package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mocap_odom_bridge_node = hardware_test_support.mocap_odom_bridge_node:main',
            'operator_arm_node = hardware_test_support.operator_arm_node:main',
            'geofence_monitor_node = hardware_test_support.geofence_monitor_node:main',
        ],
    },
)
