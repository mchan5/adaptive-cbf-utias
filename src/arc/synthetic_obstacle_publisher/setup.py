import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'synthetic_obstacle_publisher'

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
    description='Bench-test synthetic obstacle publisher (MarkerArray on /arc/obstacles)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'synthetic_obstacle_publisher = synthetic_obstacle_publisher.synthetic_obstacle_publisher_node:main',
        ],
    },
)
