from setuptools import find_packages, setup

package_name = 'arc_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mchan5',
    maintainer_email='matthew.chan0607@gmail.com',
    description='Obstacle perception pipeline for ARC',
    license='MIT',
    entry_points={
        'console_scripts': [
            'obstacle_pub = arc_perception.obstacle_publisher:main',
        ],
    },
)
