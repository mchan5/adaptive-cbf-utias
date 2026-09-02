import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hitl_bench_support'

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
    # extras_require['test'] is what colcon's pytest step keys off of to run
    # test/ with pytest rather than falling back to `python -m unittest`
    # (which cannot collect the pytest-fixture-based smoke test).
    tests_require=['pytest'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='mchan5',
    maintainer_email='matthew.chan0607@gmail.com',
    description='Shadow-HITL bench echo node: SITL body-rate commands -> real props-off Pixhawk.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'hitl_rate_echo_node = hitl_bench_support.hitl_rate_echo_node:main',
        ],
    },
)
