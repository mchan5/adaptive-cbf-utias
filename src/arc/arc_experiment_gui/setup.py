from setuptools import find_packages, setup

package_name = 'arc_experiment_gui'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matthew',
    maintainer_email='pizzaparty522@gmail.com',
    description='Operator GUI for the adaptive-vs-fixed-gamma obstacle-avoidance campaign.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard = arc_experiment_gui.main:main',
        ],
    },
)
