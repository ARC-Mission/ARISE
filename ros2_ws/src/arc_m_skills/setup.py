"""
ARC-M Skills Package
ROS 2 package for deploying learned recovery policies
"""

from setuptools import setup, find_packages

package_name = 'arc_m_skills'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/ros2_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/arc_m_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ARC-M Team',
    maintainer_email='team@arc-m.dev',
    description='ROS 2 skill library for ARC-M autonomous recovery',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_manager = arc_m_skills.policy_manager:main',
            'robot_interface = arc_m_skills.robot_interface:main',
            'state_estimator = arc_m_skills.state_estimator:main',
            'magistral_planner = arc_m_skills.magistral_planner:main',
        ],
    },
)
