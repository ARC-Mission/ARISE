
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Package parameters
    share_dir = get_package_share_directory('arc_m_skills')
    config_path = os.path.join(share_dir, 'config', 'ros2_params.yaml')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Policy Manager Node
    policy_manager = Node(
        package='arc_m_skills',
        executable='policy_manager',
        name='policy_manager',
        parameters=[config_path],
        output='screen'
    )
    
    # State Estimator Node
    state_estimator = Node(
        package='arc_m_skills',
        executable='state_estimator',
        name='state_estimator',
        parameters=[config_path],
        output='screen'
    )
    
    # Magistral Planner Node (Optional - enables LLM reasoning)
    magistral_planner = Node(
        package='arc_m_skills',
        executable='magistral_planner',
        name='magistral_planner',
        parameters=[config_path],
        output='screen'
    )
    
    # Robot Interface Node
    # Note: In simulation, this might Bridge to Isaac Sim
    robot_interface = Node(
        package='arc_m_skills',
        executable='robot_interface',
        name='robot_interface',
        parameters=[config_path],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo/Isaac) clock if true'),
            
        state_estimator,
        policy_manager,
        magistral_planner,
        robot_interface
    ])
