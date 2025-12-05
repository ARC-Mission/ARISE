#!/usr/bin/env python3
"""
ARC-M Policy Manager Node
Manages switching between learned recovery policies based on robot state.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

# ROS 2 message types
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, TwistStamped


class PolicyState(Enum):
    """Robot operation states."""
    NORMAL = "normal"
    STUCK = "stuck"
    FALLEN = "fallen"
    RECOVERY = "recovery"


@dataclass
class PolicyConfig:
    """Configuration for a single policy."""
    name: str
    path: str
    description: str
    observation_dim: int = 48
    action_dim: int = 12


class PolicyManager(Node):
    """
    ROS 2 node for managing learned RL policies.
    
    Responsibilities:
    - Load ONNX policy models
    - Construct observation vectors from sensor data
    - Run policy inference at high frequency
    - Switch between policies based on robot state
    - Publish action commands
    """
    
    def __init__(self):
        super().__init__('arc_m_policy_manager')
        
        # Declare parameters
        self.declare_parameter('update_rate', 100.0)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('default_policy', 'locomotion_flat')
        self.declare_parameter('auto_switch_enabled', True)
        self.declare_parameter('stuck_velocity_threshold', 0.05)
        self.declare_parameter('stuck_duration', 2.0)
        self.declare_parameter('orientation_threshold', 0.5)
        
        # Get parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.device = self.get_parameter('device').value
        self.default_policy_name = self.get_parameter('default_policy').value
        
        # Initialize state
        self.current_state = PolicyState.NORMAL
        self.current_policy_name = self.default_policy_name
        self.policies: Dict[str, ort.InferenceSession] = {}
        self.policy_configs: Dict[str, PolicyConfig] = {}
        
        # Observation buffer
        self.obs_buffer = {
            'base_lin_vel': np.zeros(3, dtype=np.float32),
            'base_ang_vel': np.zeros(3, dtype=np.float32),
            'projected_gravity': np.zeros(3, dtype=np.float32),
            'commands': np.zeros(3, dtype=np.float32),
            'joint_pos': np.zeros(12, dtype=np.float32),
            'joint_vel': np.zeros(12, dtype=np.float32),
            'prev_actions': np.zeros(12, dtype=np.float32),
        }
        
        # Stuck detection
        self.velocity_history: List[float] = []
        self.stuck_start_time: Optional[float] = None
        
        # Load policies
        self._load_policies()
        
        # Setup subscribers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, qos)
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self._imu_callback, qos)
        
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, qos)
        
        # Setup publishers
        self.action_pub = self.create_publisher(
            Float32MultiArray, '/policy/action', 10)
        
        self.current_policy_pub = self.create_publisher(
            String, '/policy/current', 10)
        
        # Control loop timer
        self.timer = self.create_timer(1.0 / self.update_rate, self._control_loop)
        
        self.get_logger().info(f'Policy Manager initialized with {len(self.policies)} policies')
        self.get_logger().info(f'Running at {self.update_rate} Hz on {self.device}')
        
    def _load_policies(self):
        """Load ONNX policy models."""
        
        # Define available policies
        policy_list = [
            PolicyConfig('locomotion_flat', 'models/locomotion_flat.onnx', 'Flat terrain walking'),
            PolicyConfig('locomotion_rough', 'models/locomotion_rough.onnx', 'Rough terrain walking'),
            PolicyConfig('recovery_stuck', 'models/recovery_stuck.onnx', 'Recovery from stuck'),
            PolicyConfig('recovery_fallen', 'models/recovery_fallen.onnx', 'Self-righting'),
        ]
        
        # ONNX runtime providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        
        for config in policy_list:
            policy_path = Path(config.path)
            
            if policy_path.exists():
                try:
                    session = ort.InferenceSession(str(policy_path), providers=providers)
                    self.policies[config.name] = session
                    self.policy_configs[config.name] = config
                    self.get_logger().info(f'Loaded policy: {config.name}')
                except Exception as e:
                    self.get_logger().warn(f'Failed to load {config.name}: {e}')
            else:
                self.get_logger().warn(f'Policy file not found: {policy_path}')
        
        # Ensure default policy is available
        if self.default_policy_name not in self.policies:
            if self.policies:
                self.default_policy_name = list(self.policies.keys())[0]
                self.get_logger().warn(f'Default policy not found, using: {self.default_policy_name}')
            else:
                self.get_logger().error('No policies loaded!')
                
    def _joint_state_callback(self, msg: JointState):
        """Process joint state messages."""
        if len(msg.position) >= 12:
            self.obs_buffer['joint_pos'] = np.array(msg.position[:12], dtype=np.float32)
        if len(msg.velocity) >= 12:
            self.obs_buffer['joint_vel'] = np.array(msg.velocity[:12], dtype=np.float32)
            
    def _imu_callback(self, msg: Imu):
        """Process IMU messages."""
        # Angular velocity
        self.obs_buffer['base_ang_vel'] = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float32)
        
        # Linear acceleration (approximate velocity with integration - simplified)
        # In practice, use state estimator
        self.obs_buffer['base_lin_vel'] = np.array([
            msg.linear_acceleration.x * 0.01,  # Crude approximation
            msg.linear_acceleration.y * 0.01,
            msg.linear_acceleration.z * 0.01
        ], dtype=np.float32)
        
        # Projected gravity from orientation
        quat = msg.orientation
        # Convert quaternion to projected gravity (simplified)
        self.obs_buffer['projected_gravity'] = np.array([
            2 * (quat.x * quat.z - quat.w * quat.y),
            2 * (quat.y * quat.z + quat.w * quat.x),
            1 - 2 * (quat.x * quat.x + quat.y * quat.y)
        ], dtype=np.float32) * -9.81
        
    def _cmd_vel_callback(self, msg: Twist):
        """Process velocity command messages."""
        self.obs_buffer['commands'] = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.angular.z
        ], dtype=np.float32)
        
    def _construct_observation(self) -> np.ndarray:
        """Construct observation vector for policy."""
        obs = np.concatenate([
            self.obs_buffer['base_lin_vel'],
            self.obs_buffer['base_ang_vel'],
            self.obs_buffer['projected_gravity'],
            self.obs_buffer['commands'],
            self.obs_buffer['joint_pos'],
            self.obs_buffer['joint_vel'],
            self.obs_buffer['prev_actions'],
        ])
        return obs.reshape(1, -1).astype(np.float32)
    
    def _detect_state(self) -> PolicyState:
        """Detect robot state for automatic policy switching."""
        
        # Check for fallen state
        gravity = self.obs_buffer['projected_gravity']
        if np.abs(gravity[0]) > 3.0 or np.abs(gravity[1]) > 3.0:
            return PolicyState.FALLEN
        
        # Check for stuck state
        velocity = np.linalg.norm(self.obs_buffer['base_lin_vel'][:2])
        commanded = np.linalg.norm(self.obs_buffer['commands'][:2])
        
        if commanded > 0.1 and velocity < self.get_parameter('stuck_velocity_threshold').value:
            if self.stuck_start_time is None:
                self.stuck_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            
            stuck_duration = self.get_clock().now().seconds_nanoseconds()[0] - self.stuck_start_time
            if stuck_duration > self.get_parameter('stuck_duration').value:
                return PolicyState.STUCK
        else:
            self.stuck_start_time = None
            
        return PolicyState.NORMAL
    
    def _select_policy(self, state: PolicyState) -> str:
        """Select appropriate policy for current state."""
        
        policy_map = {
            PolicyState.NORMAL: self.default_policy_name,
            PolicyState.STUCK: 'recovery_stuck',
            PolicyState.FALLEN: 'recovery_fallen',
            PolicyState.RECOVERY: 'recovery_stuck',
        }
        
        selected = policy_map.get(state, self.default_policy_name)
        
        # Fall back to default if selected policy not available
        if selected not in self.policies:
            selected = self.default_policy_name
            
        return selected
    
    def _run_inference(self, policy_name: str, observation: np.ndarray) -> np.ndarray:
        """Run policy inference."""
        
        if policy_name not in self.policies:
            return np.zeros(12, dtype=np.float32)
        
        session = self.policies[policy_name]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        action = session.run([output_name], {input_name: observation})[0]
        return action.flatten()
    
    def _control_loop(self):
        """Main control loop."""
        
        if not self.policies:
            return
        
        # Detect state
        if self.get_parameter('auto_switch_enabled').value:
            self.current_state = self._detect_state()
            
        # Select policy
        self.current_policy_name = self._select_policy(self.current_state)
        
        # Construct observation
        observation = self._construct_observation()
        
        # Run inference
        action = self._run_inference(self.current_policy_name, observation)
        
        # Update previous actions
        self.obs_buffer['prev_actions'] = action.copy()
        
        # Publish action
        action_msg = Float32MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)
        
        # Publish current policy name
        policy_msg = String()
        policy_msg.data = f'{self.current_policy_name} ({self.current_state.value})'
        self.current_policy_pub.publish(policy_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
