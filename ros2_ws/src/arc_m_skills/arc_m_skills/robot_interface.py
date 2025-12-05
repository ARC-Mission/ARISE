#!/usr/bin/env python3
"""
ARC-M Robot Interface Node
Hardware abstraction layer for robot control.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32MultiArray

class RobotInterface(Node):
    """
    Interface for robot hardware or simulation bridge.
    Abducts hardware specifics and exposes standard ROS 2 interfaces.
    """
    
    def __init__(self):
        super().__init__('robot_interface')
        
        # Load parameters
        self.declare_parameter('robot_type', 'anymal_d')
        self.declare_parameter('control_mode', 'position')
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        
        # Subscribers
        self.action_sub = self.create_subscription(
            Float32MultiArray, 
            '/policy/action', 
            self._action_callback, 
            10
        )
        
        # Simulaton timer
        self.create_timer(0.01, self._sim_loop)
        
        self.get_logger().info('Robot Interface initialized')
        
    def _action_callback(self, msg):
        """Handle control actions from policy."""
        # In real robot, send to hardware SDK
        # In sim, this might be handled by bridge, but we log here for debug
        pass
        
    def _sim_loop(self):
        """Simulate hardware reading (placeholder)."""
        # Publish dummy joint states
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in range(12)]
        msg.position = [0.0] * 12
        msg.velocity = [0.0] * 12
        msg.effort = [0.0] * 12
        self.joint_state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
