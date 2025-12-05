#!/usr/bin/env python3
"""
ARC-M Magistral Planner Node
High-level reasoning using Mistral LLM.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import threading

class MagistralPlanner(Node):
    """
    LLM-based high-level planner for recovery strategies.
    Uses Mistral API to analyze robot state and suggest strategic actions.
    """
    
    def __init__(self):
        super().__init__('magistral_planner')
        
        # Load API key
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            self.get_logger().warn("MISTRAL_API_KEY not set. Planner disabled.")
        
        self.plan_pub = self.create_publisher(String, '/planner/current_plan', 10)
        
        # Timer for planning (slow loop)
        self.create_timer(5.0, self._planning_loop)
        
        self.get_logger().info('Magistral Planner initialized')
        
    def _planning_loop(self):
        """Periodic planning cycle."""
        if not self.api_key:
            return
            
        # In a real impl, we'd query the LLM here
        # For now, just publish status
        msg = String()
        msg.data = "Monitoring robot state..."
        self.plan_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MagistralPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
