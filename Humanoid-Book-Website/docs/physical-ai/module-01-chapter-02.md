---
id: module-01-chapter-02
title: Chapter 02 - Nodes, Topics, Services
sidebar_position: 2
---

# Chapter 02 - Nodes, Topics, Services

## Table of Contents
- [Overview](#overview)
- [Nodes in Depth](#nodes-in-depth)
- [Topics and Message Passing](#topics-and-message-passing)
- [Services for Request/Response](#services-for-requestresponse)
- [Practical Examples with Humanoid Robots](#practical-examples-with-humanoid-robots)
- [Quality of Service (QoS) Settings](#quality-of-service-qos-settings)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

In the previous chapter, we introduced the core concepts of ROS 2. Now, we'll dive deeper into three fundamental communication patterns: nodes, topics, and services. These patterns form the backbone of any ROS 2 application and are especially critical in humanoid robotics where multiple subsystems need to communicate efficiently and reliably.

Understanding these communication patterns is essential for building robust humanoid robots, as they determine how sensory information flows through the system, how control commands are distributed, and how different components coordinate their activities.

## Nodes in Depth

### What is a Node?

A node is an executable process that uses ROS 2 to communicate with other nodes. In humanoid robotics, nodes represent different components of the robot:

- **Sensor Nodes**: Camera, IMU, force sensors
- **Controller Nodes**: Joint controllers, walking controllers
- **Processing Nodes**: Vision processing, path planning
- **Interface Nodes**: Human-robot interaction, remote control

### Creating a Node

Here's the basic structure of a ROS 2 node in Python:

```python
import rclpy
from rclpy.node import Node

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.get_logger().info('Simple node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

ROS 2 nodes can be configured with lifecycle management, which is particularly important for humanoid robots that need to transition through different states safely:

1. **Unconfigured**: Node created but not configured
2. **Inactive**: Node configured but not running
3. **Active**: Node running and operational
4. **Finalized**: Node shut down and cleaned up

This pattern is crucial for humanoid robots that need to go through calibration, homing, and safety checks.

### Node Parameters

Nodes can accept parameters to customize their behavior:

```python
self.declare_parameter('robot_name', 'humanoid')
self.declare_parameter('control_rate', 100)
```

Parameters allow the same node to behave differently across different robot platforms.

## Topics and Message Passing

### Publish-Subscribe Pattern

Topics implement a publish-subscribe communication pattern where:
- **Publishers** send messages to a topic
- **Subscribers** receive messages from a topic
- Communication is asynchronous and loosely coupled

This pattern is ideal for continuous data streams like sensor data, joint states, or camera images.

### Message Types

ROS 2 comes with many standard message types defined in ROS message definition files (.msg). Common message types in humanoid robotics:

- **sensor_msgs**: Camera images, IMU data, joint states
- **geometry_msgs**: Positions, orientations, velocities
- **std_msgs**: Basic data types (int, float, string)
- **trajectory_msgs**: Joint trajectories for motion planning

### Creating Custom Messages

For humanoid-specific applications, you might need custom messages. For example, a message for humanoid balance data:

```
# HumanoidState.msg
std_msgs/Header header
geometry_msgs/Pose torso_pose
sensor_msgs/JointState joint_states
geometry_msgs/Twist com_velocity
float64[] zmp_positions  # Zero Moment Point positions
```

### Quality of Service (QoS) for Topics

QoS settings allow you to customize how messages are delivered:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)
```

For humanoid robots:
- **Joint states**: Use reliable delivery with small buffer
- **Camera images**: Use best-effort delivery with larger buffer
- **IMU data**: Use reliable delivery with small buffer

### Example: Publishing Joint States

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.01, self.publish_joint_state)  # 100 Hz
        self.joint_names = ['hip_joint', 'knee_joint', 'ankle_joint']
        self.position = [0.0, 0.0, 0.0]
        
    def publish_joint_state(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.position
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        self.publisher.publish(msg)
        
        # Update position for demonstration
        self.position = [math.sin(self.get_clock().now().nanoseconds * 1e-9),
                         math.cos(self.get_clock().now().nanoseconds * 1e-9),
                         0.0]

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Services for Request/Response

### When to Use Services

Services implement a request-response pattern where:
- A client sends a request
- A server processes the request and sends a response
- Communication is synchronous

Services are ideal for operations that need a guaranteed response, such as:
- Calibration routines
- State changes (sitting, standing)
- Configuration updates
- Emergency stops

### Service Types

Standard service types include:
- **std_srvs**: Basic services like trigger or set_bool
- **sensor_msgs**: Services related to sensors
- **custom services**: User-defined service types (.srv files)

### Creating Custom Services

For humanoid robots, you might need custom services. For example, a service to change the walking gait:

```
# SetGait.srv
string gait_type  # 'walk', 'trot', 'amble'
float64 speed     # Speed of the gait
---
bool success      # Whether the gait was successfully set
string message    # Additional information
```

### Example: Service Server for Joint Homing

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import time

class JointHomingServer(Node):
    def __init__(self):
        super().__init__('joint_homing_server')
        self.srv = self.create_service(Trigger, 'home_joints', self.homing_callback)
        
    def homing_callback(self, request, response):
        self.get_logger().info('Homing joints initiated')
        # Simulate homing process
        time.sleep(2)
        
        response.success = True
        response.message = 'All joints homed successfully'
        self.get_logger().info('Homing completed')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = JointHomingServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Example: Service Client for Joint Homing

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class HomingClient(Node):
    def __init__(self):
        super().__init__('homing_client')
        self.cli = self.create_client(Trigger, 'home_joints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = Trigger.Request()
        
    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = HomingClient()
    response = client.send_request()
    if response is not None:
        print(f'Result: {response.success}, {response.message}')
    else:
        print('Service call failed')
    client.destroy_node()
    rclpy.shutdown()
```

## Practical Examples with Humanoid Robots

### Sensor Data Processing Pipeline

A typical sensor data processing pipeline in a humanoid robot might look like:

```
Camera Node (publishes /camera/image_raw)
         ↓ (image topic)
Image Processing Node (subscribes to /camera/image_raw, 
                    processes image, publishes /object_detected)
         ↓ (object detection topic)
Behavior Node (subscribes to /object_detected, 
            decides robot behavior)
```

This decoupled approach allows each component to develop and run independently.

### Control Architecture

A humanoid's control system might have:

- **High-level planner**: Decides where to walk, what actions to perform
- **Mid-level controller**: Translates high-level goals into joint commands
- **Low-level controller**: Executes precise joint movements

Each communicates via topics and services:
- Topic: `/joint_commands` (high-level to mid-level)
- Topic: `/joint_states` (from sensors back to controllers)
- Service: `/execute_behavior` (for triggering specific behaviors)

### Example: Simple Humanoid Control Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        
        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)
        
        # Command subscriber (e.g., from navigation)
        self.cmd_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        self.current_joint_states = JointState()
        
    def joint_state_callback(self, msg):
        self.current_joint_states = msg
        # Process joint states, implement control logic
        
    def cmd_vel_callback(self, msg):
        # Convert velocity command to joint commands
        joint_commands = Float64MultiArray()
        # Implementation would convert Twist to joint movements
        
        self.joint_cmd_pub.publish(joint_commands)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Quality of Service (QoS) Settings

### Understanding QoS

QoS settings control how messages are delivered, which is crucial for humanoid robots with real-time requirements:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# For critical data like joint states
qos_critical = QoSProfile(
    depth=5,  # Small buffer since we only need recent data
    reliability=ReliabilityPolicy.RELIABLE,  # Must receive all messages
    durability=DurabilityPolicy.VOLATILE,  # Don't need to keep old messages
    history=HistoryPolicy.KEEP_LAST  # Keep only the last N messages
)

# For best-effort data like images
qos_best_effort = QoSProfile(
    depth=10,  # Larger buffer for more data points
    reliability=ReliabilityPolicy.BEST_EFFORT,  # OK to drop some messages
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)
```

### QoS for Humanoid Applications

| Data Type | Reliability | Depth | History | Reason |
|-----------|-------------|-------|---------|--------|
| Joint states | RELIABLE | 1-5 | KEEP_LAST | Critical for safety |
| IMU data | RELIABLE | 1-10 | KEEP_LAST | Needed for balance |
| Camera images | BEST_EFFORT | 5-15 | KEEP_LAST | OK to drop frames |
| LIDAR scans | BEST_EFFORT | 1-5 | KEEP_LAST | Prefer fresh data |

## Summary

In this chapter, we've explored the fundamental communication patterns in ROS 2: nodes, topics, and services. We've seen how these patterns enable the complex, distributed nature of humanoid robotics applications, with real examples of how they might be used in a humanoid robot system.

Understanding these communication patterns is essential for developing robust, maintainable humanoid robot software. The publish-subscribe pattern via topics handles continuous data streams like sensor readings, while services provide guaranteed request-response communication for critical operations.

## Exercises

1. Create a simple ROS 2 node that publishes joint states for a 6-DOF arm. Include position, velocity, and effort for each joint.

2. Design custom message types for a humanoid robot's specific needs:
   - A message for robot balance state (center of mass, ZMP, stability measures)
   - A message for describing walking gaits
   - A service to transition between different robot states (sitting, standing, walking)

3. Implement a node that subscribes to IMU data and publishes a simplified balance state based on the received information.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
