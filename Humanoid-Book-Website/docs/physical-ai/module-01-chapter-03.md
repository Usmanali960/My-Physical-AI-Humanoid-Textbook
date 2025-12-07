---
id: module-01-chapter-03
title: Chapter 03 - Python rclpy Integration
sidebar_position: 3
---

# Chapter 03 - Python rclpy Integration

## Table of Contents
- [Overview](#overview)
- [Why Python for Robotics?](#why-python-for-robotics)
- [Introduction to rclpy](#introduction-to-rclpy)
- [Setting Up Python Development](#setting-up-python-development)
- [Creating Nodes with rclpy](#creating-nodes-with-rclpy)
- [Publishing and Subscribing](#publishing-and-subscribing)
- [Services and Actions](#services-and-actions)
- [Working with Parameters](#working-with-parameters)
- [Advanced rclpy Concepts](#advanced-rclpy-concepts)
- [Best Practices for Humanoid Robotics](#best-practices-for-humanoid-robotics)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Python has become one of the most popular languages for robotics development, and for good reason. Its simplicity, extensive libraries, and strong community support make it an ideal choice for prototyping and implementing robot behaviors. In this chapter, we'll explore rclpy, the Python client library for ROS 2, and see how to use it effectively in humanoid robotics applications.

rclpy provides a Python interface to ROS 2, allowing you to create nodes, publish/subscribe to topics, provide/request services, and manage parameters—all essential building blocks for humanoid robot software.

## Why Python for Robotics?

### Advantages

1. **Rapid Prototyping**: Python's concise syntax allows for quick development and testing of ideas
2. **Rich Ecosystem**: Libraries like NumPy, SciPy, OpenCV, and TensorFlow make complex robotics algorithms accessible
3. **Readability**: Code is easier to understand, making collaboration and maintenance simpler
4. **Community Support**: Large community with extensive documentation and examples
5. **Integration**: Easy integration with other languages and systems

### When to Use Python vs. C++

While Python excels at high-level planning and control, C++ is typically used for performance-critical components like:
- Real-time control loops (kHz frequencies)
- Low-level sensor/actuator interfaces
- Computationally intensive algorithms

For humanoid robotics:
- **Python**: High-level behaviors, path planning, vision processing, user interfaces
- **C++**: Joint controllers, real-time safety systems, sensor fusion, inverse kinematics

## Introduction to rclpy

### What is rclpy?

rclpy is the Python client library for ROS 2. It provides Python bindings for the ROS 2 middleware (rcl) and allows Python programs to interact with ROS 2.

### Key Components

- **Node**: The basic execution unit
- **Publisher/Subscriber**: For topic communication
- **Service/Client**: For request/response communication
- **Action Server/Client**: For goal-oriented communication with feedback
- **Timer**: For periodic execution
- **Parameter**: For runtime configuration

### Installation and Setup

rclpy is typically installed with ROS 2. Verify your installation with:

```python
import rclpy
print(rclpy.__version__)
```

## Setting Up Python Development

### Virtual Environments

It's recommended to use virtual environments for Python robotics projects:

```bash
python -m venv ros2_env
source ros2_env/bin/activate  # On Windows: ros2_env\Scripts\activate
```

### Project Structure

```
my_humanoid_project/
├── nodes/
│   ├── sensor_processing.py
│   ├── behavior_controller.py
│   └── high_level_planner.py
├── launch/
├── config/
├── test/
└── setup.py
```

### Package Dependencies

For robotics development, common dependencies include:
- `numpy`: Numerical computations
- `scipy`: Scientific computing
- `opencv-python`: Computer vision
- `matplotlib`: Visualization
- `pybullet`: Physics simulation

## Creating Nodes with rclpy

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class HumanoidNode(Node):
    def __init__(self):
        super().__init__('humanoid_node')
        self.get_logger().info('Humanoid node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node with Parameters

```python
import rclpy
from rclpy.node import Node

class ConfigurableNode(Node):
    def __init__(self):
        super().__init__('configurable_node')
        
        # Declare parameters with defaults
        self.declare_parameter('robot_name', 'humanoid')
        self.declare_parameter('control_rate', 100)
        self.declare_parameter('safety_threshold', 0.5)
        
        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_rate = self.get_parameter('control_rate').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        
        self.get_logger().info(f'Initialized for robot: {self.robot_name}')

def main(args=None):
    rclpy.init(args=args)
    node = ConfigurableNode()
    
    # Spin with a rate limiter
    rate = node.create_rate(node.control_rate)  # Hz
    try:
        while rclpy.ok():
            # Your control logic here
            node.get_logger().info(f'Control loop running for {node.robot_name}')
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Lifecycle Nodes in Python

```python
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.executors import SingleThreadedExecutor

class LifecycleHumanoidNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_humanoid_node')
        self.get_logger().info('Lifecycle node created')

    def on_configure(self, state):
        self.get_logger().info('Configuring')
        # Initialize resources here
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating')
        # Activate subscribers/publishers
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating')
        # Deactivate subscribers/publishers
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up')
        # Clean up resources
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleHumanoidNode()
    
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishing and Subscribing

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer to periodically publish data
        self.timer = self.create_timer(0.01, self.publish_joint_state)  # 100Hz
        
        # Initialize joint data
        self.joint_names = ['left_hip', 'left_knee', 'left_ankle', 
                           'right_hip', 'right_knee', 'right_ankle']
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
    def publish_joint_state(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        
        # Create subscriber
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10  # QoS depth
        )
        
        # Store latest joint state
        self.latest_joint_state = None
        
    def joint_state_callback(self, msg):
        self.latest_joint_state = msg
        self.get_logger().info(f'Received joint state with {len(msg.name)} joints')
        
        # Example: Check if any joint is at limit
        for i, pos in enumerate(msg.position):
            if abs(pos) > 3.0:  # Check if joint is near limit (3 radians)
                self.get_logger().warn(f'Joint {msg.name[i]} near limit: {pos}')

def main(args=None):
    rclpy.init(args=args)
    node = JointStateSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Publisher/Subscriber Pattern

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionProcessingNode(Node):
    def __init__(self):
        super().__init__('vision_processing_node')
        
        # Create publisher for processed image
        self.processed_image_pub = self.create_publisher(Image, 'processed_image', 10)
        
        # Create subscriber for raw image
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        # Create subscriber for joint states (for head position)
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Store head joint position
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Perform vision processing (example: face detection)
        processed_image = self.detect_faces(cv_image)
        
        # Convert back to ROS Image
        processed_msg = self.cv_bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
        processed_msg.header = msg.header
        
        # Publish processed image
        self.processed_image_pub.publish(processed_msg)
        
    def joint_state_callback(self, msg):
        # Update head joint positions
        for i, name in enumerate(msg.name):
            if name == 'head_yaw_joint':
                self.head_yaw = msg.position[i]
            elif name == 'head_pitch_joint':
                self.head_pitch = msg.position[i]
                
    def detect_faces(self, image):
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        return image

def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and Actions

### Creating a Service Server

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger, SetBool
from humanoid_msgs.srv import SetGait  # Custom service
import time

class HumanoidServiceServer(Node):
    def __init__(self):
        super().__init__('humanoid_service_server')
        
        # Create service for homing joints
        self.homing_service = self.create_service(
            Trigger, 
            'home_joints', 
            self.homing_callback
        )
        
        # Create service for enabling/disabling robot
        self.enable_service = self.create_service(
            SetBool,
            'set_enabled',
            self.enable_callback
        )
        
        # Create custom service for setting gait
        self.gait_service = self.create_service(
            SetGait,
            'set_gait',
            self.gait_callback
        )
        
        self.enabled = False
        
    def homing_callback(self, request, response):
        self.get_logger().info('Homing joints initiated')
        
        # Simulate homing process
        time.sleep(2)
        
        response.success = True
        response.message = 'All joints homed successfully'
        self.get_logger().info('Homing completed')
        return response
        
    def enable_callback(self, request, response):
        self.enabled = request.data
        self.get_logger().info(f'Robot {"enabled" if self.enabled else "disabled"}')
        
        response.success = True
        response.message = f'Robot {"enabled" if self.enabled else "disabled"}'
        return response
        
    def gait_callback(self, request, response):
        if not self.enabled:
            response.success = False
            response.message = 'Robot not enabled'
            return response
            
        # Process gait request
        gait_type = request.gait_type
        speed = request.speed
        
        self.get_logger().info(f'Setting gait: {gait_type} at speed {speed}')
        
        # Here you would implement the actual gait setting logic
        response.success = True
        response.message = f'Gait {gait_type} set successfully'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidServiceServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Service Client

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger, SetBool
from humanoid_msgs.srv import SetGait

class ServiceClientDemo(Node):
    def __init__(self):
        super().__init__('service_client_demo')
        
        # Create clients
        self.homing_client = self.create_client(Trigger, 'home_joints')
        self.enable_client = self.create_client(SetBool, 'set_enabled')
        self.gait_client = self.create_client(SetGait, 'set_gait')
        
        # Wait for services to be available
        while not self.homing_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Homing service not available, waiting...')
        while not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Enable service not available, waiting...')
        while not self.gait_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gait service not available, waiting...')
        
        self.req = Trigger.Request()
        
    def home_joints(self):
        self.get_logger().info('Sending homing request')
        future = self.homing_client.call_async(self.req)
        # Process response asynchronously or synchronously
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(f'Homing result: {response.success}, {response.message}')
        
    def enable_robot(self, enable=True):
        req = SetBool.Request()
        req.data = enable
        future = self.enable_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(f'Enable result: {response.success}, {response.message}')
        
    def set_gait(self, gait_type, speed):
        req = SetGait.Request()
        req.gait_type = gait_type
        req.speed = speed
        future = self.gait_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(f'Set gait result: {response.success}, {response.message}')

def main(args=None):
    rclpy.init(args=args)
    client = ServiceClientDemo()
    
    # Example usage
    client.enable_robot(True)
    client.home_joints()
    client.set_gait('walk', 0.5)
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Working with Actions

Actions are perfect for long-running tasks with feedback, which is common in humanoid robotics:

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
from humanoid_msgs.action import WalkToPose  # Custom action

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')
        
        # Create action server with reentrant callback group for concurrency
        self._action_server = ActionServer(
            self,
            WalkToPose,
            'walk_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info('Received goal request')
        # Check if the goal is valid
        if goal_request.target_pose.position.z != 0.0:
            self.get_logger().warn('Walking to non-zero Z position not supported')
            return GoalResponse.REJECT
            
        return GoalResponse.ACCEPT
        
    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
        
    def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        
        # Get the goal
        target_pose = goal_handle.request.target_pose
        tolerance = goal_handle.request.tolerance
        
        # Feedback and result messages
        feedback_msg = WalkToPose.Feedback()
        result_msg = WalkToPose.Result()
        
        # Simulate walking
        steps = 50  # Number of steps in the simulation
        for i in range(steps):
            if goal_handle.is_cancel_requested:
                result_msg.success = False
                result_msg.message = 'Goal canceled'
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return result_msg
                
            # Update feedback
            progress = float(i) / float(steps)
            feedback_msg.current_pose = target_pose  # In a real implementation, this would be the actual pose
            feedback_msg.distance_remaining = 1.0 - progress  # In a real implementation, calculate actual distance
            
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Walking... {(progress*100):.2f}% complete')
            
            # Simulate walking by sleeping
            time.sleep(0.1)
        
        # Check if we reached the goal (in a real implementation, check actual robot position)
        result_msg.success = True
        result_msg.message = 'Successfully reached target pose'
        goal_handle.succeed()
        
        self.get_logger().info('Goal succeeded')
        return result_msg

def main(args=None):
    rclpy.init(args=args)
    node = WalkActionServer()
    
    # Use multi-threaded executor to handle multiple goals
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Parameters

### Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterDemoNode(Node):
    def __init__(self):
        super().__init__('parameter_demo_node')
        
        # Declare parameters with defaults and descriptions
        self.declare_parameter('robot_name', 'nao', 
                              ParameterDescriptor(description='Name of the robot'))
        self.declare_parameter('walking_speed', 0.5, 
                              ParameterDescriptor(description='Default walking speed (m/s)'))
        self.declare_parameter('max_joint_speed', 1.0, 
                              ParameterDescriptor(description='Maximum joint velocity (rad/s)'))
        
        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.walking_speed = self.get_parameter('walking_speed').value
        self.max_joint_speed = self.get_parameter('max_joint_speed').value
        
        # Set up parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.get_logger().info(f'Initialized with robot: {self.robot_name}')
        
    def parameter_callback(self, params):
        """Callback for parameter changes."""
        for param in params:
            if param.name == 'walking_speed' and param.type_ == Parameter.Type.DOUBLE:
                if param.value < 0.0 or param.value > 2.0:
                    self.get_logger().warn(f'Invalid walking speed: {param.value}, clamping to range [0.0, 2.0]')
                    return SetParametersResult(successful=False)
                else:
                    self.get_logger().info(f'Walking speed changed to: {param.value}')
                    self.walking_speed = param.value
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterDemoNode()
    
    # Change parameter programmatically
    node.set_parameters([Parameter('walking_speed', Parameter.Type.DOUBLE, 0.7)])
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Parameter with YAML Configuration

```python
import rclpy
from rclpy.node import Node

class ConfigurableHumanoidNode(Node):
    def __init__(self):
        super().__init__('configurable_humanoid_node')
        
        # Declare parameters that will be loaded from YAML
        self.declare_parameter('robot_description_file', '')
        self.declare_parameter('controller_configs', {})
        self.declare_parameter('safety_limits', {})
        
        # Get parameter values
        self.robot_description_file = self.get_parameter('robot_description_file').value
        self.controller_configs = self.get_parameter('controller_configs').value
        self.safety_limits = self.get_parameter('safety_limits').value
        
        self.get_logger().info(f'Loaded robot description from: {self.robot_description_file}')
        
        # Use the parameters in your initialization
        self.initialize_robot()

    def initialize_robot(self):
        """Initialize the robot using the loaded parameters."""
        # Implementation would use the loaded configuration
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ConfigurableHumanoidNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Example YAML configuration file (`config/humanoid_config.yaml`):
```yaml
configurable_humanoid_node:
  ros__parameters:
    robot_description_file: "package://my_robot_description/urdf/robot.urdf"
    controller_configs:
      joint_state_controller:
        type: joint_state_controller/JointStateController
      position_controllers:
        type: position_controllers/JointGroupPositionController
        joints: ["hip_joint", "knee_joint", "ankle_joint"]
    safety_limits:
      max_velocity: 2.0
      max_torque: 100.0
      max_temperature: 80.0
```

## Advanced rclpy Concepts

### Custom Callback Groups

```python
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class AdvancedCallbackNode(Node):
    def __init__(self):
        super().__init__('advanced_callback_node')
        
        # Create different callback groups
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.background_group = ReentrantCallbackGroup()
        
        # Create subscribers with different callback groups
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10,
            callback_group=self.sensor_group
        )
        
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            10,
            callback_group=self.control_group
        )
        
        # Create timer with background group
        self.status_timer = self.create_timer(
            1.0,  # 1 second
            self.status_callback,
            callback_group=self.background_group
        )
        
    def joint_callback(self, msg):
        self.get_logger().info(f'Received joint state with {len(msg.name)} joints')
        # Simulate processing time
        import time
        time.sleep(0.05)
        
    def cmd_callback(self, msg):
        self.get_logger().info(f'Received cmd_vel: {msg.linear.x}, {msg.angular.z}')
        # Simulate processing time
        import time
        time.sleep(0.03)
        
    def status_callback(self):
        self.get_logger().info('Status update')

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedCallbackNode()
    
    # Use multi-threaded executor to handle different callback groups
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Async/Await Support

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
import asyncio
from humanoid_msgs.action import WalkToPose

class AsyncNode(Node):
    def __init__(self):
        super().__init__('async_node')
        
        # Create publisher
        self.publisher = self.create_publisher(String, 'async_status', 10)
        
        # Create action client
        self._action_client = ActionClient(self, WalkToPose, 'walk_to_pose')
        
    async def async_service_call(self):
        """Simulate an async service call."""
        self.get_logger().info('Starting async operation')
        await asyncio.sleep(2)  # Simulate async work
        self.get_logger().info('Async operation completed')
        return "Operation successful"
        
    async def async_action_client(self):
        """Send an action goal asynchronously."""
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        # Create goal
        goal_msg = WalkToPose.Goal()
        goal_msg.target_pose.position.x = 1.0
        goal_msg.target_pose.position.y = 0.0
        goal_msg.target_pose.position.z = 0.0
        goal_msg.target_pose.orientation.w = 1.0
        goal_msg.tolerance = 0.05
        
        # Send goal asynchronously
        self.get_logger().info('Sending goal...')
        send_goal_future = await self._action_client.send_goal_async(goal_msg)
        
        # Wait for result
        goal_handle = await send_goal_future
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return False
            
        self.get_logger().info('Goal accepted, waiting for result...')
        get_result_future = await goal_handle.get_result_async()
        result = await get_result_future
        self.get_logger().info(f'Action completed: {result.result.success}')
        
        return result.result.success

def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    node = AsyncNode()
    
    async def run_async_tasks():
        # Run multiple async operations concurrently
        task1 = node.async_service_call()
        task2 = node.async_action_client()
        
        results = await asyncio.gather(task1, task2)
        node.get_logger().info(f'All async tasks completed: {results}')
    
    # Run the async function
    rclpy.spin_once(node, timeout_sec=0)  # Process any pending callbacks first
    asyncio.run(run_async_tasks())
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid Robotics

### Error Handling and Safety

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import traceback

class SafeHumanoidNode(Node):
    def __init__(self):
        super().__init__('safe_humanoid_node')
        
        # Store latest joint state with timestamp
        self.last_joint_state = None
        self.last_joint_state_time = None
        
        # Create subscriber with appropriate QoS
        qos_profile = QoSProfile(depth=1, reliability=2)  # Reliable, depth 1
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        # Safety timer - check for data timeouts
        self.safety_timer = self.create_timer(0.1, self.safety_check)
        
    def joint_state_callback(self, msg):
        try:
            # Validate joint state message
            if len(msg.name) != len(msg.position):
                self.get_logger().error('Joint state name/position length mismatch')
                return
                
            # Store with timestamp
            self.last_joint_state = msg
            self.last_joint_state_time = self.get_clock().now()
            
            # Perform safety checks on joint positions
            for i, pos in enumerate(msg.position):
                # Check for extreme values (example: joint limits)
                if abs(pos) > 5.0:  # Assume 5 rad is extreme
                    self.get_logger().warn(f'Joint {msg.name[i]} at extreme position: {pos}')
                    
        except Exception as e:
            self.get_logger().error(f'Error processing joint state: {e}')
            self.get_logger().error(traceback.format_exc())
            
    def safety_check(self):
        """Check for safety conditions."""
        if self.last_joint_state_time is not None:
            # Check if joint data is too old (>100ms)
            time_diff = self.get_clock().now() - self.last_joint_state_time
            if time_diff.nanoseconds * 1e-9 > 0.1:  # More than 100ms old
                self.get_logger().error('Joint state data timeout - implementing safety stop')
                self.emergency_stop()
                
    def emergency_stop(self):
        """Implement emergency stop logic."""
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        # Publish zero velocity commands, disable actuators, etc.
        # Implementation would depend on your robot's control system

def main(args=None):
    rclpy.init(args=args)
    node = SafeHumanoidNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Resource Management

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
import threading
from collections import deque

class EfficientHumanoidNode(Node):
    def __init__(self):
        super().__init__('efficient_humanoid_node')
        
        # Use deques for time-series data instead of lists
        self.joint_positions = deque(maxlen=100)  # Only keep last 100 values
        self.timestamps = deque(maxlen=100)
        
        # Pre-allocate numpy arrays for computations
        self.computation_buffer = np.zeros(100, dtype=np.float64)
        
        # Threading lock for shared data
        self.data_lock = threading.Lock()
        
        # Create publisher for processed data
        qos_profile = QoSProfile(depth=10)
        self.result_pub = self.create_publisher(JointState, 'processed_joint_states', qos_profile)
        
    def add_joint_data(self, joint_name, position, timestamp):
        """Thread-safe addition of joint data."""
        with self.data_lock:
            self.joint_positions.append(position)
            self.timestamps.append(timestamp)
            
    def process_joint_data(self):
        """Process joint data efficiently."""
        with self.data_lock:
            # Convert to numpy array for efficient computation
            if len(self.joint_positions) > 10:  # Need at least 10 points
                positions = np.array(list(self.joint_positions))
                
                # Efficient filtering using numpy
                filtered_positions = np.convolve(positions, np.ones(5)/5, mode='valid')
                
                # Use pre-allocated buffer
                self.computation_buffer[:len(filtered_positions)] = filtered_positions
                
                # Create result message
                result_msg = JointState()
                result_msg.position = filtered_positions.tolist()
                
                # Publish result
                self.result_pub.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EfficientHumanoidNode()
    
    # Process data periodically
    timer = node.create_timer(0.01, node.process_joint_data)  # 100 Hz
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this chapter, we've thoroughly explored rclpy, the Python client library for ROS 2, demonstrating how it enables the development of humanoid robotics applications. We covered everything from basic node creation to advanced concepts like async programming and proper safety handling.

Python with rclpy provides an excellent platform for implementing the high-level behaviors and control logic needed in humanoid robots. Its ease of use and rich ecosystem make it ideal for rapid development of complex algorithms like path planning, vision processing, and behavior trees.

## Exercises

1. Create a Python node that subscribes to joint states and publishes a filtered version of the joint velocities using a simple moving average filter.

2. Implement a Python node that acts as a client for a custom service that computes inverse kinematics for a humanoid arm. The service should take a target end-effector pose and return joint angles.

3. Create a parameter server node that manages robot-specific configurations for a humanoid robot, including joint limits, control gains, and safety thresholds. Create a launch file to load parameters from a YAML file.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
