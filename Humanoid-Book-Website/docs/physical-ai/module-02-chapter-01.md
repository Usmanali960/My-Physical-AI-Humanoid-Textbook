---
id: module-02-chapter-01
title: Chapter 01 - Gazebo Simulation Basics
sidebar_position: 5
---

# Chapter 01 - Gazebo Simulation Basics

## Table of Contents
- [Overview](#overview)
- [Introduction to Gazebo](#introduction-to-gazebo)
- [Gazebo Architecture](#gazebo-architecture)
- [Setting Up Gazebo for Humanoid Robotics](#setting-up-gazebo-for-humanoid-robotics)
- [Basic Simulation Concepts](#basic-simulation-concepts)
- [Creating Your First Humanoid Simulation](#creating-your-first-humanoid-simulation)
- [SDF vs. URDF in Gazebo](#sdf-vs-urdf-in-gazebo)
- [Physics Engine Fundamentals](#physics-engine-fundamentals)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Gazebo is a powerful 3D simulation environment that plays a crucial role in humanoid robotics development. It allows you to test algorithms, validate control systems, and develop robot behaviors in a safe, repeatable environment before deploying to physical hardware. In this chapter, we'll build a foundation in Gazebo simulation, focusing on concepts and techniques specifically relevant to humanoid robots.

Simulation is particularly important for humanoid robotics because these systems are complex, expensive to build, and potentially dangerous if control algorithms fail. Gazebo enables rapid iteration and testing of complex behaviors without risk to hardware or humans.

## Introduction to Gazebo

### What is Gazebo?

Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides:

- **Realistic physics simulation**: Accurate modeling of forces, collisions, and dynamics
- **High-quality graphics**: Photo-realistic rendering and visualization
- **Extensive sensor simulation**: Cameras, LIDAR, IMUs, force/torque sensors, and more
- **Multiple physics engines**: Options like ODE, Bullet, Simbody, and DART
- **ROS integration**: Seamless integration with the Robot Operating System

### Why Gazebo for Humanoid Robotics?

Humanoid robots present unique challenges that make simulation particularly valuable:

1. **Complex kinematics**: Multiple limbs and joints require coordinated control
2. **Balance and locomotion**: Bipedal walking is inherently unstable
3. **Expensive hardware**: Physical humanoid robots are costly to build and maintain
4. **Safety concerns**: Testing on physical robots can damage hardware or cause injury
5. **Repeatability**: Simulation allows for consistent testing conditions

### Gazebo vs. Other Simulation Platforms

Compared to other simulation environments:

- **Webots**: Gazebo has more complex scene simulation capabilities
- **PyBullet**: Gazebo offers better visualization and ROS integration
- **Mujoco**: Gazebo is open-source and has more robotic-specific features
- **Unity**: Gazebo is specifically designed for robotics applications

## Gazebo Architecture

### Core Components

Gazebo's architecture consists of several key components:

1. **Server (gazebo)**: The core simulation engine that handles physics, rendering, and sensors
2. **Client (gzclient)**: The visualization interface that connects to the server
3. **Plugins**: Extensible components that add functionality to models and the world
4. **Transport layer**: Handles message passing between components

### Plugin System

Gazebo's plugin system is one of its most powerful features:

- **Model plugins**: Attach to specific models to provide custom behavior
- **World plugins**: Affect the entire simulation world
- **Sensor plugins**: Process data from Gazebo sensors
- **System plugins**: Provide global simulation functionality

Example of a simple world plugin:

```xml
<sdf version="1.7">
  <world name="default">
    <!-- World plugin for custom simulation behavior -->
    <plugin name="custom_world_plugin" filename="libCustomWorldPlugin.so">
      <frequency>10</frequency>
    </plugin>
  </world>
</sdf>
```

## Setting Up Gazebo for Humanoid Robotics

### Installation

Gazebo is typically installed with ROS 2:

```bash
# On Ubuntu 22.04 (for ROS 2 Humble)
sudo apt update
sudo apt install ros-humble-gazebo-*

# Install Gazebo Garden (standalone)
sudo apt install gazebo
```

### Basic Launch

Start Gazebo with:

```bash
# Launch Gazebo server only
gzserver

# Launch Gazebo client only (connects to running server)
gzclient

# Launch both server and client
gazebo
```

### ROS 2 Integration

To use Gazebo with ROS 2, you'll typically use Gazebo ROS packages:

```xml
<!-- In your robot's URDF -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_humanoid</robotNamespace>
  </plugin>
</gazebo>
```

### Launch Files for Humanoid Simulation

A typical launch file for a humanoid in Gazebo:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Arguments
    model_arg = DeclareLaunchArgument(
        name='model',
        default_value='my_humanoid',
        description='Robot model name'
    )
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ])
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', PathJoinSubstitution([
                FindPackageShare('my_humanoid_description'),
                'urdf/my_humanoid.urdf.xacro'
            ])])
        }]
    )
    
    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', LaunchConfiguration('model')
        ],
        output='screen'
    )
    
    return LaunchDescription([
        model_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## Basic Simulation Concepts

### World Definition (SDF)

Gazebo worlds are defined using SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.2 -0.4 -0.9</direction>
    </light>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sample objects -->
    <include>
      <uri>model://table</uri>
      <pose>1 0 0 0 0 0</pose>
    </include>
    
    <!-- Your humanoid robot (defined elsewhere) -->
    <include>
      <uri>model://my_humanoid</uri>
      <pose>0 0 0.8 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Time in Simulation

Gazebo operates in simulation time, which can progress differently from real time:

- **Real Time Factor (RTF)**: Ratio of sim time to real time
  - RTF = 1.0: Simulation runs at real-time speed
  - RTF > 1.0: Simulation runs faster than real-time
  - RTF < 1.0: Simulation runs slower than real-time

- **Step Size**: How much time advances per physics calculation
  - Smaller steps = more accurate but slower simulation
  - For humanoid robots, typical step sizes are 0.001s to 0.0001s

### Coordinate Systems

In Gazebo, the coordinate system follows the right-hand rule:
- **X**: Forward (relative to the model)
- **Y**: Left (relative to the model)
- **Z**: Up (opposite to gravity direction)

This is typically the same as ROS, ensuring consistency when moving between simulation and reality.

## Creating Your First Humanoid Simulation

### Basic Humanoid Model in SDF

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_humanoid">
    <!-- Base link (pelvis for humanoid) -->
    <link name="base_link">
      <pose>0 0 0.8 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.1</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
          <iyy>0.1</iyy> <iyz>0.0</iyz> <izz>0.1</izz>
        </inertia>
      </inertial>
      
      <visual name="visual">
        <geometry>
          <box>
            <size>0.2 0.3 0.1</size>
          </box>
        </geometry>
      </visual>
      
      <collision name="collision">
        <geometry>
          <box>
            <size>0.2 0.3 0.1</size>
          </box>
        </geometry>
      </collision>
    </link>
    
    <!-- Left leg -->
    <link name="left_thigh">
      <pose>0 -0.1 -0.3 0 0 0</pose>
      <inertial>
        <mass>3.0</mass>
        <inertia>
          <ixx>0.05</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
          <iyy>0.05</iyy> <iyz>0.0</iyz> <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>
    
    <joint name="left_hip_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_thigh</child>
      <pose>-0.1 -0.15 -0.1 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.0</lower>
          <upper>1.0</upper>
          <effort>100</effort>
          <velocity>2.0</velocity>
        </limit>
      </axis>
    </joint>
    
    <!-- Controllers would go here using plugins -->
  </model>
</sdf>
```

### Loading the Simulation

To run this simulation, save it as "simple_humanoid.sdf" and run:

```bash
gazebo simple_humanoid.sdf
```

### Robot Control in Simulation

For humanoid robots, you'll want to add control plugins:

```xml
<!-- In your model definition -->
<plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
  <ros>
    <namespace>/simple_humanoid</namespace>
    <remapping>~/out:=joint_states</remapping>
  </ros>
  <update_rate>30</update_rate>
  <joint_name>left_hip_joint</joint_name>
</plugin>

<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <ros>
    <namespace>/simple_humanoid</namespace>
  </ros>
  <left_joint>left_wheel_joint</left_joint>
  <right_joint>right_wheel_joint</right_joint>
  <wheel_separation>0.4</wheel_separation>
  <wheel_diameter>0.2</wheel_diameter>
  <command_topic>cmd_vel</command_topic>
  <odometry_topic>odom</odometry_topic>
  <odometry_frame>odom</odometry_frame>
  <robot_base_frame>base_link</robot_base_frame>
</plugin>
```

## SDF vs. URDF in Gazebo

### When to Use Each

- **URDF**: For robot description in ROS ecosystem
  - Defines kinematics, dynamics, and visual/collision properties
  - Used with robot_state_publisher, moveit, etc.
  - Better for describing the robot's physical structure

- **SDF**: For Gazebo simulation environments
  - Defines entire simulation worlds with physics, lighting, etc.
  - Better for describing simulation-specific aspects
  - Required for Gazebo plugins and simulation physics

### Converting URDF to SDF

You can convert a URDF model to SDF:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or include URDF as a model in SDF world
<sdf version="1.7">
  <world name="my_world">
    <include>
      <uri>model://my_robot.urdf</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Combining Both Approaches

In practice, you'll often use both:
1. Define your robot in URDF (with Gazebo plugins)
2. Define your world in SDF
3. Include the URDF robot in the SDF world

```xml
<!-- In your URDF with Gazebo-specific elements -->
<robot name="my_humanoid">
  <!-- Your URDF content -->
  <link name="base_link">
    <!-- ... -->
  </link>
  
  <!-- Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>
  
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/my_humanoid</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

## Physics Engine Fundamentals

### Physics Engine Selection

Gazebo supports multiple physics engines:

- **ODE (Open Dynamics Engine)**: Default choice, good for most applications
- **Bullet**: Better for complex contact scenarios
- **DART**: More robust for articulated systems like humanoid robots
- **Simbody**: Good for biomechanics, but less commonly used

For humanoid robots, DART is often preferred due to its robust handling of articulated systems and contacts.

### Physics Parameters for Humanoids

The physics configuration significantly impacts humanoid simulation:

```xml
<physics type="dart">  <!-- DART for humanoid robots -->
  <max_step_size>0.001</max_step_size>  <!-- Small steps for stability -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time if possible -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- High update rate -->
  
  <!-- Solver parameters -->
  <solver>
    <type>PGS</type>  <!-- Projected Gauss-Seidel -->
    <iters>1000</iters>  <!-- More iterations for stability -->
    <sor>1.3</sor>  <!-- Successive Over Relaxation -->
  </solver>
  
  <!-- Constraints -->
  <constraints>
    <cfm>0</cfm>  <!-- Constraint Force Mixing -->
    <erp>0.2</erp>  <!-- Error Reduction Parameter -->
    <contact_max_correcting_vel>100</contact_max_correcting_vel>
    <contact_surface_layer>0.001</contact_surface_layer>
  </constraints>
</physics>
```

### Balancing Simulation Realism and Performance

For humanoid robots, you need to balance:
- **Stability**: Prevent joint oscillations and artifacts
- **Accuracy**: Properly simulate forces and contacts
- **Performance**: Maintain real-time or near real-time simulation

Key parameters to tune:
- `max_step_size`: Smaller for more accuracy but slower simulation
- `real_time_update_rate`: Higher for real-time performance
- `solver/iters`: More iterations for stability but slower computation
- `constraints/erp`: Higher for faster error correction but potential instability

### Contact Modeling for Humanoid Robots

Humanoid robots have many potential contact points (feet, hands, etc.), making contact modeling critical:

```xml
<!-- For feet in a humanoid robot -->
<gazebo reference="left_foot">
  <collision name="foot_collision">
    <surface>
      <contact>
        <ode>
          <kp>10000000</kp>  <!-- Contact stiffness -->
          <kd>10000</kd>     <!-- Contact damping -->
          <max_vel>100</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.8</mu>    <!-- Static friction coefficient -->
          <mu2>0.8</mu2>  <!-- Secondary friction coefficient -->
          <fdir1>0 0 1</fdir1>
        </ode>
      </friction>
    </surface>
  </collision>
</gazebo>
```

## Summary

In this chapter, we've established the foundation for Gazebo simulation in humanoid robotics. We explored Gazebo's architecture, its importance for humanoid development, and the key concepts for setting up and running humanoid simulations.

Simulation is an essential tool for humanoid robotics development because it allows for safe, repeatable testing of complex behaviors. The physics engine, contact modeling, and real-time performance considerations are all critical for creating accurate humanoid simulations.

Understanding the relationship between URDF (for robot description) and SDF (for simulation environment) is crucial for effective simulation setup.

## Exercises

1. Install Gazebo on your system and launch the empty world. Explore the interface and try adding some basic objects using the Insert panel.

2. Create a simple URDF model of a humanoid with a torso, head, two arms, and two legs. Include the necessary Gazebo plugins and launch it in simulation.

3. Modify the physics parameters of a Gazebo world file and observe how they affect simulation stability and performance. Try different physics engines (ODE vs. DART) and document the differences for a simple humanoid model.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
