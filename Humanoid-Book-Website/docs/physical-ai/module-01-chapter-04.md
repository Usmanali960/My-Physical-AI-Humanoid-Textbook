---
id: module-01-chapter-04
title: Chapter 04 - URDF for Humanoids
sidebar_position: 4
---

# Chapter 04 - URDF for Humanoids

## Table of Contents
- [Overview](#overview)
- [What is URDF?](#what-is-urdf)
- [URDF for Humanoid Robots](#urdf-for-humanoid-robots)
- [URDF Structure and Components](#urdf-structure-and-components)
- [Links, Joints, and Transmissions](#links-joints-and-transmissions)
- [Visual and Collision Elements](#visual-and-collision-elements)
- [Inertial Properties](#inertial-properties)
- [Gazebo-Specific Elements](#gazebo-specific-elements)
- [Creating a Humanoid URDF](#creating-a-humanoid-urdf)
- [URDF Tools and Validation](#urdf-tools-and-validation)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

URDF (Unified Robot Description Format) is the standard XML format for representing robot models in ROS. For humanoid robots, which are complex articulated systems with many degrees of freedom, a well-designed URDF is crucial for simulation, visualization, motion planning, and control. In this chapter, we'll explore how to create and structure URDF files specifically for humanoid robots, covering best practices and advanced techniques.

A properly structured URDF provides the foundation for all downstream robotics applications, from simulation to control, making it one of the most important components in humanoid robot development.

## What is URDF?

### Definition and Purpose

URDF is an XML-based format that describes robot models, their kinematic and dynamic properties, and their visual representation. It specifies:
- Link/joint relationships
- Physical properties (mass, inertia)
- Visual and collision geometry
- Sensors and actuators
- Robot kinematics

### The Robot Model in URDF

A robot is modeled as a tree structure of links connected by joints. For a humanoid robot, this structure typically resembles:

```
base_link (pelvis) 
├── torso
│   ├── head
│   ├── left_arm
│   │   ├── left_forearm
│   │   └── left_hand
│   └── right_arm
│       ├── right_forearm
│       └── right_hand
└── left_leg
    ├── left_lower_leg
    └── left_foot
└── right_leg
    ├── right_lower_leg
    └── right_foot
```

### URDF vs. SDF

- **URDF**: Primarily for ROS ecosystem, represents robot models
- **SDF**: For Gazebo simulation, can represent entire worlds with robots, objects, and environments
- **Integration**: URDF can be converted to SDF for simulation, or SDF can include URDF models

## URDF for Humanoid Robots

### Unique Challenges

Humanoid robots present specific challenges in URDF definition:
- **Many degrees of freedom**: Typically 20-40+ joints
- **Complex kinematics**: Bipedal locomotion, manipulation, balance
- **Symmetry**: Arms and legs often have symmetric structures
- **Multi-body systems**: Head, torso, limbs connected in complex ways
- **Sensor integration**: IMUs, cameras, force-torque sensors

### Design Considerations

When creating URDF for humanoid robots, consider:
1. **Kinematic chains**: How limbs connect and affect each other
2. **Center of Mass**: Critical for balance and simulation
3. **Collision detection**: Properly defined for safe simulation
4. **Visualization**: Accurate representation for debugging
5. **Simulation performance**: Balance detail with computational cost

## URDF Structure and Components

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links define the rigid bodies -->
  <link name="base_link">
    <inertial>...</inertial>
    <visual>...</visual>
    <collision>...</collision>
  </link>
  
  <!-- Joints define the connections between links -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="link_name"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
  
  <!-- Other elements -->
  <transmission>...</transmission>
  <gazebo>...</gazebo>
</robot>
```

### Key Components of a Humanoid URDF

1. **Base Link**: Usually the pelvis/torso for bipedal robots
2. **Kinematic Chains**: Left/right arms, left/right legs
3. **End Effectors**: Hands and feet
4. **Sensors**: IMUs, cameras, force-torque sensors

## Links, Joints, and Transmissions

### Links

Links represent rigid bodies in the robot. For humanoid robots, important links include:

```xml
<link name="base_link">
  <inertial>
    <mass value="10.0"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <inertia ixx="0.4" ixy="0" ixz="0" iyy="0.4" iyz="0" izz="0.2"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_humanoid_description/meshes/pelvis.dae"/>
    </geometry>
    <material name="grey">
      <color rgba="0.5 0.5 0.5 1.0"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_humanoid_description/meshes/pelvis_collision.stl"/>
    </geometry>
  </collision>
</link>
```

### Joint Types for Humanoids

1. **Revolute**: Single-axis rotation, used for most humanoid joints
2. **Continuous**: Like revolute but unlimited rotation (wheels)
3. **Prismatic**: Linear motion
4. **Fixed**: No motion, used for attaching sensors
5. **Floating**: 6 DOF (rarely used in humanoid URDFs)

For humanoid robots, most joints are revolute. Example of a hip joint:

```xml
<joint name="left_hip_yaw_joint" type="revolute">
  <parent link="base_link"/>
  <child link="left_thigh"/>
  <origin xyz="0 0.1 -0.05" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="2.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Joint Limits and Dynamics

For humanoid robots, proper limits and dynamics are crucial for realistic simulation:

```xml
<joint name="left_knee_joint" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_lower_leg"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0.0" upper="2.5" effort="200" velocity="3.0"/>
  <dynamics damping="1.0" friction="0.5"/>
  <safety_controller k_position="10" k_velocity="10" soft_lower_limit="0.05" soft_upper_limit="2.45"/>
</joint>
```

### Transmissions

Transmissions define how actuators connect to joints. For humanoid robots:

```xml
<transmission name="left_knee_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_knee_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_knee_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Visual and Collision Elements

### Visual Elements

Visual elements determine how the robot appears in RViz, Gazebo, and other visualization tools:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Options: box, cylinder, sphere, or mesh -->
    <mesh filename="package://my_humanoid_description/meshes/upper_arm.dae" scale="1 1 1"/>
  </geometry>
  <material name="light_grey">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>
</visual>
```

For humanoid robots, using mesh files is common since the complex shapes are difficult to represent with basic primitives.

### Collision Elements

Collision elements are used for collision detection in simulation and motion planning:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Common choices for humanoid collision geometry -->
    <mesh filename="package://my_humanoid_description/meshes/upper_arm_collision.stl"/>
    <!-- Or simplified geometry for performance -->
    <cylinder length="0.2" radius="0.08"/>
  </geometry>
</collision>
```

### Collision Optimization

For humanoid robots with many links, collision performance is important:

1. **Simplified geometry**: Use simpler shapes (cylinders, capsules, boxes) instead of complex meshes
2. **Convex decomposition**: Break complex meshes into convex parts
3. **Collision filtering**: Use Gazebo's collision groups to avoid unnecessary checks

## Inertial Properties

### Mass and Inertia

Accurate inertial properties are crucial for humanoid simulation and control:

```xml
<inertial>
  <mass value="2.5"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
</inertial>
```

### Computing Inertia

For complex humanoid links, computing inertia can be done:

1. **CAD software**: Most CAD tools can compute mass properties
2. **Formulas for simple shapes**:
   - Solid cylinder (about central axis): Izz = ½mr²
   - Solid cylinder (about end face): Ixx = Iyy = m(¼r² + ⅓h²), Izz = ½mr²
   - Solid sphere: Ixx = Iyy = Izz = ⅖mr²

3. **Online calculators** or dedicated tools

### Inertial Guidelines for Humanoids

```xml
<!-- Example for a humanoid head link -->
<link name="head">
  <inertial>
    <mass value="3.0"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <!-- Approximate as sphere for head -->
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
  ...
</link>
```

## Gazebo-Specific Elements

### Gazebo Plugins

For simulation, Gazebo-specific elements are needed:

```xml
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_humanoid</robotNamespace>
  </plugin>
</gazebo>
```

### Link-Specific Gazebo Elements

```xml
<gazebo reference="head">
  <material>Gazebo/Blue</material>
  <sensor type="camera" name="head_camera">
    <update_rate>30.0</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <format>R8G8B8</format>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>my_humanoid/head_camera</cameraName>
      <imageTopicName>image_raw</imageTopicName>
      <cameraInfoTopicName>camera_info</cameraInfoTopicName>
      <frameName>head_camera_frame</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### Joint-Specific Gazebo Elements

```xml
<gazebo reference="left_knee_joint">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

## Creating a Humanoid URDF

### Complete Example: Simplified Humanoid

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="mass_torso" value="10.0" />
  <xacro:property name="mass_limb" value="2.0" />
  <xacro:property name="mass_head" value="3.0" />
  
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="${mass_torso}"/>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.4"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="10" velocity="1"/>
  </joint>
  
  <link name="head">
    <inertial>
      <mass value="${mass_head}"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0.15 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="2"/>
  </joint>
  
  <link name="left_upper_arm">
    <inertial>
      <mass value="${mass_limb}"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.05"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="30" velocity="2"/>
  </joint>
  
  <link name="left_lower_arm">
    <inertial>
      <mass value="${mass_limb}"/>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.004"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.15" radius="0.04"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.15" radius="0.04"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0 -0.075 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/2}" effort="100" velocity="1.5"/>
  </joint>
  
  <link name="left_thigh">
    <inertial>
      <mass value="${mass_limb}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.008"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.06"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.06"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="100" velocity="1.5"/>
  </joint>
  
  <link name="left_lower_leg">
    <inertial>
      <mass value="${mass_limb}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.006"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Right side (similar to left, with appropriate mirroring) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0 0.075 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/2}" effort="100" velocity="1.5"/>
  </joint>
  
  <link name="right_thigh">
    <inertial>
      <mass value="${mass_limb}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.008"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.06"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.06"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="100" velocity="1.5"/>
  </joint>
  
  <link name="right_lower_leg">
    <inertial>
      <mass value="${mass_limb}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.006"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Gazebo elements -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/simple_humanoid</robotNamespace>
    </plugin>
  </gazebo>
  
  <!-- IMU sensor -->
  <gazebo reference="base_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>
  
</robot>
```

### Using Xacro for Complex Humanoid URDFs

For complex humanoid robots, Xacro (XML Macros) is essential to avoid repetition:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_name" value="my_humanoid" />
  
  <!-- Macro for leg definition -->
  <xacro:macro name="leg" params="side prefix reflect joint_limits">
    <joint name="${prefix}_${side}_hip_yaw_joint" type="revolute">
      <parent link="base_link"/>
      <child link="${prefix}_${side}_thigh"/>
      <origin xyz="0 ${reflect*0.1} -0.05" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${joint_limits[0]}" upper="${joint_limits[1]}" effort="100" velocity="2.0"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>
    
    <link name="${prefix}_${side}_thigh">
      <inertial>
        <mass value="3.0"/>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.008"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.3" radius="0.06"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.3" radius="0.06"/>
        </geometry>
      </collision>
    </link>
    
    <!-- Additional joints and links for the leg -->
    <joint name="${prefix}_${side}_hip_roll_joint" type="revolute">
      <parent link="${prefix}_${side}_thigh"/>
      <child link="${prefix}_${side}_shank"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="1.0" effort="100" velocity="2.0"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>
    
    <link name="${prefix}_${side}_shank">
      <inertial>
        <mass value="2.5"/>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.006"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>
  
  <!-- Use the macro to create both legs -->
  <xacro:leg side="left" prefix="leg" reflect="1" joint_limits="[-0.5, 0.5]"/>
  <xacro:leg side="right" prefix="leg" reflect="-1" joint_limits="[-0.5, 0.5]"/>
  
</robot>
```

## URDF Tools and Validation

### Checking URDF Validity

```bash
# Check if URDF is well-formed XML
check_urdf /path/to/robot.urdf

# Visualize the URDF structure
urdf_to_graphiz /path/to/robot.urdf
```

### Visualization in RViz

```bash
# Launch a URDF with joint states publisher
roslaunch urdf_tutorial display.launch model:=/path/to/robot.urdf
```

### Python Scripts for URDF Processing

```python
import xml.etree.ElementTree as ET
from collections import defaultdict

def analyze_urdf(urdf_path):
    """Analyze URDF for humanoid-specific properties."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    joints = root.findall('.//joint')
    links = root.findall('.//link')
    
    # Count joint types
    joint_types = defaultdict(int)
    for joint in joints:
        joint_type = joint.get('type')
        joint_types[joint_type] += 1
    
    print(f"URDF Analysis:")
    print(f"  Links: {len(links)}")
    print(f"  Joints: {len(joints)}")
    print(f"  Joint types: {dict(joint_types)}")
    
    # Check for common humanoid joints
    humanoid_joints = [
        'hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist', 
        'neck', 'torso', 'waist'
    ]
    
    found_humanoid_joints = []
    for joint in joints:
        joint_name = joint.get('name').lower()
        if any(humanoid_joint in joint_name for humanoid_joint in humanoid_joints):
            found_humanoid_joints.append(joint_name)
    
    print(f"  Humanoid joints found: {len(found_humanoid_joints)}")
    for joint in found_humanoid_joints:
        print(f"    - {joint}")
    
    return dict(joint_types), found_humanoid_joints

# Example usage
# joint_types, humanoid_joints = analyze_urdf('/path/to/humanoid.urdf')
```

## Best Practices

### Modeling Best Practices

1. **Proper scaling**: Ensure all dimensions are in meters and masses in kg
2. **Consistent naming**: Use clear, consistent naming conventions (e.g., `left_shoulder_yaw_joint`)
3. **Realistic inertials**: Use CAD tools or formulas for accurate inertial properties
4. **Symmetry for limbs**: Use Xacro macros to define symmetrical parts
5. **Collision optimization**: Use simplified geometry for collision detection

### Performance Best Practices

1. **Mesh optimization**: Reduce polygon count for real-time applications
2. **LOD (Level of Detail)**: Use simpler collision meshes than visual meshes
3. **Joint limits**: Set appropriate limits to prevent self-collision
4. **Fixed joints**: Use fixed joints to combine multiple collision objects if needed

### Simulation Best Practices

1. **Damping values**: Add appropriate damping to prevent unrealistic oscillations
2. **Friction coefficients**: Set realistic friction values
3. **Realistic limits**: Set torque and velocity limits based on hardware specs
4. **Sensor placement**: Consider where sensors like IMUs should be placed

## Summary

In this chapter, we've covered URDF in depth, focusing specifically on its application to humanoid robots. We've explored the structure of URDF files, the various elements needed for humanoid modeling, and best practices for creating accurate and efficient robot models.

A well-crafted URDF is fundamental to humanoid robotics development, providing the basis for simulation, visualization, motion planning, and control. The complex structure of humanoid robots requires careful attention to kinematic chains, inertial properties, and both visual and collision representations.

## Exercises

1. Create a complete URDF for a simplified humanoid robot with at least 12 degrees of freedom (6 per leg, 2 per arm, and 2 for the torso/neck). Include proper inertial properties and visual elements.

2. Using Xacro, create a macro that defines a humanoid arm structure that can be instantiated for both left and right arms with appropriate mirroring.

3. Write a Python script that analyzes an existing humanoid URDF and reports:
   - Total number of links and joints
   - Joint types distribution
   - Center of mass calculation for the complete robot
   - Potential kinematic chain issues (e.g., disconnected links)

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
