---
id: module-02-chapter-02
title: Chapter 02 - Physics, Gravity, Collisions
sidebar_position: 6
---

# Chapter 02 - Physics, Gravity, Collisions

## Table of Contents
- [Overview](#overview)
- [Physics Simulation Fundamentals](#physics-simulation-fundamentals)
- [Gravity Modeling in Humanoid Robotics](#gravity-modeling-in-humanoid-robotics)
- [Collision Detection and Response](#collision-detection-and-response)
- [Inertial Properties and Dynamics](#inertial-properties-and-dynamics)
- [Contact Modeling for Humanoid Robots](#contact-modeling-for-humanoid-robots)
- [Stability Considerations](#stability-considerations)
- [Tuning Physics Parameters](#tuning-physics-parameters)
- [Common Physics Issues and Solutions](#common-physics-issues-and-solutions)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Physics simulation is fundamental to humanoid robotics, as these robots must interact with the physical world through forces, gravity, and collisions. Unlike other robot types, humanoid robots have complex balance requirements and multiple potential contact points with the environment. In this chapter, we'll explore the physics principles and simulation techniques specifically relevant to humanoid robots, focusing on how to achieve realistic and stable simulation of human-like movement and interactions.

The physics simulation of humanoid robots is particularly challenging because of their complex kinematics, multiple degrees of freedom, and the need to maintain balance while performing various tasks. Getting the physics right is critical for ensuring that control algorithms developed in simulation will transfer effectively to the physical robot.

## Physics Simulation Fundamentals

### Core Physics Concepts

Physics simulation in Gazebo is based on several fundamental concepts:

1. **Newtonian Mechanics**: Simulation of forces, torques, velocities, and accelerations
2. **Rigid Body Dynamics**: Modeling links as rigid bodies connected by joints
3. **Constraint Solving**: Handling joint constraints and collision contacts
4. **Time Integration**: Advancing the simulation through time with numerical methods

### Physics Simulation Loop

The physics simulation proceeds in discrete time steps:

1. **Force Application**: Apply external forces (gravity, actuators)
2. **Collision Detection**: Identify contacts between objects
3. **Constraint Solving**: Compute contact forces and joint reactions
4. **Integration**: Update positions and velocities
5. **Visualization**: Render the scene

For humanoid robots, this loop typically runs at 1000+ Hz to maintain stability for underactuated systems like bipedal walkers.

### Physics Engines in Gazebo

Gazebo supports several physics engines, each with tradeoffs:

- **ODE (Open Dynamics Engine)**: Default, good general-purpose engine
- **Bullet**: Good for complex contact scenarios
- **Simbody**: Biologically-inspired, good for articulated systems
- **DART (Dynamic Animation and Robotics Toolkit)**: Robust for complex articulated systems

For humanoid robots, DART is often preferred due to its superior handling of articulated systems and contact stability.

## Gravity Modeling in Humanoid Robotics

### Gravity in Simulation

Gravity is a fundamental force in humanoid simulation, representing Earth's gravitational field:

```xml
<!-- In a world file -->
<world name="humanoid_world">
  <gravity>0 0 -9.8</gravity>  <!-- 9.8 m/s^2 downward -->
  
  <!-- Physics parameters -->
  <physics type="dart">
    <gravity>0 0 -9.8</gravity>
  </physics>
</world>
```

### Gravity Compensation

Humanoid robots need to actively counteract gravity to maintain posture:

```xml
<!-- Example of a gravity-compensated joint -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="1.0" effort="100" velocity="2.0"/>
  <dynamics damping="1.0" friction="0.5"/>
  <!-- The controller must compensate for gravity on the arm -->
</joint>
```

### Center of Mass Considerations

For humanoid robots, the center of mass (CoM) is critical for balance:

- **Standing**: CoM should be within the support polygon (area between feet)
- **Walking**: CoM follows a complex trajectory to maintain dynamic balance
- **Stability**: Lower CoM generally means more stability

```python
# Example: Calculating CoM position
import numpy as np

def calculate_com(robot_masses, robot_positions):
    """
    Calculate center of mass for a humanoid robot
    :param robot_masses: List of link masses
    :param robot_positions: List of link positions [x, y, z]
    :return: Center of mass position [x, y, z]
    """
    total_mass = sum(robot_masses)
    com_x = sum(mass * pos[0] for mass, pos in zip(robot_masses, robot_positions)) / total_mass
    com_y = sum(mass * pos[1] for mass, pos in zip(robot_masses, robot_positions)) / total_mass
    com_z = sum(mass * pos[2] for mass, pos in zip(robot_masses, robot_positions)) / total_mass
    
    return [com_x, com_y, com_z]
```

### Gravity in World Frames

The gravity vector can be adjusted for different environments:

```xml
<!-- For robot on an incline -->
<world name="incline_world">
  <gravity>1.7 -1.0 -9.6</gravity>  <!-- Adjusted for 10-degree incline -->
</world>
```

## Collision Detection and Response

### Collision Geometry

For humanoid robots, collision detection is critical for:
- Self-collision prevention
- Environment collision avoidance
- Foot-ground contact for balance

Collision geometry should be:
- **Accurate enough** to catch important collisions
- **Simple enough** to maintain simulation performance
- **Conservative enough** to prevent missed collisions

Types of collision geometry:
- **Primitives**: Box, sphere, cylinder (fastest)
- **Mesh**: Complex shapes from 3D models (most accurate)
- **Compound**: Combination of primitives (balance of speed/accuracy)

### Self-Collision Avoidance

Humanoid robots have many potential self-collision pairs. In URDF:

```xml
<link name="left_upper_arm">
  <collision>
    <geometry>
      <cylinder length="0.25" radius="0.05"/>
    </geometry>
  </collision>
</link>

<link name="left_lower_arm">
  <collision>
    <geometry>
      <cylinder length="0.25" radius="0.04"/>
    </geometry>
  </collision>
</link>

<!-- Disable collision between adjacent links (they're meant to touch) -->
<disable_collisions link1="torso" link2="left_upper_arm"/>
<disable_collisions link1="left_upper_arm" link2="left_lower_arm"/>
```

### Ground Contact Modeling

Ground contact is critical for bipedal locomotion:

```xml
<gazebo reference="left_foot">
  <collision name="foot_collision">
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>    <!-- Static friction coefficient -->
          <mu2>0.8</mu2>  <!-- Secondary friction coefficient -->
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <soft_cfm>0.001</soft_cfm>
          <soft_erp>0.8</soft_erp>
          <kp>1e6</kp>  <!-- Contact stiffness -->
          <kd>1e4</kd>  <!-- Contact damping -->
        </ode>
      </contact>
    </surface>
  </collision>
</gazebo>
```

### Collision Detection Algorithms

Gazebo uses hierarchical collision detection:
1. **Broad phase**: Quick rejection using bounding boxes
2. **Narrow phase**: Precise collision checking between potential pairs
3. **Contact generation**: Computing contact points and normals

For humanoid robots with many links, consider:
- **Reduced precision models** for collision detection
- **Selective collision checking** only for likely interactions
- **Spatial partitioning** for large environments

## Inertial Properties and Dynamics

### Inertial Parameters

Accurate inertial properties are essential for realistic humanoid simulation:

```xml
<link name="left_thigh">
  <inertial>
    <mass value="3.0"/>  <!-- Mass in kg -->
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>  <!-- Center of mass offset -->
    <inertia 
      ixx="0.05" ixy="0.0" ixz="0.0"
      iyy="0.05" iyz="0.0" 
      izz="0.01"/>  <!-- Moments of inertia -->
  </inertial>
</link>
```

### Computing Inertial Properties

For complex humanoid links:

1. **CAD software**: Most CAD tools can export inertial properties
2. **Mathematical formulas**: For simple shapes
   - Cylinder (about center): Ixx = Iyy = m(h²/12 + r²/4), Izz = mr²/2
   - Box: Ixx = m(h² + d²)/12, etc.
3. **Approximation**: Treat as combinations of simple shapes

### Inertia and Balance

Inertial properties significantly affect humanoid balance:

- **Larger moments of inertia**: More resistance to angular acceleration
- **Center of mass position**: Affects balance and required control effort
- **Inertial coupling**: Motion in one joint affects others (especially important for humanoids)

### Dynamic Simulation

For realistic humanoid motion, consider:

```xml
<!-- Joint dynamics properties -->
<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_lower_leg"/>
  <axis xyz="0 1 0"/>
  <limit lower="0.0" upper="2.5" effort="200" velocity="3.0"/>
  <!-- Dynamic properties affect stability -->
  <dynamics damping="2.0" friction="0.5"/>
</joint>
```

### Actuator Dynamics

Model the real dynamics of actuators:

```xml
<!-- Transmission for realistic actuator modeling -->
<transmission name="left_knee_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_knee">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>100</mechanicalReduction>
  </joint>
  <actuator name="left_knee_motor">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>100</mechanicalReduction>
  </actuator>
</transmission>
```

## Contact Modeling for Humanoid Robots

### Contact Stability Challenges

Humanoid robots face unique contact challenges:

1. **Multiple contact points**: Both feet during standing, variable during walking
2. **Variable contact areas**: Flat feet, toes, heels
3. **Dynamic contacts**: Contacts that appear and disappear during motion
4. **High forces**: Humanoid robots can generate significant ground reaction forces

### Contact Parameters

Key contact parameters for humanoid simulation:

```xml
<!-- In a link's collision element -->
<surface>
  <contact>
    <ode>
      <soft_cfm>0.001</soft_cfm>    <!-- Constraint Force Mixing -->
      <soft_erp>0.8</soft_erp>      <!-- Error Reduction Parameter -->
      <kp>10000000</kp>             <!-- Stiffness -->
      <kd>10000</kd>                <!-- Damping -->
      <max_vel>100</max_vel>         <!-- Maximum contact correction rate -->
      <min_depth>0.001</min_depth>   <!-- Penetration depth threshold -->
    </ode>
  </contact>
  <friction>
    <ode>
      <mu>0.8</mu>      <!-- Primary friction coefficient -->
      <mu2>0.8</mu2>    <!-- Secondary friction coefficient -->
      <fdir1>1 0 0</fdir1>  <!-- Friction direction 1 -->
    </ode>
  </friction>
</surface>
```

### Foot Contact Modeling

For humanoid walking, foot contact modeling is crucial:

```xml
<!-- Detailed foot contact model -->
<gazebo reference="left_foot">
  <collision name="foot_contact">
    <geometry>
      <!-- Use multiple contact points if needed -->
      <box size="0.15 0.08 0.02"/>  <!-- Simplified contact surface -->
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.0001</soft_cfm>
          <soft_erp>0.9</soft_erp>  <!-- Higher ERP for better contact stability -->
          <kp>1e7</kp>              <!-- High stiffness for foot contact -->
          <kd>1e5</kd>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>1.0</mu>    <!-- High friction for stable walking -->
          <mu2>1.0</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
</gazebo>
```

### Contact Stability Techniques

To improve contact stability:

1. **Increase solver iterations** for more accurate contact resolution
2. **Adjust ERP and CFM** parameters for the right balance of stability and accuracy
3. **Use realistic stiffness values** - too high causes numerical issues, too low causes excessive penetration
4. **Consider using a different physics engine** like DART for articulated systems

## Stability Considerations

### Numerical Stability

Humanoid simulations require careful attention to numerical stability:

```xml
<!-- Physics engine configuration for stability -->
<physics type="dart">
  <max_step_size>0.0005</max_step_size>  <!-- Small steps for stability -->
  <real_time_factor>0.5</real_time_factor>  <!-- Allow slower than real-time -->
  <real_time_update_rate>2000</real_time_update_rate>
  <solver>
    <type>quick</type>  <!-- or "pgs" for Projected Gauss-Seidel -->
    <iters>2000</iters>  <!-- More iterations for accuracy -->
    <sor>1.2</sor>      <!-- Successive Over-Relaxation parameter -->
  </solver>
</physics>
```

### Z-Fighting and Penetration

Common issues in humanoid simulation:

- **Z-fighting**: Objects oscillating as they pass through each other
- **Penetration**: Objects sinking into each other
- **Jittering**: High-frequency oscillations at contacts

### Stability Tuning Process

1. **Start with conservative parameters** (small step size, many iterations)
2. **Test basic poses** (standing) for stability
3. **Gradually adjust parameters** for performance while maintaining stability
4. **Test dynamic behaviors** (walking, manipulation) to ensure they remain stable

### Balance Control and Physics Interaction

Physics parameters affect stability control:

```python
# Example: Balance controller affected by physics parameters
class BalanceController:
    def __init__(self, robot_mass, com_height, gravity=9.81):
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = gravity
        self.control_gain = self.calculate_control_gain()
        
    def calculate_control_gain(self):
        # Control gain depends on physical parameters
        # and simulation time step
        omega = (self.gravity / self.com_height) ** 0.5
        return (self.robot_mass * self.gravity) / omega**2
    
    def compute_control(self, measured_com_pos, desired_com_pos):
        # Control is affected by simulation physics
        return self.control_gain * (desired_com_pos - measured_com_pos)
```

## Tuning Physics Parameters

### Parameter Tuning Process

1. **Start with default values** from stable simulations
2. **Identify the issue** you're trying to solve
3. **Change one parameter at a time** to understand its effect
4. **Test with realistic scenarios** (not just static poses)
5. **Document the effects** of changes

### Common Parameter Ranges

For humanoid robots:

```xml
<physics type="dart">
  <!-- Time stepping -->
  <max_step_size>0.0005</max_step_size>    <!-- 0.5ms - 1ms typical -->
  
  <!-- Solver -->
  <solver>
    <type>pgs</type>      <!-- PGS or Dantzig -->
    <iters>1000</iters>   <!-- 500-2000 for humanoid robots -->
    <sor>1.3</sor>        <!-- 1.0-2.0 -->
  </solver>
  
  <!-- Constraints -->
  <constraints>
    <contact_surface_layer>0.001</contact_surface_layer>  <!-- 0.1-2mm -->
    <contact_max_correcting_vel>100</contact_max_correcting_vel>  <!-- 10-100 m/s -->
  </constraints>
</physics>
```

### Performance vs. Accuracy Tradeoffs

| Parameter | Stability Impact | Performance Impact | Recommended Value |
|-----------|------------------|-------------------|-------------------|
| Step Size | Higher = More stable | Higher = Slower | 0.0005-0.001 s |
| Solver Iterations | Higher = More stable | Higher = Slower | 1000-2000 |
| ERP | Higher = More responsive | Higher = Less stable | 0.8-0.95 |
| CFM | Higher = Less stable | Higher = More stable | 0.0001-0.01 |

### Validation of Physics Settings

Test physics settings with:
1. **Static poses**: Robot standing still
2. **Simple movements**: Joint oscillations
3. **Balance tasks**: Maintaining upright position
4. **Dynamic tasks**: Walking or manipulation

## Common Physics Issues and Solutions

### Joint Drift

**Problem**: Joints slowly move away from commanded positions over time.

**Solution**: 
- Increase joint damping
- Add joint friction
- Verify control loop frequency

```xml
<joint name="problematic_joint" type="revolute">
  <dynamics damping="10.0" friction="5.0"/>  <!-- Increase damping/friction -->
</joint>
```

### Unstable Contacts

**Problem**: Robot bounces, jitters, or falls through the ground.

**Solutions**:
- Increase solver iterations
- Adjust ERP/CFM parameters
- Use a different physics engine
- Increase contact stiffness

### Excessive Penetration

**Problem**: Robot parts sink into each other or the ground.

**Solutions**:
- Increase contact stiffness (kp)
- Reduce contact damping (kd) relative to stiffness
- Ensure collision geometry is properly sized

### Simulation Instability

**Problem**: Robot suddenly explodes or exhibits chaotic behavior.

**Solutions**:
- Reduce max step size
- Add joint limits and safety controllers
- Verify inertial properties are physically realistic
- Add damping to joints

### Control-Physics Loop Interaction

**Problem**: Control algorithm works in theory but fails in simulation.

**Solutions**:
- Match simulation time step to control loop time step
- Add sensor noise models to simulation
- Include actuator dynamics in simulation

## Summary

In this chapter, we've explored the physics simulation aspects that are particularly important for humanoid robots. We've covered gravity modeling, collision detection and response, inertial properties, contact modeling, and stability considerations. 

Physics simulation is crucial for humanoid robotics because these robots must interact with the physical world through forces, gravity, and contacts while maintaining balance. Proper physics parameters are essential for achieving stable, realistic simulation that allows for effective development and testing of humanoid control algorithms.

The key challenges in humanoid physics simulation include managing multiple contact points (especially feet), maintaining stability during dynamic behaviors like walking, and ensuring that the simulation matches the physical robot's behavior.

## Exercises

1. Create a simple humanoid model with realistic inertial properties and test its stability in standing position. Adjust physics parameters to achieve a stable simulation.

2. Implement a simple balance controller that maintains the robot's center of mass within the support polygon defined by its feet. Test how physics parameters affect the controller's performance.

3. Model foot-ground contact with different friction coefficients and observe how they affect the robot's ability to stand and walk in simulation.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
