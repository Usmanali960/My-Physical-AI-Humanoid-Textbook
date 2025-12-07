---
id: module-02-chapter-03
title: Chapter 03 - Sensor Simulation
sidebar_position: 7
---

# Chapter 03 - Sensor Simulation

## Table of Contents
- [Overview](#overview)
- [Sensor Types in Humanoid Robotics](#sensor-types-in-humanoid-robotics)
- [Camera Simulation](#camera-simulation)
- [Inertial Measurement Units (IMU)](#inertial-measurement-units-imu)
- [Force/Torque Sensors](#forcetorque-sensors)
- [LIDAR and Range Sensors](#lidar-and-range-sensors)
- [Sensor Fusion in Simulation](#sensor-fusion-in-simulation)
- [Sensor Noise and Realism](#sensor-noise-and-realism)
- [Sensor Placement for Humanoid Robots](#sensor-placement-for-humanoid-robots)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Sensors form the sensory system of humanoid robots, enabling them to perceive their environment, maintain balance, and interact with objects. In simulation, accurately modeling these sensors is crucial for developing robust perception and control algorithms that will transfer effectively to real robots. This chapter explores the simulation of various sensor types commonly found on humanoid robots, with a focus on Gazebo implementation and realistic behavior modeling.

The complexity of humanoid robots requires a diverse array of sensors working together. Unlike simpler robots, humanoid robots must maintain balance while moving, perceive the world from a human-like perspective, and interact with environments designed for humans. This requires careful attention to sensor placement, accuracy, and integration.

## Sensor Types in Humanoid Robotics

### Primary Sensor Categories

Humanoid robots typically use several sensor categories:

1. **Vision sensors**: Cameras for environment perception
2. **Inertial sensors**: IMUs for orientation and acceleration
3. **Proprioceptive sensors**: Joint encoders for position feedback
4. **Force/torque sensors**: For contact detection and manipulation
5. **Range sensors**: LIDAR, ultrasonic, or infrared for distance measurement
6. **Tactile sensors**: For object manipulation and contact (emerging technology)

### Sensor Integration Challenges

For humanoid robots specifically:
- **Multiple sensor fusion**: Combining data from many different sensors
- **Real-time processing**: Handling high data rates from multiple sensors
- **Synchronization**: Managing data from sensors with different rates
- **Calibration**: Ensuring sensors maintain accurate relationships

## Camera Simulation

### RGB Camera Simulation

Cameras are essential for humanoid robots to perceive their environment visually:

```xml
<!-- Camera sensor in URDF/Gazebo -->
<gazebo reference="head_camera">
  <sensor type="camera" name="head_camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees in radians -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera Simulation

Many humanoid robots use RGB-D cameras for 3D perception:

```xml
<!-- Depth camera sensor -->
<gazebo reference="depth_camera">
  <sensor type="depth" name="head_depth_camera">
    <update_rate>30</update_rate>
    <camera name="head_depth_camera">
      <horizontal_fov>1.0472</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera_name>head_depth_camera</camera_name>
      <image_topic_name>rgb/image_raw</image_topic_name>
      <depth_image_topic_name>depth/image_raw</depth_image_topic_name>
      <point_cloud_topic_name>depth/points</point_cloud_topic_name>
      <camera_info_topic_name>rgb/camera_info</camera_info_topic_name>
      <depth_image_camera_info_topic_name>depth/camera_info</depth_image_camera_info_topic_name>
      <frame_name>head_depth_camera_frame</frame_name>
      <point_cloud_cutoff>0.1</point_cloud_cutoff>
      <point_cloud_cutoff_max>5.0</point_cloud_cutoff_max>
    </plugin>
  </sensor>
</gazebo>
```

### Multi-Camera Systems

Humanoid robots often use multiple cameras:

```xml
<!-- Stereo cameras for depth perception -->
<gazebo reference="left_camera">
  <sensor type="camera" name="left_camera">
    <pose>0.05 0.06 0 0 0 0</pose> <!-- 6cm baseline -->
    <!-- Camera properties same as above -->
  </sensor>
</gazebo>

<gazebo reference="right_camera">
  <sensor type="camera" name="right_camera">
    <pose>0.05 -0.06 0 0 0 0</pose> <!-- 6cm baseline -->
    <!-- Camera properties same as above -->
  </sensor>
</gazebo>
```

### Camera Performance Considerations

When simulating cameras on humanoid robots:

- **Resolution**: Higher resolution = more realistic but slower simulation
- **Update rate**: Should match real camera capabilities (typically 15-30 Hz)
- **Field of view**: Wider FOV provides more scene coverage but less detail
- **Noise models**: Include realistic noise for robust algorithm development

## Inertial Measurement Units (IMU)

### IMU Fundamentals

IMUs are critical for humanoid balance, providing:
- **Orientation**: Quaternions representing robot's orientation
- **Angular velocity**: Rate of rotation around each axis
- **Linear acceleration**: Acceleration in the sensor frame

### IMU Simulation in Gazebo

```xml
<!-- IMU sensor simulation -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <topic>__default_topic__</topic>
    <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
      <topicName>imu/data</topicName>
      <bodyName>imu_link</bodyName>
      <updateRateHZ>100</updateRateHZ>
      <gaussianNoise>0.001</gaussianNoise>
      <xyzOffset>0 0 0</xyzOffset>
      <rpyOffset>0 0 0</rpyOffset>
      <frameName>imu_link</frameName>
    </plugin>
    <imu>
      <!-- Angular velocity noise -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <!-- Linear acceleration noise -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### IMU Placement on Humanoid Robots

For humanoid balance, IMUs are typically placed at strategic locations:

```xml
<!-- IMU in torso for overall orientation -->
<gazebo reference="torso_imu">
  <sensor name="torso_imu" type="imu">
    <!-- Configuration -->
  </sensor>
</gazebo>

<!-- IMUs in feet for balance and gait detection -->
<gazebo reference="left_foot_imu">
  <sensor name="left_foot_imu" type="imu">
    <!-- Configuration -->
  </sensor>
</gazebo>
```

### Integration with Control Systems

IMU data is critical for humanoid control:

```python
# Example IMU-based balance control
import numpy as np
from scipy.spatial.transform import Rotation as R

class BalanceController:
    def __init__(self):
        self.orientation = np.array([0, 0, 0, 1])  # w, x, y, z quaternion
        self.angular_velocity = np.array([0, 0, 0])  # x, y, z angular rates
        self.last_update_time = None
        
    def update_from_imu(self, imu_msg):
        # Update internal state with IMU data
        self.orientation = np.array([
            imu_msg.orientation.w,
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z
        ])
        
        self.angular_velocity = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
        
        # Calculate roll and pitch for balance control
        rotation = R.from_quat(self.orientation)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler_angles
        
        # Use tilt angles for balance control
        self.adjust_for_balance(roll, pitch)
```

## Force/Torque Sensors

### Force/Torque Sensor Simulation

Force/torque sensors are crucial for manipulation and contact detection:

```xml
<!-- Force/torque sensor in hand for manipulation -->
<gazebo reference="left_hand_force_torque_sensor">
  <sensor name="left_hand_force_torque" type="force_torque">
    <update_rate>100</update_rate>
    <always_on>true</always_on>
    <visualize>true</visualize>
    <force_torque>
      <frame>child</frame>  <!-- Measure forces in child frame -->
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
    <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
      <frame_name>left_hand_force_torque_frame</frame_name>
      <topic>left_hand/force_torque</topic>
      <body_name>left_hand</body_name>
    </plugin>
  </sensor>
</gazebo>
```

### Ground Contact Detection

For bipedal robots, foot force sensors are essential:

```xml
<!-- Force sensors in feet for ground contact detection -->
<gazebo reference="left_foot_contact_sensor">
  <sensor name="left_foot_contact" type="contact">
    <update_rate>100</update_rate>
    <contact>
      <collision>left_foot_collision</collision>
    </contact>
    <plugin name="contact_plugin" filename="libgazebo_ros_bumper.so">
      <frame_name>left_foot_frame</frame_name>
      <topic>left_foot/contact</topic>
    </plugin>
  </sensor>
</gazebo>
```

### Force/Torque Processing

```python
class ForceTorqueProcessor:
    def __init__(self):
        self.force_threshold = 5.0  # Newtons
        self.torque_threshold = 2.0  # N-m
        
    def process_force_torque(self, ft_msg):
        force = np.array([ft_msg.wrench.force.x, 
                         ft_msg.wrench.force.y, 
                         ft_msg.wrench.force.z])
        
        torque = np.array([ft_msg.wrench.torque.x,
                          ft_msg.wrench.torque.y,
                          ft_msg.wrench.torque.z])
        
        # Check for contact
        if np.linalg.norm(force) > self.force_threshold:
            self.handle_contact(force, torque)
        
        # Check for grasp success
        if (abs(force[2]) > 10.0 and  # Normal force
            abs(torque[0]) < 0.5 and   # Minimal roll torque
            abs(torque[1]) < 0.5):     # Minimal pitch torque
            return "grasp_successful"
        
        return "no_contact"
```

## LIDAR and Range Sensors

### LIDAR Simulation for Humanoid Robots

LIDAR sensors provide 360-degree range measurements:

```xml
<!-- 2D LIDAR simulation -->
<gazebo reference="base_lidar">
  <sensor name="base_lidar" type="ray">
    <pose>0 0 0.5 0 0 0</pose> <!-- Mount at 50cm height -->
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -π radians -->
          <max_angle>3.14159</max_angle>   <!-- π radians -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
      <topicName>base_scan</topicName>
      <frameName>base_lidar_frame</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### Multi-Planar LIDAR for 3D Perception

For more comprehensive perception:

```xml
<!-- 3D LIDAR simulation -->
<gazebo reference="head_3d_lidar">
  <sensor name="head_3d_lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>640</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>64</samples>
          <resolution>1</resolution>
          <min_angle>-0.5236</min_angle> <!-- -30 degrees -->
          <max_angle>0.3491</max_angle>   <!-- 20 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>20.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_laser.so">
      <topicName>head_points</topicName>
      <frameName>head_3d_lidar_frame</frameName>
      <min_range>0.1</min_range>
      <max_range>20.0</max_range>
    </plugin>
  </sensor>
</gazebo>
```

### Ultrasonic Range Sensors

For close-range detection:

```xml
<!-- Ultrasonic sensors for collision avoidance -->
<gazebo reference="front_ultrasonic">
  <sensor name="front_ultrasonic" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>1</samples>
          <min_angle>0</min_angle>
          <max_angle>0.1745</max_angle> <!-- 10 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.02</min>
        <max>4.0</max>
      </range>
    </ray>
    <plugin name="ultrasonic_plugin" filename="libgazebo_ros_ray.so">
      <ros>
        <namespace>ultrasonic</namespace>
        <remapping>~/out:=range</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

## Sensor Fusion in Simulation

### Combining Multiple Sensors

Humanoid robots must fuse data from multiple sensors:

```python
class SensorFusion:
    def __init__(self):
        self.imu_orientation = None
        self.camera_pose = None
        self.lidar_pose = None
        self.kalman_filter = self.initialize_kalman_filter()
        
    def initialize_kalman_filter(self):
        # Simplified Kalman filter for position estimation
        # State: [x, y, z, vx, vy, vz]
        dt = 0.01  # 100 Hz update rate
        
        # State transition model
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise
        Q = np.eye(6) * 0.1
        
        # Measurement matrix (we can measure position directly)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        R = np.eye(3) * 0.1
        
        # Error covariance
        P = np.eye(6)
        
        return KalmanFilter(F, H, Q, R, P)
        
    def update_state_estimate(self, imu_data, camera_data, lidar_data):
        # Predict step (using IMU data for motion model)
        self.kalman_filter.predict()
        
        # Update with camera measurement
        if camera_data is not None:
            camera_measurement = np.array([camera_data.x, camera_data.y, camera_data.z])
            self.kalman_filter.update(camera_measurement)
            
        # Update with LIDAR measurement
        if lidar_data is not None:
            lidar_measurement = np.array([lidar_data.x, lidar_data.y, lidar_data.z])
            self.kalman_filter.update(lidar_measurement)
            
        # Return fused estimate
        return self.kalman_filter.state()
```

### Visual-Inertial Odometry

```python
class VisualInertialOdometry:
    def __init__(self):
        self.prev_image = None
        self.vio_pose = np.eye(4)  # 4x4 transformation matrix
        self.imu_bias = np.zeros(6)  # 3 for gyro, 3 for accelerometer
        
    def process_frame(self, image_data, imu_data):
        if self.prev_image is None:
            self.prev_image = image_data
            return self.vio_pose
            
        # Extract visual features
        features = self.extract_features(image_data)
        
        # Match features with previous frame
        matches = self.match_features(features, self.prev_features)
        
        # Estimate visual motion
        visual_motion = self.estimate_motion(matches)
        
        # Integrate IMU data for motion prediction
        imu_motion = self.integrate_imu(imu_data)
        
        # Fuse visual and IMU estimates
        fused_motion = self.fuse_visual_imu(visual_motion, imu_motion)
        
        # Update pose estimate
        self.vio_pose = self.vio_pose @ fused_motion
        
        self.prev_image = image_data
        self.prev_features = features
        
        return self.vio_pose
```

## Sensor Noise and Realism

### Adding Realistic Noise Models

Real sensors have various noise characteristics:

```xml
<!-- Camera with realistic noise -->
<sensor type="camera" name="noisy_camera">
  <camera>
    <!-- Camera properties -->
  </camera>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev> <!-- 0.7% noise -->
  </noise>
</sensor>

<!-- IMU with bias and drift modeling -->
<sensor name="realistic_imu" type="imu">
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev> <!-- Random walk -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>1e-5</bias_stddev>
        </noise>
      </x>
      <!-- Similar for Y and Z -->
    </angular_velocity>
  </imu>
</sensor>
```

### Simulating Sensor Limitations

```python
class RealisticCamera:
    def __init__(self):
        self.depth_limit = 10.0  # meters
        self.fov = 80  # degrees
        self.resolution = (640, 480)
        
    def simulate_sensor_behavior(self, scene_data):
        # Apply depth limitations
        scene_data = self.apply_depth_limits(scene_data)
        
        # Add motion blur for fast movements
        scene_data = self.add_motion_blur(scene_data)
        
        # Add lens distortion
        scene_data = self.add_lens_distortion(scene_data)
        
        # Add quantization noise based on bit depth
        scene_data = self.add_quantization_noise(scene_data)
        
        return scene_data
        
    def apply_depth_limits(self, scene_data):
        # Objects beyond depth limit are not visible
        scene_data[scene_data > self.depth_limit] = float('inf')
        return scene_data
```

### Sensor Validation and Calibration

```python
def validate_sensor_simulation(robot_model, sensor_data):
    """Validate that simulated sensors behave realistically."""
    
    # Check update rates
    expected_rates = {
        'imu': 100,    # Hz
        'camera': 30,  # Hz
        'lidar': 10,   # Hz
    }
    
    for sensor_type, expected_rate in expected_rates.items():
        actual_rate = measure_sensor_rate(sensor_data[sensor_type])
        if abs(actual_rate - expected_rate) > 5:  # Allow 5Hz tolerance
            print(f"Warning: {sensor_type} rate is {actual_rate}Hz, expected ~{expected_rate}Hz")
    
    # Check data ranges
    if 'imu' in sensor_data:
        # Check for realistic acceleration values
        accel_magnitude = np.linalg.norm(sensor_data['imu']['linear_acceleration'])
        if accel_magnitude > 20:  # Greater than 2g
            print("Warning: IMU acceleration exceeds realistic range")
    
    # Check for sensor consistency
    if 'left_foot_contact' in sensor_data and 'imu' in sensor_data:
        # When foot is in contact, IMU should show appropriate forces
        pass  # Validation logic here
    
    return True
```

## Sensor Placement for Humanoid Robots

### Strategic Sensor Placement

```xml
<!-- Head sensors for perception -->
<link name="head">
  <inertial>...</inertial>
  <visual>...</visual>
  <collision>...</collision>
</link>

<!-- Camera placement -->
<joint name="head_camera_joint" type="fixed">
  <parent link="head"/>
  <child link="head_camera_frame"/>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
</joint>

<sensor name="head_camera" type="camera">
  <parent link="head_camera_frame"/>
  <!-- Camera definition -->
</sensor>

<!-- Torso sensors for balance -->
<sensor name="torso_imu" type="imu">
  <parent link="torso"/>
  <!-- IMU definition -->
</sensor>
```

### Multi-Sensor Integration

For humanoid robots, sensors must work together:

1. **Head sensors**: Vision, audio for environment perception
2. **Torso sensors**: IMUs for balance and orientation
3. **Limb sensors**: Joint encoders, force/torque for manipulation
4. **Foot sensors**: Pressure, contact for locomotion
5. **Hand sensors**: Force/torque, tactile for manipulation

### Sensor Redundancy

Humanoid robots often implement sensor redundancy:

- **Multiple IMUs**: One in torso, one in head, one in each foot
- **Multiple cameras**: Stereo vision, wide-angle for navigation
- **Multiple contact sensors**: In both feet, hands for manipulation
- **Multiple range sensors**: For robust obstacle detection

## Summary

In this chapter, we've explored the simulation of various sensor types essential for humanoid robotics. We've covered camera simulation (both RGB and depth), IMU simulation for balance and orientation, force/torque sensors for manipulation and contact detection, and LIDAR for environment perception.

The key to effective sensor simulation for humanoid robots is to model not just the ideal sensor behavior, but also the realistic limitations, noise characteristics, and failure modes that real sensors exhibit. This allows for the development of robust perception and control algorithms that will transfer effectively to physical robots.

Sensor fusion is particularly important for humanoid robots due to their complex multi-sensor requirements. Properly integrating data from cameras, IMUs, force sensors, and other modalities is essential for robust operation in real environments.

## Exercises

1. Create a complete sensor suite for a humanoid robot model, including cameras, IMUs, and contact sensors in the feet. Simulate the robot standing and walking to validate sensor behavior.

2. Implement a simple sensor fusion algorithm that combines IMU and camera data to improve pose estimation. Test the algorithm with simulated sensor noise.

3. Model the specific sensor placement for a humanoid robot designed for indoor navigation and manipulation. Include appropriate noise models and validate the sensor configurations.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
