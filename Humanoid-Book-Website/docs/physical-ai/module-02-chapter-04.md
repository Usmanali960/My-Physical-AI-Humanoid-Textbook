
---
id: module-02-chapter-04
title: Chapter 04 - Unity Visualization
sidebar_position: 8
---

# Chapter 04 - Unity Visualization

## Table of Contents
- [Overview](#overview)
- [Introduction to Unity for Robotics](#introduction-to-unity-for-robotics)
- [Unity vs. Gazebo: When to Use Each](#unity-vs-gazebo-when-to-use-each)
- [Setting Up Unity for Robot Visualization](#setting-up-unity-for-robot-visualization)
- [Importing Robot Models](#importing-robot-models)
- [Implementing Robot Kinematics in Unity](#implementing-robot-kinematics-in-unity)
- [ROS-Unity Integration](#ros-unity-integration)
- [Humanoid-Specific Visualization](#humanoid-specific-visualization)
- [Creating Interactive Environments](#creating-interactive-environments)
- [Performance Optimization](#performance-optimization)
- [Advanced Visualization Techniques](#advanced-visualization-techniques)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

While Gazebo is the standard simulation environment for robotics, Unity offers unique advantages for visualization, especially for humanoid robots. Unity's powerful rendering engine, extensive asset store, and intuitive interface make it ideal for creating realistic, interactive environments and visualizing complex humanoid behaviors. This chapter explores how to leverage Unity for humanoid robot visualization, from importing robot models to creating interactive environments that can complement traditional robotics simulation tools.

Unity's strength lies in its ability to create visually rich, interactive experiences that can help developers and end users better understand robot behavior. For humanoid robots, this is particularly valuable as their human-like form factor makes visualization more intuitive and engaging for stakeholders who may not be robotics experts.

## Introduction to Unity for Robotics

### Why Unity for Robotics?

Unity, originally designed for game development, has found significant applications in robotics for several reasons:

1. **High-Quality Graphics**: Photorealistic rendering capabilities
2. **Intuitive Interface**: Visual editor for scene design
3. **Extensive Asset Library**: Thousands of pre-made objects, environments, and tools
4. **Real-time Interaction**: Supports user interaction during simulation
5. **Cross-Platform Deployment**: Applications can run on multiple platforms
6. **VR/AR Support**: Can integrate with virtual and augmented reality systems

### Unity Robotics Packages

Unity provides several packages specifically for robotics:

- **Unity Robotics Hub**: Central repository for robotics packages
- **ROS-TCP-Connector**: Connects Unity to ROS/Ros2
- **Unity Robotics Package**: Tools for robotics simulation
- **ML-Agents**: For training AI agents in Unity

### Key Advantages for Humanoid Robots

- **Human-like Appearance**: Can create more realistic humanoid models
- **Interactive Environments**: Create real-world scenarios for testing
- **User-Friendly Visualization**: Non-technical stakeholders can easily understand
- **Animation System**: Advanced animation tools for humanoid movements
- **Character Controller**: Built-in tools for humanoid locomotion

## Unity vs. Gazebo: When to Use Each

### Gazebo Strengths
- **Physics Simulation**: Accurate physics for testing control algorithms
- **ROS Integration**: Native integration with ROS/Ros2
- **Robot Validation**: Validates real robot performance
- **Multi-Robot Simulation**: Handles many robots simultaneously
- **Performance**: Optimized for real-time physics

### Unity Strengths
- **Visual Quality**: Higher-quality rendering for presentations
- **User Interaction**: Real-time user interaction during visualization
- **Asset Library**: Extensive library of objects and environments
- **Animation Tools**: Advanced animation for humanoid movements
- **VR/AR Support**: Can create immersive experiences

### When to Use Unity for Humanoid Robots

1. **Presentation and Demo**: High-quality visualizations for stakeholders
2. **User Training**: Training humans to interact with robots
3. **Algorithm Visualization**: Understanding complex robot behaviors
4. **Human-Robot Interaction**: Testing interaction scenarios
5. **VR Training Environments**: Immersive robot programming

### Complementary Usage

For humanoid robotics development, Gazebo and Unity can be used together:
- Use Gazebo for physics-based simulation and control algorithm validation
- Use Unity for visualization and human-robot interaction testing
- Connect both through ROS communication for integrated workflows

## Setting Up Unity for Robot Visualization

### Unity Installation and Setup

For robotics visualization, use Unity Hub to install Unity with the following packages:
1. Unity Editor with the Universal Render Pipeline
2. Visual Studio integration for scripting
3. Linux Build Support (if deploying to ROS robots)

### Required Packages for Robotics

```bash
# Install Unity Robotics packages via Package Manager
# In Unity Editor: Window -> Package Manager -> Unity Registry
# Install:
# - ROS-TCP-Connector
# - Unity Machine Learning Agents (ML-Agents)
# - XR Interaction Toolkit (for VR support)
```

### Unity Project Structure for Robotics

```
UnityRobotVisualization/
├── Assets/
│   ├── Models/              # Robot models (imported from URDF/SDF)
│   ├── Materials/           # Materials for robot parts
│   ├── Scripts/             # Custom scripts for robot control
│   ├── Scenes/              # Different simulation environments
│   └── Plugins/             # External libraries (e.g., ROS connection)
├── ProjectSettings/
└── Packages/
```

### Basic Project Setup

1. Create a new 3D Unity project with Universal Render Pipeline (URP)
2. Configure physics settings to match your robot (gravity, fixed timestep)
3. Set up layer collisions appropriately
4. Install ROS-TCP-Connector package

## Importing Robot Models

### Converting from URDF to Unity

Robot models defined in URDF need to be converted to Unity format:

```bash
# Tools for converting URDF to Unity:
# 1. SW2URDF (SolidWorks to URDF converter that can export to other formats)
# 2. FreeCAD with URDF import/export
# 3. Manual conversion with Blender
```

### Manual Import Process

1. **Import Meshes**: Import robot mesh files (STL, OBJ, FBX) into Unity
2. **Create Hierarchy**: Organize meshes into parent-child hierarchy
3. **Set Colliders**: Add appropriate colliders to robot parts
4. **Configure Materials**: Add realistic materials to robot parts

### Robot Model Structure in Unity

```csharp
// Example robot hierarchy in Unity
public class RobotModel : MonoBehaviour
{
    public Transform head;
    public Transform torso;
    public Transform leftArm;
    public Transform rightArm;
    public Transform leftLeg;
    public Transform rightLeg;
    
    // Joint limits
    public float headYawMin = -1.0f, headYawMax = 1.0f;
    public float headPitchMin = -0.5f, headPitchMax = 0.5f;
    
    // References to joint objects
    public HingeJoint headYawJoint;
    public HingeJoint headPitchJoint;
    // ... other joints
}
```

### Importing with Proper Scaling

When importing robot models:
- Ensure consistent units (typically meters for robotics)
- Check that model dimensions match real-world dimensions
- Verify that the robot's origin point is at the base link
- Align coordinate systems: Unity uses Y-up, ROS uses Z-up

### Coordinate System Conversion

```csharp
// Function to convert ROS coordinates to Unity
public Vector3 ROS2Unity(Vector3 rosPoint)
{
    // ROS: X-forward, Y-left, Z-up
    // Unity: X-right, Y-up, Z-forward
    return new Vector3(rosPoint.y, rosPoint.z, rosPoint.x);
}

// Function to convert Unity coordinates to ROS
public Vector3 Unity2ROS(Vector3 unityPoint)
{
    // Unity: X-right, Y-up, Z-forward
    // ROS: X-forward, Y-left, Z-up
    return new Vector3(unityPoint.z, unityPoint.x, unityPoint.y);
}
```

## Implementing Robot Kinematics in Unity

### Forward Kinematics

```csharp
// Example forward kinematics implementation in Unity
public class RobotKinematics : MonoBehaviour
{
    public Transform[] joints;  // Array of joint transforms
    public float[] jointAngles; // Current joint angles
    public Transform[] links;   // Link transforms
    
    // DH parameters for each joint (example)
    public float[] linkLengths;
    public float[] linkTwists;
    public float[] linkOffsets;
    
    void UpdateRobotPose()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            // Set joint rotation based on current angle
            joints[i].localRotation = Quaternion.Euler(0, jointAngles[i] * Mathf.Rad2Deg, 0);
        }
    }
    
    Transform ComputeForwardKinematics(int linkIndex)
    {
        Transform current = joints[0];
        
        // Apply transformations up to the requested link
        for (int i = 0; i < linkIndex && i < joints.Length; i++)
        {
            current = current * CalculateTransform(i, jointAngles[i]);
        }
        
        return current;
    }
    
    Transform CalculateTransform(int jointIndex, float angle)
    {
        // Calculate transformation based on DH parameters
        // [Implementation of DH parameter transformation]
        return Transform.identity;
    }
}
```

### Inverse Kinematics

Unity has built-in support for inverse kinematics:

```csharp
using UnityEngine;

public class HumanoidIK : MonoBehaviour
{
    [Header("IK Targets")]
    public Transform leftHandTarget;
    public Transform rightHandTarget;
    public Transform leftFootTarget;
    public Transform rightFootTarget;
    
    [Header("IK Solvers")]
    public Animator animator;
    
    [Header("IK Weights")]
    public float leftHandPositionWeight = 1.0f;
    public float leftHandRotationWeight = 1.0f;
    public float rightHandPositionWeight = 1.0f;
    public float rightHandRotationWeight = 1.0f;
    
    void OnAnimatorIK(int layerIndex)
    {
        // Apply inverse kinematics for hand and feet positioning
        if (animator) {
            // Left hand IK
            animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, leftHandPositionWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, leftHandRotationWeight);
            if (leftHandTarget != null)
                animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandTarget.position);
            
            // Right hand IK
            animator.SetIKPositionWeight(AvatarIKGoal.RightHand, rightHandPositionWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.RightHand, rightHandRotationWeight);
            if (rightHandTarget != null)
                animator.SetIKPosition(AvatarIKGoal.RightHand, rightHandTarget.position);
                
            // Foot IK
            animator.SetIKPositionWeight(AvatarIKGoal.LeftFoot, 1.0f);
            animator.SetIKRotationWeight(AvatarIKGoal.LeftFoot, 1.0f);
            if (leftFootTarget != null)
                animator.SetIKPosition(AvatarIKGoal.LeftFoot, leftFootTarget.position);
        }
    }
}
```

### Realistic Joint Constraints

```csharp
// Custom joint constraint script
public class JointConstraint : MonoBehaviour
{
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float maxVelocity = 10f;
    
    private HingeJoint joint;
    private JointLimits limits;
    
    void Start()
    {
        joint = GetComponent<HingeJoint>();
        limits = joint.limits;
        limits.min = minAngle;
        limits.max = maxAngle;
        joint.limits = limits;
        joint.useLimits = true;
        
        // Also constrain velocity
        ConfigurableJoint configJoint = GetComponent<ConfigurableJoint>();
        if (configJoint != null)
        {
            SoftJointLimit lowLimit = configJoint.lowAngularXLimit;
            lowLimit.limit = minAngle * Mathf.Deg2Rad;
            configJoint.lowAngularXLimit = lowLimit;
            
            SoftJointLimit highLimit = configJoint.highAngularXLimit;
            highLimit.limit = maxAngle * Mathf.Deg2Rad;
            configJoint.highAngularXLimit = highLimit;
        }
    }
    
    public void SetJointAngle(float angle)
    {
        // Ensure requested angle is within limits
        angle = Mathf.Clamp(angle, minAngle, maxAngle);
        
        // Use motor to move to desired position
        JointMotor motor = joint.motor;
        motor.targetVelocity = CalculateVelocity(angle);
        motor.force = 100f; // Adjust as needed
        joint.motor = motor;
        joint.useMotor = true;
    }
    
    float CalculateVelocity(float targetAngle)
    {
        float currentAngle = joint.angle;
        float angleDiff = targetAngle - currentAngle;
        // Simple PD controller for velocity
        float velocity = 10f * angleDiff; // P term
        velocity = Mathf.Clamp(velocity, -maxVelocity, maxVelocity);
        return velocity;
    }
}
```

## ROS-Unity Integration

### Using ROS-TCP-Connector

The ROS-TCP-Connector package enables Unity to communicate with ROS/Ros2:

```csharp
using ROS2;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotName = "my_humanoid_robot";
    
    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>($"joint_states");
        
        // Subscribe to robot commands
        ros.Subscribe<JointStateMsg>($"/{robotName}/joint_commands", OnJointCommand);
    }
    
    void OnJointCommand(JointStateMsg jointState)
    {
        // Process joint state message and update Unity visualization
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];
            
            // Find corresponding joint in Unity and update position
            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint position (implementation depends on joint type)
                UpdateJointPosition(jointTransform, jointPosition);
            }
        }
    }
    
    void UpdateVisualization()
    {
        // Publish current joint states for any ROS nodes listening
        JointStateMsg jointState = new JointStateMsg();
        jointState.header = new std_msgs.HeaderMsg();
        jointState.header.stamp = new builtin_interfaces.TimeMsg();
        
        // Set joint names and positions for visualization
        // Implementation depends on your robot structure
    }
    
    Transform FindJointByName(string name)
    {
        // Search through robot hierarchy to find joint by name
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach(Transform child in allChildren)
        {
            if (child.name == name)
                return child;
        }
        return null;
    }
    
    void UpdateJointPosition(Transform joint, float position)
    {
        // Update the joint transform based on position
        // This depends on the joint type (revolute, prismatic, etc.)
        joint.localEulerAngles = new Vector3(0, position * Mathf.Rad2Deg, 0);
    }
}
```

### Publishing Visualization Data

```csharp
public class UnityVisualizationPublisher : MonoBehaviour
{
    ROSConnection ros;
    string robotNamespace = "humanoid_visualization";
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        // Publishers for visualization topics
        ros.RegisterPublisher<MarkerArrayMsg>($"visualization_marker_array");
        ros.RegisterPublisher<OdometryMsg>($"odom");
    }
    
    void PublishRobotPose()
    {
        // Publish robot position and orientation for RViz visualization
        OdometryMsg odom = new OdometryMsg();
        odom.header = new std_msgs.HeaderMsg();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        
        // Set position
        odom.pose.pose.position = new geometry_msgs.PointMsg(
            transform.position.x, 
            transform.position.y, 
            transform.position.z
        );
        
        // Set orientation
        odom.pose.pose.orientation = new geometry_msgs.QuaternionMsg(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        );
        
        // Publish the message
        ros.Publish($"/{robotNamespace}/odom", odom);
    }
    
    void PublishVisualizationMarkers()
    {
        // Create markers for visualization in RViz
        MarkerArrayMsg markerArray = new MarkerArrayMsg();
        
        // Example: Create a marker for the robot's center of mass
        MarkerMsg comMarker = new MarkerMsg();
        comMarker.header = new std_msgs.HeaderMsg();
        comMarker.header.frame_id = "base_link";
        comMarker.ns = "robot_com";
        comMarker.id = 0;
        comMarker.type = MarkerMsg.SPHERE;
        comMarker.action = MarkerMsg.ADD;
        comMarker.pose.position = new geometry_msgs.PointMsg(0, 0, 0.5f); // Example CoM position
        comMarker.scale = new geometry_msgs.Vector3Msg(0.1f, 0.1f, 0.1f);
        comMarker.color = new std_msgs.ColorRGBAMsg(1.0f, 0.0f, 0.0f, 1.0f); // Red color
        
        markerArray.markers.Add(comMarker);
        
        ros.Publish($"/{robotNamespace}/visualization_marker_array", markerArray);
    }
}
```

## Humanoid-Specific Visualization

### Humanoid Animation Systems

Unity's Animation system is well-suited for humanoid robots:

```csharp
using UnityEngine;
using UnityEngine.Animations;

public class HumanoidAnimator : MonoBehaviour
{
    [Header("Animation Parameters")]
    public Animator animator;
    public float walkSpeed = 1.0f;
    public float turnSpeed = 50.0f;
    
    [Header("Gait Patterns")]
    public AnimationCurve walkCycle;
    public AnimationClip idleClip;
    public AnimationClip walkClip;
    public AnimationClip turnLeftClip;
    public AnimationClip turnRightClip;
    
    void Update()
    {
        if (animator == null) return;
        
        // Control walking animation
        float moveX = Input.GetAxis("Horizontal");
        float moveZ = Input.GetAxis("Vertical");
        
        // Update animation parameters
        animator.SetFloat("MoveX", moveX);
        animator.SetFloat("MoveZ", moveZ);
        animator.SetFloat("Speed", Mathf.Sqrt(moveX*moveX + moveZ*moveZ));
        
        // Blend between clips based on movement
        if (Mathf.Abs(moveX) > 0.1f)
        {
            // Turning
            animator.CrossFade(moveX > 0 ? "TurnRight" : "TurnLeft", 0.2f);
        }
        else if (Mathf.Abs(moveZ) > 0.1f)
        {
            // Walking forward/backward
            animator.CrossFade("Walk", 0.2f);
        }
        else
        {
            // Idle
            animator.CrossFade("Idle", 0.2f);
        }
    }
}
```

### Visualizing Internal States

```csharp
public class StateVisualizer : MonoBehaviour
{
    public GameObject[] stateIndicators; // Different visual indicators for states
    public Renderer[] statusLights;      // LED-like status indicators
    public Material[] statusMaterials;   // Different materials for states
    
    public enum RobotState { Idle, Walking, Standing, Falling, Manipulating }
    
    private RobotState currentState = RobotState.Idle;
    
    void Update()
    {
        UpdateStateVisualization();
    }
    
    void SetRobotState(RobotState newState)
    {
        currentState = newState;
        UpdateStateVisualization();
    }
    
    void UpdateStateVisualization()
    {
        // Update visual indicators based on state
        for (int i = 0; i < stateIndicators.Length; i++)
        {
            stateIndicators[i].SetActive(i == (int)currentState);
        }
        
        // Update status lights
        if (statusLights.Length > 0)
        {
            int materialIndex = (int)currentState % statusMaterials.Length;
            foreach (Renderer light in statusLights)
            {
                light.material = statusMaterials[materialIndex];
            }
        }
    }
    
    // Method to visualize center of mass
    public void VisualizeCenterOfMass(Vector3 comPosition, bool isStable)
    {
        // Create a visual indicator for center of mass
        GameObject comIndicator = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        comIndicator.transform.position = comPosition;
        comIndicator.transform.localScale = Vector3.one * 0.05f;
        
        // Color based on stability
        Renderer renderer = comIndicator.GetComponent<Renderer>();
        renderer.material = new Material(Shader.Find("Standard"));
        renderer.material.color = isStable ? Color.green : Color.red;
        
        // Auto-destroy after a few seconds
        Destroy(comIndicator, 3.0f);
    }
}
```

### Expressive Features for Humanoid Robots

```csharp
public class ExpressiveFeatures : MonoBehaviour
{
    [Header("Face Components")]
    public SkinnedMeshRenderer faceRenderer;
    public BlendShapeSetting[] eyesShapes;
    public BlendShapeSetting[] mouthShapes;
    
    [Header("LED Indicators")]
    public Light[] eyeLights;
    public Light[] statusLights;
    
    // Expressions
    [Range(0, 100)] public float happiness = 50;
    [Range(0, 100)] public float surprise = 0;
    [Range(0, 100)] public float sadness = 0;
    
    void Update()
    {
        UpdateFaceExpression();
        UpdateLEDStatus();
    }
    
    void UpdateFaceExpression()
    {
        if (faceRenderer != null)
        {
            // Apply blend shapes for different expressions
            faceRenderer.SetBlendShapeWeight(0, happiness);    // Smiling
            faceRenderer.SetBlendShapeWeight(1, surprise);     // Wide eyes
            faceRenderer.SetBlendShapeWeight(2, sadness);      // Frowning
        }
    }
    
    void UpdateLEDStatus()
    {
        // Update eye and status lights based on robot state
        foreach (Light eye in eyeLights)
        {
            eye.intensity = happiness / 100.0f * 2.0f; // Max intensity based on happiness
            eye.color = happiness > 50 ? Color.blue : Color.white;
        }
    }
}

[System.Serializable]
public class BlendShapeSetting
{
    public string name;
    public int index;
    [Range(0, 100)] public float weight;
}
```

## Creating Interactive Environments

### Environment Design Principles

For humanoid robot visualization, environments should:
- Be appropriately scaled for human-sized robots
- Include interactive objects of various sizes
- Have realistic lighting and materials
- Support the robot's intended functions

### Interactive Object System

```csharp
using UnityEngine;

public class InteractiveObject : MonoBehaviour
{
    public enum ObjectType { Unknown, Furniture, Tool, Food, Door }
    public ObjectType objectType = ObjectType.Unknown;
    
    [Header("Interaction Properties")]
    public bool canBeGrasped = true;
    public bool canCollide = true;
    public bool affectedByPhysics = true;
    
    [Header("Grasp Points")]
    public Transform[] graspPoints;
    
    [Header("Highlight Settings")]
    public Material originalMaterial;
    public Material highlightMaterial;
    private Material[] originalMaterials;
    
    void Start()
    {
        // Store original materials
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            originalMaterials = rend.materials;
        }
    }
    
    public void Highlight(bool state)
    {
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            Material[] materials = new Material[rend.materials.Length];
            for (int i = 0; i < materials.Length; i++)
            {
                materials[i] = state ? highlightMaterial : originalMaterials[i];
            }
            rend.materials = materials;
        }
    }
    
    void OnMouseEnter()
    {
        Highlight(true);
    }
    
    void OnMouseExit()
    {
        Highlight(false);
    }
    
    void OnMouseDown()
    {
        // Handle object interaction
        if (canBeGrasped)
        {
            Debug.Log($"Can grasp object: {gameObject.name}");
            // Trigger grasping animation or ROS command
        }
    }
    
    void StartInteraction(Transform hand)
    {
        // Handle more complex interaction
        // This might involve grasping, manipulation, etc.
        Debug.Log($"Starting interaction with {gameObject.name}");
    }
}
```

### Humanoid-Friendly Environments

```csharp
public class HumanoidEnvironment : MonoBehaviour
{
    [Header("Environment Elements")]
    public Transform[] doorWaypoints;      // Waypoints through doorways
    public Transform[] chairPositions;      // Sitting positions
    public Transform[] tablePositions;      // Interaction points
    public Transform[] shelfPositions;      // Reachable positions
    
    [Header("Humanoid Constraints")]
    public float minPassageWidth = 0.8f;    // Minimum door width
    public float maxStepHeight = 0.15f;     // Maximum step up height
    public float maxReachDistance = 1.2f;   // Maximum arm reach
    
    void Start()
    {
        ValidateEnvironment();
    }
    
    void ValidateEnvironment()
    {
        // Check if environment is suitable for humanoid robot
        ValidateDoorWidths();
        ValidateStepHeights();
        ValidateReachability();
    }
    
    void ValidateDoorWidths()
    {
        // Check all doors/doorways in the environment
        foreach (Transform door in doorWaypoints)
        {
            // Check width of door opening
            float doorWidth = CalculateDoorWidth(door);
            if (doorWidth < minPassageWidth)
            {
                Debug.LogWarning($"Door at {door.position} is too narrow for humanoid robot");
            }
        }
    }
    
    float CalculateDoorWidth(Transform door)
    {
        // Implement logic to determine door width
        // This might involve raycasting or checking mesh dimensions
        return 1.0f; // Placeholder
    }
    
    void ValidateReachability()
    {
        // Check if table/shelf positions are within humanoid reach
        foreach (Transform pos in tablePositions)
        {
            if (Vector3.Distance(transform.position, pos.position) > maxReachDistance)
            {
                Debug.LogWarning($"Position {pos.position} is out of reach for humanoid robot");
            }
        }
    }
    
    public bool CanNavigateTo(Vector3 target)
    {
        // Pathfinding to check if humanoid can navigate to target
        // This would typically use Unity's NavMesh system
        NavMeshPath path = new NavMeshPath();
        bool pathExists = NavMesh.CalculatePath(transform.position, target, NavMesh.AllAreas, path);
        
        // Additional humanoid-specific validation
        if (pathExists)
        {
            // Check path for humanoid constraints
            return ValidatePathForHumanoid(path);
        }
        
        return false;
    }
    
    bool ValidatePathForHumanoid(NavMeshPath path)
    {
        // Check if path is suitable for humanoid robot
        // Consider step height, narrow passages, etc.
        return true; // Simplified implementation
    }
}
```

## Performance Optimization

### LOD (Level of Detail) for Robot Models

```csharp
using UnityEngine;

public class RobotLOD : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public GameObject lodObject;
        public float distance;
        public ComputeShader lodComputeShader;
    }
    
    public LODLevel[] lodLevels;
    public Transform viewer; // Camera position for distance calculation
    
    void Start()
    {
        if (viewer == null)
            viewer = Camera.main.transform;
    }
    
    void Update()
    {
        float distance = Vector3.Distance(transform.position, viewer.position);
        
        // Find the appropriate LOD level
        GameObject activeLOD = null;
        foreach (LODLevel level in lodLevels)
        {
            if (distance <= level.distance)
            {
                activeLOD = level.lodObject;
            }
            else
            {
                level.lodObject.SetActive(false);
            }
        }
        
        if (activeLOD != null)
            activeLOD.SetActive(true);
    }
}
```

### Optimizing Scene Rendering

```csharp
public class RenderingOptimizer : MonoBehaviour
{
    [Header("Culling Settings")]
    public float maxRenderDistance = 20.0f;
    public float updateInterval = 0.1f;
    
    [Header("Frustum Culling")]
    public Camera mainCamera;
    private float lastUpdateTime;
    
    void Start()
    {
        if (mainCamera == null)
            mainCamera = Camera.main;
    }
    
    void Update()
    {
        if (Time.time - lastUpdateTime > updateInterval)
        {
            UpdateObjectVisibility();
            lastUpdateTime = Time.time;
        }
    }
    
    void UpdateObjectVisibility()
    {
        // Iterate through all robot parts and environment objects
        Renderer[] allRenderers = FindObjectsOfType<Renderer>();
        
        foreach (Renderer r in allRenderers)
        {
            float distance = Vector3.Distance(r.transform.position, transform.position);
            
            if (distance > maxRenderDistance)
            {
                r.enabled = false; // Disable rendering
            }
            else if (!mainCamera.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0)).IsRayIntersecting(r.bounds))
            {
                r.enabled = false; // Not in view, disable rendering
            }
            else
            {
                r.enabled = true; // Within distance and view, enable rendering
            }
        }
    }
}
```

### Reducing Physics Load

```csharp
public class PhysicsOptimizer : MonoBehaviour
{
    [Header("Simulation Optimization")]
    public float simulationDistance = 10.0f; // Only simulate nearby objects
    public bool useContinuousDetection = false; // Use for fast-moving objects
    public float fixedTimestep = 0.02f; // 50 Hz physics update
    
    void Start()
    {
        // Optimize physics settings
        Time.fixedDeltaTime = fixedTimestep;
        
        Rigidbody[] allRigidbodies = FindObjectsOfType<Rigidbody>();
        foreach (Rigidbody rb in allRigidbodies)
        {
            // Optimize rigidbody for performance
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            rb.sleepThreshold = 0.005f;
            
            // Use continuous collision detection only for fast-moving objects
            rb.collisionDetectionMode = useContinuousDetection ? 
                CollisionDetectionMode.Continuous : 
                CollisionDetectionMode.Discrete;
        }
    }
    
    void Update()
    {
        // Only enable physics simulation for objects within distance
        Rigidbody[] allRigidbodies = FindObjectsOfType<Rigidbody>();
        foreach (Rigidbody rb in allRigidbodies)
        {
            if (Vector3.Distance(rb.position, transform.position) > simulationDistance)
            {
                rb.isKinematic = true; // Disable physics simulation
            }
            else
            {
                rb.isKinematic = false; // Enable physics simulation
            }
        }
    }
}
```

## Advanced Visualization Techniques

### Real-time Rendering Techniques

```csharp
using UnityEngine.Rendering.Universal;

public class AdvancedRobotRendering : MonoBehaviour
{
    [Header("Rendering Effects")]
    public bool useBloom = true;
    public bool useSSR = false; // Screen Space Reflections
    public bool useDoF = false; // Depth of Field
    
    [Header("Custom Shaders")]
    public Shader robotHighlightShader;
    public float highlightIntensity = 2.0f;
    
    private UniversalRenderPipelineAsset renderPipelineAsset;
    
    void Start()
    {
        SetupAdvancedRendering();
    }
    
    void SetupAdvancedRendering()
    {
        // Configure render pipeline effects
        if (useBloom)
        {
            // Configure bloom effect
            var renderer = RenderPipelineManager.currentPipeline as UniversalRendererData;
            // Add bloom renderer feature
        }
        
        // Configure custom materials for robot
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        foreach (Renderer r in renderers)
        {
            // Apply robot-specific materials and shaders
            Material[] materials = r.materials;
            for (int i = 0; i < materials.Length; i++)
            {
                if (materials[i].name.Contains("Robot"))
                {
                    materials[i].SetFloat("_HighlightIntensity", highlightIntensity);
                }
            }
            r.materials = materials;
        }
    }
    
    public void HighlightRobot(bool state)
    {
        // Apply visual highlight effect to entire robot
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        foreach (Renderer r in renderers)
        {
            Material[] materials = r.materials;
            for (int i = 0; i < materials.Length; i++)
            {
                materials[i].SetFloat("_Highlighted", state ? 1.0f : 0.0f);
            }
            r.materials = materials;
        }
    }
}
```

### Multi-camera Visualization

```csharp
public class MultiCameraView : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera mainCamera;
    public Camera headCamera;        // First-person view
    public Camera externalCamera;    // Third-person view
    public Camera sensorCamera;      // Simulated sensor view
    
    [Header("Camera Switching")]
    public KeyCode switchCamKey = KeyCode.Tab;
    private Camera[] cameras;
    private int currentCameraIndex = 0;
    
    void Start()
    {
        // Initialize cameras
        cameras = new Camera[] { mainCamera, headCamera, externalCamera, sensorCamera };
        SwitchToCamera(0);
    }
    
    void Update()
    {
        if (Input.GetKeyDown(switchCamKey))
        {
            SwitchToNextCamera();
        }
    }
    
    void SwitchToNextCamera()
    {
        SwitchToCamera((currentCameraIndex + 1) % cameras.Length);
    }
    
    void SwitchToCamera(int index)
    {
        for (int i = 0; i < cameras.Length; i++)
        {
            if (cameras[i] != null)
            {
                cameras[i].gameObject.SetActive(i == index);
            }
        }
        currentCameraIndex = index;
        
        // Optionally update UI or ROS with camera switch
        Debug.Log($"Switched to camera: {cameras[index].name}");
    }
    
    public Texture2D CaptureCameraView(Camera cam)
    {
        // Capture current camera view for processing
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;
        
        cam.Render();
        
        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        image.Apply();
        
        RenderTexture.active = currentRT;
        return image;
    }
}
```

### Data Visualization Integration

```csharp
public class DataVisualization : MonoBehaviour
{
    [Header("Visualization Elements")]
    public LineRenderer pathRenderer;
    public GameObject[] trajectoryPoints;
    public TextMeshProUGUI statusText;
    
    [Header("Data Sources")]
    public float[] sensorData;
    public Vector3[] positionHistory;
    public float[] balanceMetrics;
    
    void Update()
    {
        UpdateTrajectoryVisualization();
        UpdateStatusDisplay();
        UpdateSensorVisualization();
    }
    
    void UpdateTrajectoryVisualization()
    {
        if (positionHistory.Length > 1)
        {
            pathRenderer.positionCount = positionHistory.Length;
            pathRenderer.SetPositions(positionHistory);
        }
    }
    
    void UpdateStatusDisplay()
    {
        if (statusText != null)
        {
            // Create a comprehensive status display
            string status = $"Position: {transform.position}\n" +
                           $"Velocity: {GetComponent<Rigidbody>().velocity.magnitude:F2}\n" +
                           $"Balance: {GetBalanceStatus():P1}\n" +
                           $"Battery: {GetBatteryLevel():P1}";
            
            statusText.text = status;
        }
    }
    
    void UpdateSensorVisualization()
    {
        // Visualize sensor data as graphs or indicators
        for (int i = 0; i < sensorData.Length; i++)
        {
            if (trajectoryPoints[i] != null)
            {
                // Adjust position based on sensor value
                trajectoryPoints[i].transform.localPosition = 
                    new Vector3(i * 0.1f, sensorData[i] * 0.5f, 0);
            }
        }
    }
    
    float GetBalanceStatus()
    {
        // Calculate current balance status (0-1)
        // Implementation depends on your balance control system
        return 0.95f; // Placeholder
    }
    
    float GetBatteryLevel()
    {
        // Calculate current battery level (0-1)
        return 0.75f; // Placeholder
    }
}
```

## Summary

In this chapter, we've explored Unity as a powerful visualization tool for humanoid robotics. We've covered how to import robot models from URDF, implement kinematics in Unity, integrate with ROS systems, and create humanoid-specific visualizations that can enhance understanding and interaction.

Unity provides unique advantages over traditional simulation tools like Gazebo, particularly in terms of visual quality, user interaction, and the ability to create engaging, interactive environments for testing human-robot interaction scenarios. While Gazebo excels at physics simulation for control algorithm validation, Unity excels at creating visually rich, user-friendly visualizations and interactive environments.

The key to effective Unity visualization for humanoid robots is to leverage Unity's strength in graphics and interaction to complement rather than replace traditional robotics simulation tools, creating an integrated workflow that serves different aspects of the development process.

## Exercises

1. Create a Unity scene with a simplified humanoid robot model and implement basic kinematics to visualize joint movements. Connect to a ROS system to visualize real joint state data.

2. Implement a basic animation system for the humanoid robot that shows different states (idle, walking, manipulating) and integrate it with ROS communication.

3. Create an interactive environment in Unity that includes furniture and objects appropriate for humanoid interaction, with collision detection and visualization features.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*
