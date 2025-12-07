---
id: module-06-chapter-03
title: Chapter 03 - Manipulation and Grasping Control
sidebar_position: 23
---

# Chapter 03 - Manipulation and Grasping Control

## Table of Contents
- [Overview](#overview)
- [Introduction to Robotic Manipulation](#introduction-to-robotic-manipulation)
- [Grasp Planning and Analysis](#grasp-planning-and-analysis)
- [Hand Design and Actuation](#hand-design-and-actuation)
- [Force and Tactile Control](#force-and-tactile-control)
- [Grasp Stability and Compliance](#grasp-stability-and-compliance)
- [Learning-Based Grasping](#learning-based-grasping)
- [Multi-Fingered Hand Control](#multi-fingered-hand-control)
- [Grasp Adaptation and Recovery](#grasp-adaptation-and-recovery)
- [Safety in Manipulation](#safety-in-manipulation)
- [Evaluation Metrics](#evaluation-metrics)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Manipulation and grasping control form the foundation of physical interaction for humanoid robots. The ability to grasp and manipulate objects with human-like dexterity enables these robots to perform complex tasks in human environments, using the same tools and interfaces designed for humans. This chapter explores the technical challenges and solutions in robotic manipulation, from grasp planning and hand design to force control and learning-based approaches.

Robotic manipulation encompasses the perception, planning, and control aspects required for a robot to interact with objects in its environment. Grasping, a critical component of manipulation, involves the stable holding of objects using robot hands, requiring precise control of contact forces and hand configuration. The chapter covers both traditional analytical approaches and modern learning-based methods, highlighting how these techniques are adapted for humanoid platforms with anthropomorphic hands.

The complexity of manipulation control stems from the physical interaction between the robot and objects, which involves complex contact mechanics, force distributions, and dynamic behavior. Unlike rigid robotic arms that move in free space, manipulation tasks require compliance control and adaptive responses to changing contact conditions.

## Introduction to Robotic Manipulation

### Manipulation Fundamentals

Robotic manipulation involves the controlled interaction between a robot and objects in its environment to achieve specific goals. The fundamental components of manipulation include:

```python
# Key components of robotic manipulation
MANIPULATION_COMPONENTS = {
    'perception': {
        'function': 'Identify and locate objects in the environment',
        'techniques': ['Computer vision', 'Depth sensing', 'Object recognition'],
        'output': 'Object poses, shapes, physical properties'
    },
    'planning': {
        'function': 'Determine sequence of actions to achieve manipulation goals',
        'techniques': ['Motion planning', 'Grasp planning', 'Task planning'],
        'output': 'Trajectories, grasp configurations, action sequences'
    },
    'control': {
        'function': 'Execute planned actions with appropriate force and motion control',
        'techniques': ['Impedance control', 'Force control', 'Position control'],
        'output': 'Joint torques, end-effector forces, hand configurations'
    },
    'learning': {
        'function': 'Improve manipulation skills through experience',
        'techniques': ['Reinforcement learning', 'Imitation learning', 'Supervised learning'],
        'output': 'Improved policies, skill models, adaptation mechanisms'
    }
}

class ManipulationSystem:
    def __init__(self):
        self.perception_system = PerceptionSystem()
        self.planning_system = PlanningSystem()
        self.control_system = ControlSystem()
        self.learning_system = LearningSystem()
        
    def execute_manipulation_task(self, task_description, environment_state):
        """Execute a complete manipulation task"""
        # Step 1: Perceive the environment
        scene_info = self.perception_system.analyze_scene(environment_state)
        
        # Step 2: Plan manipulation sequence
        manipulation_plan = self.planning_system.plan_manipulation(
            task_description, scene_info
        )
        
        # Step 3: Execute with appropriate control
        execution_result = self.control_system.execute_plan(
            manipulation_plan, scene_info
        )
        
        # Step 4: Learn from the experience
        self.learning_system.update_from_experience(
            task_description, manipulation_plan, execution_result
        )
        
        return execution_result
```

### Manipulation Task Categories

Manipulation tasks can be categorized based on their complexity and interaction requirements:

```cpp
// Manipulation task categories
enum ManipulationTaskType {
    REACHING,        // Simple reaching to a location
    GRASPING,        // Grasping a target object
    TRANSPORT,       // Moving object from one location to another
    PLACEMENT,       // Precise placement of object
    COMPLIANT_TASK,  // Tasks requiring compliance (e.g., insertion)
    DEFORMABLE_MANIPULATION, // Manipulation of deformable objects
    DUAL_ARM_TASK    // Tasks requiring both arms
};

class ManipulationTaskPlanner {
public:
    ManipulationPlan planTask(ManipulationTaskType task_type,
                             const TaskParameters& params) {
        switch(task_type) {
            case REACHING:
                return planReachingTask(params);
            case GRASPING:
                return planGraspingTask(params);
            case TRANSPORT:
                return planTransportTask(params);
            case PLACEMENT:
                return planPlacementTask(params);
            case COMPLIANT_TASK:
                return planCompliantTask(params);
            case DEFORMABLE_MANIPULATION:
                return planDeformableManipulationTask(params);
            case DUAL_ARM_TASK:
                return planDualArmTask(params);
            default:
                return planGeneralTask(params);
        }
    }

private:
    ManipulationPlan planReachingTask(const TaskParameters& params) {
        // Plan a simple reaching motion
        return generateReachingTrajectory(params.target_pose);
    }
    
    ManipulationPlan planGraspingTask(const TaskParameters& params) {
        // Plan approach, grasp, and lift motions
        ManipulationPlan plan;
        
        // Approach phase
        plan.addPhase(generateApproachTrajectory(params.object_pose));
        
        // Grasp phase
        plan.addPhase(generateGraspMotion(params.grasp_config));
        
        // Lift phase
        plan.addPhase(generateLiftTrajectory());
        
        return plan;
    }
    
    ManipulationPlan planTransportTask(const TaskParameters& params) {
        // Plan transport from current to target location
        ManipulationPlan plan;
        
        // Maintain grasp during transport
        plan.addPhase(maintainGraspDuringTransport(
            params.start_pose, params.target_pose
        ));
        
        return plan;
    }
    
    ManipulationPlan planPlacementTask(const TaskParameters& params) {
        // Plan precise placement of object
        ManipulationPlan plan;
        
        // Approach placement location
        plan.addPhase(generateApproachForPlacement(params.target_pose));
        
        // Execute placement with appropriate compliance
        plan.addPhase(executePlacementWithCompliance());
        
        // Release object
        plan.addPhase(generateReleaseMotion());
        
        return plan;
    }
    
    ManipulationPlan planCompliantTask(const TaskParameters& params) {
        // Plan for tasks requiring compliance (e.g., insertion)
        ManipulationPlan plan;
        
        // Use force control for compliant behavior
        plan.addPhase(generateCompliantApproach(params.target_region));
        plan.addPhase(generateCompliantInsertion(params.insertion_path));
        
        return plan;
    }
    
    ManipulationPlan planDeformableManipulationTask(const TaskParameters& params) {
        // Plan for manipulating deformable objects
        ManipulationPlan plan;
        
        // Use special control strategies for deformable objects
        plan.addPhase(deformableObjectApproach(params.object_properties));
        plan.addPhase(deformableObjectManipulation(
            params.target_shape, params.material_properties
        ));
        
        return plan;
    }
    
    ManipulationPlan planDualArmTask(const TaskParameters& params) {
        // Plan coordinated dual-arm manipulation
        ManipulationPlan plan;
        
        // Generate coordinated motion for both arms
        auto dual_arm_trajectories = generateDualArmTrajectory(
            params.left_arm_task, params.right_arm_task
        );
        
        plan.addPhase(dual_arm_trajectories);
        
        return plan;
    }
    
    ManipulationPlan planGeneralTask(const TaskParameters& params) {
        // Default general-purpose manipulation planning
        return ManipulationPlan();
    }
};
```

### Degrees of Freedom and Configuration Space

The complexity of manipulation depends significantly on the robot's degrees of freedom and the dimensionality of its configuration space:

```python
# Analysis of manipulation DOF requirements
MANIPULATION_DOF_REQUIREMENTS = {
    'simple_grasping': {
        'requisite_dof': 6,  # Position (3) + Orientation (3)
        'redundancy_benefit': 'Avoid singularities, obstacle avoidance',
        'typical_implementation': '6-DOF arm + 1-DOF gripper'
    },
    'dexterous_manipulation': {
        'requisite_dof': 15,  # Human hand has ~16 DOF
        'redundancy_benefit': 'Finger dexterity, multiple grasp types',
        'typical_implementation': '7-DOF arm + 12-DOF hand'
    },
    'bimanual_manipulation': {
        'requisite_dof': 30,  # Two arms + two hands
        'redundancy_benefit': 'Complex tasks, tool use, assembly',
        'typical_implementation': 'Two 7-DOF arms + two dexterous hands'
    }
}

def analyze_configuration_space(robot_model):
    """Analyze the configuration space of a robot for manipulation tasks"""
    # Calculate joint limits
    joint_limits = robot_model.get_joint_limits()
    
    # Calculate configuration space dimensionality
    cs_dimension = len(joint_limits)
    
    # Identify singularities
    singularities = find_singularities(robot_model)
    
    # Calculate manipulability measures
    manipulability_measures = calculate_manipulability(robot_model)
    
    return {
        'dimensionality': cs_dimension,
        'joint_limits': joint_limits,
        'singularities': singularities,
        'manipulability_measures': manipulability_measures,
        'workspace_volume': calculate_workspace_volume(robot_model)
    }

class ConfigurationSpaceAnalyzer:
    def __init__(self, robot_description):
        self.robot = self.load_robot_model(robot_description)
        
    def analyze_for_manipulation(self, task_pose):
        """Analyze configuration space for a specific manipulation task"""
        # Find all configurations that can reach the task pose
        valid_configs = self.find_reachable_configurations(task_pose)
        
        # Evaluate each configuration for manipulation quality
        config_evaluations = []
        for config in valid_configs:
            evaluation = self.evaluate_configuration_for_manipulation(
                config, task_pose
            )
            config_evaluations.append({
                'configuration': config,
                'evaluation': evaluation
            })
        
        # Sort by manipulation quality
        sorted_configs = sorted(
            config_evaluations, 
            key=lambda x: x['evaluation']['quality'], 
            reverse=True
        )
        
        return sorted_configs[:10]  # Return top 10 configurations
    
    def evaluate_configuration_for_manipulation(self, config, task_pose):
        """Evaluate a specific configuration for manipulation quality"""
        # Calculate kinematic dexterity
        jacobian = self.calculate_jacobian(config)
        manipulability = calculate_manipulability_index(jacobian)
        
        # Calculate force transmission capability
        force_ellipse = calculate_force_ellipse(jacobian)
        force_capability = force_ellipse.volume
        
        # Calculate dexterity indices
        condition_number = np.linalg.cond(jacobian)
        dexterity_index = 1.0 / condition_number if condition_number != 0 else 0
        
        # Calculate proximity to joint limits
        limit_proximity = calculate_distance_to_limits(config, self.robot.joint_limits)
        
        return {
            'manipulability': manipulability,
            'force_capability': force_capability,
            'dexterity_index': dexterity_index,
            'limit_proximity': limit_proximity,
            'quality': manipulability * force_capability * dexterity_index
        }
```

## Grasp Planning and Analysis

### Grasp Representations

Grasps can be represented in various ways depending on the planning approach and application:

```python
# Different grasp representations
GRASP_REPRESENTATIONS = {
    'grasp_frame': {
        'description': 'Position and orientation of the grasp relative to the object',
        'parameters': ['position', 'orientation (quaternion)', 'approach_direction', 'grasp_type'],
        'use_case': 'Geometric grasp planning, grasp database storage'
    },
    'contact_points': {
        'description': '3D positions of contact points between hand and object',
        'parameters': ['position', 'normal_vector', 'friction_coefficient', 'force_application'],
        'use_case': 'Physics-based grasp analysis, grasp synthesis'
    },
    'hand_configuration': {
        'description': 'Joint angles of the robotic hand in the grasp',
        'parameters': ['joint_angles', 'actuator_commands', 'finger_positions'],
        'use_case': 'Direct control, hardware-specific grasping'
    },
    'grasp_template': {
        'description': 'Canonical grasp pose parameterized on object geometry',
        'parameters': ['template_type', 'object_features', 'parameterization'],
        'use_case': 'Template-based grasp planning for known object classes'
    }
}

class Grasp:
    def __init__(self, approach_direction, grasp_width, finger_positions, 
                 grasp_type='cylindrical', quality=0.0):
        self.approach_direction = np.array(approach_direction)
        self.grasp_width = grasp_width
        self.finger_positions = finger_positions
        self.grasp_type = grasp_type
        self.quality = quality
        self.stability = 0.0
        self.force_closure = False
        self.contact_points = []
        
    def to_grasp_frame(self):
        """Convert to canonical grasp frame representation"""
        return {
            'approach': self.approach_direction,
            'width': self.grasp_width,
            'type': self.grasp_type,
            'quality': self.quality
        }
        
    def from_contact_points(self, contact_points):
        """Initialize grasp from contact points"""
        self.contact_points = contact_points
        # Calculate grasp properties from contact points
        self.calculate_grasp_properties()
        
    def calculate_grasp_properties(self):
        """Calculate properties like stability and force closure from contact points"""
        # Implement physics-based grasp analysis
        # Check force closure, grasp stability, etc.
        pass

class GraspPlanner:
    def __init__(self):
        self.object_analyzer = ObjectAnalyzer()
        self.grasp_generator = GraspGenerator()
        self.grasp_evaluator = GraspEvaluator()
        
    def plan_grasp(self, target_object, grasp_constraints=None):
        """Plan an optimal grasp for the target object"""
        # Analyze object properties
        object_properties = self.object_analyzer.analyze(target_object)
        
        # Generate candidate grasps
        candidate_grasps = self.grasp_generator.generate_candidates(
            object_properties, grasp_constraints
        )
        
        # Evaluate grasps
        evaluated_grasps = []
        for grasp in candidate_grasps:
            score = self.grasp_evaluator.evaluate(grasp, object_properties)
            evaluated_grasps.append((grasp, score))
        
        # Return best grasp
        best_grasp, best_score = max(evaluated_grasps, key=lambda x: x[1])
        return best_grasp
```

### Physics-Based Grasp Analysis

Physics-based analysis evaluates grasp stability and quality using principles of mechanics:

```cpp
// Physics-based grasp analysis
class PhysicsBasedGraspAnalyzer {
public:
    PhysicsBasedGraspAnalyzer() {
        initializeFrictionModels();
        initializeStabilityMetrics();
    }

    GraspQuality evaluateGrasp(const Grasp& grasp, const Object& object) {
        GraspQuality quality;
        
        // Calculate force closure
        quality.force_closure = checkForceClosure(grasp, object);
        
        // Calculate grasp stability
        quality.stability = calculateStabilityMetric(grasp, object);
        
        // Calculate grasp robustness
        quality.robustness = calculateRobustness(grasp, object);
        
        // Calculate grasp wrench space
        quality.wrench_space = calculateWrenchSpace(grasp, object);
        
        return quality;
    }

private:
    struct GraspQuality {
        bool force_closure;
        double stability;
        double robustness;
        WrenchSpace wrench_space;
    };

    bool checkForceClosure(const Grasp& grasp, const Object& object) {
        // Check if the grasp can resist any arbitrary wrench
        // using the concept of form closure or force closure
        
        std::vector<ContactPoint> contacts = grasp.contact_points;
        
        if (contacts.size() < 2) {
            // Need at least 2 contact points for 2D, 4 for 3D
            return false;
        }
        
        // Form the grasp matrix G
        Eigen::MatrixXd G = formGraspMatrix(contacts);
        
        // Check if the grasp can resist any arbitrary wrench
        // This involves checking the rank of the grasp matrix
        Eigen::FullPivLU<Eigen::MatrixXd> lu(G);
        int rank = lu.rank();
        
        // For 3D objects, need at least 6 linearly independent wrenches
        // For 2D objects, need at least 3 linearly independent wrenches
        int required_rank = object.is_3d ? 6 : 3;
        
        return rank >= required_rank;
    }
    
    double calculateStabilityMetric(const Grasp& grasp, const Object& object) {
        // Calculate a quantitative stability measure
        // This could be based on the volume of the wrench space
        // or the minimum force required to cause slip
        
        // For this example, we'll use a simplified metric based on
        // contact point distribution and friction
        double stability = 0.0;
        
        std::vector<ContactPoint> contacts = grasp.contact_points;
        
        // Calculate how well the contacts distribute around the object
        for (const auto& contact : contacts) {
            // Consider contact location, normal direction, and friction
            stability += calculateContactQuality(contact, object);
        }
        
        // Normalize by number of contacts
        if (!contacts.empty()) {
            stability /= contacts.size();
        }
        
        // Also consider the object's center of mass relative to contacts
        Eigen::Vector3d com = object.center_of_mass;
        double com_stability = calculateCOMStability(com, contacts);
        stability = (stability + com_stability) / 2.0;
        
        return stability;
    }
    
    Eigen::MatrixXd formGraspMatrix(const std::vector<ContactPoint>& contacts) {
        // Form the grasp matrix G where each column represents
        // the wrench that can be applied by a contact
        int num_wrenches = contacts.size() * 2; // Assuming point contacts with friction
        int wrench_dim = 6; // 3 forces + 3 torques for 3D
        
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(wrench_dim, num_wrenches);
        
        for (size_t i = 0; i < contacts.size(); ++i) {
            const ContactPoint& cp = contacts[i];
            
            // Column 1: Normal force
            G(0, 2*i) = cp.normal[0];
            G(1, 2*i) = cp.normal[1];
            G(2, 2*i) = cp.normal[2];
            
            // Calculate moment arm from object center
            Eigen::Vector3d r = cp.position - object_com_; // object center of mass
            
            // Moments from normal force
            G(3, 2*i) = r[1]*cp.normal[2] - r[2]*cp.normal[1]; // mx
            G(4, 2*i) = r[2]*cp.normal[0] - r[0]*cp.normal[2]; // my
            G(5, 2*i) = r[0]*cp.normal[1] - r[1]*cp.normal[0]; // mz
            
            // Column 2: Friction force (tangent direction)
            Eigen::Vector3d tangent = calculateTangent(cp.normal);
            G(0, 2*i+1) = tangent[0];
            G(1, 2*i+1) = tangent[1];
            G(2, 2*i+1) = tangent[2];
            
            // Moments from friction force
            G(3, 2*i+1) = r[1]*tangent[2] - r[2]*tangent[1];
            G(4, 2*i+1) = r[2]*tangent[0] - r[0]*tangent[2];
            G(5, 2*i+1) = r[0]*tangent[1] - r[1]*tangent[0];
        }
        
        return G;
    }
    
    Eigen::Vector3d calculateTangent(const Eigen::Vector3d& normal) {
        // Calculate a tangent vector perpendicular to the normal
        Eigen::Vector3d tangent = Eigen::Vector3d::UnitX();
        if (std::abs(normal.dot(Eigen::Vector3d::UnitX())) > 0.9) {
            tangent = Eigen::Vector3d::UnitY();  // Use Y if normal is X-like
        }
        // Make tangent perpendicular to normal
        tangent = tangent - normal * normal.dot(tangent);
        return tangent.normalized();
    }
    
    double calculateContactQuality(const ContactPoint& contact, const Object& object) {
        // Calculate quality based on contact properties
        double quality = 0.0;
        
        // Prefer contacts that support the object's weight
        Eigen::Vector3d weight_vector = Eigen::Vector3d(0, 0, -object.mass * 9.81);
        if (contact.normal.dot(-weight_vector) > 0) {
            quality += 0.3;  // Good for supporting weight
        }
        
        // Prefer contacts with high friction
        quality += 0.4 * contact.friction_coefficient;
        
        // Prefer contacts not too close to edges
        double edge_distance = calculateDistanceToEdge(contact.position, object);
        quality += 0.3 * std::min(1.0, edge_distance / 0.02); // 2cm threshold
        
        return quality;
    }
    
    double calculateCOMStability(const Eigen::Vector3d& com, 
                                const std::vector<ContactPoint>& contacts) {
        // Calculate stability based on how well contacts support the COM
        double stability = 0.0;
        
        // Find the support polygon projected to the horizontal plane
        std::vector<Eigen::Vector2d> support_points;
        for (const auto& contact : contacts) {
            support_points.push_back(Eigen::Vector2d(contact.position[0], contact.position[1]));
        }
        
        Eigen::Vector2d com_projected = Eigen::Vector2d(com[0], com[1]);
        
        // Calculate distance from COM projection to support polygon
        double distance_to_support = calculateDistanceToPolygon(com_projected, support_points);
        
        // Higher stability if COM is well within support polygon
        if (distance_to_support > 0.01) { // 1cm threshold
            stability = 1.0;
        } else {
            stability = std::max(0.0, distance_to_support / 0.01);
        }
        
        return stability;
    }

    Eigen::Vector3d object_com_;
    
    void initializeFrictionModels();
    void initializeStabilityMetrics();
    
    double calculateRobustness(const Grasp& grasp, const Object& object);
    WrenchSpace calculateWrenchSpace(const Grasp& grasp, const Object& object);
    double calculateDistanceToEdge(const Eigen::Vector3d& point, const Object& object);
    double calculateDistanceToPolygon(const Eigen::Vector2d& point, 
                                    const std::vector<Eigen::Vector2d>& polygon);
};
```

### Learning-Based Grasp Synthesis

Learning approaches can synthesize grasps based on experience and data:

```python
class LearningBasedGraspSynthesizer:
    def __init__(self):
        self.grasp_dataset = GraspDataset()
        self.convolutional_network = ConvolutionalGraspNetwork()
        self.geometric_features = GeometricFeatureExtractor()
        self.sampling_strategy = GraspSamplingStrategy()
        
    def synthesize_grasp(self, object_pcd):
        """Synthesize a grasp for an unknown object using learned models"""
        # Extract geometric features from point cloud
        geometric_features = self.geometric_features.extract(object_pcd)
        
        # Use convolutional network to predict grasp affordances
        grasp_affordance_map = self.convolutional_network.predict_affordance(object_pcd)
        
        # Sample potential grasp locations from high-affordance areas
        candidate_grasps = self.sampling_strategy.sample_from_affordance(
            grasp_affordance_map, n_samples=100
        )
        
        # Evaluate candidates and select best one
        best_grasp = self.select_best_grasp(candidate_grasps, object_pcd)
        
        return best_grasp
    
    def select_best_grasp(self, candidate_grasps, object_pcd):
        """Select the best grasp from candidates using geometric and learned criteria"""
        scores = []
        
        for grasp in candidate_grasps:
            # Combine learned score with geometric validation
            learned_score = self.evaluate_grasp_with_network(grasp, object_pcd)
            geometric_score = self.validate_grasp_geometry(grasp, object_pcd)
            
            combined_score = 0.7 * learned_score + 0.3 * geometric_score
            scores.append((grasp, combined_score))
        
        # Return grasp with highest score
        best_grasp, best_score = max(scores, key=lambda x: x[1])
        return best_grasp
    
    def evaluate_grasp_with_network(self, grasp, object_pcd):
        """Evaluate a grasp using learned network"""
        # Prepare input for the network
        network_input = self.prepare_network_input(grasp, object_pcd)
        
        # Run network to get success probability
        success_prob = self.convolutional_network.predict_success_probability(network_input)
        
        return success_prob
        
    def validate_grasp_geometry(self, grasp, object_pcd):
        """Validate grasp using geometric constraints"""
        # Check if grasp is geometrically feasible
        # - Fingers don't collide with object
        # - Grasp width is appropriate
        # - Contact points are on object surface
        
        geometric_score = 1.0
        
        # Check grasp width constraints
        grasp_width = self.calculate_grasp_width(grasp)
        obj_size = self.calculate_object_size(object_pcd)
        
        if grasp_width < 0.5 * obj_size or grasp_width > 2.0 * obj_size:
            geometric_score *= 0.1  # Penalize inappropriate grasp width
        
        # Check collision between gripper and object
        if self.check_gripper_collision(grasp, object_pcd):
            geometric_score *= 0.0  # Invalid if gripper collides with object
        
        # Check if contact points are on object surface
        contact_points = self.calculate_contact_points(grasp)
        for point in contact_points:
            if not self.is_on_object_surface(point, object_pcd):
                geometric_score *= 0.5  # Penalize if contact point not on surface
        
        return geometric_score
        
    def prepare_network_input(self, grasp, object_pcd):
        """Prepare input for the grasp evaluation network"""
        # Combine grasp parameters with object representation
        grasp_params = np.array([
            grasp.approach_direction,
            grasp.grasp_width,
            grasp.finger_positions
        ])
        
        # Object representation (could be voxel grid, point cloud, image, etc.)
        object_repr = self.encode_object(object_pcd)
        
        return {
            'grasp_params': grasp_params,
            'object_repr': object_repr
        }
        
    def calculate_grasp_width(self, grasp):
        """Calculate the width of the grasp"""
        # Implementation depends on grasp representation
        return grasp.grasp_width

class ConvolutionalGraspNetwork:
    def __init__(self):
        self.model = self.build_model()
        
    def build_model(self):
        """Build convolutional neural network for grasp evaluation"""
        import tensorflow as tf
        from tensorflow.keras import layers
        
        # Input for object (e.g., depth image or voxel grid)
        object_input = layers.Input(shape=(64, 64, 1), name='object_input')
        
        # Convolutional layers to extract object features
        conv1 = layers.Conv2D(32, 3, activation='relu')(object_input)
        conv2 = layers.Conv2D(64, 3, activation='relu')(conv1)
        pool1 = layers.MaxPooling2D(2)(conv2)
        conv3 = layers.Conv2D(64, 3, activation='relu')(pool1)
        conv4 = layers.Conv2D(128, 3, activation='relu')(conv3)
        pool2 = layers.MaxPooling2D(2)(conv4)
        flattened = layers.Flatten()(pool2)
        dense1 = layers.Dense(512, activation='relu')(flattened)
        
        # Input for grasp parameters
        grasp_input = layers.Input(shape=(10,), name='grasp_input')  # Example shape
        grasp_dense = layers.Dense(64, activation='relu')(grasp_input)
        
        # Combine object and grasp features
        combined = layers.Concatenate()([dense1, grasp_dense])
        dense2 = layers.Dense(256, activation='relu')(combined)
        dense3 = layers.Dense(128, activation='relu')(dense2)
        
        # Output: success probability
        output = layers.Dense(1, activation='sigmoid', name='success_probability')(dense3)
        
        model = tf.keras.Model(inputs=[object_input, grasp_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
        
    def predict_affordance(self, object_pcd):
        """Predict grasp affordance map for the object"""
        # Preprocess point cloud to appropriate input format
        input_repr = self.preprocess_point_cloud(object_pcd)
        
        # Predict affordance for each location
        affordance_map = self.model.predict(input_repr)
        
        return affordance_map
        
    def predict_success_probability(self, network_input):
        """Predict success probability for a specific grasp"""
        # Run the network with the provided input
        prediction = self.model.predict(network_input)
        return float(prediction[0][0])  # Return the probability
        
    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud to network input format"""
        # Convert point cloud to voxel grid, depth image, or other format
        # that the network expects
        pass
```

## Hand Design and Actuation

### Anthropomorphic Hand Design

Creating hands that mimic human capabilities while being practical for robotic applications:

```cpp
// Anthropomorphic hand design considerations
class AnthropomorphicHandDesign {
public:
    AnthropomorphicHandDesign() {
        initializeFingerKinematics();
        defineActuationScheme();
        installSensors();
    }

    HandSpecification designHand(const HumanHandSpecifications& target_specs) {
        HandSpecification design;
        
        // Design fingers based on human hand characteristics
        design.thumb = designDigit(THUMB, target_specs.thumb);
        for (int i = 0; i < 4; ++i) {  // Index, middle, ring, little
            DigitType type = static_cast<DigitType>(INDEX_FINGER + i);
            design.fingers[i] = designDigit(type, target_specs.fingers[i]);
        }
        
        // Design palm and hand structure
        design.palm = designPalm(target_specs.palm);
        
        // Integrate actuation and control
        design.actuation = designActuation(target_specs);
        design.control = designControlSystem(target_specs);
        
        return design;
    }

private:
    struct HumanHandSpecifications {
        DigitSpecification thumb;
        DigitSpecification fingers[4];  // Index, middle, ring, little
        PalmSpecification palm;
        int total_dof;
        double max_force;
        dexterity_metrics;
    };

    struct HandSpecification {
        Digit thumb;
        Digit fingers[4];
        Palm palm;
        ActuationSystem actuation;
        ControlSystem control;
    };

    Digit designDigit(DigitType type, const DigitSpecification& target) {
        Digit digit;
        
        // Number of joints based on finger type
        digit.joints = (type == THUMB) ? 4 : 3;  // Thumb has 4 joints, others have 3
        
        // Range of motion for each joint
        for (int i = 0; i < digit.joints; ++i) {
            Joint joint = designJoint(target.joint_specs[i], type, i);
            digit.joint_configurations.push_back(joint);
        }
        
        // Link lengths based on proportions
        digit.links = calculateLinkLengths(target, type);
        
        // Actuation: decide between underactuated or fully actuated
        digit.actuation = designDigitActuation(target, type);
        
        return digit;
    }
    
    Joint designJoint(const JointSpecification& target, DigitType type, int joint_idx) {
        Joint joint;
        
        // Joint type (revolute, prismatic, etc.)
        joint.type = determineJointType(type, joint_idx);
        
        // Range of motion
        joint.min_angle = target.min_angle;
        joint.max_angle = target.max_angle;
        
        // Mechanical design
        joint.gear_ratio = target.gear_ratio;
        joint.max_torque = target.max_torque;
        joint.max_velocity = target.max_velocity;
        
        // Sensing capabilities
        joint.position_sensor = true;
        joint.torque_sensor = (joint_idx == 0);  // Only first joint sometimes has torque sensing
        
        return joint;
    }
    
    std::vector<Link> calculateLinkLengths(const DigitSpecification& target, DigitType type) {
        // Calculate link lengths based on human finger proportions
        // Human fingers have specific length ratios
        std::vector<Link> links;
        
        if (type == THUMB) {
            // Thumb proportions (typically shorter than other fingers)
            links.push_back({target.length * 0.3, target.diameter * 0.8});  // MCP
            links.push_back({target.length * 0.25, target.diameter * 0.8}); // IP
            links.push_back({target.length * 0.25, target.diameter * 0.7}); // tip
            links.push_back({target.length * 0.2, target.diameter * 0.6});  // tip segment
        } else {
            // Other fingers have 3 phalanges: proximal, middle, distal
            links.push_back({target.length * 0.35, target.diameter});   // Proximal
            links.push_back({target.length * 0.30, target.diameter * 0.9}); // Middle
            links.push_back({target.length * 0.25, target.diameter * 0.8}); // Distal
        }
        
        return links;
    }
    
    ActuationScheme designDigitActuation(const DigitSpecification& target, DigitType type) {
        // Choose between different actuation schemes
        ActuationScheme scheme;
        
        if (target.dexterity_requirement > 0.8) {
            // Fully actuated for high dexterity
            scheme.type = FULLY_ACTUATED;
            scheme.motors_per_digit = target.num_joints;
        } else if (target.dexterity_requirement > 0.5) {
            // Underactuated for moderate dexterity
            scheme.type = UNDERACTUATED;
            scheme.motors_per_digit = std::ceil(target.num_joints / 2.0);
        } else {
            // Simplified for basic grasping
            scheme.type = SIMPLIFIED_ACTUATION;
            scheme.motors_per_digit = 1;
        }
        
        // Select actuators based on required force and speed
        scheme.actuators = selectActuators(target);
        
        return scheme;
    }
    
    void installSensors() {
        // Install various sensors needed for dexterous manipulation
        tactile_sensors_ = TactileSensorArray();
        force_torque_sensors_ = ForceTorqueSensors();
        position_encoders_ = JointPositionEncoders();
        imu_ = IMU();
    }
    
    void initializeFingerKinematics();
    void defineActuationScheme();
    std::vector<Actuator> selectActuators(const DigitSpecification& target);
    
    TactileSensorArray tactile_sensors_;
    ForceTorqueSensors force_torque_sensors_;
    JointPositionEncoders position_encoders_;
    IMU imu_;
};
```

### Underactuated vs. Fully Actuated Hands

Different actuation strategies offer trade-offs between complexity, cost, and dexterity:

```python
# Comparison of underactuated vs fully actuated hand strategies
ACTUATION_STRATEGIES = {
    'underactuated': {
        'definition': 'Fewer actuators than degrees of freedom',
        'mechanism': 'Coupled joints, spring actuation, tendon routing',
        'advantages': [
            'Lower cost',
            'Lighter weight', 
            'Self-adaptive grasping',
            'Simpler control'
        ],
        'disadvantages': [
            'Limited independent control',
            'Less dexterity',
            'Harder kinematic modeling'
        ],
        'best_for': ['Basic grasping tasks', 'Cost-sensitive applications'],
        'example_designs': ['LARM, Adaptive hand', 'Robotiq 2F-85']
    },
    'fully_actuated': {
        'definition': 'One actuator per degree of freedom',
        'mechanism': 'Direct drive, gear reduction for each joint',
        'advantages': [
            'Full independent control',
            'Maximum dexterity',
            'Precise control'
        ],
        'disadvantages': [
            'Higher cost',
            'Heavier',
            'More complex control',
            'More failure points'
        ],
        'best_for': ['Dexterous manipulation', 'Research applications'],
        'example_designs': ['i-HY hand', 'DLR hand', 'Shadow hand']
    },
    'simplified': {
        'definition': 'Minimal actuation for basic functions',
        'mechanism': 'Single or few actuators for basic grips',
        'advantages': [
            'Lowest cost',
            'Simplest control',
            'Most reliable'
        ],
        'disadvantages': [
            'Very limited dexterity',
            'Few grasp types',
            'Less adaptive'
        ],
        'best_for': ['Simple pick-and-place', 'Industrial applications'],
        'example_designs': ['Parallel jaw grippers', 'Basic 2-finger grippers']
    }
}

class HandActuationDesigner:
    def __init__(self):
        self.mechanical_designer = MechanicalDesigner()
        self.actuator_selector = ActuatorSelector()
        self.control_designer = ControlDesigner()
        
    def design_hand_actuation(self, requirements):
        """Design hand actuation based on requirements"""
        # Determine appropriate actuation strategy
        strategy = self.select_actuation_strategy(requirements)
        
        if strategy == 'underactuated':
            return self.design_underactuated_hand(requirements)
        elif strategy == 'fully_actuated':
            return self.design_fully_actuated_hand(requirements)
        elif strategy == 'simplified':
            return self.design_simplified_hand(requirements)
        else:
            raise ValueError(f"Unknown actuation strategy: {strategy}")
    
    def select_actuation_strategy(self, requirements):
        """Select appropriate actuation strategy based on requirements"""
        # Weight different factors
        cost_weight = 0.3
        dexterity_weight = 0.4
        reliability_weight = 0.2
        weight_requirement = 0.1
        
        # Calculate score for each strategy
        scores = {}
        
        # Underactuated score
        scores['underactuated'] = (
            (1.0 - requirements.get('cost', 0.5)) * cost_weight +
            requirements.get('dexterity', 0.5) * 0.7 * dexterity_weight +
            requirements.get('reliability', 0.7) * reliability_weight +
            (1.0 - requirements.get('weight', 0.3)) * weight_requirement
        )
        
        # Fully actuated score
        scores['fully_actuated'] = (
            (1.0 - requirements.get('cost', 0.8)) * cost_weight * 0.5 +  # Higher cost penalty
            requirements.get('dexterity', 0.9) * dexterity_weight +
            requirements.get('reliability', 0.5) * 0.8 * reliability_weight +  # More failure points
            (1.0 - requirements.get('weight', 0.8)) * 0.5 * weight_requirement  # Heavier
        )
        
        # Simplified score
        scores['simplified'] = (
            (1.0 - requirements.get('cost', 0.2)) * 1.2 * cost_weight +  # Lowest cost benefit
            requirements.get('dexterity', 0.2) * 0.3 * dexterity_weight +  # Lowest dexterity
            requirements.get('reliability', 0.9) * reliability_weight +  # Highest reliability
            (1.0 - requirements.get('weight', 0.2)) * 1.2 * weight_requirement  # Lightest
        )
        
        # Return strategy with highest score
        return max(scores, key=scores.get)
    
    def design_underactuated_hand(self, requirements):
        """Design an underactuated hand"""
        hand_design = {
            'type': 'underactuated',
            'actuation_scheme': 'tendon-driven with coupling',
            'dof': requirements.get('total_dof', 12),
            'actuators': requirements.get('total_dof', 12) * 0.5,  # Fewer actuators than DOF
            'mechanism': self.design_coupledd_joints(requirements),
            'control': self.design_position_based_control(requirements),
            'tactile_sensing': requirements.get('tactile_sensing', True)
        }
        
        # Add underactuation mechanisms
        hand_design['coupling_mechanism'] = self.design_coupling_mechanism()
        hand_design['spring_elements'] = self.design_spring_elements()
        
        return hand_design
    
    def design_fully_actuated_hand(self, requirements):
        """Design a fully actuated hand"""
        hand_design = {
            'type': 'fully_actuated',
            'actuation_scheme': 'direct drive with individual motors',
            'dof': requirements.get('total_dof', 16),
            'actuators': requirements.get('total_dof', 16),  # Same number of actuators as DOF
            'mechanism': self.design_direct_drive_joints(requirements),
            'control': self.design_torque_based_control(requirements),
            'tactile_sensing': requirements.get('tactile_sensing', True)
        }
        
        # Add individual actuation for each joint
        hand_design['actuators'] = self.design_individual_actuators(requirements)
        
        return hand_design
    
    def design_coupling_mechanism(self):
        """Design coupling mechanism for underactuated hand"""
        return {
            'type': 'tendon_coupling',
            'coupling_ratio': [1.0, 1.0, 0.8],  # MCP, PIP, DIP coupling
            'adjustability': 'passive adaptability',
            'self_adaptive': True
        }
    
    def design_individual_actuators(self, requirements):
        """Design individual actuators for fully actuated hand"""
        actuators = []
        
        # Each joint gets its own actuator
        for i in range(requirements.get('total_dof', 16)):
            actuator = self.actuator_selector.select_by_requirements({
                'torque': requirements.get('joint_torque', 1.0),
                'speed': requirements.get('joint_speed', 1.0),
                'precision': requirements.get('control_precision', 0.01),
                'size': requirements.get('size_constraint', 'medium')
            })
            actuators.append(actuator)
            
        return actuators
        
    def design_direct_drive_joints(self, requirements):
        """Design direct drive joints for fully actuated hand"""
        joints = []
        
        for i in range(requirements.get('total_dof', 16)):
            joint = {
                'joint_type': 'revolute',
                'actuator_type': 'direct_drive_brushless',
                'gear_ratio': 1.0,  # Direct drive
                'range_of_motion': requirements.get('joint_rom', 90.0),
                'max_torque': requirements.get('joint_torque', 1.0),
                'max_speed': requirements.get('joint_speed', 120.0),
                'position_resolution': 0.01  # High precision
            }
            joints.append(joint)
            
        return joints
```

## Force and Tactile Control

### Impedance Control for Manipulation

Impedance control enables robots to interact safely and effectively with their environment:

```cpp
// Impedance control for robotic manipulation
class ImpedanceController {
public:
    ImpedanceController() {
        // Initialize with default parameters for manipulation
        setDefaultImpedanceParameters();
    }

    void setDesiredImpedance(const Matrix3& mass, 
                            const Matrix3& damping, 
                            const Matrix3& stiffness) {
        M_desired_ = mass;
        D_desired_ = damping; 
        K_desired_ = stiffness;
    }

    Wrench calculateImpedanceForce(const Pose& current_pose,
                                  const Pose& desired_pose,
                                  const Twist& current_twist,
                                  const Twist& desired_twist) {
        // Calculate position and velocity errors
        Vector6 pose_error = calculatePoseError(current_pose, desired_pose);
        Vector6 twist_error = desired_twist - current_twist;
        
        // Implementation of impedance control law:
        // F = M*(xddot_d - xddot) + D*(xdot_d - xdot) + K*(x_d - x)
        Vector6 acceleration_error = calculateAccelerationError(
            desired_twist, current_twist, desired_pose, current_pose
        );
        
        Wrench impedance_force;
        impedance_force.force = (M_desired_ * acceleration_error.segment(0, 3) + 
                                D_desired_ * twist_error.segment(0, 3) + 
                                K_desired_ * pose_error.segment(0, 3));
        impedance_force.torque = (M_desired_ * acceleration_error.segment(3, 3) + 
                                 D_desired_ * twist_error.segment(3, 3) + 
                                 K_desired_ * pose_error.segment(3, 3));
        
        return impedance_force;
    }

    void adaptImpedanceForContact(const ContactState& contact_state) {
        // Adapt impedance based on contact state
        if (contact_state.status == ContactStatus::NO_CONTACT) {
            // Free space - may want lower stiffness for safety
            setFreeSpaceImpedance();
        } else if (contact_state.status == ContactStatus::PARTIAL_CONTACT) {
            // Partial contact - adjust for stability
            setPartialContactImpedance();
        } else if (contact_state.status == ContactStatus::FULL_CONTACT) {
            // Full contact - adjust for manipulation task
            setFullContactImpedance(contact_state.surface_properties);
        }
    }

private:
    Matrix3 M_desired_;  // Desired mass matrix
    Matrix3 D_desired_;  // Desired damping matrix
    Matrix3 K_desired_;  // Desired stiffness matrix

    void setDefaultImpedanceParameters() {
        // Set default parameters for manipulation tasks
        M_desired_ = Matrix3::Identity() * 1.0;  // 1 kg equivalent mass
        D_desired_ = Matrix3::Identity() * 10.0; // Damping ratio ~1.0
        K_desired_ = Matrix3::Identity() * 1000.0; // Stiffness for precise control
    }

    void setFreeSpaceImpedance() {
        // In free space, we might want lower stiffness for safety
        K_desired_ = Matrix3::Identity() * 500.0;
        D_desired_ = Matrix3::Identity() * 5.0;
    }

    void setPartialContactImpedance() {
        // When partially in contact, moderate impedance for stability
        K_desired_ = Matrix3::Identity() * 800.0;
        D_desired_ = Matrix3::Identity() * 8.0;
    }

    void setFullContactImpedance(const SurfaceProperties& surface) {
        // When fully in contact, adapt to surface properties
        double surface_stiffness_factor = surface.stiffness / 1e6; // Normalize
        
        K_desired_ = Matrix3::Identity() * 1000.0 * surface_stiffness_factor;
        D_desired_ = Matrix3::Identity() * 10.0 * std::sqrt(surface_stiffness_factor);
    }

    Vector6 calculatePoseError(const Pose& current, const Pose& desired) {
        // Calculate pose error in task space
        Vector6 error;
        
        // Position error
        error.segment(0, 3) = desired.position - current.position;
        
        // Orientation error using rotation vector
        Matrix3 rotation_error = desired.orientation * current.orientation.transpose();
        error.segment(3, 3) = rotationMatrixToVector(rotation_error);
        
        return error;
    }

    Vector6 calculateAccelerationError(const Twist& desired_twist,
                                     const Twist& current_twist,
                                     const Pose& desired_pose,
                                     const Pose& current_pose) {
        // Numerical differentiation to get acceleration error estimate
        // In practice, this would use a more sophisticated estimator
        static Twist prev_twist = current_twist;
        static ros::Time prev_time = ros::Time::now();
        
        ros::Time curr_time = ros::Time::now();
        double dt = (curr_time - prev_time).toSec();
        
        if (dt > 0.001) {  // Avoid division by zero
            Vector6 current_accel = (current_twist - prev_twist) / dt;
            Vector6 desired_accel = (desired_twist - prev_twist) / dt; // Simplified
            
            prev_twist = current_twist;
            prev_time = curr_time;
            
            return desired_accel - current_accel;
        }
        
        return Vector6::Zero(); // Return zero if dt too small
    }

    Vector3 rotationMatrixToVector(const Matrix3& R) {
        // Convert rotation matrix to rotation vector (axis-angle representation)
        double trace = R(0,0) + R(1,1) + R(2,2);
        double angle = acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
        
        if (angle < 1e-6) {
            // Small angle approximation
            Vector3 rot_vec;
            rot_vec(0) = (R(2,1) - R(1,2)) / 2.0;
            rot_vec(1) = (R(0,2) - R(2,0)) / 2.0;
            rot_vec(2) = (R(1,0) - R(0,1)) / 2.0;
            return rot_vec;
        } else {
            Vector3 rot_vec;
            rot_vec(0) = (R(2,1) - R(1,2));
            rot_vec(1) = (R(0,2) - R(2,0));
            rot_vec(2) = (R(1,0) - R(0,1));
            rot_vec = rot_vec * angle / (2.0 * sin(angle));
            return rot_vec;
        }
    }
};
```

### Tactile Sensing and Feedback

Tactile sensing provides crucial feedback for dexterous manipulation:

```python
class TactileFeedbackSystem:
    def __init__(self):
        self.tactile_sensors = self.initialize_tactile_sensors()
        self.tactile_processing = TactileSignalProcessor()
        self.slam = SimultaneousLocalizationAndMapping()
        
    def initialize_tactile_sensors(self):
        """Initialize tactile sensors across the robotic hand"""
        sensors = {}
        
        # Add tactile sensors to fingertips
        for digit_idx in range(5):  # thumb + 4 fingers
            sensors[f'finger_{digit_idx}_tip'] = {
                'type': 'biotac',
                'resolution': 'high',
                'coverage': 'center_of_fingerpad'
            }
            
        # Add tactile sensors to finger pads if available
        for digit_idx in range(5):
            for joint_idx in range(3):  # PIP, DIP joints for fingers; MCP, IP for thumb
                sensors[f'finger_{digit_idx}_pad_{joint_idx}'] = {
                    'type': 'gelsight',
                    'resolution': 'medium',
                    'coverage': 'finger_pad'
                }
                
        return sensors
    
    def process_tactile_data(self, raw_tactile_signals):
        """Process raw tactile sensor data into meaningful information"""
        processed_data = {}
        
        for sensor_id, signal in raw_tactile_signals.items():
            # Process individual sensor data
            sensor_type = self.sensors[sensor_id]['type']
            
            if sensor_type == 'biotac':
                processed_data[sensor_id] = self.process_biotac_signal(signal)
            elif sensor_type == 'gelsight':
                processed_data[sensor_id] = self.process_gelsight_signal(signal)
            else:
                processed_data[sensor_id] = self.process_generic_tactile_signal(signal)
        
        # Integrate multi-sensor data
        integrated_feedback = self.integrate_sensor_data(processed_data)
        
        return integrated_feedback
        
    def process_biotac_signal(self, signal):
        """Process Biotac tactile sensor signal"""
        # Biotac sensors provide multiple modalities:
        # - DC (Direct Current) - contact detection
        # - AC (Alternating Current) - texture and slip
        # - Taxel readings - spatial tactile information
        
        dc_value = signal['dc']
        ac_values = signal['ac']
        taxel_readings = signal['taxels']
        
        feedback = {
            'contact_detected': dc_value > self.biotac_thresholds['contact'],
            'slip_detected': self.detect_slip_from_ac(ac_values),
            'contact_location': self.estimate_contact_location(taxel_readings),
            'contact_force': self.estimate_contact_force(dc_value),
            'texture_estimate': self.estimate_texture(ac_values)
        }
        
        return feedback
        
    def process_gelsight_signal(self, signal):
        """Process GelSight tactile sensor signal"""
        # GelSight provides high-resolution images of the contact surface
        image = signal['image']
        
        # Extract contact features from the image
        contact_area = self.calculate_contact_area(image)
        contact_centroid = self.calculate_contact_centroid(image)
        surface_normals = self.estimate_surface_normals(image)
        contact_geometry = self.estimate_contact_geometry(image)
        
        feedback = {
            'contact_detected': contact_area > 0,
            'contact_area': contact_area,
            'contact_centroid': contact_centroid,
            'surface_normals': surface_normals,
            'contact_geometry': contact_geometry,
            'high_resolution_shape': self.reconstruct_shape_from_image(image)
        }
        
        return feedback
        
    def integrate_sensor_data(self, processed_data):
        """Integrate data from multiple tactile sensors"""
        # Create a unified tactile perception of the grasp
        unified_feedback = {
            'contact_map': self.create_contact_map(processed_data),
            'object_slip_state': self.assess_object_slip(processed_data),
            'grasp_stability': self.assess_grasp_stability(processed_data),
            'object_properties': self.estimate_object_properties(processed_data),
            'manipulation_plan_updates': self.assess_manipulation_feasibility(processed_data)
        }
        
        return unified_feedback
        
    def assess_grasp_stability(self, tactile_data):
        """Assess grasp stability based on tactile feedback"""
        # Factors affecting grasp stability:
        # - Number of contact points
        # - Distribution of contact forces
        # - Detection of slip events
        # - Object properties estimation
        
        contact_points = []
        contact_forces = []
        slip_events = []
        
        for sensor_id, data in tactile_data.items():
            if data.get('contact_detected', False):
                contact_points.append(data.get('contact_location'))
                contact_forces.append(data.get('contact_force'))
                
            if data.get('slip_detected', False):
                slip_events.append({
                    'sensor': sensor_id,
                    'time': data.get('timestamp'),
                    'severity': data.get('slip_severity', 1.0)
                })
        
        # Calculate grasp stability metrics
        num_contacts = len(contact_points)
        avg_contact_force = np.mean(contact_forces) if contact_forces else 0
        slip_frequency = len(slip_events) / (self.time_window if hasattr(self, 'time_window') else 1.0)
        
        # Combine metrics into stability score
        stability_score = (
            min(1.0, num_contacts / 4.0) * 0.4 +  # At least 4 contacts preferred
            min(1.0, avg_contact_force / 5.0) * 0.3 +  # Reasonable force level
            max(0.0, 1.0 - slip_frequency) * 0.3  # Less slip is better
        )
        
        return {
            'stability_score': stability_score,
            'number_of_contacts': num_contacts,
            'average_contact_force': avg_contact_force,
            'slip_frequency': slip_frequency,
            'recommendation': self.get_stability_recommendation(stability_score)
        }
    
    def get_stability_recommendation(self, stability_score):
        """Get action recommendation based on grasp stability"""
        if stability_score > 0.8:
            return "maintain_current_grasp"
        elif stability_score > 0.5:
            return "consider_adjusting_grasp"
        else:
            return "regrasp_needed"

class TactileControlSystem:
    def __init__(self):
        self.impedance_controller = ImpedanceController()
        self.force_controller = ForceController()
        self.tactile_feedback_system = TactileFeedbackSystem()
        
    def execute_compliant_manipulation(self, desired_trajectory, tactile_feedback):
        """Execute manipulation with compliance based on tactile feedback"""
        # Adjust control parameters based on tactile feedback
        adjusted_parameters = self.adapt_control_to_tactile_feedback(tactile_feedback)
        
        # Apply adaptive control law
        control_output = self.apply_adaptive_control(
            desired_trajectory, adjusted_parameters
        )
        
        return control_output
    
    def adapt_control_to_tactile_feedback(self, tactile_feedback):
        """Adapt control parameters based on tactile feedback"""
        adaptation = {}
        
        # If slip is detected, increase grasp force
        if tactile_feedback.get('slip_detected', False):
            adaptation['grasp_force_increase'] = 0.5  # Newtons
            adaptation['compliance_decrease'] = 0.2   # Make grasp more rigid
            
        # If grasp is unstable, adjust impedance
        stability = tactile_feedback.get('grasp_stability', {})
        stability_score = stability.get('stability_score', 1.0)
        
        if stability_score < 0.5:
            adaptation['impedance_increase'] = 0.3  # Increase stiffness
            adaptation['exploration_movement'] = True  # Try to improve grasp
            
        # If object properties are uncertain, use conservative parameters
        object_uncertainty = tactile_feedback.get('object_uncertainty', 0.0)
        if object_uncertainty > 0.7:
            adaptation['conservative_control'] = True
            adaptation['reduced_speed'] = 0.5  # Reduce movement speed by 50%
            
        return adaptation
    
    def apply_adaptive_control(self, desired_trajectory, adaptations):
        """Apply control with adaptive parameters"""
        # Get current state
        current_state = self.get_current_state()
        
        # Apply adaptations to control parameters
        if 'grasp_force_increase' in adaptations:
            self.increase_grasp_force(adaptations['grasp_force_increase'])
            
        if 'compliance_decrease' in adaptations:
            self.reduce_compliance(adaptations['compliance_decrease'])
            
        if 'impedance_increase' in adaptations:
            self.increase_impedance(adaptations['impedance_increase'])
            
        # Execute trajectory following with adapted parameters
        return self.follow_trajectory_adaptively(desired_trajectory)
```

## Grasp Stability and Compliance

### Stability Analysis Methods

Analyzing grasp stability is crucial for maintaining object hold during manipulation:

```python
# Methods for grasp stability analysis
STABILITY_ANALYSIS_METHODS = {
    'force_closure': {
        'description': 'Ability to resist any arbitrary wrench',
        'mathematical_basis': 'Convex hull of wrenches that contacts can apply',
        'computation_complexity': 'Medium',
        'applicability': 'Quasi-static grasps, rigid objects'
    },
    'form_closure': {
        'description': 'Geometric constraint without friction',
        'mathematical_basis': 'Complete geometric constraint of object DOF',
        'computation_complexity': 'Low',
        'applicability': 'Precision grasps, fixtures'
    },
    'wrench_space_analysis': {
        'description': 'Volume of wrenches the grasp can resist',
        'mathematical_basis': 'Volume of the grasp wrench space',
        'computation_complexity': 'High',
        'applicability': 'Quantitative stability comparison'
    },
    'l1_stability': {
        'description': 'Distance to grasp failure under perturbation',
        'mathematical_basis': 'Distance in wrench space to failure',
        'computation_complexity': 'Medium',
        'applicability': 'Robust grasp evaluation'
    }
}

class GraspStabilityAnalyzer:
    def __init__(self):
        self.wrench_calculator = WrenchCalculator()
        self.convex_hull_computer = ConvexHullComputer()
        
    def analyze_stability(self, grasp, object_properties, environment_context):
        """Comprehensive stability analysis of a grasp"""
        analysis_results = {}
        
        # Force closure analysis
        analysis_results['force_closure'] = self.analyze_force_closure(grasp, object_properties)
        
        # Form closure analysis
        analysis_results['form_closure'] = self.analyze_form_closure(grasp, object_properties)
        
        # Wrench space analysis
        analysis_results['wrench_space'] = self.analyze_wrench_space(grasp, object_properties)
        
        # L1 stability metric
        analysis_results['l1_stability'] = self.calculate_l1_stability(grasp, object_properties)
        
        # Dynamic stability considering object motion
        analysis_results['dynamic_stability'] = self.analyze_dynamic_stability(
            grasp, object_properties, environment_context
        )
        
        # Overall stability score
        analysis_results['overall_stability'] = self.calculate_overall_stability_score(
            analysis_results
        )
        
        return analysis_results
    
    def analyze_force_closure(self, grasp, object_properties):
        """Analyze if grasp has force closure properties"""
        # Get contact points and normals
        contact_points = grasp.contact_points
        normals = [contact.normal for contact in contact_points]
        positions = [contact.position for contact in contact_points]
        
        # Construct grasp matrix G
        # Each contact contributes 3 wrenches in 2D (1 normal + 2 friction)
        # or 6 wrenches in 3D (1 normal + 5 friction)
        G = self.construct_grasp_matrix(contact_points, object_properties)
        
        # Check if the grasp can resist any arbitrary wrench
        # This means checking if the positive span of G contains a ball around origin
        has_force_closure = self.check_positive_span_contains_origin(G)
        
        # Calculate force closure margin
        force_closure_margin = self.calculate_force_closure_margin(G)
        
        return {
            'has_force_closure': has_force_closure,
            'force_closure_margin': force_closure_margin,
            'grasp_matrix_condition': np.linalg.cond(G) if G.shape[0] == G.shape[1] else None
        }
    
    def construct_grasp_matrix(self, contact_points, object_properties):
        """Construct the grasp matrix from contact points"""
        # For 3D objects, each frictional contact can apply wrenches in a cone
        # We approximate this with multiple ray representations
        num_cone_rays = 4  # Number of rays to approximate friction cone
        
        # Calculate total possible wrenches
        total_wrenches = len(contact_points) * num_cone_rays
        
        # Grasp matrix: rows = object DOF (6 for 3D), cols = possible wrenches
        G = np.zeros((6, total_wrenches))
        
        for i, contact in enumerate(contact_points):
            # Calculate moment arms
            r = contact.position - object_properties['center_of_mass']
            
            # For each contact, add wrenches in friction cone approximation
            normal = np.array(contact.normal)
            tangent1, tangent2 = self.calculate_tangent_vectors(normal)
            
            for j in range(num_cone_rays):
                # Calculate wrench direction based on friction cone approximation
                if j == 0:
                    wrench_direction = normal  # Normal force
                elif j == 1:
                    wrench_direction = tangent1  # Friction in first direction
                elif j == 2:
                    wrench_direction = -tangent1  # Friction in opposite first direction
                else:
                    wrench_direction = tangent2  # Friction in second direction
                
                # Add force component to grasp matrix
                G[0:3, i*num_cone_rays + j] = wrench_direction
                
                # Add moment component to grasp matrix
                moment = np.cross(r, wrench_direction)
                G[3:6, i*num_cone_rays + j] = moment
        
        return G
    
    def calculate_tangent_vectors(self, normal):
        """Calculate two orthogonal tangent vectors to the normal"""
        # Create arbitrary vector not parallel to normal
        if abs(normal[2]) < 0.9:
            arbitrary = np.array([0, 0, 1])
        else:
            arbitrary = np.array([1, 0, 0])
            
        # Calculate first tangent using cross product
        tangent1 = np.cross(normal, arbitrary)
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        
        # Calculate second tangent perpendicular to both
        tangent2 = np.cross(normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)
        
        return tangent1, tangent2
    
    def check_positive_span_contains_origin(self, G):
        """Check if positive span of G columns contains the origin"""
        # This is equivalent to checking if the origin is in the convex hull
        # of the columns of G (when augmented with a slack variable)
        
        try:
            # Use linear programming to check if 0 is in the positive span
            n_cols = G.shape[1]
            
            # Minimize sum of coefficients (c^T * x) where c = [1, 1, ..., 1]
            c = np.ones(n_cols)
            
            # Subject to G * x = 0 (origin) and x >= 0 (positive span)
            A_eq = G
            b_eq = np.zeros(G.shape[0])
            
            # Bounds for variables (all must be non-negative)
            bounds = [(0, None) for _ in range(n_cols)]
            
            # Solve LP
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, 
                           method='highs', options={'presolve': True})
            
            # If solution exists with all positive coefficients, origin is in positive span
            if result.success and all(x > -1e-10 for x in result.x):
                return True
            else:
                return False
        except:
            return False  # If optimization fails, assume no force closure
    
    def calculate_force_closure_margin(self, G):
        """Calculate the force closure margin (distance to rank deficiency)"""
        # Calculate SVD of grasp matrix
        U, s, Vt = np.linalg.svd(G)
        
        # Smallest singular value indicates margin to rank deficiency
        min_singular_value = np.min(s)
        
        # Normalize by largest singular value
        condition_number = s[0] / min_singular_value if min_singular_value > 0 else np.inf
        
        return min_singular_value
    
    def analyze_wrench_space(self, grasp, object_properties):
        """Analyze the wrench space of the grasp"""
        # Calculate the set of wrenches that the grasp can resist
        contact_points = grasp.contact_points
        
        # Sample the wrench space by testing if grasp can resist various wrenches
        wrench_space_volume = self.approximate_wrench_space_volume(
            contact_points, object_properties
        )
        
        # Calculate wrench space shape metrics
        wrench_space_metrics = self.calculate_wrench_space_metrics(
            contact_points, object_properties
        )
        
        return {
            'volume': wrench_space_volume,
            'metrics': wrench_space_metrics,
            'resistible_wrenches': self.get_resistible_wrenches(contact_points)
        }
    
    def approximate_wrench_space_volume(self, contact_points, object_properties):
        """Approximate the volume of the grasp wrench space"""
        # This is a complex calculation that typically involves 
        # computing the volume of the intersection of friction cones
        
        # For simplicity, we'll calculate a proxy based on contact distribution
        # and friction coefficients
        total_friction_area = 0
        for contact in contact_points:
            # Approximate friction area contribution
            friction_area = np.pi * (contact.friction_coefficient ** 2)
            total_friction_area += friction_area
        
        # Scale by contact distribution
        contact_distribution = self.calculate_contact_distribution(contact_points)
        
        # Proxy volume score
        volume_score = total_friction_area * contact_distribution
        
        return volume_score
    
    def calculate_contact_distribution(self, contact_points):
        """Calculate how well contacts distribute around the object"""
        if len(contact_points) < 2:
            return 0.0
            
        # Calculate average distance between contact points
        total_distance = 0
        for i in range(len(contact_points)):
            for j in range(i+1, len(contact_points)):
                pos1 = np.array(contact_points[i].position)
                pos2 = np.array(contact_points[j].position)
                distance = np.linalg.norm(pos1 - pos2)
                total_distance += distance
        
        avg_distance = total_distance / (len(contact_points) * (len(contact_points) - 1) / 2)
        
        return avg_distance

class ComplianceControlSystem:
    def __init__(self):
        self.stability_analyzer = GraspStabilityAnalyzer()
        self.impedance_controller = ImpedanceController()
        self.admittance_controller = AdmittanceController()
        
    def adjust_compliance_based_on_stability(self, grasp, object_properties, environment):
        """Adjust robot compliance based on grasp stability analysis"""
        # Analyze current grasp stability
        stability_analysis = self.stability_analyzer.analyze_stability(
            grasp, object_properties, environment
        )
        
        # Determine appropriate compliance based on stability
        required_compliance = self.determine_compliance_from_stability(
            stability_analysis
        )
        
        # Apply compliance control
        self.set_compliance_parameters(required_compliance)
        
        # Return stability-informed control parameters
        return {
            'stability_analysis': stability_analysis,
            'compliance_settings': required_compliance,
            'control_mode': self.select_control_mode(stability_analysis)
        }
    
    def determine_compliance_from_stability(self, stability_analysis):
        """Determine compliance parameters from stability analysis"""
        stability_score = stability_analysis['overall_stability']
        
        # Map stability to compliance parameters
        if stability_score > 0.8:
            # Very stable grasp - can be more compliant for safe interaction
            return {
                'stiffness': 500,   # Lower stiffness
                'damping': 20,      # Lower damping
                'safety_margin': 1.2
            }
        elif stability_score > 0.5:
            # Moderately stable - balanced compliance
            return {
                'stiffness': 1000,  # Medium stiffness
                'damping': 30,      # Medium damping
                'safety_margin': 1.5
            }
        else:
            # Unstable grasp - be more rigid to maintain grasp
            return {
                'stiffness': 2000,  # Higher stiffness
                'damping': 50,      # Higher damping
                'safety_margin': 2.0
            }
    
    def select_control_mode(self, stability_analysis):
        """Select appropriate control mode based on grasp stability"""
        stability_score = stability_analysis['overall_stability']
        
        if stability_score > 0.7:
            return "compliant_force_control"  # Can use compliant control
        elif stability_score > 0.4:
            return "hybrid_position_force_control"  # Need to mix control types
        else:
            return "stiff_position_control"  # Need precise position control to maintain grasp
```

## Learning-Based Grasping

### Reinforcement Learning for Grasping

Using reinforcement learning to improve grasp success rates through experience:

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class GraspingDQN(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(GraspingDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Convolutional layers for processing object representation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out_size(state_size)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
    def _get_conv_out_size(self, shape):
        """Calculate the output size after convolution layers"""
        # Assuming shape is (channels, height, width) for image-like input
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, state):
        """Forward pass through the network"""
        conv_out = self.conv_layers(state).view(state.size(0), -1)
        return self.fc_layers(conv_out)

class GraspReinforcementLearner:
    def __init__(self, state_size, action_size, learning_rate=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(42)
        
        # Q-Networks
        self.qnetwork_local = GraspingDQN(state_size, action_size)
        self.qnetwork_target = GraspingDQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.update_every = 4
        
        # Initialize time step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Take a step in the environment and learn from it"""
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0:
            # Learn if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                experiences = self.sample_from_memory(self.batch_size)
                self.learn(experiences)
    
    def sample_from_memory(self, batch_size):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.array([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.array([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.array([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.array([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.array([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def act(self, state, eps=0.01):
        """Select an action based on the current state"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Set network to evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        # Set network back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (0.99 * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class GraspLearningEnvironment:
    def __init__(self):
        self.robot = RobotInterface()
        self.object_sampler = ObjectSampler()
        self.grasp_evaluator = GraspEvaluator()
        self.visualizer = GraspVisualizer()
        
    def reset(self):
        """Reset the environment to a new grasp scenario"""
        # Sample a new object
        self.current_object = self.object_sampler.sample_object()
        
        # Place object in the workspace
        self.robot.place_object(self.current_object)
        
        # Initialize state representation
        self.state = self.get_state_representation(self.current_object)
        
        return self.state
    
    def get_state_representation(self, obj):
        """Get state representation for the reinforcement learning agent"""
        # State could be:
        # - Depth image of the scene
        # - Point cloud of the object
        # - Semantic features of the object
        # - Robot hand configuration
        
        # For this example, we'll use a depth image
        depth_image = self.robot.get_depth_image()
        
        # Preprocess the image for the neural network
        processed_state = self.preprocess_image(depth_image)
        
        return processed_state
    
    def step(self, action):
        """Execute a grasp action and return environment feedback"""
        # Convert action to grasp parameters
        grasp_params = self.decode_action(action)
        
        # Execute the grasp
        grasp_result = self.robot.execute_grasp(grasp_params)
        
        # Evaluate success
        reward = self.calculate_reward(grasp_result)
        
        # Check if episode is done
        done = self.is_episode_done(grasp_result)
        
        # Update state
        next_state = self.get_state_representation(self.current_object)
        
        return next_state, reward, done, grasp_result
    
    def decode_action(self, action):
        """Decode action index to grasp parameters"""
        # This mapping depends on how actions are defined
        # For example, action could represent:
        # - Discrete grasp type and position
        # - Continuous grasp pose parameters
        # - High-level grasp strategy
        
        # Simple example: discrete grid of grasp positions with orientations
        grid_size = 32  # 32x32 grid over the object
        orientations = 8  # 8 different orientations
        
        if action < grid_size * grid_size:
            # Position-only grasps with default orientation
            x = (action % grid_size) / grid_size
            y = (action // grid_size) / grid_size
            return {'position': (x, y), 'orientation': 0, 'width': 0.05}
        else:
            # Grasps with both position and orientation
            action_remaining = action - (grid_size * grid_size)
            pos_idx = action_remaining // orientations
            orientation_idx = action_remaining % orientations
            
            x = (pos_idx % grid_size) / grid_size
            y = (pos_idx // grid_size) / grid_size
            orientation = (2 * np.pi * orientation_idx) / orientations
            
            return {'position': (x, y), 'orientation': orientation, 'width': 0.05}
    
    def calculate_reward(self, grasp_result):
        """Calculate reward based on grasp outcome"""
        # Define rewards for different outcomes
        if grasp_result.success:
            if grasp_result.stable:
                return 10.0  # Large reward for successful stable grasp
            else:
                return 5.0   # Medium reward for successful but unstable grasp
        else:
            return -1.0      # Small penalty for failed grasp
    
    def is_episode_done(self, grasp_result):
        """Determine if the episode is finished"""
        # For grasping, each action is a complete episode
        return True
    
    def preprocess_image(self, image):
        """Preprocess the image for neural network input"""
        # Resize image if needed
        import cv2
        resized = cv2.resize(image, (64, 64))
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add channel dimension (64, 64) -> (1, 64, 64) for CNN
        return np.expand_dims(normalized, axis=0)

class LearningBasedGraspPlanner:
    def __init__(self, rl_agent):
        self.rl_agent = rl_agent
        self.exploration_rate = 0.1
        self.use_learning = True
        
    def plan_grasp(self, object_pcd, environment_context):
        """Plan a grasp using the learned policy"""
        if self.use_learning:
            # Use the RL agent to select the best action
            state = self.encode_state(object_pcd, environment_context)
            action = self.rl_agent.act(state, eps=self.exploration_rate)
            
            # Convert action back to grasp parameters
            grasp = self.decode_action_to_grasp(action, object_pcd)
            
            return grasp
        else:
            # Fallback to traditional grasp planning
            return self.fallback_grasp_planner(object_pcd, environment_context)
    
    def encode_state(self, object_pcd, environment_context):
        """Encode the environment into a state suitable for RL"""
        # This could involve:
        # - Converting point cloud to depth image
        # - Extracting object features
        # - Encoding robot state
        pass
    
    def decode_action_to_grasp(self, action, object_pcd):
        """Decode the RL action into a grasp configuration"""
        # Convert the discrete action to a continuous grasp pose
        grasp_pose = self.calculate_grasp_pose_from_action(action, object_pcd)
        
        return Grasp(
            approach_direction=grasp_pose['approach'],
            grasp_width=grasp_pose['width'],
            finger_positions=grasp_pose['finger_positions'],
            quality=1.0  # Assume well-learned quality
        )
```

### Imitation Learning for Grasping

Learning to grasp by observing human demonstrations:

```python
class ImitationGraspLearner:
    def __init__(self):
        self.demonstration_buffer = []
        self.imitation_network = ImitationNetwork()
        self.data_processor = DemonstrationProcessor()
        self.augmentation_pipeline = DataAugmentationPipeline()
        
    def learn_from_demonstration(self, demonstrations):
        """Learn grasping policy from human demonstrations"""
        # Process demonstrations
        processed_demos = self.process_demonstrations(demonstrations)
        
        # Augment demonstrations
        augmented_demos = self.augmentation_pipeline.augment(processed_demos)
        
        # Train imitation network
        self.train_imitation_network(augmented_demos)
        
        return self.imitation_network
    
    def process_demonstrations(self, raw_demonstrations):
        """Process raw demonstrations into training format"""
        processed_demos = []
        
        for demo in raw_demonstrations:
            # Extract relevant features
            processed_demo = {
                'object_repr': self.extract_object_representation(demo.object_state),
                'robot_state': demo.robot_state,
                'action': demo.action_taken,
                'image_sequence': demo.camera_feed,
                'tactile_data': demo.tactile_feedback,
                'success': demo.success
            }
            
            processed_demos.append(processed_demo)
        
        return processed_demos
    
    def extract_object_representation(self, object_state):
        """Extract object representation from state"""
        if hasattr(object_state, 'point_cloud'):
            return self.point_cloud_to_representation(object_state.point_cloud)
        elif hasattr(object_state, 'mesh'):
            return self.mesh_to_representation(object_state.mesh)
        else:
            return self.bounding_box_to_representation(object_state.bbox)
    
    def train_imitation_network(self, demonstrations):
        """Train the imitation learning network"""
        # Prepare dataset
        states = [demo['object_repr'] for demo in demonstrations]
        actions = [demo['action'] for demo in demonstrations]
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(states)
        action_tensor = torch.FloatTensor(actions)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.imitation_network.parameters(), lr=1e-3)
        
        # Training loop
        self.imitation_network.train()
        for epoch in range(100):  # Number of epochs
            optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = self.imitation_network(state_tensor)
            
            # Calculate loss
            loss = criterion(predicted_actions, action_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

class ImitationNetwork(nn.Module):
    def __init__(self, input_size=256, output_size=8):  # 8 DoF for grasp parameters
        super(ImitationNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),  # Adaptive pooling to fixed size
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class DemonstrationProcessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.action_encoder = ActionEncoder()
        
    def process_demonstration(self, raw_demo):
        """Process a single demonstration"""
        # Extract object features
        object_features = self.feature_extractor.extract(raw_demo.object_state)
        
        # Encode human action as robot-appropriate action
        robot_action = self.action_encoder.encode_for_robot(
            raw_demo.human_action, raw_demo.robot_config
        )
        
        # Normalize action to appropriate range
        normalized_action = self.normalize_action(robot_action)
        
        processed_demo = {
            'state': object_features,
            'action': normalized_action,
            'metadata': {
                'object_type': raw_demo.object_type,
                'success': raw_demo.success,
                'execution_time': raw_demo.execution_time
            }
        }
        
        return processed_demo
    
    def normalize_action(self, action):
        """Normalize action to appropriate range for neural network"""
        # Example normalization for grasp parameters
        normalized = action.copy()
        
        # Normalize position (0-1 range)
        if 'position' in normalized:
            normalized['position'] = np.clip(normalized['position'], 0, 1)
        
        # Normalize orientation (0-2π range -> -1 to 1 range)
        if 'orientation' in normalized:
            normalized['orientation'] = np.sin(normalized['orientation'])  # Use sin/cos encoding
        
        # Normalize grasp width (scale to 0-1 based on hand limits)
        if 'grasp_width' in normalized:
            min_width, max_width = 0.01, 0.1  # Example hand limits
            normalized['grasp_width'] = (normalized['grasp_width'] - min_width) / (max_width - min_width)
        
        return normalized

class DataAugmentationPipeline:
    def __init__(self):
        self.rotation_augmenter = RotationAugmenter()
        self.noise_augmenter = NoiseAugmenter()
        self.scale_augmenter = ScaleAugmenter()
        
    def augment(self, demonstrations):
        """Augment demonstrations with various transformations"""
        augmented_demos = []
        
        for demo in demonstrations:
            # Add original demo
            augmented_demos.append(demo)
            
            # Add rotated versions
            rotated_demos = self.rotation_augmenter.augment(demo, n_rotations=4)
            augmented_demos.extend(rotated_demos)
            
            # Add noisy versions
            noisy_demos = self.noise_augmenter.augment(demo, n_noise_samples=2)
            augmented_demos.extend(noisy_demos)
            
            # Add scaled versions
            scaled_demos = self.scale_augmenter.augment(demo, n_scale_samples=2)
            augmented_demos.extend(scaled_demos)
        
        return augmented_demos
```

## Multi-Fingered Hand Control

### Coordinated Finger Control

Controlling multiple fingers in coordination for complex grasps:

```cpp
// Multi-fingered hand control system
class MultiFingerController {
public:
    MultiFingerController(int num_fingers) : num_fingers_(num_fingers) {
        initializeFingerControllers();
        initializeCoordinationManager();
    }

    void controlHand(const HandCommand& command, const HandState& current_state) {
        // Distribute the overall grasp command to individual fingers
        auto finger_commands = coordinateFingers(command, current_state);
        
        // Execute commands for each finger
        for (int i = 0; i < num_fingers_; ++i) {
            finger_controllers_[i]->execute(finger_commands[i]);
        }
    }

    void executePredefinedGrasp(const GraspType& grasp_type) {
        // Execute a predefined grasp pattern
        auto grasp_pattern = getGraspPattern(grasp_type);
        
        for (int i = 0; i < num_fingers_; ++i) {
            std::vector<double> finger_trajectory = calculateFingerTrajectory(
                grasp_pattern.finger_trajectories[i]
            );
            
            finger_controllers_[i]->executeTrajectory(finger_trajectory);
        }
    }

    void adaptGraspBasedOnTactile(const TactileData& tactile_data) {
        // Adjust finger positions/forces based on tactile feedback
        auto adjustments = calculateTactileAdjustments(tactile_data);
        
        for (int i = 0; i < num_fingers_; ++i) {
            if (adjustments.finger_adjustments[i].magnitude > 0.01) { // Threshold
                finger_controllers_[i]->applyAdjustment(
                    adjustments.finger_adjustments[i]
                );
            }
        }
    }

private:
    struct FingerCommand {
        std::vector<double> joint_positions;
        std::vector<double> joint_velocities;
        std::vector<double> joint_torques;
        std::vector<double> contact_forces;
    };

    struct CoordinationCommand {
        std::vector<FingerCommand> finger_commands;
        CoordinationStrategy coordination_strategy;
    };

    int num_fingers_;
    std::vector<std::unique_ptr<FingerController>> finger_controllers_;
    std::unique_ptr<CoordinationManager> coordination_manager_;

    CoordinationCommand coordinateFingers(const HandCommand& command, 
                                         const HandState& current_state) {
        CoordinationCommand coord_command;
        
        switch(command.coordination_type) {
            case COORDINATION_SYNERGISTIC:
                coord_command = calculateSynergisticCommand(command, current_state);
                break;
            case COORDINATION_INDEPENDENT:
                coord_command = calculateIndependentCommand(command, current_state);
                break;
            case COORDINATION_FORCE_BALANCED:
                coord_command = calculateForceBalancedCommand(command, current_state);
                break;
            default:
                coord_command = calculateDefaultCommand(command, current_state);
                break;
        }
        
        return coord_command;
    }
    
    CoordinationCommand calculateSynergisticCommand(const HandCommand& command,
                                                   const HandState& current_state) {
        // Calculate synergistic finger coordination
        // Based on human hand synergies research
        CoordinationCommand result;
        
        // Define synergy patterns (simplified example)
        std::vector<double> synergy_activation = calculateSynergyActivation(
            command.intended_task, current_state
        );
        
        // Map synergies to individual finger movements
        for (int i = 0; i < num_fingers_; ++i) {
            auto finger_command = mapSynergyToFinger(
                synergy_activation, i, current_state.fingers[i]
            );
            result.finger_commands.push_back(finger_command);
        }
        
        return result;
    }
    
    CoordinationCommand calculateForceBalancedCommand(const HandCommand& command,
                                                     const HandState& current_state) {
        // Calculate finger commands to balance forces for stable grasp
        CoordinationCommand result;
        
        // Calculate required contact forces for stable grasp
        auto required_forces = calculateStableGraspForces(
            command.object_properties, command.external_wrench
        );
        
        // Distribute forces among fingers based on their capabilities
        for (int i = 0; i < num_fingers_; ++i) {
            auto finger_force = distributeForceToFinger(
                required_forces, i, current_state.fingers[i]
            );
            
            FingerCommand finger_cmd;
            finger_cmd.contact_forces = finger_force;
            result.finger_commands.push_back(finger_cmd);
        }
        
        return result;
    }
    
    std::vector<double> calculateSynergyActivation(const TaskIntent& task,
                                                  const HandState& state) {
        // Calculate activation levels for different hand synergies
        // Based on the intended task and current state
        
        std::vector<double> synergies(5, 0.0); // 5 basic synergies
        
        // Example: For precision grasp, activate thumb-index synergy highly
        if (task.type == TaskType::PRECISION_GRASP) {
            synergies[0] = 0.9; // Thumb-index synergy
            synergies[1] = 0.3; // Other fingers supporting
        } else if (task.type == TaskType::POWER_GRASP) {
            synergies[2] = 0.8; // Closing synergy
            synergies[3] = 0.6; // Stabilization synergy
        }
        
        return synergies;
    }
    
    FingerCommand mapSynergyToFinger(const std::vector<double>& synergies,
                                    int finger_idx, const FingerState& finger_state) {
        // Map synergy activations to specific finger joint commands
        FingerCommand command;
        
        // For each joint in the finger, calculate command based on synergies
        command.joint_positions.resize(finger_state.num_joints);
        command.joint_velocities.resize(finger_state.num_joints);
        
        for (int j = 0; j < finger_state.num_joints; ++j) {
            // Calculate joint contribution from each synergy
            double joint_pos = finger_state.joint_positions[j];
            for (size_t s = 0; s < synergies.size(); ++s) {
                joint_pos += synergies[s] * synergy_joint_mapping_[finger_idx][j][s];
            }
            
            command.joint_positions[j] = joint_pos;
        }
        
        return command;
    }
    
    void initializeFingerControllers() {
        for (int i = 0; i < num_fingers_; ++i) {
            finger_controllers_.push_back(std::make_unique<FingerController>(i));
        }
    }
    
    void initializeCoordinationManager();
    
    std::vector<std::vector<std::vector<double>>> synergy_joint_mapping_; // [finger][joint][synergy]
};

// Predefined grasp types for multi-fingered hands
enum GraspType {
    PARALLEL_GRASP,
    CYLINDRICAL_GRASP,
    SPHERICAL_GRASP,
    LATERAL_GRASP,
    PINCER_GRASP,
    FINGERTIP_GRASP,
    HOOK_GRASP,
    POWER_GRASP,
    PRECISION_GRASP
};

class GraspPatternLibrary {
public:
    GraspPatternLibrary() {
        initializePredefinedGrasps();
    }

    GraspPattern getGraspPattern(GraspType type) {
        auto it = grasp_patterns_.find(type);
        if (it != grasp_patterns_.end()) {
            return it->second;
        } else {
            return getDefaultGraspPattern();
        }
    }

private:
    struct GraspPattern {
        std::vector<std::vector<double>> finger_trajectories; // [finger][timestep][joint_value]
        std::vector<std::vector<double>> force_profiles;      // [finger][timestep][force]
        std::vector<int> contact_finger_indices;
        std::string description;
        double success_rate;
    };

    std::map<GraspType, GraspPattern> grasp_patterns_;

    void initializePredefinedGrasps() {
        // Define cylindrical grasp pattern
        GraspPattern cylindrical;
        cylindrical.description = "Wrap fingers around cylindrical object";
        cylindrical.success_rate = 0.85;
        
        // For a 4-fingered hand + thumb
        cylindrical.finger_trajectories.resize(5); // 4 fingers + thumb
        for (int i = 0; i < 4; ++i) {  // regular fingers
            cylindrical.finger_trajectories[i] = generateCylindricalGraspTrajectory(i);
        }
        cylindrical.finger_trajectories[4] = generateThumbSupportTrajectory(); // thumb
        
        cylindrical.contact_finger_indices = {0, 1, 2, 3, 4}; // all fingers
        grasp_patterns_[CYLINDRICAL_GRASP] = cylindrical;
        
        // Define precision grasp pattern (thumb + index finger)
        GraspPattern precision;
        precision.description = "Pinch grasp with thumb and index finger";
        precision.success_rate = 0.92;
        
        precision.finger_trajectories.resize(5);
        precision.finger_trajectories[0] = generateIndexFingerTrajectory(); // index
        precision.finger_trajectories[4] = generateThumbOppositionTrajectory(); // thumb
        // Other fingers remain in supporting position
        
        precision.contact_finger_indices = {0, 4}; // index + thumb
        grasp_patterns_[PRECISION_GRASP] = precision;
        
        // Add other grasp patterns similarly...
    }
    
    std::vector<std::vector<double>> generateCylindricalGraspTrajectory(int finger_idx) {
        // Generate trajectory for cylindrical grasp for a specific finger
        std::vector<std::vector<double>> trajectory;
        
        // Simple example: close finger joints gradually
        for (int step = 0; step < 10; ++step) {
            std::vector<double> joints(3); // 3 joints per finger (example)
            double progress = static_cast<double>(step) / 9.0;
            
            // Each joint closes progressively
            joints[0] = 0.2 + 0.7 * progress; // MCP joint
            joints[1] = 0.1 + 0.8 * progress; // PIP joint  
            joints[2] = 0.05 + 0.9 * progress; // DIP joint
            
            trajectory.push_back(joints);
        }
        
        return trajectory;
    }
    
    std::vector<std::vector<double>> generateThumbSupportTrajectory() {
        // Generate trajectory for thumb in cylindrical grasp
        std::vector<std::vector<double>> trajectory;
        
        for (int step = 0; step < 10; ++step) {
            std::vector<double> joints(4); // 4 joints for thumb
            double progress = static_cast<double>(step) / 9.0;
            
            joints[0] = 0.3 + 0.5 * progress; // CMC joint
            joints[1] = 0.1 + 0.7 * progress; // MCP joint
            joints[2] = 0.05 + 0.8 * progress; // IP joint
            joints[3] = 0.02 + 0.85 * progress; // Tip joint
            
            trajectory.push_back(joints);
        }
        
        return trajectory;
    }
    
    GraspPattern getDefaultGraspPattern() {
        // Return a default grasp pattern if specific one not found
        GraspPattern default_pattern;
        default_pattern.description = "Default enveloping grasp";
        default_pattern.success_rate = 0.70;
        return default_pattern;
    }
};
```

## Grasp Adaptation and Recovery

### Grasp Failure Detection and Recovery

Robots need to detect and recover from grasp failures:

```python
class GraspFailureDetector:
    def __init__(self):
        self.force_thresholds = self.initialize_force_thresholds()
        self.tactile_analyzer = TactileAnalyzer()
        self.vision_analyzer = VisionAnalyzer()
        self.slip_detector = SlipDetector()
        
    def initialize_force_thresholds(self):
        """Initialize force thresholds for failure detection"""
        return {
            'sudden_force_drop': 5.0,  # N, indicates object slipped out
            'excessive_force': 40.0,   # N, indicates crushing or jamming
            'force_asymmetry': 0.6,    # ratio, unbalanced grasping
            'torque_limit': 5.0        # Nm, joint torque limits
        }
    
    def detect_failure(self, current_state, tactile_data, vision_data):
        """Detect if the grasp has failed"""
        failure_indicators = {}
        
        # Check force-based indicators
        failure_indicators['force_sudden_drop'] = self.detect_force_drop(
            current_state, self.force_thresholds['sudden_force_drop']
        )
        
        failure_indicators['force_excessive'] = self.detect_excessive_force(
            current_state, self.force_thresholds['excessive_force']
        )
        
        failure_indicators['force_asymmetry'] = self.detect_force_asymmetry(
            current_state, self.force_thresholds['force_asymmetry']
        )
        
        # Check tactile indicators
        failure_indicators['slip_detected'] = self.slip_detector.detect_slip(tactile_data)
        
        failure_indicators['contact_loss'] = self.detect_contact_loss(tactile_data)
        
        # Check vision indicators
        failure_indicators['object_moved'] = self.detect_object_movement(vision_data)
        
        failure_indicators['object_oriented_wrong'] = self.detect_wrong_orientation(vision_data)
        
        # Aggregate failure detection
        overall_failure = self.aggregate_failure_indicators(failure_indicators)
        
        return {
            'failed': overall_failure,
            'indicators': failure_indicators,
            'failure_type': self.classify_failure_type(failure_indicators)
        }
    
    def detect_force_drop(self, state, threshold):
        """Detect sudden drop in grasp force indicating object loss"""
        if len(state.force_history) < 2:
            return False
            
        current_force = np.linalg.norm(state.get_end_effector_force())
        previous_force = np.linalg.norm(state.force_history[-2])
        
        force_change = abs(current_force - previous_force)
        
        # Check if force dropped significantly
        return force_change > threshold and current_force < threshold/2
    
    def detect_excessive_force(self, state, threshold):
        """Detect excessive force indicating crushing or jamming"""
        current_force = np.linalg.norm(state.get_end_effector_force())
        return current_force > threshold
    
    def detect_force_asymmetry(self, state, threshold):
        """Detect unbalanced grasp forces"""
        if not hasattr(state, 'finger_forces') or len(state.finger_forces) < 2:
            return False
            
        forces = state.finger_forces
        force_variability = np.std(forces) / np.mean(forces)
        return force_variability > threshold
    
    def detect_contact_loss(self, tactile_data):
        """Detect loss of expected contact points"""
        expected_contacts = tactile_data.get('expected_contacts', 0)
        current_contacts = tactile_data.get('current_contacts', 0)
        
        # If we have significantly fewer contacts than expected
        if expected_contacts > 0:
            contact_loss_ratio = (expected_contacts - current_contacts) / expected_contacts
            return contact_loss_ratio > 0.3  # 30% loss is significant
        
        return False
    
    def detect_object_movement(self, vision_data):
        """Detect unexpected object movement during grasp hold"""
        if 'object_pose_history' not in vision_data:
            return False
            
        if len(vision_data['object_pose_history']) < 2:
            return False
            
        prev_pose = vision_data['object_pose_history'][-2]
        curr_pose = vision_data['object_pose_history'][-1]
        
        # Calculate movement magnitude
        position_change = np.linalg.norm(
            np.array(curr_pose.position) - np.array(prev_pose.position)
        )
        
        orientation_change = self.calculate_orientation_change(
            prev_pose.orientation, curr_pose.orientation
        )
        
        # If movement exceeds thresholds, object may have slipped
        movement_threshold = 0.005  # 5mm
        orientation_threshold = 0.1  # 0.1 rad
        
        return position_change > movement_threshold or orientation_change > orientation_threshold
    
    def calculate_orientation_change(self, q1, q2):
        """Calculate the angular difference between two orientations"""
        # Convert quaternions to rotation matrices and calculate angle
        import tf.transformations as tft
        matrix1 = tft.quaternion_matrix(q1)[:3, :3]
        matrix2 = tft.quaternion_matrix(q2)[:3, :3]
        
        # Calculate relative rotation matrix
        rel_matrix = np.dot(matrix2, matrix1.T)
        
        # Extract rotation angle
        angle = np.arccos(np.clip((np.trace(rel_matrix) - 1) / 2, -1, 1))
        
        return angle

class GraspRecoveryPlanner:
    def __init__(self):
        self.failure_detector = GraspFailureDetector()
        self.recovery_strategies = self.initialize_recovery_strategies()
        self.motion_planner = MotionPlanner()
        
    def initialize_recovery_strategies(self):
        """Initialize different recovery strategies for different failure types"""
        return {
            'slippage': {
                'increase_grasp_force': 0.3,
                'reposition_fingers': 0.5,
                'adjust_approach_angle': 0.2
            },
            'contact_loss': {
                'regrasp': 0.8,
                'adjust_grasp_width': 0.2
            },
            'object_moved': {
                'regrasp': 0.6,
                'reposition_hand': 0.4
            },
            'force_asymmetry': {
                'adjust_force_distribution': 0.7,
                'reposition_grasp': 0.3
            }
        }
    
    def plan_recovery(self, failure_report, current_state, object_properties):
        """Plan recovery action based on failure type"""
        failure_type = failure_report['failure_type']
        
        if failure_type not in self.recovery_strategies:
            return self.plan_generic_recovery(current_state, object_properties)
        
        strategy_weights = self.recovery_strategies[failure_type]
        
        # Select the highest weighted recovery action
        best_action = max(strategy_weights.items(), key=lambda x: x[1])
        action_type = best_action[0]
        
        # Plan specific recovery
        if action_type == 'increase_grasp_force':
            return self.plan_force_increase_recovery(current_state)
        elif action_type == 'reposition_fingers':
            return self.plan_reposition_fingers_recovery(current_state, object_properties)
        elif action_type == 'regrasp':
            return self.plan_regrasp_recovery(current_state, object_properties)
        elif action_type == 'adjust_grasp_width':
            return self.plan_grasp_width_adjustment_recovery(current_state, object_properties)
        else:
            return self.plan_generic_recovery(current_state, object_properties)
    
    def plan_force_increase_recovery(self, current_state):
        """Plan recovery by increasing grasp force"""
        recovery_plan = {
            'action': 'increase_grasp_force',
            'force_increase_ratio': 1.3,  # Increase force by 30%
            'monitoring_duration': 2.0,   # Monitor for 2 seconds after increase
            'success_criteria': ['stable_force', 'no_slip_detected']
        }
        
        return recovery_plan
    
    def plan_regrasp_recovery(self, current_state, object_properties):
        """Plan complete regrasp action"""
        # First, release the current grasp
        release_plan = {
            'action': 'release_grasp',
            'gripper_command': 'open',
            'duration': 1.0
        }
        
        # Plan approach trajectory for regrasp
        approach_pose = self.calculate_approach_pose(object_properties)
        
        approach_plan = {
            'action': 'move_to_approach_pose',
            'target_pose': approach_pose,
            'trajectory': self.motion_planner.plan_trajectory(
                current_state.pose, approach_pose
            )
        }
        
        # Plan new grasp
        new_grasp = self.select_better_grasp(object_properties)
        
        grasp_plan = {
            'action': 'execute_new_grasp',
            'grasp_params': new_grasp,
            'success_criteria': ['firm_contact', 'appropriate_force']
        }
        
        recovery_plan = {
            'sequence': [release_plan, approach_plan, grasp_plan],
            'monitoring_after_each': True
        }
        
        return recovery_plan
    
    def select_better_grasp(self, object_properties):
        """Select a more stable grasp based on object properties"""
        # If the previous grasp failed due to slippage, 
        # try a grasp with better friction contact
        if object_properties.get('slip_resistant_regions'):
            # Try grasp that contacts slip-resistant regions
            return self.select_slip_resistant_grasp(object_properties)
        else:
            # Try a grasp with more contact points
            return self.select_multi_contact_grasp(object_properties)
    
    def calculate_approach_pose(self, object_properties):
        """Calculate good approach pose for regrasp"""
        # Find a stable approach direction
        approach_direction = self.find_stable_approach_direction(object_properties)
        
        # Calculate approach position a safe distance from object
        object_center = object_properties['center']
        approach_position = object_center + approach_direction * 0.05  # 5cm away
        
        # Calculate approach orientation (typically perpendicular to grasp surface)
        approach_orientation = self.calculate_approach_orientation(
            approach_direction
        )
        
        return {
            'position': approach_position,
            'orientation': approach_orientation
        }
    
    def plan_generic_recovery(self, current_state, object_properties):
        """Plan a generic recovery when specific strategy is not available"""
        return {
            'action': 'regrasp_with_adjustments',
            'sequence': [
                {'sub_action': 'slightly_open_gripper', 'duration': 0.5},
                {'sub_action': 'reposition_for_better_grasp', 'adjustment': 'lateral'},
                {'sub_action': 'execute_adjusted_grasp', 'correction_factor': 1.1}
            ],
            'monitoring': True,
            'fallback_to_regrasp': True
        }

class AdaptiveGraspController:
    def __init__(self):
        self.failure_detector = GraspFailureDetector()
        self.recovery_planner = GraspRecoveryPlanner()
        self.current_grasp_stability = 1.0
        self.successful_grasp_count = 0
        
    def monitor_grasp(self, current_state, tactile_data, vision_data):
        """Continuously monitor the grasp for signs of failure"""
        failure_report = self.failure_detector.detect_failure(
            current_state, tactile_data, vision_data
        )
        
        if failure_report['failed']:
            # Initiate recovery
            recovery_plan = self.recovery_planner.plan_recovery(
                failure_report, current_state, self.get_object_properties(vision_data)
            )
            
            return {
                'status': 'needs_recovery',
                'failure_report': failure_report,
                'recovery_plan': recovery_plan
            }
        else:
            # Update stability metrics
            self.update_stability_metrics(current_state, tactile_data)
            
            return {
                'status': 'stable',
                'stability_score': self.current_grasp_stability
            }
    
    def update_stability_metrics(self, state, tactile_data):
        """Update grasp stability metrics based on current data"""
        # Calculate stability based on various indicators
        force_stability = self.calculate_force_stability(state)
        tactile_stability = self.calculate_tactile_stability(tactile_data)
        geometric_stability = self.calculate_geometric_stability(state)
        
        # Weighted combination of stability measures
        self.current_grasp_stability = (
            0.4 * force_stability + 
            0.4 * tactile_stability + 
            0.2 * geometric_stability
        )
        
        return self.current_grasp_stability
    
    def calculate_force_stability(self, state):
        """Calculate stability based on force measurements"""
        # A stable grasp has consistent, appropriate force levels
        if hasattr(state, 'grasp_force'):
            force = state.grasp_force
            # Optimal force range (example values)
            min_optimal = 5.0  # N
            max_optimal = 20.0 # N
            
            if min_optimal <= force <= max_optimal:
                return 1.0
            elif force < min_optimal:
                # Too little force - may drop
                return force / min_optimal
            else:
                # Too much force - inefficient/unsafe
                return max_optimal / force
        else:
            return 0.5  # Unknown stability
    
    def calculate_tactile_stability(self, tactile_data):
        """Calculate stability based on tactile feedback"""
        # Stable grasp has consistent contact patterns
        if 'contact_stability' in tactile_data:
            return tactile_data['contact_stability']
        else:
            # Default: look at number and consistency of contacts
            contacts = tactile_data.get('active_contacts', 0)
            # Prefer 3+ contacts for stability
            return min(1.0, max(0.0, (contacts - 1) / 3.0))
    
    def calculate_geometric_stability(self, state):
        """Calculate stability based on geometric grasp properties"""
        # Consider how grasp contacts support object's center of mass
        if hasattr(state, 'contact_points') and hasattr(state, 'object_com'):
            # Calculate how well contacts support the object
            # This is a simplified calculation
            com = state.object_com
            contacts = state.contact_points
            
            # Find support polygon formed by contacts
            # Calculate distance from COM to support polygon
            support_distance = self.calculate_com_distance_to_support(contacts, com)
            
            # Return stability measure based on support
            return max(0.0, min(1.0, 0.02 / max(0.001, support_distance)))  # 2cm threshold
        else:
            return 0.5  # Unknown stability
```

## Safety in Manipulation

### Safety Considerations for Manipulation

Safety is paramount in robotic manipulation, especially with human interaction:

```python
class ManipulationSafetySystem:
    def __init__(self):
        self.collision_detector = CollisionDetector()
        self.force_monitor = ForceMonitor()
        self.velocity_limiter = VelocityLimiter()
        self.emergency_stop = EmergencyStopSystem()
        self.human_proximity_detector = HumanProximityDetector()
        self.risk_assessment_engine = RiskAssessmentEngine()
        
    def check_manipulation_safety(self, planned_trajectory, environment_state):
        """Check if a manipulation trajectory is safe to execute"""
        safety_report = {
            'trajectory_safe': True,
            'collision_risk': 0.0,
            'force_risk': 0.0,
            'velocity_risk': 0.0,
            'human_safety_risk': 0.0,
            'overall_safety_score': 1.0,
            'issues': []
        }
        
        # Check for collisions along trajectory
        collision_check = self.collision_detector.check_trajectory_safety(
            planned_trajectory, environment_state
        )
        safety_report['collision_risk'] = collision_check['risk']
        if collision_check['collision_detected']:
            safety_report['trajectory_safe'] = False
            safety_report['issues'].append('collision_detected')
        
        # Check force constraints
        force_check = self.force_monitor.check_trajectory_forces(
            planned_trajectory, environment_state
        )
        safety_report['force_risk'] = force_check['risk']
        if force_check['excessive_force']:
            safety_report['trajectory_safe'] = False
            safety_report['issues'].append('excessive_force')
        
        # Check velocity constraints
        velocity_check = self.velocity_limiter.check_trajectory_velocities(
            planned_trajectory
        )
        safety_report['velocity_risk'] = velocity_check['risk']
        if velocity_check['excessive_velocity']:
            safety_report['trajectory_safe'] = False
            safety_report['issues'].append('excessive_velocity')
        
        # Check human safety
        human_safety_check = self.human_proximity_detector.check_human_safety(
            planned_trajectory, environment_state
        )
        safety_report['human_safety_risk'] = human_safety_check['risk']
        if human_safety_check['human_at_risk']:
            safety_report['trajectory_safe'] = False
            safety_report['issues'].append('human_at_risk')
        
        # Calculate overall safety score
        safety_report['overall_safety_score'] = self.calculate_safety_score(
            safety_report
        )
        
        return safety_report
    
    def calculate_safety_score(self, safety_report):
        """Calculate overall safety score based on all safety checks"""
        # Weight different safety factors
        weights = {
            'collision_risk': 0.3,
            'force_risk': 0.25,
            'velocity_risk': 0.2,
            'human_safety_risk': 0.25
        }
        
        # Calculate inverse of risk scores (lower risk = higher safety)
        collision_safety = max(0.0, 1.0 - safety_report['collision_risk'])
        force_safety = max(0.0, 1.0 - safety_report['force_risk'])
        velocity_safety = max(0.0, 1.0 - safety_report['velocity_risk'])
        human_safety = max(0.0, 1.0 - safety_report['human_safety_risk'])
        
        overall_score = (
            weights['collision_risk'] * collision_safety +
            weights['force_risk'] * force_safety +
            weights['velocity_risk'] * velocity_safety +
            weights['human_safety_risk'] * human_safety
        )
        
        return overall_score

class ForceSafetyController:
    def __init__(self):
        self.max_force_thresholds = {
            'fingertip': 20.0,    # N
            'palm': 50.0,         # N
            'object_interaction': 30.0  # N
        }
        self.force_rate_limits = {
            'max_force_rate': 100.0  # N/s
        }
        self.compliance_controller = ComplianceController()
        
    def limit_interaction_force(self, desired_force, contact_type):
        """Limit interaction force based on contact type"""
        max_force = self.max_force_thresholds.get(contact_type, 30.0)
        limited_force = min(desired_force, max_force)
        
        return limited_force
    
    def check_force_rate(self, current_force, previous_force, dt):
        """Check that force is not changing too rapidly"""
        force_change = abs(current_force - previous_force)
        force_rate = force_change / dt if dt > 0 else 0
        
        if force_rate > self.force_rate_limits['max_force_rate']:
            # Apply rate limiting
            return previous_force + np.sign(current_force - previous_force) * \
                   self.force_rate_limits['max_force_rate'] * dt
        else:
            return current_force

class SafeExecutionMonitor:
    def __init__(self):
        self.safety_system = ManipulationSafetySystem()
        self.intervention_thresholds = self.define_intervention_thresholds()
        self.execution_history = []
        
    def define_intervention_thresholds(self):
        """Define thresholds for safety intervention"""
        return {
            'collision_risk': 0.1,      # 10% risk triggers warning
            'force_risk': 0.15,         # 15% risk triggers warning
            'human_safety_risk': 0.05,  # 5% risk triggers immediate stop
            'overall_safety_score': 0.7 # Below 70% triggers review
        }
    
    def monitor_execution(self, current_state, tactile_data, vision_data, 
                         execution_context):
        """Monitor execution for safety violations"""
        safety_status = {
            'execution_safe': True,
            'warnings': [],
            'interventions': [],
            'actions_taken': []
        }
        
        # Check current state against safety criteria
        if self.detect_immediate_danger(current_state, tactile_data):
            safety_status['execution_safe'] = False
            action = self.emergency_stop.activate()
            safety_status['interventions'].append('emergency_stop')
            safety_status['actions_taken'].append(action)
            return safety_status
        
        # Check for potential safety issues
        if self.detect_force_anomaly(tactile_data):
            warning = self.slow_down_execution()
            safety_status['warnings'].append('high_force_detected')
            safety_status['actions_taken'].append(warning)
        
        if self.detect_unstable_object(vision_data):
            warning = self.pause_execution_for_verification()
            safety_status['warnings'].append('object_instability')
            safety_status['actions_taken'].append(warning)
        
        # Record safety metrics
        self.execution_history.append(self.extract_safety_metrics(
            current_state, tactile_data, vision_data
        ))
        
        return safety_status
    
    def detect_immediate_danger(self, state, tactile_data):
        """Detect conditions requiring immediate safety action"""
        # Check for excessive joint torques
        if hasattr(state, 'joint_torques'):
            max_torque = max(abs(t) for t in state.joint_torques)
            if max_torque > 50.0:  # N*m threshold
                return True
        
        # Check for slip with high force (indicating crush risk)
        if tactile_data.get('slip_detected', False) and tactile_data.get('contact_force', 0) > 40.0:
            return True
            
        # Check for human proximity with high speed
        if state.get('end_effector_speed', 0) > 0.2 and self.near_humans(state):  # 0.2 m/s and 1m proximity
            return True
            
        return False
    
    def detect_force_anomaly(self, tactile_data):
        """Detect anomalous force patterns"""
        return tactile_data.get('max_contact_force', 0) > 35.0  # High force threshold
    
    def detect_unstable_object(self, vision_data):
        """Detect object instability during manipulation"""
        if 'object_pose_history' in vision_data and len(vision_data['object_pose_history']) > 5:
            recent_poses = vision_data['object_pose_history'][-5:]
            # Check for excessive movement
            position_variance = np.var([pose.position for pose in recent_poses])
            return position_variance > 0.001  # m^2 threshold
    
    def slow_down_execution(self):
        """Reduce speed for safer execution"""
        return "speed_reduction_applied"
    
    def pause_execution_for_verification(self):
        """Pause execution to verify safety"""
        return "execution_paused_for_verification"
    
    def near_humans(self, state):
        """Check if robot is near humans"""
        # Implementation would check robot position against human positions
        return False  # Simplified implementation
```

## Evaluation Metrics

### Metrics for Manipulation Performance

Evaluating manipulation performance requires comprehensive metrics:

```python
# Metrics for manipulation performance evaluation
MANIPULATION_EVALUATION_METRICS = {
    'grasp_success_rate': {
        'definition': 'Percentage of grasp attempts that successfully acquire and hold the object',
        'calculation': 'Successful grasps / Total grasp attempts',
        'target': 0.85,
        'importance': 'Primary measure of grasping capability'
    },
    'grasp_stability': {
        'definition': 'Ability to maintain object hold during manipulation',
        'calculation': 'Objects held for required duration / Total grasps',
        'target': 0.90,
        'importance': 'Critical for successful manipulation tasks'
    },
    'precision_placement': {
        'definition': 'Accuracy in placing objects at specified locations',
        'calculation': 'Mean position error from target (in meters)',
        'target': 0.01,  # 1cm
        'importance': 'Essential for many manipulation tasks'
    },
    'task_completion_rate': {
        'definition': 'Percentage of manipulation tasks completed successfully',
        'calculation': 'Successful task completions / Total task attempts',
        'target': 0.80,
        'importance': 'Measures overall system effectiveness'
    },
    'manipulation_speed': {
        'definition': 'Time to complete standard manipulation tasks',
        'calculation': 'Average time per task type (in seconds)',
        'target': 10.0,  # seconds for basic tasks
        'importance': 'Affects practical usability'
    },
    'energy_efficiency': {
        'definition': 'Energy used per manipulation task',
        'calculation': 'Total energy consumed / Number of tasks completed',
        'target': 50.0,  # Joules per task
        'importance': 'Affects operational cost and sustainability'
    },
    'safety_incidents': {
        'definition': 'Number of safety-related incidents during manipulation',
        'calculation': 'Safety incidents / Total operation hours',
        'target': 0.0,  # No incidents
        'importance': 'Critical for human-robot interaction'
    }
}

class ManipulationPerformanceEvaluator:
    def __init__(self):
        self.metrics = MANIPULATION_EVALUATION_METRICS
        self.experiment_logger = ExperimentLogger()
        self.video_analyzer = VideoAnalyzer()
        self.force_analyzer = ForceAnalyzer()
        
    def evaluate_manipulation_system(self, system, test_scenarios):
        """Comprehensive evaluation of manipulation system"""
        evaluation_results = {}
        
        # Evaluate grasp success rate
        evaluation_results['grasp_success_rate'] = self.evaluate_grasp_success(
            system, test_scenarios.grasp_tests
        )
        
        # Evaluate grasp stability
        evaluation_results['grasp_stability'] = self.evaluate_grasp_stability(
            system, test_scenarios.stability_tests
        )
        
        # Evaluate precision placement
        evaluation_results['precision_placement'] = self.evaluate_precision_placement(
            system, test_scenarios.placement_tests
        )
        
        # Evaluate task completion rate
        evaluation_results['task_completion_rate'] = self.evaluate_task_completion(
            system, test_scenarios.complex_tasks
        )
        
        # Evaluate manipulation speed
        evaluation_results['manipulation_speed'] = self.evaluate_speed(
            system, test_scenarios.speed_tests
        )
        
        # Evaluate energy efficiency
        evaluation_results['energy_efficiency'] = self.evaluate_energy_efficiency(
            system, test_scenarios.energy_tests
        )
        
        # Evaluate safety
        evaluation_results['safety_incidents'] = self.evaluate_safety(
            system, test_scenarios.safety_tests
        )
        
        # Calculate overall score
        evaluation_results['overall_performance'] = self.calculate_overall_score(
            evaluation_results
        )
        
        # Generate detailed report
        report = self.generate_evaluation_report(
            evaluation_results, test_scenarios
        )
        
        return evaluation_results, report
    
    def evaluate_grasp_success(self, system, grasp_tests):
        """Evaluate grasp success rate on various objects"""
        successful_grasps = 0
        total_attempts = 0
        
        for test in grasp_tests:
            # Execute grasp
            grasp_result = system.execute_grasp(test.object, test.grasp_params)
            
            # Record result
            if grasp_result.success and grasp_result.stable:
                successful_grasps += 1
            
            total_attempts += 1
            
            # Log detailed information
            self.experiment_logger.log_grasp_attempt(
                object_type=test.object.type,
                grasp_params=test.grasp_params,
                result=grasp_result,
                timestamp=time.time()
            )
        
        success_rate = successful_grasps / total_attempts if total_attempts > 0 else 0.0
        return {
            'success_rate': success_rate,
            'successful_grasps': successful_grasps,
            'total_attempts': total_attempts,
            'by_object_type': self.calculate_success_by_object_type(grasp_tests)
        }
    
    def evaluate_grasp_stability(self, system, stability_tests):
        """Evaluate grasp stability over time"""
        stable_grasps = 0
        total_grasps = 0
        
        for test in stability_tests:
            # Execute grasp
            grasp_result = system.execute_grasp(test.object, test.grasp_params)
            
            if grasp_result.success:
                # Monitor grasp during specified task/duration
                stability_result = self.monitor_grasp_stability(
                    system, grasp_result, test.monitoring_duration, test.task
                )
                
                if stability_result['remained_stable']:
                    stable_grasps += 1
                
                total_grasps += 1
        
        stability_rate = stable_grasps / total_grasps if total_grasps > 0 else 0.0
        return {
            'stability_rate': stability_rate,
            'stable_grasps': stable_grasps,
            'total_stable_tests': total_grasps
        }
    
    def evaluate_precision_placement(self, system, placement_tests):
        """Evaluate accuracy of object placement"""
        position_errors = []
        
        for test in placement_tests:
            # Execute placement task
            placement_result = system.execute_placement(
                test.object, test.start_pose, test.target_pose
            )
            
            # Calculate placement error
            if placement_result.success:
                error = self.calculate_placement_error(
                    placement_result.final_pose, test.target_pose
                )
                position_errors.append(error)
        
        if position_errors:
            mean_error = np.mean(position_errors)
            std_error = np.std(position_errors)
            return {
                'mean_position_error': mean_error,
                'std_position_error': std_error,
                'median_error': np.median(position_errors),
                'success_count': len([e for e in position_errors if e < 0.01]),  # Within 1cm
                'total_evaluations': len(position_errors)
            }
        else:
            return {'mean_position_error': float('inf'), 'total_evaluations': 0}
    
    def calculate_overall_score(self, evaluation_results):
        """Calculate overall performance score"""
        # Weighted combination of all metrics
        weights = {
            'grasp_success_rate': 0.25,
            'grasp_stability': 0.20,
            'precision_placement': 0.15,
            'task_completion_rate': 0.20,
            'manipulation_speed': 0.10,  # Inverted (lower time = higher score)
            'energy_efficiency': 0.05,
            'safety_incidents': 0.05  # Inverted (fewer incidents = higher score)
        }
        
        # Normalize each metric to 0-1 scale
        normalized_scores = {}
        
        # Grasp success rate (higher is better, max 1.0)
        normalized_scores['grasp_success_rate'] = min(
            1.0, evaluation_results['grasp_success_rate']['success_rate']
        )
        
        # Grasp stability (higher is better, max 1.0)
        normalized_scores['grasp_stability'] = min(
            1.0, evaluation_results['grasp_stability']['stability_rate']
        )
        
        # Precision placement (inverse - lower error is better)
        placement_error = evaluation_results['precision_placement'].get('mean_position_error', float('inf'))
        if placement_error != float('inf'):
            normalized_scores['precision_placement'] = max(0.0, 1.0 - (placement_error / 0.1))  # 10cm baseline
        else:
            normalized_scores['precision_placement'] = 0.0
        
        # Task completion rate (higher is better, max 1.0)
        normalized_scores['task_completion_rate'] = min(
            1.0, evaluation_results['task_completion_rate']['completion_rate']
        )
        
        # Manipulation speed (inverse - faster is better)
        avg_time = evaluation_results['manipulation_speed'].get('avg_time', float('inf'))
        if avg_time != float('inf'):
            normalized_scores['manipulation_speed'] = max(0.0, 1.0 - (avg_time / 20.0))  # 20s baseline
        else:
            normalized_scores['manipulation_speed'] = 0.0
        
        # Energy efficiency (inverse - lower consumption is better)
        avg_energy = evaluation_results['energy_efficiency'].get('avg_energy_per_task', float('inf'))
        if avg_energy != float('inf'):
            normalized_scores['energy_efficiency'] = max(0.0, 1.0 - (avg_energy / 100.0))  # 100J baseline
        else:
            normalized_scores['energy_efficiency'] = 0.0
        
        # Safety incidents (inverse - fewer incidents is better)
        incident_rate = evaluation_results['safety_incidents']['incident_rate']
        normalized_scores['safety_incidents'] = max(0.0, 1.0 - incident_rate)
        
        # Calculate weighted average
        overall_score = sum(
            weights[metric] * score 
            for metric, score in normalized_scores.items()
        )
        
        return overall_score
    
    def generate_evaluation_report(self, results, test_scenarios):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_summary': {
                'overall_performance_score': results['overall_performance'],
                'evaluation_date': datetime.now().isoformat(),
                'total_test_scenarios': len(test_scenarios.all_tests),
                'system_configuration': test_scenarios.system_config
            },
            'detailed_results': results,
            'strengths': self.identify_strengths(results),
            'weaknesses': self.identify_weaknesses(results),
            'recommendations': self.generate_recommendations(results),
            'comparisons': self.compare_to_benchmarks(results)
        }
        
        return report
    
    def identify_strengths(self, results):
        """Identify system strengths based on evaluation results"""
        strengths = []
        
        if results['grasp_success_rate']['success_rate'] > 0.9:
            strengths.append("Excellent grasp success rate")
            
        if results['grasp_stability']['stability_rate'] > 0.95:
            strengths.append("Very stable grasps")
            
        if results['precision_placement']['mean_position_error'] < 0.005:  # <5mm
            strengths.append("High-precision placement")
            
        if results['task_completion_rate']['completion_rate'] > 0.9:
            strengths.append("High task completion rate")
            
        return strengths
    
    def identify_weaknesses(self, results):
        """Identify system weaknesses based on evaluation results"""
        weaknesses = []
        
        if results['grasp_success_rate']['success_rate'] < 0.7:
            weaknesses.append("Low grasp success rate")
            
        if results['grasp_stability']['stability_rate'] < 0.8:
            weaknesses.append("Grasp instability issues")
            
        if results['precision_placement']['mean_position_error'] > 0.02:  # >2cm
            weaknesses.append("Low placement precision")
            
        if results['safety_incidents']['incident_rate'] > 0.01:  # >1% incident rate
            weaknesses.append("Safety concerns")
            
        return weaknesses
        
    def generate_recommendations(self, results):
        """Generate improvement recommendations based on evaluation"""
        recommendations = []
        
        if results['grasp_success_rate']['success_rate'] < 0.8:
            recommendations.append({
                'area': 'Grasp Planning',
                'recommendation': 'Implement learning-based grasp synthesis to improve success rate',
                'priority': 'high'
            })
            
        if results['grasp_stability']['stability_rate'] < 0.85:
            recommendations.append({
                'area': 'Grasp Control',
                'recommendation': 'Add tactile feedback-based grasp adjustment',
                'priority': 'high'
            })
            
        if results['manipulation_speed']['avg_time'] > 15.0:
            recommendations.append({
                'area': 'Motion Planning',
                'recommendation': 'Optimize trajectories for faster execution',
                'priority': 'medium'
            })
            
        if results['energy_efficiency']['avg_energy_per_task'] > 80.0:
            recommendations.append({
                'area': 'Control Optimization',
                'recommendation': 'Implement energy-aware control strategies',
                'priority': 'medium'
            })
            
        return recommendations

class ExperimentLogger:
    def __init__(self, log_dir="manipulation_logs"):
        self.log_dir = log_dir
        self.current_session = self.start_new_session()
        
    def start_new_session(self):
        """Start a new logging session"""
        import uuid
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(self.log_dir, session_id)
        
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def log_grasp_attempt(self, object_type, grasp_params, result, timestamp):
        """Log details of a grasp attempt"""
        log_entry = {
            'timestamp': timestamp,
            'object_type': object_type,
            'grasp_params': grasp_params,
            'result': {
                'success': result.success,
                'stable': result.stable,
                'grasp_force': getattr(result, 'grasp_force', None),
                'slip_detected': getattr(result, 'slip_detected', False),
                'error_message': getattr(result, 'error_message', None)
            }
        }
        
        # Save to JSON file
        log_file = os.path.join(self.current_session, f"grasp_log_{int(timestamp)}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
```

## Summary

Manipulation and grasping control are fundamental capabilities that enable humanoid robots to interact physically with their environment. This chapter explored the technical challenges and solutions in robotic manipulation, from grasp planning and hand design to force control and learning-based approaches.

The chapter covered various approaches to grasp planning, including physics-based analysis for stable grasps and learning-based methods that can handle novel objects. It detailed the design considerations for anthropomorphic hands, comparing underactuated and fully actuated approaches, each with their own trade-offs in complexity, cost, and dexterity.

Force and tactile control were discussed in depth, including impedance control strategies that allow robots to safely interact with objects and environments. The chapter addressed the importance of compliance control and how tactile feedback can be used to improve grasp stability and adapt to changing conditions.

The learning-based approaches to manipulation, including reinforcement learning and imitation learning, were explored for their potential to improve grasp success rates and adapt to new situations. The chapter also covered multi-fingered hand control, grasp adaptation and recovery strategies, and the critical safety considerations in manipulation.

Finally, comprehensive evaluation metrics were presented to assess manipulation performance, including grasp success rate, stability, precision, and safety measures. The chapter emphasized that successful manipulation requires the integration of perception, planning, and control systems working together.

## Exercises

1. Implement a grasp stability evaluation algorithm that can predict the probability of grasp failure based on contact forces and object properties. How would you validate this algorithm experimentally?

2. Design a multi-fingered hand controller that can execute complex grasps using synergistic coordination. What biomechanical principles would you incorporate into your design?

3. Create a learning-based grasp planning system that can improve its performance through interaction with various objects. What features would you use to represent objects and grasps?

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*