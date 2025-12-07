---
id: module-03-chapter-02
title: Chapter 02 - Machine Learning for Humanoid Perception
sidebar_position: 10
---

# Chapter 02 - Machine Learning for Humanoid Perception

## Table of Contents
- [Overview](#overview)
- [Perception in Humanoid Robotics](#perception-in-humanoid-robotics)
- [Vision Systems and Deep Learning](#vision-systems-and-deep-learning)
- [Audio Processing and Natural Language](#audio-processing-and-natural-language)
- [Tactile and Proprioceptive Sensing](#tactile-and-proprioceptive-sensing)
- [Sensor Fusion Techniques](#sensor-fusion-techniques)
- [Real-time Perception Challenges](#real-time-perception-challenges)
- [Learning from Human Demonstrations](#learning-from-human-demonstrations)
- [Perception in Dynamic Environments](#perception-in-dynamic-environments)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Perception is the foundation of intelligent behavior in humanoid robots, enabling them to understand and interact with their environment. Unlike traditional robots operating in controlled environments, humanoid robots must perceive and interpret complex, unstructured real-world scenes. This chapter explores how machine learning techniques enable humanoid robots to extract meaningful information from various sensors, combining multiple modalities to form a coherent understanding of their environment.

The perception capabilities of humanoid robots need to match human-level performance in many situations. This requires sophisticated machine learning models that can process visual, auditory, tactile, and proprioceptive information in real-time, adapting to changing conditions while maintaining reliability and safety.

## Perception in Humanoid Robotics

### Unique Challenges for Humanoid Perception

Humanoid robots face specific perception challenges:

1. **Human-level sensory requirements**: Must perceive the world similarly to humans for effective interaction
2. **Real-time processing**: Balance control and social interaction require immediate responses
3. **Multi-modal integration**: Combining vision, audition, touch, and proprioception
4. **Dynamic environments**: Operating in spaces designed for and shared with humans
5. **Social perception**: Recognizing and interpreting human emotions, gestures, and intentions

### Perception Pipeline Architecture

```python
class HumanoidPerceptionSystem:
    def __init__(self):
        self.preprocessing = PreprocessingModule()
        self.vision_system = VisionPerceptionModule()
        self.audio_system = AudioPerceptionModule()
        self.tactile_system = TactilePerceptionModule()
        self.fusion_module = SensorFusionModule()
        self.context_interpreter = ContextInterpreterModule()
        
    def process_perception_cycle(self, raw_sensor_data):
        # Preprocess raw sensor data
        preprocessed_data = self.preprocessing.process(raw_sensor_data)
        
        # Process modalities separately
        visual_info = self.vision_system.process(preprocessed_data['camera'])
        audio_info = self.audio_system.process(preprocessed_data['microphones'])
        tactile_info = self.tactile_system.process(preprocessed_data['tactile'])
        
        # Fuse information from modalities
        fused_percepts = self.fusion_module.fuse(visual_info, audio_info, tactile_info)
        
        # Interpret in context
        interpreted_scene = self.context_interpreter.interpret(fused_percepts)
        
        return interpreted_scene
```

### Perception Goals for Humanoid Robots

Humanoid robots need to achieve several perception goals:

1. **Object Recognition**: Identifying objects in the environment
2. **Human Detection and Tracking**: Locating and following humans
3. **Scene Understanding**: Comprehending the layout and affordances of spaces
4. **Action Recognition**: Understanding what humans are doing
5. **Emotion Recognition**: Detecting human emotional states
6. **Spatial Mapping**: Creating and maintaining maps of the environment
7. **Intention Inference**: Predicting what humans will do next

## Vision Systems and Deep Learning

### Convolutional Neural Networks for Vision

CNNs form the backbone of humanoid vision systems:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class HumanoidVisionNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(HumanoidVisionNet, self).__init__()
        
        # Use a pretrained backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Add specialized heads for different perception tasks
        self.object_detection_head = nn.Linear(1000, num_classes)
        self.pose_estimation_head = nn.Linear(1000, 24)  # 24 joint angles
        self.emotion_recognition_head = nn.Linear(1000, 7)  # 7 basic emotions
        
    def forward(self, x):
        features = self.backbone(x)
        
        # Multi-task outputs
        object_logits = self.object_detection_head(features)
        pose_estimates = self.pose_estimation_head(features)
        emotion_logits = self.emotion_recognition_head(features)
        
        return {
            'objects': object_logits,
            'pose': pose_estimates,
            'emotion': emotion_logits
        }

# Initialize the vision system
vision_system = HumanoidVisionNet()
```

### Real-time Object Detection

For humanoid robots, object detection must be both accurate and fast:

```python
import torch
import torchvision.transforms as transforms

class RealTimeObjectDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ])
        self.confidence_threshold = 0.5
        
    def detect_objects(self, image):
        """Detect objects in real-time"""
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run detection
        with torch.no_grad():
            detections = self.model(input_tensor)
        
        # Filter by confidence
        high_conf_detections = detections[detections[:, 4] > self.confidence_threshold]
        
        return self.post_process(high_conf_detections)
        
    def post_process(self, detections):
        """Apply non-maximum suppression and return clean detections"""
        # Implementation would include NMS and bounding box refinement
        return detections
```

### Human Pose Estimation

Understanding human pose is crucial for humanoid interaction:

```python
import cv2
import numpy as np

class HumanPoseEstimator:
    def __init__(self, model_path):
        self.model = self.load_pose_model(model_path)
        
    def estimate_pose(self, image):
        """Estimate human pose keypoints"""
        # Run pose estimation
        keypoints = self.model.predict(image)
        
        # Structure keypoints for humanoid interaction
        person_pose = self.structure_keypoints(keypoints)
        
        # Calculate pose-based features for interaction
        interaction_features = self.extract_interaction_features(person_pose)
        
        return person_pose, interaction_features
        
    def extract_interaction_features(self, pose):
        """Extract features for interaction decision-making"""
        features = {}
        
        # Calculate where person is looking
        features['gaze_direction'] = self.calculate_gaze_direction(pose)
        
        # Calculate attention direction
        features['attention'] = self.calculate_attention(pose)
        
        # Calculate gesture recognition
        features['gesture'] = self.recognize_gesture(pose)
        
        return features
```

### Scene Understanding

Humanoid robots need to understand scenes contextually:

```python
class SceneUnderstandingModule:
    def __init__(self):
        self.semantic_segmentation = SemanticSegmentationModel()
        self.depth_estimator = DepthEstimationModel()
        self.object_detector = ObjectDetectionModel()
        
    def understand_scene(self, image, camera_params):
        """Comprehensively understand a scene"""
        # Get semantic segmentation
        semantic_map = self.semantic_segmentation.predict(image)
        
        # Estimate depth
        depth_map = self.depth_estimator.predict(image)
        
        # Detect objects
        objects = self.object_detector.detect(image)
        
        # Combine information into scene graph
        scene_graph = self.build_scene_graph(
            semantic_map, depth_map, objects, camera_params
        )
        
        # Extract affordances (possibilities for action)
        affordances = self.extract_affordances(scene_graph)
        
        return scene_graph, affordances
        
    def extract_affordances(self, scene_graph):
        """Extract action possibilities from scene"""
        affordances = []
        
        for obj in scene_graph['objects']:
            if obj['class'] == 'chair':
                affordances.append({
                    'type': 'sit',
                    'object': obj['id'],
                    'location': obj['position']
                })
            elif obj['class'] == 'mug':
                affordances.append({
                    'type': 'grasp',
                    'object': obj['id'],
                    'location': obj['position']
                })
            # Add more affordances based on object types
        
        return affordances
```

## Audio Processing and Natural Language

### Audio Signal Processing

Humanoid robots need sophisticated audio processing:

```python
import librosa
import numpy as np
from scipy import signal

class AudioProcessingModule:
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = 1024
        self.hop_length = 512
        
    def preprocess_audio(self, audio_data):
        """Preprocess raw audio for higher-level processing"""
        # Noise reduction
        denoised_audio = self.reduce_noise(audio_data)
        
        # Voice activity detection
        vad_segments = self.detect_voice_activity(denoised_audio)
        
        # Feature extraction
        mfcc_features = librosa.feature.mfcc(
            y=denoised_audio, 
            sr=self.sample_rate, 
            n_mfcc=13
        )
        
        return {
            'audio_clean': denoised_audio,
            'vad_segments': vad_segments,
            'mfcc': mfcc_features
        }
        
    def reduce_noise(self, audio_data):
        """Apply noise reduction techniques"""
        # Use spectral subtraction or other noise reduction methods
        return audio_data  # Simplified
        
    def detect_voice_activity(self, audio_data):
        """Detect segments containing speech"""
        # Implementation using energy-based or ML-based VAD
        return [0, len(audio_data)]  # Simplified
```

### Speech Recognition for Humanoid Interaction

```python
import speech_recognition as sr

class RobotSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
        # Language model for humanoid interaction
        self.interaction_grammar = self.load_interaction_grammar()
        
    def recognize_speech(self, audio_data):
        """Recognize speech and extract meaning for humanoid interaction"""
        try:
            # Convert to text
            text = self.recognizer.recognize_google(audio_data)
            
            # Parse for interaction intent
            intent = self.parse_interaction_intent(text)
            
            # Extract named entities
            entities = self.extract_entities(text)
            
            return {
                'text': text,
                'intent': intent,
                'entities': entities,
                'confidence': 1.0  # Would use actual confidence score
            }
        except sr.UnknownValueError:
            return {'error': 'Could not understand audio'}
        except sr.RequestError as e:
            return {'error': f'Service error: {e}'}
            
    def parse_interaction_intent(self, text):
        """Parse text for interaction intent"""
        # Use NLU model to determine intent
        # Examples: "move forward", "wave hello", "pick up object", etc.
        return self.classify_intent(text)
        
    def extract_entities(self, text):
        """Extract named entities relevant to humanoid interaction"""
        # Extract objects, locations, people, etc.
        return self.entity_recognizer(text)
```

### Natural Language Understanding

```python
import spacy
import transformers
from transformers import pipeline

class NaturalLanguageUnderstanding:
    def __init__(self):
        # Load spaCy model for linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load transformer model for deeper understanding
        self.qa_pipeline = pipeline("question-answering")
        
    def understand_utterance(self, text, context=None):
        """Comprehend human utterance in context"""
        # Linguistic analysis
        doc = self.nlp(text)
        
        # Extract linguistic features
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Named entity recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Coreference resolution
        coreferences = self.resolve_coreferences(doc)
        
        # Dialogue act classification
        dialogue_act = self.classify_dialogue_act(text)
        
        # Intent classification for robot action
        robot_intent = self.classify_robot_intent(text)
        
        return {
            'linguistic_analysis': {
                'tokens': tokens,
                'pos_tags': pos_tags,
                'dependencies': dependencies,
                'entities': entities
            },
            'coreferences': coreferences,
            'dialogue_act': dialogue_act,
            'robot_intent': robot_intent
        }
        
    def classify_robot_intent(self, text):
        """Classify text for appropriate robot response"""
        # Map utterance to robot behavior
        # Examples: GREETING, INSTRUCTION, QUESTION, EMOTIONAL_STATE
        return self.intent_classifier(text)
```

## Tactile and Proprioceptive Sensing

### Tactile Perception

Tactile sensing enables manipulation and social interaction:

```python
class TactilePerceptionSystem:
    def __init__(self, tactile_sensor_config):
        self.sensor_grid = tactile_sensor_config['grid']
        self.pressure_threshold = tactile_sensor_config['pressure_threshold']
        self.temporal_window = tactile_sensor_config['temporal_window']
        
    def process_tactile_data(self, tactile_readings):
        """Process tactile sensor array data"""
        # Detect contact points
        contact_points = self.detect_contact(tactile_readings)
        
        # Classify contact type (grasp, touch, pat, etc.)
        contact_type = self.classify_contact(tactile_readings, contact_points)
        
        # Estimate object properties
        object_properties = self.estimate_object_properties(tactile_readings)
        
        return {
            'contact_points': contact_points,
            'contact_type': contact_type,
            'object_properties': object_properties
        }
        
    def detect_contact(self, readings):
        """Detect contact points from tactile sensor readings"""
        contacts = []
        for i, row in enumerate(readings):
            for j, pressure in enumerate(row):
                if pressure > self.pressure_threshold:
                    contacts.append((i, j, pressure))
        return contacts
        
    def classify_contact(self, readings, contacts):
        """Classify type of contact"""
        if len(contacts) == 0:
            return 'none'
            
        # Analyze pressure distribution and temporal patterns
        pressure_pattern = self.analyze_pressure_pattern(readings)
        temporal_pattern = self.analyze_temporal_pattern(readings)
        
        # Classify based on patterns
        if pressure_pattern['max'] > 0.9 and len(contacts) < 5:
            return 'tap'
        elif temporal_pattern['slope'] > 0.7:
            return 'grip'
        elif len(contacts) > 10 and pressure_pattern['mean'] < 0.3:
            return 'handshake'
        else:
            return 'touch'
```

### Proprioceptive Processing

Understanding the robot's own body state:

```python
class ProprioceptiveProcessor:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.joint_limits = robot_model.joint_limits
        self.mass_properties = robot_model.mass_properties
        
    def process_proprioceptive_data(self, joint_positions, joint_velocities, joint_torques):
        """Process proprioceptive sensor data"""
        # Calculate current pose
        current_pose = self.calculate_forward_kinematics(joint_positions)
        
        # Check safety constraints
        safety_status = self.check_safety_constraints(
            joint_positions, joint_velocities, joint_torques
        )
        
        # Estimate balance state
        balance_state = self.estimate_balance_state(
            joint_positions, joint_torques, current_pose
        )
        
        # Calculate center of mass
        com_position = self.calculate_center_of_mass(joint_positions)
        
        return {
            'pose': current_pose,
            'safety_status': safety_status,
            'balance_state': balance_state,
            'center_of_mass': com_position
        }
        
    def estimate_balance_state(self, joint_positions, joint_torques, current_pose):
        """Estimate robot's balance state"""
        # Calculate zero moment point (ZMP)
        zmp = self.calculate_zmp(joint_positions, joint_torques)
        
        # Calculate center of mass projection
        com_projection = self.project_com_to_ground(current_pose)
        
        # Define support polygon based on foot positions
        support_polygon = self.calculate_support_polygon()
        
        # Determine balance state
        if self.is_in_support_polygon(com_projection, support_polygon):
            balance_state = 'stable'
        elif self.is_near_support_polygon(com_projection, support_polygon):
            balance_state = 'cautious'
        else:
            balance_state = 'unstable'
            
        return balance_state
```

## Sensor Fusion Techniques

### Multi-Sensor Data Integration

Combining information from multiple sensors:

```python
class SensorFusionModule:
    def __init__(self):
        self.kalman_filters = {}
        self.particle_filters = {}
        
    def fuse_vision_audio(self, visual_data, audio_data):
        """Fuse visual and audio information"""
        # Track objects with both visual and audio cues
        fused_objects = self.track_objects_multimodal(visual_data, audio_data)
        
        # Locate sound sources in visual space
        sound_locations = self.localize_sounds(visual_data['camera_params'], audio_data)
        
        # Associate sounds with objects
        associations = self.associate_audio_with_objects(
            sound_locations, fused_objects
        )
        
        return {
            'fused_objects': fused_objects,
            'sound_locations': sound_locations,
            'associations': associations
        }
        
    def track_objects_multimodal(self, visual_data, audio_data):
        """Track objects using both visual and audio data"""
        # Initialize object tracks
        tracks = self.initialize_tracks(visual_data['detections'])
        
        # Update with visual data
        updated_tracks = self.update_with_vision(tracks, visual_data)
        
        # Update with audio data
        updated_tracks = self.update_with_audio(updated_tracks, audio_data)
        
        return updated_tracks
        
    def kalman_filter_fusion(self, sensor_measurements):
        """Fusion using Kalman filtering"""
        # Initialize Kalman filter if not already done
        if 'main' not in self.kalman_filters:
            self.kalman_filters['main'] = self.initialize_kalman_filter()
        
        # Predict state
        predicted_state = self.kalman_filters['main'].predict()
        
        # Update with each sensor measurement
        for sensor_id, measurement in sensor_measurements.items():
            predicted_state = self.kalman_filters['main'].update(measurement)
            
        return predicted_state
```

### Extended Kalman Filter for Humanoid Perception

```python
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State vector [x, y, z, vx, vy, vz, orientation, angular_rate]
        self.state = np.zeros(state_dim)
        
        # Error covariance matrix
        self.P = np.eye(state_dim) * 1000  # High initial uncertainty
        
        # Process noise
        self.Q = np.eye(state_dim) * 0.1
        
        # Measurement noise
        self.R_vision = np.eye(measurement_dim) * 0.01
        self.R_imu = np.eye(6) * 0.1  # [pos, orientation]
        
    def predict(self, dt, control_input=None):
        """Predict next state"""
        # State transition (simplified model)
        F = self.calculate_jacobian_f(self.state, dt)
        self.state = self.state_transition(self.state, dt, control_input)
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurement, sensor_type='vision'):
        """Update state with measurement"""
        # Calculate measurement Jacobian
        H = self.calculate_jacobian_h(self.state, sensor_type)
        
        # Calculate Kalman gain
        R = self.R_vision if sensor_type == 'vision' else self.R_imu
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        innovation = measurement - self.h(self.state, sensor_type)
        self.state = self.state + K @ innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
        
    def calculate_jacobian_f(self, state, dt):
        """Calculate the Jacobian of the state transition function"""
        # Linear approximation of state transition
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * dt  # Position from velocity
        return F
        
    def state_transition(self, state, dt, control_input):
        """Nonlinear state transition function"""
        new_state = state.copy()
        
        # Update positions based on velocities
        new_state[0:3] += state[3:6] * dt
        
        # Update velocities based on accelerations (from control or gravity)
        if control_input is not None:
            new_state[3:6] += control_input[:3] * dt
            
        return new_state
```

### Particle Filtering for Non-linear Systems

```python
class ParticleFilter:
    def __init__(self, state_dim, num_particles=1000):
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.particles = np.random.normal(0, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        
    def predict(self, control, noise_std):
        """Predict next state of particles"""
        # Apply control with noise
        noise = np.random.normal(0, noise_std, (self.num_particles, self.state_dim))
        self.particles += control + noise
        
    def update(self, measurement, measurement_function, measurement_noise):
        """Update particle weights based on measurement"""
        # Calculate likelihood of measurement for each particle
        for i, particle in enumerate(self.particles):
            predicted_measurement = measurement_function(particle)
            likelihood = self.calculate_likelihood(
                measurement, predicted_measurement, measurement_noise
            )
            self.weights[i] *= likelihood
            
        # Normalize weights
        self.weights += 1e-300  # Avoid numerical issues
        self.weights /= np.sum(self.weights)
        
    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = self.systematic_resample()
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)
        
    def estimate(self):
        """Estimate state from particles"""
        # Weighted mean of particles
        return np.average(self.particles, axis=0, weights=self.weights)
        
    def systematic_resample(self):
        """Systematic resampling algorithm"""
        indices = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        
        u = np.random.uniform(0, 1/self.num_particles)
        i, j = 0, 0
        
        while i < self.num_particles:
            while cumulative_sum[j] < u:
                j += 1
            indices[i] = j
            u += 1/self.num_particles
            i += 1
            
        return indices
```

## Real-time Perception Challenges

### Computational Optimization

Real-time perception requires careful optimization:

```python
class OptimizedPerceptionPipeline:
    def __init__(self):
        self.models = self.load_optimized_models()
        self.execution_plan = self.create_execution_plan()
        
    def load_optimized_models(self):
        """Load quantized and optimized models for real-time execution"""
        # Load models optimized for the target hardware
        models = {}
        
        # Vision model optimized for edge deployment
        models['vision'] = self.load_quantized_model('optimized_vision.onnx')
        
        # Audio model optimized for low latency
        models['audio'] = self.load_quantized_model('optimized_audio.onnx')
        
        return models
        
    def create_execution_plan(self):
        """Create schedule for processing tasks"""
        # Define pipeline stages
        stages = [
            {'name': 'preprocessing', 'period': 0.01, 'deadline': 0.005},
            {'name': 'vision', 'period': 0.033, 'deadline': 0.025},  # 30 FPS
            {'name': 'audio', 'period': 0.0625, 'deadline': 0.05},  # 16 FPS
            {'name': 'fusion', 'period': 0.1, 'deadline': 0.08},  # 10 FPS
        ]
        
        return stages
        
    def run_pipeline(self):
        """Run perception pipeline with real-time constraints"""
        import time
        
        while True:
            start_time = time.time()
            
            # Preprocess data
            if self.should_execute('preprocessing', start_time):
                preprocessed_data = self.preprocess_sensors()
                
            # Run vision processing
            if self.should_execute('vision', start_time):
                vision_output = self.run_vision(preprocessed_data['camera'])
                
            # Run audio processing
            if self.should_execute('audio', start_time):
                audio_output = self.run_audio(preprocessed_data['microphones'])
                
            # Run fusion
            if self.should_execute('fusion', start_time):
                fused_output = self.fuse_modalities(vision_output, audio_output)
                
            # Maintain timing
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.01 - elapsed)  # 100Hz minimum
            time.sleep(sleep_time)
```

### Adaptive Perception

Adjusting perception based on context and performance:

```python
class AdaptivePerceptionSystem:
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.current_mode = 'normal'
        
    def adapt_to_context(self, context_features):
        """Adapt perception parameters based on current context"""
        # Determine appropriate perception mode
        if context_features['crowd_density'] > 0.5:
            # High density - focus on collision avoidance
            self.set_mode('navigation')
            self.increase_vision_frequency()
        elif context_features['interaction_level'] > 0.7:
            # High interaction - focus on social perception
            self.set_mode('social')
            self.increase_audio_frequency()
        else:
            # Normal mode
            self.set_mode('normal')
            
    def adapt_to_performance(self, computational_load):
        """Adapt perception based on computational constraints"""
        if computational_load > 0.8:
            # Reduce resolution to meet timing constraints
            self.reduce_vision_resolution()
            self.simplify_audio_processing()
        elif computational_load < 0.5:
            # Increase resolution if resources available
            self.increase_vision_resolution()
            self.enhance_audio_processing()
            
    def set_mode(self, mode):
        """Set perception system to specific mode"""
        if mode != self.current_mode:
            if mode == 'navigation':
                # Prioritize obstacle detection and collision avoidance
                self.models['vision'].set_task('detection')
                self.vision_frequency = 60  # Higher for safety
            elif mode == 'social':
                # Prioritize face detection and emotion recognition
                self.models['vision'].set_task('social')
                self.audio_frequency = 50  # Higher for interaction
            elif mode == 'normal':
                # Balanced processing
                self.models['vision'].set_task('general')
                self.vision_frequency = 30
                self.audio_frequency = 16
                
            self.current_mode = mode
```

## Learning from Human Demonstrations

### Imitation Learning for Perception Tasks

Humanoid robots can learn perception capabilities through imitation:

```python
class ImitationPerceptionLearner:
    def __init__(self):
        self.perception_policy = self.initialize_policy_network()
        self.demonstration_buffer = []
        
    def learn_from_demonstration(self, expert_trajectory):
        """Learn perception mappings from expert demonstration"""
        # Extract state-action pairs from demonstration
        states, actions = self.extract_pairs(expert_trajectory)
        
        # Add to demonstration buffer
        self.demonstration_buffer.extend(zip(states, actions))
        
        # Train perception policy
        self.train_policy()
        
    def extract_pairs(self, trajectory):
        """Extract state-action pairs from demonstration trajectory"""
        states = []
        actions = []
        
        for t in range(len(trajectory)-1):
            current_state = trajectory[t]['sensor_data']
            next_state = trajectory[t+1]['sensor_data']
            action = trajectory[t]['action']
            
            # State could be: [current_vision, current_audio, proprioception]
            state = self.encode_state(current_state)
            
            states.append(state)
            actions.append(action)
            
        return np.array(states), np.array(actions)
        
    def encode_state(self, sensor_data):
        """Encode sensor data into state representation"""
        # Combine different modalities
        vision_features = self.vision_encoder(sensor_data['camera'])
        audio_features = self.audio_encoder(sensor_data['microphones'])
        proprio_features = sensor_data['proprioception'].flatten()
        
        # Concatenate into single state vector
        return np.concatenate([vision_features, audio_features, proprio_features])
        
    def train_policy(self):
        """Train perception policy using behavior cloning"""
        if len(self.demonstration_buffer) < 100:
            return  # Not enough data yet
            
        # Separate states and actions
        states, actions = zip(*self.demonstration_buffer)
        states = np.array(states)
        actions = np.array(actions)
        
        # Train the policy network
        self.perception_policy.fit(states, actions, epochs=50)
```

### Active Learning for Perception

```python
class ActivePerceptionLearner:
    def __init__(self, initial_model):
        self.model = initial_model
        self.uncertainty_threshold = 0.3
        self.query_strategy = 'uncertainty_sampling'
        
    def select_informative_samples(self, unlabeled_data):
        """Select most informative samples for labeling"""
        uncertainties = []
        
        for data_point in unlabeled_data:
            # Get prediction and uncertainty
            prediction, uncertainty = self.model.predict_with_uncertainty(data_point)
            uncertainties.append(uncertainty)
            
        # Sort by uncertainty (highest first)
        sorted_indices = np.argsort(uncertainties)[::-1]
        
        # Select top N most uncertain samples
        num_to_select = min(10, len(unlabeled_data) // 10)  # 10% of data
        selected_indices = sorted_indices[:num_to_select]
        
        return [unlabeled_data[i] for i in selected_indices], selected_indices
        
    def improve_perception_with_user(self, raw_sensor_data):
        """Request user feedback on uncertain perceptions"""
        # Process sensor data
        perception_result = self.model.predict(raw_sensor_data)
        uncertainty = self.model.estimate_uncertainty(raw_sensor_data)
        
        if uncertainty > self.uncertainty_threshold:
            # Request user confirmation
            user_feedback = self.request_user_feedback(
                perception_result, raw_sensor_data
            )
            
            # Update model with new labeled data
            self.model.update_with_feedback(
                raw_sensor_data, user_feedback, uncertainty
            )
            
        return perception_result
        
    def request_user_feedback(self, perception_result, sensor_data):
        """Request user confirmation of perception"""
        # In a real system, this might show data to a human operator
        # or present options for the human to select
        return self.simulate_user_feedback(perception_result, sensor_data)
```

## Perception in Dynamic Environments

### Dynamic Object Tracking

```python
class DynamicScenePerceptor:
    def __init__(self):
        self.object_trackers = {}
        self.tracking_history = {}
        self.motion_models = {}
        
    def track_dynamic_objects(self, sensor_data):
        """Track objects that are moving in the environment"""
        # Get current detections
        current_detections = self.detect_objects(sensor_data['camera'])
        
        # Update existing tracks
        updated_tracks = self.update_existing_tracks(current_detections)
        
        # Initialize new tracks
        new_tracks = self.initialize_new_tracks(current_detections, updated_tracks)
        
        # Predict future positions
        predicted_positions = self.predict_motion(updated_tracks)
        
        return {
            'tracked_objects': updated_tracks,
            'predicted_motion': predicted_positions,
            'motion_vectors': self.compute_motion_vectors(updated_tracks)
        }
        
    def update_existing_tracks(self, detections):
        """Update existing object tracks with new detections"""
        updated_tracks = {}
        
        for obj_id, track in self.object_trackers.items():
            # Find best matching detection
            best_match = self.find_best_match(track, detections)
            
            if best_match is not None:
                # Update track with new measurement
                track['state'] = self.update_kalman(track['filter'], best_match)
                track['last_seen'] = time.time()
                
                # Update motion model
                self.update_motion_model(obj_id, track['state'])
                
                updated_tracks[obj_id] = track
            elif time.time() - track['last_seen'] < 2.0:  # 2 seconds
                # Object not detected but may be temporarily occluded
                # Use prediction
                track['state'] = self.predict_kalman(track['filter'])
                updated_tracks[obj_id] = track
            # Else: object is considered lost
        
        return updated_tracks
        
    def predict_motion(self, tracks):
        """Predict future positions of tracked objects"""
        predictions = {}
        
        for obj_id, track in tracks.items():
            # Use learned motion model to predict future path
            motion_model = self.motion_models.get(obj_id, 'constant_velocity')
            
            if motion_model == 'constant_velocity':
                prediction = self.constant_velocity_predict(track['state'])
            elif motion_model == 'constant_acceleration':
                prediction = self.constant_acceleration_predict(track['state'])
            else:
                prediction = self.learning_based_predict(motion_model, track['state'])
                
            predictions[obj_id] = prediction
            
        return predictions
```

### Change Detection and Adaptation

```python
class ChangeDetectionSystem:
    def __init__(self):
        self.background_model = None
        self.environment_map = None
        self.change_threshold = 0.1
        
    def detect_environment_changes(self, current_sensors):
        """Detect changes in the environment"""
        # Update background model
        if self.background_model is None:
            self.background_model = self.initialize_background(current_sensors)
        
        # Compare with stored environment map
        changes = self.compare_with_map(current_sensors)
        
        # Classify types of changes
        change_types = self.classify_changes(changes)
        
        # Update internal map
        self.update_environment_map(current_sensors, changes)
        
        return {
            'changes_detected': len(changes) > 0,
            'change_types': change_types,
            'change_locations': [c['location'] for c in changes]
        }
        
    def compare_with_map(self, current_sensors):
        """Compare current sensor data with stored map"""
        changes = []
        
        for sensor_type, data in current_sensors.items():
            if sensor_type == 'camera':
                visual_changes = self.compare_visual_map(data)
                changes.extend(visual_changes)
            elif sensor_type == 'depth':
                depth_changes = self.compare_depth_map(data)
                changes.extend(depth_changes)
                
        return changes
        
    def classify_changes(self, changes):
        """Classify detected changes"""
        change_classes = {
            'appearance': [],
            'disappearance': [],
            'movement': [],
            'modification': []
        }
        
        for change in changes:
            # Classify based on nature of change
            if change['type'] == 'new_object':
                change_classes['appearance'].append(change)
            elif change['type'] == 'object_removed':
                change_classes['disappearance'].append(change)
            elif change['type'] == 'object_moved':
                change_classes['movement'].append(change)
            else:
                change_classes['modification'].append(change)
                
        return change_classes
```

## Summary

This chapter has explored how machine learning enables humanoid robots to perceive and understand their environment. We've covered the unique challenges of perception in humanoid robotics, including the need for real-time processing, multi-modal integration, and human-like interpretation of sensory information.

We've examined the implementation of vision systems using deep learning, including object detection, pose estimation, and scene understanding. Audio processing and natural language understanding were discussed, showing how humanoid robots process speech and extract meaning for interaction.

The chapter also covered tactile and proprioceptive sensing, which are crucial for manipulation and balance in humanoid robots. Sensor fusion techniques, including Kalman filtering and particle filtering, were presented as ways to combine information from multiple sensors into coherent perceptions.

Real-time perception challenges were addressed, including computational optimization and adaptive perception systems. The chapter concluded with discussions on learning from human demonstrations and handling dynamic environments.

## Exercises

1. Implement a simple sensor fusion system that combines data from a camera and IMU to track an object in 3D space, using either Kalman filtering or particle filtering.

2. Design a perception pipeline for a humanoid robot that needs to recognize human emotions and respond appropriately. Consider the different modalities needed and how they would be integrated.

3. Create an active learning system for a humanoid robot's perception that identifies when its confidence is low and requests human confirmation to improve its models.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*