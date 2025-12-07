---
id: module-04-chapter-03
title: Chapter 03 - Tactile and Proprioceptive Sensing
sidebar_position: 15
---

# Chapter 03 - Tactile and Proprioceptive Sensing

## Table of Contents
- [Overview](#overview)
- [Tactile Sensing Technologies](#tactile-sensing-technologies)
- [Tactile Sensor Arrays](#tactile-sensor-arrays)
- [Tactile Processing and Interpretation](#tactile-processing-and-interpretation)
- [Proprioceptive Sensing](#proprioceptive-sensing)
- [Joint Position and Velocity Sensing](#joint-position-and-velocity-sensing)
- [Inertial Measurement Units (IMUs)]#inertial-measurement-units-imus)
- [Force and Torque Sensing](#force-and-torque-sensing)
- [Sensor Fusion for Proprioception](#sensor-fusion-for-proprioception)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Tactile and proprioceptive sensing form the foundation of physical awareness in humanoid robots, enabling them to perceive their own body state and interact with objects through touch. Unlike traditional robots that rely primarily on visual feedback, humanoid robots must have sophisticated tactile sensing capabilities to perform human-like manipulation tasks and maintain balance. This chapter explores the technologies, implementation, and integration of tactile and proprioceptive sensors in humanoid robotics.

The integration of tactile and proprioceptive senses is essential for safe and effective human-robot interaction. These senses enable robots to perceive contact forces, object properties, and their own body configuration, allowing for dexterous manipulation and stable locomotion. Understanding how to effectively implement and process these sensory modalities is crucial for developing capable humanoid robots.

## Tactile Sensing Technologies

### Resistive Tactile Sensors

Resistive tactile sensors are widely used in humanoid robotics due to their simplicity and reliability:

```python
class ResistiveTactileSensor:
    def __init__(self, rows=8, cols=8, pressure_range=(0, 200)):
        self.rows = rows
        self.cols = cols
        self.pressure_range = pressure_range  # range in kPa
        self.grid = np.zeros((rows, cols))
        self.calibration_data = None
        
    def read_tactile_data(self):
        """Read raw resistive values from the sensor array"""
        # Simulate reading from hardware
        raw_readings = self.simulate_hardware_read()
        
        # Apply calibration
        calibrated_values = self.apply_calibration(raw_readings)
        
        # Convert to pressure values
        pressure_map = self.convert_to_pressure(calibrated_values)
        
        return pressure_map
        
    def simulate_hardware_read(self):
        """Simulate reading from resistive sensor array"""
        # In real implementation, this would interface with actual hardware
        raw_values = np.random.rand(self.rows, self.cols) * 1024  # 10-bit ADC values
        return raw_values
        
    def apply_calibration(self, raw_values):
        """Apply calibration parameters to raw sensor readings"""
        if self.calibration_data is None:
            self.calibrate_sensor()
            
        # Apply linear calibration: calibrated = gain * raw + offset
        calibrated = (raw_values * self.calibration_data['gain'] + 
                     self.calibration_data['offset'])
        return np.clip(calibrated, 0, 1)  # Ensure values are between 0 and 1
        
    def convert_to_pressure(self, calibrated_values):
        """Convert calibrated values to pressure values"""
        # Convert normalized values to actual pressure
        pressure = calibrated_values * (self.pressure_range[1] - self.pressure_range[0]) + self.pressure_range[0]
        return pressure
        
    def calibrate_sensor(self):
        """Calibrate the tactile sensor"""
        # Apply known pressures and record responses
        print("Starting tactile sensor calibration...")
        
        # Apply minimum and maximum pressures to determine gain and offset
        min_response = self.simulate_hardware_read()  # With no pressure
        max_response = self.simulate_hardware_read()  # With maximum pressure (simulated)
        
        # Calculate calibration parameters
        max_adc = 1024  # 10-bit ADC
        gain = max_adc / (self.pressure_range[1] - self.pressure_range[0])
        offset = -self.pressure_range[0] * gain
        
        self.calibration_data = {
            'gain': 1.0 / gain if gain != 0 else 1.0,
            'offset': -offset * (1.0 / gain if gain != 0 else 1.0),
            'max_adc': max_adc
        }
        
        print("Calibration complete.")
```

### Capacitive Tactile Sensors

Capacitive sensors offer high sensitivity and are commonly used in advanced humanoid robots:

```python
class CapacitiveTactileSensor:
    def __init__(self, resolution=(16, 16)):
        self.cols = resolution[0]
        self.rows = resolution[1]
        self.baseline_capacitance = np.ones((self.rows, self.cols)) * 100  # pF
        self.pressure_sensitivity = 0.1  # pF per kPa
        self.grid = np.zeros((self.rows, self.cols))
        
    def read_tactile_data(self):
        """Read capacitive values from the sensor array"""
        # Simulate capacitive measurements
        capacitance_map = self.measure_capacitance()
        
        # Convert capacitance to pressure
        pressure_map = self.capacitance_to_pressure(capacitance_map)
        
        return pressure_map
        
    def measure_capacitance(self):
        """Measure capacitance values at each sensor element"""
        # In real implementation, this would interact with hardware
        # For simulation, add baseline with pressure-induced changes
        baseline = self.baseline_capacitance.copy()
        
        # Simulate contact by reducing capacitance (pressure increases contact area)
        contact_pressure = np.random.rand(self.rows, self.cols) * 50  # Simulate pressure up to 50 kPa
        capacitance_change = contact_pressure * self.pressure_sensitivity
        
        capacitance_map = baseline - capacitance_change  # Capacitance typically decreases with pressure
        
        return np.clip(capacitance_map, 0, baseline*2)  # Ensure valid range
        
    def capacitance_to_pressure(self, capacitance_map):
        """Convert capacitance measurements to pressure estimates"""
        # Convert the change in capacitance to pressure
        baseline = self.baseline_capacitance
        delta_capacitance = baseline - capacitance_map
        
        pressure_map = delta_capacitance / self.pressure_sensitivity
        return pressure_map
        
    def detect_contact_events(self, current_data, previous_data, threshold=1.0):
        """Detect contact and release events"""
        delta = current_data - previous_data
        
        contact_events = np.where(delta > threshold)
        release_events = np.where(delta < -threshold)
        
        return {
            'contacts': list(zip(contact_events[0], contact_events[1])),
            'releases': list(zip(release_events[0], release_events[1]))
        }
```

### Vision-Based Tactile Sensors

Vision-based tactile sensors like GelSight provide high-resolution tactile information:

```python
class VisionBasedTactileSensor:
    def __init__(self, camera_config):
        self.camera = self.initialize_camera(camera_config)
        self.gel_reference = None  # Reference image of undistorted gel
        self.marker_positions = {}  # Positions of internal markers
        
    def initialize_camera(self, config):
        """Initialize camera for tactile sensing"""
        # This would typically interface with a real camera
        # For simulation, we'll just store the configuration
        return {
            'resolution': config['resolution'],
            'fps': config['fps'],
            'sensor_type': 'vision_tactile'
        }
        
    def capture_tactile_image(self):
        """Capture tactile image from the sensor"""
        # In a real implementation, this would capture from the camera
        # For simulation, we'll generate a synthetic tactile image
        height, width = self.camera['resolution']
        
        # Create a base texture that simulates the tactile gel
        base_image = np.random.rand(height, width) * 255
        
        # Simulate an object pressing into the gel
        object_size = np.random.randint(20, 50)
        center_x = np.random.randint(object_size, width - object_size)
        center_y = np.random.randint(object_size, height - object_size)
        
        # Create a circular depression in the gel
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= object_size**2
        base_image[mask] = 100  # Darker where pressed
        
        # Add some texture distortion to simulate the effect of pressure
        distortion = np.random.rand(height, width) * 50
        base_image = np.clip(base_image + distortion, 0, 255)
        
        return base_image.astype(np.uint8)
        
    def estimate_shape_and_pressure(self, tactile_image):
        """Estimate shape and pressure from tactile image"""
        if self.gel_reference is None:
            self.gel_reference = tactile_image
            return np.zeros_like(tactile_image, dtype=np.float32)
            
        # Compute difference from reference
        difference = cv2.absdiff(tactile_image, self.gel_reference)
        
        # Enhance the tactile features
        enhanced = cv2.bilateralFilter(difference, 9, 75, 75)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Convert to pressure estimates (this is a simplified model)
        pressure_map = enhanced.astype(np.float32) / 255.0
        
        return pressure_map
        
    def detect_slip(self, tactile_images_sequence):
        """Detect slip based on changes in tactile image patterns"""
        if len(tactile_images_sequence) < 2:
            return False
            
        # Compare consecutive images to detect rapid changes (indicative of slip)
        prev_img = tactile_images_sequence[-2]
        curr_img = tactile_images_sequence[-1]
        
        # Calculate difference
        diff = cv2.absdiff(prev_img, curr_img)
        diff_norm = np.linalg.norm(diff)
        
        # Empirically determined threshold for slip detection
        slip_threshold = 5000  # This value would need to be tuned
        
        return diff_norm > slip_threshold
```

### Tactile Sensor Integration in Humanoid Systems

```python
class TactileIntegrationSystem:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.finger_sensors = self.setup_finger_tactile_sensors()
        self.palm_sensors = self.setup_palm_tactile_sensors()
        self.foot_sensors = self.setup_foot_tactile_sensors()
        
    def setup_finger_tactile_sensors(self):
        """Set up tactile sensors on robot fingers"""
        sensors = {}
        
        for hand in ['left', 'right']:
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                # High-resolution tactile sensors for manipulation
                sensors[f'{hand}_{finger}_tip'] = ResistiveTactileSensor(
                    rows=16, cols=16, pressure_range=(0, 100)
                )
                
                # Lower resolution for finger pads
                sensors[f'{hand}_{finger}_pad'] = CapacitiveTactileSensor(
                    resolution=(8, 8)
                )
                
        return sensors
        
    def setup_palm_tactile_sensors(self):
        """Set up tactile sensors on robot palms"""
        sensors = {}
        
        for hand in ['left', 'right']:
            # Large area sensor for palm contact detection
            sensors[f'{hand}_palm'] = ResistiveTactileSensor(
                rows=20, cols=20, pressure_range=(0, 50)
            )
            
        return sensors
        
    def setup_foot_tactile_sensors(self):
        """Set up tactile sensors on robot feet for balance"""
        sensors = {}
        
        for foot in ['left', 'right']:
            # High-resolution sensor for precise balance control
            sensors[f'{foot}_sole'] = CapacitiveTactileSensor(
                resolution=(32, 24)
            )
            
            # Additional sensors on heel and toe
            sensors[f'{foot}_heel'] = ResistiveTactileSensor(
                rows=4, cols=4, pressure_range=(0, 200)
            )
            sensors[f'{foot}_toe'] = ResistiveTactileSensor(
                rows=4, cols=4, pressure_range=(0, 200)
            )
            
        return sensors
        
    def process_tactile_frame(self):
        """Process all tactile data from the robot"""
        tactile_data = {}
        
        # Read all finger sensors
        for name, sensor in self.finger_sensors.items():
            tactile_data[name] = sensor.read_tactile_data()
            
        # Read all palm sensors
        for name, sensor in self.palm_sensors.items():
            tactile_data[name] = sensor.read_tactile_data()
            
        # Read all foot sensors
        for name, sensor in self.foot_sensors.items():
            tactile_data[name] = sensor.read_tactile_data()
            
        return tactile_data
        
    def detect_contacts(self, tactile_data, threshold=5.0):
        """Detect contact points across all tactile sensors"""
        contacts = {}
        
        for sensor_name, pressure_map in tactile_data.items():
            # Find locations where pressure exceeds threshold
            contact_positions = np.where(pressure_map > threshold)
            
            if len(contact_positions[0]) > 0:
                # Get pressure values at contact points
                contact_pressures = pressure_map[contact_positions]
                
                contacts[sensor_name] = {
                    'positions': list(zip(contact_positions[0], contact_positions[1])),
                    'pressures': contact_pressures.tolist(),
                    'total_force': np.sum(contact_pressures),
                    'contact_count': len(contact_pressures)
                }
                
        return contacts
```

## Tactile Sensor Arrays

### High-Density Tactile Arrays

```python
class HighDensityTactileArray:
    def __init__(self, rows=32, cols=32, sensor_type='resistive'):
        self.rows = rows
        self.cols = cols
        self.sensor_type = sensor_type
        self.array = np.zeros((rows, cols))
        self.data_buffer = np.zeros((rows, cols, 10))  # Buffer for temporal information
        
    def update_reading(self, new_data):
        """Update tactile array reading"""
        # Shift buffer and add new reading
        self.data_buffer = np.roll(self.data_buffer, shift=-1, axis=2)
        self.data_buffer[:, :, -1] = new_data
        
        self.array = new_data  # Update current reading
        
    def compute_temporal_features(self):
        """Compute temporal features from buffered data"""
        temporal_features = np.zeros((self.rows, self.cols, 4))  # [velocity, acceleration, variance, mean]
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Extract time series for this sensor element
                time_series = self.data_buffer[i, j, :]
                
                # Compute velocity (first derivative)
                velocity = np.gradient(time_series)
                
                # Compute acceleration (second derivative)
                acceleration = np.gradient(velocity)
                
                # Compute statistical features
                variance = np.var(time_series)
                mean = np.mean(time_series)
                
                temporal_features[i, j] = [
                    np.mean(velocity),    # Average velocity
                    np.mean(acceleration), # Average acceleration
                    variance,             # Variability
                    mean                  # Average pressure
                ]
                
        return temporal_features
        
    def detect_events(self, temporal_features, threshold=0.1):
        """Detect tactile events based on temporal features"""
        events = {
            'contact_onsets': [],
            'contact_offsets': [],
            'slip_events': [],
            'vibration_events': []
        }
        
        for i in range(self.rows):
            for j in range(self.cols):
                velocity = temporal_features[i, j, 0]
                acceleration = temporal_features[i, j, 1]
                variance = temporal_features[i, j, 2]
                
                # Contact onset: rapid pressure increase
                if velocity > threshold and self.array[i, j] > 0.1:
                    events['contact_onsets'].append((i, j))
                    
                # Contact offset: rapid pressure decrease
                elif velocity < -threshold and self.array[i, j] < 0.1:
                    events['contact_offsets'].append((i, j))
                    
                # Slip detection: high acceleration in shear directions
                elif abs(acceleration) > 0.5:
                    events['slip_events'].append((i, j))
                    
                # Vibration: high variance
                elif variance > 0.01:
                    events['vibration_events'].append((i, j))
                    
        return events
```

### Distributed Tactile Processing

```python
class DistributedTactileProcessor:
    def __init__(self, sensor_layout):
        self.sensor_layout = sensor_layout
        self.processors = self.create_processors()
        
    def create_processors(self):
        """Create specialized processors for different tactile regions"""
        processors = {}
        
        for region, sensor_data in self.sensor_layout.items():
            if 'finger' in region:
                # Fingers need high-resolution processing for manipulation
                processors[region] = FingerTactileProcessor(sensor_data)
            elif 'palm' in region:
                # Palms need contact detection and force estimation
                processors[region] = PalmTactileProcessor(sensor_data)
            elif 'foot' in region:
                # Feet need balance and gait information
                processors[region] = FootTactileProcessor(sensor_data)
            else:
                # General purpose tactile processing
                processors[region] = GeneralTactileProcessor(sensor_data)
                
        return processors
        
    def process_all_regions(self, tactile_data):
        """Process tactile data for all regions"""
        results = {}
        
        for region, processor in self.processors.items():
            if region in tactile_data:
                results[region] = processor.process(tactile_data[region])
                
        return results

class FingerTactileProcessor:
    def __init__(self, sensor_config):
        self.config = sensor_config
        
    def process(self, tactile_map):
        """Process tactile data for finger manipulation"""
        # Detect contact locations
        contact_points = np.where(tactile_map > 0.1)  # Threshold for contact
        
        if len(contact_points[0]) == 0:
            return {'contact': False, 'slip_risk': 0, 'manipulation_readiness': 0}
            
        # Compute contact features
        contact_pressures = tactile_map[contact_points]
        
        return {
            'contact': True,
            'contact_points': list(zip(contact_points[0], contact_points[1])),
            'total_force': np.sum(contact_pressures),
            'pressure_cofm': self.compute_cofm(tactile_map),  # Center of pressure
            'slip_risk': self.estimate_slip_risk(tactile_map),
            'object_diameter': self.estimate_object_diameter(tactile_map),
            'friction_coeff': self.estimate_friction(tactile_map)
        }
        
    def compute_cofm(self, tactile_map):
        """Compute center of pressure for contact"""
        total_pressure = np.sum(tactile_map)
        if total_pressure == 0:
            return (0, 0)
            
        rows, cols = np.mgrid[0:tactile_map.shape[0], 0:tactile_map.shape[1]]
        weighted_row = np.sum(rows * tactile_map) / total_pressure
        weighted_col = np.sum(cols * tactile_map) / total_pressure
        
        return (weighted_row, weighted_col)
        
    def estimate_slip_risk(self, tactile_map):
        """Estimate risk of slip based on pressure distribution"""
        # Simplified slip risk based on pressure variance
        if np.sum(tactile_map) == 0:
            return 0
            
        pressure_variance = np.var(tactile_map[tactile_map > 0])
        normalized_variance = pressure_variance / np.mean(tactile_map[tactile_map > 0])
        
        # Higher variance indicates uneven pressure distribution, higher slip risk
        return min(normalized_variance, 1.0)

class PalmTactileProcessor:
    def __init__(self, sensor_config):
        self.config = sensor_config
        
    def process(self, tactile_map):
        """Process tactile data for palm contact"""
        # Find contact regions
        contact_mask = tactile_map > 0.05
        labeled_contacts, num_contacts = ndimage.label(contact_mask)
        
        contact_regions = []
        for i in range(1, num_contacts + 1):
            region_mask = (labeled_contacts == i)
            region_indices = np.where(region_mask)
            
            if len(region_indices[0]) > 5:  # Only consider regions with >5 active sensors
                region_pressures = tactile_map[region_mask]
                contact_regions.append({
                    'center_of_region': (
                        np.mean(region_indices[0]),
                        np.mean(region_indices[1])
                    ),
                    'total_pressure': np.sum(region_pressures),
                    'area': len(region_pressures),
                    'avg_pressure': np.mean(region_pressures)
                })
                
        return {
            'contact_regions': contact_regions,
            'num_contacts': len(contact_regions),
            'total_force': np.sum(tactile_map),
            'contact_distribution': self.analyze_contact_distribution(tactile_map)
        }
        
    def analyze_contact_distribution(self, tactile_map):
        """Analyze how contact pressure is distributed"""
        if np.sum(tactile_map) == 0:
            return {'concentration': 0, 'symmetry': 0}
            
        # Calculate pressure distribution metrics
        total_force = np.sum(tactile_map)
        max_pressure = np.max(tactile_map)
        concentration = max_pressure / total_force if total_force > 0 else 0
        
        # Analyze symmetry (for palm, check left-right balance)
        left_half = tactile_map[:, :tactile_map.shape[1]//2]
        right_half = tactile_map[:, tactile_map.shape[1]//2:]
        symmetry = 1 - abs(np.sum(left_half) - np.sum(right_half)) / total_force if total_force > 0 else 0
        
        return {
            'concentration': concentration,
            'symmetry': symmetry
        }
```

## Tactile Processing and Interpretation

### Object Property Estimation

```python
class TactileObjectPropertyEstimator:
    def __init__(self):
        self.object_classifier = self.train_object_classifier()
        self.material_classifier = self.train_material_classifier()
        
    def train_object_classifier(self):
        """Train classifier for object recognition through tactile sensing"""
        # In a real implementation, this would be a trained model
        # For simulation, we'll use a simple rule-based approach
        return lambda features: self.simple_object_classification(features)
        
    def train_material_classifier(self):
        """Train classifier for material recognition through tactile sensing"""
        # In a real implementation, this would be a trained model
        # For simulation, we'll use a simple rule-based approach
        return lambda features: self.simple_material_classification(features)
        
    def estimate_object_properties(self, tactile_data, contact_location):
        """Estimate properties of object being touched"""
        # Extract features from tactile data
        features = self.extract_tactile_features(tactile_data)
        
        # Classify object
        object_class = self.object_classifier(features)
        
        # Classify material
        material_class = self.material_classifier(features)
        
        # Estimate geometric properties
        geometric_props = self.estimate_geometric_properties(tactile_data)
        
        return {
            'object_class': object_class,
            'material_class': material_class,
            'geometric_properties': geometric_props,
            'estimated_weight': self.estimate_weight(tactile_data, geometric_props),
            'surface_properties': self.estimate_surface_properties(tactile_data)
        }
        
    def extract_tactile_features(self, tactile_map):
        """Extract features from tactile sensor data"""
        features = {}
        
        # Statistical features
        features['mean_pressure'] = np.mean(tactile_map)
        features['std_pressure'] = np.std(tactile_map)
        features['max_pressure'] = np.max(tactile_map)
        features['pressure_variance'] = np.var(tactile_map)
        
        # Spatial features
        features['contact_area'] = np.sum(tactile_map > 0.1)  # Number of active sensors
        features['pressure_distribution'] = self.compute_pressure_distribution(tactile_map)
        
        # Temporal features (if temporal data is available)
        features['temporal_changes'] = 0  # Placeholder for temporal features
        
        return features
        
    def compute_pressure_distribution(self, tactile_map):
        """Compute spatial distribution metrics of pressure"""
        if np.sum(tactile_map) == 0:
            return {'eccentricity': 0, 'orientation': 0}
            
        # Find center of pressure
        total_pressure = np.sum(tactile_map)
        y_coords, x_coords = np.mgrid[0:tactile_map.shape[0], 0:tactile_map.shape[1]]
        
        center_y = np.sum(y_coords * tactile_map) / total_pressure
        center_x = np.sum(x_coords * tactile_map) / total_pressure
        
        # Compute covariance matrix
        var_x = np.sum((x_coords - center_x)**2 * tactile_map) / total_pressure
        var_y = np.sum((y_coords - center_y)**2 * tactile_map) / total_pressure
        cov_xy = np.sum((x_coords - center_x) * (y_coords - center_y) * tactile_map) / total_pressure
        
        cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        
        # Eigenvalues to determine eccentricity
        eigenvals = np.linalg.eigvals(cov_matrix)
        if eigenvals[0] > 0:
            eccentricity = np.sqrt(1 - (min(eigenvals)/max(eigenvals))**2)
        else:
            eccentricity = 0
            
        # Orientation (angle of major axis)
        orientation = 0.5 * np.arctan2(2*cov_xy, var_x - var_y)
        
        return {
            'eccentricity': eccentricity,
            'orientation': orientation,
            'center': (center_x, center_y)
        }
        
    def estimate_geometric_properties(self, tactile_data):
        """Estimate geometric properties from tactile contact"""
        # Find contact points
        contact_mask = tactile_data > 0.1
        y_coords, x_coords = np.where(contact_mask)
        
        if len(x_coords) == 0:
            return {'diameter': 0, 'shape': 'unknown', 'size': 0}
            
        # Estimate size based on contact area
        contact_area = len(x_coords)
        
        # Estimate object diameter based on contact spread
        x_span = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
        y_span = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1
        
        # Combine to get overall size estimate
        estimated_diameter = np.sqrt(x_span * y_span) * 2  # Rough geometric estimate
        
        # Shape classification based on aspect ratio
        aspect_ratio = max(x_span, y_span) / min(x_span, y_span) if min(x_span, y_span) > 0 else 1
        
        if aspect_ratio > 3:
            shape = 'elongated'
        elif aspect_ratio < 1.5:
            shape = 'round'
        else:
            shape = 'intermediate'
            
        return {
            'diameter': estimated_diameter,
            'shape': shape,
            'size': contact_area,
            'aspect_ratio': aspect_ratio
        }
        
    def estimate_surface_properties(self, tactile_data):
        """Estimate surface properties from tactile data"""
        # Analyze pressure variation for texture estimation
        active_sensors = tactile_data[tactile_data > 0.1]
        
        if len(active_sensors) == 0:
            return {'roughness': 0, 'compliance': 0}
            
        # Estimate roughness from pressure variance
        pressure_variance = np.var(active_sensors)
        normalized_roughness = pressure_variance / np.mean(active_sensors) if np.mean(active_sensors) > 0 else 0
        
        # Estimate compliance (softness) from contact area vs. total force
        total_force = np.sum(active_sensors)
        contact_area = len(active_sensors)
        compliance = contact_area / total_force if total_force > 0 else 0
        
        return {
            'roughness': min(normalized_roughness, 1.0),
            'compliance': min(compliance, 1.0),
            'friction': self.estimate_friction_coefficient(tactile_data)
        }
        
    def estimate_friction_coefficient(self, tactile_data):
        """Estimate friction coefficient from tactile data"""
        # Simplified estimation based on pressure distribution
        # In real implementation, this would use slip detection
        if np.sum(tactile_data) == 0:
            return 0.1
            
        # Higher friction materials typically show more uniform pressure distribution
        cv = np.std(tactile_data) / np.mean(tactile_data) if np.mean(tactile_data) > 0 else 0
        friction_coeff = max(0.1, min(1.0, 1.0 - cv))  # More uniform = higher friction
        
        return friction_coeff
```

### Grasp Stability Assessment

```python
class GraspStabilityAssessor:
    def __init__(self):
        self.stability_threshold = 0.7  # Minimum stability score for stable grasp
        self.slip_detection_threshold = 0.8  # Threshold for slip detection
        
    def assess_grasp_stability(self, tactile_data, grasp_type='three_finger'):
        """Assess stability of a grasp based on tactile data"""
        stability_metrics = self.compute_stability_metrics(tactile_data, grasp_type)
        
        # Weight different metrics based on their importance
        stability_score = (
            0.3 * stability_metrics['force_distribution'] +
            0.25 * stability_metrics['contact_coverage'] +
            0.25 * stability_metrics['pressure_uniformity'] +
            0.2 * stability_metrics['friction_estimation']
        )
        
        return {
            'stability_score': stability_score,
            'is_stable': stability_score >= self.stability_threshold,
            'stability_metrics': stability_metrics,
            'recommended_action': self.get_recommended_action(stability_score, tactile_data)
        }
        
    def compute_stability_metrics(self, tactile_data, grasp_type='three_finger'):
        """Compute various metrics for grasp stability"""
        metrics = {}
        
        # 1. Force Distribution - how evenly forces are distributed among contact points
        contact_pressures = tactile_data[tactile_data > 0.05]
        if len(contact_pressures) > 1:
            pressure_cv = np.std(contact_pressures) / np.mean(contact_pressures)
            metrics['force_distribution'] = max(0, 1 - pressure_cv)  # More uniform = higher score
        else:
            metrics['force_distribution'] = 0 if len(contact_pressures) == 0 else 1
            
        # 2. Contact Coverage - how much of the sensor area is covered by contacts
        total_sensors = tactile_data.size
        active_sensors = np.sum(tactile_data > 0.05)
        metrics['contact_coverage'] = active_sensors / total_sensors
        
        # 3. Pressure Uniformity - how consistent the pressure is across contact points
        if len(contact_pressures) > 0:
            # Lower variance in pressure = more uniform grasp = more stable
            pressure_variance = np.var(contact_pressures)
            max_possible_variance = (np.max(contact_pressures) - np.min(contact_pressures))**2
            if max_possible_variance > 0:
                uniformity = 1 - (pressure_variance / max_possible_variance)
            else:
                uniformity = 1.0
            metrics['pressure_uniformity'] = uniformity
        else:
            metrics['pressure_uniformity'] = 0
            
        # 4. Friction Estimation - estimated friction at contact points
        avg_pressure = np.mean(contact_pressures) if len(contact_pressures) > 0 else 0
        # Higher pressure generally indicates better friction (to a point)
        metrics['friction_estimation'] = min(avg_pressure * 5, 1.0)  # Scaled to [0,1]
        
        return metrics
        
    def get_recommended_action(self, stability_score, tactile_data):
        """Get recommended action based on stability assessment"""
        if stability_score < 0.3:
            return "release_object"
        elif stability_score < 0.6:
            return "increase_grip_force"
        elif stability_score < self.stability_threshold:
            return "adjust_grip_position"
        else:
            return "maintain_current_grip"
            
    def detect_slip(self, tactile_sequence):
        """Detect slip based on temporal tactile patterns"""
        if len(tactile_sequence) < 2:
            return False, 0.0
            
        # Compare consecutive tactile images
        prev_tactile = tactile_sequence[-2]
        curr_tactile = tactile_sequence[-1]
        
        # Calculate change in tactile pattern
        delta_tactile = np.abs(curr_tactile - prev_tactile)
        
        # Detect rapid changes that indicate slip
        slip_score = np.mean(delta_tactile)  # Average change across all sensors
        
        # Use threshold to determine if slip is occurring
        is_slipping = slip_score > self.slip_detection_threshold
        return is_slipping, slip_score
```

## Proprioceptive Sensing

### Joint Position Sensing Fundamentals

```python
class JointPositionSensor:
    def __init__(self, joint_name, sensor_type='encoder', resolution=16):
        self.joint_name = joint_name
        self.sensor_type = sensor_type  # 'encoder', 'potentiometer', 'resolver'
        self.resolution = resolution  # bits for encoder
        self.max_position = 2 * np.pi  # radians for rotary joint
        self.min_position = -2 * np.pi
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0
        self.bias = 0.0  # Calibration bias
        self.noise_level = 0.001  # Radians of noise
        
    def read_position(self):
        """Read current position of the joint"""
        # Simulated reading with noise
        true_position = self.simulate_physical_position()
        noisy_reading = true_position + np.random.normal(0, self.noise_level)
        
        # Apply bias offset
        corrected_reading = noisy_reading + self.bias
        
        # Keep within limits
        corrected_reading = np.clip(corrected_reading, self.min_position, self.max_position)
        
        return corrected_reading
        
    def simulate_physical_position(self):
        """Simulate actual physical position of joint"""
        # In a real robot, this would interface with the physical sensor
        # For simulation, we'll return the last set position with some dynamics
        return self.position
        
    def calibrate_sensor(self, known_position):
        """Calibrate the sensor using a known position"""
        current_reading = self.read_position()
        self.bias = known_position - current_reading
        print(f"Joint {self.joint_name} calibrated with bias: {self.bias}")
        
    def update_state(self, new_position, new_velocity, new_effort):
        """Update joint state from control system"""
        self.position = new_position
        self.velocity = new_velocity
        self.effort = new_effort

class JointPositionSensorArray:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.joint_sensors = {}
        
        # Create sensor for each joint in the robot model
        for joint_name in robot_model.joint_names:
            joint_limits = robot_model.joint_limits[joint_name]
            sensor_type = robot_model.joint_sensor_types[joint_name]
            
            self.joint_sensors[joint_name] = JointPositionSensor(
                joint_name, sensor_type, resolution=16
            )
            
    def read_all_positions(self):
        """Read positions from all joint sensors"""
        positions = {}
        
        for joint_name, sensor in self.joint_sensors.items():
            positions[joint_name] = sensor.read_position()
            
        return positions
        
    def get_joint_state(self):
        """Get complete joint state (position, velocity, effort)"""
        joint_state = {
            'position': self.read_all_positions(),
            'velocity': self.estimate_velocities(),
            'effort': self.estimate_efforts()
        }
        
        return joint_state
        
    def estimate_velocities(self):
        """Estimate velocities (would use encoder differences in real implementation)"""
        velocities = {}
        
        # In a real implementation, this would use encoder differences
        # For simulation, we'll return zeros
        for joint_name in self.joint_sensors.keys():
            velocities[joint_name] = 0.0
            
        return velocities
        
    def estimate_efforts(self):
        """Estimate efforts/torques (would use motor current in real implementation)"""
        efforts = {}
        
        # In a real implementation, this would be based on motor current
        # For simulation, we'll return zeros
        for joint_name in self.joint_sensors.keys():
            efforts[joint_name] = 0.0
            
        return efforts
```

### Proprioceptive State Estimation

```python
class ProprioceptiveStateEstimator:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.kinematic_model = robot_model.kinematic_model
        self.state_estimate = np.zeros(robot_model.num_joints * 2)  # pos, vel for each joint
        self.state_covariance = np.eye(robot_model.num_joints * 2) * 0.1
        self.process_noise = np.eye(robot_model.num_joints * 2) * 0.01
        
    def update_from_sensors(self, joint_state):
        """Update state estimate using sensor readings"""
        # Extract positions and velocities from joint state
        measured_positions = np.array(list(joint_state['position'].values()))
        measured_velocities = np.array(list(joint_state['velocity'].values()))
        
        # Create measurement vector
        measurement = np.hstack([measured_positions, measured_velocities])
        
        # Predict step (using dynamics model)
        self.state_estimate = self.predict_state(self.state_estimate)
        
        # Update step (correction based on measurements)
        self.state_estimate = self.update_state(self.state_estimate, measurement)
        
        # Compute forward kinematics to get end-effector positions
        end_effector_positions = self.compute_end_effector_positions()
        
        return {
            'joint_positions': measured_positions,
            'joint_velocities': measured_velocities,
            'end_effector_positions': end_effector_positions,
            'center_of_mass': self.compute_center_of_mass(measured_positions),
            'stability_metrics': self.estimate_stability(measured_positions)
        }
        
    def predict_state(self, current_state):
        """Predict next state based on dynamics model"""
        # Simplified prediction: assume current state continues with slight changes
        # In a real implementation, this would use robot dynamics
        predicted_state = current_state + np.random.normal(0, 0.001, size=current_state.shape)
        return predicted_state
        
    def update_state(self, predicted_state, measurement):
        """Update state with new measurement using Kalman filter approach"""
        # In a real implementation, this would be a proper Kalman filter update
        # For this simulation, we'll blend prediction and measurement
        update_ratio = 0.1  # How much to trust new measurements
        updated_state = (1 - update_ratio) * predicted_state + update_ratio * measurement
        return updated_state
        
    def compute_end_effector_positions(self):
        """Compute end-effector positions from joint angles"""
        # Use forward kinematics to compute end-effector positions
        # This would interface with the robot's kinematic model
        joint_positions = self.state_estimate[:len(self.state_estimate)//2]
        
        # Example computation (simplified)
        # In real implementation, would use full kinematic model
        end_effectors = {}
        
        # Example: Compute position of right hand
        right_hand_position = self.kinematic_model.forward_kinematics(
            joint_positions, 'right_hand'
        )
        end_effectors['right_hand'] = right_hand_position
        
        # Compute position of left hand
        left_hand_position = self.kinematic_model.forward_kinematics(
            joint_positions, 'left_hand'
        )
        end_effectors['left_hand'] = left_hand_position
        
        # Compute head position
        head_position = self.kinematic_model.forward_kinematics(
            joint_positions, 'head'
        )
        end_effectors['head'] = head_position
        
        return end_effectors
        
    def compute_center_of_mass(self, joint_positions):
        """Compute center of mass based on joint positions"""
        # Simplified CoM calculation
        # In real implementation, would use detailed mass distribution
        return np.array([0.0, 0.0, 0.8])  # Typical human-like CoM height
        
    def estimate_stability(self, joint_positions):
        """Estimate overall stability based on joint configuration"""
        # Compute Zero Moment Point (ZMP) as a stability metric
        zmp = self.compute_zmp(joint_positions)
        
        # Define support polygon based on feet positions
        support_polygon = self.compute_support_polygon(joint_positions)
        
        # Check if ZMP is within support polygon
        is_stable = self.is_zmp_in_support_polygon(zmp, support_polygon)
        
        return {
            'zmp': zmp,
            'support_polygon': support_polygon,
            'is_stable': is_stable,
            'stability_margin': self.compute_stability_margin(zmp, support_polygon)
        }
        
    def compute_zmp(self, joint_positions):
        """Compute Zero Moment Point"""
        # Simplified ZMP computation
        # In real implementation, would use full dynamics model
        return np.array([0.0, 0.0])  # Simplified
        
    def compute_support_polygon(self, joint_positions):
        """Compute support polygon based on contact feet"""
        # Simplified support polygon
        # In real implementation, would check ground contact for each foot
        return np.array([[0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]])  # Square around origin
        
    def is_zmp_in_support_polygon(self, zmp, polygon):
        """Check if ZMP is within support polygon"""
        # Use ray casting algorithm to check if point is in polygon
        x, y = zmp
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
```

## Joint Position and Velocity Sensing

### Encoder Technologies

```python
class EncoderSensor:
    def __init__(self, joint_name, encoder_type='incremental', resolution=12):
        self.joint_name = joint_name
        self.encoder_type = encoder_type  # 'incremental' or 'absolute'
        self.resolution = resolution  # bits
        self.max_counts = 2**resolution - 1
        self.current_count = 0
        self.previous_count = 0
        self.position_offset = 0
        self.velocity_estimator = VelocityEstimator()
        
    def read_encoder(self):
        """Read encoder value"""
        # In real implementation, this would read from hardware
        # For simulation, we'll return a changing value
        self.previous_count = self.current_count
        self.current_count = (self.current_count + 10) % self.max_counts  # Simulate movement
        
        return self.current_count
        
    def get_position(self):
        """Get position in radians from encoder count"""
        position = (self.current_count / self.max_counts) * 2 * np.pi  # 0 to 2pi
        position -= np.pi  # Center at -pi to pi
        return position + self.position_offset
        
    def get_velocity(self, dt=0.01):
        """Get velocity from encoder counts"""
        delta_count = self.current_count - self.previous_count
        # Handle wraparound
        if abs(delta_count) > self.max_counts / 2:
            delta_count = delta_count - np.sign(delta_count) * self.max_counts
            
        delta_pos = (delta_count / self.max_counts) * 2 * np.pi
        velocity = delta_pos / dt
        
        return velocity
        
    def calibrate(self, known_position):
        """Calibrate encoder to known position"""
        current_pos = self.get_position()
        self.position_offset = known_position - current_pos

class JointStateObserver:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.encoders = self.create_encoders()
        self.state_history = []  # For velocity and acceleration estimation
        self.max_history = 10  # Number of states to keep for estimation
        
    def create_encoders(self):
        """Create encoder instance for each joint"""
        encoders = {}
        for joint_name in self.robot_model.joint_names:
            joint_type = self.robot_model.joint_types[joint_name]
            resolution = 12  # Default resolution
            
            encoders[joint_name] = EncoderSensor(joint_name, resolution=resolution)
            
        return encoders
        
    def observe_joints(self):
        """Observe all joints and return comprehensive state"""
        positions = {}
        velocities = {}
        accelerations = {}
        
        # Read encoder values
        for joint_name, encoder in self.encoders.items():
            positions[joint_name] = encoder.get_position()
            velocities[joint_name] = encoder.get_velocity()
        
        # Estimate accelerations from history
        self.state_history.append((time.time(), positions, velocities))
        
        # Keep history limited
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            
        # Compute accelerations from velocity history
        if len(self.state_history) >= 2:
            accelerations = self.estimate_accelerations()
        
        return {
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
            'timestamp': time.time()
        }
        
    def estimate_accelerations(self):
        """Estimate accelerations using velocity history"""
        accelerations = {}
        
        if len(self.state_history) < 2:
            # Return zeros if not enough history
            for joint_name in self.encoders.keys():
                accelerations[joint_name] = 0.0
            return accelerations
            
        # Use the last two states to estimate acceleration
        prev_time, _, prev_velocities = self.state_history[-2]
        curr_time, _, curr_velocities = self.state_history[-1]
        
        dt = curr_time - prev_time
        if dt <= 0:
            dt = 0.01  # Default dt
            
        for joint_name in self.encoders.keys():
            if joint_name in prev_velocities and joint_name in curr_velocities:
                dv = curr_velocities[joint_name] - prev_velocities[joint_name]
                accelerations[joint_name] = dv / dt
            else:
                accelerations[joint_name] = 0.0
                
        return accelerations
        
    def detect_anomalies(self, joint_state):
        """Detect potential sensor or joint anomalies"""
        anomalies = []
        
        for joint_name, position in joint_state['position'].items():
            # Check if position is out of joint limits
            joint_limits = self.robot_model.joint_limits.get(joint_name, (-5, 5))
            if not (joint_limits[0] <= position <= joint_limits[1]):
                anomalies.append({
                    'joint': joint_name,
                    'type': 'position_limit_exceeded',
                    'value': position,
                    'limit': joint_limits
                })
                
        return anomalies
```

## Inertial Measurement Units (IMUs)

### IMU Implementation

```python
class IMUSensor:
    def __init__(self, link_name, sample_rate=100, noise_params=None):
        self.link_name = link_name
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Noise parameters
        self.noise_params = noise_params or {
            'gyro_noise_density': 2.0e-3,  # rad/s/sqrt(Hz)
            'gyro_random_walk': 8.0e-5,   # rad/s^2/sqrt(Hz)
            'accel_noise_density': 4.0e-3,  # m/s^2/sqrt(Hz)
            'accel_random_walk': 4.0e-4    # m/s^3/sqrt(Hz)
        }
        
        # Internal state
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z quaternion
        self.angular_velocity = np.zeros(3)
        self.linear_acceleration = np.zeros(3)
        
        # Bias tracking
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # For integration
        self.prev_time = time.time()
        
    def read_imu(self):
        """Read IMU data (simulated)"""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Simulate actual values (in robot's reference frame)
        # These would come from the simulation/physics engine in a real scenario
        true_angular_velocity = self.get_true_angular_velocity()
        true_linear_acceleration = self.get_true_linear_acceleration()
        
        # Apply noise and bias
        noisy_angular_velocity = self.add_noise_and_bias(
            true_angular_velocity, self.gyro_bias, 
            self.noise_params['gyro_noise_density'] * np.sqrt(self.sample_rate)
        )
        
        noisy_linear_acceleration = self.add_noise_and_bias(
            true_linear_acceleration, self.accel_bias,
            self.noise_params['accel_noise_density'] * np.sqrt(self.sample_rate)
        )
        
        # Update internal state
        self.angular_velocity = noisy_angular_velocity
        self.linear_acceleration = noisy_linear_acceleration
        
        # Integrate to update orientation (simplified)
        self.integrate_orientation(dt)
        
        self.prev_time = current_time
        
        return {
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'linear_acceleration': self.linear_acceleration.copy()
        }
        
    def get_true_angular_velocity(self):
        """Get true angular velocity from simulation (in real implementation, this would come from physics engine)"""
        # For simulation, return small random values
        return np.random.normal(0, 0.1, size=3)
        
    def get_true_linear_acceleration(self):
        """Get true linear acceleration from simulation"""
        # Include gravity in the IMU frame
        gravity = np.array([0, 0, -9.81])  # gravity vector in world frame
        
        # Rotate to IMU frame using current orientation
        gravity_imu_frame = self.rotate_vector_to_imu_frame(gravity)
        
        # Add small motion accelerations
        motion_acc = np.random.normal(0, 0.5, size=3)
        
        return gravity_imu_frame + motion_acc
        
    def rotate_vector_to_imu_frame(self, vector):
        """Rotate a vector from world frame to IMU frame using current orientation"""
        # Convert quaternion to rotation matrix
        w, x, y, z = self.orientation
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return rotation_matrix @ vector
        
    def add_noise_and_bias(self, true_value, bias, noise_level):
        """Add realistic noise and bias to true sensor values"""
        noise = np.random.normal(0, noise_level, size=true_value.shape)
        return true_value + bias + noise
        
    def integrate_orientation(self, dt):
        """Integrate angular velocity to update orientation"""
        # Convert angular velocity to quaternion derivative
        omega = self.angular_velocity
        
        # Quaternion derivative: dq/dt = 0.5 * q * omega_quat
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        
        # Multiply quaternion by omega_quat (simplified)
        q_dot = np.zeros(4)
        q_dot[0] = -0.5 * (omega[0] * self.orientation[1] + 
                          omega[1] * self.orientation[2] + 
                          omega[2] * self.orientation[3])
        q_dot[1] = 0.5 * (omega[0] * self.orientation[0] - 
                         omega[1] * self.orientation[3] + 
                         omega[2] * self.orientation[2])
        q_dot[2] = 0.5 * (omega[0] * self.orientation[3] + 
                         omega[1] * self.orientation[0] - 
                         omega[2] * self.orientation[1])
        q_dot[3] = 0.5 * (-omega[0] * self.orientation[2] + 
                         omega[1] * self.orientation[1] + 
                         omega[2] * self.orientation[0])
        
        # Update orientation
        new_orientation = self.orientation + q_dot * dt
        
        # Normalize quaternion
        self.orientation = new_orientation / np.linalg.norm(new_orientation)
        
    def calibrate_bias(self, stationary_time=5.0):
        """Calibrate sensor biases while robot is stationary"""
        print(f"Starting IMU {self.link_name} calibration...")
        
        gyro_readings = []
        accel_readings = []
        
        # Collect data while stationary
        start_time = time.time()
        while time.time() - start_time < stationary_time:
            data = self.read_imu()
            gyro_readings.append(data['angular_velocity'])
            accel_readings.append(data['linear_acceleration'])
            time.sleep(0.1)
            
        # Calculate biases
        self.gyro_bias = np.mean(gyro_readings, axis=0)
        # For accelerometer, remove gravity to get true bias
        avg_accel = np.mean(accel_readings, axis=0)
        gravity = np.array([0, 0, -9.81])
        self.accel_bias = avg_accel - gravity  # What's left after removing gravity
        
        print(f"Calibration complete for IMU {self.link_name}")
        print(f"Gyro bias: {self.gyro_bias}")
        print(f"Accel bias: {self.accel_bias}")
```

### Multi-IMU State Estimation

```python
class MultiIMUStateEstimator:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.imus = self.setup_body_imus()
        
        # State: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z]
        self.state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(10) * 0.1  # Initial uncertainty
        
    def setup_body_imus(self):
        """Set up IMUs on different parts of the robot body"""
        imus = {}
        
        # Torso IMU (primary reference)
        imus['torso'] = IMUSensor('torso', sample_rate=500)
        
        # Additional IMUs for better estimation
        imus['head'] = IMUSensor('head', sample_rate=200)
        imus['left_foot'] = IMUSensor('left_foot', sample_rate=200)
        imus['right_foot'] = IMUSensor('right_foot', sample_rate=200)
        
        return imus
        
    def estimate_state(self):
        """Estimate full body state using multiple IMUs"""
        # Read all IMUs
        imu_readings = {}
        for name, imu in self.imus.items():
            imu_readings[name] = imu.read_imu()
        
        # Perform sensor fusion to estimate state
        self.update_state_with_imu_data(imu_readings)
        
        return {
            'position': self.state[0:3],
            'orientation': self.state[3:7],
            'velocity': self.state[7:10],
            'imu_readings': imu_readings,
            'balance_metrics': self.compute_balance_metrics(imu_readings)
        }
        
    def update_state_with_imu_data(self, imu_readings):
        """Update state estimate using IMU data"""
        # Simplified fusion approach
        # In a real implementation, this would use an Extended Kalman Filter or other fusion algorithm
        
        # Use torso IMU as primary reference for orientation
        torso_data = imu_readings['torso']
        self.state[3:7] = torso_data['orientation']  # Update orientation
        
        # Use foot IMUs to detect contact and estimate ZMP
        left_foot_data = imu_readings['left_foot']
        right_foot_data = imu_readings['right_foot']
        
        # Estimate velocity by integrating torso acceleration
        # Remove gravity from linear acceleration
        gravity_corrected_acc = self.remove_gravity(
            torso_data['linear_acceleration'], 
            torso_data['orientation']
        )
        
        # Integrate to get velocity
        dt = 0.01  # Hardcoded for example
        self.state[7:10] += gravity_corrected_acc * dt  # Update velocity
        
        # Integrate to get position
        self.state[0:3] += self.state[7:10] * dt  # Update position
        
    def remove_gravity(self, acceleration, orientation):
        """Remove gravity from measured acceleration"""
        # Convert quaternion to rotation matrix
        w, x, y, z = orientation
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Gravity vector in world frame
        gravity_world = np.array([0, 0, 9.81])
        
        # Transform gravity to IMU frame
        gravity_imu = rotation_matrix @ gravity_world
        
        # Remove gravity from measured acceleration
        corrected_acceleration = acceleration - gravity_imu
        
        return corrected_acceleration
        
    def compute_balance_metrics(self, imu_readings):
        """Compute balance-related metrics using IMU data"""
        # Compute Zero Moment Point (ZMP) approximation
        zmp = self.compute_zmp_approximation(imu_readings)
        
        # Compute stability margin based on foot positions
        support_polygon = self.compute_support_polygon()
        stability_margin = self.compute_stability_margin(zmp, support_polygon)
        
        # Compute body orientation relative to gravity
        torso_orientation = imu_readings['torso']['orientation']
        body_tilt = self.compute_body_tilt(torso_orientation)
        
        return {
            'zmp': zmp,
            'stability_margin': stability_margin,
            'body_tilt': body_tilt,
            'foot_contacts': self.estimate_foot_contacts(imu_readings)
        }
        
    def compute_zmp_approximation(self, imu_readings):
        """Compute ZMP approximation using IMU data"""
        # Simplified ZMP calculation based on body orientation and acceleration
        torso_acc = imu_readings['torso']['linear_acceleration']
        
        # Project torso acceleration to get ZMP (simplified)
        # In reality, this requires complex dynamics
        zmp_x = -torso_acc[0] / 9.81 * 0.8  # 0.8m is approximate CoM height
        zmp_y = -torso_acc[1] / 9.81 * 0.8
        
        return np.array([zmp_x, zmp_y])
        
    def compute_support_polygon(self):
        """Compute support polygon based on foot positions"""
        # Simplified polygon
        # In reality, this would be based on contact points of feet
        return np.array([[-0.1, -0.05], [-0.1, 0.05], [0.1, 0.05], [0.1, -0.05]])
        
    def compute_stability_margin(self, zmp, support_polygon):
        """Compute distance from ZMP to edge of support polygon"""
        # Simplified calculation
        # Find the closest point on the polygon to the ZMP
        distances = []
        for i in range(len(support_polygon)):
            edge_start = support_polygon[i]
            edge_end = support_polygon[(i + 1) % len(support_polygon)]
            
            # Calculate distance from ZMP to this edge
            distance = self.distance_point_to_line_segment(zmp, edge_start, edge_end)
            distances.append(distance)
            
        return min(distances)  # Closest distance to any edge
        
    def distance_point_to_line_segment(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        # Vector calculations for line segment
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point - line_start)
            
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        
        t = np.dot(line_unitvec, point_vec_scaled)
        t = np.clip(t, 0, 1)  # Clamp to line segment
        
        nearest = line_start + t * line_vec
        return np.linalg.norm(point - nearest)
        
    def compute_body_tilt(self, orientation):
        """Compute body tilt from orientation quaternion"""
        # Convert quaternion to roll, pitch, yaw
        w, x, y, z = orientation
        
        # Calculate roll and pitch
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        # Return tilt as combination of roll and pitch
        tilt = np.sqrt(roll**2 + pitch**2)
        return tilt
        
    def estimate_foot_contacts(self, imu_readings):
        """Estimate foot contacts using IMU data"""
        foot_contacts = {}
        
        # Use foot IMU data to estimate contact
        # When acceleration is low, the foot is likely in contact
        for foot in ['left_foot', 'right_foot']:
            linear_acc = imu_readings[foot]['linear_acceleration']
            angular_vel = imu_readings[foot]['angular_velocity']
            
            # Compute magnitude of acceleration and angular velocity
            acc_magnitude = np.linalg.norm(linear_acc)
            vel_magnitude = np.linalg.norm(angular_vel)
            
            # If acceleration is close to gravity (1g) and angular velocity is low, likely in contact
            is_contact = (abs(acc_magnitude - 9.81) < 2) and (vel_magnitude < 0.5)
            foot_contacts[foot] = is_contact
            
        return foot_contacts
```

## Force and Torque Sensing

### Force/Torque Sensor Implementation

```python
class ForceTorqueSensor:
    def __init__(self, sensor_name, position, orientation, noise_params=None):
        self.sensor_name = sensor_name
        self.position = position  # Position in parent link frame [x, y, z]
        self.orientation = orientation  # Orientation as quaternion [w, x, y, z]
        
        # Noise and bias parameters
        self.noise_params = noise_params or {
            'force_noise_density': 1e-3,    # N/sqrt(Hz)
            'torque_noise_density': 1e-4,   # Nm/sqrt(Hz)
            'force_bias': [0, 0, 0],
            'torque_bias': [0, 0, 0]
        }
        
        # True values (from simulation/real sensor)
        self.true_force = np.zeros(3)
        self.true_torque = np.zeros(3)
        
        # Measured values (with noise and bias)
        self.measured_force = np.zeros(3)
        self.measured_torque = np.zeros(3)
        
    def read_sensor(self):
        """Read force and torque values from sensor"""
        # Get true values (in a real implementation, these would come from the physical sensor)
        true_force, true_torque = self.get_true_values()
        
        # Add noise and bias
        self.measured_force = self.add_noise_and_bias(
            true_force, 
            np.array(self.noise_params['force_bias']), 
            self.noise_params['force_noise_density']
        )
        
        self.measured_torque = self.add_noise_and_bias(
            true_torque, 
            np.array(self.noise_params['torque_bias']), 
            self.noise_params['torque_noise_density']
        )
        
        # Apply temperature compensation if available
        self.apply_temperature_compensation()
        
        return {
            'force': self.measured_force.copy(),
            'torque': self.measured_torque.copy()
        }
        
    def get_true_values(self):
        """Get true force and torque values (for simulation)"""
        # In a real implementation, these would come from the physical sensor
        # For simulation, we'll return values based on robot's state
        return np.array([0.5, 0.2, -9.3]), np.array([0.02, 0.01, 0.05])
        
    def add_noise_and_bias(self, true_value, bias, noise_density):
        """Add noise and bias to true sensor values"""
        # Add bias
        biased_value = true_value + bias
        
        # Add noise (simplified model)
        noise = np.random.normal(0, noise_density, size=true_value.shape)
        
        return biased_value + noise
        
    def apply_temperature_compensation(self):
        """Apply temperature-based compensation if available"""
        # In real sensors, temperature affects the readings
        # For this simulation, we'll skip this
        pass
        
    def calibrate_sensor(self, no_load_duration=5.0):
        """Calibrate the sensor by measuring bias with no load"""
        print(f"Starting calibration for sensor {self.sensor_name}")
        
        force_readings = []
        torque_readings = []
        
        # Collect readings with no load
        start_time = time.time()
        while time.time() - start_time < no_load_duration:
            data = self.read_sensor()
            force_readings.append(data['force'])
            torque_readings.append(data['torque'])
            time.sleep(0.1)
            
        # Calculate new biases
        measured_force_bias = np.mean(force_readings, axis=0)
        measured_torque_bias = np.mean(torque_readings, axis=0)
        
        # The true bias is the negative of the measured average (assuming no external forces)
        self.noise_params['force_bias'] = -measured_force_bias
        self.noise_params['torque_bias'] = -measured_torque_bias
        
        print(f"Calibration complete for {self.sensor_name}")
        print(f"Force bias: {self.noise_params['force_bias']}")
        print(f"Torque bias: {self.noise_params['torque_bias']}")

class WristForceTorqueSensor:
    def __init__(self, hand_name):
        # 6-axis F/T sensor at wrist
        self.ft_sensor = ForceTorqueSensor(
            f"{hand_name}_wrist_ft", 
            position=[0, 0, 0],  # At the wrist
            orientation=[1, 0, 0, 0]  # Identity quaternion
        )
        self.hand_name = hand_name
        self.temperature = 25  # Current temperature in Celsius
        
    def get_wrench(self):
        """Get the 6-axis wrench (force + torque) from the sensor"""
        raw_data = self.ft_sensor.read_sensor()
        
        # Transform to robot's base frame if needed
        wrench = self.transform_to_base_frame(raw_data)
        
        return wrench
        
    def transform_to_base_frame(self, raw_data):
        """Transform wrench from sensor frame to robot base frame"""
        # This would involve transformation matrices based on robot's kinematics
        # For this example, we'll return the raw data
        return {
            'force': raw_data['force'],
            'torque': raw_data['torque'],
            'timestamp': time.time()
        }
        
    def detect_contact(self, threshold=2.0):
        """Detect if the hand is in contact with something"""
        wrench = self.get_wrench()
        
        # Calculate magnitude of force vector
        force_magnitude = np.linalg.norm(wrench['force'])
        
        return force_magnitude > threshold, force_magnitude
        
    def detect_slip(self):
        """Detect slip based on force and torque patterns"""
        # In a real implementation, this would analyze temporal patterns
        # For this example, we'll use a simplified approach
        wrench = self.get_wrench()
        
        # Rapid changes in force might indicate slip
        # This would typically use a history of readings
        return False  # Simplified
```

### Force-Based Control and Interaction

```python
class ForceBasedController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.wrist_sensors = self.setup_wrist_sensors()
        self.impedance_controllers = self.setup_impedance_controllers()
        
    def setup_wrist_sensors(self):
        """Set up force/torque sensors on wrists"""
        sensors = {}
        
        for hand in ['left', 'right']:
            sensors[hand] = WristForceTorqueSensor(hand)
            
        return sensors
        
    def setup_impedance_controllers(self):
        """Set up impedance controllers for compliant manipulation"""
        controllers = {}
        
        for hand in ['left', 'right']:
            controllers[hand] = ImpedanceController(
                mass=1.0, damping=10.0, stiffness=100.0
            )
            
        return controllers
        
    def compliant_control(self, hand, desired_force, current_pose, dt):
        """Perform compliant control based on force feedback"""
        # Get current force from sensor
        current_wrench = self.wrist_sensors[hand].get_wrench()
        current_force = current_wrench['force']
        
        # Calculate force error
        force_error = desired_force - current_force
        
        # Use impedance controller to compute position adjustment
        position_correction = self.impedance_controllers[hand].compute_impedance(
            force_error, dt
        )
        
        # Calculate new desired pose
        new_pose = current_pose.copy()
        new_pose[:3] += position_correction  # Update position part of pose
        
        return new_pose
        
    def grasp_control(self, hand, object_properties):
        """Control grasp force based on object properties"""
        # Get current force
        wrench = self.wrist_sensors[hand].get_wrench()
        current_force_magnitude = np.linalg.norm(wrench['force'])
        
        # Determine appropriate grasp force based on object properties
        appropriate_force = self.calculate_grasp_force(object_properties)
        
        # Calculate force error
        force_error = appropriate_force - current_force_magnitude
        
        # Adjust motor commands to correct force error
        # This would interface with the robot's motor controller
        adjustment = self.force_to_motor_command(force_error)
        
        return adjustment
        
    def calculate_grasp_force(self, object_properties):
        """Calculate appropriate grasp force based on object properties"""
        # Example calculation based on object weight and fragility
        weight = object_properties.get('weight', 0.1)  # kg
        fragility = object_properties.get('fragility', 0.5)  # 0-1 scale
        
        # Base grasp force on object weight
        base_force = weight * 9.81 * 1.5  # Weight * safety factor
        
        # Adjust for fragility (more fragile = less force)
        fragility_factor = max(0.1, 1 - fragility)
        adjusted_force = base_force * fragility_factor
        
        # Ensure minimum force for stability
        final_force = max(adjusted_force, 2.0)  # Minimum 2N
        
        return final_force
        
    def force_to_motor_command(self, force_error):
        """Convert force error to motor command adjustment"""
        # Simple proportional control
        kp = 0.1  # Proportional gain
        command_adjustment = kp * force_error
        return command_adjustment
        
    def detect_unexpected_contacts(self, expected_wrench, threshold=5.0):
        """Detect contacts that weren't expected"""
        unexpected_contacts = {}
        
        for hand in ['left', 'right']:
            current_wrench = self.wrist_sensors[hand].get_wrench()
            
            # Calculate deviation from expected wrench
            force_deviation = np.linalg.norm(
                current_wrench['force'] - expected_wrench[hand]['force']
            )
            torque_deviation = np.linalg.norm(
                current_wrench['torque'] - expected_wrench[hand]['torque']
            )
            
            # Check if deviation exceeds threshold
            if force_deviation > threshold or torque_deviation > threshold:
                unexpected_contacts[hand] = {
                    'force_deviation': force_deviation,
                    'torque_deviation': torque_deviation,
                    'current_wrench': current_wrench
                }
                
        return unexpected_contacts

class ImpedanceController:
    def __init__(self, mass=1.0, damping=10.0, stiffness=100.0):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        
    def compute_impedance(self, force, dt):
        """Compute position change based on applied force using impedance model"""
        # Solve mass-spring-damper equation: M*a + B*v + K*x = F
        # For small timesteps, we can approximate as: 
        # dx = (F*dt^2) / M  (simplified model)
        
        acceleration = force / self.mass
        position_change = 0.5 * acceleration * dt**2
        
        return position_change
```

## Sensor Fusion for Proprioception

### Kalman Filtering for Proprioception

```python
class ProprioceptiveKalmanFilter:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints
        
        # State vector: [pos1, vel1, pos2, vel2, ..., posN, velN]
        self.state_dim = 2 * self.num_joints
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 0.1
        
        # Process and measurement noise
        self.Q = np.eye(self.state_dim) * 0.01  # Process noise
        self.R = np.eye(self.num_joints) * 0.1  # Measurement noise (positions only)
        
        # Control input matrix (simplified)
        self.B = np.zeros((self.state_dim, self.num_joints))
        
    def predict(self, control_input, dt):
        """Predict state based on control input and dynamics"""
        # Continuous time dynamics matrix for mass-spring-damper system
        # A = [[0 I], [-K/M -B/M]] where I is identity, K/M and B/M are diagonal
        
        A_continuous = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.num_joints):
            # Position derivative is velocity
            A_continuous[i*2, i*2+1] = 1.0
            
        # Discretize the system
        A_discrete = np.eye(self.state_dim) + A_continuous * dt
        
        # Predict state: x_k = A*x_{k-1} + B*u_{k-1}
        self.state = A_discrete @ self.state + self.B @ control_input
        
        # Predict covariance: P_k = A*P_{k-1}*A^T + Q
        self.covariance = A_discrete @ self.covariance @ A_discrete.T + self.Q
        
    def update(self, measurements):
        """Update state with new measurements"""
        # Measurement model: only positions are measured directly
        H = np.zeros((self.num_joints, self.state_dim))
        for i in range(self.num_joints):
            H[i, i*2] = 1.0  # Measure position of each joint
            
        # Calculate Kalman gain
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Innovation
        y = measurements - H @ self.state
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance

class ExtendedKalmanFilterFusion:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.state_vector = self.initialize_state_vector()
        
        # Initialize various sensors
        self.encoders = self.initialize_encoders()
        self.imus = self.initialize_imus()
        self.force_sensors = self.initialize_force_sensors()
        
    def initialize_state_vector(self):
        """Initialize state vector with position, velocity, and orientation"""
        # State: [joint_positions, joint_velocities, base_position, base_orientation, base_velocity]
        state_size = (2 * self.robot_model.num_joints) + 3 + 4 + 3
        return np.zeros(state_size)
        
    def initialize_encoders(self):
        """Initialize encoder sensors for all joints"""
        encoders = {}
        for joint_name in self.robot_model.joint_names:
            encoders[joint_name] = EncoderSensor(joint_name)
        return encoders
        
    def initialize_imus(self):
        """Initialize IMUs at various body points"""
        return MultiIMUStateEstimator(self.robot_model)
        
    def initialize_force_sensors(self):
        """Initialize force/torque sensors"""
        return ForceBasedController(self.robot_model)
        
    def fuse_sensors(self):
        """Fuse data from all proprioceptive sensors"""
        # Get encoder data
        joint_state = {}
        for name, encoder in self.encoders.items():
            joint_state[name] = {
                'position': encoder.get_position(),
                'velocity': encoder.get_velocity()
            }
            
        # Get IMU data and state estimates
        imu_state = self.imus.estimate_state()
        
        # Get force data
        force_data = {}
        for hand in ['left', 'right']:
            wrench = self.force_sensors.wrist_sensors[hand].get_wrench()
            force_data[f'{hand}_wrench'] = wrench
            
        # Perform fusion to get consistent state
        fused_state = self.perform_fusion(joint_state, imu_state, force_data)
        
        return fused_state
        
    def perform_fusion(self, joint_state, imu_state, force_data):
        """Perform sensor fusion to get consistent state estimate"""
        # Use complementary filtering to combine different sensor modalities
        # Position: primarily from encoders with IMU corrections
        # Orientation: primarily from IMUs
        # Velocity: derived from encoders with IMU corrections
        
        fused_state = {
            'joint_positions': self.fuse_joint_positions(joint_state),
            'joint_velocities': self.fuse_joint_velocities(joint_state),
            'base_position': imu_state['position'],
            'base_orientation': imu_state['orientation'],
            'base_velocity': imu_state['velocity'],
            'contact_status': self.estimate_contact_status(force_data),
            'stability_metrics': imu_state['balance_metrics']
        }
        
        return fused_state
        
    def fuse_joint_positions(self, joint_state):
        """Fuse position data from encoders and other sources"""
        # For now, primarily use encoder values
        # In real implementation, these would be adjusted based on other sensor data
        positions = {}
        for joint_name, state in joint_state.items():
            positions[joint_name] = state['position']
            
        return positions
        
    def fuse_joint_velocities(self, joint_state):
        """Fuse velocity data from encoders and other sources"""
        # For now, primarily use encoder values
        velocities = {}
        for joint_name, state in joint_state.items():
            velocities[joint_name] = state['velocity']
            
        return velocities
        
    def estimate_contact_status(self, force_data):
        """Estimate contact status from force data"""
        contact_status = {}
        
        # Check for contact based on force threshold
        for hand in ['left', 'right']:
            wrench = force_data[f'{hand}_wrench']
            force_magnitude = np.linalg.norm(wrench['force'])
            
            contact_status[f'{hand}_contact'] = {
                'in_contact': force_magnitude > 1.0,  # 1N threshold
                'force_magnitude': force_magnitude,
                'force_vector': wrench['force'],
                'timestamp': wrench['timestamp']
            }
            
        return contact_status
```

## Summary

This chapter has covered the critical proprioceptive and tactile sensing systems that enable humanoid robots to perceive their own state and interact with objects through touch. We've explored various tactile sensing technologies, including resistive, capacitive, and vision-based sensors, highlighting their implementation and use in humanoid robotics.

The chapter detailed how tactile sensors are implemented in arrays and processed for object recognition and manipulation. We covered the estimation of object properties through tactile sensing and how these capabilities support dexterous manipulation and grasp stability assessment.

Proprioceptive sensing was thoroughly examined, including joint position and velocity sensing using encoders, inertial measurement units for orientation and motion detection, and force/torque sensors for interaction control. We discussed how these sensors work together to provide awareness of the robot's configuration and its interaction with the environment.

The chapter concluded with sensor fusion techniques that combine multiple proprioceptive sources to create consistent state estimates. This fusion is essential for the reliable operation of humanoid robots, as it allows the integration of complementary information from different sensor types to create a comprehensive understanding of the robot's state.

Effective proprioceptive and tactile sensing is fundamental to the dexterous manipulation and stable locomotion capabilities of humanoid robots. These systems provide the physical awareness necessary for safe interaction with humans and objects, forming a critical foundation for autonomous robot operation.

## Exercises

1. Implement a tactile sensor array system for a robot hand with 5 fingers, including contact detection, slip detection, and object property estimation.

2. Design a sensor fusion algorithm that combines data from joint encoders, IMUs, and force/torque sensors to accurately estimate the pose of a robot's end-effector.

3. Create a grasp stability assessment system that uses tactile sensor data to determine if an object is securely grasped and recommend adjustments if needed.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*