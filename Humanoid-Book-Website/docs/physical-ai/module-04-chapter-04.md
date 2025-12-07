---
id: module-04-chapter-04
title: Chapter 04 - Sensor Fusion and Integration
sidebar_position: 16
---

# Chapter 04 - Sensor Fusion and Integration

## Table of Contents
- [Overview](#overview)
- [Sensor Fusion Fundamentals](#sensor-fusion-fundamentals)
- [Kalman Filtering for Sensor Fusion](#kalman-filtering-for-sensor-fusion)
- [Particle Filtering](#particle-filtering)
- [Multi-Sensor Data Association](#multi-sensor-data-association)
- [Time Synchronization](#time-synchronization)
- [Sensor Calibration and Registration](#sensor-calibration-and-registration)
- [Bayesian Sensor Fusion](#bayesian-sensor-fusion)
- [Deep Learning for Sensor Fusion](#deep-learning-for-sensor-fusion)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Sensor fusion is the process of combining information from multiple sensors to achieve improved accuracy, reliability, and robustness compared to using any single sensor alone. In humanoid robotics, effective sensor fusion is critical for creating a coherent understanding of the robot's state and environment. This chapter explores the theoretical foundations, practical implementations, and advanced techniques for fusing data from the diverse array of sensors found on humanoid robots, including cameras, IMUs, force/torque sensors, and tactile sensors.

Humanoid robots operate in complex environments where no single sensor can provide complete information. Visual sensors may fail in poor lighting conditions, force sensors might not detect contact until it occurs, and IMUs can drift over time. By intelligently combining data from multiple sensors, humanoid robots can maintain awareness of their state and environment even when individual sensors fail or provide incomplete information.

## Sensor Fusion Fundamentals

### Why Sensor Fusion is Critical for Humanoid Robots

Humanoid robots face unique challenges that make sensor fusion essential:

1. **Diverse Sensor Modalities**: Humanoids have many different types of sensors - visual, inertial, force, tactile - each providing different types of information
2. **Environmental Challenges**: Operating in dynamic, unstructured environments where individual sensors may fail
3. **Real-time Requirements**: Need for immediate processing to maintain balance and safe interaction
4. **Robustness Needs**: Must operate safely even when some sensors fail

### Types of Sensor Fusion

Sensor fusion can be categorized by the level of processing:

1. **Data Level Fusion**: Combining raw sensor data
2. **Feature Level Fusion**: Combining extracted features from sensor data
3. **Decision Level Fusion**: Combining decisions or classifications from different sensors
4. **Hybrid Fusion**: Combining different levels of information

```python
class SensorFusionLevel:
    """Class to demonstrate different levels of sensor fusion"""
    
    def __init__(self):
        self.fusion_methods = {
            'data': self.data_level_fusion,
            'feature': self.feature_level_fusion,
            'decision': self.decision_level_fusion,
            'hybrid': self.hybrid_fusion
        }
        
    def data_level_fusion(self, raw_sensor_data):
        """Combine raw sensor readings"""
        # Process raw data from all sensors together
        # Example: Combine camera pixel values with LIDAR points
        fused_data = {}
        for sensor_type, data in raw_sensor_data.items():
            if sensor_type == 'camera':
                fused_data['image'] = data
            elif sensor_type == 'lidar':
                fused_data['point_cloud'] = data
            elif sensor_type == 'imu':
                fused_data['inertial'] = data
        return fused_data
        
    def feature_level_fusion(self, features):
        """Combine extracted features"""
        # Combine features from different sensors
        # Example: Combine visual features with inertial features
        combined_features = {}
        for sensor_type, sensor_features in features.items():
            combined_features.update({
                f"{sensor_type}_{key}": value 
                for key, value in sensor_features.items()
            })
        return combined_features
        
    def decision_level_fusion(self, decisions):
        """Combine final decisions from different sensors"""
        # Combine final decisions or classifications
        # Example: Weighted voting of different sensor decisions
        final_decision = {}
        for sensor_type, decision in decisions.items():
            # Apply confidence weighting based on sensor reliability
            confidence = decision.get('confidence', 1.0)
            final_decision[sensor_type] = {'decision': decision, 'weight': confidence}
        return final_decision
        
    def hybrid_fusion(self, data):
        """Combine multiple fusion levels"""
        # Implement a combination of different fusion levels
        # This might combine raw data for some applications and features for others
        pass
```

### Mathematical Foundations

The most common mathematical framework for sensor fusion is probability theory:

```python
import numpy as np
from scipy.stats import multivariate_normal

class ProbabilisticFusion:
    """Basic probabilistic sensor fusion framework"""
    
    def __init__(self):
        self.sensor_covariances = {}
        self.state_estimate = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.uncertainty = np.eye(6) * 0.1  # Initial uncertainty
        
    def bayesian_update(self, prior, likelihood):
        """Perform Bayesian update of state estimate"""
        # Prior: p(x) - our belief before new measurement
        # Likelihood: p(z|x) - probability of measurement given state
        # Posterior: p(x|z) ∝ p(z|x) * p(x)
        
        # For Gaussian distributions, this has a closed-form solution
        posterior_mean, posterior_cov = self.update_gaussian(
            prior['mean'], prior['cov'], 
            likelihood['mean'], likelihood['cov']
        )
        
        return {
            'mean': posterior_mean,
            'cov': posterior_cov
        }
        
    def update_gaussian(self, prior_mean, prior_cov, meas_mean, meas_cov):
        """Update Gaussian state estimate with measurement"""
        # Kalman filter update equations
        # Innovation covariance
        S = prior_cov + meas_cov
        
        # Kalman gain
        K = prior_cov @ np.linalg.inv(S)
        
        # Updated mean and covariance
        updated_mean = prior_mean + K @ (meas_mean - prior_mean)
        updated_cov = prior_cov - K @ S @ K.T
        
        return updated_mean, updated_cov
        
    def fuse_two_sensors(self, meas1, meas2):
        """Fusion of two sensor measurements"""
        # Calculate fused estimate
        # Method: Inverse covariance weighting
        cov1_inv = np.linalg.inv(meas1['cov'])
        cov2_inv = np.linalg.inv(meas2['cov'])
        
        fused_cov = np.linalg.inv(cov1_inv + cov2_inv)
        fused_mean = fused_cov @ (cov1_inv @ meas1['mean'] + cov2_inv @ meas2['mean'])
        
        return {
            'mean': fused_mean,
            'cov': fused_cov
        }
```

### Sensor Characteristics and Modeling

Understanding sensor characteristics is crucial for effective fusion:

```python
class SensorModel:
    """Model for different sensor characteristics"""
    
    def __init__(self, sensor_type, noise_params):
        self.sensor_type = sensor_type
        self.noise_params = noise_params
        self.bias = np.zeros(self.get_output_dim())
        self.scale_factor = 1.0
        
    def get_output_dim(self):
        """Get dimension of sensor output"""
        mapping = {
            'camera': 3,      # x, y, z position
            'lidar': 3,       # x, y, z or range, bearing, elevation
            'imu': 6,         # 3 axes each for acceleration and angular velocity
            'force_torque': 6, # 3 forces + 3 torques
            'encoder': 1,     # joint position
            'gps': 3          # x, y, z position
        }
        return mapping.get(self.sensor_type, 1)
        
    def model_reading(self, true_value):
        """Model sensor reading with noise and bias"""
        # Add bias
        biased_value = true_value + self.bias
        
        # Add noise
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.get_output_dim()), 
            cov=np.eye(self.get_output_dim()) * self.noise_params['noise_density']**2
        )
        
        # Apply scale factor
        final_reading = self.scale_factor * (biased_value + noise)
        
        return final_reading
        
    def compute_likelihood(self, measurement, state_prediction):
        """Compute likelihood of measurement given state prediction"""
        # Calculate expected measurement from state
        expected_measurement = self.predict_measurement(state_prediction)
        
        # Measurement residual
        residual = measurement - expected_measurement
        
        # Compute likelihood using sensor noise model
        covariance = np.eye(len(residual)) * self.noise_params['noise_density']**2
        likelihood = multivariate_normal.pdf(
            residual, 
            mean=np.zeros(len(residual)), 
            cov=covariance
        )
        
        return likelihood
        
    def predict_measurement(self, state):
        """Predict what measurement should be given state"""
        # This depends on sensor type and mounting position
        # Simplified implementation
        return state[:self.get_output_dim()]
```

## Kalman Filtering for Sensor Fusion

### Extended Kalman Filter (EKF)

The Extended Kalman Filter handles non-linear systems by linearizing around the current state estimate:

```python
class ExtendedKalmanFilter:
    def __init__(self, state_dim, control_dim):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim) * 0.1
        
        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.01  # Process noise covariance
        self.R = np.eye(3) * 0.1          # Measurement noise covariance (example)
        
    def predict(self, control_input, dt):
        """Prediction step with non-linear dynamics"""
        # Non-linear state transition function f(x, u)
        # For example, integrating position and velocity
        new_state = self.nonlinear_dynamics(self.state, control_input, dt)
        
        # Jacobian of the state transition function
        F = self.jacobian_state_transition(self.state, control_input, dt)
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
        
        self.state = new_state
        
    def nonlinear_dynamics(self, state, control, dt):
        """Non-linear dynamics model"""
        # Example: position-velocity model
        # state = [x, y, z, vx, vy, vz, roll, pitch, yaw]
        new_state = state.copy()
        
        # Update positions based on velocities
        new_state[0:3] = state[0:3] + state[3:6] * dt  # positions += velocities * dt
        
        # Update velocities based on accelerations
        # Control input might be accelerations
        new_state[3:6] = state[3:6] + control * dt  # velocities += accelerations * dt
        
        # Update orientation
        # Simplified: integrate angular velocities
        if len(state) > 6:
            new_state[6:9] = state[6:9] + control[3:6] * dt  # orientation += angular_velocities * dt
            
        return new_state
        
    def jacobian_state_transition(self, state, control, dt):
        """Jacobian of the state transition function"""
        F = np.eye(self.state_dim)
        
        # For position-velocity model:
        # dx/dx = 1, dx/dv = dt, etc.
        F[0:3, 3:6] = np.eye(3) * dt  # Position derivatives w.r.t. velocity
        
        return F
        
    def update(self, measurement, measurement_model):
        """Update step with non-linear measurement model"""
        # Expected measurement
        expected_meas = self.nonlinear_measurement_model(self.state)
        
        # Jacobian of measurement model
        H = self.jacobian_measurement_model(self.state)
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        innovation = measurement - expected_meas
        self.state = self.state + K @ innovation
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance
        
    def nonlinear_measurement_model(self, state):
        """Non-linear measurement model h(x)"""
        # Example: measure position only
        return state[0:3]  # Return position part of state
        
    def jacobian_measurement_model(self, state):
        """Jacobian of measurement model"""
        H = np.zeros((3, self.state_dim))  # 3D measurement
        H[0:3, 0:3] = np.eye(3)  # Measure position directly
        return H
```

### Unscented Kalman Filter (UKF)

The UKF provides better accuracy than EKF for highly non-linear systems:

```python
class UnscentedKalmanFilter:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim) * 0.1
        
        # UKF parameters
        self.alpha = 1e-3  # Spread of sigma points
        self.kappa = 0     # Secondary scaling parameter
        self.beta = 2      # Parameter for distribution (2 for Gaussian)
        
        # Derived parameters
        self.lmbda = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.gamma = np.sqrt(self.state_dim + self.lmbda)
        
        # Weights
        self.Wm = np.full(2 * self.state_dim + 1, 1.0 / (2 * (self.state_dim + self.lmbda)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lmbda / (self.state_dim + self.lmbda)
        self.Wc[0] = self.lmbda / (self.state_dim + self.lmbda) + (1 - self.alpha**2 + self.beta)
        
        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(3) * 0.1  # Example measurement covariance
        
    def predict(self, dt):
        """Prediction step using unscented transform"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points through non-linear function
        propagated_points = []
        for point in sigma_points:
            propagated_point = self.nonlinear_dynamics(point, dt)
            propagated_points.append(propagated_point)
        
        # Calculate predicted state and covariance
        self.state = sum(self.Wm[i] * propagated_points[i] for i in range(len(propagated_points)))
        
        # Calculate covariance
        self.covariance = self.Q  # Process noise
        for i in range(len(propagated_points)):
            diff = (propagated_points[i] - self.state).reshape(-1, 1)
            self.covariance += self.Wc[i] * diff @ diff.T
            
    def generate_sigma_points(self):
        """Generate sigma points for UKF"""
        sigma_points = [self.state]  # Start with mean
        
        # Calculate matrix square root
        U = np.linalg.cholesky((self.state_dim + self.lmbda) * self.covariance)
        
        # Generate points
        for i in range(self.state_dim):
            sigma_points.append(self.state + U[i, :])
            sigma_points.append(self.state - U[i, :])
            
        return np.array(sigma_points)
        
    def nonlinear_dynamics(self, state, dt):
        """Non-linear dynamics model"""
        new_state = state.copy()
        
        # Example: integrate position and velocity
        new_state[0:3] += state[3:6] * dt  # Update positions
        new_state[3:6] += state[6:9] * dt  # Update velocities based on accelerations (if applicable)
        
        return new_state
        
    def update(self, measurement):
        """Update step using unscented transform"""
        # Generate sigma points around current state
        sigma_points = self.generate_sigma_points()
        
        # Transform sigma points through measurement function
        measurement_points = []
        for point in sigma_points:
            meas_point = self.nonlinear_measurement_model(point)
            measurement_points.append(meas_point)
        
        # Calculate mean and covariance of measurements
        expected_measurement = sum(self.Wm[i] * measurement_points[i] 
                                 for i in range(len(measurement_points)))
        
        # Measurement covariance
        P_zz = self.R  # Measurement noise
        for i in range(len(measurement_points)):
            diff = (measurement_points[i] - expected_measurement).reshape(-1, 1)
            P_zz += self.Wc[i] * diff @ diff.T
            
        # Cross covariance
        P_xz = np.zeros((self.state_dim, len(expected_measurement)))
        for i in range(len(sigma_points)):
            x_diff = (sigma_points[i] - self.state).reshape(-1, 1)
            z_diff = (measurement_points[i] - expected_measurement).reshape(-1, 1)
            P_xz += self.Wc[i] * x_diff @ z_diff.T
            
        # Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)
        
        # Update state and covariance
        innovation = measurement - expected_measurement
        self.state = self.state + K @ innovation
        self.covariance = self.covariance - K @ P_zz @ K.T
        
    def nonlinear_measurement_model(self, state):
        """Non-linear measurement model"""
        # Example: measure position only
        return state[0:3]
```

### Information Filter

Information filters work with the inverse of the covariance matrix for better numerical stability:

```python
class InformationFilter:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        
        # Information state (inverse of covariance matrix times state)
        self.y = np.zeros(state_dim)  # Information state vector
        
        # Information matrix (inverse of covariance matrix)
        self.Y = np.eye(state_dim) * 0.1  # Initially low certainty
        self.Y = np.linalg.inv(np.eye(state_dim) * 10)  # Invert initial covariance
        
        # Process and measurement information matrices
        self.inv_Q = np.eye(state_dim) * 100  # High certainty in process model
        self.inv_R = np.eye(3) * 10         # Measurement information matrix
        
    def predict(self, F):
        """Prediction step in information form"""
        # Predict information matrix
        # Y_k|k-1 = (F @ Y_k-1 @ F.T + Q)^(-1)
        # In information form: Y_k|k-1 = (Y_k-1^(-1) - Q^(-1))^(-1)
        # This is equivalent to Y_k|k-1 = Y_k-1 - F.T @ (F @ Y_k-1^(-1) @ F.T + Q)^(-1) @ F
        # Use the information form equation:
        Y_inverse = np.linalg.inv(self.Y)
        predicted_Y_inv = F @ Y_inverse @ F.T + np.linalg.inv(self.inv_Q)
        self.Y = np.linalg.inv(predicted_Y_inv)
        
        # Predict information state
        # y_k|k-1 = Y_k|k-1 @ F @ Y_k-1^(-1) @ y_k-1
        Y_inv_prev = np.linalg.inv(self.Y)  # This should be the old Y before update
        # In practice, this is complex and often we use the standard form for prediction
        self.y = F @ np.linalg.inv(self.Y) @ self.y  # Simplified; real implementation more complex
        
    def update(self, measurement, H):
        """Update step in information form"""
        # Update information matrix
        self.Y = self.Y + H.T @ self.inv_R @ H
        
        # Update information state vector
        self.y = self.y + H.T @ self.inv_R @ measurement
```

## Particle Filtering

### Basic Particle Filter

Particle filters are effective for non-Gaussian, non-linear systems:

```python
class ParticleFilter:
    def __init__(self, state_dim, num_particles=1000):
        self.state_dim = state_dim
        self.num_particles = num_particles
        
        # Initialize particles randomly
        self.particles = np.random.normal(0, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        
        # Process noise
        self.process_noise = np.eye(state_dim) * 0.1
        
    def predict(self, control_input, dt):
        """Predict particle states based on control input"""
        for i in range(self.num_particles):
            # Apply motion model with noise
            self.particles[i] = self.motion_model(
                self.particles[i], control_input, dt
            ) + np.random.multivariate_normal(
                np.zeros(self.state_dim), self.process_noise
            )
            
    def motion_model(self, state, control, dt):
        """Model how state changes with control input"""
        new_state = state.copy()
        
        # Example: integrate position and velocity
        new_state[0:3] += state[3:6] * dt  # Update position
        new_state[3:6] += control[0:3] * dt  # Update velocity with control input
        
        return new_state
        
    def update(self, measurement):
        """Update particle weights based on measurement"""
        # Calculate likelihood of measurement for each particle
        for i in range(self.num_particles):
            predicted_measurement = self.measurement_model(self.particles[i])
            likelihood = self.calculate_likelihood(measurement, predicted_measurement)
            self.weights[i] *= likelihood
            
        # Normalize weights
        self.weights += 1e-300  # Avoid numerical issues
        self.weights /= np.sum(self.weights)
        
    def measurement_model(self, state):
        """Model what measurement should look like for given state"""
        # Example: measure position
        return state[0:3]
        
    def calculate_likelihood(self, measurement, predicted_measurement):
        """Calculate likelihood of measurement given prediction"""
        # Assume Gaussian measurement noise
        diff = measurement - predicted_measurement
        # Simplified likelihood calculation
        return np.exp(-0.5 * np.dot(diff, diff))  # Gaussian kernel
        
    def estimate_state(self):
        """Estimate state from particles"""
        # Weighted average of particles
        estimated_state = np.zeros(self.state_dim)
        for i in range(self.num_particles):
            estimated_state += self.weights[i] * self.particles[i]
        return estimated_state
        
    def resample(self):
        """Resample particles based on weights to avoid degeneracy"""
        # Systematic resampling
        indices = self.systematic_resample()
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)
        
    def systematic_resample(self):
        """Systematic resampling algorithm"""
        indices = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        
        # Generate uniform samples
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

### Rao-Blackwellized Particle Filter

For high-dimensional state spaces, this approach keeps some dimensions in parametric form:

```python
class RaoBlackwellizedParticleFilter:
    def __init__(self, discrete_dim, continuous_dim):
        self.discrete_dim = discrete_dim      # Dimensions handled by particles
        self.continuous_dim = continuous_dim  # Dimensions handled by Kalman filters
        
        self.num_particles = 100
        self.particles = np.random.rand(self.num_particles, discrete_dim)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # For each particle, maintain a Kalman filter for continuous dimensions
        self.kalman_filters = [
            ExtendedKalmanFilter(continuous_dim, 0) 
            for _ in range(self.num_particles)
        ]
        
    def predict(self, control_input, dt):
        """Predict step for both discrete and continuous dimensions"""
        # Predict discrete part using particles
        for i in range(self.num_particles):
            self.particles[i] = self.discrete_motion_model(
                self.particles[i], control_input, dt
            )
            
        # Predict continuous part for each particle
        for kf in self.kalman_filters:
            kf.predict(control_input, dt)
            
    def discrete_motion_model(self, state, control, dt):
        """Model for discrete dimensions"""
        # Example motion model
        new_state = state.copy()
        # Implement specific motion model for your application
        return new_state
        
    def update(self, measurement):
        """Update weights based on measurement"""
        for i in range(self.num_particles):
            # Update the corresponding Kalman filter
            self.kalman_filters[i].update(measurement, None)  # Simplified
            
            # Calculate likelihood of measurement given particle
            likelihood = self.calculate_particle_likelihood(i, measurement)
            self.weights[i] *= likelihood
            
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
    def calculate_particle_likelihood(self, particle_idx, measurement):
        """Calculate likelihood of measurement given a particle"""
        # The likelihood is calculated using the associated Kalman filter
        # This is a simplified version
        predicted_measurement = self.kalman_filters[particle_idx].state[0:3]  # Example
        diff = measurement - predicted_measurement
        likelihood = np.exp(-0.5 * np.dot(diff, diff))  # Gaussian
        return likelihood
        
    def estimate_state(self):
        """Estimate state using weighted combination of all particles"""
        # Estimate discrete part
        discrete_estimate = np.zeros(self.discrete_dim)
        for i in range(self.num_particles):
            discrete_estimate += self.weights[i] * self.particles[i]
            
        # Estimate continuous part (weighted combination of Kalman filter states)
        continuous_estimate = np.zeros(self.continuous_dim)
        for i in range(self.num_particles):
            continuous_estimate += self.weights[i] * self.kalman_filters[i].state
            
        return np.concatenate([discrete_estimate, continuous_estimate])
```

## Multi-Sensor Data Association

### Nearest Neighbor Data Association

```python
from scipy.optimize import linear_sum_assignment

class DataAssociation:
    def __init__(self):
        self.assignment_method = 'hungarian'  # or 'nearest_neighbor'
        
    def associate_detections(self, predictions, measurements, max_distance=1.0):
        """
        Associate predicted object locations with actual measurements
        """
        if len(predictions) == 0 or len(measurements) == 0:
            return [], []
            
        # Calculate distance matrix between predictions and measurements
        distance_matrix = np.zeros((len(predictions), len(measurements)))
        
        for i, pred in enumerate(predictions):
            for j, meas in enumerate(measurements):
                # Calculate distance between predicted and measured position
                distance_matrix[i][j] = np.linalg.norm(pred - meas)
                
        # Handle the association based on the chosen method
        if self.assignment_method == 'nearest_neighbor':
            return self.nearest_neighbor_assignment(distance_matrix, max_distance)
        elif self.assignment_method == 'hungarian':
            return self.hungarian_assignment(distance_matrix, max_distance)
            
    def nearest_neighbor_assignment(self, distance_matrix, max_distance):
        """Simple nearest neighbor assignment"""
        assignments = []
        used_measurements = set()
        
        for i in range(distance_matrix.shape[0]):  # For each prediction
            # Find closest unused measurement
            min_dist = float('inf')
            best_j = -1
            
            for j in range(distance_matrix.shape[1]):  # For each measurement
                if j not in used_measurements and distance_matrix[i][j] < min_dist:
                    min_dist = distance_matrix[i][j]
                    best_j = j
                    
            # If a valid association is found within distance threshold
            if best_j != -1 and min_dist < max_distance:
                assignments.append((i, best_j))
                used_measurements.add(best_j)
                
        return assignments, [i for i in range(len(distance_matrix)) 
                           if i not in [a[0] for a in assignments]]
                           
    def hungarian_assignment(self, distance_matrix, max_distance):
        """Optimal assignment using Hungarian algorithm"""
        # Use scipy's linear_sum_assignment
        pred_indices, meas_indices = linear_sum_assignment(distance_matrix)
        
        # Filter out assignments that exceed distance threshold
        valid_assignments = []
        for pred_idx, meas_idx in zip(pred_indices, meas_indices):
            if distance_matrix[pred_idx][meas_idx] < max_distance:
                valid_assignments.append((pred_idx, meas_idx))
                
        # Return valid assignments and unassigned predictions
        assigned_preds = {a[0] for a in valid_assignments}
        unassigned_preds = [i for i in range(len(distance_matrix)) 
                          if i not in assigned_preds]
                          
        return valid_assignments, unassigned_preds
```

### Joint Probabilistic Data Association (JPDA)

JPDA handles uncertainty in data association by considering all possible associations:

```python
class JPDAFilter:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
    def update_with_jpda(self, predicted_state, predicted_covariance, 
                        measurements, clutter_intensity=0.1, detection_prob=0.9):
        """
        Update state estimate using JPDA
        """
        # Calculate association probabilities for each measurement
        prob_assoc = []
        innovations = []
        innovation_covariances = []
        
        for meas in measurements:
            # Calculate innovation (difference between measurement and prediction)
            innovation = meas - predicted_state[0:3]  # Assuming first 3 dims are position
            S = predicted_covariance[0:3, 0:3] + np.eye(3) * 0.1  # Innovation covariance
            
            # Calculate likelihood of this measurement
            likelihood = multivariate_normal.pdf(innovation, 
                                               mean=np.zeros(3), 
                                               cov=S)
            prob_assoc.append(likelihood)
            innovations.append(innovation)
            innovation_covariances.append(S)
            
        # Normalize association probabilities
        if len(prob_assoc) > 0:
            # Add probability of no detection (for clutter)
            prob_detection = detection_prob
            prob_clutter = clutter_intensity  # Density of false measurements
            
            # Calculate association probabilities
            assoc_probs = [p / (sum(prob_assoc) + prob_clutter) for p in prob_assoc]
            
            # Add probability of no association (false measurement)
            prob_no_assoc = prob_clutter / (sum(prob_assoc) + prob_clutter)
            
            # Calculate updated state and covariance
            updated_state = self.calculate_jpda_update(
                predicted_state, innovations, assoc_probs, prob_no_assoc
            )
            
            updated_cov = self.calculate_jpda_cov_update(
                predicted_covariance, innovations, innovation_covariances, 
                assoc_probs, prob_no_assoc
            )
            
            return updated_state, updated_cov
            
        return predicted_state, predicted_covariance
        
    def calculate_jpda_update(self, predicted_state, innovations, assoc_probs, prob_no_assoc):
        """Calculate state update using JPDA"""
        # The JPDA update is a weighted sum of all possible associations
        innovation_weighted = np.zeros(self.state_dim)
        
        for i, innovation in enumerate(innovations):
            # Extend innovation to full state size
            full_innovation = np.zeros(self.state_dim)
            full_innovation[0:3] = innovation  # Assuming position measurements
            
            innovation_weighted += assoc_probs[i] * full_innovation
            
        # JPDA state update
        updated_state = predicted_state + innovation_weighted  # Simplified
        return updated_state
        
    def calculate_jpda_cov_update(self, predicted_cov, innovations, innovation_covs, 
                                assoc_probs, prob_no_assoc):
        """Calculate covariance update using JPDA"""
        # This is a simplified version; full JPDA covariance update is more complex
        updated_cov = predicted_cov  # Simplified
        return updated_cov
```

## Time Synchronization

### Hardware and Software Timestamping

```python
import time
from collections import deque

class TimeSynchronizer:
    def __init__(self, max_buffer_size=100):
        self.sensors = {}
        self.buffer_size = max_buffer_size
        self.time_offsets = {}  # Per-sensor time offsets
        self.buffer = {}
        
    def register_sensor(self, sensor_name, hardware_timestamping=False):
        """Register a sensor with the synchronizer"""
        self.sensors[sensor_name] = {
            'hardware_timestamping': hardware_timestamping,
            'buffer': deque(maxlen=self.buffer_size)
        }
        self.time_offsets[sensor_name] = 0.0  # Initialize offset
        
    def add_reading(self, sensor_name, data, timestamp=None):
        """Add a sensor reading with its timestamp"""
        if timestamp is None:
            timestamp = time.time()  # Use system time if not provided
            
        # Apply time offset correction
        corrected_timestamp = timestamp + self.time_offsets[sensor_name]
        
        # Add to buffer
        entry = {
            'data': data,
            'timestamp': corrected_timestamp,
            'raw_timestamp': timestamp
        }
        
        self.sensors[sensor_name]['buffer'].append(entry)
        
    def calibrate_time_offsets(self, calibration_event_callback, duration=10.0):
        """Calibrate time offsets between sensors"""
        print("Starting time offset calibration...")
        
        # Record a known event across all sensors
        event_time = time.time()
        event_data = calibration_event_callback()
        
        # Wait for the event to be recorded in all sensors
        time.sleep(0.5)
        
        # Calculate time differences
        for sensor_name, sensor_info in self.sensors.items():
            if sensor_info['buffer']:
                latest_reading = sensor_info['buffer'][-1]
                measured_offset = event_time - latest_reading['raw_timestamp']
                self.time_offsets[sensor_name] = measured_offset
                
        print("Time offset calibration complete.")
        for sensor, offset in self.time_offsets.items():
            print(f"  {sensor}: {offset:.6f}s")
            
    def get_synchronized_readings(self, target_time, time_window=0.01):
        """Get synchronized readings from all sensors around target time"""
        synchronized_data = {}
        
        for sensor_name, sensor_info in self.sensors.items():
            # Find the closest reading to target_time within time_window
            closest_reading = self.find_closest_reading(
                sensor_info['buffer'], target_time, time_window
            )
            
            if closest_reading is not None:
                synchronized_data[sensor_name] = {
                    'data': closest_reading['data'],
                    'timestamp': closest_reading['timestamp']
                }
            else:
                # No reading found in time window
                synchronized_data[sensor_name] = None
                
        return synchronized_data
        
    def find_closest_reading(self, buffer, target_time, time_window):
        """Find the closest reading to target time within time window"""
        if not buffer:
            return None
            
        best_reading = None
        min_diff = float('inf')
        
        for entry in buffer:
            time_diff = abs(entry['timestamp'] - target_time)
            if time_diff < time_window and time_diff < min_diff:
                min_diff = time_diff
                best_reading = entry
                
        return best_reading
```

### Temporal Data Fusion

```python
class TemporalFusion:
    def __init__(self, fusion_horizon=0.1):  # 100ms fusion horizon
        self.fusion_horizon = fusion_horizon
        self.synchronizer = TimeSynchronizer()
        self.temporal_models = {}
        
    def register_sensor_type(self, sensor_type, temporal_model):
        """Register how to process data from a particular sensor type"""
        self.temporal_models[sensor_type] = temporal_model
        
    def fuse_temporal_data(self, sensor_readings):
        """Fuse sensor data across time"""
        # Group readings by sensor type
        grouped_readings = {}
        for sensor_name, reading in sensor_readings.items():
            if reading is not None:  # Only include valid readings
                sensor_type = self.get_sensor_type(sensor_name)
                if sensor_type not in grouped_readings:
                    grouped_readings[sensor_type] = []
                grouped_readings[sensor_type].append(reading)
                
        # Apply temporal fusion for each sensor type
        fused_results = {}
        for sensor_type, readings in grouped_readings.items():
            if sensor_type in self.temporal_models:
                fused_results[sensor_type] = self.temporal_models[sensor_type](readings)
            else:
                # Default fusion: use most recent reading
                fused_results[sensor_type] = readings[-1]['data']
                
        return fused_results
        
    def get_sensor_type(self, sensor_name):
        """Extract sensor type from sensor name"""
        # Example: 'camera_left' -> 'camera', 'imu_torso' -> 'imu'
        return sensor_name.split('_')[0]
        
    def predict_state_to_common_time(self, sensor_readings, common_time):
        """Predict all sensor readings to a common time"""
        predicted_readings = {}
        
        for sensor_name, reading in sensor_readings.items():
            if reading is not None:
                # Predict the reading to common_time using temporal model
                predicted_readings[sensor_name] = self.predict_to_time(
                    reading, common_time
                )
            else:
                predicted_readings[sensor_name] = None
                
        return predicted_readings
        
    def predict_to_time(self, reading, target_time):
        """Predict a reading to a target time using temporal model"""
        # This would use the sensor's temporal dynamics model
        # For example, integrate velocity to predict position
        # Simplified implementation:
        return reading
```

## Sensor Calibration and Registration

### Extrinsic Calibration

```python
class ExtrinsicCalibrator:
    def __init__(self):
        self.transforms = {}  # Store transforms between sensors
        
    def calibrate_cameras(self, left_camera, right_camera, calibration_board):
        """Calibrate stereo camera setup"""
        # Capture images from both cameras simultaneously
        left_images = []
        right_images = []
        
        print("Capturing calibration images...")
        for i in range(30):  # Capture 30 image pairs
            left_img, right_img = self.capture_stereo_pair(left_camera, right_camera)
            left_images.append(left_img)
            right_images.append(right_img)
            print(f"Captured pair {i+1}/30")
            time.sleep(0.5)
            
        # Prepare object points (real-world coordinates of calibration board)
        pattern_size = (9, 6)  # Chessboard pattern
        objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        left_imgpoints = []  # 2d points in image plane for left camera
        right_imgpoints = []  # 2d points in image plane for right camera
        
        # Find corners in all images
        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size, None)
            
            # If found, add object points and image points
            if ret_left and ret_right:
                objpoints.append(objp)
                
                # Refine corner positions
                corners_left = cv2.cornerSubPix(
                    left_gray, corners_left, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                left_imgpoints.append(corners_left)
                
                corners_right = cv2.cornerSubPix(
                    right_gray, corners_right, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                right_imgpoints.append(corners_right)
        
        # Perform stereo calibration
        if len(objpoints) > 5:  # Need at least 5 valid image pairs
            ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, \
                R, T, E, F = cv2.stereoCalibrate(
                    objpoints, left_imgpoints, right_imgpoints,
                    camera_matrix_left=self.get_camera_matrix(left_camera),
                    dist_coeffs_left=self.get_dist_coeffs(left_camera),
                    camera_matrix_right=self.get_camera_matrix(right_camera),
                    dist_coeffs_right=self.get_dist_coeffs(right_camera),
                    image_size=left_gray.shape[::-1],
                    flags=cv2.CALIB_FIX_INTRINSIC
                )
                
            # Store transformation (rotation R and translation T)
            self.transforms['left_to_right'] = {'R': R, 'T': T, 'E': E, 'F': F}
            
            return ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right
        
        return False, None, None, None, None
        
    def calibrate_imu_to_camera(self, camera, imu, calibration_target):
        """Calibrate transformation between IMU and camera"""
        # This requires a calibration target with known visual features
        # that can be tracked while the IMU measures orientation changes
        
        # The algorithm would:
        # 1. Track visual features across multiple camera orientations
        # 2. Record corresponding IMU measurements
        # 3. Solve for the rigid transformation between the two sensors
        
        # Simplified implementation would use a known calibration object
        # and solve AX=XB problem for hand-eye calibration
        pass
        
    def get_camera_matrix(self, camera):
        """Get camera matrix from camera parameters"""
        # This would interface with the camera's intrinsic calibration
        return np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])  # Placeholder
        
    def get_dist_coeffs(self, camera):
        """Get distortion coefficients from camera calibration"""
        # This would interface with the camera's distortion calibration
        return np.zeros(5)  # Placeholder
    
    def capture_stereo_pair(self, left_camera, right_camera):
        """Capture synchronized stereo image pair (simplified)"""
        # In real implementation, this would capture from actual cameras
        # For simulation, return placeholder images
        left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        right_img = np.zeros((480, 640, 3), dtype=np.uint8)
        return left_img, right_img
```

### Online Calibration

```python
class OnlineCalibrator:
    def __init__(self):
        self.calibration_data = {}
        self.is_calibrated = False
        
    def update_calibration(self, sensor_data, truth_data=None):
        """Update calibration parameters as new data comes in"""
        # If truth data is available, use it for calibration
        if truth_data is not None:
            self.compute_calibration_adjustment(sensor_data, truth_data)
        else:
            # Use self-calibration techniques if available
            self.self_calibration_update(sensor_data)
            
    def compute_calibration_adjustment(self, sensor_data, truth_data):
        """Compute adjustment to calibration parameters"""
        # Example: Adjust for drift in IMU bias
        for sensor_name, measurements in sensor_data.items():
            if sensor_name not in self.calibration_data:
                self.calibration_data[sensor_name] = {}
                
            # Update bias estimates using Kalman filter approach
            if 'gyro_bias' not in self.calibration_data[sensor_name]:
                self.calibration_data[sensor_name]['gyro_bias'] = np.zeros(3)
                
            # Compare with truth data to estimate bias
            predicted = self.apply_calibration(measurements, sensor_name)
            error = truth_data[sensor_name] - predicted
            
            # Update bias estimate
            # Simplified bias update
            self.calibration_data[sensor_name]['gyro_bias'] += error * 0.001
            
    def self_calibration_update(self, sensor_data):
        """Update calibration using internal consistency checks"""
        # Use the fact that sensor readings should be consistent
        # across different sensors and time steps
        pass
        
    def apply_calibration(self, raw_data, sensor_name):
        """Apply current calibration to raw sensor data"""
        if sensor_name in self.calibration_data:
            bias = self.calibration_data[sensor_name].get('gyro_bias', np.zeros(3))
            return raw_data - bias
        else:
            return raw_data
```

## Bayesian Sensor Fusion

### Bayesian Networks for Sensor Fusion

```python
class BayesianSensorFusion:
    def __init__(self):
        self.variables = {}
        self.conditional_probabilities = {}
        self.priors = {}
        
    def add_sensor_variable(self, sensor_name, state_dim):
        """Add a sensor variable to the Bayesian network"""
        self.variables[sensor_name] = {
            'dimension': state_dim,
            'type': 'continuous',  # or 'discrete'
            'evidence': None
        }
        
        # Initialize prior belief
        self.priors[sensor_name] = {
            'mean': np.zeros(state_dim),
            'covariance': np.eye(state_dim)
        }
        
    def define_sensor_dependencies(self, child_sensor, parent_sensors):
        """Define conditional dependencies between sensors"""
        # Define how child sensor readings depend on parent sensor readings
        self.conditional_probabilities[child_sensor] = {
            'parents': parent_sensors,
            'conditional_model': self.linear_gaussian_model
        }
        
    def linear_gaussian_model(self, parent_values, noise_cov):
        """Linear Gaussian conditional probability model"""
        # y = A*x + b + noise
        # where x is concatenated parent values
        if len(parent_values) > 0:
            combined_parent = np.concatenate(parent_values)
            A = np.eye(len(combined_parent))  # Simplified: identity
            b = np.zeros(len(combined_parent))
            mean = A @ combined_parent + b
            return mean, noise_cov
        else:
            return np.zeros(3), noise_cov  # Default values
            
    def infer_state(self, evidence_dict):
        """Infer state using Bayesian inference"""
        # Update evidence
        for sensor, evidence in evidence_dict.items():
            if sensor in self.variables:
                self.variables[sensor]['evidence'] = evidence
                
        # Perform inference (simplified using Kalman filtering approaches)
        # In a real implementation, this would use message passing or other inference methods
        return self.approximate_bayesian_update(evidence_dict)
        
    def approximate_bayesian_update(self, evidence_dict):
        """Approximate Bayesian update using Kalman filter equivalent"""
        # Start with prior
        posterior_mean = self.priors['state']['mean'].copy()
        posterior_cov = self.priors['state']['covariance'].copy()
        
        # Update with each piece of evidence
        for sensor_name, measurement in evidence_dict.items():
            # Calculate Kalman gain equivalent
            innovation = measurement - posterior_mean
            innovation_cov = posterior_cov + self.get_sensor_noise(sensor_name)
            kalman_gain = posterior_cov @ np.linalg.inv(innovation_cov)
            
            # Update estimate
            posterior_mean = posterior_mean + kalman_gain @ innovation
            posterior_cov = posterior_cov - kalman_gain @ innovation_cov @ kalman_gain.T
            
        return posterior_mean, posterior_cov
        
    def get_sensor_noise(self, sensor_name):
        """Get noise characteristics for sensor"""
        # This would be specific to each sensor type
        noise_levels = {
            'camera': np.eye(3) * 0.01,     # 1cm accuracy
            'lidar': np.eye(3) * 0.005,     # 5mm accuracy
            'imu': np.eye(6) * 0.1,         # Higher uncertainty
            'gps': np.eye(3) * 2.0          # 2m accuracy
        }
        return noise_levels.get(sensor_name, np.eye(3))
```

### Markov Chain Monte Carlo for Fusion

```python
class MCMCSensorFusion:
    def __init__(self, state_dim, num_samples=10000):
        self.state_dim = state_dim
        self.num_samples = num_samples
        self.samples = np.zeros((num_samples, state_dim))
        
    def fuse_with_mcmc(self, sensor_readings, initial_state=None):
        """Fuse sensor readings using Markov Chain Monte Carlo"""
        if initial_state is None:
            initial_state = np.zeros(self.state_dim)
            
        # Initialize Markov chain
        current_state = initial_state
        accepted_count = 0
        
        for i in range(self.num_samples):
            # Generate proposal state
            proposal = self.proposal_distribution(current_state)
            
            # Calculate acceptance probability
            current_likelihood = self.calculate_likelihood(current_state, sensor_readings)
            proposal_likelihood = self.calculate_likelihood(proposal, sensor_readings)
            
            acceptance_ratio = min(1, proposal_likelihood / current_likelihood)
            
            # Accept or reject proposal
            if np.random.uniform(0, 1) < acceptance_ratio:
                current_state = proposal
                accepted_count += 1
                
            # Store sample
            self.samples[i] = current_state.copy()
            
        # Calculate acceptance rate
        acceptance_rate = accepted_count / self.num_samples
        print(f"MCMC acceptance rate: {acceptance_rate:.3f}")
        
        # Return statistics of the posterior distribution
        mean_estimate = np.mean(self.samples, axis=0)
        covariance_estimate = np.cov(self.samples, rowvar=False)
        
        return mean_estimate, covariance_estimate
        
    def proposal_distribution(self, current_state):
        """Generate a new state proposal"""
        # Use Gaussian random walk
        noise = np.random.normal(0, 0.1, size=self.state_dim)
        return current_state + noise
        
    def calculate_likelihood(self, state, sensor_readings):
        """Calculate likelihood of sensor readings given state"""
        log_likelihood = 0.0
        
        for sensor_name, reading in sensor_readings.items():
            if reading is not None:
                # Predict what this sensor should measure given the state
                predicted_reading = self.predict_sensor_reading(state, sensor_name)
                
                # Calculate likelihood of actual reading given prediction
                diff = reading - predicted_reading
                sensor_noise = self.get_sensor_noise(sensor_name)
                
                # Gaussian likelihood
                log_likelihood += self.gaussian_log_likelihood(diff, sensor_noise)
                
        return np.exp(log_likelihood)  # Return probability (not log)
        
    def predict_sensor_reading(self, state, sensor_name):
        """Predict what a sensor should read given state"""
        # This depends on sensor type
        if 'camera' in sensor_name:
            # Extract position part of state
            return state[0:3]  # x, y, z
        elif 'imu' in sensor_name:
            # Extract orientation part of state
            return state[3:6]  # For example, roll, pitch, yaw
        else:
            return state[0:3]  # Default to position
            
    def gaussian_log_likelihood(self, diff, covariance):
        """Calculate log-likelihood of a Gaussian distribution"""
        d = len(diff)
        normalization = d * np.log(2 * np.pi) + np.log(np.linalg.det(covariance))
        exponent = diff.T @ np.linalg.inv(covariance) @ diff
        return -0.5 * (normalization + exponent)
        
    def get_sensor_noise(self, sensor_name):
        """Get noise characteristics for sensor"""
        # Similar to previous implementation
        noise_levels = {
            'camera': np.eye(3) * 0.01,
            'lidar': np.eye(3) * 0.005,
            'imu': np.eye(3) * 0.1,
            'gps': np.eye(3) * 2.0
        }
        return noise_levels.get(sensor_name, np.eye(3))
```

## Deep Learning for Sensor Fusion

### Neural Network-Based Fusion

```python
import torch
import torch.nn as nn

class NeuralSensorFusion(nn.Module):
    def __init__(self, sensor_dims, output_dim):
        super(NeuralSensorFusion, self).__init__()
        
        self.sensor_dims = sensor_dims
        self.output_dim = output_dim
        
        # Create input layers for each sensor
        self.sensor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            ) for dim in sensor_dims
        ])
        
        # Fusion layer - combines encoded sensor information
        total_encoded_dim = 64 * len(sensor_dims)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoded_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, sensor_inputs):
        """Fusion of sensor inputs using neural network"""
        encoded_features = []
        
        # Encode each sensor's input
        for i, (encoder, sensor_input) in enumerate(zip(self.sensor_encoders, sensor_inputs)):
            encoded = encoder(sensor_input)
            encoded_features.append(encoded)
            
        # Combine all encoded features
        combined_features = torch.cat(encoded_features, dim=-1)
        
        # Apply fusion
        fused_output = self.fusion_layer(combined_features)
        
        return fused_output
        
    def train_fusion_network(self, training_data, labels, epochs=100):
        """Train the fusion network"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_idx, (sensor_inputs_batch, targets) in enumerate(zip(training_data, labels)):
                # Forward pass
                outputs = self.forward(sensor_inputs_batch)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(training_data):.4f}")

class AttentionBasedFusion(nn.Module):
    def __init__(self, sensor_dims, output_dim):
        super(AttentionBasedFusion, self).__init__()
        
        self.sensor_dims = sensor_dims
        self.output_dim = output_dim
        
        # Linear transformations for each sensor
        self.sensor_transforms = nn.ModuleList([
            nn.Linear(dim, 64) for dim in sensor_dims
        ])
        
        # Attention mechanism
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4,
            dropout=0.1
        )
        
        # Output layer
        self.output_projection = nn.Linear(64, output_dim)
        
    def forward(self, sensor_inputs):
        """Fusion with attention mechanism"""
        transformed_inputs = []
        
        # Transform each sensor input
        for transform, sensor_input in zip(self.sensor_transforms, sensor_inputs):
            transformed = transform(sensor_input)
            transformed_inputs.append(transformed)
            
        # Stack for attention mechanism
        stacked_inputs = torch.stack(transformed_inputs, dim=0)  # [num_sensors, batch, features]
        
        # Apply attention
        attended_output, attention_weights = self.attention_layer(
            stacked_inputs, stacked_inputs, stacked_inputs
        )
        
        # Pool the attention outputs (e.g., take the mean)
        pooled_output = torch.mean(attended_output, dim=0)
        
        # Project to output dimension
        final_output = self.output_projection(pooled_output)
        
        return final_output
```

### Deep Learning-Based State Estimation

```python
class DeepStateEstimator(nn.Module):
    def __init__(self, input_dim, state_dim, sequence_length):
        super(DeepStateEstimator, self).__init__()
        
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        
        # Encoder for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder to state space
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, state_dim)
        )
        
    def forward(self, sensor_sequences):
        """Estimate state from sensor sequences"""
        # sensor_sequences shape: (batch_size, seq_len, feature_dim)
        
        # Process sequence through LSTM
        lstm_out, (hidden, cell) = self.lstm(sensor_sequences)
        
        # Use the last output for state estimation
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Decode to state space
        state_estimate = self.decoder(last_output)
        
        return state_estimate, lstm_out, hidden
        
    def train_estimator(self, sensor_data, true_states, epochs=100):
        """Train the state estimator"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_data, batch_states in zip(sensor_data, true_states):
                # Forward pass
                state_pred, _, _ = self(batch_data)
                loss = criterion(state_pred, batch_states)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(sensor_data):.4f}")
```

## Summary

This chapter has covered the fundamental and advanced concepts of sensor fusion and integration for humanoid robotics. We explored the mathematical foundations of sensor fusion, including probabilistic approaches and the need for combining information from diverse sensors to achieve improved accuracy, reliability, and robustness.

We delved into various filtering techniques essential for sensor fusion, from classical Kalman filtering approaches to more advanced methods like Extended Kalman Filter, Unscented Kalman Filter, and Particle Filters. Each method has its place in humanoid robotics depending on the linearity of the system and the distribution of uncertainties.

The chapter addressed the critical issue of data association - the process of determining which measurements correspond to which objects or features - and presented solutions like nearest neighbor assignment and Joint Probabilistic Data Association.

Time synchronization was covered as a crucial element for multi-sensor systems, ensuring that measurements from different sensors correspond to the same point in time for accurate fusion. We explored both hardware and software approaches to achieving proper synchronization.

Sensor calibration and registration were discussed as prerequisites for accurate sensor fusion, covering both offline calibration procedures and online self-calibration techniques to handle drift and changes in sensor characteristics over time.

Advanced probabilistic approaches like Bayesian networks and Markov Chain Monte Carlo methods were presented for situations requiring rigorous uncertainty handling.

Finally, we explored modern deep learning approaches to sensor fusion, including neural networks with attention mechanisms that can learn complex fusion strategies from data, potentially outperforming traditional model-based approaches.

Effective sensor fusion is essential for humanoid robots to operate reliably in complex environments, providing the integrated awareness necessary for dexterous manipulation, stable locomotion, and safe human interaction.

## Exercises

1. Implement an Extended Kalman Filter to fuse data from a camera and IMU for position and orientation estimation.

2. Design a sensor fusion architecture for a humanoid robot that combines visual, inertial, and force/torque sensors to estimate the state of a grasped object.

3. Create a deep learning model that learns to fuse multimodal sensor data from different timesteps to predict the robot's future state.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*