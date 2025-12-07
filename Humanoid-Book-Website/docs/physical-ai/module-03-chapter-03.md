---
id: module-03-chapter-03
title: Chapter 03 - Deep Learning for Humanoid Control
sidebar_position: 11
---

# Chapter 03 - Deep Learning for Humanoid Control

## Table of Contents
- [Overview](#overview)
- [Control Challenges in Humanoid Robotics](#control-challenges-in-humanoid-robotics)
- [Classical Control vs. Deep Learning](#classical-control-vs-deep-learning)
- [Deep Reinforcement Learning for Locomotion](#deep-reinforcement-learning-for-locomotion)
- [Neural Networks for Motion Planning](#neural-networks-for-motion-planning)
- [Learning Motor Skills](#learning-motor-skills)
- [Humanoid-Specific Control Architectures](#humanoid-specific-control-architectures)
- [Safety and Stability in Deep Learning Control](#safety-and-stability-in-deep-learning-control)
- [Transfer Learning and Domain Adaptation](#transfer-learning-and-domain-adaptation)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Deep learning has revolutionized the field of humanoid robotics control, enabling robots to learn complex behaviors, adapt to new situations, and perform tasks that were previously impossible with traditional control methods. This chapter explores how deep neural networks are applied to control the complex dynamics of humanoid robots, from basic balance and locomotion to sophisticated manipulation and interaction tasks.

Traditional control approaches for humanoid robots rely on precise mathematical models and predefined control laws. While effective in many scenarios, these approaches struggle with the complexity, variability, and uncertainty of real-world environments. Deep learning approaches, on the other hand, can learn control policies directly from data, making them more adaptable and robust to environmental variations and modeling errors.

## Control Challenges in Humanoid Robotics

### Complexity of Humanoid Dynamics

Humanoid robots present unique control challenges due to their complex dynamics:

1. **High Degrees of Freedom (DoF)**: Humanoid robots typically have 20-40+ joints, creating a high-dimensional control space
2. **Underactuation**: Humanoid robots are often underactuated during walking, making balance control difficult
3. **Nonlinear Dynamics**: Humanoid systems have complex, nonlinear dynamics with many coupled joints
4. **Contact Dynamics**: Walking involves dynamic contact with the environment, creating discontinuous dynamics
5. **Real-time Constraints**: Control decisions must be made at high frequency to maintain balance (typically >1000 Hz)

### Stability and Safety Requirements

```python
class StabilityController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.zmp_calculator = ZMPCalculator(robot_model)
        self.capture_point_calculator = CapturePointCalculator(robot_model)
        
    def ensure_stability(self, state, desired_action):
        """Ensure that the desired action maintains stability"""
        # Calculate current Zero Moment Point (ZMP)
        current_zmp = self.zmp_calculator.calculate(state.joint_positions, 
                                                   state.joint_velocities, 
                                                   state.joint_torques)
        
        # Calculate capture point for balance
        capture_point = self.capture_point_calculator.calculate(state.com_position, 
                                                              state.com_velocity)
        
        # Define support polygon based on foot positions
        support_polygon = self.calculate_support_polygon(state)
        
        # Check if ZMP is within support polygon
        if not self.is_in_support_polygon(current_zmp, support_polygon):
            # Modify action to ensure stability
            desired_action = self.modify_for_stability(desired_action, state)
            
        # Verify that modified action maintains stability
        future_state = self.simulate_action(desired_action, state)
        future_zmp = self.zmp_calculator.calculate(future_state.joint_positions,
                                                 future_state.joint_velocities,
                                                 future_state.joint_torques)
        
        if not self.is_in_support_polygon(future_zmp, support_polygon):
            # Fall back to basic balance behavior
            desired_action = self.get_balance_action(state)
            
        return desired_action
```

### Real-time Performance Requirements

Humanoid control systems must operate under strict real-time constraints:

- **Balance Control**: 1000 Hz (1ms response time)
- **Locomotion Planning**: 100-200 Hz (5-10ms processing time)
- **Manipulation Control**: 500-1000 Hz (1-2ms response time)
- **High-level Planning**: 10-50 Hz (20-100ms planning time)

### Adaptation to New Situations

Humanoid robots must adapt to:
- Uneven terrain
- Uncertain object properties
- Dynamic environments
- Changing body states (e.g., carrying loads)
- Wear and tear of mechanical components

## Classical Control vs. Deep Learning

### Limitations of Classical Control

Classical control approaches for humanoid robots include:

1. **PID Controllers**: Simple but limited to linear, time-invariant systems
2. **LQR/LQG**: Optimal for linear systems but requires accurate models
3. **Computed Torque Control**: Linearizes dynamics but sensitive to model errors
4. **ZMP-based Control**: Elegant for walking but limited to quasi-static motions

```python
class ClassicalController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.pid_controllers = self.initialize_pid_controllers()
        self.zmp_controller = ZMPController(robot_model)
        
    def compute_control_torques(self, state, reference):
        """Compute control torques using classical methods"""
        # PID control for joint position tracking
        pid_torques = self.compute_pid_torques(state, reference)
        
        # Add feedforward torques for dynamics compensation
        feedforward_torques = self.compute_feedforward_torques(state, reference)
        
        # Add balance control if needed
        balance_torques = self.zmp_controller.compute_balance_torques(state, reference)
        
        total_torques = pid_torques + feedforward_torques + balance_torques
        
        return total_torques
        
    def compute_pid_torques(self, state, reference):
        """Compute torques using PID control"""
        errors = reference.positions - state.joint_positions
        d_errors = reference.velocities - state.joint_velocities
        
        # PID control law
        torques = (self.kp * errors + 
                  self.kd * d_errors + 
                  self.ki * self.integrated_errors)
                  
        return torques
```

### Advantages of Deep Learning Control

Deep learning approaches offer several advantages:

1. **Learning from Experience**: Can improve performance over time
2. **Robustness**: Handle uncertainties and disturbances better
3. **Adaptation**: Adjust to changes in robot dynamics
4. **Generalization**: Apply learned skills to new situations
5. **Complex Behavior**: Learn sophisticated control strategies

### Hybrid Approaches

Modern humanoid control often combines classical and deep learning methods:

```python
class HybridControlSystem:
    def __init__(self, robot_model):
        self.classical_controller = ClassicalController(robot_model)
        self.deep_controller = DeepController(robot_model)
        self.arbiter = ControllerArbiter(robot_model)
        
    def compute_control(self, state, task):
        """Compute control using hybrid approach"""
        # Use classical controller for safety-critical aspects
        safety_control = self.classical_controller.compute_safe_control(state)
        
        # Use deep controller for complex behavior learning
        behavior_control = self.deep_controller.compute_behavior(state, task)
        
        # Combine using task-specific arbiter
        final_control = self.arbiter.combine_controls(
            safety_control, 
            behavior_control, 
            state, 
            task
        )
        
        return final_control
```

## Deep Reinforcement Learning for Locomotion

### Reinforcement Learning Framework

Deep RL formulates humanoid control as a Markov Decision Process (MDP):

```python
import torch
import torch.nn as nn
import numpy as np

class HumanoidLocomotionEnvironment:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.target_velocity = 1.0  # m/s
        
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Convert neural network output to joint torques
        torques = self.action_to_torques(action)
        
        # Apply torques to robot simulation
        next_state = self.robot_model.simulate_step(torques)
        
        # Calculate reward based on walking performance
        reward = self.calculate_locomotion_reward(next_state)
        
        # Check termination conditions
        done = self.check_termination(next_state)
        
        # Additional information
        info = self.get_environment_info(next_state)
        
        return next_state, reward, done, info
        
    def calculate_locomotion_reward(self, state):
        """Calculate reward for locomotion task"""
        # Velocity reward - encourage forward movement
        forward_vel = self.get_forward_velocity(state)
        velocity_reward = 0.1 * max(0, forward_vel)  # Only reward forward movement
        
        # Balance reward - penalize deviation from upright position
        com_pos = state.center_of_mass_position
        com_vel = state.center_of_mass_velocity
        upright_reward = -0.1 * (abs(com_pos[1]) + abs(com_vel[1]))  # Penalize roll/pitch deviation
        
        # Energy efficiency - penalize excessive actuation
        energy_penalty = -0.001 * np.sum(np.abs(state.joint_torques))
        
        # Joint limit penalty - penalize approaching joint limits
        joint_limit_penalty = -0.1 * self.get_joint_limit_violation(state)
        
        # Alive bonus - encourage long walking episodes
        alive_bonus = 0.1
        
        total_reward = (velocity_reward + upright_reward + 
                       energy_penalty + joint_limit_penalty + alive_bonus)
        
        return total_reward
```

### Deep Deterministic Policy Gradient (DDPG)

DDPG is effective for continuous control tasks like humanoid locomotion:

```python
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        q = self.l3(q)
        return q

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action)
        self.critic = DDPGCritic(state_dim, action_dim)
        self.critic_target = DDPGCritic(state_dim, action_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer()
        self.discount = 0.99
        self.tau = 0.005
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()
        
    def train(self, batch_size=100):
        """Train actor and critic networks"""
        # Sample replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Compute target Q-value
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (done * self.discount * target_Q).detach()
        
        # Get current Q estimate
        current_Q = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### Proximal Policy Optimization (PPO)

PPO is another popular algorithm for humanoid control:

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, k_epochs=80, eps_clip=0.2):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def update(self, memory):
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).detach())
        old_actions = torch.squeeze(torch.stack(memory.actions).detach())
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
```

### Deep Q-Network (DQN) for Discrete Actions

For some humanoid tasks with discrete action spaces:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HumanoidDQN(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(HumanoidDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.qnetwork_local = HumanoidDQN(state_size, action_size, seed)
        self.qnetwork_target = HumanoidDQN(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)
        
        # Experience replay
        self.memory = ReplayBuffer(action_size, 10000, 64, seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            # If enough samples available in memory, get random subset and learn
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, 0.99)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Neural Networks for Motion Planning

### Learning Motion Primitives

```python
class MotionPrimitiveLearner:
    def __init__(self, primitive_dim, context_dim):
        # Network to generate motion primitives
        self.primitive_generator = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, primitive_dim),
        )
        
        # Network to blend motion primitives
        self.blender = nn.Sequential(
            nn.Linear(primitive_dim * 2 + context_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, primitive_dim),
        )
        
    def generate_primitive(self, context):
        """Generate a motion primitive based on context"""
        context_tensor = torch.FloatTensor(context)
        primitive = self.primitive_generator(context_tensor)
        return primitive
        
    def blend_primitives(self, primitive1, primitive2, context):
        """Blend two motion primitives"""
        blend_input = torch.cat([
            primitive1, 
            primitive2, 
            torch.FloatTensor(context)
        ])
        blended_primitive = self.blender(blend_input)
        return blended_primitive
```

### Deep Path Planning Networks

```python
class DeepPathPlanner(nn.Module):
    def __init__(self, environment_dim, robot_dim, max_path_length):
        super(DeepPathPlanner, self).__init__()
        self.max_path_length = max_path_length
        
        # Process environment information
        self.env_encoder = nn.Sequential(
            nn.Linear(environment_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        # Process robot state
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        
        # Combine environment and robot information
        self.combiner = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # Generate path waypoints
        self.path_generator = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        self.waypoint_predictor = nn.Linear(256, 3)  # x, y, theta
        
    def forward(self, env_info, robot_state, goal):
        """Generate a path from current state to goal"""
        # Encode environment information
        env_features = self.env_encoder(env_info)
        
        # Encode robot state
        robot_features = self.robot_encoder(robot_state)
        
        # Combine information
        combined_features = self.combiner(
            torch.cat([env_features, robot_features], dim=-1)
        )
        
        # Prepare input for RNN
        # Repeat combined features for each time step
        rnn_input = combined_features.unsqueeze(1).repeat(1, self.max_path_length, 1)
        
        # Generate path waypoints
        path_features, _ = self.path_generator(rnn_input)
        waypoints = self.waypoint_predictor(path_features)
        
        return waypoints
```

### Model Predictive Control with Neural Networks

```python
class NeuralMPC:
    def __init__(self, prediction_horizon=10, control_horizon=5):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        
        # Dynamics model: predicts next state given current state and action
        self.dynamics_model = self.build_dynamics_model()
        
        # Cost function: evaluates state-action pairs
        self.cost_model = self.build_cost_model()
        
    def build_dynamics_model(self):
        """Build neural network for predicting system dynamics"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
    def build_cost_model(self):
        """Build neural network for computing costs"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def optimize_control_sequence(self, current_state, goal):
        """Optimize sequence of control actions"""
        # Initialize control sequence randomly
        control_sequence = torch.randn(self.control_horizon, action_dim, requires_grad=True)
        
        optimizer = torch.optim.Adam([control_sequence], lr=0.1)
        
        for iteration in range(50):  # Optimization iterations
            optimizer.zero_grad()
            
            total_cost = 0
            state = current_state.clone()
            
            # Simulate system forward using dynamics model
            for t in range(self.prediction_horizon):
                if t < self.control_horizon:
                    action = control_sequence[t]
                else:
                    action = torch.zeros(action_dim)  # Zero after control horizon
                
                # Predict next state
                state = self.dynamics_model(torch.cat([state, action]))
                
                # Compute cost
                cost = self.cost_model(torch.cat([state, action]))
                total_cost += cost
                
            # Backpropagate and update control sequence
            total_cost.backward()
            optimizer.step()
            
        # Return first control action (MPC principle)
        return control_sequence[0]
```

## Learning Motor Skills

### Skill Embedding Networks

```python
class SkillLearningNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_skills):
        super(SkillLearningNetwork, self).__init__()
        
        # Encoder for state and action
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Skill identifier network
        self.skill_identifier = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills),
            nn.Softmax(dim=-1)
        )
        
        # Skill-specific controllers
        self.skill_controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            ) for _ in range(num_skills)
        ])
        
    def forward(self, state):
        # Get skill probabilities for this state
        state_features = self.state_encoder(state)
        skill_probs = torch.ones(self.num_skills) / self.num_skills  # Simplified
        
        # Get action from each skill controller
        skill_actions = []
        for controller in self.skill_controllers:
            action = controller(state)
            skill_actions.append(action)
            
        # Weight actions by skill probabilities
        skill_actions = torch.stack(skill_actions, dim=1)  # [batch, num_skills, action_dim]
        weighted_action = torch.sum(skill_probs.unsqueeze(-1) * skill_actions, dim=1)
        
        return weighted_action, skill_probs
```

### Imitation Learning for Motor Skills

```python
class ImitationLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-3)
        
    def train_step(self, states, actions):
        """Single training step for imitation learning"""
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        
        # Get predicted actions
        predicted_actions = self.policy_network(states_tensor)
        
        # Compute imitation loss (MSE between predicted and expert actions)
        loss = nn.MSELoss()(predicted_actions, actions_tensor)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train_behavior_cloning(self, expert_data, epochs=100):
        """Train policy using behavior cloning"""
        states, actions = expert_data
        
        for epoch in range(epochs):
            loss = self.train_step(states, actions)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Imitation Loss: {loss:.4f}")
```

### Transfer Learning of Skills

```python
class SkillTransferNetwork(nn.Module):
    def __init__(self, source_task_dim, target_task_dim):
        super(SkillTransferNetwork, self).__init__()
        
        # Shared representation
        self.shared_encoder = nn.Sequential(
            nn.Linear(source_task_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Task-specific decoder for source
        self.source_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, source_task_dim)
        )
        
        # Task-specific decoder for target
        self.target_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, target_task_dim)
        )
        
        # Skill transfer network
        self.transfer_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def forward(self, source_state):
        # Encode source state in shared representation
        shared_repr = self.shared_encoder(source_state)
        
        # Reconstruct source output
        source_output = self.source_decoder(shared_repr)
        
        # Transform for target task
        transferred_repr = self.transfer_network(shared_repr)
        target_output = self.target_decoder(transferred_repr)
        
        return source_output, target_output
```

## Humanoid-Specific Control Architectures

### Hierarchical Control Architecture

```python
class HierarchicalHumanoidController:
    def __init__(self, robot_model):
        self.high_level_planner = HighLevelPlanner(robot_model)
        self.mid_level_controller = MidLevelController(robot_model)
        self.low_level_controller = LowLevelController(robot_model)
        self.integration_layer = IntegrationLayer(robot_model)
        
    def compute_action(self, task, state, context):
        """Compute action through hierarchical control"""
        # High-level planning: what to do
        high_level_goal = self.high_level_planner.generate_goal(task, context)
        
        # Mid-level control: how to achieve high-level goal
        mid_level_commands = self.mid_level_controller.generate_commands(
            high_level_goal, state
        )
        
        # Low-level control: motor commands for joints
        low_level_commands = self.low_level_controller.generate_joint_commands(
            mid_level_commands, state
        )
        
        # Integration: combine and validate commands
        final_action = self.integration_layer.integrate_commands(
            low_level_commands, state, task
        )
        
        return final_action

class HighLevelPlanner:
    def __init__(self, robot_model):
        self.goal_generator = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, goal_dim)
        )
        
    def generate_goal(self, task, context):
        """Generate high-level goal based on task and context"""
        context_tensor = torch.FloatTensor(context)
        goal = self.goal_generator(context_tensor)
        return goal

class MidLevelController:
    def __init__(self, robot_model):
        self.command_generator = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, command_dim)
        )
        
    def generate_commands(self, goal, state):
        """Generate mid-level commands to achieve goal"""
        state_tensor = torch.FloatTensor(state)
        goal_tensor = torch.FloatTensor(goal)
        commands = self.command_generator(torch.cat([state_tensor, goal_tensor]))
        return commands

class LowLevelController:
    def __init__(self, robot_model):
        self.joint_command_generator = nn.Sequential(
            nn.Linear(state_dim + command_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, joint_action_dim)
        )
        
    def generate_joint_commands(self, commands, state):
        """Generate joint-level commands"""
        state_tensor = torch.FloatTensor(state)
        command_tensor = torch.FloatTensor(commands)
        joint_commands = self.joint_command_generator(
            torch.cat([state_tensor, command_tensor])
        )
        return joint_commands
```

### Modular Control Architecture

```python
class ModularHumanoidController:
    def __init__(self, robot_model):
        self.modules = {
            'balance': BalanceModule(robot_model),
            'locomotion': LocomotionModule(robot_model),
            'manipulation': ManipulationModule(robot_model),
            'social_interaction': SocialInteractionModule(robot_model)
        }
        
        # Arbitration system for combining module outputs
        self.arbiter = ModuleArbiter(list(self.modules.keys()))
        
    def compute_action(self, task, context, state):
        """Compute action using modular architecture"""
        module_outputs = {}
        
        # Run each module
        for module_name, module in self.modules.items():
            activation = self.compute_module_activation(module_name, task, context, state)
            if activation > 0.1:  # Only run modules that are relevant
                module_outputs[module_name] = module.compute(state, task, activation)
        
        # Arbitrate between module outputs
        final_action = self.arbiter.combine_outputs(module_outputs, state, task)
        
        return final_action
        
    def compute_module_activation(self, module_name, task, context, state):
        """Compute how much each module should be active"""
        # Use a neural network to determine module activation
        activation_network = self.build_activation_network(module_name)
        activation_input = self.prepare_activation_input(task, context, state)
        activation = activation_network(activation_input)
        return torch.sigmoid(activation).item()
        
    def prepare_activation_input(self, task, context, state):
        """Prepare input for activation computation"""
        # Combine task, context, and state information
        task_encoding = self.encode_task(task)
        context_tensor = torch.FloatTensor(context)
        state_tensor = torch.FloatTensor(state)
        
        return torch.cat([task_encoding, context_tensor, state_tensor])

class BalanceModule:
    def __init__(self, robot_model):
        self.balance_controller = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, joint_action_dim)
        )
        
    def compute(self, state, task, activation):
        """Compute balance actions"""
        state_tensor = torch.FloatTensor(state)
        base_action = self.balance_controller(state_tensor)
        return base_action * activation  # Scale by activation
```

## Safety and Stability in Deep Learning Control

### Safety-First Control Framework

```python
class SafeDeepController:
    def __init__(self, robot_model):
        self.deep_policy = DeepPolicyNetwork()
        self.safety_checker = SafetyChecker(robot_model)
        self.backup_controller = BackupController(robot_model)
        self.stability_threshold = 0.8
        
    def compute_safe_action(self, state):
        """Compute action with safety guarantees"""
        # Get action from deep policy
        proposed_action = self.deep_policy(state)
        
        # Check safety of proposed action
        safety_status = self.safety_checker.check_action(proposed_action, state)
        
        if safety_status.is_safe and safety_status.stability > self.stability_threshold:
            return proposed_action
        else:
            # Use backup controller or modify action for safety
            if safety_status.is_critical:
                safe_action = self.backup_controller.emergency_stop(state)
            else:
                safe_action = self.modify_action_for_safety(
                    proposed_action, state, safety_status
                )
                
            return safe_action
            
    def modify_action_for_safety(self, action, state, safety_status):
        """Modify action to satisfy safety constraints"""
        # Project action to safe region
        safe_action = action.clone()
        
        # Apply constraints based on safety violations
        for constraint_type, violation in safety_status.violations.items():
            if violation > 0:
                safe_action = self.apply_constraint(
                    safe_action, constraint_type, violation, state
                )
                
        return safe_action
```

### Constrained Policy Optimization

```python
class ConstrainedPolicyOptimization(nn.Module):
    def __init__(self, state_dim, action_dim, constraint_dim):
        super(ConstrainedPolicyOptimization, self).__init__()
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Constraint networks (one for each safety constraint)
        self.constraint_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Output probability of constraint violation
            ) for _ in range(constraint_dim)
        ])
        
        # Cost network
        self.cost_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        """Generate action following policy"""
        action = self.policy(state)
        return action
        
    def compute_cost_and_constraints(self, state, action):
        """Compute cost and constraint violations"""
        state_action = torch.cat([state, action], dim=-1)
        
        # Compute cost
        cost = self.cost_network(state_action)
        
        # Compute constraint violations
        constraint_violations = []
        for constraint_net in self.constraint_networks:
            violation = constraint_net(state_action)
            constraint_violations.append(violation)
            
        return cost, torch.stack(constraint_violations, dim=-1)
```

### Lyapunov-Based Stability

```python
class LyapunovStableController:
    def __init__(self, robot_model):
        self.lyapunov_network = self.build_lyapunov_network()
        self.controller_network = self.build_controller_network()
        self.gamma = 0.99  # Discount factor for stability
        
    def build_lyapunov_network(self):
        """Build network to learn Lyapunov function"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensure positive output
        )
        
    def build_controller_network(self):
        """Build network for control policy"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def compute_lyapunov_value(self, state):
        """Compute Lyapunov function value"""
        return self.lyapunov_network(state)
        
    def compute_control(self, state):
        """Compute control action that decreases Lyapunov function"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get current Lyapunov value
        V_current = self.compute_lyapunov_value(state_tensor)
        
        # Find action that decreases Lyapunov function
        best_action = None
        best_V_next = float('inf')
        
        # Sample multiple actions to find one that decreases Lyapunov function
        for _ in range(100):
            action = torch.randn(1, action_dim) * 0.1  # Random action with small variance
            next_state = self.simulate_state_transition(state_tensor, action)
            V_next = self.compute_lyapunov_value(next_state)
            
            if V_next < V_current and V_next < best_V_next:
                best_V_next = V_next
                best_action = action
        
        return best_action.squeeze(0) if best_action is not None else self.controller_network(state_tensor).squeeze(0)
        
    def simulate_state_transition(self, state, action):
        """Simulate one step of state transition"""
        # This would interface with the robot dynamics model
        return state + 0.001 * action  # Simplified for example
```

## Transfer Learning and Domain Adaptation

### Domain Randomization

```python
class DomainRandomizedController:
    def __init__(self, robot_model):
        self.controller = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.domain_randomizer = DomainRandomizer()
        
    def train_with_randomization(self, env, episodes=10000):
        """Train controller with randomized domain parameters"""
        for episode in range(episodes):
            # Randomize domain parameters for this episode
            randomized_params = self.domain_randomizer.randomize()
            env.update_dynamics(randomized_params)
            
            # Train on randomized environment
            episode_reward = 0
            state = env.reset()
            
            for step in range(1000):  # Max steps per episode
                action = self.controller(torch.FloatTensor(state))
                next_state, reward, done, _ = env.step(action.numpy())
                
                # Store transition for learning
                self.update_policy(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
                    
            if episode % 1000 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
                
    def evaluate_on_real_robot(self, real_env):
        """Evaluate controller trained with domain randomization"""
        # Controller trained with domain randomization should be robust to real-world variations
        total_reward = 0
        for episode in range(10):
            state = real_env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = self.controller(torch.FloatTensor(state))
                state, reward, done, _ = real_env.step(action.numpy())
                episode_reward += reward
                
                if done:
                    break
                    
            total_reward += episode_reward
            print(f"Real robot episode {episode}: {episode_reward}")
            
        avg_reward = total_reward / 10
        print(f"Average reward on real robot: {avg_reward}")
```

### Sim-to-Real Transfer

```python
class SimToRealTransfer:
    def __init__(self, sim_robot, real_robot):
        self.sim_controller = self.create_controller()
        self.real_controller = self.create_controller()  # Same architecture but different weights
        
        # Domain adaptation network
        self.domain_adaptor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
        # Cycle consistency network
        self.sim_to_real = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
        self.real_to_sim = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
    def adapt_controller(self, sim_data, real_data):
        """Adapt controller from simulation to reality"""
        # Train domain adaptation networks
        self.train_domain_adaptation(sim_data, real_data)
        
        # Fine-tune controller on real data
        self.fine_tune_on_real_data(real_data)
        
    def train_domain_adaptation(self, sim_data, real_data):
        """Train networks to adapt between simulation and reality"""
        optimizer = torch.optim.Adam(
            list(self.sim_to_real.parameters()) + 
            list(self.real_to_sim.parameters()),
            lr=1e-4
        )
        
        for epoch in range(100):
            # Cycle consistency loss
            sim_tensor = torch.FloatTensor(sim_data)
            real_tensor = torch.FloatTensor(real_data)
            
            # Sim to real to sim cycle
            real_pred = self.sim_to_real(sim_tensor)
            sim_recon = self.real_to_sim(real_pred)
            cycle_sim_loss = nn.MSELoss()(sim_tensor, sim_recon)
            
            # Real to sim to real cycle
            sim_pred = self.real_to_sim(real_tensor)
            real_recon = self.sim_to_real(sim_pred)
            cycle_real_loss = nn.MSELoss()(real_tensor, real_recon)
            
            # Total cycle loss
            cycle_loss = cycle_sim_loss + cycle_real_loss
            
            optimizer.zero_grad()
            cycle_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Domain adaptation epoch {epoch}, cycle loss: {cycle_loss.item():.4f}")
                
    def fine_tune_on_real_data(self, real_data):
        """Fine-tune controller on real-world data"""
        # Use the adapted simulation model as starting point
        self.real_controller.load_state_dict(self.sim_controller.state_dict())
        
        # Train on real data with smaller learning rate
        optimizer = torch.optim.Adam(self.real_controller.parameters(), lr=1e-5)
        
        for epoch in range(50):
            # Training code here
            pass
```

## Summary

This chapter has explored how deep learning transforms humanoid robot control, moving from traditional model-based approaches to learning-based methods that can adapt and improve with experience. We've covered reinforcement learning algorithms like DDPG and PPO that enable humanoid robots to learn complex behaviors like walking and manipulation through trial and error. 

The chapter discussed neural networks for motion planning and how they can learn to generate effective motion trajectories. We explored how humanoid robots can learn motor skills through imitation learning and how these skills can be transferred between different tasks and robots. 

The importance of safety and stability in deep learning control was emphasized, with techniques for ensuring that learned controllers remain stable and safe. Finally, we covered transfer learning and domain adaptation techniques that allow controllers trained in simulation to work effectively on real robots.

Deep learning has enabled humanoid robots to achieve behaviors that were previously impossible with classical control methods, opening new possibilities for human-robot interaction and task performance in unstructured environments.

## Exercises

1. Implement a Deep Q-Network (DQN) controller for a simplified 2D bipedal walker and train it to walk forward as fast as possible.

2. Design a hierarchical control architecture for a humanoid robot that needs to navigate through a room and pick up objects, with separate modules for navigation, manipulation, and balance.

3. Create a domain randomization environment for training a humanoid walking controller, where physical parameters like friction, mass, and actuator properties are randomly varied during training.

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*