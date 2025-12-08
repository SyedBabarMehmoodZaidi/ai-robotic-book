---
sidebar_position: 5
---

# Reinforcement Learning for Locomotion: AI-Driven Movement Control

## Learning Objectives

By the end of this section, you will be able to:
- Understand reinforcement learning fundamentals for robotics applications
- Implement deep reinforcement learning algorithms for locomotion control
- Train bipedal robots using Isaac Gym and reinforcement learning
- Create perception-action loops for dynamic locomotion
- Evaluate and optimize locomotion policies for real-world deployment
- Integrate learned locomotion policies with navigation and perception systems

## Introduction to Reinforcement Learning in Robotics

**Reinforcement Learning (RL)** is a branch of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. In robotics, RL has emerged as a powerful approach for learning complex behaviors, particularly locomotion, manipulation, and navigation tasks that are difficult to program using traditional methods.

Reinforcement learning is particularly well-suited for robotics because it:
- Learns from trial and error in simulation
- Adapts to environmental changes and disturbances
- Generalizes across different terrains and conditions
- Optimizes complex, multi-objective behaviors
- Handles high-dimensional continuous action spaces

### Key RL Concepts for Robotics

#### Agent-Environment Interaction
```
┌─────────────┐    Action    ┌──────────────┐
│             │ ────────────▶│              │
│   Robot     │              │  Environment │
│   (Agent)   │ ◀────────────│   (World)    │
│             │  Reward,Obs  │              │
└─────────────┘              └──────────────┘
```

#### Core RL Components
- **State (s)**: Robot's current configuration (joint angles, velocities, IMU readings)
- **Action (a)**: Motor commands or joint torques applied to the robot
- **Reward (r)**: Feedback signal indicating task success or failure
- **Policy (π)**: Strategy that maps states to actions
- **Value Function (V)**: Expected future rewards from a given state

## Isaac Gym for Robotics RL

### Introduction to Isaac Gym

**Isaac Gym** is NVIDIA's physics simulation environment specifically designed for reinforcement learning. It provides:
- GPU-accelerated physics simulation (hundreds of environments in parallel)
- Direct integration with PyTorch for seamless RL training
- Built-in RL examples for various robotic tasks
- Support for contact sensors, IMUs, and other proprioceptive sensors
- Domain randomization capabilities for robust policy learning

### Isaac Gym Architecture

```python
# Isaac Gym environment structure
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

class IsaacGymEnvironment:
    def __init__(self):
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()

        # Create simulation
        self.sim = self.gym.create_sim(
            device_id=0,
            pipeline='gpu'
        )

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.Vec3(0, 0, 1))

        # Initialize environment parameters
        self.envs = []
        self.num_envs = 1024  # Parallel environments
        self.env_spacing = 2.0
```

### Installing Isaac Gym

```bash
# Install Isaac Gym
pip install isaacgym

# Verify installation
python -c "import isaacgym; print('Isaac Gym installed successfully')"
```

## Deep Reinforcement Learning Algorithms

### Proximal Policy Optimization (PPO)

PPO is one of the most popular algorithms for continuous control tasks like locomotion:

```python
# ppo_locomotion_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are normalized to [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Action standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        value = self.critic(state)
        return action_mean, action_std, value

    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def update(self, states, actions, rewards, log_probs, values, dones):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Calculate advantages
        with torch.no_grad():
            _, _, current_values = self.actor_critic(states)
            advantages = rewards + self.gamma * current_values.squeeze() * (1 - dones.float()) - old_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy
        for _ in range(4):  # PPO epochs
            action_means, action_stds, current_values = self.actor_critic(states)
            dist = torch.distributions.Normal(action_means, action_stds)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratios = torch.exp(new_log_probs - old_log_probs)

            # PPO loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = self.MseLoss(current_values.squeeze(), rewards + self.gamma * old_values.squeeze() * (1 - dones.float()))

            total_loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

### Twin Delayed DDPG (TD3) for Locomotion

TD3 is another effective algorithm for continuous control:

```python
# td3_locomotion_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

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

## Bipedal Locomotion Training

### Environment Setup for Bipedal Robots

```python
# bipedal_env.py
import numpy as np
import torch
import gym
from gym import spaces

class BipedalLocomotionEnv(gym.Env):
    def __init__(self):
        super(BipedalLocomotionEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32  # 12 joint torques
        )

        # Observation space: joint positions, velocities, IMU readings, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )

        # Robot parameters
        self.max_episode_steps = 1000
        self.current_step = 0

        # Reward parameters
        self.forward_weight = 1.0
        self.velocity_weight = 0.5
        self.energy_weight = 0.01
        self.balance_weight = 0.3

    def reset(self):
        # Reset robot to initial position
        self.current_step = 0
        self.robot_state = self.initialize_robot()

        return self.get_observation()

    def step(self, action):
        # Apply action to robot
        self.apply_action(action)

        # Simulate physics
        self.simulate_step()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if episode is done
        done = self.is_episode_done()

        # Update step count
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        # Return current robot state as observation
        obs = np.concatenate([
            self.robot_state.joint_positions,
            self.robot_state.joint_velocities,
            self.robot_state.imu_readings,
            self.robot_state.contact_states
        ])
        return obs.astype(np.float32)

    def calculate_reward(self):
        # Forward velocity reward
        forward_vel = self.robot_state.linear_velocity[0]
        forward_reward = self.forward_weight * forward_vel

        # Velocity tracking reward
        target_vel = 1.0  # Target forward velocity
        vel_reward = self.velocity_weight * max(0, 1.0 - abs(forward_vel - target_vel))

        # Energy efficiency penalty
        energy_penalty = self.energy_weight * np.sum(np.abs(self.robot_state.torques))

        # Balance reward (keep torso upright)
        torso_angle = self.robot_state.imu_readings[0]  # Roll angle
        balance_reward = self.balance_weight * max(0, np.cos(torso_angle))

        # Total reward
        total_reward = forward_reward + vel_reward - energy_penalty + balance_reward

        return total_reward

    def is_episode_done(self):
        # Check if robot has fallen
        torso_height = self.robot_state.torso_position[2]
        if torso_height < 0.3:  # Robot has fallen
            return True

        # Check if robot is walking backwards
        forward_vel = self.robot_state.linear_velocity[0]
        if forward_vel < -0.5:  # Walking backwards too fast
            return True

        return False

    def apply_action(self, action):
        # Apply torques to robot joints
        self.robot_state.torques = action

    def simulate_step(self):
        # Simulate one physics step
        # This would interface with the physics engine
        pass

    def initialize_robot(self):
        # Initialize robot in standing position
        # This would interface with the physics engine
        pass
```

### Training Loop Implementation

```python
# locomotion_training.py
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class LocomotionTrainer:
    def __init__(self, agent, env, max_episodes=2000):
        self.agent = agent
        self.env = env
        self.max_episodes = max_episodes

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []

    def train(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0

            done = False
            while not done:
                # Select action
                action = self.agent.select_action(state)

                # Add noise for exploration
                noise = np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action + noise, -1, 1)

                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)

                # Store transition
                self.agent.replay_buffer.push(state, action, next_state, reward, done)

                # Update agent
                if len(self.agent.replay_buffer) > 1000:
                    loss = self.agent.train()
                    if loss is not None:
                        episode_loss += loss

                state = next_state
                episode_reward += reward
                step_count += 1

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
            if step_count > 0:
                self.episode_losses.append(episode_loss / step_count)

            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

            # Save best model
            if episode % 500 == 0:
                self.save_model(f"locomotion_model_episode_{episode}.pth")

        # Plot training curves
        self.plot_training_curves()

    def save_model(self, filepath):
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'episode_rewards': self.episode_rewards
        }, filepath)

    def plot_training_curves(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')

        # Plot losses
        ax3.plot(self.episode_losses)
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')

        plt.tight_layout()
        plt.savefig('locomotion_training_curves.png')
        plt.show()
```

## Isaac Gym Integration for Locomotion

### Isaac Gym Bipedal Environment

```python
# isaac_gym_bipedal.py
import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
import numpy as np

class IsaacGymBipedalEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg["device"]

        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(
            device_id=0,
            pipeline='gpu'
        )

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.Vec3(0, 0, 1))

        # Initialize environments
        self._create_envs()

        # Initialize tensors
        self._initialize_tensors()

        # RL parameters
        self.dt = 1.0 / 60.0  # Physics update rate
        self.max_episode_length = 1000

    def _create_envs(self):
        # Create terrain
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # Load bipedal robot asset
        asset_root = "path/to/bipedal/robot"
        asset_file = "bipedal_robot.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        self.bipedal_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # Create environments
        self.envs = []
        num_envs = self.cfg["num_envs"]
        spacing = 2.0

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, 1)

            # Add robot to environment
            pos = gymapi.Vec3(0.0, 0.0, 1.0)
            pose = gymapi.Transform.from_rotation_translation(
                gymapi.Quat.from_euler_zyx(0, 0, 0), pos
            )

            actor_handle = self.gym.create_actor(
                env, self.bipedal_asset, pose, "bipedal", i, 0
            )

            # Configure DOF properties
            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"].fill(0.0)
            dof_props["damping"].fill(10.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            self.envs.append(env)

    def _initialize_tensors(self):
        # Get gym tensors
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.actor_root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.dof_command_tensor = self.gym.acquire_dof_command_force_tensor(self.sim)

        # Wrap tensors in torch tensors
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor).view(
            self.cfg["num_envs"], -1, 2
        )
        self.root_states = gymtorch.wrap_tensor(self.actor_root_tensor).view(
            self.cfg["num_envs"], -1, 13
        )

        # Initialize command tensor
        self.dof_commands = torch.zeros(
            self.cfg["num_envs"], self.cfg["num_dof"],
            device=self.device, dtype=torch.float32
        )

    def get_observation(self):
        # Extract observation from current state
        # Joint positions and velocities
        joint_pos = self.dof_state[:, :, 0]
        joint_vel = self.dof_state[:, :, 1]

        # Root state (position, orientation, linear velocity, angular velocity)
        root_pos = self.root_states[:, 0, 0:3]
        root_orn = self.root_states[:, 0, 3:7]
        root_lin_vel = self.root_states[:, 0, 7:10]
        root_ang_vel = self.root_states[:, 0, 10:13]

        # Combine observations
        obs = torch.cat([
            joint_pos,
            joint_vel,
            root_pos,
            root_orn,
            root_lin_vel,
            root_ang_vel
        ], dim=-1)

        return obs

    def apply_action(self, actions):
        # Apply actions to robot joints
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_commands)
        )

    def reset(self):
        # Reset all environments to initial state
        pass

    def step(self, actions):
        # Apply actions
        self.apply_action(actions)

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Update tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # Get observations
        obs = self.get_observation()

        # Calculate rewards
        rewards = self.calculate_rewards()

        # Check if done
        dones = self.check_termination()

        return obs, rewards, dones, {}

    def calculate_rewards(self):
        # Calculate rewards based on robot performance
        # This is a simplified reward function
        joint_pos = self.dof_state[:, :, 0]
        joint_vel = self.dof_state[:, :, 1]

        # Forward velocity reward
        root_lin_vel = self.root_states[:, 0, 7:10]
        forward_vel = root_lin_vel[:, 0]  # x-axis velocity
        forward_reward = torch.clamp(forward_vel, 0.0, 2.0)

        # Energy penalty
        energy_penalty = torch.sum(torch.square(joint_vel), dim=1) * 0.001

        # Balance reward
        root_orn = self.root_states[:, 0, 3:7]
        roll, pitch = get_euler_xyz(root_orn)
        balance_reward = torch.cos(pitch) * torch.cos(roll) * 2.0

        # Combined reward
        rewards = forward_reward - energy_penalty + balance_reward

        return rewards

    def check_termination(self):
        # Check if episode should terminate
        root_pos = self.root_states[:, 0, 0:3]
        root_orn = self.root_states[:, 0, 3:7]

        # Check if robot has fallen
        roll, pitch = get_euler_xyz(root_orn)
        fallen = torch.abs(pitch) > 1.0  # Fallen if pitch > 60 degrees

        # Check if robot is too low
        too_low = root_pos[:, 2] < 0.3  # Fallen if z < 0.3

        return torch.logical_or(fallen, too_low)
```

## Perception-Action Integration

### Vision-Based Locomotion

```python
# vision_locomotion.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VisionLocomotionPolicy(nn.Module):
    def __init__(self, action_dim, visual_feature_dim=512):
        super(VisionLocomotionPolicy, self).__init__()

        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, visual_feature_dim),
            nn.ReLU()
        )

        # Proprioceptive encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(24, 128),  # 12 joint positions + 12 joint velocities
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Combined policy
        self.policy = nn.Sequential(
            nn.Linear(visual_feature_dim + 128 + 6, 256),  # +6 for IMU
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, visual_input, proprio_input, imu_input):
        visual_features = self.visual_encoder(visual_input)
        proprio_features = self.proprio_encoder(proprio_input)

        combined_features = torch.cat([
            visual_features,
            proprio_features,
            imu_input
        ], dim=-1)

        return self.policy(combined_features)

class VisionLocomotionAgent:
    def __init__(self, action_dim=12):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = VisionLocomotionPolicy(action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_action(self, image, proprio_data, imu_data):
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        proprio_tensor = torch.FloatTensor(proprio_data).unsqueeze(0).to(self.device)
        imu_tensor = torch.FloatTensor(imu_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy(image_tensor, proprio_tensor, imu_tensor)

        return action.cpu().numpy().flatten()

    def train_step(self, batch_images, batch_proprio, batch_imu, batch_actions):
        # Convert to tensors
        images = torch.stack([self.transform(img) for img in batch_images]).to(self.device)
        proprio = torch.FloatTensor(batch_proprio).to(self.device)
        imu = torch.FloatTensor(batch_imu).to(self.device)
        actions = torch.FloatTensor(batch_actions).to(self.device)

        # Forward pass
        predicted_actions = self.policy(images, proprio, imu)

        # Compute loss
        loss = torch.nn.functional.mse_loss(predicted_actions, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## Policy Transfer and Deployment

### Sim-to-Real Transfer

```python
# sim_to_real_transfer.py
import torch
import numpy as np
from collections import deque

class SimToRealTransfer:
    def __init__(self, sim_policy_path, real_robot_interface):
        # Load simulation policy
        self.sim_policy = torch.load(sim_policy_path)

        # Real robot interface
        self.real_robot = real_robot_interface

        # Domain randomization parameters
        self.domain_params = {
            'friction': [0.5, 1.5],  # Randomize friction in sim
            'mass': [0.8, 1.2],      # Randomize mass in sim
            'com_offset': [-0.02, 0.02]  # Randomize center of mass
        }

        # Adaptation networks
        self.adaptation_net = self._create_adaptation_network()

    def _create_adaptation_network(self):
        """Create network to adapt sim policy to real robot"""
        return torch.nn.Sequential(
            torch.nn.Linear(48, 128),  # Observation space
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12),  # Action space
            torch.nn.Tanh()
        )

    def collect_real_data(self, num_episodes=10):
        """Collect data from real robot for adaptation"""
        real_data = []

        for episode in range(num_episodes):
            obs = self.real_robot.reset()
            episode_data = []

            for step in range(200):  # 200 steps per episode
                # Get action from sim policy
                action = self.sim_policy.get_action(obs)

                # Apply action to real robot
                next_obs, reward, done, info = self.real_robot.step(action)

                # Store transition
                episode_data.append({
                    'obs': obs,
                    'action': action,
                    'next_obs': next_obs,
                    'reward': reward
                })

                obs = next_obs

                if done:
                    break

            real_data.extend(episode_data)

        return real_data

    def adapt_policy(self, real_data):
        """Adapt policy using real robot data"""
        # Train adaptation network
        optimizer = torch.optim.Adam(self.adaptation_net.parameters(), lr=1e-3)

        for epoch in range(100):
            total_loss = 0

            for transition in real_data:
                obs_tensor = torch.FloatTensor(transition['obs']).unsqueeze(0)
                action_tensor = torch.FloatTensor(transition['action']).unsqueeze(0)

                predicted_action = self.adaptation_net(obs_tensor)
                loss = torch.nn.functional.mse_loss(predicted_action, action_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Adaptation epoch {epoch}, loss: {total_loss/len(real_data):.4f}")

    def deploy_policy(self):
        """Deploy adapted policy to real robot"""
        def policy_function(observation):
            # Use adapted policy for real robot
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action = self.adaptation_net(obs_tensor)
            return action.detach().numpy().flatten()

        return policy_function
```

## Performance Evaluation

### Locomotion Metrics

```python
# locomotion_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class LocomotionEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_gait(self, joint_positions, joint_velocities, time_steps):
        """Evaluate gait quality metrics"""
        # Calculate gait cycle parameters
        gait_cycle_duration = self.calculate_gait_cycle(joint_positions)

        # Calculate stability metrics
        stability = self.calculate_stability(joint_positions, joint_velocities)

        # Calculate energy efficiency
        energy_efficiency = self.calculate_energy_efficiency(joint_velocities)

        # Calculate forward velocity
        forward_velocity = self.calculate_forward_velocity(time_steps)

        self.metrics.update({
            'gait_cycle_duration': gait_cycle_duration,
            'stability': stability,
            'energy_efficiency': energy_efficiency,
            'forward_velocity': forward_velocity
        })

        return self.metrics

    def calculate_gait_cycle(self, joint_positions):
        """Calculate gait cycle duration from joint position data"""
        # Find peaks in hip joint movement to identify gait cycles
        hip_joint_data = joint_positions[:, 0]  # Assuming hip joint is first

        # Find peaks using scipy
        peaks, _ = signal.find_peaks(hip_joint_data, height=np.std(hip_joint_data))

        if len(peaks) > 1:
            # Calculate average cycle duration
            cycle_durations = np.diff(peaks)  # Time steps between peaks
            avg_cycle_duration = np.mean(cycle_durations)
            return avg_cycle_duration
        else:
            return 0  # Not enough cycles to calculate

    def calculate_stability(self, joint_positions, joint_velocities):
        """Calculate stability metrics"""
        # Zero Moment Point (ZMP) stability
        # Simplified calculation - in practice, this requires full kinematics
        com_position = self.calculate_center_of_mass(joint_positions)
        com_velocity = self.calculate_center_of_mass_velocity(joint_velocities)

        # Calculate stability margin
        stability_margin = np.std(com_position, axis=0)  # Lower std = more stable
        return 1.0 / (1.0 + np.mean(stability_margin))  # Higher is better

    def calculate_center_of_mass(self, joint_positions):
        """Calculate approximate center of mass"""
        # Simplified COM calculation
        # In practice, use full kinematic model with link masses
        return np.mean(joint_positions, axis=1)

    def calculate_center_of_mass_velocity(self, joint_velocities):
        """Calculate COM velocity"""
        return np.mean(joint_velocities, axis=1)

    def calculate_energy_efficiency(self, joint_velocities):
        """Calculate energy efficiency based on joint velocities"""
        # Energy is proportional to squared velocity
        energy_usage = np.mean(np.sum(np.square(joint_velocities), axis=1))
        return 1.0 / (1.0 + energy_usage)  # Higher is more efficient

    def calculate_forward_velocity(self, time_steps):
        """Calculate average forward velocity"""
        # This would be calculated from actual robot movement
        # For now, return a placeholder
        return 1.0  # m/s

    def plot_gait_analysis(self, joint_positions, time_steps):
        """Plot gait analysis"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot joint positions over time
        for i in range(min(6, joint_positions.shape[1])):  # Plot first 6 joints
            axes[0].plot(time_steps, joint_positions[:, i], label=f'Joint {i}')
        axes[0].set_title('Joint Positions Over Time')
        axes[0].set_ylabel('Position (rad)')
        axes[0].legend()

        # Plot gait phases
        axes[1].plot(time_steps, np.zeros_like(time_steps), 'k--', alpha=0.3)
        axes[1].set_title('Gait Phases')
        axes[1].set_ylabel('Phase')

        # Plot stability metrics
        stability = self.calculate_stability_over_time(joint_positions)
        axes[2].plot(time_steps, stability)
        axes[2].set_title('Stability Over Time')
        axes[2].set_ylabel('Stability')
        axes[2].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig('gait_analysis.png')
        plt.show()

    def calculate_stability_over_time(self, joint_positions):
        """Calculate stability metrics over time"""
        stability = []
        window_size = 10

        for i in range(len(joint_positions)):
            start_idx = max(0, i - window_size)
            window_data = joint_positions[start_idx:i+1]

            if len(window_data) > 1:
                stability_val = 1.0 / (1.0 + np.std(window_data))
                stability.append(stability_val)
            else:
                stability.append(1.0)

        return np.array(stability)

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        report = f"""
        Locomotion Performance Evaluation Report
        ========================================

        Gait Analysis:
        - Average gait cycle duration: {self.metrics.get('gait_cycle_duration', 'N/A')}
        - Stability score: {self.metrics.get('stability', 'N/A'):.3f}
        - Energy efficiency: {self.metrics.get('energy_efficiency', 'N/A'):.3f}
        - Forward velocity: {self.metrics.get('forward_velocity', 'N/A'):.2f} m/s

        Performance Assessment:
        """

        # Add qualitative assessment
        stability = self.metrics.get('stability', 0)
        if stability > 0.8:
            report += "- Excellent stability performance\n"
        elif stability > 0.6:
            report += "- Good stability performance\n"
        elif stability > 0.4:
            report += "- Adequate stability performance\n"
        else:
            report += "- Poor stability performance - requires improvement\n"

        velocity = self.metrics.get('forward_velocity', 0)
        if velocity > 1.0:
            report += "- Good forward speed performance\n"
        elif velocity > 0.5:
            report += "- Adequate forward speed performance\n"
        else:
            report += "- Low forward speed - may need optimization\n"

        return report
```

## Troubleshooting RL for Locomotion

### Common Issues and Solutions

#### 1. Training Instability
```python
# Solution: Implement curriculum learning
class CurriculumLearning:
    def __init__(self):
        self.difficulty_level = 0
        self.curriculum = [
            {'terrain': 'flat', 'speed': 0.5, 'duration': 500},
            {'terrain': 'flat', 'speed': 1.0, 'duration': 500},
            {'terrain': 'sloped', 'speed': 1.0, 'duration': 500},
            {'terrain': 'rough', 'speed': 1.0, 'duration': 500}
        ]

    def get_current_task(self):
        return self.curriculum[self.difficulty_level]

    def increase_difficulty(self, success_rate):
        if success_rate > 0.8 and self.difficulty_level < len(self.curriculum) - 1:
            self.difficulty_level += 1
            print(f"Increasing difficulty to level {self.difficulty_level}")
```

#### 2. Sim-to-Real Gap
```python
# Solution: Domain randomization
def apply_domain_randomization(env, episode):
    # Randomize physical parameters each episode
    friction = np.random.uniform(0.5, 1.5)
    mass_multiplier = np.random.uniform(0.8, 1.2)
    com_offset = np.random.uniform(-0.02, 0.02, 3)

    env.set_friction(friction)
    env.set_mass_multiplier(mass_multiplier)
    env.set_com_offset(com_offset)
```

#### 3. Reward Function Design
```python
# Well-designed reward function
def locomotion_reward(robot_state, action, prev_robot_state):
    # Forward velocity reward (primary objective)
    forward_vel = robot_state.linear_velocity[0]
    forward_reward = max(0, forward_vel) * 10.0

    # Energy efficiency penalty
    energy_penalty = np.sum(np.square(action)) * 0.01

    # Balance reward (keep upright)
    roll, pitch = robot_state.imu_euler
    balance_reward = np.cos(pitch) * np.cos(roll) * 5.0

    # Smoothness penalty
    action_diff = action - prev_robot_state.prev_action
    smoothness_penalty = np.sum(np.square(action_diff)) * 0.1

    # Avoid falling
    height_penalty = 0
    if robot_state.torso_height < 0.5:
        height_penalty = -100  # Large penalty for falling

    total_reward = forward_reward - energy_penalty + balance_reward - smoothness_penalty + height_penalty
    return total_reward
```

## Exercise: Reinforcement Learning Locomotion

1. Install Isaac Gym and set up a bipedal robot environment
2. Implement a PPO agent for locomotion control
3. Design a reward function that promotes stable forward walking
4. Train the agent in simulation with domain randomization
5. Evaluate the learned policy's performance using gait analysis
6. Implement sim-to-real transfer techniques for real robot deployment
7. Test the policy on different terrains and conditions
8. Analyze the gait patterns and stability metrics

This exercise will give you hands-on experience with AI-driven locomotion control.

## Summary

Reinforcement learning provides a powerful approach to learning complex locomotion behaviors that are difficult to program manually. By combining Isaac Gym's parallel simulation capabilities with deep RL algorithms like PPO and TD3, robots can learn to walk, run, and navigate diverse terrains. The key to success lies in proper reward function design, curriculum learning, and sim-to-real transfer techniques that bridge the gap between simulation and reality.

In the next section, we'll create exercises that integrate all the concepts from Module 3.