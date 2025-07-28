#!/usr/bin/env python3
"""
Unified example usage of both MuJoCo and Robotics environments.
This script demonstrates how to use both environment wrappers.
"""

import torch
import sys
import os

# Add the parent directory to the path to import the environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environment.gym_mujoco_env import GymMuJoCoEnv, get_available_mujoco_envs
from Environment.gym_robotics_env import (
    GymRoboticsEnv, 
    get_available_maze_envs,
    get_available_adroit_envs,
    get_available_franka_envs,
    get_available_robotics_envs
)


def main():
    """Main function demonstrating both MuJoCo and Robotics environment usage."""
    
    print("=" * 60)
    print("Gymnasium Environment Examples")
    print("=" * 60)
    
    # MuJoCo Environments
    print("\n1. MuJoCo Environments:")
    mujoco_envs = get_available_mujoco_envs()
    print(f"Available MuJoCo environments: {len(mujoco_envs)}")
    for env_name in mujoco_envs[:5]:  # Show first 5
        print(f"  - {env_name}")
    print(f"  ... and {len(mujoco_envs)-5} more")
    
    # Robotics Environments
    print("\n2. Robotics Environments:")
    maze_envs = get_available_maze_envs()
    adroit_envs = get_available_adroit_envs()
    franka_envs = get_available_franka_envs()
    all_robotics_envs = get_available_robotics_envs()
    
    print(f"Maze environments: {len(maze_envs)}")
    print(f"Adroit Arm environments: {len(adroit_envs)}")
    print(f"Franka Kitchen environments: {len(franka_envs)}")
    print(f"Total Robotics environments: {len(all_robotics_envs)}")
    
    print("\n" + "=" * 60)
    print("Example 1: MuJoCo Environment (HalfCheetah)")
    print("=" * 60)
    
    # Example 1: MuJoCo environment
    env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=2)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    print(f"Observation space size: {env.num_obs}")
    print(f"Action space size: {env.num_actions}")
    
    # Take some random steps
    for step in range(3):
        actions = torch.randn(2, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Rewards = {rewards.numpy()}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Example 2: Robotics Environment (Maze)")
    print("=" * 60)
    
    # Example 2: Robotics environment (Maze)
    env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=1)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    print(f"Observation space size: {env.num_obs}")
    print(f"Action space size: {env.num_actions}")
    
    # Take some random steps
    for step in range(3):
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Reward = {rewards.item():.3f}, Done = {dones.item()}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Example 3: Robotics Environment (Adroit Arm)")
    print("=" * 60)
    
    # Example 3: Robotics environment (Adroit Arm)
    env = GymRoboticsEnv("pen-human-v0", num_envs=1)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    print(f"Observation space size: {env.num_obs}")
    print(f"Action space size: {env.num_actions}")
    
    # Take some random steps
    for step in range(3):
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Reward = {rewards.item():.3f}, Done = {dones.item()}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Example 4: Robotics Environment (Franka Kitchen)")
    print("=" * 60)
    
    # Example 4: Robotics environment (Franka Kitchen)
    env = GymRoboticsEnv("kitchen-complete-v0", num_envs=1)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    print(f"Observation space size: {env.num_obs}")
    print(f"Action space size: {env.num_actions}")
    
    # Take some random steps
    for step in range(3):
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Reward = {rewards.item():.3f}, Done = {dones.item()}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Example 5: Multi-Environment Comparison")
    print("=" * 60)
    
    # Example 5: Compare different environment types
    env_configs = [
        ("MuJoCo", "HalfCheetah-v4", GymMuJoCoEnv),
        ("Maze", "maze2d-umaze-v1", GymRoboticsEnv),
        ("Adroit Arm", "pen-human-v0", GymRoboticsEnv),
        ("Franka Kitchen", "kitchen-complete-v0", GymRoboticsEnv),
    ]
    
    for env_type, env_name, env_class in env_configs:
        print(f"\n{env_type} Environment ({env_name}):")
        env = env_class(env_name, num_envs=1)
        observations = env.reset()
        print(f"  Observation shape: {observations.shape}")
        print(f"  Observation dim: {env.num_obs}")
        print(f"  Action dim: {env.num_actions}")
        
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"  Step reward: {rewards.item():.3f}")
        
        env.close()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 