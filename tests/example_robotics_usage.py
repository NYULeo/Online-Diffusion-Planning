#!/usr/bin/env python3
"""
Example usage of the GymRoboticsEnv class.
This script demonstrates how to use the robotics environment wrapper.
"""

import torch
import sys
import os

# Add the parent directory to the path to import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environment.gym_robotics_env import (
    GymRoboticsEnv, 
    get_available_maze_envs,
    get_available_adroit_envs,
    get_available_franka_envs,
    get_available_robotics_envs
)


def main():
    """Main function demonstrating robotics environment usage."""
    
    print("Available Robotics Environments:")
    print("\nMaze Environments:")
    maze_envs = get_available_maze_envs()
    for env_name in maze_envs[:5]:  # Show first 5
        print(f"  - {env_name}")
    print(f"  ... and {len(maze_envs)-5} more")
    
    print("\nAdroit Arm Environments:")
    adroit_envs = get_available_adroit_envs()
    for env_name in adroit_envs[:5]:  # Show first 5
        print(f"  - {env_name}")
    print(f"  ... and {len(adroit_envs)-5} more")
    
    print("\nFranka Kitchen Environments:")
    franka_envs = get_available_franka_envs()
    for env_name in franka_envs:
        print(f"  - {env_name}")
    
    print(f"\nTotal environments: {len(get_available_robotics_envs())}")
    print()
    
    # Example 1: Maze environment
    print("Example 1: Maze environment")
    env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=1)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    print(f"Observation space size: {env.num_obs}")
    print(f"Action space size: {env.num_actions}")
    
    # Take some random steps
    for step in range(5):
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Reward = {rewards.item():.3f}, Done = {dones.item()}")
    
    env.close()
    print()
    
    # Example 2: Adroit Arm environment
    print("Example 2: Adroit Arm environment")
    env = GymRoboticsEnv("pen-human-v0", num_envs=2)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    
    # Take some random steps
    for step in range(3):
        actions = torch.randn(2, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}:")
        print(f"  Rewards: {rewards.numpy()}")
        print(f"  Dones: {dones.numpy()}")
    
    env.close()
    print()
    
    # Example 3: Franka Kitchen environment
    print("Example 3: Franka Kitchen environment")
    env = GymRoboticsEnv("kitchen-complete-v0", num_envs=1)
    
    observations = env.reset()
    print(f"Kitchen observations shape: {observations.shape}")
    print(f"Kitchen observation space size: {env.num_obs}")
    print(f"Kitchen action space size: {env.num_actions}")
    
    # Take a few steps
    for step in range(3):
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Reward = {rewards.item():.3f}")
    
    env.close()
    print()
    
    # Example 4: Multi-environment with different types
    print("Example 4: Multi-environment setup")
    env_names = ["maze2d-umaze-v1", "pen-human-v0", "kitchen-complete-v0"]
    
    for env_name in env_names:
        print(f"\nTesting {env_name}:")
        env = GymRoboticsEnv(env_name, num_envs=1)
        observations = env.reset()
        print(f"  Observation shape: {observations.shape}")
        print(f"  Observation dim: {env.num_obs}")
        print(f"  Action dim: {env.num_actions}")
        
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"  Step reward: {rewards.item():.3f}")
        
        env.close()
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main() 