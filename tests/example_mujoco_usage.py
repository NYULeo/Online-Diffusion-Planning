#!/usr/bin/env python3
"""
Example usage of the GymMuJoCoEnv class.
This script demonstrates how to use the MuJoCo environment wrapper.
"""

import torch
import sys
import os

# Add the parent directory to the path to import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environment.gym_mujoco_env import GymMuJoCoEnv, get_available_mujoco_envs


def main():
    """Main function demonstrating MuJoCo environment usage."""
    
    print("Available MuJoCo environments:")
    available_envs = get_available_mujoco_envs()
    for env_name in available_envs:
        print(f"  - {env_name}")
    print()
    
    # Example 1: Single environment
    print("Example 1: Single environment")
    env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=1)
    
    # Reset the environment
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
    
    # Example 2: Multiple environments
    print("Example 2: Multiple environments")
    env = GymMuJoCoEnv("Hopper-v4", num_envs=4)
    
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    
    # Take some random steps
    for step in range(3):
        actions = torch.randn(4, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}:")
        print(f"  Rewards: {rewards.cpu().numpy()}")
        print(f"  Dones: {dones.cpu().numpy()}")
    
    env.close()
    print()
    
    # Example 3: Different environment
    print("Example 3: Walker2d environment")
    env = GymMuJoCoEnv("Walker2d-v4", num_envs=2)
    
    observations = env.reset()
    print(f"Walker2d observations shape: {observations.shape}")
    print(f"Walker2d observation space size: {env.num_obs}")
    print(f"Walker2d action space size: {env.num_actions}")
    
    # Take a few steps
    for step in range(3):
        actions = torch.randn(2, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Rewards = {rewards.cpu().numpy()}")
    
    env.close()
    print("All examples completed successfully!")


if __name__ == "__main__":
    main() 