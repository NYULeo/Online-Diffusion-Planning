#!/usr/bin/env python3
"""
Example usage of the GymRoboticsEnv class (Farama / gymnasium-robotics).
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
    get_available_robotics_envs,
)

# Canonical env ids for examples (current Farama naming)
ENV_ID_MAZE = "AntMaze_UMaze-v4"
ENV_ID_ADROIT = "AdroitHandPen-v1"
ENV_ID_FRANKA = "FrankaKitchen-v1"


def _print_list_with_head(name, items, head=5):
    print(f"\n{name}:")
    if not items:
        print("  (none found)")
        return
    show = min(len(items), head)
    for env_name in items[:show]:
        print(f"  - {env_name}")
    remaining = max(len(items) - show, 0)
    if remaining > 0:
        print(f"  ... and {remaining} more")


def main():
    """Main function demonstrating robotics environment usage."""

    print("Available Robotics Environments:")
    _print_list_with_head("Maze Environments", get_available_maze_envs())
    _print_list_with_head("Adroit Hand Environments", get_available_adroit_envs())
    _print_list_with_head("Franka Kitchen Environments", get_available_franka_envs(), head=10)

    print(f"\nTotal environments: {len(get_available_robotics_envs())}\n")

    # ---------------------------------------------------------------------
    # Example 1: Maze environment (vectorized = 1 for a simple smoke test)
    # ---------------------------------------------------------------------
    print("Example 1: Maze environment")
    env = GymRoboticsEnv(ENV_ID_MAZE, num_envs=1)
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    print(f"Observation space size: {env.num_obs}")
    print(f"Action space size: {env.num_actions}")

    for step in range(5):
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: Reward = {rewards.item():.3f}, Done = {bool(dones.item())}")

    env.close()
    print()

    # ---------------------------------------------------------------------
    # Example 2: Adroit Hand environment (vectorized = 2)
    # ---------------------------------------------------------------------
    print("Example 2: Adroit Hand environment")
    env = GymRoboticsEnv(ENV_ID_ADROIT, num_envs=2)
    observations = env.reset()
    print(f"Initial observations shape: {observations.shape}")

    for step in range(3):
        actions = torch.randn(2, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}:")
        print(f"  Rewards: {rewards.cpu().numpy()}")
        print(f"  Dones: {dones.cpu().numpy()}")

    env.close()
    print()

    # ---------------------------------------------------------------------
    # Example 3: Franka Kitchen environment
    # Note: may require additional assets/dependencies. We handle failure
    # gracefully so the example still runs on lightweight setups/CI.
    # ---------------------------------------------------------------------
    print("Example 3: Franka Kitchen environment")
    try:
        env = GymRoboticsEnv(ENV_ID_FRANKA, num_envs=1)
    except Exception as e:
        print(f"  Skipping {ENV_ID_FRANKA} (not available on this machine): {e}")
    else:
        observations = env.reset()
        print(f"Kitchen observations shape: {observations.shape}")
        print(f"Kitchen observation space size: {env.num_obs}")
        print(f"Kitchen action space size: {env.num_actions}")

        for step in range(3):
            actions = torch.randn(1, env.num_actions)
            observations, rewards, dones, infos = env.step(actions)
            print(f"Step {step + 1}: Reward = {rewards.item():.3f}")

        env.close()
    print()

    # ---------------------------------------------------------------------
    # Example 4: Multi-environment with different types (sequential)
    # We iterate through a couple of different families.
    # ---------------------------------------------------------------------
    print("Example 4: Multi-environment setup")
    env_names = [ENV_ID_MAZE, ENV_ID_ADROIT, ENV_ID_FRANKA]

    for env_name in env_names:
        print(f"\nTesting {env_name}:")
        try:
            env = GymRoboticsEnv(env_name, num_envs=1)
        except Exception as e:
            print(f"  Skipping {env_name} (unavailable): {e}")
            continue

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
