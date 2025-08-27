"""
Franka Kitchen Environment

This module provides a wrapper for the official Gymnasium-Robotics Franka Kitchen environment.
Based on the official documentation: https://robotics.farama.org/envs/franka_kitchen/

The Franka Kitchen environment is a multitask environment in which a 9-DoF Franka robot 
is placed in a kitchen containing several common household items. The goal of each task 
is to interact with the items in order to reach a desired goal configuration.

Available tasks:
- bottom burner: twist control knob to activate bottom left burner in the stove
- top burner: twist control knob to activate top left burner in the stove  
- light switch: move a lever switch to turn on a light over the burners
- slide cabinet: slide open the cabinet door
- hinge cabinet: open a hinge cabinet door
- microwave: open the microwave door
- kettle: move the kettle from the bottom burner to the top burner
"""

import sys
import os
import logging
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
from loguru import logger as log
import minari

# Disable all logging below CRITICAL level
log.remove()
log.add(lambda msg: False, level="CRITICAL")

# Try to import Gymnasium-Robotics
try:
    import gymnasium_robotics
    GYMNASIUM_ROBOTICS_AVAILABLE = True
    # Register the environments
    gym.register_envs(gymnasium_robotics)
except ImportError:
    GYMNASIUM_ROBOTICS_AVAILABLE = False
    print("Warning: Gymnasium-Robotics not available. Franka Kitchen environment will use fallback.")

# Available Franka Kitchen tasks (from official documentation)
FRANKA_KITCHEN_TASKS = [
    "bottom burner",
    "top burner", 
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

# Default task configuration (from official documentation)
DEFAULT_TASKS = ["microwave", "kettle", "bottom burner", "light switch"]


def make_env(env_name="FrankaKitchen-v1", tasks_to_complete=None, rank=0, render_mode=None, seed=0):
    """
    Utility function for multiprocessed environment creation.
    
    Args:
        env_name: Name of the environment (default: 'FrankaKitchen-v1')
        tasks_to_complete: List of tasks to complete (default: DEFAULT_TASKS)
        rank: Index of the subprocess
        render_mode: Render mode for the environment
        seed: Initial seed for RNG
        
    Returns:
        Environment initialization function
    """
    max_episode_steps = 280  # Standard for Franka Kitchen environments
    
    if tasks_to_complete is None:
        tasks_to_complete = DEFAULT_TASKS

    def _init():
        if GYMNASIUM_ROBOTICS_AVAILABLE:
            try:
                # Create official Gymnasium-Robotics Franka Kitchen environment
                env = gym.make(env_name, tasks_to_complete=tasks_to_complete, render_mode=render_mode)
                print(f"Created Gymnasium-Robotics Franka Kitchen environment: {env_name}")
                print(f"Tasks to complete: {tasks_to_complete}")
            except Exception as e:
                print(f"Warning: Could not create Gymnasium-Robotics environment {env_name}: {e}")
                print("Using fallback environment.")
                env = gym.make("Pendulum-v1", render_mode=render_mode)
        else:
            print("Warning: Gymnasium-Robotics not available. Creating fallback environment.")
            env = gym.make("Pendulum-v1", render_mode=render_mode)
        
        # Wrap with time limit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # Handle seeding
        try:
            if hasattr(env.unwrapped, 'seed'):
                env.unwrapped.seed(seed + rank)
            else:
                env.reset(seed=seed + rank)
        except Exception:
            pass
            
        return env

    return _init


class FrankaKitchenEnv:
    """
    Wrapper for Franka Kitchen environment to support parallel environments.
    
    This wrapper handles the GoalEnv observation structure and provides
    a consistent interface for the diffusion planning framework.
    """
    
    def __init__(self, env_name="FrankaKitchen-v1", tasks_to_complete=None, num_envs=1, 
                 render_mode=None, seed=0, sim_device="cpu"):
        """
        Initialize Franka Kitchen environment wrapper.
        
        Args:
            env_name: Name of the environment (default: 'FrankaKitchen-v1')
            tasks_to_complete: List of tasks to complete (default: DEFAULT_TASKS)
            num_envs: Number of parallel environments
            render_mode: Render mode for the environment
            seed: Initial seed for RNG
            sim_device: Device for simulation (cpu/cuda)
        """
        self.env_name = env_name
        self.tasks_to_complete = tasks_to_complete if tasks_to_complete is not None else DEFAULT_TASKS
        self.num_envs = num_envs
        self.sim_device = sim_device
        
        # Validate tasks
        for task in self.tasks_to_complete:
            if task not in FRANKA_KITCHEN_TASKS:
                raise ValueError(f"Unknown task: {task}. Available tasks: {FRANKA_KITCHEN_TASKS}")
        
        # Create parallel environments
        if num_envs == 1:
            self.envs = make_env(env_name, self.tasks_to_complete, 0, render_mode, seed)()
        else:
            self.envs = SubprocVecEnv([
                make_env(env_name, self.tasks_to_complete, i, render_mode, seed) 
                for i in range(num_envs)
            ])
        
        # Get environment dimensions
        self._get_env_dimensions()
        
        print(f"✅ Franka Kitchen environment initialized: {env_name}")
        print(f"📊 Tasks: {self.tasks_to_complete}")
        print(f"📏 Observation space: {self.num_obs} dimensions")
        print(f"🎯 Action space: {self.num_actions} dimensions")
        print(f"🔄 Number of environments: {self.num_envs}")
    
    def _get_env_dimensions(self):
        """Extract environment dimensions from observation and action spaces."""
        # Handle GoalEnv observation space structure
        if hasattr(self.envs.observation_space, 'spaces') and 'observation' in self.envs.observation_space.spaces:
            # GoalEnv with dictionary observation space (like Franka Kitchen)
            self.num_obs = self.envs.observation_space.spaces['observation'].shape[-1]
            self.asymmetric_obs = True  # GoalEnv has asymmetric observation space
        elif hasattr(self.envs.observation_space, 'shape'):
            self.num_obs = self.envs.observation_space.shape[-1]
            self.asymmetric_obs = False
        else:
            self.num_obs = self.envs.observation_space.n  # For discrete spaces
            self.asymmetric_obs = False
        
        # Handle action space
        if hasattr(self.envs.action_space, 'shape') and len(self.envs.action_space.shape) > 0:
            self.num_actions = self.envs.action_space.shape[-1]
        else:
            self.num_actions = self.envs.action_space.n  # For discrete spaces
    
    def reset(self):
        """Reset the environment."""
        reset_result = self.envs.reset()
        
        # Handle different return types from environment reset
        if isinstance(reset_result, tuple):
            observations, info = reset_result
        else:
            observations = reset_result
            info = {}
        
        # Handle GoalEnv observation structure
        if self.asymmetric_obs and isinstance(observations, dict):
            # For GoalEnv, extract the 'observation' key
            observations = observations['observation']
        
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        return observations
    
    def step(self, actions):
        """Take a step in the environment."""
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()
        
        # Handle single environment case - remove batch dimension
        if self.num_envs == 1 and actions.ndim == 2:
            actions = actions.squeeze(0)

        step_result = self.envs.step(actions)

        # Handle different return types from environment step
        if len(step_result) == 5:  # New Gymnasium API: obs, reward, terminated, truncated, info
            observations, rewards, terminated, truncated, raw_infos = step_result
            dones = terminated | truncated
        else:  # Old API: obs, reward, done, info
            observations, rewards, dones, raw_infos = step_result
            truncated = np.zeros_like(dones)

        # Handle GoalEnv observation structure
        if self.asymmetric_obs and isinstance(observations, dict):
            # For GoalEnv, extract the 'observation' key
            observations = observations['observation']

        # Process info for getting 'true' next observations
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones)
        
        # Handle single environment case
        if self.num_envs == 1:
            if raw_infos.get("TimeLimit.truncated", False):
                truncateds[0] = True
                infos["observations"]["raw"]["obs"] = raw_infos["terminal_observation"]
        else:
            # Handle multiple environments case
            for i in range(self.num_envs):
                if raw_infos[i].get("TimeLimit.truncated", False):
                    truncateds[i] = True
                    infos["observations"]["raw"]["obs"][i] = raw_infos[i]["terminal_observation"]

        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        
        # Handle scalar vs array rewards and dones
        if np.isscalar(rewards):
            rewards = torch.tensor([rewards], dtype=torch.float, device=self.sim_device)
        else:
            rewards = torch.from_numpy(rewards).to(device=self.sim_device, dtype=torch.float)
            
        if np.isscalar(dones):
            dones = torch.tensor([dones], dtype=torch.bool, device=self.sim_device)
        else:
            dones = torch.from_numpy(dones).to(device=self.sim_device)
            
        if np.isscalar(truncateds):
            truncateds = torch.tensor([truncateds], dtype=torch.bool, device=self.sim_device)
        else:
            truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds

        return observations, rewards, dones, infos
    
    def close(self):
        """Close the environment."""
        self.envs.close()


class FrankaKitchenTaskEnv:
    """
    Task-specific environment for Franka Kitchen with goal computation.
    
    This class provides utilities for computing task rewards and goals
    based on the Franka Kitchen environment structure.
    """
    
    def __init__(self, tasks_to_complete=None):
        """
        Initialize Franka Kitchen task environment.
        
        Args:
            tasks_to_complete: List of tasks to complete (default: DEFAULT_TASKS)
        """
        self.tasks_to_complete = tasks_to_complete if tasks_to_complete is not None else DEFAULT_TASKS
        
        # Validate tasks
        for task in self.tasks_to_complete:
            if task not in FRANKA_KITCHEN_TASKS:
                raise ValueError(f"Unknown task: {task}. Available tasks: {FRANKA_KITCHEN_TASKS}")
        
        print(f"✅ Franka Kitchen task environment initialized with tasks: {self.tasks_to_complete}")
    
    def get_task_goal(self):
        """
        Get the goal configuration for the specified tasks.
        
        Returns:
            Goal configuration array
        """
        # For Franka Kitchen, the goal is typically the final state
        # where all specified tasks are completed
        # This would need to be implemented based on the specific task requirements
        # For now, return a placeholder
        return np.zeros(60)  # Standard Franka Kitchen observation dimension
    
    def compute_task_reward(self, observation):
        """
        Compute reward based on task completion.
        
        Args:
            observation: Current observation
            
        Returns:
            Task reward
        """
        # This would need to be implemented based on the specific task requirements
        # For now, return a placeholder reward
        return 0.0
    
    def check_task_completion(self, observation):
        """
        Check if all tasks are completed.
        
        Args:
            observation: Current observation
            
        Returns:
            Boolean indicating if all tasks are completed
        """
        # This would need to be implemented based on the specific task requirements
        # For now, return False
        return False


def create_franka_kitchen_env(tasks_to_complete=None, num_envs=1, **kwargs):
    """
    Convenience function to create a Franka Kitchen environment.
    
    Args:
        tasks_to_complete: List of tasks to complete (default: DEFAULT_TASKS)
        num_envs: Number of parallel environments
        **kwargs: Additional arguments for FrankaKitchenEnv
        
    Returns:
        FrankaKitchenEnv instance
    """
    return FrankaKitchenEnv(tasks_to_complete=tasks_to_complete, num_envs=num_envs, **kwargs)


def create_franka_kitchen_task_env(tasks_to_complete=None):
    """
    Convenience function to create a Franka Kitchen task environment.
    
    Args:
        tasks_to_complete: List of tasks to complete (default: DEFAULT_TASKS)
        
    Returns:
        FrankaKitchenTaskEnv instance
    """
    return FrankaKitchenTaskEnv(tasks_to_complete=tasks_to_complete)


# Example usage and testing
if __name__ == "__main__":
    # Test environment creation
    print("🧪 Testing Franka Kitchen Environment")
    print("=" * 50)
    
    # Create environment with default tasks
    env = create_franka_kitchen_env(num_envs=1)
    
    # Test reset and step
    obs = env.reset()
    print(f"✅ Reset successful, obs shape: {obs.shape}")
    
    action = torch.randn(1, env.num_actions)
    next_obs, reward, done, info = env.step(action)
    print(f"✅ Step successful, reward: {reward.item():.4f}, done: {done.item()}")
    
    # Test task environment
    task_env = create_franka_kitchen_task_env()
    print(f"✅ Task environment created with tasks: {task_env.tasks_to_complete}")
    
    print("🎉 All tests passed!")
