from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from loguru import logger as log
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit

# Disable all logging below CRITICAL level
log.remove()
log.add(lambda msg: False, level="CRITICAL")


def make_env(env_name, rank, render_mode=None, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=1000)
        env.unwrapped.seed(seed + rank)
        return env
    return _init


class GymRoboticsEnv:
    """Wraps Gymnasium Robotics environments to support parallel environments."""

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs
        self.env_name = env_name

        # Create the base environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
        )

        self.max_episode_steps = 1000
        self.asymmetric_obs = False
        self.num_obs = self._get_observation_dim()
        self.num_actions = self.envs.action_space.shape[-1]

    def _get_observation_dim(self):
        """Get the observation dimension for the environment."""
        obs_space = self.envs.observation_space
        if hasattr(obs_space, 'spaces') and 'observation' in obs_space.spaces:
            # Multi-goal API: return the dimension of the 'observation' key
            return obs_space.spaces['observation'].shape[-1]
        else:
            # Standard API: return the full observation dimension
            return obs_space.shape[-1]

    def reset(self):
        """Reset the environment."""
        observations = self.envs.reset()
        
        # Handle multi-goal API observations
        if isinstance(observations, dict):
            flattened_obs = self._flatten_observation(observations)
        else:
            flattened_obs = observations
            
        flattened_obs = torch.from_numpy(flattened_obs).to(
            device=self.sim_device, dtype=torch.float
        )
        return flattened_obs

    def _flatten_observation(self, obs_dict):
        """Flatten multi-goal observation dictionary."""
        if isinstance(obs_dict, dict):
            # Concatenate observation, desired_goal, and achieved_goal
            obs = obs_dict['observation']
            desired_goal = obs_dict['desired_goal']
            achieved_goal = obs_dict['achieved_goal']
            
            # Concatenate along the last dimension
            flattened = np.concatenate([obs, desired_goal, achieved_goal], axis=-1)
            return flattened
        else:
            return obs_dict

    def render(self):
        """Render the environment."""
        assert self.num_envs == 1, "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        """Take a step in the environment."""
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos = self.envs.step(actions)

        # Handle truncated episodes
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                infos["observations"]["raw"]["obs"][i] = raw_infos[i]["terminal_observation"]

        # Handle multi-goal API observations
        if isinstance(observations, dict):
            flattened_obs = self._flatten_observation(observations)
        else:
            flattened_obs = observations

        flattened_obs = torch.from_numpy(flattened_obs).to(
            device=self.sim_device, dtype=torch.float
        )
        rewards = torch.from_numpy(rewards).to(device=self.sim_device, dtype=torch.float)
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds

        return flattened_obs, rewards, dones, infos

    def close(self):
        """Close the environment."""
        self.envs.close()


def get_available_maze_envs():
    """Get list of available maze environments."""
    return [
        "maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1",
        "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1",
        "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0", "antmaze-large-play-v0", "antmaze-large-diverse-v0"
    ]


def get_available_adroit_envs():
    """Get list of available adroit arm environments."""
    return [
        "pen-human-v0", "pen-cloned-v0", "pen-expert-v0",
        "hammer-human-v0", "hammer-cloned-v0", "hammer-expert-v0",
        "door-human-v0", "door-cloned-v0", "door-expert-v0",
        "relocate-human-v0", "relocate-cloned-v0", "relocate-expert-v0"
    ]


def get_available_franka_envs():
    """Get list of available franka kitchen environments."""
    return ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]


def get_available_robotics_envs():
    """Get list of all available robotics environments."""
    return get_available_maze_envs() + get_available_adroit_envs() + get_available_franka_envs() 