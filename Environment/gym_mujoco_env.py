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
    """
    Create a function that returns an environment.
    
    :param env_name: (str) name of the environment
    :param rank: (int) index of the subprocess
    :param render_mode: (str) rendering mode
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=1000)
        # Use reset with seed instead of deprecated seed method
        env.reset(seed=seed + rank)
        return env

    return _init


class GymMuJoCoEnv:
    """Wraps Gymnasium MuJoCo environments to support parallel environments."""

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        """
        Initialize the MuJoCo environment wrapper.
        
        :param env_name: (str) name of the MuJoCo environment (e.g., "HalfCheetah-v4")
        :param num_envs: (int) number of parallel environments
        :param render_mode: (str) rendering mode
        :param device: (torch.device) device to use for tensors
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs
        self.env_name = env_name

        # Create the base environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
        )

        # Set episode length based on environment
        if env_name in [ "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]:
            self.max_episode_steps = 1000
        elif env_name in ["Ant-v5", "Humanoid-v5"]:
            self.max_episode_steps = 1000
        else:
            self.max_episode_steps = 1000

        # For compatibility with other environments
        self.asymmetric_obs = False
        self.num_obs = self.envs.observation_space.shape[-1]
        self.num_actions = self.envs.action_space.shape[-1]

    def reset(self):
        """Reset the environment."""
        observations = self.envs.reset()
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        return observations

    def render(self):
        """Render the environment."""
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        """
        Take a step in the environment.
        
        :param actions: (torch.Tensor) actions to take
        :return: (observations, rewards, dones, infos)
        """
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
                infos["observations"]["raw"]["obs"][i] = raw_infos[i][
                    "terminal_observation"
                ]

        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds

        return observations, rewards, dones, infos

    def close(self):
        """Close the environment."""
        self.envs.close()


def get_available_mujoco_envs():
    """Get list of available MuJoCo environments."""
    return [
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "Ant-v5",
        "Humanoid-v5",
        "Swimmer-v5",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "Reacher-v5",
        "Pusher-v5",
        "HumanoidStandup-v5"
    ]