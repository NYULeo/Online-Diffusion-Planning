from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from loguru import logger as log
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import FlattenObservation

# Disable all logging below CRITICAL level
log.remove()
log.add(lambda msg: False, level="CRITICAL")


def make_env(env_id: str, rank: int, render_mode: str | None = None, seed: int = 0, max_episode_steps: int = 1000):
    """Utility function for multiprocessed env creation (SB3 pattern)."""
    def _init():
        gym.register_envs(gymnasium_robotics)
        env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
        
        # Flatten dict observations (e.g., Maze/Fetch/Hand multi-goal API) for SB3
        if hasattr(env.observation_space, "spaces"):
            env = FlattenObservation(env)

        env.reset(seed=seed + rank)
        return env
    return _init


class GymRoboticsEnv:
    """Wraps Gymnasium Robotics environments to support parallel environments."""

    def __init__(self, env_name: str, num_envs: int = 1, render_mode: str | None = None, device=None, max_episode_steps: int = 1000):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps

        # Create the vectorized environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode, max_episode_steps=max_episode_steps) for i in range(num_envs)]
        )

        self.asymmetric_obs = False
        self.num_obs = self._get_observation_dim()
        self.num_actions = self.envs.action_space.shape[-1]

    def _get_observation_dim(self) -> int:
        """Get the observation dimension for the (possibly flattened) environment."""
        obs_space = self.envs.observation_space
        # After FlattenObservation, this will always be a Box
        return int(np.prod(obs_space.shape))

    def reset(self) -> torch.Tensor:
        """Reset the environment and return torch observations."""
        observations = self.envs.reset()  # SB3 VecEnv -> np.ndarray
        obs = torch.as_tensor(observations, device=self.sim_device, dtype=torch.float32)
        return obs

    def render(self):
        """Render the environment (single env only)."""
        assert self.num_envs == 1, "Rendering is only supported for a single environment."
        return self.envs.render()

    def step(self, actions: torch.Tensor):
        """Take a step in the environment."""
        assert isinstance(actions, torch.Tensor)
        observations, rewards, dones, raw_infos = self.envs.step(actions.cpu().numpy())

        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones, dtype=bool)

        # SB3’s VecEnv merges Gymnasium terminated|truncated into dones.
        # TimeLimit truncation is exposed via info["TimeLimit.truncated"].
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                if "terminal_observation" in raw_infos[i]:
                    infos["observations"]["raw"]["obs"][i] = raw_infos[i]["terminal_observation"]

        obs = torch.as_tensor(observations, device=self.sim_device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device=self.sim_device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.sim_device, dtype=torch.bool)
        truncateds = torch.as_tensor(truncateds, device=self.sim_device, dtype=torch.bool)
        infos["observations"]["raw"]["obs"] = torch.as_tensor(infos["observations"]["raw"]["obs"],
                                                              device=self.sim_device, dtype=torch.float32)
        infos["time_outs"] = truncateds
        return obs, rewards, dones, infos

    def close(self):
        self.envs.close()


def get_available_franka_envs():
    """Franka Kitchen in Gymnasium-Robotics (choose tasks via tasks_to_complete)."""
    return ["FrankaKitchen-v1"]  # tasks set at env creation, see docs


def get_available_maze_envs():
    """Point/Ant Maze ids as of v2025 docs (dense variants available via *Dense* suffix)."""
    return [
        # AntMaze (v4)
        "AntMaze_UMaze-v4",
        "AntMaze_BigMaze-v4",
        "AntMaze_HardestMaze-v4",
        "AntMaze_BigMaze_DG-v4",
        "AntMaze_HardestMaze_DG-v4",
        "AntMaze_BigMaze_DGR-v4",
        "AntMaze_HardestMaze_DGR-v4",
        # PointMaze (v3) — if you use Point variant
        "PointMaze_UMaze-v3",
        "PointMaze_BigMaze-v3",
        "PointMaze_HardestMaze-v3",
        "PointMaze_BigMaze_DG-v3",
        "PointMaze_HardestMaze_DG-v3",
        "PointMaze_BigMaze_DGR-v3",
        "PointMaze_HardestMaze_DGR-v3",
    ]


def get_available_adroit_envs():
    """Adroit hand envs (dense + *Sparse* variants)."""
    return [
        "AdroitHandDoor-v1", "AdroitHandDoorSparse-v1",
        "AdroitHandHammer-v1", "AdroitHandHammerSparse-v1",
        "AdroitHandPen-v1", "AdroitHandPenSparse-v1",
        "AdroitHandRelocate-v1", "AdroitHandRelocateSparse-v1",
    ]


def get_available_robotics_envs():
    """All current robotics env ids this wrapper targets."""
    return get_available_franka_envs() + get_available_maze_envs() + get_available_adroit_envs()
