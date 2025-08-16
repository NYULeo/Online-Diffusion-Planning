import pytest
import torch
import numpy as np
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

# Canonical, currently-valid env ids for smoke tests
ENV_ID_MAZE = "AntMaze_UMaze-v4"          # Maze family (v4)
ENV_ID_ADROIT = "AdroitHandPen-v1"        # Adroit family (v1)
ENV_ID_FRANKA = "FrankaKitchen-v1"        # Franka Kitchen (task config via env kwargs)


class TestGymRoboticsEnv:
    """Test cases for the GymRoboticsEnv class."""

    def test_available_envs(self):
        """We can retrieve lists of available robotics env ids."""
        maze_envs = get_available_maze_envs()
        adroit_envs = get_available_adroit_envs()
        franka_envs = get_available_franka_envs()
        all_envs = get_available_robotics_envs()

        assert isinstance(maze_envs, list)
        assert isinstance(adroit_envs, list)
        assert isinstance(franka_envs, list)
        assert isinstance(all_envs, list)

        assert len(maze_envs) > 0
        assert len(adroit_envs) > 0
        assert len(franka_envs) > 0
        assert len(all_envs) > 0

        # Spot-check that our canonical ids are present
        assert ENV_ID_MAZE in maze_envs
        assert ENV_ID_ADROIT in adroit_envs
        assert ENV_ID_FRANKA in franka_envs

    def test_maze_env_initialization(self):
        """Maze env initialization with vectorization."""
        env = GymRoboticsEnv(ENV_ID_MAZE, num_envs=2)
        assert env.num_envs == 2
        assert env.env_name == ENV_ID_MAZE
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs is False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_adroit_env_initialization(self):
        """Adroit env initialization."""
        env = GymRoboticsEnv(ENV_ID_ADROIT, num_envs=1)
        assert env.num_envs == 1
        assert env.env_name == ENV_ID_ADROIT
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs is False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_franka_env_initialization(self):
        """Franka env initialization (skip if assets/deps unavailable)."""
        try:
            env = GymRoboticsEnv(ENV_ID_FRANKA, num_envs=1)
        except Exception as e:
            pytest.skip(f"Skipping FrankaKitchen init (missing deps/assets?): {e}")
            return
        assert env.num_envs == 1
        assert env.env_name == ENV_ID_FRANKA
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs is False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_reset(self):
        """Environment reset returns correctly-shaped torch tensors."""
        env = GymRoboticsEnv(ENV_ID_MAZE, num_envs=1)
        observations = env.reset()
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[0] == 1  # num_envs
        assert observations.shape[1] == env.num_obs
        env.close()

    def test_step(self):
        """Single-env step works and shapes line up."""
        env = GymRoboticsEnv(ENV_ID_ADROIT, num_envs=1)
        env.reset()

        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)

        assert isinstance(observations, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
        assert isinstance(infos, dict)

        assert observations.shape[0] == 1  # num_envs
        assert observations.shape[1] == env.num_obs
        assert rewards.shape[0] == 1
        assert dones.shape[0] == 1

        env.close()

    def test_multi_env_step(self):
        """Stepping with multiple parallel envs."""
        env = GymRoboticsEnv(ENV_ID_MAZE, num_envs=4)
        env.reset()

        actions = torch.randn(4, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)

        assert observations.shape[0] == 4
        assert rewards.shape[0] == 4
        assert dones.shape[0] == 4

        env.close()

    def test_device_assignment(self):
        """Tensors should live on the requested device."""
        device = torch.device("cpu")
        env = GymRoboticsEnv(ENV_ID_ADROIT, num_envs=1, device=device)
        observations = env.reset()
        assert observations.device == device

        actions = torch.randn(1, env.num_actions, device=device)
        observations, rewards, dones, infos = env.step(actions)

        assert observations.device == device
        assert rewards.device == device
        assert dones.device == device

        env.close()

    def test_different_env_types(self):
        """Smoke test a couple of different env families."""
        env_names = [ENV_ID_MAZE, ENV_ID_ADROIT, ENV_ID_FRANKA]
        for env_name in env_names:
            try:
                env = GymRoboticsEnv(env_name, num_envs=1)
            except Exception:
                # Skip only the ones not available in the runtime (e.g., Franka assets)
                pytest.skip(f"Skipping {env_name} due to unavailable runtime requirements.")
                continue

            observations = env.reset()
            assert observations.shape[1] == env.num_obs

            actions = torch.randn(1, env.num_actions)
            observations, rewards, dones, infos = env.step(actions)

            assert observations.shape[1] == env.num_obs
            assert actions.shape[1] == env.num_actions

            env.close()

    def test_multi_goal_api(self):
        """Obs should be flat tensors even for multi-goal envs (FlattenObservation)."""
        env = GymRoboticsEnv(ENV_ID_MAZE, num_envs=1)
        observations = env.reset()
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[1] == env.num_obs
        env.close()

    def test_render_mode(self):
        """Instantiate with a render mode that works in headless (rgb_array)."""
        env = GymRoboticsEnv(ENV_ID_MAZE, num_envs=1, render_mode="rgb_array")
        env.reset()
        assert env.num_envs == 1
        env.close()

    def test_invalid_env_name(self):
        """Invalid ids should raise an exception."""
        with pytest.raises(Exception):
            env = GymRoboticsEnv("InvalidEnv-v0", num_envs=1)
            env.reset()


if __name__ == "__main__":
    pytest.main([__file__])
