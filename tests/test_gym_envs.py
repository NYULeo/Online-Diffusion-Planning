import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environment.gym_mujoco_env import GymMuJoCoEnv, get_available_mujoco_envs
from Environment.gym_robotics_env import GymRoboticsEnv, get_available_robotics_envs


class TestGymEnvironments:
    """Test cases for both MuJoCo and Robotics environments."""

    def test_available_envs(self):
        """Test that we can get the list of available environments."""
        mujoco_envs = get_available_mujoco_envs()
        robotics_envs = get_available_robotics_envs()
        
        assert isinstance(mujoco_envs, list)
        assert isinstance(robotics_envs, list)
        assert len(mujoco_envs) > 0
        assert len(robotics_envs) > 0

    def test_mujoco_env(self):
        """Test MuJoCo environment."""
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=1)
        observations = env.reset()
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[0] == 1
        
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        assert isinstance(observations, torch.Tensor)
        env.close()

    def test_robotics_env(self):
        """Test Robotics environment."""
        env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=1)
        observations = env.reset()
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[0] == 1
        
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        assert isinstance(observations, torch.Tensor)
        env.close()

    def test_device_assignment(self):
        """Test device assignment."""
        device = torch.device("cpu")
        
        mujoco_env = GymMuJoCoEnv("Hopper-v4", num_envs=1, device=device)
        observations = mujoco_env.reset()
        assert observations.device == device
        mujoco_env.close()
        
        robotics_env = GymRoboticsEnv("pen-human-v0", num_envs=1, device=device)
        observations = robotics_env.reset()
        assert observations.device == device
        robotics_env.close()


if __name__ == "__main__":
    pytest.main([__file__]) 