import pytest
import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environment.gym_mujoco_env import GymMuJoCoEnv, get_available_mujoco_envs


class TestGymMuJoCoEnv:
    """Test cases for the GymMuJoCoEnv class."""

    def test_available_envs(self):
        """Test that we can get the list of available MuJoCo environments."""
        envs = get_available_mujoco_envs()
        assert isinstance(envs, list)
        assert len(envs) > 0
        assert "HalfCheetah-v4" in envs
        assert "Hopper-v4" in envs

    def test_env_initialization(self):
        """Test environment initialization."""
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=2)
        assert env.num_envs == 2
        assert env.env_name == "HalfCheetah-v4"
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs == False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=1)
        observations = env.reset()
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[0] == 1  # num_envs
        assert observations.shape[1] == env.num_obs
        env.close()

    def test_step(self):
        """Test environment step."""
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=1)
        env.reset()
        
        # Create random actions
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
        """Test stepping with multiple environments."""
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=4)
        env.reset()
        
        # Create random actions for multiple environments
        actions = torch.randn(4, env.num_actions)
        
        observations, rewards, dones, infos = env.step(actions)
        
        assert observations.shape[0] == 4  # num_envs
        assert rewards.shape[0] == 4
        assert dones.shape[0] == 4
        
        env.close()

    def test_device_assignment(self):
        """Test that tensors are assigned to the correct device."""
        device = torch.device("cpu")
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=1, device=device)
        observations = env.reset()
        assert observations.device == device
        
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        
        assert observations.device == device
        assert rewards.device == device
        assert dones.device == device
        
        env.close()

    def test_different_envs(self):
        """Test different MuJoCo environments."""
        env_names = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
        
        for env_name in env_names:
            env = GymMuJoCoEnv(env_name, num_envs=1)
            observations = env.reset()
            assert observations.shape[1] == env.num_obs
            
            actions = torch.randn(1, env.num_actions)
            observations, rewards, dones, infos = env.step(actions)
            
            assert observations.shape[1] == env.num_obs
            assert actions.shape[1] == env.num_actions
            
            env.close()

    def test_render_mode(self):
        """Test environment with render mode."""
        env = GymMuJoCoEnv("HalfCheetah-v4", num_envs=1, render_mode="human")
        env.reset()
        
        # Note: render test might fail in headless environments
        # This is just to test that the render_mode parameter works
        assert env.num_envs == 1
        
        env.close()

    def test_invalid_env_name(self):
        """Test that invalid environment names raise appropriate errors."""
        with pytest.raises(Exception):
            # This should fail as "InvalidEnv-v4" doesn't exist
            env = GymMuJoCoEnv("InvalidEnv-v4", num_envs=1)
            env.reset()


if __name__ == "__main__":
    pytest.main([__file__]) 