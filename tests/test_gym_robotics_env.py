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
    get_available_robotics_envs
)


class TestGymRoboticsEnv:
    """Test cases for the GymRoboticsEnv class."""

    def test_available_envs(self):
        """Test that we can get the list of available robotics environments."""
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

    def test_maze_env_initialization(self):
        """Test maze environment initialization."""
        env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=2)
        assert env.num_envs == 2
        assert env.env_name == "maze2d-umaze-v1"
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs == False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_adroit_env_initialization(self):
        """Test adroit environment initialization."""
        env = GymRoboticsEnv("pen-human-v0", num_envs=1)
        assert env.num_envs == 1
        assert env.env_name == "pen-human-v0"
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs == False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_franka_env_initialization(self):
        """Test franka environment initialization."""
        env = GymRoboticsEnv("kitchen-complete-v0", num_envs=1)
        assert env.num_envs == 1
        assert env.env_name == "kitchen-complete-v0"
        assert env.max_episode_steps == 1000
        assert env.asymmetric_obs == False
        assert env.num_obs > 0
        assert env.num_actions > 0
        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=1)
        observations = env.reset()
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[0] == 1  # num_envs
        assert observations.shape[1] == env.num_obs
        env.close()

    def test_step(self):
        """Test environment step."""
        env = GymRoboticsEnv("pen-human-v0", num_envs=1)
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
        env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=4)
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
        env = GymRoboticsEnv("kitchen-complete-v0", num_envs=1, device=device)
        observations = env.reset()
        assert observations.device == device
        
        actions = torch.randn(1, env.num_actions)
        observations, rewards, dones, infos = env.step(actions)
        
        assert observations.device == device
        assert rewards.device == device
        assert dones.device == device
        
        env.close()

    def test_different_env_types(self):
        """Test different robotics environment types."""
        env_names = ["maze2d-umaze-v1", "pen-human-v0", "kitchen-complete-v0"]
        
        for env_name in env_names:
            env = GymRoboticsEnv(env_name, num_envs=1)
            observations = env.reset()
            assert observations.shape[1] == env.num_obs
            
            actions = torch.randn(1, env.num_actions)
            observations, rewards, dones, infos = env.step(actions)
            
            assert observations.shape[1] == env.num_obs
            assert actions.shape[1] == env.num_actions
            
            env.close()

    def test_multi_goal_api(self):
        """Test that multi-goal API observations are handled correctly."""
        env = GymRoboticsEnv("pen-human-v0", num_envs=1)
        observations = env.reset()
        
        # For multi-goal environments, observations should be flattened
        # and contain observation + desired_goal + achieved_goal
        assert isinstance(observations, torch.Tensor)
        assert observations.shape[1] == env.num_obs
        
        env.close()

    def test_render_mode(self):
        """Test environment with render mode."""
        env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=1, render_mode="human")
        env.reset()
        
        # Note: render test might fail in headless environments
        # This is just to test that the render_mode parameter works
        assert env.num_envs == 1
        
        env.close()

    def test_invalid_env_name(self):
        """Test that invalid environment names raise appropriate errors."""
        with pytest.raises(Exception):
            # This should fail as "InvalidEnv-v0" doesn't exist
            env = GymRoboticsEnv("InvalidEnv-v0", num_envs=1)
            env.reset()


if __name__ == "__main__":
    pytest.main([__file__]) 