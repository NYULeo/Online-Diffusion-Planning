import sys
import os
import logging
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
from loguru import logger as log
import minari
from Dataset import KitchenDataset


from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any
import torch.nn as nn
from Backbone import UNet1D
from Dataset import get_dataset, get_dataset
import torch
import math

def compare_environment_states(env1, env2):
    """Compare the current state of two environments"""
    
    # Reset both environments
    obs1 = env1.reset()
    obs2 = env2.reset()
    
    # Compare initial observations
    if not np.array_equal(obs1, obs2):
        print("❌ Initial observations differ")
        return False
    
    # Take same action and compare
    action = env1.action_space.sample()
    obs1_next, reward1, done1, info1 = env1.step(action)
    obs2_next, reward2, done2, info2 = env2.step(action)
    
    if not np.array_equal(obs1_next, obs2_next):
        print("❌ Next observations differ")
        return False
        
    if reward1 != reward2:
        print("❌ Rewards differ")
        return False
        
    print("✅ Environment states are equal")
    return True
    
def compare_environments(env1, env2):
    """Compare two environments for equality"""
    
    # Check basic properties
    if env1.observation_space != env2.observation_space:
        print("❌ Observation spaces differ")
        return False
        
    if env1.action_space != env2.action_space:
        print("❌ Action spaces differ")
        return False
        
    if env1.reward_range != env2.reward_range:
        print("❌ Reward ranges differ")
        return False
        
    # Check environment names/IDs
    if env1.unwrapped.spec.id != env2.unwrapped.spec.id:
        print("❌ Environment IDs differ")
        return False
        
    print("✅ Environments are equal")
    return True

gym.register_envs(gymnasium_robotics)

env1 = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle', 'light switch', 'slide cabinet'])

data = get_dataset('kitchen', 'complete')
env2 = data.get_env()

# Compare the environments
print("=== Environment Comparison ===")
print(f"env1: {env1.unwrapped.spec.id}")
print(f"env2: {env2.unwrapped.spec.id}")
print()

# Check if environments are equal
are_equal = compare_environments(env1, env2)
print()

# Check if environment states are equal
if are_equal:
    print("=== State Comparison ===")
    states_equal = compare_environment_states(env1, env2)
else:
    print("Skipping state comparison since environments are not equal")

print("\n=== Summary ===")
if are_equal:
    print("✅ Environments are equal")
else:
    print("❌ Environments are not equal")








