from pstats import StatsProfile
import sys
import os
import logging
import numpy as np
import torch
import gymnasium as gym# Conditional import to avoid GLFW3 errors on headless servers

from loguru import logger as log
import minari
from scipy.ndimage import gaussian_filter1d, convolve

from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any
import torch.nn as nn
from Dataset import get_dataset, get_dataset

import torch
import math
from utils import set_seed
from Dataset import get_env, get_dataset
import gymnasium_robotics
import mediapy as media
from collections import namedtuple

from Planners.Backbone.utils import get_pretrained_planner
from torch.utils.data import DataLoader
from Dataset import PlannerDataset
from Rewards.nets import gaussian_rewards
import scipy
import scipy.ndimage
from sympy import factorint





"""

env, d_s, d_a= get_env('kitchen', 'mixed')
data = get_dataset('kitchen', 'mixed')
trajs = data.get_trajectories()
traj = trajs[0]
print(traj.keys())




set_seed(0)
env.reset()
frames = []
for i in range(len(traj['actions'])):
    action = traj['actions'][i]
    action = np.clip(action, -1.0, 1.0)
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

media.write_video("demo.mp4", frames, fps=30)
"""
# === COMPREHENSIVE GAUSSIAN FILTER DIAGNOSTICS ===


"""

print("=== SYSTEM DIAGNOSTICS ===")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"SciPy ndimage available: {hasattr(scipy.ndimage, 'gaussian_filter1d')}")

# Test 1: Check if function exists and is callable
print(f"\n=== FUNCTION AVAILABILITY ===")
print(f"gaussian_filter1d function: {gaussian_filter1d}")
print(f"Function type: {type(gaussian_filter1d)}")
print(f"Is callable: {callable(gaussian_filter1d)}")

# Test 2: Basic functionality test
print(f"\n=== BASIC FUNCTIONALITY TEST ===")
a = np.array([0,0,1,0,0], dtype=np.float64)
print(f"Test array: {a}")
print(f"Array dtype: {a.dtype}")
print(f"Array shape: {a.shape}")

try:
    result = gaussian_filter1d(a, 0.5, mode='nearest')
    print(f"Result: {result}")
    print(f"Result dtype: {result.dtype}")
    print(f"Result shape: {result.shape}")
    print(f"Result sum: {np.sum(result)}")
    print(f"Result max: {np.max(result)}")
    print(f"Result min: {np.min(result)}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Test with different sigma values
print(f"\n=== SIGMA TESTS ===")
for sigma in [0.1, 0.5, 1.0, 2.0]:
    try:
        result = gaussian_filter1d(a, sigma, mode='nearest')
        print(f"Sigma {sigma}: {result}")
    except Exception as e:
        print(f"Sigma {sigma} ERROR: {e}")

# Test 4: Test with different modes
print(f"\n=== MODE TESTS ===")
modes = ['nearest', 'constant', 'reflect', 'mirror', 'wrap']
for mode in modes:
    try:
        result = gaussian_filter1d(a, 1.0, mode=mode)
        print(f"Mode {mode}: {result}")
    except Exception as e:
        print(f"Mode {mode} ERROR: {e}")

# Test 5: Manual kernel test
print(f"\n=== MANUAL KERNEL TEST ===")
try:
    # Create manual gaussian kernel
    kernel_size = 5
    sigma = 1.0
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    print(f"Manual kernel: {kernel}")
    
    # Manual convolution
    manual_result = np.convolve(a, kernel, mode='same')
    print(f"Manual convolution: {manual_result}")
except Exception as e:
    print(f"Manual kernel ERROR: {e}")

# Test 6: Test with all ones
print(f"\n=== ONES TEST ===")
ones_array = np.ones(5, dtype=np.float64)
print(f"Ones array: {ones_array}")
try:
    ones_result = gaussian_filter1d(ones_array, 1.0, mode='nearest')
    print(f"Ones result: {ones_result}")
except Exception as e:
    print(f"Ones ERROR: {e}")

# Test 7: Check if it's a scipy import issue
print(f"\n=== IMPORT VERIFICATION ===")
try:
    from scipy.ndimage import gaussian_filter1d as gaussian_filter1d_direct
    print("Direct import successful")
    result_direct = gaussian_filter1d_direct(a, 1.0, mode='nearest')
    print(f"Direct import result: {result_direct}")
except Exception as e:
    print(f"Direct import ERROR: {e}")

# Test 8: Alternative scipy functions
print(f"\n=== ALTERNATIVE FUNCTIONS ===")
try:
    from scipy.ndimage import gaussian_filter
    result_2d = gaussian_filter(a.reshape(1, -1), sigma=1.0, mode='nearest')
    print(f"gaussian_filter result: {result_2d.flatten()}")
except Exception as e:
    print(f"gaussian_filter ERROR: {e}")

# Test 9: Kitchen data test
print(f"\n=== KITCHEN DATA TEST ===")
try:
    data = get_dataset('kitchen', 'complete')
    traj = data.get_trajectories()
    kitchen_rewards = traj[0]['rewards']
    print(f"Kitchen rewards dtype: {kitchen_rewards.dtype}")
    print(f"Kitchen rewards shape: {kitchen_rewards.shape}")
    print(f"Kitchen rewards (first 10): {kitchen_rewards[:10]}")
    print(f"Kitchen rewards sum: {np.sum(kitchen_rewards)}")
    
    kitchen_smoothed = gaussian_filter1d(kitchen_rewards, 1.0, mode='nearest')
    print(f"Kitchen smoothed (first 10): {kitchen_smoothed[:10]}")
    print(f"Kitchen smoothed sum: {np.sum(kitchen_smoothed)}")
except Exception as e:
    print(f"Kitchen data ERROR: {e}")
# Simple targeted test
print("=== SIMPLE TEST ===")
a = np.array([0,0,1,0,0])
print(f"Input array: {a}")
print(f"Array dtype: {a.dtype}")

# Test the function
result = gaussian_filter1d(a, 1.0, mode='nearest')
print(f"Result: {result}")
print(f"Result dtype: {result.dtype}")
print(f"Result sum: {np.sum(result)}")

# Test with explicit float64
a_float = np.array([0,0,1,0,0], dtype=np.float64)
print(f"\nFloat64 input: {a_float}")
result_float = gaussian_filter1d(a_float, 1.0, mode='nearest')
print(f"Float64 result: {result_float}")

# Test with all ones
ones = np.ones(5)
print(f"\nOnes input: {ones}")
result_ones = gaussian_filter1d(ones, 1.0, mode='nearest')
print(f"Ones result: {result_ones}")

# Manual verification
print(f"\n=== MANUAL VERIFICATION ===")
# Create gaussian kernel manually
sigma = 1.0
kernel_size = 5
x = np.arange(kernel_size) - kernel_size // 2
kernel = np.exp(-(x**2) / (2 * sigma**2))
kernel = kernel / np.sum(kernel)
print(f"Manual kernel: {kernel}")
manual_result = np.convolve(a, kernel, mode='same')
print(f"Manual result: {manual_result}")

# Check if the function is actually working
print(f"\n=== FUNCTION CHECK ===")
print(f"Function object: {gaussian_filter1d}")
print(f"Function module: {gaussian_filter1d.__module__}")
print(f"Function doc: {gaussian_filter1d.__doc__[:100]}...")
"""
"""
a = np.array([0,0,1,0,0], dtype=np.float64)
smoothed = gaussian_filter1d(a, 1.0, mode='nearest')
print(f"Float64 result: {smoothed}")
"""



with open('Rollouts/kitchen/partial/Generated_trajs_Info.pkl', 'rb') as f:
    trajs_info = pickle.load(f)

trajs = trajs_info['trajs']
print(len(trajs))
