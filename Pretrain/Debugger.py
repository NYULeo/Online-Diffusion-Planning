import sys
import os
import logging
import numpy as np
import torch
import gymnasium as gym# Conditional import to avoid GLFW3 errors on headless servers

from loguru import logger as log
import minari


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

Xfix = torch.zeros(2, 3)
M = torch.zeros(2,3)

M[0, :] = 1.0 
print(M)
Xfix[0, :] = torch.tensor([1,2,3])
print(Xfix)

x = torch.tensor( [ [4,5,6, 7, 8], [7,8,9, 10, 11]          ]       )
print(x)
"""
x = M * Xfix + (1.0 - M) * x
print(x)
"""

print(int(1e6 // 10000))

