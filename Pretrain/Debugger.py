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
from Dataset import KitchenDataset, get_env


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
from pretrain_planner import get_PlannerName



env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle'], render_mode = 'rgb_array')
print(env)



