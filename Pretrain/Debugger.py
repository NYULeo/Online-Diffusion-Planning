import sys
import os
import logging
import numpy as np
import torch
import gymnasium as gym# Conditional import to avoid GLFW3 errors on headless servers

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
from loguru import logger as log
import minari
from Dataset import KitchenDataset, get_env

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


x = torch.tensor([[ [1,1,1], [1,1,1]], [ [1,1,1], [1,1,1]]]).sum(dim=(1, 2)) 
print(x)
