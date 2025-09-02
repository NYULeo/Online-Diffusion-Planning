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
from Dataset import get_dataset
import torch
import math



def cosine_beta(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Continuous-time VP drift g(t)^2 = beta(t) for the cosine schedule.
    Using beta(t) = -2 d/dt log alpha(t) = (pi/(1+s)) * tan(a).
    """

    t = t.clamp(0.0, 1.0 - 1e-6)
    a = (math.pi / 2.0) * (t + s) / (1.0 + s)
    return (math.pi / (1.0 + s)) * torch.tan(a)

print(cosine_beta(torch.tensor(1), s = 0.008))