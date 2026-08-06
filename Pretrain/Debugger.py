#from pstats import StatsProfile
import sys
import os

from torch.distributed import batch_isend_irecv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import numpy as np
import torch
from loguru import logger as log
import minari
from scipy.ndimage import gaussian_filter1d,  gaussian_filter

from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any
import torch.nn as nn
from Dataset import get_dataset, get_dataset
import gymnasium as gym
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
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from Planners.Backbone.Dit import DiT1d
from Planners.Backbone.utils import compute_dot_alpha_beta
from Planners.Backbone.Sampler import sample_reverse_sde
from Dataset import Planner_Processor
import torch.nn.functional as F
from Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
from Dataset import get_dataset
from Rewards.nets import Reward
from Finetuning.traj_reward4 import TotalReward
from Rewards.Reward_Backbone import Train_Dataset, RewardDataset
import random
from Critic.train_critic import get_critic_model, get_critic_stats
from Critic.nets import Critic
try:
    import matplotlib
    #matplotlib.use('Agg')  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be skipped.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
import ogbench


def plot_function(func, x_range=(-10, 10), num_points=1000, title="Function Plot", xlabel="x", ylabel="f(x)"):
    
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))





