import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import ogbench as og
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import minari
import sys

from sympy import Max
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from collections import deque
import gymnasium as gym
import gymnasium_robotics  # registers the envs
import numpy as np
import torch
import pickle
from scipy.ndimage import gaussian_filter1d
from Pretrain.Dataset import get_dataset
import ogbench
from Finetuning.Rollout import load_success_trajs
from Finetuning.Rollout import load_success_trajs
from Finetuning.utils import train_critic, test_critic
from Pretrain.utils import set_seed


if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
       env_name = 'cube'
       specific_env = 'single-play'
       horizon = 50
       task_id = 4
       traj_length = None
       step = 0
      
       
        
       trajs = load_success_trajs(env_name, specific_env, task_id, step)
       
       train_critic(trajs, 
             dataset_name = env_name, 
             specific_dataset = specific_env, 
             hidden_layers = 4, 
             hidden_dim = 512, 
             sigma = 3.0,
             batch_size = 256, 
             num_steps = 1000, 
             gamma = 0.99, 
             lam = 0.95, 
             horizon = horizon, 
             lr = 1e-05, 
             min_lr = 1e-06, 
             tau = 0.005, 
             old_step = None, 
             new_step = step, 
             momentum = 0.005, 
             target_reward = 80.0,
             task_id = task_id)

       trajs = load_success_trajs(env_name, specific_env, task_id, step)
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = step, 
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 3.0, 
            target_reward = 80.0, 
            trajs = trajs,
            task_id = task_id)
 