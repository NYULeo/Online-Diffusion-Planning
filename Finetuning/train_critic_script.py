import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import ogbench as og
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import gymnasium as gym
import numpy as np
import torch
import pickle
from scipy.ndimage import gaussian_filter1d
from Pretrain.Dataset import get_dataset
import ogbench
from Finetuning.Rollout import load_success_trajs
from Finetuning.utils import (
    train_critic_with_reward,
    train_critic_with_planner,
    train_critic_with_planner3,
    train_critic_with_planner4,
    train_critic_with_planner5,
    train_critic,
    test_critic,
    KernelConfig,
)
from Pretrain.utils import set_seed
from accelerate import Accelerator
import random 
import wandb

"""
if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
      
       
       env_name = 'cube'
       specific_env = 'single-play'
       traj_length = 200
       horizon = 128
       task_id = 4
       step = 0
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
       
      
       
       mean, std = train_critic_with_reward(trajs,
                             dataset_name  = env_name,
                             specific_dataset = specific_env,
                             reward_hidden_layers = 4,
                             reward_hidden_dim  = 512,
                             reward_checkpoint  = 0,
                             critic_hidden_layers = 4,
                             critic_hidden_dim  = 512,
                             batch_size = 256,
                             num_steps  = 10000,
                             gamma = 0.99,
                             lam = 0.95,
                             horizon = horizon,
                             #lr = 1e-04, 
                             lr = 1e-04,
                             #min_lr = 1e-05, 
                             min_lr = 1e-05,
                             tau = 0.005, 
                             old_step = None,    # from scratch
                             new_step = step,
                             momentum = 0.005,   # unused when old_step is None
                             task_id = task_id)
      
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            mean = None,
            std = None,
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 4.0, 
            #sigma = None,
            target_reward = 500.0, 
            trajs = trajs,
            task_id = task_id)
"""


"""
if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
       
       
       env_name = 'cube'
       specific_env = 'double-play'
       traj_length = 500
       horizon = 480
       task_id = 4
       step = 0
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
       
       
       
       mean, std = train_critic_with_reward(trajs,
                             dataset_name  = env_name,
                             specific_dataset = specific_env,
                             reward_hidden_layers = 4,
                             reward_hidden_dim  = 512,
                             reward_checkpoint  = 0,
                             critic_hidden_layers = 4,
                             critic_hidden_dim  = 512,
                             batch_size = 256,
                             num_steps  = 5000,
                             gamma = 0.99,
                             lam = 0.95,
                             horizon = horizon,
                             lr = 1e-04, 
                             min_lr = 1e-05, 
                             tau = 0.005, 
                             old_step = None,    # from scratch
                             new_step = step,
                             momentum = 0.005,   # unused when old_step is None
                             task_id = task_id)
    
       
       
       
    
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            mean = None,
            std = None,
            gamma = 0.99, 
            horizon = horizon,  
            #sigma = 3.0, 
            sigma = None,
            #target_reward = 10.0, 
            target_reward = None, 
            trajs = trajs,
            task_id = task_id)

"""
    

if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
       env_name = 'antmaze'
       specific_env = 'large'
       traj_length = 1000
       horizon = 800
       task_id = 4
       step = 0
       wandb.init(
           entity="kaiwen_hu-uc-berkeley",
           project="ODP",
           name=f"{env_name}-{specific_env}-task{task_id}-critic_1",
           config={
               "dataset_name": env_name,
               "specific_dataset": specific_env,
               "task_id": task_id,
               "traj_length": traj_length,
               "horizon": horizon,
               "reward_hidden_layers": 4,
               "reward_hidden_dim": 512,
               "reward_checkpoint": 0,
               "critic_hidden_layers": 4,
               "critic_hidden_dim": 512,
               "batch_size": 1024,
               "num_steps": 10000,
               "gamma": 0.99,
               "lam": 0.95,
               "lr": 1e-04,
               "min_lr": 1e-05,
               "tau": 0.005,
               "old_step": None,
               "new_step": step,
               "momentum": 0.005,
           }
       )
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
       
       mean, std = train_critic_with_reward(trajs,
                             dataset_name  = env_name,
                             specific_dataset = specific_env,
                             reward_hidden_layers = 4,
                             reward_hidden_dim  = 512,
                             reward_checkpoint  = 0,
                             critic_hidden_layers = 4,
                             critic_hidden_dim  = 512,
                             #batch_size = 256,
                             batch_size = 1024,
                             num_steps  = 10000,
                             gamma = 0.99,
                             lam = 0.95,
                             horizon = horizon,
                             #lr = 1e-04, 
                             lr = 1e-04,
                             #min_lr = 1e-05, 
                             min_lr = 1e-05,
                             tau = 0.005, 
                             old_step = None,    # from scratch
                             new_step = step,
                             momentum = 0.005,   # unused when old_step is None
                             task_id = task_id)
       #wandb.finish()
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            mean = None,
            std = None,
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 6.0, 
            #sigma = None,
            target_reward = 2000.0, 
            trajs = trajs,
            task_id = task_id)
       wandb.finish()
      




"""
if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
       env_name = 'humanoidmaze'
       specific_env = 'large'
       traj_length = 2000
       horizon = 1800
       task_id = 2
       step = 0
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
       
       mean, std = train_critic_with_reward(trajs,
                             dataset_name  = env_name,
                             specific_dataset = specific_env,
                             reward_hidden_layers = 4,
                             reward_hidden_dim  = 1024,
                             reward_checkpoint  = 0,
                             critic_hidden_layers = 4,
                             critic_hidden_dim  = 512,
                             batch_size = 256,
                             num_steps  = 10000,
                             gamma = 0.99,
                             lam = 0.95,
                             horizon = horizon,
                             #lr = 1e-04, 
                             lr = 1e-04,
                             #min_lr = 1e-05, 
                             min_lr = 1e-05,
                             tau = 0.005, 
                             old_step = None,    # from scratch
                             new_step = step,
                             momentum = 0.005,   # unused when old_step is None
                             task_id = task_id)
      
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            mean = None,
            std = None,
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 4.0, 
            #sigma = None,
            target_reward = 500.0, 
            trajs = trajs,
            task_id = task_id)

"""







