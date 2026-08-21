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
    train_critic_with_planner6,
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
    
       accelerator = Accelerator(mixed_precision='bf16')
       kernel_config = KernelConfig(
                checkpoint = 0,
                type_kernel = 'mog',
                num_hidden_layers = 4,
                hidden_dim = 514,
                num_modes = 10,
                noise_floor = 5e-4,
                min_log_prob = -110.0,
                #min_log_prob = -130.0,
                oversample = 5,
        )
       
       mean, std = train_critic_with_planner6(
                               trajs                  = trajs,
                               dataset_name           = env_name,
                               specific_dataset       = specific_env,
                               planner_checkpoint     = 0,
                               reward_checkpoint      = 0,
                               old_critic_checkpoint  = 0,
                               backbone_layers        = 2,
                               hidden_layers          = 4,
                               hidden_dim             = 512,
                               kernel_config          = kernel_config,
                               reward_hidden_layers   = 4,
                               reward_hidden_dim      = 512,
                               batch_size             = 64,
                               num_steps              = 100,
                               horizon                = 32,
                               gamma                  = 0.99,
                               lam                    = None,
                               rho                    = 1.0,
                               lr                     = 1e-04,
                               min_lr                 = 1e-05,
                               tau                    = 0.005,
                               steps_T                = 10,
                               num_karras             = 1,
                               eta                    = 0.0,
                               new_step               = 0,
                               task_id                = task_id,
                               log_every              = 20,
                               accelerator            = accelerator) 
      
       accelerator.wait_for_everyone()
       
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
       
       
      
       
       
       
       accelerator = Accelerator(mixed_precision='bf16')
       #accelerator.wait_for_everyone()
       kernel_config = KernelConfig(
                checkpoint = 0,
                type_kernel = 'mog',
                num_hidden_layers = 4,
                hidden_dim = 514,
                num_modes = 10,
                noise_floor = 5e-4,
                min_log_prob = -170.0,
                oversample = 20,
        )
       
       
       mean, std = train_critic_with_planner6(
                               trajs                  = trajs,
                               dataset_name           = env_name,
                               specific_dataset       = specific_env,
                               planner_checkpoint     = 0,
                               reward_checkpoint      = 0,
                               old_critic_checkpoint  = 0,
                               backbone_layers        = 4,
                               hidden_layers          = 4,
                               hidden_dim             = 512,
                               kernel_config          = kernel_config,
                               reward_hidden_layers   = 4,
                               reward_hidden_dim      = 512,
                               batch_size             = 64,
                               num_steps              = 100,
                               horizon                = 32,
                               gamma                  = 0.99,
                               lam                    = None,
                               rho                    = 1.0,
                               lr                     = 1e-04,
                               min_lr                 = 1e-06,
                               tau                    = 0.005,
                               steps_T                = 10,
                               num_karras             = 1,
                               eta                    = 0.0,
                               new_step               = 0,
                               task_id                = task_id,
                               log_every              = 20,
                               accelerator            = accelerator) 
       accelerator.wait_for_everyone()
       
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
            #sigma = 6.0, 
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
       accelerator = Accelerator(mixed_precision='bf16')
       if accelerator.is_main_process:
           wandb.init(
               entity="kaiwen_hu-uc-berkeley",
               project="ODP",
               name=f"{env_name}-{specific_env}-task{task_id}-critic_2",
               config={
                   "dataset_name": env_name,
                   "specific_dataset": specific_env,
                   "task_id": task_id,
                   "traj_length": traj_length,
                   "horizon": horizon,
                   "planner_checkpoint": 0,
                   "reward_checkpoint": 0,
                   "old_critic_checkpoint": 0,
                   "backbone_layers": 4,
                   "hidden_layers": 4,
                   "hidden_dim": 512,
                   "reward_hidden_layers": 4,
                   "reward_hidden_dim": 512,
                   "batch_size": 128,
                   "num_steps": 100,
                   "train_horizon": 32,
                   "gamma": 0.99,
                   "lam": None,
                   "rho": 1.0,
                   "lr": 1e-04,
                   "min_lr": 1e-05,
                   "tau": 0.005,
                   "steps_T": 10,
                   "num_karras": 1,
                   "eta": 0.0,
                   "new_step": 0,
                   "log_every": 20,
                   "kernel_type": "mog",
                   "kernel_checkpoint": 0,
                   "num_modes": 10,
                   "oversample": 15,
               }
           )
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
    
       kernel_config = KernelConfig(
                checkpoint = 0,
                type_kernel = 'mog',
                num_hidden_layers = 4,
                hidden_dim = 514,
                num_modes = 10,
                noise_floor = 5e-4,
                min_log_prob = -110.0,
                #min_log_prob = -130.0,
                oversample = 5,
                #oversample = 15
        )
       
       mean, std = train_critic_with_planner6(
                               trajs                  = trajs,
                               dataset_name           = env_name,
                               specific_dataset       = specific_env,
                               planner_checkpoint     = 0,
                               reward_checkpoint      = 0,
                               old_critic_checkpoint  = 0,
                               backbone_layers        = 4,
                               hidden_layers          = 4,
                               hidden_dim             = 512,
                               kernel_config          = kernel_config,
                               reward_hidden_layers   = 4,
                               reward_hidden_dim      = 512,
                               batch_size             = 64,
                               #batch_size             = 128,
                               num_steps              = 100,
                               horizon                = 32,
                               gamma                  = 0.99,
                               lam                    = None,
                               rho                    = 1.0,
                               lr                     = 1e-04,
                               min_lr                 = 1e-05,
                               tau                    = 0.005,
                               steps_T                = 10,
                               num_karras             = 1,
                               eta                    = 0.0,
                               new_step               = 0,
                               task_id                = task_id,
                               log_every              = 20,
                               accelerator            = accelerator) 
      
       accelerator.wait_for_everyone()

       #if accelerator.is_main_process:
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
       if accelerator.is_main_process:
           wandb.finish()











