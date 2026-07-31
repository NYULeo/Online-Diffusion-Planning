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

from torch._C import NoneType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import gymnasium as gym
import gymnasium_robotics  # registers the envs
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
                             num_steps  = 20000,
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
      
       
       accelerator = Accelerator(mixed_precision='bf16')
       #accelerator.wait_for_everyone()
       kernel_config = KernelConfig(
                checkpoint = 0,
                type_kernel = 'mog',
                num_hidden_layers = 4,
                hidden_dim = 514,
                num_modes = 10,
                noise_floor = 5e-4,
                min_log_prob = -110.0,
                oversample = 5,
        )
       
       mean, std = train_critic_with_planner4(
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
                               num_steps              = 1000,
                               horizon                = 32,
                               gamma                  = 0.99,
                               lam                    = 0.95,
                               lr                     = 1e-06,
                               min_lr                 = 1e-09,
                               tau                    = 0.005,
                               steps_T                = 10,
                               num_karras             = 1,
                               eta                    = 0.0,
                               new_step               = 0,
                               task_id                = task_id,
                               log_every              = 100,
                               accelerator            = accelerator) 
      
       accelerator.wait_for_everyone()
       
       #if accelerator.is_main_process:
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            mean = mean,
            std = std,
            gamma = 0.99, 
            horizon = horizon,  
            #sigma = 4.0, 
            sigma = None,
            target_reward = 500.0, 
            trajs = trajs,
            task_id = task_id)
"""

def filter_trajs(trajs):
    successes = [t for t in trajs if t['rewards'][-1] > 0.0]
    failures = [t for t in trajs if t['rewards'][-1] <= 0.0]
    k = min(2 * len(successes), len(failures))
    return successes + random.sample(failures, k=k)

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
       
       trajs = filter_trajs(trajs)
       
       mean, std = train_critic_with_reward(trajs,
                             dataset_name  = env_name,
                             specific_dataset = specific_env,
                             reward_hidden_layers = 4,
                             reward_hidden_dim  = 512,
                             reward_checkpoint  = 0,
                             critic_hidden_layers = 4,
                             critic_hidden_dim  = 512,
                             batch_size = 256,
                             num_steps  = 20000,
                             gamma = 0.99,
                             lam = 0.95,
                             horizon = horizon,
                             lr = 1e-07, 
                             min_lr = 1e-08, 
                             tau = 0.005, 
                             old_step = None,    # from scratch
                             new_step = step,
                             momentum = 0.005,   # unused when old_step is None
                             task_id = task_id)
    
       
       """
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
                oversample = 5,
        )
       
       
       mean, std = train_critic_with_planner4(
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
                               num_steps              = 1000,
                               horizon                = 32,
                               gamma                  = 0.99,
                               lam                    = 0.95,
                               lr                     = 1e-07,
                               min_lr                 = 1e-10,
                               tau                    = 0.005,
                               steps_T                = 10,
                               num_karras             = 1,
                               eta                    = 0.0,
                               new_step               = 0,
                               task_id                = task_id,
                               log_every              = 100,
                               accelerator            = accelerator) 
      
       accelerator.wait_for_everyone()
       """
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
            target_reward = 800.0, 
            trajs = trajs,
            task_id = task_id)


    













