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
    train_critic_with_planner2,
    train_critic,
    test_critic,
    KernelConfig,
)
from Pretrain.utils import set_seed


if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
       env_name = 'cube'
       specific_env = 'double-play'
       traj_length = None
       horizon = 100
       #horizon = 300
       task_id = 4
       step = 0
      
       
        
       #trajs = load_success_trajs(env_name, specific_env, task_id, step)
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
       """
       train_critic(trajs, 
             dataset_name = env_name, 
             specific_dataset = specific_env, 
             hidden_layers = 4, 
             hidden_dim = 512, 
             sigma = 4.0,
             batch_size = 256, 
             num_steps = 20000, 
             gamma = 0.99, 
             lam = 0.95, 
             horizon = horizon, 
             #lr = 1e-04, 
             #min_lr = 1e-05, 
             lr = 1e-05, 
             min_lr = 5e-07, 
             tau = 0.005, 
             old_step = None, 
             new_step = step, 
             momentum = 0.005, 
             target_reward = 300.0,
             task_id = task_id)  
       """
       
       train_critic_with_reward(trajs,
                      dataset_name  = env_name,
                      specific_dataset = specific_env,
                      reward_hidden_layers = 5,
                      reward_hidden_dim  = 512,
                      reward_checkpoint  = 0,
                      critic_hidden_layers = 5,
                      critic_hidden_dim  = 512,
                      batch_size = 256,
                      num_steps  = 50000,
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
            finetune = False,
            hidden_layers = 5, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 4.0, 
            target_reward = 300.0, 
            trajs = trajs,
            task_id = task_id)
       
       """
       
       kernel_config = KernelConfig(
             checkpoint        = 0,           # which kernel checkpoint to load
             type_kernel       = 'mog',    # 'robust' or 'mog'
             num_hidden_layers = 4,           # must match training-time arch
             hidden_dim        = 514,         # must match training-time arch
             num_modes         = 10,           # only used when type_kernel == 'mog'
             noise_floor       = 5e-4,        # only used when type_kernel == 'mog'
             min_log_prob      = -110.0,       # feasibility threshold (tune per kernel type)
             oversample        = 4,           # try up to oversample * batch_size candidates
       )


       train_critic_with_planner2(
            trajs                  = trajs,
            dataset_name           = env_name,
            specific_dataset       = specific_env,
            planner_checkpoint     = 0,
            reward_checkpoint      = 0,
            old_critic_checkpoint  = 0,
            hidden_layers          = 4,
            hidden_dim             = 512,
            kernel_config          = kernel_config,
            reward_hidden_layers   = 4,
            reward_hidden_dim      = 512,
            batch_size             = 128,
            num_steps              = 10,
            horizon                = 32,
            gamma                  = 0.99,
            lr                     = 5e-5,
            min_lr                 = 1e-6,
            tau                    = 0.005,
            steps_T                = 10,
            num_karras             = 1,
            eta                    = 0.0,
            new_step               = 1,
            task_id                = task_id,
            log_every              = 1,
         )


       #trajs = load_success_trajs(env_name, specific_env, task_id, step)
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            finetune = True,
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 1, 
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 4.0, 
            target_reward = 500.0, 
            trajs = trajs,
            task_id = task_id)
      """
