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
    train_critic_with_planner3,
    train_critic_with_planner4,
    train_critic_with_planner5,
    train_critic,
    test_critic,
    KernelConfig,
)
from Pretrain.utils import set_seed
from accelerate import Accelerator


if __name__ == '__main__':  # pragma: no cover
       set_seed(1)
       env_name = 'cube'
       specific_env = 'single-play'
       traj_length = 200
       horizon = 128
       task_id = 4
       step = 0
      
       
       """
       #trajs = load_success_trajs(env_name, specific_env, task_id, step)
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
       train_critic_with_reward(trajs,
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
     
      
       
       trajs = data.get_trajectories()
       test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = horizon,  
            sigma = 4.0, 
            target_reward = 500.0, 
            trajs = trajs,
            task_id = task_id)
       """
       
       
       data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
       trajs = data.get_trajectories()
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
       accelerator = Accelerator()
       accelerator.wait_for_everyone()
       """
       mean, std = train_critic_with_planner3(
            trajs                  = trajs,
            dataset_name           = env_name,
            specific_dataset       = specific_env,
            planner_checkpoint     = 0,
            reward_checkpoint      = 0,
            old_critic_checkpoint  = None,
            backbone_layers        = 2,
            hidden_layers          = 4,
            hidden_dim             = 512,
            kernel_config          = kernel_config,
            reward_hidden_layers   = 4,
            reward_hidden_dim      = 512,
            batch_size             = 128,
            num_steps              = 1000,
            horizon                = 32,
            gamma                  = 0.99,
            lr                     = 5e-5,
            min_lr                 = 1e-8,
            tau                    = 0.005,
            steps_T                = 10,
            num_karras             = 1,
            eta                    = 0.0,
            new_step               = 0,
            task_id                = task_id,
            log_every              = 100,
         )
       """
       train_critic_with_planner5(
                trajs=trajs,
                dataset_name=env_name,
                specific_dataset=specific_env,
                planner_checkpoint=0,
                reward_checkpoint=0,
                old_critic_checkpoint=None,
                backbone_layers=2,
                hidden_layers=4,
                hidden_dim=512,
                kernel_config=kernel_config,
                reward_hidden_layers=4,
                reward_hidden_dim=512,
                batch_size=64,
                num_steps=200,
                horizon=32,
                gamma=0.99,
                lr=5e-6,
                min_lr=1e-9,
                tau=0.005,
                steps_T=10,
                num_karras=1,
                eta=0.0,
                new_step=0,
                task_id=task_id,
                log_every=20,
                use_multi_horizon = True,
                accelerator=accelerator,
        )

       accelerator.wait_for_everyone()
       #trajs = load_success_trajs(env_name, specific_env, task_id, step)
       trajs = data.get_trajectories()
       if accelerator.is_main_process:
           test_critic(dataset_name = env_name, 
                       specific_dataset = specific_env, 
                       hidden_layers = 4, 
                       hidden_dim = 512, 
                       checkpoint_step = 0, 
                       gamma = 0.99, 
                       horizon = horizon,  
                       sigma = 4.0, 
                       target_reward = 500.0, 
                       trajs = trajs,
                       task_id = task_id)
     
