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
from collections import deque
import gymnasium as gym
import gymnasium_robotics  # registers the envs
import numpy as np
import torch
import pickle
from scipy.ndimage import gaussian_filter1d
from Pretrain.utils import ema_smooth
from Pretrain.Dataset import get_dataset
import ogbench
from Finetuning.Rollout import load_success_trajs
from Finetuning.utils import reward_processor
from typing import Optional, List
from torch.utils.data import Dataset
import torch.nn as nn
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras

def check_increase(rewards):
    for i in range(1, len(rewards)):
        if( rewards[i] < rewards[i-1]):
            return False
    
    return True


goals = {'task_1': np.array( [ 0.0,       -1.0,        0.199599]), 
         'task_2': np.array([7.50000000e-01, 8.02418254e-18, 1.99598996e-01]),
         'task_3': np.array([-7.50000000e-01,  1.21832368e-19,  1.99598996e-01]),
         'task_4': np.array([0.75,     2.0,       0.199599]),
         'task_5': np.array([ 0.75,     -2.0,        0.199599])}
    
def check_cube_single_goal_reach(trajs, task_id):   
    goals = {'task_1': np.array( [ 0.0,       -1.0,        0.199599]), 
         'task_2': np.array([7.50000000e-01, 8.02418254e-18, 1.99598996e-01]),
         'task_3': np.array([-7.50000000e-01,  1.21832368e-19,  1.99598996e-01]),
         'task_4': np.array([0.75,     2.0,       0.199599]),
         'task_5': np.array([ 0.75,     -2.0,        0.199599])}
    
    total_dist = 0.0
    for traj in trajs:
           position = traj['observations'][-1][19:22]
           total_dist += np.linalg.norm(position - goals[f"task_{task_id}"])
    average_dist = total_dist/len(trajs)
    print(f"Task {task_id} average distance: {average_dist}")

def check_cube_double_goal_reach(trajs, task_id):   
    goals = {   'task_1': [np.array([0.00000000e+00, 4.40762988e-19, 1.99598996e-01]),  np.array([0.0,   1.0,   0.199599])], 
                'task_2': [np.array([-0.75,      1.0,        0.199599]),  np.array([0.75,     1.0,       0.199599])],
                'task_3': [np.array([0.0,       -2.0,        0.199599]),  np.array([0.0,      2.0,       0.199599])],
                'task_4': [np.array([0.0,        1.0,        0.199599]),  np.array([0.0,       -1.0,        0.199599])],
                'task_5': [np.array([0.00000000e+00,  -3.99397428e-18,   1.99213779e-01]),  np.array([0.00000000e+00,   9.37726514e-18,   5.99039293e-01])]     }
    total_dist = 0.0
    for traj in trajs:
           position_1 = traj['observations'][-1][19:22]
           position_2 = traj['observations'][-1][28:31]
           dist_1 = np.linalg.norm(position_1 - goals[f"task_{task_id}"][0])
           dist_2 = np.linalg.norm(position_2 - goals[f"task_{task_id}"][1])
           total_dist += dist_1 + dist_2
    average_dist = total_dist/len(trajs)
    print(f"Task {task_id} average distance: {average_dist}")



env, dataset, eval_dataset = ogbench.make_env_and_datasets(
                 "cube-double-play-singletask-task4-v0", render_mode="rgb_array"
                  )
last_start = 0
print(len(dataset['actions'][1]))
exit()
for i in range(1, len(dataset['rewards'])):
    if(dataset['rewards'][i] == 0  or dataset['terminals'][i] == 1):
          if(not check_increase(dataset['rewards'][last_start: i+1])):
             print('False')
             exit()
          rews_slice = reward_processor(dataset['rewards'][last_start:i+1].copy(), 'cube')
          if len(rews_slice) < 10:
                    last_start = i + 1
                    continue
          print()
          last_start = i + 1

exit()


"""
env_steps = [0, 1592, 1590, 1600, 1411, 1416, 1600, 1555, 1422, 1600, 1599, 1554]
total_steps = [0]
for i in range(1, len(env_steps)):
    total_steps.append(total_steps[i-1] + env_steps[i])

print(len(total_steps))
print(len(env_steps))
"""



"""
data = get_dataset('cube', 'single-play', task_id = 5)
trajs = data.get_trajectories()
"""



"""
for traj in trajs:
    rews = traj['rewards']
    rews[-1] = 50.0
    print(len(rews))
    rews = rews[len(rews)-50:]
    print(len(rews))
    rews = gaussian_filter1d(rews, sigma = 8.0, mode = 'nearest')
    print(rews)
    exit()
"""
"""
import ogbench
env, dataset, eval_dataset = ogbench.make_env_and_datasets(
                'cube-single-play-singletask-v0', render_mode="rgb_array"
            )

#for i in range(len(dataset['rewards'])):
print((dataset['rewards'].shape))
"""




"""
from Finetuning.Rollout import rollout
from Finetuning.utils import check_device
from Pretrain.utils import set_seed
device = check_device()
env_name = 'cube'
specific_train_dataset = 'single-play'
horizon = 32
checkpoint = 0
set_seed(8)
reward  =  rollout(
               env_name, 
               specific_train_dataset, 
               horizon, 
               steps_T = 200, 
               num_karras = 10, 
               eta = 0.8, 
               episode_length = 3000, 
               checkpoint_steps = checkpoint, 
               render = True,  
               base_seed = 1, 
               task_id = 4,
               continual_rollout = True,
               chunk_size = 32,
               device = device)

print(reward)
"""


from Finetuning.Rollout import load_success_trajs
from Finetuning.utils import TrajectoryDict, reward_processor, train_critic, test_critic
from Pretrain.utils import set_seed

env_name = 'cube'
specific_env = 'single-play'
task_id = 4
step = 0
traj_length = 200

set_seed(1)
"""
trajs = load_success_trajs(env_name, specific_env, task_id, step)
test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 80.0, 
            trajs = trajs,
            task_id = task_id)
"""
trajs = load_success_trajs(env_name, specific_env, task_id, step)
train_critic(trajs, 
             dataset_name = env_name, 
             specific_dataset = specific_env, 
             hidden_layers = 4, 
             hidden_dim = 512, 
             sigma = 3.0,
             batch_size = 256, 
             num_steps = 20000, 
             gamma = 0.99, 
             lam = 0.95, 
             horizon = 32, 
             lr = 5e-05, 
             min_lr = 1e-06, 
             tau = 0.005, 
             old_step = None, 
             new_step = 0, 
             momentum = 0.005, 
             target_reward = 80.0,
             task_id = task_id)

trajs = load_success_trajs(env_name, specific_env, task_id, step)
test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 80.0, 
            trajs = trajs,
            task_id = task_id)





"""

from Finetuning.Rollout import get_success_trajs
from Finetuning.utils import train_critic
from Pretrain.utils import set_seed
from Finetuning.utils import test_critic
env_name = 'pointmaze'
specific_env = 'medium'
save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_0.pkl'
with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
trajs = get_success_trajs(trajs)


set_seed(1)
test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 3, 
            hidden_dim = 256, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 20.0, 
            trajs = trajs)

with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
trajs = get_success_trajs(trajs)


train_critic(trajs, 
             dataset_name = env_name, 
             specific_dataset = specific_env, 
             hidden_layers = 3, 
             hidden_dim = 256, 
             sigma = 7.0,
             batch_size = 256, 
             num_steps = 5000, 
             gamma = 0.99, 
             lam = 0.95, 
             horizon = 32, 
             lr = 1e-05, 
             min_lr = 1e-06, 
             tau = 0.005, 
             old_step = 0, 
             new_step = 10, 
             momentum = 0.005, 
             target_reward = 20.0)


with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
trajs = get_success_trajs(trajs)


test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 3, 
            hidden_dim = 256, 
            checkpoint_step = 10, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 20.0, 
            trajs = trajs)

"""

