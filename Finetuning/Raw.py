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
from Pretrain.Dataset import get_dataset

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


"""
path = f'./Finetuning/Rollouts/cube/single-play/task_4/trajs_task4_success_0.pkl'
with open(path, 'rb') as f:
    trajs = pickle.load(f)

data = get_dataset('cube', 'double-play', task_id = 5, traj_length = None)
trajs = data.get_trajectories()
"""

a = torch.tensor([[1,2,3,3]])
a = a.squeeze(0)
print(a)


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






