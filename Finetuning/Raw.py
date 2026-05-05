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




path = f'./Finetuning/Rollouts/cube/single-play/task_4/trajs_task4_success_0.pkl'
with open(path, 'rb') as f:
    trajs = pickle.load(f)


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






