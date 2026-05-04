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



path = f'./Finetuning/Rollouts/cube/single-play/task_1/Generated_trajs_Info_0.pkl'
with open(path, 'rb') as f:
    trajs = pickle.load(f)

data = get_dataset('cube', 'single-play', task_id = 4)
trajs = data.get_trajectories()

print(trajs[9]['rewards'][-1])
print(trajs[9]['observations'][-1][19:22])

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






