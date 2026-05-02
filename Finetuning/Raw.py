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



path = f'./Finetuning/Rollouts/cube/single-play/task_1/Generated_trajs_Info_0.pkl'
with open(path, 'rb') as f:
    trajs = pickle.load(f)
from Finetuning.utils import CriticDataset
dataset = CriticDataset(trajs, sigma = 7.0, dataset_name = 'cube', specific_dataset = 'single-play', step = 0, goal = None, target_reward = None, horizon = 32, gamma = 0.99)










