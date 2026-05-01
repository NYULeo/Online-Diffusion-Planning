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

from collections import deque
import gymnasium as gym
import gymnasium_robotics  # registers the envs
import numpy as np
import torch
import pickle


path = f'./Finetuning/Rollouts/cube/single-play/task_1/Generated_trajs_Info_0.pkl'
with open(path, 'rb') as f:
    trajs = pickle.load(f)

print(len(trajs))










