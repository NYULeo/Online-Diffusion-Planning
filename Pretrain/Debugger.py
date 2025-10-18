from pstats import StatsProfile
import sys
import os
import logging
import numpy as np
from pandas._libs.algos import take_2d_axis0_float32_float32
import torch
import gymnasium as gym# Conditional import to avoid GLFW3 errors on headless servers

from loguru import logger as log
import minari
from scipy.ndimage import gaussian_filter1d, convolve

from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any
import torch.nn as nn
from Dataset import get_dataset, get_dataset

import torch
import math
from utils import set_seed
from Dataset import get_env, get_dataset
import gymnasium_robotics
import mediapy as media
from collections import namedtuple

from Planners.Backbone.utils import get_pretrained_planner
from torch.utils.data import DataLoader
from Dataset import PlannerDataset
from Rewards.nets import gaussian_rewards
import scipy
import scipy.ndimage
from sympy import factorint
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from Planners.Backbone.Dit import DiT1d
from Planners.Backbone.utils import compute_dot_alpha_beta
from Planners.Backbone.Sampler import sample_reverse_sde
from Dataset import Planner_Processor




def plot_function(func, x_range=(-10, 10), num_points=1000, title="Function Plot", xlabel="x", ylabel="f(x)"):
    """
    Plot a mathematical function.
    
    Args:
        func: A function that takes x and returns f(x)
        x_range: Tuple of (min_x, max_x) for the plotting range
        num_points: Number of points to plot
        title: Title of the plot
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))


"""

def kt(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    
    t = t.clamp(0.0, 1.0 - 1e-3)
    a = (math.pi / 2.0) * ((t + s) / (1.0 + s))
    return (-0.5)* (math.pi / (1.0 + s)) * torch.tan(a)

@torch.no_grad()
def sample_reverse_sde2(
    s0: np.ndarray,
    score_model: DiT1d,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int,
    eta: float,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0_t = torch.tensor(s0, device=device, dtype=torch.float32)
    if ( (s0_t.shape[0] != d_s)   ):
        raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")
    dim = d_s + d_a
    t_asc = torch.linspace(1.0, 0.0, steps_T + 1, device=device)
    #beta = cosine_beta(t_asc, s=0.008)
    #alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)
    k = kt(t_asc, s = 0.008)
    
    # Initialize x_T ~ N(0, I) with shape (horizon, dim)
    x = torch.randn(horizon, dim, dtype=torch.float32, device=device).unsqueeze(0)
    conditions = s0_t.unsqueeze(0)
    mask = torch.zeros((1, horizon, dim), dtype = torch.float32, device = device)
    mask[:, 0, :d_s] = 1
    y = torch.zeros((1, horizon, dim), dtype = torch.float32, device = device)
    y[:, 0, :d_s] = conditions.clone()
    #x = apply_conditioning(x, conditions, d_s)
    x = mask * y + (1 - mask) * x
    
    

    for i in range(len(t_asc) - 1):
        t_now, t_next = t_asc[i], t_asc[i + 1]
        dt = (t_next - t_now).item()
        score = score_model(x, t_now.unsqueeze(0))
        drift = k[i] * x
        
       

        if eta > 0:
            noise = torch.randn_like(x)
            noise_scale = eta * math.sqrt((-2*k[i]) * (-dt))
            x = x + (drift +  2*k[i] * score ) * dt + noise_scale * noise
        else:
            x = x + (drift + 2*k[i] * score) * dt
        
        x = mask * y + (1 - mask) * x
        
        
        #x = apply_conditioning(x, conditions, d_s)

    return x.squeeze(0).detach().cpu().numpy()






def rollout(env_name, specific_env, horizon, steps_T, eta, episode_length, critic, checkpoint_steps, render = False):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checpoint_steps; {checkpoint_steps}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device {device}")
     
     
     #get environment
     if(render):
         env, d_s, d_a = get_env(env_name, specific_env, 'rgb_array')
     else:
         env, d_s, d_a = get_env(env_name, specific_env, None)

     #get Planner
     state_dict = get_pretrained_planner(env_name, specific_env, checkpoint_steps)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()

    #get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     
     #reset
     s0 = env.reset(seed=1)
     s0 = s0[0]['observation']
     current_state = s0
     frames = []
     observations = []
     actions = []
     rewards = []
     for i in range(episode_length):
           current_state_norm = planner_processor.preprocess(current_state)
    
           #x1 = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
           x2 = sample_reverse_sde3(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)



           action = x2[0, d_s:(d_s+d_a)].copy()
           
           obs, reward, terminated, truncated, info = env.step(action)
           if(render):
                frames.append(env.render())
           
           observations.append(obs['observation'].copy())
           actions.append(action.copy())
           rewards.append(reward)
           current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     
     env.close()
     traj = {'observations': observations, 'actions': actions, 'rewards': rewards}
     traj_info = {'sequence': traj, 'env_name': env_name, 'specific_env': specific_env }
     if(render):
          media.write_video("demo.mp4", frames, fps=50)
    


x = torch.tensor([[1,2,3], [4,5,6]]).unsqueeze(0)
print(x.view(-1))
"""

#plot_function(function, x_range=(-10, 10), num_points=1000, title="Function Plot", xlabel="x", ylabel="f(x)")
print(function(0))

