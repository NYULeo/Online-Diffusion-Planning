import sys
import os

#from Finetuning.heatmap_plot import critic_heatmap
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import torch
import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset
from Pretrain.utils import SAStats
from scipy.ndimage import gaussian_filter1d
from typing import TypedDict, List, Union
from typing import Optional
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from Pretrain.Dataset import get_PlannerName
from typing import Tuple, Dict
from Pretrain.Transition_Kernel.Kernel_Backbone import count_files_in_folder
import copy
from Pretrain.Rewards.nets import SimpleReward, EnsembleReward
from torch.utils.data import DataLoader
import torch.optim as optim
from Pretrain.Transition_Kernel.Kernel_Net import MoGTransitionKernel, RobustTransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_total_mahalanobis_score, compute_log_density_mog, compute_log_density, compute_total_mahalanobis_score_mog
from Pretrain.Dataset import get_env, get_dataset, Planner_Processor
from gymnasium.vector import AsyncVectorEnv
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Critic.nets import Critic
from Pretrain.Dataset import get_dataset
import json
import torch.nn as nn
import random
import torch.distributed as dist
import wandb




class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray

class Q_Stats:
    Q_mean: float
    Q_std:  float
    def get_Q_stats(self):
        return self.Q_mean, self.Q_std

class Q_Scale:
    Q_scale: float
    def get_Q_scale(self):
        return self.Q_scale


def get_Q_stats(dataset_name: str, specific_dataset: str, task_id: Optional[int] = None, step: int = 0) -> Q_Stats:
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Q_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'rb') as f:
             Q_stats = pickle.load(f)
        return Q_stats

def save_Q_stats(Q_stats: Q_Stats, dataset_name: str, specific_dataset: str, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Q_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(Q_stats, f)
        print(f"saved stats to {savepath}")

def get_Q_scale(dataset_name: str, specific_dataset: str, task_id: Optional[int] = None) -> Q_Scale:
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Q_scale.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'rb') as f:
             Q_scale = pickle.load(f)
        return Q_scale

def save_Q_scale(Q_scale: Q_Scale, dataset_name: str, specific_dataset: str, task_id: Optional[int] = None):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Q_scale.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(Q_scale, f)
        print(f"saved Q_scale to {savepath}")


def build_dit(
    d_s: int,
    d_a: int,
    depth: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    env_name: Optional[str] = None,
) -> DiT1d:
    """Standard planner DiT used everywhere in this repo.
    Fixed: emb_dim=128, d_model=256, n_heads=4, timestep_emb_type='fourier'.
    Only depth (and dims / device) vary by call site.
    """
    in_dim = d_s + d_a
    model = DiT1d(
        in_dim=in_dim,
        emb_dim=128,
        d_model=256,
        n_heads=256 // 64,
        depth=depth,
        timestep_emb_type='fourier',
    )
    if device is not None:
        model = model.to(device)
    return model

def load_dit(
    d_s: int,
    d_a: int,
    state_dict: dict,
    depth: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    env_name: Optional[str] = None,
    eval_mode: bool = True,
) -> DiT1d:
    """Build the standard DiT and load a checkpoint."""
    model = build_dit(d_s, d_a, depth=depth, device=device, env_name=env_name)
    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()
    return model

def divide_trajs(trajs):
    success_trajs = []
    failed_trajs = []
    for traj in trajs:
        if(traj['rewards'][-1] == 1.0):
            success_trajs.append(traj)
        else:
            failed_trajs.append(traj)
    return success_trajs, failed_trajs

def drop_trajs(trajs, percentage):
    success_trajs, failed_trajs = divide_trajs(trajs)
    random.shuffle(failed_trajs)
    failed_trajs = failed_trajs[:int(len(failed_trajs) * percentage)]
    return success_trajs + failed_trajs

def _bootstrap_per_member(s, a, r, ensemble_size, device):
    B = s.shape[0]
    idx = torch.randint(0, B, (ensemble_size, B), device=device)
    return s[idx], a[idx], r[idx]

def check_specific_dataset(dataset_name):
    if(dataset_name in ['kitchen', 'scene']):
         return False
    elif dataset_name in ['pointmaze', 'cube', 'ogpointmaze', 'puzzle', 'antmaze', 'humanoidmaze']:
        return True

def reward_name_converter(specific_dataset):
    cube = {
        "single-play": "single", "single-noisy": "single",
        "double-play": "double", "double-noisy": "double",
        "triple-play": "triple", "triple-noisy": "triple",
        "quadruple-play": "quadruple", "quadruple-noisy": "quadruple",
    }
    puzzle = {
        "3x3-play": "3x3", "3x3-noisy": "3x3",
        "4x4-play": "4x4", "4x4-noisy": "4x4",
        "4x5-play": "4x5", "4x5-noisy": "4x5",
        "4x6-play": "4x6", "4x6-noisy": "4x6",
    }
    scene = {"play": "play", "noisy": "play"}  # shared Scene reward/kernel
    return cube.get(specific_dataset) or puzzle.get(specific_dataset) or scene.get(specific_dataset) or specific_dataset

def reward_processor(rewards, name: str):
    def spare_reward_processor(rewards):
        Temp = []
        for i in range(1, len(rewards)):
             if(rewards[i] == rewards[i-1]+1):
                 Temp.append(i)
        new_rewards = [0]*len(rewards)
        for i in range(len(rewards)):
           if(i in Temp):
                new_rewards[i] = 1.0
           else:
                new_rewards[i] = 0.0
        return np.array(new_rewards, dtype = np.float64)

    def ogbench_reward_processor(rewards):
         if(not isinstance(rewards, np.ndarray)):
              rewards = np.array(rewards)
         Min = np.min(rewards)
         dist = 0 - Min
         rews = rewards + dist
         return rews
    
    if(name in ('cube', 'ogpointmaze', 'antmaze', 'humanoidmaze', 'puzzle', 'scene')):
         return ogbench_reward_processor(rewards)
    else:
         return spare_reward_processor(rewards)

def reward_filter(obs, rews, goal):
    #target_goals = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5], [-2.5, 2.5]])
    for i in range(1, len(obs)):
        pos = obs[i][:2] 
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        #goal_coord = np.asarray(goal_coord, dtype=np.float32).reshape(-1)  
        dist = np.linalg.norm(pos - g) 
        if (dist < 0.5):
            rews[i-1] = 1.0
        else:
            rews[i-1] = 0.0
    return rews

def reward_filter_goals(trajs: List[TrajectoryDict], goal) -> List[TrajectoryDict]:
    def reward_filter2(traj: TrajectoryDict, goal) -> List[TrajectoryDict]:
        last_step = 1
        #i = 1
        new_trajs = []
        new_rews = [0]*len(traj['rewards'])
        traj['rewards'] = new_rews
        
        #while(i < len(traj['observations'])):
        for i in range(1, len(traj['observations'])):
          pos = traj['observations'][i][:2]
          g = np.asarray(goal, dtype=np.float32).reshape(-1)
          dist = np.linalg.norm(pos - g) 
          if(dist < 0.5):
              if((i - last_step) < 3):
                  last_step = i+1
                  continue
              else:
                  rews = traj['rewards'][last_step:i-1]
                  rews[-1] = 1.0
                  new_trajs.append({'observations': traj['observations'][last_step:i-1], 'actions': traj['actions'][last_step:i-1], 'rewards': rews})
                  last_step = i+1
        return new_trajs
    
    new_trajs = []
    for traj in trajs:
        new_trajs.extend(reward_filter2(traj, goal))
    return new_trajs
   
def save_reward_model(reward_net, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    reward_net.eval()
    net_dict = reward_net.state_dict()
    specific_dataset = reward_name_converter(specific_dataset)
    #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
    if(check_specific_dataset(dataset_name)):
          os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
          save_path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{reward_name}_Reward_{str(step)}.pkl'
    else: 
          os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Models/', exist_ok=True)
          save_path = f'./Finetuning/Rewards/{dataset_name}/Models/{reward_name}_Reward_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)

def save_kernel_model(kernel_net, dataset_name, specific_dataset, step, ensemble_idx):
    kernel_net.eval()
    specific_dataset = reward_name_converter(specific_dataset)
    name = getName2(dataset_name, specific_dataset)
    net_dict = kernel_net.state_dict()
    if(check_specific_dataset(dataset_name)):
          os.makedirs(f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}', exist_ok=True)
          save_path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    else: 
          os.makedirs(f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}', exist_ok=True)
          save_path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    torch.save(net_dict, save_path)
    #print(f"Kernel model save to {name}_{str(step)}_{str(ensemble_idx)}.pkl")

def get_reward_model(dataset_name, specific_dataset, step, task_id: Optional[int] = None):
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    specific_dataset = reward_name_converter(specific_dataset)
    #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{reward_name}_Reward_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rewards/{dataset_name}/Models/{reward_name}_Reward_{str(step)}.pkl'
    model_state_dict = torch.load(path, weights_only=True, map_location='cpu')
    return model_state_dict, obs_dim, act_dim

def get_reward_stats(dataset_name, specific_dataset, step, task_id: Optional[int] = None):
    specific_dataset = reward_name_converter(specific_dataset)
    #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/{reward_name}_Reward_stats_{str(step)}.pkl'
        
    else:
        path = f'./Finetuning/Rewards/{dataset_name}/Stats/{reward_name}_Reward_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats  

def get_kernel(dataset_name, specific_dataset, step):
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    specific_dataset = reward_name_converter(specific_dataset)
    name = getName2(dataset_name, specific_dataset)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}'
    else:
        path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}'
    file_count = count_files_in_folder(path)
    kernel_state_dicts = []
    for i in range(file_count):
        if(check_specific_dataset(dataset_name)):
            dir = f"./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}/{name}_Kernel_{str(i)}.pkl"
        else:
            dir = f"./Finetuning/Kernels/{dataset_name}/Models/{str(step)}/{name}_Kernel_{str(i)}.pkl"
        kernel_state_dicts.append(torch.load(dir, weights_only=True, map_location='cpu'))
    return kernel_state_dicts, obs_dim, act_dim

def get_kernel_stats(dataset_name, specific_dataset, step):
    specific_dataset = reward_name_converter(specific_dataset)
    name = getName2(dataset_name, specific_dataset)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Stats/{name}_Kernel_stats_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Kernels/{dataset_name}/Stats/{name}_Kernel_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats  

"""
def save_planner(model, dataset_name, specific_dataset, step: int):
    model.eval()
    data = {
            'dataset_name': dataset_name,
            'specific_dataset': specific_dataset,
            'step': step,
            'ema': model.state_dict()
    }
    name = getName(dataset_name, specific_dataset)
    savepath = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}/{name}_Planner_{str(step)}.pt"
    torch.save(data, savepath)
    print(f"saved model to {savepath}")
"""

def save_planner(model, dataset_name, specific_dataset, step: int,
                 task_id: Optional[int] = None):              # NEW arg
    model.eval()
    data = {
        'dataset_name': dataset_name,
        'specific_dataset': specific_dataset,
        'task_id': task_id,                                   # NEW field
        'step': step,
        'ema': model.state_dict(),
    }
    base = getName(dataset_name, specific_dataset)
    tid  = f"_task{task_id}" if task_id is not None else ""
    fname = f"{base}{tid}_Planner_{step}.pt"
    dir   = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}"
    os.makedirs(dir, exist_ok=True)
    savepath = f"{dir}/{fname}"
    torch.save(data, savepath)
    print(f"saved model to {savepath}")

def get_planner(dataset_name, specific_dataset, step,
                task_id: Optional[int] = None):               # NEW arg
    base = getName(dataset_name, specific_dataset)
    tid  = f"_task{task_id}" if task_id is not None else ""
    path = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}/{base}{tid}_Planner_{step}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, weights_only=True, map_location='cpu')['ema']

"""
def get_planner(dataset_name, specific_dataset, step):
    name = getName(dataset_name, specific_dataset)
    path = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}/{name}_Planner_{str(step)}.pt"
    if not os.path.exists(path):
          raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, weights_only = True,map_location='cpu')
    #checkpoint = torch.load(checkpoint_path,  weights_only=True)
    return checkpoint['ema']
"""

def save_critic(model, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    model.eval()
    critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
    net_dict = model.state_dict()
    os.makedirs(f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
    save_path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{critic_name}_Critic_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)
    print(f"critic model save to {critic_name}_{str(step)}.pkl")

def get_critic_model(dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{critic_name}_Critic_{str(step)}.pkl'
    model_state_dict = torch.load(path, weights_only=True, map_location='cpu')
    return model_state_dict, obs_dim

def get_critic_stats(dataset_name, specific_dataset, task_id: Optional[int] = None,  step: int = 0) -> SAStats:
    critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/{critic_name}_Critic_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats 

def save_trajs(trajs, env_name, specific_env, step, task_id: Optional[int] = None):
    if(task_id is not None):
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{str(task_id)}'
    else:
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/'
    os.makedirs(path, exist_ok=True)
    save_path =  f'{path}/Generated_trajs_Info_{str(step)}.pkl'
    with open(save_path, 'wb') as f:
         pickle.dump(trajs, f)
    print(f"trajectories saved")

def get_trajs(env_name, specific_env, step, task_id: Optional[int] = None):
    if(task_id is not None):
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{str(task_id)}/Generated_trajs_Info_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{str(step)}.pkl'
    with open(path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs

def save_success_trajs_for_reward(trajs, env_name, specific_env, task_id, step):
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/trajs_task{task_id}_success_{step}.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajs, f)
    print("trajectories saved")

def load_success_trajs(env_name, specific_env, task_id, step):
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/trajs_task{task_id}_success_{step}.pkl'
    with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs

class Lambda:
    def __init__(self, lam: float, beta: float, eta_lam: float):
        self.lam = lam
        self.base_lam = lam
        self.beta = beta
        self.eta_lam = eta_lam
    
    def update(self, C):
        self.lam = np.maximum(self.base_lam, self.lam + (self.eta_lam * C))
        self.lam = np.clip(self.lam, 0.0, 5.0)
    
    def set_lam(self, lam: float):
        self.lam = lam

    def get_lam(self):
        return self.lam

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))

def getName(env_name, specific_env):
     if(env_name == 'kitchen'):
        
          return 'Kitchen'
     elif(env_name == 'pointmaze'):
          if specific_env == 'open_dense':
               return 'PointMaze_OpenDense'
          elif specific_env == 'umaze':
               return 'PointMaze_Umaze'
          elif specific_env == 'large_dense':
               return 'PointMaze_LargeDense'
          elif specific_env== 'medium':
               return 'PointMaze_Medium'
          elif specific_env == 'umaze_dense':
               return 'PointMaze_UmazeDense'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env == 'open':
               return 'PointMaze_Open'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")

     elif(env_name == 'antmaze'):
          if specific_env == 'medium':
               return 'AntMaze_Medium'
          elif specific_env == 'large':
               return 'AntMaze_Large'
          elif specific_env == 'giant':
               return 'AntMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     
     elif(env_name == 'cube'):
          if specific_env == 'single-play':
                return 'Cube_SinglePlay'
          elif specific_env == 'single-noisy':
                return 'Cube_SingleNoisy'
          elif specific_env == 'double-play':
                return 'Cube_DoublePlay'
          elif specific_env == 'double-noisy':
                return 'Cube_DoubleNoisy'
          elif specific_env == 'triple-play':
                return 'Cube_TriplePlay'
          elif specific_env == 'triple-noisy':
                return 'Cube_TripleNoisy'
          elif specific_env == 'quadruple-play':
                return 'Cube_QuadruplePlay'
          elif specific_env == 'quadruple-noisy':
                return 'Cube_QuadrupleNoisy'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     
     elif(env_name == 'puzzle'):
          if specific_env == '3x3-play':
                return 'Puzzle_3x3Play'
          elif specific_env == '3x3-noisy':
                return 'Puzzle_3x3Noisy'
          elif specific_env == '4x4-play':
                return 'Puzzle_4x4Play'
          elif specific_env == '4x4-noisy':
                return 'Puzzle_4x4Noisy'
          elif specific_env == '4x5-play':
                return 'Puzzle_4x5Play'
          elif specific_env == '4x5-noisy':
                return 'Puzzle_4x5Noisy'
          elif specific_env == '4x6-play':
                return 'Puzzle_4x6Play'
          elif specific_env == '4x6-noisy':
                return 'Puzzle_4x6Noisy'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'scene'):
          if specific_env == 'play':
                return 'Scene_Play'
          elif specific_env == 'noisy':
                return 'Scene_Noisy'
          else:
                raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
          if specific_env == 'medium':
                return 'OG2DMaze_Medium'
          elif specific_env == 'large':
                return 'OG2DMaze_Large'
          elif specific_env == 'giant':
                return 'OG2DMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
    
     elif(env_name == 'humanoidmaze'):
          if specific_env == 'medium':
                return 'HumanoidMaze_Medium'
          elif specific_env == 'large':
                return 'HumanoidMaze_Large'
          elif specific_env == 'giant':
                return 'HumanoidMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def getName2(env_name, specific_env):
     if(env_name == 'kitchen'):
          return 'Kitchen'

     elif(env_name == 'pointmaze'):
          if specific_env == 'umaze':
               return 'PointMaze_Umaze'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env== 'medium':
               return 'PointMaze_Medium'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")

     elif(env_name == 'antmaze'):
          if specific_env == 'medium':
               return 'AntMaze_Medium'
          elif specific_env == 'large':
               return 'AntMaze_Large'
          elif specific_env == 'giant':
               return 'AntMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'cube'):
          if specific_env == 'single':
                return 'Cube_Single'
          elif specific_env == 'double':
                return 'Cube_Double'
          elif specific_env == 'triple':
                return 'Cube_Triple'
          elif specific_env == 'quadruple':
                return 'Cube_Quadruple'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     
     elif(env_name == 'puzzle'):
          if specific_env == '3x3':
                return 'Puzzle_3x3'
          elif specific_env == '4x4':
                return 'Puzzle_4x4'
          elif specific_env == '4x5':
                return 'Puzzle_4x5'
          elif specific_env == '4x6':
                return 'Puzzle_4x6'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'scene'):
          return 'Scene'

     elif(env_name == 'ogpointmaze'):
          if specific_env == 'medium':
                return 'OG2DMaze_Medium'
          elif specific_env == 'large':
                return 'OG2DMaze_Large'
          elif specific_env == 'giant':
                return 'OG2DMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'humanoidmaze'):
          if specific_env == 'medium':
                return 'HumanoidMaze_Medium'
          elif specific_env == 'large':
                return 'HumanoidMaze_Large'
          elif specific_env == 'giant':
                return 'HumanoidMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def get_CriticName(env_name, specific_env, task_id: Optional[int] = None):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     
     elif(env_name == 'pointmaze'):
         if(specific_env == 'large'):
              return 'PointMaze_Large'
         elif(specific_env == 'medium'):
              return 'PointMaze_Medium'
         elif(specific_env == 'unmaze'):
              return 'PointMaze_Unmaze'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")

     elif(env_name == 'cube'):
         if specific_env == 'single-play':
              return f'Cube_SinglePlay_task{task_id}'
         elif specific_env == 'single-noisy':
             return f'Cube_SingleNoisy_task{task_id}'
         elif specific_env == 'double-play':
             return f'Cube_DoublePlay_task{task_id}'
         elif specific_env == 'double-noisy':
             return f'Cube_DoubleNoisy_task{task_id}'
         elif specific_env == 'triple-play':
             return f'Cube_TriplePlay_task{task_id}'
         elif specific_env == 'triple-noisy':
             return f'Cube_TripleNoisy_task{task_id}'
         elif specific_env == 'quadruple-play':
             return f'Cube_QuadruplePlay_task{task_id}'
         elif specific_env == 'quadruple-noisy':
             return f'Cube_QuadrupleNoisy_task{task_id}'
         else:
             raise ValueError(f"Invalid cube dataset name: {specific_env}")
     
     elif(env_name == 'puzzle'):
          if specific_env == '3x3-play':
                return f'Puzzle_3x3Play_task{task_id}'
          elif specific_env == '3x3-noisy':
                return f'Puzzle_3x3Noisy_task{task_id}'
          elif specific_env == '4x4-play':
                return f'Puzzle_4x4Play_task{task_id}'
          elif specific_env == '4x4-noisy':
                return f'Puzzle_4x4Noisy_task{task_id}'
          elif specific_env == '4x5-play':
                return f'Puzzle_4x5Play_task{task_id}'
          elif specific_env == '4x5-noisy':
                return f'Puzzle_4x5Noisy_task{task_id}'
          elif specific_env == '4x6-play':
                return f'Puzzle_4x6Play_task{task_id}'
          elif specific_env == '4x6-noisy':
                return f'Puzzle_4x6Noisy_task{task_id}'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'scene'):
         if specific_env == 'play':
             return f'Scene_Play_task{task_id}'
         elif specific_env == 'noisy':
             return f'Scene_Noisy_task{task_id}'
         else:
             raise ValueError(f"Invalid scene dataset name: {specific_env}")
    
     elif(env_name == 'antmaze'):
          if(task_id is None):
               raise ValueError('Task ID is required for antmaze dataset')
          elif specific_env == 'medium':
               return f'AntMaze_Medium_task{task_id}'
          elif specific_env == 'large':
               return f'AntMaze_Large_task{task_id}'
          elif specific_env == 'giant':
               return f'AntMaze_Giant_task{task_id}'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     
     elif(env_name == 'humanoidmaze'):
          if(task_id is None):
               raise ValueError('Task ID is required for humanoidmaze dataset')
          elif specific_env == 'medium':
               return f'HumanoidMaze_Medium_task{task_id}'
          elif specific_env == 'large':
               return f'HumanoidMaze_Large_task{task_id}'
          elif specific_env == 'giant':
               return f'HumanoidMaze_Giant_task{task_id}'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
         if(task_id is None):
              raise ValueError('Task ID is required for ogpointmaze dataset')
         if(specific_env == 'medium'):
              return f'OG2DMaze_Medium_task{task_id}'
         elif(specific_env == 'large'):
              return f'OG2DMaze_Large_task{task_id}'
         elif(specific_env == 'giant'):
              return f'OG2DMaze_Giant_task{task_id}'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def get_RewardName(env_name, specific_env, task_id: Optional[int] = None):
     if(env_name == 'kitchen'):
          return 'Kitchen'
     elif(env_name == 'pointmaze'):
          if specific_env == 'open_dense':
               return 'PointMaze_OpenDense'
          elif specific_env == 'umaze':
               return 'PointMaze_Umaze'
          elif specific_env == 'large_dense':
               return 'PointMaze_LargeDense'
          elif specific_env== 'medium':
               return 'PointMaze_Medium'
          elif specific_env == 'medium_dense':
               return 'PointMaze_MediumDense'
          elif specific_env == 'umaze_dense':
               return 'PointMaze_UmazeDense'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env == 'open':
               return 'PointMaze_Open'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     
     elif(env_name == 'antmaze'):
          if(task_id is None):
               raise ValueError('Task ID is required for antmaze dataset')
          elif specific_env == 'medium':
               return f'AntMaze_Medium_Task{task_id}'
          elif specific_env == 'large':
               return f'AntMaze_Large_Task{task_id}'
          elif specific_env == 'giant':
               return f'AntMaze_Giant_Task{task_id}'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     
     elif(env_name == 'humanoidmaze'):
          if(task_id is None):
               raise ValueError('Task ID is required for humanoidmaze dataset')
          elif specific_env == 'medium':
               return f'HumanoidMaze_Medium_Task{task_id}'
          elif specific_env == 'large':
               return f'HumanoidMaze_Large_Task{task_id}'
          elif specific_env == 'giant':
               return f'HumanoidMaze_Giant_Task{task_id}'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'cube'):
         if(task_id is None):
            raise ValueError('Task ID is required for cube dataset')
         if specific_env == 'single' or specific_env == 'single-play':
              return f'Cube_Single_Task{task_id}'
         elif specific_env == 'double'  or specific_env == 'double-play':
              return f'Cube_Double_Task{task_id}'
         elif specific_env == 'triple' or specific_env == 'triple-play':
              return f'Cube_Triple_Task{task_id}'
         elif specific_env == 'quadruple' or specific_env == 'quadruple-play':
              return f'Cube_Quadruple_Task{task_id}'
         else:
              raise ValueError(f"Invalid dataset name: {specific_env}")
     
     elif(env_name == 'puzzle'):
         if(task_id is None):
            raise ValueError('Task ID is required for puzzle dataset')
         if specific_env == '3x3' or specific_env == '3x3-play':
              return f'Puzzle_3x3_Task{task_id}'
         elif specific_env == '4x4'  or specific_env == '4x4-play':
              return f'Puzzle_4x4_Task{task_id}'
         elif specific_env == '4x5' or specific_env == '4x5-play':
              return f'Puzzle_4x5_Task{task_id}'
         elif specific_env == '4x6' or specific_env == '4x6-play':
              return f'Puzzle_4x6_Task{task_id}'
         else:
              raise ValueError(f"Invalid dataset name: {specific_env}")

     elif(env_name == 'scene'):
          return f"Scene_Task{task_id}"

     elif(env_name == 'ogpointmaze'):
         if(task_id is None):
            raise ValueError('Task ID is required for ogpointmaze dataset')
         if(specific_env == 'medium'):
              return f'OG2DMaze_Medium_Task{task_id}'
         elif(specific_env == 'large'):
              return f'OG2DMaze_Large_Task{task_id}'
         elif(specific_env == 'giant'):
              return f'OG2DMaze_Giant_Task{task_id}'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

class KernelDataset(Dataset):
    def __init__(self, trajectories: List[TrajectoryDict], dataset_name: str, specific_dataset: str, step: int):
         obs_list, act_list = [], []
        
         for traj in trajectories:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
         obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]
         #act_all = np.concatenate(act_list, axis=0)  # [N, d_a]
        
        #get stats
         self.stats = SAStats()
         self.stats.obs_mean = obs_all.mean(axis=0)
         self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
         data = []
         for traj in trajectories:
            obs = traj['observations']
            acts = traj['actions']
            for t in range(len(obs)-1):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                data.append((s_t, a_t, s_tp1))
         self.data = data
         self.save_stats(dataset_name, specific_dataset, step)
    
    def save_stats(self, dataset_name, specific_dataset, step):
        specific_dataset = reward_name_converter(specific_dataset)
        name = getName2(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Kernel_stats_{str(step)}.pkl'
        if(check_specific_dataset(dataset_name)):
             stats_dir = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Stats/'
        else:
             stats_dir = f'./Finetuning/Kernels/{dataset_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, s_next = self.data[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32)
        )

class RewardDataset(Dataset):
    def __init__(self, trajs: List[TrajectoryDict], sigma: float, dataset_name: str, specific_dataset: str, step: int, goal: Optional[np.array] = None, target_reward: Optional[float] = None, task_id: Optional[int] = None):
            
        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        
        for traj in trajs:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]
        #act_all = np.concatenate(act_list, axis=0)  # [N, d_a]
        
        
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            acts = traj['actions']
            rews = traj['rewards']
            """
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            """
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            rews = gaussian_filter1d(rews, sigma)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, step)
    
    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = 0, step = 0):
        specific_dataset = reward_name_converter(specific_dataset)
        #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
        reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
        stats_name =  str(reward_name) + f'_Reward_stats_{str(step)}.pkl'
        if(check_specific_dataset(dataset_name)):
            stats_dir = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/'
        else:
            stats_dir = f'./Finetuning/Rewards/{dataset_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )
    
    def boost_signal(self, target_reward, rews):
         rews = np.asarray(rews, dtype=np.float64).copy()
         rews = rews * target_reward
         return rews

def train_reward(trajs: List[TrajectoryDict], 
                 dataset_name: str,
                 hidden_layers: int, 
                 hidden_dim: int, 
                 batch_size, 
                 num_steps, 
                 lr, min_lr, sigma, 
                 step, 
                 target_reward: Optional[float] = None, 
                 specific_dataset: Optional[str] = None, 
                 goal: Optional[np.array] = None, 
                 task_id: Optional[int] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    print(f"Training reward approximator for {dataset_name}_{specific_dataset} Dataset") 
    dataset = RewardDataset(trajs, sigma, dataset_name, specific_dataset, step, goal, target_reward, task_id)
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8))
    reward_net = SimpleReward(obs_dim, act_dim, hidden_dim, hidden_layers).to(device)
    optimizer = optim.AdamW(reward_net.parameters(), lr = lr, weight_decay = 1e-4)
    total_loss = 0
    counter = 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = num_steps,   # one scheduler step per training step
            eta_min = min_lr
        )
    for i in range(num_steps):
           s, a, r = next(dataloader)
           s = s.to(device)
           a = a.to(device)
           r = r.to(device)
        
           # Predicted Reward
           optimizer.zero_grad()
           pred = reward_net(s, a)
           #loss = F.mse_loss(pred, r)
           loss = F.smooth_l1_loss(pred, r, beta = 1)
           loss.backward()
           #torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm = 1.0)
           optimizer.step()
           scheduler.step()
           total_loss += loss.item()
           counter += 1
    save_reward_model(reward_net, dataset_name, specific_dataset, task_id, step)
    print(f"reward model saved")

def train_reward_ensemble(
    trajs: List[TrajectoryDict], 
    dataset_name: str,
    hidden_layers: int,
    hidden_dim: int,
    batch_size: int,
    num_steps: int,
    lr: float,
    min_lr: float,
    ensemble_size: int = 5,
    bootstrap: bool = True,
    save_percentage: float = 0.0,
    sigma: Optional[float] = None,
    step: int = 0,
    target_reward: Optional[float] = None,
    specific_dataset: Optional[str] = None,
    goal: Optional[np.ndarray] = None,
    task_id: Optional[int] = None,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
):  
   
    device = check_device()
    trajs = drop_trajs(trajs, save_percentage)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    dataset = RewardDataset(trajs, sigma, dataset_name, specific_dataset, step, goal, target_reward, task_id)
    dataloader = cycle(DataLoader(
        dataset, batch_size = batch_size, shuffle = True,
        pin_memory = True, num_workers = 8,
    ))
    # --- build model + optim
    reward_net = EnsembleReward(
        obs_dim, act_dim, hidden_dim, hidden_layers,
        ensemble_size=ensemble_size,
    ).to(device)
    optimizer = optim.AdamW(
        reward_net.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )
    running_loss = 0.0
    for step in range(1, num_steps + 1):
        s, a, r = next(dataloader)
        s = s.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        if bootstrap and ensemble_size > 1:
            s_e, a_e, r_e = _bootstrap_per_member(s, a, r, ensemble_size, device)
        else:
            # diversity from random init only
            s_e = s.unsqueeze(0).expand(ensemble_size, -1, -1)
            a_e = a.unsqueeze(0).expand(ensemble_size, -1, -1)
            r_e = r.unsqueeze(0).expand(ensemble_size, -1)
        optimizer.zero_grad()
        pred_e = reward_net(s_e, a_e)                     # (E, B)
        # mean over (E*B) ≡ mean of per-member SmoothL1 losses
        """
        per_elem = F.smooth_l1_loss(pred_e, r_e, beta=1.0, reduction='none')
        positive_weight = 50.0                       # try 8.0 ~ 30.0
        weights = torch.where(r_e > 0, positive_weight, 1.0)
        loss = (weights * per_elem).mean()
        """
        loss = F.smooth_l1_loss(pred_e, r_e, beta = 1.0)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(reward_net.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    save_reward_model(reward_net, dataset_name, specific_dataset, task_id, step)
    print(f"reward model saved")

def train_kernel(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    batch_size=256,
    lr=1e-3,
    num_steps=10000,
    ensemble_size=10,
    λ_reg=1e-3,
    num_hidden_layers=2,
    hidden_dim=256,
    step: int = 0,
    constraint_type: str = "mahalanobis",
    quantile: float = 0.95,
    x_generated_plans: Optional[list] = None,
    accelerator=None,
):
   
    if accelerator is not None and accelerator.is_main_process:
          print(f"Training kernel for {dataset_name}_{specific_dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # distributed role info
    if accelerator is None:
        is_main, rank, world = True, 0, 1
    else:
        is_main = accelerator.is_main_process
        rank = accelerator.process_index
        world = accelerator.num_processes

    # normalize constraint string
    ctype = "log_prob" if constraint_type in ("log_prob", "log_density") else "mahalanobis"

    # -----------------------------
    # Phase A: train on main only
    # -----------------------------
    ensemble = None
    if is_main:
        dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
        loader = cycle(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=8,
            )
        )
        ensemble = [
            RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers, hidden_dim).to(device)
            for _ in range(ensemble_size)
        ]
        optimizers = [optim.Adam(m.parameters(), lr, weight_decay=1e-5) for m in ensemble]

        for _ in range(1, num_steps + 1):
            s, a, s_next = next(loader)
            s, a, s_next = s.to(device), a.to(device), s_next.to(device)

            losses, mus, log_stds = [], [], []
            for m in ensemble:
                mu, log_std = m(s, a)
                mus.append(mu)
                log_stds.append(log_std)
                losses.append(m.gaussian_nll(s_next, mu, log_std))

            mus_stack = torch.stack(mus, dim=0)
            mu_mean = mus_stack.mean(dim=0)
            disagreement = ((mus_stack - mu_mean.unsqueeze(0)) ** 2).mean(dim=0).detach()

            for i, m in enumerate(ensemble):
                penalty = (disagreement / (torch.exp(2 * log_stds[i]) + m.noise_floor)).sum(dim=-1).mean()
                losses[i] = losses[i] + λ_reg * penalty

            for i, (m, opt) in enumerate(zip(ensemble, optimizers)):
                opt.zero_grad()
                losses[i].backward()
                opt.step()

        # save trained kernels for all ranks to load
        for idx, m in enumerate(ensemble):
            save_kernel_model(copy.deepcopy(m).cpu(), dataset_name, specific_dataset, step, idx)
        print("Kernel model saved")

    if accelerator is not None:
        accelerator.wait_for_everyone()

    # ----------------------------------------
    # Phase B: threshold by all GPUs in parallel
    # ----------------------------------------
    threshold = None
    if x_generated_plans is not None:
        # every rank loads saved kernels
        kernel_state_dicts, _, _ = get_kernel(dataset_name, specific_dataset, step)
        eval_ensemble = [
            RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers, hidden_dim).to(device)
            for _ in range(len(kernel_state_dicts))
        ]
        for m, sd in zip(eval_ensemble, kernel_state_dicts):
            m.load_state_dict(sd)
            m.eval()

        kernel_stats = get_kernel_stats(dataset_name, specific_dataset, step)

        # shard plans across ranks
        local_plans = x_generated_plans[rank::world]
        local_values = []
        for x in local_plans:
            for j in range(1, len(x) - 1):
                obs = torch.tensor(kernel_stats.norm_obs(x[j, :obs_dim].copy()), dtype=torch.float32).unsqueeze(0).to(device)
                act = torch.tensor(x[j, obs_dim:obs_dim + act_dim].copy(), dtype=torch.float32).unsqueeze(0).to(device)
                s_next = torch.tensor(kernel_stats.norm_obs(x[j + 1, :obs_dim].copy()), dtype=torch.float32).unsqueeze(0).to(device)

                if ctype == "log_prob":
                    v = compute_log_density(eval_ensemble, obs, act, s_next).item()
                else:
                    v = compute_total_mahalanobis_score(eval_ensemble, obs, act, s_next).item()
                local_values.append(v)

        # gather local values from all ranks
        if accelerator is not None:
            gathered = accelerator.gather_for_metrics([local_values], use_gather_object=True)
        else:
            gathered = [local_values]

        if is_main:
            values = []
            for chunk in gathered:
                values.extend(chunk)
            if ctype == "log_prob":
                threshold = float(np.quantile(values, 1 - quantile))
            else:
                threshold = float(np.quantile(values, quantile))
            print(f"New Threshold for {ctype}: {threshold}")
        else:
            threshold = 0.0

        # broadcast scalar threshold
        if accelerator is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([threshold], device=device, dtype=torch.float32)
            torch.distributed.broadcast(t, src=0)
            threshold = float(t.item())
    
    if accelerator is not None:
        accelerator.wait_for_everyone()
        
    return threshold

def train_kernel_mog(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    batch_size=256,
    lr=1e-3,
    num_steps=10000,
    ensemble_size=10,
    λ_reg=1e-3,
    num_modes: Optional[int] = 8,
    num_hidden_layers=2,
    hidden_dim=256,
    kernel_noise_floor: Optional[float] = 1e-4,
    step: int = 0,
    constraint_type: str = "mahalanobis",
    quantile: float = 0.95,
    x_generated_plans: Optional[List] = None,
    accelerator=None,
):   
    if accelerator is not None and accelerator.is_main_process:
          print(f"Training kernel for {dataset_name}_{specific_dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    if accelerator is None:
        is_main, rank, world = True, 0, 1
    else:
        is_main = accelerator.is_main_process
        rank = accelerator.process_index
        world = accelerator.num_processes

    ctype = "log_prob" if constraint_type in ("log_prob", "log_density") else "mahalanobis"

    # -----------------------------
    # Phase A: train on main only
    # -----------------------------
    if is_main:
        dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
        loader = cycle(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=8,
                persistent_workers=True,
                prefetch_factor=4,
                drop_last=True,
            )
        )

        ensemble = [
            MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, kernel_noise_floor).to(device)
            for _ in range(ensemble_size)
        ]
        optimizers = [optim.Adam(m.parameters(), lr, weight_decay=1e-5) for m in ensemble]

        for _ in range(1, num_steps + 1):
            s, a, s_next = next(loader)
            s, a, s_next = s.to(device), a.to(device), s_next.to(device)

            losses = []
            for m in ensemble:
                mu, log_std, weights = m(s, a)
                loss = m.mog_nll(s_next, mu, log_std, weights)

                mu_mean = mu.mean(dim=1)
                disagreement = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1).mean(dim=0)
                var = torch.exp(2 * log_std) + m.noise_floor
                penalty = (disagreement / (var.mean(dim=1) + 1e-6)).mean()
                losses.append(loss + λ_reg * penalty)

            for m, opt, loss in zip(ensemble, optimizers, losses):
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=5.0)
                opt.step()

        for idx, m in enumerate(ensemble):
            save_kernel_model(copy.deepcopy(m).cpu(), dataset_name, specific_dataset, step, idx)
        print("Kernel model saved")

    if accelerator is not None:
        accelerator.wait_for_everyone()

    # ----------------------------------------
    # Phase B: threshold by all GPUs in parallel
    # ----------------------------------------
    threshold = None
    if x_generated_plans is not None:
        kernel_state_dicts, _, _ = get_kernel(dataset_name, specific_dataset, step)
        eval_ensemble = [
            MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, kernel_noise_floor).to(device)
            for _ in range(len(kernel_state_dicts))
        ]
        for m, sd in zip(eval_ensemble, kernel_state_dicts):
            m.load_state_dict(sd)
            m.eval()

        kernel_stats = get_kernel_stats(dataset_name, specific_dataset, step)

        local_plans = x_generated_plans[rank::world]
        local_values = []
        for x in local_plans:
            for j in range(1, len(x) - 1):
                obs = torch.tensor(kernel_stats.norm_obs(x[j, :obs_dim].copy()), dtype=torch.float32).unsqueeze(0).to(device)
                act = torch.tensor(x[j, obs_dim:obs_dim + act_dim].copy(), dtype=torch.float32).unsqueeze(0).to(device)
                s_next = torch.tensor(kernel_stats.norm_obs(x[j + 1, :obs_dim].copy()), dtype=torch.float32).unsqueeze(0).to(device)

                if ctype == "log_prob":
                    v = compute_log_density_mog(eval_ensemble, obs, act, s_next).item()
                else:
                    v = compute_total_mahalanobis_score_mog(eval_ensemble, obs, act, s_next).item()
                local_values.append(v)

        if accelerator is not None:
            gathered = accelerator.gather_for_metrics([local_values], use_gather_object=True)
        else:
            gathered = [local_values]

        if is_main:
            values = []
            for chunk in gathered:
                values.extend(chunk)
            if ctype == "log_prob":
                threshold = float(np.quantile(values, 1 - quantile))
            else:
                threshold = float(np.quantile(values, quantile))
            print(f"New Threshold for {ctype}: {threshold}")
        else:
            threshold = 0.0

        if accelerator is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([threshold], device=device, dtype=torch.float32)
            torch.distributed.broadcast(t, src=0)
            threshold = float(t.item())
    
    if accelerator is not None:
        accelerator.wait_for_everyone()

    return threshold

def compute_threshold_mog(kernels, kernel_stats, obs_dim, act_dim, x, constraint_type: str = 'log_prob', quantile: float = 0.999, device: str = 'cuda'):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    values = []
    for i in range(len(x)):
       for j in range(1, len(x[i])-1):
           obs = torch.tensor(kernel_stats.norm_obs(x[i][j, :obs_dim].copy()), dtype = torch.float32).unsqueeze(0).to(device)
           act = torch.tensor(x[i][j, obs_dim:(obs_dim+act_dim)].copy(), dtype = torch.float32).unsqueeze(0).to(device)
           s_next = torch.tensor(kernel_stats.norm_obs(x[i][j+1, :obs_dim].copy()), dtype = torch.float32).unsqueeze(0).to(device)
           if(constraint_type == 'log_prob'):
               value = compute_log_density_mog(kernels, obs, act, s_next).item()
           else:
               value = compute_total_mahalanobis_score_mog(kernels, obs, act, s_next).item()
           values.append(value)
    if(constraint_type == 'log_prob'):
         threshold = np.quantile(values, (1 - quantile))
    elif(constraint_type == 'mahalanobis'):
         threshold = np.quantile(values, quantile)
    else:
         raise ValueError(f"Invalid constraint type: {constraint_type}")
    return threshold

def compute_threshold(kernels, kernel_stats, obs_dim, act_dim, x, constraint_type: str = 'log_prob', quantile: float = 0.999, device: str = 'cuda'):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    values = []
    for i in range(len(x)):
       for j in range(1, len(x[i])-1):
           obs = torch.tensor(kernel_stats.norm_obs(x[i][j, :obs_dim].copy()), dtype = torch.float32).unsqueeze(0).to(device)
           act = torch.tensor(x[i][j, obs_dim:(obs_dim+act_dim)].copy(), dtype = torch.float32).unsqueeze(0).to(device)
           s_next = torch.tensor(kernel_stats.norm_obs(x[i][j+1, :obs_dim].copy()), dtype = torch.float32).unsqueeze(0).to(device)
           if(constraint_type == 'log_prob'):
                value = compute_log_density(kernels, obs, act, s_next).item()
           else:
                value = compute_total_mahalanobis_score(kernels, obs, act, s_next).item()
           values.append(value)
    if(constraint_type == 'log_prob'):
         threshold = np.quantile(values, (1 - quantile))
    elif(constraint_type == 'mahalanobis'):
         threshold = np.quantile(values, quantile)
    else:
         raise ValueError(f"Invalid constraint type: {constraint_type}")
    return threshold
     
def check_Critic(dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = f"./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl"
    if(os.path.exists(path)):
        return True
    else:
        return False
         
def update_critic_stats(dataset_name, specific_dataset, new_stats: SAStats, task_id: Optional[int] = None, old_step: int = 0, momentum: float = 0.005) -> SAStats:
    old_stats = get_critic_stats(dataset_name, specific_dataset, task_id = task_id, step = old_step)
    stats = SAStats()
    stats.obs_mean = ((1 - momentum) * old_stats.obs_mean) + (momentum * new_stats.obs_mean)
    stats.obs_std = ((1 - momentum) * old_stats.obs_std) + (momentum * new_stats.obs_std)
    return stats

def get_new_critic_stats(trajs: List[TrajectoryDict]) -> SAStats:
    obs_all = []
    for traj in trajs:
        obs_all.append(traj['observations'])
    obs_all = np.concatenate(obs_all, axis = 0)
    stats = SAStats()
    stats.obs_mean = obs_all.mean(axis=0)
    stats.obs_std = obs_all.std(axis=0)+ 1e-8
    return stats

class Critic_Buffer():
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       trajs:  List[TrajectoryDict],
                       sigma: float,
                       target_reward: Optional[float] = None, 
                       horizon: int = 32,
                       gamma: float = 0.99,
                       lam: float = 0.95,
                       task_id: Optional[int] = None,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005):
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.data = CriticDataset(dataset_name,
                                  specific_dataset, 
                                  trajs, 
                                  sigma,
                                  target_reward,
                                  horizon,
                                  task_id, 
                                  old_step,  
                                  new_step, 
                                  momentum)
       
     
    def obtain_training_data(self, target_critic: nn.Module, batch_size: int, device: str):
        loader = cycle(DataLoader(
            self.data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ))
        obs_chunks, rews_chunks = next(loader)      # (B, T, dim), (B, T)
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]

        with torch.no_grad():
            values = target_critic(obs_chunks)            # (B, T)

            deltas = (
                  rews_chunks[:, :-1]
                  + self.gamma * values[:, 1:]
                   - values[:, :-1]
              )                                             # (B, T-1)

            advantages = torch.zeros(B, T - 1, device=device)
            last_adv = torch.zeros(B, device=device)
            for t in reversed(range(T - 1)):
                last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                advantages[:, t] = last_adv

            value_targets = values[:, 0] + advantages[:, 0]   # (B,)

        return obs_chunks[:, 0], value_targets

class CriticDataset(Dataset):
    def __init__(self, dataset_name: str, 
                       specific_dataset: str, 
                       trajs: List[TrajectoryDict], 
                       sigma: float,
                       target_reward: Optional[float] = None,
                       horizon: int = 32,
                       task_id: Optional[int] = None, 
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005):
        # ----- gather raw obs/actions to fit stats -----
        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0)+ 1e-8
        if(old_step is not None):
             self.stats = update_critic_stats(dataset_name, specific_dataset, stats, task_id, old_step, momentum)
        else:
             self.stats = stats
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            rews = traj['rewards']
            """
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            """
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            if(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            if len(obs) < horizon:
                continue 
            for t in range(len(obs) - horizon):
                 obs_chunk = self.stats.norm_obs(obs[t : t + horizon]).astype(np.float32)
                 rews_chunk = rews[t: min(t+horizon, len(rews))]
                 transitions.append((obs_chunk, rews_chunk))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, new_step)
    
    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __getitem__(self, idx):
        obs_chunk, rews_chunk = self.transitions[idx]
        return (
            torch.tensor(obs_chunk, dtype = torch.float32),
            torch.tensor(rews_chunk, dtype = torch.float32)
        )
    def __len__(self):
        return len(self.transitions)

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

def train_critic(trajs: List[TrajectoryDict], 
                 dataset_name: str, 
                 specific_dataset: str, 
                 hidden_layers: int, 
                 hidden_dim: int, 
                 sigma: float, 
                 batch_size, 
                 num_steps, 
                 gamma, lam, horizon, 
                 lr, 
                 min_lr, 
                 tau, 
                 old_step: Optional[int] = None, 
                 new_step: int = 0, 
                 momentum: float = 0.005, 
                 target_reward = 1.0, 
                 task_id: Optional[int] = None):
    device = check_device()
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    if(old_step is not None):
        critic_state_dict, _ = get_critic_model(dataset_name, specific_dataset, task_id = task_id, step = old_step)
        critic.load_state_dict(critic_state_dict)
    target_critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    optimizer = optim.AdamW(critic.parameters(), lr = lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = num_steps,   # one scheduler step per training step
            eta_min = min_lr
        )
    critic.train()
    buffer = Critic_Buffer(
            dataset_name=dataset_name,
            specific_dataset=specific_dataset,
            trajs=trajs,
            sigma=sigma,
            target_reward=target_reward,
            horizon=horizon,
            gamma=gamma,
            lam=lam,
            task_id=task_id,
            old_step=old_step,
            new_step=new_step,
            momentum=momentum)
    print(f"Training critic for {dataset_name}-{specific_dataset}")
    total_loss = 0.0
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, target_value = buffer.obtain_training_data(target_critic, batch_size, device)
           s = s.to(device)
           target_value = target_value.to(device)

           # Predicted Q-values
           q_pred = critic(s)
           loss = F.smooth_l1_loss(q_pred, target_value, beta = 1.0)
           #loss = F.mse_loss(q_pred, target_value)
           total_loss += loss.item()

           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
           
           if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/200}")
                total_loss = 0.0
            
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")

"""
class Critic_Test_Dataset(Dataset):
    def __init__(self, 
                 dataset_name: str, 
                 specific_dataset: str, 
                 checkpoint_step: int,
                 trajs: List[TrajectoryDict],
                 sigma: Optional[float] = None,
                 task_id: Optional[int] = None,
                 target_reward: Optional[float] = None,
                 horizon: int = 32,
                 gamma: float = 0.99):
        
        self.stats = get_critic_stats(dataset_name, specific_dataset, task_id, checkpoint_step)
        self.horizon = horizon
        self.gamma = gamma

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards'].copy()
             
            
            if target_reward is not None:
                rews = self.boost_signal(target_reward, rews)
            if sigma is not None:
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate=200/sigma)

            for t in range(len(obs) - horizon):        # consistent with training
                obs_t = self.stats.norm_obs(obs[t])
                rews_chunk = rews[t : t + horizon]
                transitions.append((obs_t, rews_chunk))

        self.transitions = transitions
        print(f"Test dataset created: {len(self.transitions)} samples (horizon={horizon})")

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        obs_t, rews_chunk = self.transitions[idx]
        return (
            torch.tensor(obs_t, dtype=torch.float32),
            torch.tensor(rews_chunk, dtype=torch.float32)
        )

def test_critic(dataset_name: str,
                specific_dataset: str,
                finetune: bool,
                hidden_layers: int,
                hidden_dim: int,
                checkpoint_step: int,
                gamma: float = 0.99,
                horizon: int = 32,
                sigma: Optional[float] = None,
                target_reward: float = 1.0,
                trajs: List[TrajectoryDict] = None,
                task_id: Optional[int] = None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if(finetune):
        dataset = Critic_Test_Dataset(
           dataset_name, specific_dataset, 0, trajs,
           sigma, task_id, target_reward, horizon, gamma
        )
    else:
        dataset = Critic_Test_Dataset(
           dataset_name, specific_dataset, checkpoint_step, trajs,
           sigma, task_id, target_reward, horizon, gamma
        )


    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)

    # Load model
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, checkpoint_step)
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    print(f"Testing critic at checkpoint {checkpoint_step} (consistent with training)...")

    with torch.no_grad():
        for s, rews_chunk in dataloader:
            s = s.to(device)
            rews_chunk = rews_chunk.to(device)          # (B, horizon)

            pred = model(s)                             # V(s) - shape (B, 1) or (B,)

            if pred.dim() == 2:
                pred = pred.squeeze(1)

            # Compute same style target as training: n-step return
            target = torch.zeros_like(pred)
            for i in range(rews_chunk.shape[1]):
                target += (gamma ** i) * rews_chunk[:, i]

            loss = F.smooth_l1_loss(pred, target, beta=1.0)
            total_loss += loss.item() * s.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

    print(f"Test Results (Checkpoint {checkpoint_step}):")
    print(f"   Smooth L1 Loss : {avg_loss:.4f}")
    print(f"   MAE            : {mae:.4f}")
    print(f"   Mean Pred      : {np.mean(all_preds):.3f}")
    print(f"   Mean Target    : {np.mean(all_targets):.3f}")
    print(f"   Pred Std       : {np.std(all_preds):.3f}")

    return avg_loss, mae
"""

class Critic_Test_Dataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 specific_dataset: str,
                 checkpoint_step: int,
                 trajs: List[TrajectoryDict],
                 sigma: Optional[float] = None,
                 task_id: Optional[int] = None,
                 target_reward: Optional[float] = None,
                 horizon: int = 32,
                 gamma: float = 0.99):
        self.stats = get_critic_stats(dataset_name, specific_dataset, task_id, checkpoint_step)
        self.horizon = horizon
        self.gamma = gamma

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards'].copy()

            if target_reward is not None:
                rews = self.boost_signal(target_reward, rews)
            if sigma is not None:
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate=200/sigma)

            for t in range(len(obs) - horizon):
                obs_t = self.stats.norm_obs(obs[t])
                rews_chunk = rews[t : t + horizon]
                transitions.append((obs_t, rews_chunk))

        self.transitions = transitions
        print(f"Test dataset created: {len(self.transitions)} samples")

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        obs_t, rews_chunk = self.transitions[idx]
        return (
            torch.tensor(obs_t, dtype=torch.float32),
            torch.tensor(rews_chunk, dtype=torch.float32)
        )

def test_critic(dataset_name: str,
                specific_dataset: str,
                hidden_layers: int,
                hidden_dim: int,
                checkpoint_step: int,
                critic_checkpoint: int,
                gamma: float = 0.99,
                horizon: int = 32,
                value_scale: float = 5.0,
                sigma: Optional[float] = None,
                target_reward: float = 10.0,      # ← must match reward model
                trajs: List[TrajectoryDict] = None,
                task_id: Optional[int] = None):
    device = check_device()
    
    NS = 0 if critic_checkpoint == -1 else critic_checkpoint
    dataset = Critic_Test_Dataset(
        dataset_name, specific_dataset, NS, trajs,
        sigma, task_id, target_reward, horizon, gamma
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)

    # Load model
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, critic_checkpoint)
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    """
    if(mean is not None and std is not None):
         mean_pred = torch.tensor(mean, device = device, dtype = torch.float32)
         std_pred = torch.tensor(std, device = device, dtype = torch.float32)
    """
    print(f"Testing critic at checkpoint {checkpoint_step}...")

    with torch.no_grad():
        for s, rews_chunk in dataloader:               # s: (B,), rews_chunk: (B, horizon)
            s = s.to(device)
            rews_chunk = rews_chunk.to(device)

            pred = model(s).squeeze(-1)                # (B,)  ← normalized V(s)
            pred = value_scale * pred
            """
            if(mean is not None and std is not None):
                pred = (pred * std_pred) + mean_pred
            """
            
            
            # Compute raw n-step return
            gamma_pow = torch.tensor([gamma ** i for i in range(horizon)], device=device, dtype=torch.float32)
            raw_target = (gamma_pow.unsqueeze(0) * rews_chunk).sum(dim=1)
            
            
            """
            # === Normalize target (CRITICAL) ===
            tgt_mean = raw_target.mean()
            tgt_std = raw_target.std(unbiased=False) + 1e-8
            target = (raw_target - tgt_mean) / tgt_std
            """
        
            
            #loss = F.smooth_l1_loss(pred, target, beta=1.0)
            loss = F.smooth_l1_loss(pred, raw_target, beta=1.0)
            total_loss += loss.item() * s.size(0)

            all_preds.extend(pred.cpu().numpy())
            #all_targets.extend(target.cpu().numpy())
            all_targets.extend(raw_target.cpu().numpy())


    avg_loss = total_loss / len(dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

    print(f"Test Results (Checkpoint {checkpoint_step}):")
    print(f"  Smooth L1 Loss : {avg_loss:.4f}")
    print(f"  MAE            : {mae:.4f}")
    print(f"  Mean Pred      : {np.mean(all_preds):.3f}")
    print(f"  Mean Target    : {np.mean(all_targets):.3f}")
    print(f"  Pred Std       : {np.std(all_preds):.3f}")

    return avg_loss, mae

def traj_cutoff(trajs, length):
    new_trajs = []
    for traj in trajs:
        L = len(traj['observations'])
        if(L > (length + 1)):
             index_obs = L - (length + 1)
             index_acts = L - length
             index_rews = L - length
             traj['observations'] = traj['observations'][index_obs:]
             traj['actions'] = traj['actions'][index_acts:]
             traj['rewards'] = traj['rewards'][index_rews:]
        new_trajs.append(traj)
    return new_trajs

def get_success_trajs(trajs):
    success_trajs = []
    for traj in trajs:
        if(traj['rewards'][-1] == 1.0):
            success_trajs.append(traj)
    return success_trajs

class PlannerDataset(Dataset):
    def __init__(self, trajs: List[TrajectoryDict], horizon: int, dataset_name: str, specific_dataset: str, task_id: Optional[int] = None, cutoff_length: Optional[int] = None):
        self.trajs = copy.deepcopy(trajs)
        if(cutoff_length is not None):
            self.trajs = traj_cutoff(self.trajs, cutoff_length)
        print(f"total steps for Finetuning: {np.sum([len(traj['observations']) for traj in self.trajs])}")
        self.conditions = []
        self.horizon = horizon
        self.planner_processor = Planner_Processor(dataset_name, specific_dataset, task_id)
        for traj in self.trajs:
            obs = traj['observations']
            for t in range(len(obs)):
                s_norm = self.planner_processor.preprocess(obs[t])
                s_norm = torch.tensor(s_norm, dtype=torch.float32)
                self.conditions.append(s_norm)
    
    def __len__(self):
        return len(self.conditions)
   
    def __getitem__(self, idx):
        return self.conditions[idx]

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class RewardTracker:
    """Track and plot rewards during finetuning (mirrors LossTracker API)."""

    def __init__(self, save_dir: str = "./logs/"):
        self.save_dir = save_dir
        self.steps = []
        self.rewards = []
        #self.learning_rates = []
        self.constraints = []
        os.makedirs(save_dir, exist_ok=True)

    def log_reward(self, step: int, reward: float, constraint: Optional[float] = None):
        self.steps.append(step)
        self.rewards.append(reward)
        if constraint is not None:
            self.constraints.append(constraint)

    def save_logs(self, filename: str = "reward_logs.pkl"):
        data = {
            'steps': self.steps,
            'rewards': self.rewards,
            'constraints': self.constraints
        }
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Reward logs saved to {save_path}")

    def plot_reward_curve(self,
                          save_path: Optional[str] = None,
                          title: str = "Finetuning Reward Curve",
                          show_constraint: bool = False,
                          smooth_window: int = 50):
        if not self.rewards:
            print("No reward data to plot!")
            return

        sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor':'black'})
        plt.rcParams.update({'font.size': 14})

        okabe_ito = ["#D55E00","#000000", "#E69F00", "#56B4E9", "#009E73",
                       "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FF0000"]
        raw_color    = okabe_ito[3]   
        smooth_color = okabe_ito[4] 
        constraint_color     = okabe_ito[9]  

        fig, ax1 = plt.subplots(figsize=(12, 8))
        steps = np.array(self.steps)
        rewards = np.array(self.rewards)


         # Plot smoothed if possible
        if len(rewards) > smooth_window and smooth_window > 1:
            smoothed = self._smooth_curve(rewards, smooth_window)
            # only plot where valid (not nan)
            valid_idx = ~np.isnan(smoothed)
            ax1.plot(steps[valid_idx], smoothed[valid_idx],
                     color=smooth_color, linewidth=2.5,
                     label=f'Smoothed Reward (window={smooth_window})')
        
        
        ax1.plot(steps, rewards, alpha=0.3, color=raw_color, linewidth=1.0, label='Raw Reward')
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12, color=raw_color)
        ax1.tick_params(axis='y', labelcolor=raw_color)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fancybox=True, fontsize=12)
        sns.despine()

        if show_constraint and self.constraints:
            ax2 = ax1.twinx()
            C_vals = np.array(self.constraints)
            ax2.plot(steps[:len(C_vals)], C_vals, color=constraint_color, alpha=0.7, linewidth=1.5, label='Constraint')
            ax2.set_ylabel('Constraint', fontsize=12, color=constraint_color)
            ax2.tick_params(axis='y', labelcolor=constraint_color)
            ax2.legend(loc='upper right')
        
        sns.despine()
        #plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

       
        if save_path is None:
            save_path = os.path.join(self.save_dir, "reward_curve.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reward curve saved to {save_path}")
        plt.show()
        return fig


    def _smooth_curve(self, data: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        padded = np.full_like(data, np.nan)
        padded[window-1:] = smoothed
        return padded

def karras_beta_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: t_grid, beta_grid, sigma_grid
    beta(t) computed from VP-SDE marginals using Karras timesteps.
    """
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    sigma_k = sigma_min * (sigma_max / sigma_min) ** t
    alpha = 1.0 / torch.sqrt(1.0 + sigma_k**2)
    sigma = sigma_k * alpha

    # Compute β(t) from dσ²/dt = β(t) * σ²(t)
    # From VP-SDE: dσ²/dt = β(t) * (1 - σ²(t))
    # But we use numerical diff for stability
    
    sigma_sq = sigma**2
    d_sigma_sq = torch.diff(sigma_sq, dim=0)
    dt = torch.diff(t, dim=0)
    beta = d_sigma_sq / (1 - sigma_sq[:-1]) / dt
    beta = torch.cat([beta, beta[-1].unsqueeze(0)])  # pad last

    return t, beta, sigma

def clip_actions(x: torch.Tensor, d_s: int) -> torch.Tensor:
    actions = torch.clamp(x[..., d_s:], -1.0, 1.0)
    x[..., d_s:] = actions
    return x

def get_normalized_score(trajs, expert_score: Optional[float] = None):
    total = 0.0
    for i in range(len(trajs)):
        temp = 0.0
        for j in range(len(trajs[i]['rewards'])):
            #if(trajs[i]['rewards'][j] == 1):
            temp += (0.99**j) * trajs[i]['rewards'][j]
        total += temp
    avg_discounted_return = total / len(trajs)
    # 5. Compute normalized score
    normalized_score = 100 * avg_discounted_return 
    if(expert_score is not None):
        normalized_score = 100 * (normalized_score / expert_score)
    #print(f"Normalized Score: {normalized_score:.2f}")
    return normalized_score

def get_expert_score(dataset_name):
    if(dataset_name == 'kitchen'):
         data = get_dataset(dataset_name, 'complete')
         trajs = data.get_trajectories()
         score = get_normalized_score(trajs, None)
         return score
    else:
         return None

def get_current_state(s0, env_name):
    if isinstance(s0, dict):
        return s0['observation']
    return s0

def load_hyperparameters(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

def rollout_parallel(
    env_name, 
    specific_env, 
    horizon = 32, 
    steps_T = 50, 
    num_karras = 10, 
    eta = 0.8, 
    episode_length = 4000, 
    checkpoint_step = 1000000, 
    num_envs = 8, 
    goal_cell = None, 
    start_cells = None, 
    device: torch.device = None, 
    seed_base: int = 0):
     #print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     #print(f"Running {num_envs} environments in parallel")
     if device is None:
          device = "cuda" if torch.cuda.is_available() else "cpu"
     trajs = []
     #print(f"Using device {device}")
     
     # Uses Accelerate's RANK env var (automatically set in DDP)
     rank = int(os.environ.get("RANK", 0))
     np.random.seed(12345 + rank + seed_base)
     torch.manual_seed(12345 + rank + seed_base)
     
     # Create environment factory function
     _, d_s, d_a = get_env(env_name, specific_env)
     def make_env():
         env, _, _ = get_env(env_name, specific_env)
         return env
     
     # Create vectorized environment
     vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])
     #maze = env.unwrapped.maze  # Access the internal Maze object
     #maze_map = maze.maze_map
     #rows, cols = len(maze_map), len(maze_map[0])
    
     # Get Planner
     state_dict = get_planner(env_name, specific_env, checkpoint_step)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     
     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env)
     
     # <<< MODIFIED: Unique env reset seeds per process to prevent identical trajectories across GPUs
     reset_seeds = list(range(seed_base, seed_base + num_envs))
     
     """
     if(goal_cell is not None):
         maze = env.unwrapped.maze  # Access the internal Maze object
         maze_map = maze.maze_map
         rows, cols = len(maze_map), len(maze_map[0])
         free_cells = []
         for row in range(rows):
             for col in range(cols):
                 if maze_map[row][col] != 1:  # 1 = wall; others are free/open
                       free_cells.append(np.array([row, col]))
         free_cells = np.array(free_cells)
         
         
         free_cells = np.array([[6,6], [1,1], [1,6], [3,2], [5,4], [3,4], [4,1], [4,6], [2,4], [2,1]])
         selected_indices = np.random.choice(len(free_cells), size=4, replace=False)
         selected_free_cells = free_cells[selected_indices]
         
         selected_free_cells = np.array([[6,6], [5,4], [2,4], [2,1]])
         start_cells = []
         for i in range(len(selected_free_cells)):
             if(np.array_equal(selected_free_cells[i], goal_cell)):
                 continue
             else:
                 start_cells.append(selected_free_cells[i].copy())
         start_cells = np.array(start_cells)
     else:
         start_cells = [None]
     """
     total_steps = 0
     for start_cell in start_cells:
       # Reset all environments
       #seeds = list(range(num_envs)) 
       opt = {}
       if goal_cell is not None:
             opt["goal_cell"] = goal_cell.copy()
       else:
             opt['goal_cell'] = None
       if start_cell is not None:
             opt["reset_cell"] = start_cell.copy()
       else:
             opt['reset_cell'] = None
       s0_vec = vec_env.reset(seed = reset_seeds, options=[opt for _ in range(num_envs)])
       current_states = s0_vec[0]['observation']
     
       # Store trajectories for each environment
       all_rewards = [0.0 for _ in range(num_envs)]
       done_envs = [False for _ in range(num_envs)]
       observations = [[] for _ in range(num_envs)]
       acts = [[] for _ in range(num_envs)]
       rewards = [[] for _ in range(num_envs)]
       for env_idx in range(num_envs):
          observations[env_idx].append(current_states[env_idx].copy())
     
       for i in range(episode_length):
          actions = np.zeros((num_envs, d_a))
         
          # Generate actions for each environment
          for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             current_state = current_states[env_idx]
             current_state_norm = planner_processor.preprocess(current_state)
             x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
             action = x[0, d_s:(d_s+d_a)].copy()
             actions[env_idx] = action
         
          # Step all environments at once
          obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
         
          # Update trajectories
          for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             
             observations[env_idx].append(obs_vec['observation'][env_idx].copy())
             acts[env_idx].append(actions[env_idx].copy())
             rewards[env_idx].append(rewards_vec[env_idx])
             all_rewards[env_idx] += rewards_vec[env_idx]
             
             current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
             if terminated_vec[env_idx] or truncated_vec[env_idx]:
                 done_envs[env_idx] = True
                 #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
          # Check if all environments are done
          if all(done_envs):
             #print("All environments completed!")
             break
     
       # Find the trajectory with the maximum reward
       for env_idx in range(num_envs):
          total_steps += (len(observations[env_idx]) - 1)
          trajs.append({
              'observations': np.asarray(observations[env_idx].copy()),
              'actions': np.asarray(acts[env_idx].copy()),
              'rewards': np.asarray(reward_processor(rewards[env_idx].copy(), env_name))
          })
        
     vec_env.close()
     if(goal_cell is None):
          expert_score = get_expert_score(env_name)
          score = get_normalized_score(trajs, expert_score)
     else:
          score = get_normalized_score(trajs)
     #save_trajs(trajs, env_name, specific_env, checkpoint_step)
     #print(f"Average Normalized Score: {score:.2f}")
     return trajs, score, total_steps

def rollout_parallel2(
     env_name, 
     specific_env, 
     backbone_layers = 2,
     horizon = 32, 
     steps_T = 50, 
     num_karras = 10, 
     eta = 0.8, 
     episode_length = 4000, 
     checkpoint_step = 1000000, 
     num_envs = 8, 
     goal_cell: Optional[np.ndarray] = None, 
     start_cells: Optional[List[np.ndarray]] = None, 
     task_id: Optional[int] = None, 
     device: torch.device = None, 
     seed_base: int = 0, 
     continual_rollout = False, 
     chunk_size = 5):
     #print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     #print(f"Running {num_envs} environments in parallel")
     if device is None:
          device = "cuda" if torch.cuda.is_available() else "cpu"
     trajs = []
     #print(f"Using device {device}")
     
     # Uses Accelerate's RANK env var (automatically set in DDP)
     rank = int(os.environ.get("RANK", 0))
     np.random.seed(12345 + rank + seed_base)
     torch.manual_seed(12345 + rank + seed_base)
     
     # Create environment factory function
     _, d_s, d_a = get_env(env_name, specific_env, task_id = task_id)
     def make_env():
         env, _, _ = get_env(env_name, specific_env, task_id = task_id)
         return env
     
     # Create vectorized environment
     vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])
     #maze = env.unwrapped.maze  # Access the internal Maze object
     #maze_map = maze.maze_map
     #rows, cols = len(maze_map), len(maze_map[0])
    
     # Get Planner
     state_dict = get_planner(env_name, specific_env, checkpoint_step, task_id)
     """
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=backbone_layers, timestep_emb_type="fourier").to(device)
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=backbone_layers, timestep_emb_type="fourier").to(device)
     elif(env_name == 'antmaze'):
         model = DiT1d(in_dim = d_s, emb_dim = 128, d_model = 256, n_heads = 256//64, depth=backbone_layers, timestep_emb_type="fourier").to(device)
     elif env_name == 'cube':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=backbone_layers, timestep_emb_type="fourier").to(device)
     elif env_name == 'ogpointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=backbone_layers, timestep_emb_type="fourier").to(device)
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     """
     model = load_dit(d_s, d_a, state_dict, backbone_layers, device, env_name, eval_mode = True)
     
     






     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env, task_id)
     
     # <<< MODIFIED: Unique env reset seeds per process to prevent identical trajectories across GPUs
     reset_seeds = list(range(seed_base, seed_base + num_envs))
     
    
     total_steps = 0
     successes = []
     if (start_cells is not None):
      for start_cell in start_cells:
         # Reset all environments
         #seeds = list(range(num_envs)) 
        opt = {}
        if goal_cell is not None:
             opt["goal_cell"] = goal_cell.copy()
        else:
             opt['goal_cell'] = None
        if start_cell is not None:
             opt["reset_cell"] = start_cell.copy()
        else:
             opt['reset_cell'] = None
        
        s0_vec = vec_env.reset(seed = reset_seeds, options=[opt for _ in range(num_envs)])
        current_states = s0_vec[0]['observation']
     
        # Store trajectories for each environment
        all_rewards = [0.0 for _ in range(num_envs)]
        done_envs = [False for _ in range(num_envs)]
        observations = [[] for _ in range(num_envs)]
        acts = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        Temp_acts = [[] for _ in range(num_envs)]
        for env_idx in range(num_envs):
            observations[env_idx].append(current_states[env_idx].copy())
     
        for i in range(episode_length):
            actions = np.zeros((num_envs, d_a))
         
          # Generate actions for each environment
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
               if(continual_rollout):
                   if(len(Temp_acts[env_idx]) == 0):
                      current_state = current_states[env_idx]
                      current_state_norm = planner_processor.preprocess(current_state)
                      x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                      for k in range(len(x)):
                          Temp_acts[env_idx].append(x[k, d_s:(d_s+d_a)].copy())
                    
                   actions[env_idx] = Temp_acts[env_idx][0].copy()
                   Temp_acts[env_idx] = Temp_acts[env_idx][1:].copy()
               else:
                   current_state = current_states[env_idx]
                   current_state_norm = planner_processor.preprocess(current_state)
                   x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                   action = x[0, d_s:(d_s+d_a)].copy()
                   actions[env_idx] = action
         
            # Step all environments at once
            obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
         
             # Update trajectories
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
             
               observations[env_idx].append(obs_vec['observation'][env_idx].copy())
               acts[env_idx].append(actions[env_idx].copy())
               rewards[env_idx].append(rewards_vec[env_idx])
               all_rewards[env_idx] += rewards_vec[env_idx]
             
               current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
               if terminated_vec[env_idx] or truncated_vec[env_idx]:
                   done_envs[env_idx] = True
                   successes.append(int(info_vec['success'][env_idx]))
                   #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
             # Check if all environments are done
            if all(done_envs):
                #print("All environments completed!")
                break
        
        for env_idx in range(num_envs):
            if not done_envs[env_idx]:
                successes.append(0)
        
        for env_idx in range(num_envs):
                   total_steps += (len(observations[env_idx]) - 1)
                   trajs.append({
                      'observations': np.asarray(observations[env_idx].copy()),
                      'actions': np.asarray(acts[env_idx].copy()),
                      'rewards': np.asarray(rewards[env_idx].copy())
         }) 
     else:
        opt =  {"task_id": task_id}
        #s0_vec = vec_env.reset(seed = reset_seeds, options=[opt for _ in range(num_envs)])
        #current_states = s0_vec[0]['observation']
        obs0, _ = vec_env.reset(seed=reset_seeds, options=[opt for _ in range(num_envs)])
        #obs0, _ = vec_env.reset(seed=reset_seeds)
        if isinstance(obs0, dict):
              current_states = obs0['observation']
        else:
              current_states = obs0
     
        # Store trajectories for each environment
        all_rewards = [0.0 for _ in range(num_envs)]
        done_envs = [False for _ in range(num_envs)]
        observations = [[] for _ in range(num_envs)]
        acts = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        Temp_acts = [[] for _ in range(num_envs)]
        for env_idx in range(num_envs):
            observations[env_idx].append(current_states[env_idx].copy())
     
        for i in range(episode_length):
            actions = np.zeros((num_envs, d_a))
         
            # Generate actions for each environment
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
               if(continual_rollout):
                   if(len(Temp_acts[env_idx]) == 0):
                      current_state = current_states[env_idx]
                      current_state_norm = planner_processor.preprocess(current_state)
                      x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                      for k in range(chunk_size):
                          Temp_acts[env_idx].append(x[k, d_s:(d_s+d_a)].copy())
                    
                   actions[env_idx] = Temp_acts[env_idx][0].copy()
                   Temp_acts[env_idx] = Temp_acts[env_idx][1:].copy()
               else:
                   current_state = current_states[env_idx]
                   current_state_norm = planner_processor.preprocess(current_state)
                   x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                   action = x[0, d_s:(d_s+d_a)].copy()
                   actions[env_idx] = action
            
            # Step all environments at once
            obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
            obs_batch = obs_vec['observation'] if isinstance(obs_vec, dict) else obs_vec
         
             # Update trajectories
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
             
               observations[env_idx].append(obs_batch[env_idx].copy())
               acts[env_idx].append(actions[env_idx].copy())
               rewards[env_idx].append(rewards_vec[env_idx])
               all_rewards[env_idx] += rewards_vec[env_idx]
               current_states[env_idx] = obs_batch[env_idx].copy()
             
               if terminated_vec[env_idx] or truncated_vec[env_idx]:
                   done_envs[env_idx] = True
                   successes.append(int(info_vec['success'][env_idx]))
                   #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
             # Check if all environments are done
            if all(done_envs):
                    #print("All environments completed!")
                    break
     
            # Find the trajectory with the maximum reward
        
        
        for env_idx in range(num_envs):
            if not done_envs[env_idx]:
                 successes.append(0)

        for env_idx in range(num_envs):
                   total_steps += (len(observations[env_idx]) - 1)
                   trajs.append({
                      'observations': np.asarray(observations[env_idx].copy()),
                      'actions': np.asarray(acts[env_idx].copy()),
                      'rewards': np.asarray(reward_processor(rewards[env_idx].copy(), env_name))
        })     
     

     
     vec_env.close()
     success_rate = np.mean(successes) if len(successes) > 0 else 0.0
     print(f"success rate: {success_rate:.2f}")
     if(goal_cell is None):
            expert_score = get_expert_score(env_name)
            score = get_normalized_score(trajs, expert_score)
     else:
            score = get_normalized_score(trajs)
     #save_trajs(trajs, env_name, specific_env, checkpoint_step)
     #print(f"Average Normalized Score: {score:.2f}")
     return trajs, score, success_rate, total_steps

def rollout_parallel3(
    env_name, specific_env,
    horizon=32, steps_T=50, num_karras=10, eta=0.8,
    episode_length=4000, checkpoint_step=1000000,
    num_envs=8,
    goal_cell: Optional[np.ndarray] = None,
    start_cells: Optional[List[np.ndarray]] = None,
    task_id: Optional[int] = None,
    device: torch.device = None,
    seed_base: int = 0,
    continual_rollout=False,
    chunk_size=10,          # currently unused
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trajs = []
    total_steps = 0

    # Seeding
    rank = int(os.environ.get("RANK", 0))
    np.random.seed(12345 + rank + seed_base)
    torch.manual_seed(12345 + rank + seed_base)

    # Environment & Vector Env
    _, d_s, d_a = get_env(env_name, specific_env, task_id = task_id)

    def make_env():
        env, _, _ = get_env(env_name, specific_env, task_id = task_id)
        return env

    vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])

    # Load model
    state_dict = get_planner(env_name, specific_env, checkpoint_step, task_id)

    if env_name in ['kitchen', 'pointmaze', 'cube', 'antmaze']:
        model = DiT1d(
            in_dim=(d_s + d_a), emb_dim=128, d_model=256,
            n_heads=256//64, depth=2, timestep_emb_type="fourier"
        ).to(device)
    else:
        raise ValueError(f"Invalid Environment: {env_name}")

    model.load_state_dict(state_dict)
    model.eval()

    planner_processor = Planner_Processor(env_name, specific_env)
    reset_seeds = list(range(seed_base, seed_base + num_envs))

    def run_rollout(options_list: Optional[dict] = None):
        """Helper to run one batch of environments (avoids duplication)."""
        nonlocal total_steps
        
        if(options_list is not None):
             obs, info = vec_env.reset(seed=reset_seeds, options=options_list)
             #obs, info = vec_env.reset(options=options_list)
        else:
             obs, info = vec_env.reset(seed=reset_seeds)
             #obs, info = vec_env.reset()
        
        
        if isinstance(obs, dict):
            current_states = obs['observation']
        else:
            current_states = obs

        all_rewards = [0.0] * num_envs
        done_envs = [False] * num_envs
        observations = [[] for _ in range(num_envs)]
        acts = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        Temp_acts = [[] for _ in range(num_envs)]

        for env_idx in range(num_envs):
            observations[env_idx].append(current_states[env_idx].copy())

        for i in range(episode_length):
            actions = np.zeros((num_envs, d_a))

            for env_idx in range(num_envs):
                if done_envs[env_idx]:
                    continue

                current_state = current_states[env_idx]
                current_state_norm = planner_processor.preprocess(current_state)

                if continual_rollout and len(Temp_acts[env_idx]) > 0:
                    action = Temp_acts[env_idx].pop(0)
                else:
                    x = sample_euler_karras(
                        current_state_norm, model, d_s, d_a,
                        horizon, steps_T, num_karras, eta, device
                    )
                    if continual_rollout:
                        Temp_acts[env_idx] = [x[k, d_s:(d_s + d_a)].copy() 
                                              for k in range(chunk_size)]
                        action = Temp_acts[env_idx].pop(0)
                    else:
                        action = x[0, d_s:(d_s + d_a)].copy()

                actions[env_idx] = action

            # Step
            obs_vec, rewards_vec, terminated_vec, truncated_vec, _ = vec_env.step(actions)

            # Consistent observation extraction
            obs_batch = obs_vec['observation'] if isinstance(obs_vec, dict) else obs_vec

            # Update
            for env_idx in range(num_envs):
                if done_envs[env_idx]:
                    continue

                observations[env_idx].append(obs_batch[env_idx].copy())
                acts[env_idx].append(actions[env_idx].copy())
                rewards[env_idx].append(rewards_vec[env_idx])
                all_rewards[env_idx] += rewards_vec[env_idx]
                current_states[env_idx] = obs_batch[env_idx].copy()

                if terminated_vec[env_idx] or truncated_vec[env_idx]:
                    done_envs[env_idx] = True

            if all(done_envs):
                break

        # Append finished trajectories
        for env_idx in range(num_envs):
            total_steps += len(observations[env_idx]) - 1
            trajs.append({
                'observations': np.asarray(observations[env_idx][:-1]),
                'actions': np.asarray(acts[env_idx]),
                'rewards': np.asarray(reward_processor(rewards[env_idx].copy(), env_name))  # FIXED
            })

    # ====================== Main Logic ======================
    if start_cells is not None and len(start_cells) > 0:
        for start_cell in start_cells:
            opt = {
                "goal_cell": goal_cell.copy() if goal_cell is not None else None,
                "reset_cell": start_cell.copy() if start_cell is not None else None,
            }
           
            run_rollout([opt] * num_envs)
    else:
        run_rollout()

    vec_env.close()

    valid, success_rate = checktrajs(trajs)
    print(f"valid: {valid}, success rate: {success_rate:.2f}")

    if goal_cell is None:
        expert_score = get_expert_score(env_name)
        score = get_normalized_score(trajs, expert_score)
    else:
        score = get_normalized_score(trajs)
    
    return trajs, score, success_rate, total_steps

import math
from dataclasses import dataclass
@dataclass
class AlphaSchedulerConfig:
    alpha_start: float
    alpha_end: float
    total_steps: int
    decay: bool = True
    
class AlphaScheduler:
    def __init__(self, config: AlphaSchedulerConfig):
        self.alpha_start = config.alpha_start
        self.alpha_end = config.alpha_end
        self.total_steps = config.total_steps
        self.decay_rate = self.total_steps / 10.0
        self.current_step = 1
        self.current_alpha = self.alpha_start
        self.decay = config.decay
    
    def step_alpha(self):
        if self.decay:
           if self.current_step > self.total_steps:
              self.current_alpha = self.alpha_end
           else:
              alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * math.exp(-self.current_step / self.decay_rate)
              self.current_alpha = max(alpha, self.alpha_end)
              self.current_step += 1
        else:
           self.current_alpha = self.alpha_start

    def get_alpha(self):
        return self.current_alpha

def checktrajs(trajs):
    success = 0
    for i in range(len(trajs)):
        Dict = {1: 0, 0: 0}
        for j in range(len(trajs[i]['rewards'])):
            if(trajs[i]['rewards'][j] not in Dict.keys()):
                Dict[int(trajs[i]['rewards'][j])] = 1
            else:
                Dict[int(trajs[i]['rewards'][j])] += 1
        if(1 in Dict.keys()):
           if(Dict[1] > 1):
              return False, 0
        if(len(list(Dict.keys())) > 2):
            return False, 0
        success += Dict[1]
    success_rate = success / len(trajs)
    return True, success_rate

def check_success_rate(trajs: List[TrajectoryDict]):
    success = 0
    for traj in trajs:
        if(traj['rewards'][-1] == 1.0):
            success += 1
    return success / len(trajs)
 
def check_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using M3 GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Falling back to CPU (no GPU acceleration)")
    return device 
      
def compute_threshold_mahalanobis(kernels, dataloader, quantile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_D2_total = []
    for i, (s, a, s_next) in enumerate(dataloader):
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)
        #compute total mahalanobis distance
        with torch.no_grad():
            D2_total = compute_total_mahalanobis_score(kernels, s, a, s_next)
        all_D2_total.extend(D2_total.detach().cpu().numpy())
    
    all_D2_total = np.array(all_D2_total)
    mean_D2_total = float(all_D2_total.mean())
    min_D2_total = float(all_D2_total.min())
    max_D2_total = float(all_D2_total.max())
    var_D2_total = float(all_D2_total.var())
    tau = float(np.quantile(all_D2_total, quantile))
    print(f"mean_D2_total = {mean_D2_total:.4f}")
    print(f"min_D2_total = {min_D2_total:.4f}")
    print(f"max_D2_total = {max_D2_total:.4f}")
    print(f"variance_D2_total = {var_D2_total:.4f}")
    print(f"τ ({quantile*100:.0f}th percentile) : {tau:.4f}")
    return tau

def compute_threshold_mahalanobis_mog(kernels, dataloader, quantile, device):
    chunks = []
    with torch.no_grad():
        for s, a, s_next in dataloader:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            s_next = s_next.to(device, non_blocking=True)
            d2 = compute_total_mahalanobis_score_mog(kernels, s, a, s_next)
            chunks.append(d2.detach().float().cpu())
    all_vals = torch.cat(chunks, dim=0)
    tau = torch.quantile(all_vals, quantile).item()
    print(f"mean_D2_total = {all_vals.mean().item():.4f}")
    print(f"min_D2_total = {all_vals.min().item():.4f}")
    print(f"max_D2_total = {all_vals.max().item():.4f}")
    print(f"variance_D2_total = {all_vals.var(unbiased=False).item():.4f}")
    print(f"τ ({quantile*100:.0f}th percentile) : {tau:.4f}")
    return tau

def compute_threshold_log_prob(kernels, dataloader, quantile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_log_density_total = []
    for i, (s, a, s_next) in enumerate(dataloader):
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)
        #compute total mahalanobis distance
        with torch.no_grad():
            log_density_total = compute_log_density(kernels, s, a, s_next)
        all_log_density_total.extend(log_density_total.detach().cpu().numpy())
    
    all_log_density_total = np.array(all_log_density_total)
    mean_log_density_total = float(all_log_density_total.mean())
    min_log_density_total = float(all_log_density_total.min())
    max_log_density_total = float(all_log_density_total.max())
    var_log_density_total = float(all_log_density_total.var())
    tau = float(np.quantile(all_log_density_total, 1 - quantile))
    print(f"mean_D2_total = {mean_log_density_total:.4f}")
    print(f"min_D2_total = {min_log_density_total:.4f}")
    print(f"max_D2_total = {max_log_density_total:.4f}")
    print(f"variance_D2_total = {var_log_density_total:.4f}")
    print(f"τ ({(1 - quantile)*100:.0f}th percentile) : {tau:.4f}")
    return tau
    
def compute_threshold_log_prob_mog(kernels, dataloader, quantile, device):
    chunks = []
    with torch.no_grad():
        for s, a, s_next in dataloader:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            s_next = s_next.to(device, non_blocking=True)
            lp = compute_log_density_mog(kernels, s, a, s_next)
            chunks.append(lp.detach().float().cpu())
    all_vals = torch.cat(chunks, dim=0)
    tau = torch.quantile(all_vals, 1.0 - quantile).item()
    print(f"mean_log_density_total = {all_vals.mean().item():.4f}")
    print(f"min_log_density_total = {all_vals.min().item():.4f}")
    print(f"max_log_density_total = {all_vals.max().item():.4f}")
    print(f"variance_log_density_total = {all_vals.var(unbiased=False).item():.4f}")
    print(f"τ ({(1-quantile)*100:.0f}th percentile) : {tau:.4f}")
    return tau

def train_critic_with_planner(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 1000,
):
    @torch.no_grad()
    def _generate_plans_batch(
           s0_planner_norm: np.ndarray,   # (B, d_s) in planner-normalized space
           planner: nn.Module,
           d_s: int, d_a: int, horizon: int,
           steps_T: int, num_karras: int, eta: float,
           device: torch.device,
    ) -> torch.Tensor:
   
        plans = []
        for s0 in s0_planner_norm:
           x = sample_euler_karras(
               s0, planner, d_s, d_a, horizon,
               num_steps=steps_T, num_karras=num_karras, eta=eta, device=device,
           )
           plans.append(x)
        return torch.from_numpy(np.stack(plans, axis=0)).float().to(device)

    
    device = check_device()
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # ------------------------------------------------------------------ critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=0,
    )
    critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------- planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=2, timestep_emb_type="fourier",
    ).to(device)
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(planner_proc.stats.obs_mean, device=device, dtype=torch.float32)
    planner_std  = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ----------------------------------------------------------- reward model
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    ).to(device)
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)

    reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std  = torch.as_tensor(np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32)

    # ----------------------------------- critic stats: load once, never save
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset,
        task_id=task_id, step=old_critic_checkpoint,
    )
    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std  = torch.as_tensor(np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32)

    # ---------------------------------------------------- starting-state pool
    s0_pool = np.concatenate([t['observations'] for t in trajs], axis=0).astype(np.float32)

    # ----------------------------------------------------------------- optim
    optimizer = optim.Adam(critic.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32,
    )                                                                       # (n,)
    gamma_n = gamma ** n

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        # 1) sample raw start states
        idx    = np.random.randint(0, len(s0_pool), size=batch_size)
        s0_raw = s0_pool[idx]                                                # (B, d_s)

        with torch.no_grad():
            # 2) plan with the diffusion planner
            s0_p  = np.stack([planner_proc.preprocess(o) for o in s0_raw])
            plans = _generate_plans_batch(
                s0_p, planner, obs_dim, act_dim, horizon,
                steps_T, num_karras, eta, device,
            )                                                                # (B, H, d_s+d_a)

            # 3) recover RAW states from planner-norm; actions are already raw
            s_planner = plans[..., :obs_dim]                                 # (B, H, d_s)
            actions   = plans[..., obs_dim:]                                 # (B, H, d_a)
            s_raw     = s_planner * planner_std + planner_mean               # (B, H, d_s)

            # 4) reward model: r̂(s_t, a_t) for t = 0..n-1
            B, H, _ = s_raw.shape
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat   = reward_net(
                s_for_r.reshape(B * n, -1),
                actions[:, :n].reshape(B * n, -1),
            ).reshape(B, n)                                                  # (B, n)

            # 5) discounted return + bootstrapped target value
            disc_return  = (gamma_pow_t.unsqueeze(0) * r_hat).sum(dim=1)     # (B,)
            s_n_critic   = (s_raw[:, n] - c_mean) / c_std                    # (B, d_s)
            v_bootstrap  = target_critic(s_n_critic)                         # (B,)
            target_value = disc_return + gamma_n * v_bootstrap               # (B,)

            # 6) input for V_β(s_0)
            s0_critic = (s_raw[:, 0] - c_mean) / c_std                       # (B, d_s)

        # 7) gradient step on V_β
        v_pred = critic(s0_critic)                                           # (B,)
        #loss   = F.mse_loss(v_pred, target_value)
        loss = F.smooth_l1_loss(v_pred, target_value, beta = 1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 8) Polyak target update
        with torch.no_grad():
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()
        if k % log_every == 0:
            print(f"  step {k:>6}/{num_steps}   loss = {running / log_every:.4f}")
            running = 0.0

    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print("critic saved.")

class CriticDataset_Reward(Dataset):
    def __init__(self, dataset_name: str, 
                       specific_dataset: str, 
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs: List[TrajectoryDict], 
                       horizon: int = 32,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005,
                       value_scale: float = 5.0,
                       task_id: Optional[int] = None):
        # ----- gather raw obs/actions to fit stats -----

        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0)+ 1e-8
        if(old_step is not None):
             self.stats = update_critic_stats(dataset_name, specific_dataset, stats, task_id, old_step, momentum)
        else:
             self.stats = stats
        
        device = check_device()
        _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
        reward_state, _, _ = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )
        reward_net = SimpleReward(
            obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
        ).to(device)
        reward_net.load_state_dict(reward_state)
        reward_net.eval()
        for p in reward_net.parameters():
            p.requires_grad_(False)
        reward_stat = get_reward_stats(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )

        transitions = []
        
        for traj in trajs:
            obs = traj['observations'] 
            acts = traj['actions']   
            T_traj = min(len(obs), len(acts))
            
            if T_traj < horizon:
                continue
            
            with torch.no_grad():
                obs_for_r = reward_stat.norm_obs(obs[:T_traj]).astype(np.float32)
                s_t = torch.as_tensor(obs_for_r, dtype=torch.float32, device=device)
                a_t = torch.as_tensor(acts[:T_traj], dtype=torch.float32, device=device)
                #a_t = torch.clamp(a_t, -1.0, 1.0)
                rews = reward_net(s_t, a_t).cpu().numpy().astype(np.float32)   # (T_traj,)  
                
                """
                # Scale down predicted rewards from reward model
                rews = np.clip(rews, -20.0, 20.0)      # adjust bounds if needed
                rews = rews / 5.0                      # or use a running std
                """
                
                
                
                # Scale down predicted rewards from reward model
                rews = np.clip(rews, 0.0, 100.0)      # adjust bounds if needed
                rews = rews / value_scale                    # or use a running std
                
            
            for t in range(len(obs) - horizon):
                 obs_chunk = self.stats.norm_obs(obs[t : t + horizon]).astype(np.float32)
                 rews_chunk = rews[t: min(t+horizon, len(rews))]
                 transitions.append((obs_chunk, rews_chunk))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, new_step)
    
    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __getitem__(self, idx):
        obs_chunk, rews_chunk = self.transitions[idx]
        return (
            torch.tensor(obs_chunk, dtype = torch.float32),
            torch.tensor(rews_chunk, dtype = torch.float32)
        )
    def __len__(self):
        return len(self.transitions)

class Critic_Buffer_Reward():
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs:  List[TrajectoryDict],
                       horizon: int = 32,
                       gamma: float = 0.99,
                       lam: float = 0.95,
                       task_id: Optional[int] = None,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       value_scale: float = 5.0,
                       momentum: float = 0.005):
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.data = CriticDataset_Reward(
            dataset_name         = dataset_name,
            specific_dataset     = specific_dataset,
            reward_hidden_layers = reward_hidden_layers,
            reward_hidden_dim    = reward_hidden_dim,
            reward_checkpoint    = reward_checkpoint,
            trajs                = trajs,
            horizon              = horizon,
            old_step             = old_step,
            new_step             = new_step,
            momentum             = momentum,
            value_scale          = value_scale,
            task_id              = task_id,
        )
   
    """
    def obtain_training_data(self, target_critic: nn.Module, batch_size: int, tgt_mean: torch.Tensor, tgt_std: torch.Tensor, device: str):
        loader = cycle(DataLoader(
            self.data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ))
        obs_chunks, rews_chunks = next(loader)      # (B, T, dim), (B, T)
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]
        

        with torch.no_grad():
            values = target_critic(obs_chunks)            # (B, T)

            deltas = (
                  rews_chunks[:, :-1]
                  + self.gamma * values[:, 1:]
                   - values[:, :-1]
              )                                             # (B, T-1)

            advantages = torch.zeros(B, T - 1, device=device)
            last_adv = torch.zeros(B, device=device)
            for t in reversed(range(T - 1)):
                last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                advantages[:, t] = last_adv

            #value_targets = values[:, 0] + advantages[:, 0]   # (B,)
            with torch.no_grad():
                 values = target_critic(obs_chunks)                      # (B, T)
                 deltas = (
                       rews_chunks[:, :-1]
                       + self.gamma * values[:, 1:]
                       - values[:, :-1]
                 )                                                       # (B, T-1)

                  # GAE advantages
                 advantages = torch.zeros_like(deltas)
                 last_adv = torch.zeros(B, device=device)
                 for t in reversed(range(deltas.shape[1])):
                     last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                     advantages[:, t] = last_adv

                 # === ADD NORMALIZATION HERE ===
                 value_targets = values[:, 0] + advantages[:, 0]         # raw targets
                
                 
                 # Normalize advantages and targets (running stats or batch stats)
                 adv_mean = advantages.mean()
                 adv_std  = advantages.std() + 1e-8
                 advantages = (advantages - adv_mean) / adv_std
                 
                 alpha = 0.99
                 tgt_mean_new = value_targets.mean()
                 tgt_std_new  = value_targets.std() + 1e-8
                 tgt_mean_new = alpha * tgt_mean + ((1 - alpha) * tgt_mean_new)
                 tgt_std_new = alpha * tgt_std + ((1 - alpha) * tgt_std_new)
                 value_targets = (value_targets - tgt_mean_new) / tgt_std_new
                 # =================================
                 

        return obs_chunks[:, 0], value_targets, tgt_mean_new, tgt_std_new
        #return obs_chunks[:, 0], value_targets
    """

    def obtain_training_data(self, target_critic: nn.Module, batch, tgt_mean: torch.Tensor, tgt_std: torch.Tensor, device: str):
        
        obs_chunks, rews_chunks = batch
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]
        

        with torch.no_grad():
            values = target_critic(obs_chunks)            # (B, T)

            deltas = (
                  rews_chunks[:, :-1]
                  + self.gamma * values[:, 1:]
                   - values[:, :-1]
              )                                             # (B, T-1)

            advantages = torch.zeros(B, T - 1, device=device)
            last_adv = torch.zeros(B, device=device)
            for t in reversed(range(T - 1)):
                last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                advantages[:, t] = last_adv

            #value_targets = values[:, 0] + advantages[:, 0]   # (B,)
            with torch.no_grad():
                 values = target_critic(obs_chunks)                      # (B, T)
                 deltas = (
                       rews_chunks[:, :-1]
                       + self.gamma * values[:, 1:]
                       - values[:, :-1]
                 )                                                       # (B, T-1)

                  # GAE advantages
                 advantages = torch.zeros_like(deltas)
                 last_adv = torch.zeros(B, device=device)
                 for t in reversed(range(deltas.shape[1])):
                     last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                     advantages[:, t] = last_adv

                 # === ADD NORMALIZATION HERE ===
                 value_targets = values[:, 0] + advantages[:, 0]         # raw targets
                
                 
                 
                 # Normalize advantages and targets (running stats or batch stats)
                 adv_mean = advantages.mean()
                 adv_std  = advantages.std() + 1e-8
                 advantages = (advantages - adv_mean) / adv_std
                 
                 alpha = 0.99
                 tgt_mean_new = value_targets.mean()
                 tgt_std_new  = value_targets.std() + 1e-8
                 tgt_mean_new = alpha * tgt_mean + ((1 - alpha) * tgt_mean_new)
                 tgt_std_new = alpha * tgt_std + ((1 - alpha) * tgt_std_new)
                 #value_targets = (value_targets - tgt_mean_new) / tgt_std_new
                 # =================================
                
                 

        return obs_chunks[:, 0], value_targets, tgt_mean_new, tgt_std_new
        #return obs_chunks[:, 0], value_targets

def train_critic_with_reward(trajs: List[TrajectoryDict], 
                 dataset_name: str, 
                 specific_dataset: str, 
                 reward_hidden_layers: int,
                 reward_hidden_dim: int,
                 reward_checkpoint: int,
                 critic_hidden_layers: int, 
                 critic_hidden_dim: int, 
                 batch_size, 
                 num_steps, 
                 gamma, lam, horizon, 
                 lr, 
                 min_lr, 
                 tau, 
                 old_step: Optional[int] = None, 
                 new_step: int = 0, 
                 momentum: float = 0.005, 
                 value_scale: float = 5.0,
                 task_id: Optional[int] = None):
    device = check_device()
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers).to(device)
    if(old_step is not None):
        critic_state_dict, _ = get_critic_model(dataset_name, specific_dataset, task_id = task_id, step = old_step)
        critic.load_state_dict(critic_state_dict)
    target_critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    optimizer = optim.AdamW(critic.parameters(), lr = lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = num_steps,   # one scheduler step per training step
            eta_min = min_lr
        )
    critic.train()
    NS = 0 if new_step == -1 else new_step
    buffer = Critic_Buffer_Reward(
                       dataset_name,
                       specific_dataset,
                       reward_hidden_layers,
                       reward_hidden_dim,
                       reward_checkpoint,
                       trajs,
                       horizon,
                       gamma,
                       lam,
                       task_id,
                       old_step,  
                       NS, 
                       value_scale,
                       momentum)
    g = torch.Generator()
    g.manual_seed(1)
    loader = cycle(
        DataLoader(
            buffer.data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=g,
        )
    )
    print(f"Training critic for {dataset_name}-{specific_dataset}")
    total_loss = 0.0
    tgt_mean = torch.zeros(1, device=device)
    tgt_std = torch.ones(1, device=device)
    for k in range(1, num_steps + 1):  # number of passes over dataset
           batch = next(loader)
           s, target_value, tgt_mean, tgt_std = buffer.obtain_training_data(target_critic, batch, tgt_mean, tgt_std, device)
           s = s.to(device)
           target_value = target_value.to(device)

           # Predicted Q-values
           q_pred = critic(s)
           loss = F.smooth_l1_loss(q_pred, target_value, beta = 1.0)
           #loss = F.mse_loss(q_pred, target_value)
           total_loss += loss.item()

           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
           
           if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/1000}")
                wandb.log({"loss": total_loss/1000, "step": k})     
                total_loss = 0.0
            
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")
    q_scale = Q_Scale()
    q_scale.Q_scale = value_scale
    save_Q_scale(q_scale, dataset_name, specific_dataset, task_id)
    print(f"mean: {tgt_mean.item()}, std: {tgt_std.item()}")
    
    """
    q_stats = Q_Stats()
    q_stats.Q_mean = tgt_mean.item()
    q_stats.Q_std = tgt_std.item()
    save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
    print(f"mean: {tgt_mean.item()}, std: {tgt_std.item()}")
    return tgt_mean.item(), tgt_std.item()
    """

@dataclass
class KernelConfig:
    """Everything needed to load the kernel ensemble + run the feasibility filter."""
    checkpoint: int                                # which kernel checkpoint to load
    type_kernel: str = 'robust'                    # 'robust' or 'mog'
    num_hidden_layers: int = 2                     # must match training
    hidden_dim: int = 256                          # must match training
    num_modes: int = 8                             # mog only
    noise_floor: Optional[float] = 1e-4            # mog only
    min_log_prob: float = -10.0                    # feasibility threshold
    oversample: int = 4  

def train_critic_with_planner2(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: int,
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 1,
):

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )

        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:  # 'mog'
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(
            kernel_stat.obs_mean, device=device, dtype=torch.float32,
        )
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32,
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan:    torch.Tensor,        # (H, d_s)
        a_raw_plan:    torch.Tensor,        # (H, d_a)
        kernels:       List[nn.Module],
        k_mean:        torch.Tensor,        # (d_s,)
        k_std:         torch.Tensor,        # (d_s,)
        kernel_config: KernelConfig,
        device:        torch.device,
    ) -> bool:
        s_k   = (s_raw_plan - k_mean) / k_std
        s_t   = s_k[:-1]
        a_t   = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:  # 'mog'
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans(
        s0_pool:        np.ndarray,
        planner:        nn.Module,
        planner_proc:   Planner_Processor,
        planner_mean:   torch.Tensor,
        planner_std:    torch.Tensor,
        kernels:        List[nn.Module],
        k_mean:         torch.Tensor,
        k_std:          torch.Tensor,
        kernel_config:  KernelConfig,
        obs_dim:        int,
        act_dim:        int,
        horizon:        int,
        steps_T:        int,
        num_karras:     int,
        eta:            float,
        batch_size:     int,
        device:         torch.device,
    ):
        accepted_plans = []
        accepted_s0    = []
        max_attempts   = kernel_config.oversample * batch_size
        attempts       = 0

        while len(accepted_plans) < batch_size and attempts < max_attempts:
            idx    = np.random.randint(0, len(s0_pool))
            s0_raw = s0_pool[idx]
            s0_p   = planner_proc.preprocess(s0_raw)
            x      = sample_euler_karras(
                s0_p, planner, obs_dim, act_dim, horizon,
                num_steps=steps_T, num_karras=num_karras,
                eta=eta, device=device,
            )

            x_t       = torch.from_numpy(x).float().to(device)
            s_planner = x_t[..., :obs_dim]
            a_raw     = x_t[..., obs_dim:]
            s_raw_pl  = s_planner * planner_std + planner_mean

            if is_plan_feasible(
                s_raw_plan    = s_raw_pl,
                a_raw_plan    = a_raw,
                kernels       = kernels,
                k_mean        = k_mean,
                k_std         = k_std,
                kernel_config = kernel_config,
                device        = device,
            ):
                accepted_plans.append(x_t)
                accepted_s0.append(s0_raw)
            attempts += 1

        if len(accepted_plans) == 0:
            raise RuntimeError(
                f"No feasible plans found after {attempts} attempts. "
                f"Lower `kernel_config.min_log_prob` or raise "
                f"`kernel_config.oversample`."
            )
        if len(accepted_plans) < batch_size:
            print(f"[Critic-Online] only {len(accepted_plans)}/{batch_size} "
                  f"feasible plans after {attempts} attempts "
                  f"(min_log_prob={kernel_config.min_log_prob}); proceeding")

        plans      = torch.stack(accepted_plans, dim=0)
        s0_raw_acc = np.stack(accepted_s0, axis=0)
        return plans, s0_raw_acc

    # ------------------------------------------------------------------ setup
    device = check_device()
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # ------------------------------------------------------------------ critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
    )
    critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------- planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    ).to(device)
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32,
    )
    planner_std  = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ----------------------------------------------------------- reward model
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    ).to(device)
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(
        reward_stat.obs_mean, device=device, dtype=torch.float32,
    )
    r_std  = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ------------------------------------------------------------------ kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config,
        obs_dim, act_dim, device,
    )

    # ----------------------------------- critic stats: load once, never save
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset,
        task_id=task_id, step=0,
    )
    c_mean = torch.as_tensor(
        critic_stat.obs_mean, device=device, dtype=torch.float32,
    )
    c_std  = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ---------------------------------------------------- starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)
    
    # === NEW: Running stats for targets ===
    running_tgt_mean = torch.zeros(1, device=device)
    running_tgt_std  = torch.ones(1, device=device)
    alpha = 0.99   # momentum
    # ======================================

    # ----------------------------------------------------------------- optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32,
    )
    gamma_n = gamma ** n

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        with torch.no_grad():
            # 1) sample feasible plans (handles s_0 sampling internally)
            plans, _ = _generate_feasible_plans(
                s0_pool       = s0_pool,
                planner       = planner,
                planner_proc  = planner_proc,
                planner_mean  = planner_mean,
                planner_std   = planner_std,
                kernels       = kernels,
                k_mean        = k_mean,
                k_std         = k_std,
                kernel_config = kernel_config,
                obs_dim       = obs_dim,
                act_dim       = act_dim,
                horizon       = horizon,
                steps_T       = steps_T,
                num_karras    = num_karras,
                eta           = eta,
                batch_size    = batch_size,
                device        = device,
            )                                                                 # (B', H, d_s+d_a)

            # 2) split planner output: states (planner-norm) and raw actions
            s_planner = plans[..., :obs_dim]                                  # (B', H, d_s)
            actions   = plans[..., obs_dim:]                                  # (B', H, d_a)
            s_raw     = s_planner * planner_std + planner_mean                # (B', H, d_s)

            # 3) reward model: r̂(s_t, a_t) for t = 0..n-1
            B, H, _ = s_raw.shape
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat   = reward_net(
                s_for_r.reshape(B * n, -1),
                actions[:, :n].reshape(B * n, -1),
            ).reshape(B, n)  
            
            """
            # NEW: Strong scaling
            r_hat = torch.clamp(r_hat, -20.0, 20.0)
            r_hat = r_hat / 5.0 
            """                                # (B', n)

            # 4) discounted return + bootstrapped target value
            disc_return  = (gamma_pow_t.unsqueeze(0) * r_hat).sum(dim=1)      # (B',)
            s_n_critic   = (s_raw[:, n] - c_mean) / c_std                     # (B', d_s)
            v_bootstrap  = target_critic(s_n_critic)                          # (B',)
            target_value = disc_return + gamma_n * v_bootstrap                # (B',)

           
            # === NEW: Running normalization ===
            batch_mean = target_value.mean()
            batch_std  = target_value.std(unbiased=False) + 1e-8

            running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
            running_tgt_std  = alpha * running_tgt_std  + (1 - alpha) * batch_std

            normalized_target = (target_value - running_tgt_mean) / running_tgt_std
            # =================================

            # 5) input for V_β(s_0)
            s0_critic = (s_raw[:, 0] - c_mean) / c_std                        # (B', d_s)

        # 6) gradient step on V_β
        v_pred = critic(s0_critic)                                            # (B',)
        loss   = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 7) Polyak target update
        with torch.no_grad():
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()
        """
        if k % log_every == 0:
            print(f"  step {k:>6}/{num_steps}   loss = {running / log_every:.4f}")
            running = 0.0
        """

    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print("critic saved.")

def train_critic_with_planner3(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 0,
):

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )

        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:  # 'mog'
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(
            kernel_stat.obs_mean, device=device, dtype=torch.float32,
        )
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32,
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan:    torch.Tensor,        # (H, d_s)
        a_raw_plan:    torch.Tensor,        # (H, d_a)
        kernels:       List[nn.Module],
        k_mean:        torch.Tensor,        # (d_s,)
        k_std:         torch.Tensor,        # (d_s,)
        kernel_config: KernelConfig,
        device:        torch.device,
    ) -> bool:
        s_k   = (s_raw_plan - k_mean) / k_std
        s_t   = s_k[:-1]
        a_t   = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:  # 'mog'
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans(
        s0_pool:        np.ndarray,
        planner:        nn.Module,
        planner_proc:   Planner_Processor,
        planner_mean:   torch.Tensor,
        planner_std:    torch.Tensor,
        kernels:        List[nn.Module],
        k_mean:         torch.Tensor,
        k_std:          torch.Tensor,
        kernel_config:  KernelConfig,
        obs_dim:        int,
        act_dim:        int,
        horizon:        int,
        steps_T:        int,
        num_karras:     int,
        eta:            float,
        batch_size:     int,
        device:         torch.device,
    ):
        accepted_plans = []
        accepted_s0    = []
        max_attempts   = kernel_config.oversample * batch_size
        attempts       = 0

        while len(accepted_plans) < batch_size and attempts < max_attempts:
            idx    = np.random.randint(0, len(s0_pool))
            s0_raw = s0_pool[idx]
            s0_p   = planner_proc.preprocess(s0_raw)
            x      = sample_euler_karras(
                s0_p, planner, obs_dim, act_dim, horizon,
                num_steps=steps_T, num_karras=num_karras,
                eta=eta, device=device,
            )

            x_t       = torch.from_numpy(x).float().to(device)
            s_planner = x_t[..., :obs_dim]
            a_raw     = x_t[..., obs_dim:]
            s_raw_pl  = s_planner * planner_std + planner_mean

            if is_plan_feasible(
                s_raw_plan    = s_raw_pl,
                a_raw_plan    = a_raw,
                kernels       = kernels,
                k_mean        = k_mean,
                k_std         = k_std,
                kernel_config = kernel_config,
                device        = device,
            ):
                accepted_plans.append(x_t)
                accepted_s0.append(s0_raw)
            attempts += 1

        if len(accepted_plans) == 0:
            raise RuntimeError(
                f"No feasible plans found after {attempts} attempts. "
                f"Lower `kernel_config.min_log_prob` or raise "
                f"`kernel_config.oversample`."
            )
        if len(accepted_plans) < batch_size:
            print(f"[Critic-Online] only {len(accepted_plans)}/{batch_size} "
                  f"feasible plans after {attempts} attempts "
                  f"(min_log_prob={kernel_config.min_log_prob}); proceeding")

        plans      = torch.stack(accepted_plans, dim=0)
        s0_raw_acc = np.stack(accepted_s0, axis=0)
        return plans, s0_raw_acc

    # ------------------------------------------------------------------ setup
    device = check_device()
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # ------------------------------------------------------------------ critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    if(old_critic_checkpoint is not None):
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
         )
        critic.load_state_dict(critic_state)
    

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------- planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    ).to(device)
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32,
    )
    planner_std  = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ----------------------------------------------------------- reward model
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    ).to(device)
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(
        reward_stat.obs_mean, device=device, dtype=torch.float32,
    )
    r_std  = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ------------------------------------------------------------------ kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config,
        obs_dim, act_dim, device,
    )
    
    
    # ----------------------------------- critic stats: load once, never save
    if(old_critic_checkpoint is not None):
         critic_stat = get_critic_stats(
             dataset_name, specific_dataset,
             task_id=task_id, step=0,
         )
    else:
         critic_stat = obtain_and_save_critic_stats(trajs, dataset_name, specific_dataset, task_id, step = 0)
    
    c_mean = torch.as_tensor(
        critic_stat.obs_mean, device=device, dtype=torch.float32,
    )
    c_std  = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ---------------------------------------------------- starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)
    
    
    # === NEW: Running stats for targets ===
    running_tgt_mean = torch.zeros(1, device=device)
    running_tgt_std  = torch.ones(1, device=device)
    alpha = 0.99   # momentum
    # ======================================


    # ----------------------------------------------------------------- optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32,
    )
    gamma_n = gamma ** n

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        with torch.no_grad():
            # 1) sample feasible plans (handles s_0 sampling internally)
            plans, _ = _generate_feasible_plans(
                s0_pool       = s0_pool,
                planner       = planner,
                planner_proc  = planner_proc,
                planner_mean  = planner_mean,
                planner_std   = planner_std,
                kernels       = kernels,
                k_mean        = k_mean,
                k_std         = k_std,
                kernel_config = kernel_config,
                obs_dim       = obs_dim,
                act_dim       = act_dim,
                horizon       = horizon,
                steps_T       = steps_T,
                num_karras    = num_karras,
                eta           = eta,
                batch_size    = batch_size,
                device        = device,
            )                                                                 # (B', H, d_s+d_a)

            # 2) split planner output: states (planner-norm) and raw actions
            s_planner = plans[..., :obs_dim]                                  # (B', H, d_s)
            actions   = plans[..., obs_dim:]                                  # (B', H, d_a)
            s_raw     = s_planner * planner_std + planner_mean                # (B', H, d_s)

            # 3) reward model: r̂(s_t, a_t) for t = 0..n-1
            B, H, _ = s_raw.shape
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat   = reward_net(
                s_for_r.reshape(B * n, -1),
                actions[:, :n].reshape(B * n, -1),
            ).reshape(B, n)  
            
            """
            # NEW: Strong scaling
            r_hat = torch.clamp(r_hat, -10.0, 10.0)
            r_hat = r_hat / 5.0          
            """             # (B', n)

            # 4) discounted return + bootstrapped target value
            disc_return  = (gamma_pow_t.unsqueeze(0) * r_hat).sum(dim=1)      # (B',)
            s_n_critic   = (s_raw[:, n] - c_mean) / c_std                     # (B', d_s)
            v_bootstrap  = target_critic(s_n_critic)                          # (B',)
            target_value = disc_return + gamma_n * v_bootstrap                # (B',)

            
            
            # === NEW: Running normalization ===
            batch_mean = target_value.mean()
            batch_std  = target_value.std(unbiased=False) + 1e-8

            running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
            running_tgt_std  = alpha * running_tgt_std  + (1 - alpha) * batch_std

            normalized_target = (target_value - running_tgt_mean) / running_tgt_std
            # =================================
        

            # 5) input for V_β(s_0)
            s0_critic = (s_raw[:, 0] - c_mean) / c_std                        # (B', d_s)

        # 6) gradient step on V_β
        v_pred = critic(s0_critic)                                            # (B',)
        loss   = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)
        #loss   = F.smooth_l1_loss(v_pred, target_value, beta=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 7) Polyak target update
        with torch.no_grad():
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()
        
        if log_every > 0 and k % log_every == 0:
            print(f"  step {k:>6}/{num_steps}   loss = {running / log_every:.4f}")
            running = 0.0
    

    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    q_stats = Q_Stats()
    q_stats.Q_mean = running_tgt_mean.item()
    q_stats.Q_std = running_tgt_std.item()
    save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
    print("critic saved.")
    return running_tgt_mean.item(), running_tgt_std.item()
               
def obtain_and_save_critic_stats(trajs: List[TrajectoryDict], dataset_name: str, specific_dataset: str, task_id: Optional[int] = None, step: int = 0):
        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0)+ 1e-8
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(stats, f)
        print(f"saved stats to {savepath}")
        return stats

def train_critic_with_planner4(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lam: Optional[float] = None,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 0,
    accelerator=None,
):

    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(kernel_stat.obs_mean, device=device, dtype=torch.float32)
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan: torch.Tensor,
        a_raw_plan: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
    ) -> bool:
        s_k = (s_raw_plan - k_mean) / k_std
        s_t = s_k[:-1]
        a_t = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans_parallel(
        s0_pool: np.ndarray,
        planner: nn.Module,
        planner_proc: Planner_Processor,
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        batch_size: int,
        device: torch.device,
        accelerator,
    ):
        """
        - Sample exactly `batch_size` starting states (s0)
        - For each s0, generate `oversample` plans
        - Keep every plan that is accepted
        - Discard an s0 only if none of its plans were accepted
        - Work is split across GPUs
        """
        oversample = kernel_config.oversample

        # 1. Sample batch_size starting states (same on every rank)
        if accelerator.is_main_process:
            rng = np.random.RandomState(42)
            s0_indices = rng.randint(0, len(s0_pool), size=batch_size)
            selected_s0 = s0_pool[s0_indices]
        else:
            selected_s0 = np.empty((batch_size, s0_pool.shape[1]), dtype=np.float32)

        selected_s0_tensor = torch.from_numpy(selected_s0).to(device)
        if accelerator.num_processes > 1:
            dist.broadcast(selected_s0_tensor, src=0)
        selected_s0 = selected_s0_tensor.cpu().numpy()

        # 2. Split the batch_size s0 across GPUs
        local_s0_indices = np.array_split(
            np.arange(batch_size), accelerator.num_processes
        )[accelerator.process_index]
        local_s0 = selected_s0[local_s0_indices]

        # 3. For each local s0, generate `oversample` plans
        local_accepted = []

        for s0_raw in local_s0:
            s0_p = planner_proc.preprocess(s0_raw)
            accepted_for_this_s0 = []

            for _ in range(oversample):
                x = sample_euler_karras(
                    s0_p, planner, obs_dim, act_dim, horizon,
                    num_steps=steps_T, num_karras=num_karras,
                    eta=eta, device=device,
                )
                x_t = torch.from_numpy(x).float().to(device)

                s_planner = x_t[..., :obs_dim]
                a_raw = x_t[..., obs_dim:]
                a_raw = torch.clamp(a_raw, -1.0, 1.0)
                s_raw_pl = s_planner * planner_std + planner_mean

                if is_plan_feasible(
                    s_raw_plan=s_raw_pl,
                    a_raw_plan=a_raw,
                    kernels=kernels,
                    k_mean=k_mean,
                    k_std=k_std,
                    kernel_config=kernel_config,
                    device=device,
                ):
                    accepted_for_this_s0.append(x_t.cpu())

            local_accepted.extend(accepted_for_this_s0)

        # 4. Collect from all GPUs
        if accelerator.num_processes > 1:
            all_accepted_lists = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_accepted_lists, local_accepted)
        else:
            all_accepted_lists = [local_accepted]

        all_plans = [p for sublist in all_accepted_lists for p in sublist]
        
        """
        if len(all_plans) == 0:
            raise RuntimeError(
                f"No feasible plans found across {accelerator.num_processes} GPUs. "
                f"Lower kernel_config.min_log_prob or increase oversample."
            )

        if accelerator.is_main_process:
            print(f"[Critic-Online] collected {len(all_plans)} feasible plans "
                  f"(from {batch_size} s0 × {oversample} attempts)")
        """

        plans = torch.stack(all_plans).to(device)
        return plans, None

    # ------------------------------------------------------------------ setup
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    if old_critic_checkpoint is not None:
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
        )
        critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    target_critic = target_critic.to(device)

    # planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(device)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32
    )
    planner_std = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # reward
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_net = reward_net.to(device)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    # critic stats
    if old_critic_checkpoint is not None:
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )
    else:
        if is_main:
            critic_stat = obtain_and_save_critic_stats(
                trajs, dataset_name, specific_dataset, task_id, step=0
            )
        accelerator.wait_for_everyone()
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )

    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)

    # running target stats
    if(old_critic_checkpoint is None):
        running_tgt_mean = torch.zeros(1, device=device)
        running_tgt_std = torch.ones(1, device=device)
    else:
        q_stats = get_Q_stats(dataset_name, specific_dataset, task_id, old_critic_checkpoint)
        running_tgt_mean = q_stats.Q_mean
        running_tgt_std = q_stats.Q_std
    alpha = 0.99

    # optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    # prepare only trainable parts
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )
    gamma_n = gamma ** n

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        with torch.no_grad():
            plans, _ = _generate_feasible_plans_parallel(
                s0_pool=s0_pool,
                planner=planner,
                planner_proc=planner_proc,
                planner_mean=planner_mean,
                planner_std=planner_std,
                kernels=kernels,
                k_mean=k_mean,
                k_std=k_std,
                kernel_config=kernel_config,
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=horizon,
                steps_T=steps_T,
                num_karras=num_karras,
                eta=eta,
                batch_size=batch_size,
                device=device,
                accelerator=accelerator,
            )

            B_eff = plans.shape[0]
            if B_eff < max(8, batch_size // 4):
                continue

            s_planner = plans[..., :obs_dim]
            #actions = plans[..., obs_dim:]
            actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
            s_raw = s_planner * planner_std + planner_mean

            N, H, _ = s_raw.shape
            n = H - 1

            # rewards for t = 0 .. n-1
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat = reward_net(
                s_for_r.reshape(N * n, -1),
                actions[:, :n].reshape(N * n, -1),
            ).reshape(N, n)  # (N, n)
             
            # reward clipping -----------------------------------------------------
            #r_hat = torch.clamp(r_hat, -20.0, 20.0)
            #r_hat = r_hat / 5.0

            # ---------------------------------------------------------------
            # New multi-horizon average target for every plan:
            # y_plan = 1/(n-1) * sum_{H=2}^{n} (
            #     sum_{t=1}^{H-1} gamma^t * r_t  +  gamma^H * V(s_H)
            # )
            # In 0-based indexing this becomes:
            # for L = 1 .. n-1:
            #     sum_{t=0}^{L-1} gamma^{t+1} * r[t]  +  gamma^{L+1} * V(s[L])
            # ---------------------------------------------------------------
            """
            plan_targets = torch.zeros(N, device=device)

            for L in range(1, n):  # L = 1 .. n-1  → H = 2 .. n
                # sum_{t=0}^{L-1} gamma^{t+1} * r[t]
                #discounts = gamma_pow_t[:L] * gamma          # gamma^1 ... gamma^L
                discounts = gamma_pow_t[:L]         
                disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)

                # bootstrap at step L
                s_L = (s_raw[:, L] - c_mean) / c_std
                v_boot = target_critic(s_L)
                #partial = disc_return + (gamma ** (L + 1)) * v_boot
                partial = disc_return + (gamma ** L) * v_boot

                plan_targets += partial

            plan_targets = plan_targets / (n - 1)            # average over H=2..n
            """
            
            plan_targets = torch.zeros(N, device=device)

            if(lam is not None):
                  w = 1.0 - lam       # first weight = (1-λ)
                  weight_sum = 0.0

                  for L in range(1, n):                               # L = 1 .. n-1
                         discounts = gamma_pow_t[:L]                     # γ⁰ … γ^{L-1}
                         disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                         s_L = (s_raw[:, L] - c_mean) / c_std
                         v_boot = target_critic(s_L)
                         partial = disc_return + (gamma ** L) * v_boot   # R^{(L)}
                         plan_targets += w * partial
                         weight_sum += w
                         w *= lam
            
                  plan_targets = plan_targets / max(weight_sum, 1e-8)
            
            else:
                    # ----- equal weight on all multi-step estimators -----
                    #plan_targets = torch.zeros(N, device=device)
                    N_est = n - 1                                 # L = 1 … n-1

                    for L in range(1, n):
                        discounts = gamma_pow_t[:L]               # γ⁰ … γ^{L-1}
                        disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                        s_L = (s_raw[:, L] - c_mean) / c_std
                        v_boot = target_critic(s_L)
                        partial = disc_return + (gamma ** L) * v_boot   # classic sum return
                        plan_targets += partial / N_est                 # equal weight
            

            # ----- average targets per unique s0 -----
            s0_raw = s_raw[:, 0]
            s0_key = torch.round(s0_raw * 1e5) / 1e5

            unique_s0, inverse_indices = torch.unique(
                s0_key, dim=0, return_inverse=True
            )

            U = unique_s0.shape[0]
            averaged_targets = torch.zeros(U, device=device)
            counts = torch.zeros(U, device=device)

            averaged_targets.index_add_(0, inverse_indices, plan_targets)
            counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
            averaged_targets = averaged_targets / counts.clamp(min=1.0)

            # running normalization
            batch_mean = averaged_targets.mean()
            batch_std = averaged_targets.std(unbiased=False) + 1e-8
            running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
            running_tgt_std = alpha * running_tgt_std + (1 - alpha) * batch_std
            normalized_target = (averaged_targets - running_tgt_mean) / running_tgt_std

            # critic input
            s0_critic = (unique_s0 - c_mean) / c_std

        # gradient step
        v_pred = critic(s0_critic)
        loss = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Polyak update
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()
    
        if log_every > 0 and k % log_every == 0 and is_main:
            print(
                f" step {k:>6}/{num_steps} "
                f"loss = {running / log_every:.10f}  "
                f"B_eff={B_eff}  U={U}  "
                f"tgt_mean={running_tgt_mean.item():.3f}  "
                f"tgt_std={running_tgt_std.item():.3f}"
            )
            running = 0.0

    # final save
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)

        q_stats = Q_Stats()
        q_stats.Q_mean = running_tgt_mean.item()
        q_stats.Q_std = running_tgt_std.item()
        save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
        print("critic saved.")

    return running_tgt_mean.item(), running_tgt_std.item()


def train_critic_with_planner5(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 0,
    use_multi_horizon: bool = False,          # NEW: True → multi-horizon average, False → standard n-step
    accelerator=None,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(kernel_stat.obs_mean, device=device, dtype=torch.float32)
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan: torch.Tensor,
        a_raw_plan: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
    ) -> bool:
        s_k = (s_raw_plan - k_mean) / k_std
        s_t = s_k[:-1]
        a_t = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans_parallel(
        s0_pool: np.ndarray,
        planner: nn.Module,
        planner_proc: Planner_Processor,
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        batch_size: int,
        device: torch.device,
        accelerator,
    ):
        """
        - Sample exactly `batch_size` starting states (s0)
        - For each s0, generate `oversample` plans
        - Keep every plan that is accepted
        - Discard an s0 only if none of its plans were accepted
        - Work is split across GPUs
        """
        oversample = kernel_config.oversample

        # 1. Sample batch_size starting states (same on every rank)
        if accelerator.is_main_process:
            rng = np.random.RandomState(42)
            s0_indices = rng.randint(0, len(s0_pool), size=batch_size)
            selected_s0 = s0_pool[s0_indices]
        else:
            selected_s0 = np.empty((batch_size, s0_pool.shape[1]), dtype=np.float32)

        selected_s0_tensor = torch.from_numpy(selected_s0).to(device)
        if accelerator.num_processes > 1:
            dist.broadcast(selected_s0_tensor, src=0)
        selected_s0 = selected_s0_tensor.cpu().numpy()

        # 2. Split the batch_size s0 across GPUs
        local_s0_indices = np.array_split(
            np.arange(batch_size), accelerator.num_processes
        )[accelerator.process_index]
        local_s0 = selected_s0[local_s0_indices]

        # 3. For each local s0, generate `oversample` plans
        local_accepted = []

        for s0_raw in local_s0:
            s0_p = planner_proc.preprocess(s0_raw)
            accepted_for_this_s0 = []

            for _ in range(oversample):
                x = sample_euler_karras(
                    s0_p, planner, obs_dim, act_dim, horizon,
                    num_steps=steps_T, num_karras=num_karras,
                    eta=eta, device=device,
                )
                x_t = torch.from_numpy(x).float().to(device)

                s_planner = x_t[..., :obs_dim]
                a_raw = x_t[..., obs_dim:]
                s_raw_pl = s_planner * planner_std + planner_mean

                if is_plan_feasible(
                    s_raw_plan=s_raw_pl,
                    a_raw_plan=a_raw,
                    kernels=kernels,
                    k_mean=k_mean,
                    k_std=k_std,
                    kernel_config=kernel_config,
                    device=device,
                ):
                    accepted_for_this_s0.append(x_t.cpu())

            local_accepted.extend(accepted_for_this_s0)

        # 4. Collect from all GPUs
        if accelerator.num_processes > 1:
            all_accepted_lists = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_accepted_lists, local_accepted)
        else:
            all_accepted_lists = [local_accepted]

        all_plans = [p for sublist in all_accepted_lists for p in sublist]
        
        """
        if len(all_plans) == 0:
            raise RuntimeError(
                f"No feasible plans found across {accelerator.num_processes} GPUs. "
                f"Lower kernel_config.min_log_prob or increase oversample."
            )

        if accelerator.is_main_process:
            print(f"[Critic-Online] collected {len(all_plans)} feasible plans "
                  f"(from {batch_size} s0 × {oversample} attempts)")
        """
        plans = torch.stack(all_plans).to(device)
        return plans, None

    # ------------------------------------------------------------------ setup
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    if old_critic_checkpoint is not None:
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
        )
        critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    target_critic = target_critic.to(device)

    # planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(device)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32
    )
    planner_std = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # reward
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_net = reward_net.to(device)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    # critic stats
    if old_critic_checkpoint is not None:
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )
    else:
        if is_main:
            critic_stat = obtain_and_save_critic_stats(
                trajs, dataset_name, specific_dataset, task_id, step=0
            )
        accelerator.wait_for_everyone()
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )

    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)

     # running target stats
    if(old_critic_checkpoint is None):
        running_tgt_mean = torch.zeros(1, device=device)
        running_tgt_std = torch.ones(1, device=device)
    else:
        q_stats = get_Q_stats(dataset_name, specific_dataset, task_id, old_critic_checkpoint)
        running_tgt_mean = q_stats.Q_mean
        running_tgt_std = q_stats.Q_std
    alpha = 0.99

    # optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    # prepare only trainable parts
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )
    gamma_n = gamma ** n

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        with torch.no_grad():
            plans, _ = _generate_feasible_plans_parallel(
                s0_pool=s0_pool,
                planner=planner,
                planner_proc=planner_proc,
                planner_mean=planner_mean,
                planner_std=planner_std,
                kernels=kernels,
                k_mean=k_mean,
                k_std=k_std,
                kernel_config=kernel_config,
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=horizon,
                steps_T=steps_T,
                num_karras=num_karras,
                eta=eta,
                batch_size=batch_size,
                device=device,
                accelerator=accelerator,
            )

            B_eff = plans.shape[0]
            if B_eff < max(8, batch_size // 4):
                continue

            s_planner = plans[..., :obs_dim]
            actions = plans[..., obs_dim:]
            s_raw = s_planner * planner_std + planner_mean

            N, H, _ = s_raw.shape
            n = H - 1

            # rewards for t = 0 .. n-1
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat = reward_net(
                s_for_r.reshape(N * n, -1),
                actions[:, :n].reshape(N * n, -1),
            ).reshape(N, n)  # (N, n)

            # reward clipping -----------------------------------------------------
            r_hat = torch.clamp(r_hat, -20.0, 20.0)
            r_hat = r_hat / 5.0

            # ---------------------------------------------------------------
            # Compute per-plan targets
            # ---------------------------------------------------------------
            if use_multi_horizon:
                # λ-weighted multi-step returns (finite-horizon GAE-style)
                # y = (1/Z) * Σ_{L=1}^{n-1} (1-λ) λ^{L-1} * R^{(L)}
                # where R^{(L)} = Σ_{t=0}^{L-1} γ^t r_t + γ^L V(s_L)
                plan_targets = torch.zeros(N, device=device)
                w = 1.0 - lam
                weight_sum = 0.0

                for L in range(1, n):
                    discounts = gamma_pow_t[:L]                              # γ⁰ … γ^{L-1}
                    disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                    s_L = (s_raw[:, L] - c_mean) / c_std
                    v_boot = target_critic(s_L)
                    partial = disc_return + (gamma ** L) * v_boot           # R^{(L)}

                    plan_targets += w * partial
                    weight_sum += w
                    w *= lam

                plan_targets = plan_targets / max(weight_sum, 1e-8)

            else:
                # Standard n-step return:
                # y_plan = sum_{t=0}^{n-1} gamma^t * r_t  +  gamma^n * V(s_n)
                disc_return = (gamma_pow_t.unsqueeze(0) * r_hat).sum(dim=1)
                s_n = (s_raw[:, n] - c_mean) / c_std
                v_boot = target_critic(s_n)
                plan_targets = disc_return + gamma_n * v_boot

            # ----- average targets per unique s0 -----
            s0_raw = s_raw[:, 0]
            s0_key = torch.round(s0_raw * 1e5) / 1e5

            unique_s0, inverse_indices = torch.unique(
                s0_key, dim=0, return_inverse=True
            )

            U = unique_s0.shape[0]
            averaged_targets = torch.zeros(U, device=device)
            counts = torch.zeros(U, device=device)

            averaged_targets.index_add_(0, inverse_indices, plan_targets)
            counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
            averaged_targets = averaged_targets / counts.clamp(min=1.0)

            # running normalization
            batch_mean = averaged_targets.mean()
            batch_std = averaged_targets.std(unbiased=False) + 1e-8
            running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
            running_tgt_std = alpha * running_tgt_std + (1 - alpha) * batch_std
            normalized_target = (averaged_targets - running_tgt_mean) / running_tgt_std

            # critic input
            s0_critic = (unique_s0 - c_mean) / c_std

        # gradient step
        v_pred = critic(s0_critic)
        loss = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Polyak update
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()

        if log_every > 0 and k % log_every == 0 and is_main:
            mode = "multi-horizon" if use_multi_horizon else "n-step"
            print(
                f" step {k:>6}/{num_steps} "
                f"loss = {running / log_every:.4f}  "
                f"B_eff={B_eff}  U={U}  mode={mode}  "
                f"tgt_mean={running_tgt_mean.item():.3f}  "
                f"tgt_std={running_tgt_std.item():.3f}"
            )
            running = 0.0

    # final save
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)

        q_stats = Q_Stats()
        q_stats.Q_mean = running_tgt_mean.item()
        q_stats.Q_std = running_tgt_std.item()
        save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
        print("critic saved.")

    return running_tgt_mean.item(), running_tgt_std.item()


"""

def train_critic_with_planner6(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lam: Optional[float] = None,
    rho: float = 1.0,          # conservatism: R_target = R_mean - rho * R_std (used when lam is None)
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 0,
    accelerator=None,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(kernel_stat.obs_mean, device=device, dtype=torch.float32)
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan: torch.Tensor,
        a_raw_plan: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
    ) -> bool:
        s_k = (s_raw_plan - k_mean) / k_std
        s_t = s_k[:-1]
        a_t = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans_parallel(
        s0_pool: np.ndarray,
        planner: nn.Module,
        planner_proc: Planner_Processor,
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        batch_size: int,
        device: torch.device,
        accelerator,
    ):
        
        oversample = kernel_config.oversample

        # 1. Sample batch_size starting states (same on every rank)
        if accelerator.is_main_process:
            rng = np.random.RandomState(42)
            s0_indices = rng.randint(0, len(s0_pool), size=batch_size)
            selected_s0 = s0_pool[s0_indices]
        else:
            selected_s0 = np.empty((batch_size, s0_pool.shape[1]), dtype=np.float32)

        selected_s0_tensor = torch.from_numpy(selected_s0).to(device)
        if accelerator.num_processes > 1:
            dist.broadcast(selected_s0_tensor, src=0)
        selected_s0 = selected_s0_tensor.cpu().numpy()

        # 2. Split the batch_size s0 across GPUs
        local_s0_indices = np.array_split(
            np.arange(batch_size), accelerator.num_processes
        )[accelerator.process_index]
        local_s0 = selected_s0[local_s0_indices]

        # 3. For each local s0, generate `oversample` plans
        local_accepted = []

        for s0_raw in local_s0:
            s0_p = planner_proc.preprocess(s0_raw)
            accepted_for_this_s0 = []

            for _ in range(oversample):
                x = sample_euler_karras(
                    s0_p, planner, obs_dim, act_dim, horizon,
                    num_steps=steps_T, num_karras=num_karras,
                    eta=eta, device=device,
                )
                x_t = torch.from_numpy(x).float().to(device)

                s_planner = x_t[..., :obs_dim]
                a_raw = x_t[..., obs_dim:]
                a_raw = torch.clamp(a_raw, -1.0, 1.0)
                s_raw_pl = s_planner * planner_std + planner_mean

                if is_plan_feasible(
                    s_raw_plan=s_raw_pl,
                    a_raw_plan=a_raw,
                    kernels=kernels,
                    k_mean=k_mean,
                    k_std=k_std,
                    kernel_config=kernel_config,
                    device=device,
                ):
                    accepted_for_this_s0.append(x_t.cpu())

            local_accepted.extend(accepted_for_this_s0)

        # 4. Collect from all GPUs
        if accelerator.num_processes > 1:
            all_accepted_lists = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_accepted_lists, local_accepted)
        else:
            all_accepted_lists = [local_accepted]

        all_plans = [p for sublist in all_accepted_lists for p in sublist]

        plans = torch.stack(all_plans).to(device)
        return plans, None

    # ------------------------------------------------------------------ setup
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    if old_critic_checkpoint is not None:
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
        )
        critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    target_critic = target_critic.to(device)

    # planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(device)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32
    )
    planner_std = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # reward
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_net = reward_net.to(device)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    # critic stats
    if old_critic_checkpoint is not None:
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )
    else:
        if is_main:
            critic_stat = obtain_and_save_critic_stats(
                trajs, dataset_name, specific_dataset, task_id, step=0
            )
        accelerator.wait_for_everyone()
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )

    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)

    # running target stats
    if old_critic_checkpoint is None:
        running_tgt_mean = torch.zeros(1, device=device)
        running_tgt_std = torch.ones(1, device=device)
    else:
        
        q_stats = get_Q_stats(dataset_name, specific_dataset, task_id, old_critic_checkpoint)
        running_tgt_mean = q_stats.Q_mean
        running_tgt_std = q_stats.Q_std
    alpha = 0.99

    # optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    # prepare only trainable parts
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        with torch.no_grad():
            plans, _ = _generate_feasible_plans_parallel(
                s0_pool=s0_pool,
                planner=planner,
                planner_proc=planner_proc,
                planner_mean=planner_mean,
                planner_std=planner_std,
                kernels=kernels,
                k_mean=k_mean,
                k_std=k_std,
                kernel_config=kernel_config,
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=horizon,
                steps_T=steps_T,
                num_karras=num_karras,
                eta=eta,
                batch_size=batch_size,
                device=device,
                accelerator=accelerator,
            )

            B_eff = plans.shape[0]
            if B_eff < max(8, batch_size // 4):
                continue

            s_planner = plans[..., :obs_dim]
            actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
            s_raw = s_planner * planner_std + planner_mean

            N, H, _ = s_raw.shape
            n = H - 1

            # rewards for t = 0 .. n-1
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat = reward_net(
                s_for_r.reshape(N * n, -1),
                actions[:, :n].reshape(N * n, -1),
            ).reshape(N, n)  # (N, n)

            
             # reward clipping -----------------------------------------------------
            #r_hat = torch.clamp(r_hat, -20.0, 20.0)
            #r_hat = r_hat / 5.0
            
           
            # reward clipping -----------------------------------------------------
            r_hat = torch.clamp(r_hat, 0.0, 100.0)      # adjust bounds if needed
            #r_hat = r_hat / 5.0                      # or use a running std

            plan_targets = torch.zeros(N, device=device)

            if lam is not None:
                # λ-return (unchanged)
                w = 1.0 - lam
                weight_sum = 0.0

                for L in range(1, n):  # L = 1 .. n-1
                    discounts = gamma_pow_t[:L]
                    disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                    s_L = (s_raw[:, L] - c_mean) / c_std
                    v_boot = target_critic(s_L)
                    partial = disc_return + (gamma ** L) * v_boot
                    plan_targets += w * partial
                    weight_sum += w
                    w *= lam

                plan_targets = plan_targets / max(weight_sum, 1e-8)

            else:
                # Conservative multi-horizon target:
                #   R^K = sum_{t=0}^{K-1} γ^t r̂_t + γ^K V_bar(s_K),  K = 1..n-1
                #   R_mean = mean_K R^K
                #   R_std  = std_K(R^K)
                #   R_target = R_mean - rho * R_std
                r_list = []
                for L in range(1, n):  # L = 1 .. n-1  ↔ K = 2 .. N in 1-based form
                    discounts = gamma_pow_t[:L]
                    disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                    s_L = (s_raw[:, L] - c_mean) / c_std
                    v_boot = target_critic(s_L)
                    partial = disc_return + (gamma ** L) * v_boot
                    r_list.append(partial)

                R = torch.stack(r_list, dim=1)  # (N, n-1)
                R_mean = R.mean(dim=1)          # (N,)
                R_std = R.std(dim=1, unbiased=False).clamp(min=0.0)  # (N,)
                plan_targets = R_mean - rho * R_std

            # ----- average targets per unique s0 -----
            s0_raw = s_raw[:, 0]
            s0_key = torch.round(s0_raw * 1e5) / 1e5

            unique_s0, inverse_indices = torch.unique(
                s0_key, dim=0, return_inverse=True
            )

            U = unique_s0.shape[0]
            averaged_targets = torch.zeros(U, device=device)
            counts = torch.zeros(U, device=device)

            averaged_targets.index_add_(0, inverse_indices, plan_targets)
            counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
            averaged_targets = averaged_targets / counts.clamp(min=1.0)

            # running normalization
            batch_mean = averaged_targets.mean()
            batch_std = averaged_targets.std(unbiased=False) + 1e-8
            running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
            running_tgt_std = alpha * running_tgt_std + (1 - alpha) * batch_std
            normalized_target = (averaged_targets - running_tgt_mean) / running_tgt_std

            # critic input
            s0_critic = (unique_s0 - c_mean) / c_std

        # gradient step
        v_pred = critic(s0_critic)
        loss = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Polyak update
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()

        if log_every > 0 and k % log_every == 0 and is_main:
            wandb.log({"loss": running / log_every, 
                       "tgt_mean": running_tgt_mean.item(),
                       "tgt_std": running_tgt_std.item(),
                       "step": k})     
            print(
                f" step {k:>6}/{num_steps} "
                f"loss = {running / log_every:.10f}  "
                f"B_eff={B_eff}  U={U}  "
                f"tgt_mean={running_tgt_mean.item():.3f}  "
                f"tgt_std={running_tgt_std.item():.3f}"
            )
            running = 0.0

    # final save
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)

        q_stats = Q_Stats()
        q_stats.Q_mean = running_tgt_mean.item()
        q_stats.Q_std = running_tgt_std.item()
        save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
        print("critic saved.")

    return running_tgt_mean.item(), running_tgt_std.item()

"""


"""
def train_critic_with_planner6(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lam: Optional[float] = None,
    rho: float = 1.0,          # conservatism: R_target = R_mean - rho * R_std (used when lam is None)
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 0,
    accelerator=None,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(kernel_stat.obs_mean, device=device, dtype=torch.float32)
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan: torch.Tensor,
        a_raw_plan: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
    ) -> bool:
        s_k = (s_raw_plan - k_mean) / k_std
        s_t = s_k[:-1]
        a_t = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans_parallel(
        s0_pool: np.ndarray,
        planner: nn.Module,
        planner_proc: Planner_Processor,
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        batch_size: int,
        device: torch.device,
        accelerator,
    ):
        
        oversample = kernel_config.oversample

        # 1. Sample batch_size starting states (same on every rank)
        if accelerator.is_main_process:
            rng = np.random.RandomState(42)
            s0_indices = rng.randint(0, len(s0_pool), size=batch_size)
            selected_s0 = s0_pool[s0_indices]
        else:
            selected_s0 = np.empty((batch_size, s0_pool.shape[1]), dtype=np.float32)

        selected_s0_tensor = torch.from_numpy(selected_s0).to(device)
        if accelerator.num_processes > 1:
            dist.broadcast(selected_s0_tensor, src=0)
        selected_s0 = selected_s0_tensor.cpu().numpy()

        # 2. Split the batch_size s0 across GPUs
        local_s0_indices = np.array_split(
            np.arange(batch_size), accelerator.num_processes
        )[accelerator.process_index]
        local_s0 = selected_s0[local_s0_indices]

        # 3. For each local s0, generate `oversample` plans
        local_accepted = []

        for s0_raw in local_s0:
            s0_p = planner_proc.preprocess(s0_raw)
            accepted_for_this_s0 = []

            for _ in range(oversample):
                x = sample_euler_karras(
                    s0_p, planner, obs_dim, act_dim, horizon,
                    num_steps=steps_T, num_karras=num_karras,
                    eta=eta, device=device,
                )
                x_t = torch.from_numpy(x).float().to(device)

                s_planner = x_t[..., :obs_dim]
                a_raw = x_t[..., obs_dim:]
                a_raw = torch.clamp(a_raw, -1.0, 1.0)
                s_raw_pl = s_planner * planner_std + planner_mean

                if is_plan_feasible(
                    s_raw_plan=s_raw_pl,
                    a_raw_plan=a_raw,
                    kernels=kernels,
                    k_mean=k_mean,
                    k_std=k_std,
                    kernel_config=kernel_config,
                    device=device,
                ):
                    accepted_for_this_s0.append(x_t.cpu())

            local_accepted.extend(accepted_for_this_s0)

        # 4. Collect from all GPUs
        if accelerator.num_processes > 1:
            all_accepted_lists = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_accepted_lists, local_accepted)
        else:
            all_accepted_lists = [local_accepted]

        all_plans = [p for sublist in all_accepted_lists for p in sublist]

        plans = torch.stack(all_plans).to(device)
        return plans, None

    # ------------------------------------------------------------------ setup
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    if old_critic_checkpoint is not None:
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
        )
        critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    target_critic = target_critic.to(device)

    # planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(device)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32
    )
    planner_std = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # reward
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_net = reward_net.to(device)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    # critic stats
    if old_critic_checkpoint is not None:
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )
    else:
        if is_main:
            critic_stat = obtain_and_save_critic_stats(
                trajs, dataset_name, specific_dataset, task_id, step=0
            )
        accelerator.wait_for_everyone()
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )

    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)

    # running target stats
    if old_critic_checkpoint is None:
        running_tgt_mean = torch.zeros(1, device=device)
        running_tgt_std = torch.ones(1, device=device)
    else:
        
        q_stats = get_Q_stats(dataset_name, specific_dataset, task_id, old_critic_checkpoint)
        running_tgt_mean = q_stats.Q_mean
        running_tgt_std = q_stats.Q_std
    alpha = 0.99

    # optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    # prepare only trainable parts
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        with torch.no_grad():
            plans, _ = _generate_feasible_plans_parallel(
                s0_pool=s0_pool,
                planner=planner,
                planner_proc=planner_proc,
                planner_mean=planner_mean,
                planner_std=planner_std,
                kernels=kernels,
                k_mean=k_mean,
                k_std=k_std,
                kernel_config=kernel_config,
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=horizon,
                steps_T=steps_T,
                num_karras=num_karras,
                eta=eta,
                batch_size=batch_size,
                device=device,
                accelerator=accelerator,
            )

            B_eff = plans.shape[0]
            if B_eff < max(8, batch_size // 4):
                continue

            s_planner = plans[..., :obs_dim]
            actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
            s_raw = s_planner * planner_std + planner_mean

            N, H, _ = s_raw.shape
            n = H - 1

            # rewards for t = 0 .. n-1
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat = reward_net(
                s_for_r.reshape(N * n, -1),
                actions[:, :n].reshape(N * n, -1),
            ).reshape(N, n)  # (N, n)

            
             # reward clipping -----------------------------------------------------
            r_hat = torch.clamp(r_hat, -20.0, 20.0)
            r_hat = r_hat / 5.0
            
            
            print(f"reward value mean: {r_hat.mean().item()}")
            print(f"reward value min: {r_hat.min().item()}")
            print(f"reward value max: {r_hat.max().item()}")
            # reward clipping -----------------------------------------------------
            #r_hat = torch.clamp(r_hat, 0.0, 100.0)      # adjust bounds if needed
            #r_hat = r_hat / 5.0                      # or use a running std
        
            plan_targets = torch.zeros(N, device=device)

            if lam is not None:
                # λ-return (unchanged)
                w = 1.0 - lam
                weight_sum = 0.0

                for L in range(1, n):  # L = 1 .. n-1
                    discounts = gamma_pow_t[:L]
                    disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                    s_L = (s_raw[:, L] - c_mean) / c_std
                    v_boot = target_critic(s_L)
                    v_boot = (v_boot * running_tgt_std) + running_tgt_mean
                    partial = disc_return + (gamma ** L) * v_boot
                    plan_targets += w * partial
                    weight_sum += w
                    w *= lam

                plan_targets = plan_targets / max(weight_sum, 1e-8)

            else:
                # Conservative multi-horizon target:
                #   R^K = sum_{t=0}^{K-1} γ^t r̂_t + γ^K V_bar(s_K),  K = 1..n-1
                #   R_mean = mean_K R^K
                #   R_std  = std_K(R^K)
                #   R_target = R_mean - rho * R_std
                r_list = []
                for L in range(1, n):  # L = 1 .. n-1  ↔ K = 2 .. N in 1-based form
                    discounts = gamma_pow_t[:L]
                    disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                    s_L = (s_raw[:, L] - c_mean) / c_std
                    v_boot = target_critic(s_L)
                    print(f"critic value normalized: {v_boot.mean().item()}")
                    v_boot = (v_boot * running_tgt_std) + running_tgt_mean
                    print(f"critic value denormalized: {v_boot.mean().item()}")
                    partial = disc_return + (gamma ** L) * v_boot
                    r_list.append(partial)

                R = torch.stack(r_list, dim=1)  # (N, n-1)
                R_mean = R.mean(dim=1)          # (N,)
                R_std = R.std(dim=1, unbiased=False).clamp(min=0.0)  # (N,)
                plan_targets = R_mean - rho * R_std

            # ----- average targets per unique s0 -----
            s0_raw = s_raw[:, 0]
            s0_key = torch.round(s0_raw * 1e5) / 1e5

            unique_s0, inverse_indices = torch.unique(
                s0_key, dim=0, return_inverse=True
            )

            U = unique_s0.shape[0]
            averaged_targets = torch.zeros(U, device=device)
            counts = torch.zeros(U, device=device)

            averaged_targets.index_add_(0, inverse_indices, plan_targets)
            counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
            averaged_targets = averaged_targets / counts.clamp(min=1.0)

            # running normalization
            batch_mean = averaged_targets.mean()
            batch_std = averaged_targets.std(unbiased=False) + 1e-8
            running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
            running_tgt_std = alpha * running_tgt_std + (1 - alpha) * batch_std
            normalized_target = (averaged_targets - running_tgt_mean) / running_tgt_std

            # critic input
            s0_critic = (unique_s0 - c_mean) / c_std

        # gradient step
        v_pred = critic(s0_critic)
        loss = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Polyak update
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()

        if log_every > 0 and k % log_every == 0 and is_main:
            
            wandb.log({"loss": running / log_every, 
                       "tgt_mean": running_tgt_mean.item(),
                       "tgt_std": running_tgt_std.item(),
                       "step": k})     
            
            print(
                f" step {k:>6}/{num_steps} "
                f"loss = {running / log_every:.10f}  "
                f"B_eff={B_eff}  U={U}  "
                f"tgt_mean={running_tgt_mean.item():.3f}  "
                f"tgt_std={running_tgt_std.item():.3f}"
            )
            running = 0.0

    # final save
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)

        q_stats = Q_Stats()
        q_stats.Q_mean = running_tgt_mean.item()
        q_stats.Q_std = running_tgt_std.item()
        save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
        print("critic saved.")

    return running_tgt_mean.item(), running_tgt_std.item()

"""


def train_critic_with_planner6(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 100,
    resample_every: int = 10,
    horizon: int = 32,
    gamma: float = 0.99,
    lam: Optional[float] = None,
    rho: float = 1.0,          # conservatism: R_target = R_mean - rho * R_std (used when lam is None)
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 0,
    accelerator=None,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(kernel_stat.obs_mean, device=device, dtype=torch.float32)
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan: torch.Tensor,
        a_raw_plan: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
    ) -> bool:
        s_k = (s_raw_plan - k_mean) / k_std
        s_t = s_k[:-1]
        a_t = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def _generate_feasible_plans_parallel(
        s0_pool: np.ndarray,
        planner: nn.Module,
        planner_proc: Planner_Processor,
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        batch_size: int,
        device: torch.device,
        accelerator,
    ):
        
        oversample = kernel_config.oversample

        # 1. Sample batch_size starting states (same on every rank)
        if accelerator.is_main_process:
            rng = np.random.RandomState(42)
            s0_indices = rng.randint(0, len(s0_pool), size=batch_size)
            selected_s0 = s0_pool[s0_indices]
        else:
            selected_s0 = np.empty((batch_size, s0_pool.shape[1]), dtype=np.float32)

        selected_s0_tensor = torch.from_numpy(selected_s0).to(device)
        if accelerator.num_processes > 1:
            dist.broadcast(selected_s0_tensor, src=0)
        selected_s0 = selected_s0_tensor.cpu().numpy()

        # 2. Split the batch_size s0 across GPUs
        local_s0_indices = np.array_split(
            np.arange(batch_size), accelerator.num_processes
        )[accelerator.process_index]
        local_s0 = selected_s0[local_s0_indices]

        # 3. For each local s0, generate `oversample` plans
        local_accepted = []

        for s0_raw in local_s0:
            s0_p = planner_proc.preprocess(s0_raw)
            accepted_for_this_s0 = []

            for _ in range(oversample):
                x = sample_euler_karras(
                    s0_p, planner, obs_dim, act_dim, horizon,
                    num_steps=steps_T, num_karras=num_karras,
                    eta=eta, device=device,
                )
                x_t = torch.from_numpy(x).float().to(device)

                s_planner = x_t[..., :obs_dim]
                a_raw = x_t[..., obs_dim:]
                a_raw = torch.clamp(a_raw, -1.0, 1.0)
                s_raw_pl = s_planner * planner_std + planner_mean

                if is_plan_feasible(
                    s_raw_plan=s_raw_pl,
                    a_raw_plan=a_raw,
                    kernels=kernels,
                    k_mean=k_mean,
                    k_std=k_std,
                    kernel_config=kernel_config,
                    device=device,
                ):
                    accepted_for_this_s0.append(x_t.cpu())

            local_accepted.extend(accepted_for_this_s0)

        # 4. Collect from all GPUs
        if accelerator.num_processes > 1:
            all_accepted_lists = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_accepted_lists, local_accepted)
        else:
            all_accepted_lists = [local_accepted]

        all_plans = [p for sublist in all_accepted_lists for p in sublist]

        plans = torch.stack(all_plans).to(device)
        return plans, None

    # ------------------------------------------------------------------ setup
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    if old_critic_checkpoint is not None:
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
        )
        critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    target_critic = target_critic.to(device)

    # planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(device)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32
    )
    planner_std = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # reward
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_net = reward_net.to(device)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    # critic stats
    if old_critic_checkpoint is not None:
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )
    else:
        if is_main:
            critic_stat = obtain_and_save_critic_stats(
                trajs, dataset_name, specific_dataset, task_id, step=0
            )
        accelerator.wait_for_everyone()
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )

    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    # starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)
    
    """
    # running target stats
    if old_critic_checkpoint is None:
        running_tgt_mean = torch.zeros(1, device=device)
        running_tgt_std = torch.ones(1, device=device)
    else:
        
        q_stats = get_Q_stats(dataset_name, specific_dataset, task_id, old_critic_checkpoint)
        running_tgt_mean = q_stats.Q_mean
        running_tgt_std = q_stats.Q_std
    """
    
    Scale = get_Q_scale(dataset_name, specific_dataset, task_id)
    running_tgt_mean = torch.zeros(1, device=device)
    running_tgt_std = torch.ones(1, device=device)
    
    alpha = 0.99

    # optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    # prepare only trainable parts
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        if (k - 1) % resample_every == 0:
            with torch.no_grad():
              plans, _ = _generate_feasible_plans_parallel(
                s0_pool=s0_pool,
                planner=planner,
                planner_proc=planner_proc,
                planner_mean=planner_mean,
                planner_std=planner_std,
                kernels=kernels,
                k_mean=k_mean,
                k_std=k_std,
                kernel_config=kernel_config,
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=horizon,
                steps_T=steps_T,
                num_karras=num_karras,
                eta=eta,
                batch_size=batch_size,
                device=device,
                accelerator=accelerator,
              )

              B_eff = plans.shape[0]
              if B_eff < max(8, batch_size // 4):
                  continue

              s_planner = plans[..., :obs_dim]
              actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
              s_raw = s_planner * planner_std + planner_mean

              N, H, _ = s_raw.shape
              n = H - 1

              # rewards for t = 0 .. n-1
              s_for_r = (s_raw[:, :n] - r_mean) / r_std
              r_hat = reward_net(
                s_for_r.reshape(N * n, -1),
                actions[:, :n].reshape(N * n, -1),
              ).reshape(N, n)  # (N, n)

              """
               # reward clipping -----------------------------------------------------
              r_hat = torch.clamp(r_hat, -20.0, 20.0)
              r_hat = r_hat / 5.0
              """
            
            
              # reward clipping -----------------------------------------------------
              r_hat = torch.clamp(r_hat, 0.0, 100.0)      # adjust bounds if needed
              r_hat = r_hat / Scale.Q_scale                     # or use a running std
        
              plan_targets = torch.zeros(N, device=device)

              if lam is not None:
                # λ-return (unchanged)
                  w = 1.0 - lam
                  weight_sum = 0.0

                  for L in range(1, n):  # L = 1 .. n-1
                      discounts = gamma_pow_t[:L]
                      disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                      s_L = (s_raw[:, L] - c_mean) / c_std
                      v_boot = target_critic(s_L)
                      #v_boot = (v_boot * running_tgt_std) + running_tgt_mean
                      partial = disc_return + (gamma ** L) * v_boot
                      plan_targets += w * partial
                      weight_sum += w
                      w *= lam

                  plan_targets = plan_targets / max(weight_sum, 1e-8)

              else:
                  # Conservative multi-horizon target:
                  #   R^K = sum_{t=0}^{K-1} γ^t r̂_t + γ^K V_bar(s_K),  K = 1..n-1
                  #   R_mean = mean_K R^K
                  #   R_std  = std_K(R^K)
                  #   R_target = R_mean - rho * R_std
                  r_list = []
                  for L in range(1, n):  # L = 1 .. n-1  ↔ K = 2 .. N in 1-based form
                      discounts = gamma_pow_t[:L]
                      disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                      s_L = (s_raw[:, L] - c_mean) / c_std
                      v_boot = target_critic(s_L)
                      #print(f"critic value normalized: {v_boot.mean().item()}")
                      #v_boot = (v_boot * running_tgt_std) + running_tgt_mean
                      #print(f"critic value denormalized: {v_boot.mean().item()}")
                      partial = disc_return + (gamma ** L) * v_boot
                      r_list.append(partial)

                  R = torch.stack(r_list, dim=1)  # (N, n-1)
                  R_mean = R.mean(dim=1)          # (N,)
                  R_std = R.std(dim=1, unbiased=False).clamp(min=0.0)  # (N,)
                  plan_targets = R_mean - rho * R_std

              # ----- average targets per unique s0 -----
              s0_raw = s_raw[:, 0]
              s0_key = torch.round(s0_raw * 1e5) / 1e5

              unique_s0, inverse_indices = torch.unique(
                s0_key, dim=0, return_inverse=True
              )

              U = unique_s0.shape[0]
              averaged_targets = torch.zeros(U, device=device)
              counts = torch.zeros(U, device=device)

              averaged_targets.index_add_(0, inverse_indices, plan_targets)
              counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
              averaged_targets = averaged_targets / counts.clamp(min=1.0)
              
              averaged_targets = averaged_targets.detach()
              
              # running normalization
              batch_mean = averaged_targets.mean()
              batch_std = averaged_targets.std(unbiased=False) + 1e-8
              running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
              running_tgt_std = alpha * running_tgt_std + (1 - alpha) * batch_std
              #normalized_target = (averaged_targets - running_tgt_mean) / running_tgt_std
            

              # critic input
              s0_critic = (unique_s0 - c_mean) / c_std
              s0_critic = s0_critic.detach()
        
        # gradient step
        v_pred = critic(s0_critic)
        with torch.no_grad():
            pred_mean = v_pred.detach().mean()
            pred_std = v_pred.detach().std(unbiased=False)
        #loss = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)
        loss = F.smooth_l1_loss(v_pred, averaged_targets, beta=1.0)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Polyak update
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()

        if log_every > 0 and k % log_every == 0 and is_main:
            
            wandb.log({ "loss": running / log_every, 
                        "pred_mean": pred_mean.item(),
                        "pred_std": pred_std.item(),
                        "tgt_mean": running_tgt_mean.item(),
                        "tgt_std": running_tgt_std.item(),
                        "step": k})     
            
            print(
                f" step {k:>6}/{num_steps} "
                f"loss = {running / log_every:.10f}  "
                f"B_eff={B_eff}  U={U}  "
                f"pred_mean={pred_mean.item():.3f}  "
                f"pred_std={pred_std.item():.3f}  "
                f"tgt_mean={running_tgt_mean.item():.3f}  "
                f"tgt_std={running_tgt_std.item():.3f}"
            )
            running = 0.0

    # final save
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
         
        """
        q_stats = Q_Stats()
        q_stats.Q_mean = running_tgt_mean.item()
        q_stats.Q_std = running_tgt_std.item()
        save_Q_stats(q_stats, dataset_name, specific_dataset, task_id, new_step)
        """
        print("critic saved.")

    return running_tgt_mean.item(), running_tgt_std.item()




