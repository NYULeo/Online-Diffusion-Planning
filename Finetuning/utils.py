import sys
import os

#from Finetuning.heatmap_plot import critic_heatmap
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
#from matplotlib import color_sequences
from scipy.special import j0
import torch
import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset
from Pretrain.utils import SAStats
from scipy.ndimage import gaussian_filter1d
from typing import TypedDict, List
from typing import Optional
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from Pretrain.Dataset import get_PlannerName
from typing import Tuple, Dict
#from Pretrain.Rewards.Reward_Backbone import Train_Dataset
from Pretrain.Transition_Kernel.Kernel_Backbone import count_files_in_folder
import copy
from Pretrain.Rewards.nets import SimpleReward
from torch.utils.data import DataLoader
import torch.optim as optim
from Pretrain.Transition_Kernel.Kernel_Net import MoGTransitionKernel, RobustTransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_total_mahalanobis_score, compute_log_density_mog, compute_log_density, compute_total_mahalanobis_score_mog
#from Pretrain.Rewards.Reward_Backbone import get_reward_name
from Pretrain.Dataset import KitchenDataset, PointMazeDataset, get_env, get_dataset, Planner_Processor
from gymnasium.vector import AsyncVectorEnv
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Critic.nets import Critic
from Pretrain.Dataset import get_dataset
import json
import torch.nn as nn

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray


def check_specific_dataset(dataset_name):
    if(dataset_name == 'kitchen'):
         return False
    elif(dataset_name == 'cube'):
         return True
    elif(dataset_name == 'pointmaze'):
         return True

def reward_name_converter(specific_dataset):
    if(specific_dataset == 'single-play' or specific_dataset == 'single-noise'):
        return 'single'
    elif(specific_dataset == 'double-play' or specific_dataset == 'double-noise'):
        return 'double'
    elif(specific_dataset == 'triple-play' or specific_dataset == 'triple-noise'):
        return 'triple'
    elif(specific_dataset == 'quadruple-play' or specific_dataset == 'quadruple-noise'):
        return 'quadruple'
    else:
        return specific_dataset

def spare_reward_prcocessor(rewards):
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

def get_planner(dataset_name, specific_dataset, step):
    name = getName(dataset_name, specific_dataset)
    path = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}/{name}_Planner_{str(step)}.pt"
    if not os.path.exists(path):
          raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, weights_only = True,map_location='cpu')
    #checkpoint = torch.load(checkpoint_path,  weights_only=True)
    return checkpoint['ema']

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
          if specific_env == 'medium_play':
               return 'AntMaze_MediumPlay'
          elif specific_env == 'umaze_diverse':
               return 'AntMaze_UmazeDiverse'
          elif specific_env == 'large_diverse':
               return 'AntMaze_LargeDiverse'
          elif specific_env == 'large_play':
               return 'AntMaze_LargePlay'
          elif specific_env == 'medium_diverse':
               return 'AntMaze_MediumDiverse'
          elif specific_env == 'umaze':
               return 'AntMaze_Umaze'
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
          elif specific_env == 'umaze':
               return 'AntMaze_Umaze'
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
          elif specific_env == 'umaze_dense':
               return 'PointMaze_UmazeDense'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env == 'open':
               return 'PointMaze_Open'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'antmaze'):
          if specific_env == 'medium_play':
               return 'AntMaze_MediumPlay'
          elif specific_env == 'umaze_diverse':
               return 'AntMaze_UmazeDiverse'
          elif specific_env == 'large_diverse':
               return 'AntMaze_LargeDiverse'
          elif specific_env == 'large_play':
               return 'AntMaze_LargePlay'
          elif specific_env == 'medium_diverse':
               return 'AntMaze_MediumDiverse'
          elif specific_env == 'umaze':
               return 'AntMaze_Umaze'
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
              raise ValueError(f"Invalid cube dataset name: {specific_env}")
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
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
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
         rews[rews == 1.0] = target_reward
         return rews
"""
class CriticDataset(Dataset):
    def __init__(self, trajs: List[TrajectoryDict], sigma: float, dataset_name: str, specific_dataset: str, step: int, goal: Optional[np.array] = None, target_reward: Optional[float] = None, horizon: int = 32, gamma: float = 0.99):
        # ----- gather raw obs/actions to fit stats -----
        if(dataset_name == 'pointmaze'):
            trajs = copy.deepcopy(trajs) 
            for traj in trajs:
                traj['observations'] = traj['observations'][:,:2]
        
        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            rews = traj['rewards']
            rews = spare_reward_prcocessor(rews)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if( goal is not None):
                rews = reward_filter(obs, rews, goal)
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            rews = gaussian_filter1d(rews, sigma)
            if(len(obs) > horizon):
               rews = self.reward_processor(rews, horizon, gamma)
               for t in range(len(obs)-horizon):
                   obs_t = self.stats.norm_obs(obs[t])
                   r_t   = rews[t]
                   obs_next_t = self.stats.norm_obs(obs[min(t+horizon, len(obs)-1)])
                   transitions.append((obs_t, r_t, obs_next_t))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, step)
    
    def save_stats(self, dataset_name, specific_dataset, step):
        name = getName(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, r, s_next = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
        )
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews
    
    def reward_processor(self, rews, horizon, gamma):
        new_rews = []
        for t in range(len(rews)):
            R = 0.0
            for i in range(t, min(t + horizon, len(rews))):
                R += (gamma**(i-t))*rews[i]
            new_rews.append(R)
        return new_rews
"""
def train_reward(trajs: List[TrajectoryDict], dataset_name: str, hidden_layers: int, hidden_dim: int, batch_size, num_steps, lr, min_lr, sigma, step, target_reward: Optional[float] = None, specific_dataset: Optional[str] = None, goal: Optional[np.array] = None, task_id: Optional[int] = None):
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
           loss = F.mse_loss(pred, r)
           loss.backward()
           torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm = 1.0)
           optimizer.step()
           scheduler.step()
           total_loss += loss.item()
           counter += 1
    save_reward_model(reward_net, dataset_name, specific_dataset, task_id, step)
    print(f"reward model saved")

"""       
def train_kernel(trajs: List[TrajectoryDict], dataset_name: str, specific_dataset: str, 
                 batch_size=256, lr=1e-3, num_steps=10000,
                 ensemble_size=10, λ_reg=1e-3, num_hidden_layers=2, hidden_dim=256, step: int = 0, constraint_type: str = 'mahalanobis', quantile: float = 0.95, x_generated_plans: Optional[list] = None):
    # Prepare dataset / dataloader
    print(f"Training kernel for {dataset_name}_{specific_dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=8))
    # Create ensemble of models
    ensemble = [RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers, hidden_dim).to(device) for _ in range(ensemble_size)]
    optimizers = [optim.Adam(m.parameters(), lr, weight_decay=1e-5) for m in ensemble]
    total_loss = 0.0
    for k in range(1, num_steps + 1):
        s, a, s_next = next(loader)
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)
        # For each model in ensemble, compute loss
        losses = []
        mus = []
        log_stds = []
        for m in ensemble:
            mu, log_std = m(s, a)
            mus.append(mu)
            log_stds.append(log_std)
            loss = m.gaussian_nll(s_next, mu, log_std)
            losses.append(loss)
        # optional: variance‐disagreement inflation
        # compute mean of mus
        mus_stack = torch.stack(mus, dim=0)  # (K, B, obs_dim)
        mu_mean = mus_stack.mean(dim=0)      # (B, obs_dim)
        # disagreement = average squared deviation
        disagreement = ((mus_stack - mu_mean.unsqueeze(0)) ** 2).mean(dim=0) 
        disagreement_detached = disagreement.detach()
        # inflate each model’s loss by penalizing small variance in high disagreement dims
        for i, m in enumerate(ensemble):
            penalty = (disagreement_detached / (torch.exp(2 * log_stds[i]) + m.noise_floor)).sum(dim=-1).mean()
            losses[i] = losses[i] + λ_reg * penalty

        # Backprop & optimize each model
        for i, (m, opt) in enumerate(zip(ensemble, optimizers)):
            opt.zero_grad()
            losses[i].backward()
            opt.step()

        avg_loss = sum(losses).item() / ensemble_size
        total_loss += avg_loss
    
    
    threshold = None
    if(x_generated_plans is not None):
        kernel_stats = get_kernel_stats(dataset_name, specific_dataset, step)
        threshold = compute_threshold(ensemble, kernel_stats, obs_dim, act_dim,  x_generated_plans, constraint_type, quantile, device)
        print(f"New Threshold for {constraint_type}: {threshold}")
    for idx, m in enumerate(ensemble):
         ckpt = copy.deepcopy(m).cpu()
         save_kernel_model(ckpt, dataset_name, specific_dataset, step, idx)
    print(f"Kernel model saved")
    return threshold
"""


"""
def train_kernel_mog(trajs: List[TrajectoryDict], dataset_name: str, specific_dataset: str,
                 batch_size=256, lr=1e-3, num_steps=10000,
                 ensemble_size=10, λ_reg=1e-3, num_modes: Optional[int] = 8,   num_hidden_layers=2, hidden_dim=256, kernel_noise_floor: Optional[float] = 1e-4, step: int = 0,  constraint_type: str = 'mahalanobis', quantile: float = 0.95, x_generated_plans: Optional[List] = None):
    # Prepare dataset / dataloader
    print(f"Training kernel for {dataset_name}_{specific_dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
    loader = cycle(DataLoader(dataset,
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  pin_memory=True, 
                                  num_workers=8, 
                                  persistent_workers=True, 
                                  prefetch_factor=4, 
                                  drop_last=True))
    # Create ensemble of models
    ensemble = [MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, kernel_noise_floor).to(device) for _ in range(ensemble_size)]
    optimizers = [optim.Adam(m.parameters(), lr, weight_decay=1e-5) for m in ensemble]
    total_loss = 0.0

    for k in range(1, num_steps + 1):
        s, a, s_next = next(loader)
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)
        # For each model in ensemble, compute loss
        losses = []
        #mus = []
        #log_stds = []
        for m in ensemble:
            mu, log_std, weights = m(s, a)
            loss = m.mog_nll(s_next, mu, log_std, weights)
            # === Optional: disagreement regularization ===
            # Average over modes for disagreement calculation
            mu_mean = mu.mean(dim=1)                    # (B, obs_dim)
            disagreement = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1).mean(dim=0)
            var = torch.exp(2 * log_std) + m.noise_floor
            penalty = (disagreement / (var.mean(dim=1) + 1e-6)).mean()
            loss = loss + λ_reg * penalty
            losses.append(loss)
        # Backprop
        
        for m, opt, loss in zip(ensemble, optimizers, losses):
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=5.0)
            opt.step()
        # Logging
        avg_loss = sum(loss.item() for loss in losses) / ensemble_size
        total_loss += avg_loss
        
       
    threshold = None
    if(x_generated_plans is not None):
        kernel_stats = get_kernel_stats(dataset_name, specific_dataset, step)
        threshold = compute_threshold_mog(ensemble, kernel_stats, obs_dim, act_dim,  x_generated_plans, constraint_type, quantile, device)
        print(f"New Threshold for {constraint_type}: {threshold}")
    for idx, m in enumerate(ensemble):
         ckpt = copy.deepcopy(m).cpu()
         save_kernel_model(ckpt, dataset_name, specific_dataset, step, idx)
    print(f"Kernel model saved")
    return threshold
"""

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
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
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
        rews[rews == 1.0] = target_reward
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
    optimizer = optim.Adam(critic.parameters(), lr = lr)
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
           total_loss += loss.item()

           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
           
           if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/1000}")
                total_loss = 0.0
            
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")

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

            if not np.all(np.isin(rews, [0.0, 1.0])):
                raise ValueError(f"Rewards must be either 0 or 1, but got {rews}")

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
        rews = rews.copy()
        rews[rews == 1.0] = target_reward
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
                gamma: float = 0.99,
                horizon: int = 32,
                sigma: Optional[float] = None,
                target_reward: float = 1.0,
                trajs: List[TrajectoryDict] = None,
                task_id: Optional[int] = None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if(traj['rewards'][-1] == 1):
            success_trajs.append(traj)
    return success_trajs

class PlannerDataset(Dataset):
    def __init__(self, trajs: List[TrajectoryDict], horizon: int, dataset_name: str, specific_dataset: str, cutoff_length: Optional[int] = None):
        self.trajs = copy.deepcopy(trajs)
        if(cutoff_length is not None):
            self.trajs = traj_cutoff(self.trajs, cutoff_length)
        print(f"total steps for Finetuning: {np.sum([len(traj['observations']) for traj in self.trajs])}")
        self.conditions = []
        self.horizon = horizon
        self.planner_processor = Planner_Processor(dataset_name, specific_dataset)
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
    if env_name == 'antmaze':
        return np.concatenate([
            s0['observation'],
            s0['achieved_goal']
        ])
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
              'rewards': np.asarray(spare_reward_prcocessor(rewards[env_idx].copy()))
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
     state_dict = get_planner(env_name, specific_env, checkpoint_step)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     elif(env_name == 'antmaze'):
         model = DiT1d(in_dim = d_s, emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif env_name == 'cube':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     
     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env)
     
     # <<< MODIFIED: Unique env reset seeds per process to prevent identical trajectories across GPUs
     reset_seeds = list(range(seed_base, seed_base + num_envs))
     
    
     total_steps = 0
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
                   #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
             # Check if all environments are done
            if all(done_envs):
                #print("All environments completed!")
                break
        
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
                      'rewards': np.asarray(rewards[env_idx].copy())
        })     
     vec_env.close()
     success_rate = check_success_rate(trajs)
     print(f"success rate: {success_rate:.2f}")
     if(goal_cell is None):
            expert_score = get_expert_score(env_name)
            score = get_normalized_score(trajs, expert_score)
     else:
            score = get_normalized_score(trajs)
     #save_trajs(trajs, env_name, specific_env, checkpoint_step)
     #print(f"Average Normalized Score: {score:.2f}")
     return trajs, score, success_rate, total_steps




"""
def rollout_parallel2(
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
    chunk_size=5,          # currently unused
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
    _, d_s, d_a = get_env(env_name, specific_env)

    def make_env():
        env, _, _ = get_env(env_name, specific_env)
        return env

    vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])

    # Load model
    state_dict = get_planner(env_name, specific_env, checkpoint_step)

    if env_name in ['kitchen', 'pointmaze', 'cube']:
        model = DiT1d(
            in_dim=(d_s + d_a), emb_dim=128, d_model=256,
            n_heads=256//64, depth=2, timestep_emb_type="fourier"
        ).to(device)
    elif env_name == 'antmaze':
        model = DiT1d(
            in_dim=d_s, emb_dim=128, d_model=256,
            n_heads=256//64, depth=2, timestep_emb_type="fourier"
        ).to(device)
    else:
        raise ValueError(f"Invalid Environment: {env_name}")

    model.load_state_dict(state_dict)
    model.eval()

    planner_processor = Planner_Processor(env_name, specific_env)
    reset_seeds = list(range(seed_base, seed_base + num_envs))

    def run_rollout(options_list):
      
        nonlocal total_steps

        obs, _ = vec_env.reset(seed=reset_seeds, options=options_list)
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
                                              for k in range(len(x))]
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
                'observations': np.asarray(observations[env_idx]),
                'actions': np.asarray(acts[env_idx]),
                'rewards': np.asarray(spare_reward_prcocessor(rewards[env_idx].copy()))  # FIXED
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
        opt = {"task_id": task_id}
        run_rollout([opt] * num_envs)

    vec_env.close()

    valid, success_rate = checktrajs(trajs)
    print(f"valid: {valid}, success rate: {success_rate:.2f}")

    if goal_cell is None:
        expert_score = get_expert_score(env_name)
        score = get_normalized_score(trajs, expert_score)
    else:
        score = get_normalized_score(trajs)

    return trajs, score, success_rate, total_steps
"""


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
    state_dict = get_planner(env_name, specific_env, checkpoint_step)

    if env_name in ['kitchen', 'pointmaze', 'cube']:
        model = DiT1d(
            in_dim=(d_s + d_a), emb_dim=128, d_model=256,
            n_heads=256//64, depth=2, timestep_emb_type="fourier"
        ).to(device)
    elif env_name == 'antmaze':
        model = DiT1d(
            in_dim=d_s, emb_dim=128, d_model=256,
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
                'rewards': np.asarray(spare_reward_prcocessor(rewards[env_idx].copy()))  # FIXED
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
        if(traj['rewards'][-1] == 1):
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

"""
def compute_threshold_mahalanobis_mog(kernels, dataloader, quantile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_D2_total = []
    for i, (s, a, s_next) in enumerate(dataloader):
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)
        #compute total mahalanobis distance
        with torch.no_grad():
            D2_total = compute_total_mahalanobis_score_mog(kernels, s, a, s_next)
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
"""

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

"""
def compute_threshold_log_prob_mog(kernels, dataloader, quantile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_log_density_total = []
    for i, (s, a, s_next) in enumerate(dataloader):
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)
        #compute total mahalanobis distance
        with torch.no_grad():
            log_density_total = compute_log_density_mog(kernels, s, a, s_next)
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
"""

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