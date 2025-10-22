import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset
from Pretrain.utils import SAStats
from scipy.ndimage import gaussian_filter1d
from typing import TypedDict, List
from Pretrain.Dataset import Planner_Processor
from typing import Optional
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Lambda:
    def __init__(self, lam: float, beta: float, eta_lam: float):
        self.lam = lam
        self.beta = beta
        self.eta_lam = eta_lam
    def update(self, C):
        delta = F.softplus(torch.tensor([0.0], requires_grad = False), beta = self.beta)
        self.lam = np.maximum(0.0, self.lam + self.eta_lam * (C - delta.item()))
    def get_lam(self):
        return self.lam

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray

def getName(env_name, specific_env):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'kitchen_high'
          elif(specific_env == 'partial'):
               return 'kitchen_medium'
          elif(specific_env == 'mixed'):
               return 'kitchen_mixed'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
          if specific_env == 'open_dense':
               return '2DMaze_openDense'
          elif specific_env == 'umaze':
               return '2DMaze_umaze'
          elif specific_env == 'large_dense':
               return '2DMaze_largeDense'
          elif specific_env== 'medium':
               return '2DMaze_medium'
          elif specific_env == 'umaze_dense':
               return '2DMaze_umazeDense'
          elif specific_env == 'large':
               return '2DMaze_large'
          elif specific_env == 'open':
               return '2DMaze_open'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'antmaze'):
          if specific_env == 'medium_play':
               return 'antMaze_mediumPlay'
          elif specific_env == 'umaze_diverse':
               return 'antMaze_umazeDiverse'
          elif specific_env == 'large_diverse':
               return 'antMaze_largeDiverse'
          elif specific_env == 'large_play':
               return 'antMaze_largePlay'
          elif specific_env == 'medium_diverse':
               return 'antMaze_mediumDiverse'
          elif specific_env == 'umaze':
               return 'antMaze_umaze'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: '{env_name}")

class KernelDataset(Dataset):
    def __init__(self, trajectories: List[TrajectoryDict], dataset_name: str, specific_dataset: str):
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
            for t in range(len(acts)):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                data.append((s_t, a_t, s_tp1))
         self.data = data
         self.save_stats(dataset_name, specific_dataset)
    
    def save_stats(self, dataset_name, specific_dataset):
        name = getName(dataset_name, specific_dataset)
        stats_name =  str(name) + '_kernel_stats.pkl'
        stats_dir = f'./Results/{dataset_name}/{specific_dataset}/Kernel_Stats/'
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
    def __init__(self, trajs: List[TrajectoryDict], sigma: float, dataset_name: str, specific_dataset: str):
            
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
        
        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            acts = traj['actions']
            rews = traj['rewards']
            rews = gaussian_filter1d(rews, sigma)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset)
    
    def save_stats(self, dataset_name, specific_dataset):
        name = getName(dataset_name, specific_dataset)
        stats_name =  str(name) + '_reward_stats.pkl'
        stats_dir = f'./Results/{dataset_name}/{specific_dataset}/Reward_Stats/'
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

class PlannerDataset(Dataset):
    def __init__(self, trajs: List[TrajectoryDict], horizon: int, dataset_name: str, specific_dataset: str):
        self.trajs = trajs
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
        self.learning_rates = []
        os.makedirs(save_dir, exist_ok=True)

    def log_reward(self, step: int, reward: float, lr: Optional[float] = None):
        self.steps.append(step)
        self.rewards.append(reward)
        if lr is not None:
            self.learning_rates.append(lr)

    def save_logs(self, filename: str = "reward_logs.pkl"):
        data = {
            'steps': self.steps,
            'rewards': self.rewards,
            'learning_rates': self.learning_rates
        }
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Reward logs saved to {save_path}")

    def plot_reward_curve(self,
                          save_path: Optional[str] = None,
                          title: str = "Finetuning Reward Curve",
                          show_lr: bool = False,
                          smooth_window: int = 50):
        if not self.rewards:
            print("No reward data to plot!")
            return

        fig, ax1 = plt.subplots(figsize=(12, 8))
        steps = np.array(self.steps)
        rewards = np.array(self.rewards)

        ax1.plot(steps, rewards, alpha=0.3, color='blue', label='Raw Reward')

        if len(rewards) > smooth_window:
            smoothed = self._smooth_curve(rewards, smooth_window)
            ax1.plot(steps, smoothed, color='red', linewidth=2,
                     label=f'Smoothed Reward (window={smooth_window})')

        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if show_lr and self.learning_rates:
            ax2 = ax1.twinx()
            ax2.plot(steps, self.learning_rates, color='green', alpha=0.7, label='Learning Rate')
            ax2.set_ylabel('Learning Rate', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')

        plt.title(title, fontsize=14, fontweight='bold')
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