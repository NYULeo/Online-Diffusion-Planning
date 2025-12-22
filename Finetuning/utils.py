import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
#from matplotlib import color_sequences
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
import seaborn as sns
from Pretrain.Dataset import get_PlannerName
from typing import Tuple


class Lambda:
    def __init__(self, lam: float, beta: float, eta_lam: float):
        self.lam = lam
        self.base_lam = lam
        self.beta = beta
        self.eta_lam = eta_lam
    
    def update(self, C):
        self.lam = np.maximum(self.base_lam, self.lam + (self.eta_lam * C))
        #self.lam = np.clip(self.lam, 0.0, 5.0)
    
    def set_lam(self, lam: float):
        self.lam = lam

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

def get_pretrained_planner(dataset_name, specific_dataset, steps):
      planner_name = get_PlannerName(dataset_name, specific_dataset)
      checkpoint_path = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}/{planner_name}_{steps}.pt"
      if not os.path.exists(checkpoint_path):
          raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, weights_only = True,map_location='cpu')
      #checkpoint = torch.load(checkpoint_path,  weights_only=True)
      return checkpoint['ema']

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


