import torch
import numpy as np
from numpy import shape
from Pretrain.Rewards.nets import ScalarReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from torch import nn
import torch
import os
import pickle
from torch.utils.data import Dataset
from Pretrain.utils import SAStats
from scipy.ndimage import gaussian_filter1d
from typing import TypedDict, List


class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    gamma: float
    device: str
    d_s: int
    d_a: int

class TotalReward(nn.Module):
    def __init__(self, config: RewardConfig, reward_net: ScalarReward, kernel_net: RobustTransitionKernel):
        super().__init__()
        self.config = config
        self.reward_net = reward_net.to(self.config.device)
        self.kernel_net = kernel_net.to(self.config.device)
    
    def sigmoid(self, s, a, s_next):
        mu, log_std = self.kernel_net.forward(self.kernel_net, s, a, s_next)
        log_prob = self.kernel_net.log_prob(s_next, mu, log_std)
        x = self.config.min_log_prob - log_prob
        c = torch.softplus(x, beta = self.config.beta)
        return c

    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        C = 0.0
        for i in range(H-1):
            r = self.reward_net.predict(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0))
            var = self.reward_net.variance(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0))
            c = self.sigmoid(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0), x[i+1][:self.config.d_s].unsqueeze(0))
            C += c
            total_reward += (r.squeeze(0) + self.config.gamma * var.squeeze(0)) - (lam * c)
        
        r = self.reward_net.predict(x[H-1][:self.config.d_s].unsqueeze(0), x[H-1][self.config.d_s:].unsqueeze(0))
        var = self.reward_net.variance(x[H-1][:self.config.d_s].unsqueeze(0), x[H-1][self.config.d_s:].unsqueeze(0))
        total_reward += (r.squeeze(0) + self.config.gamma * var.squeeze(0))
        total_reward = total_reward * (1/H)
        C = C * (1/(H-1))
        return total_reward, C

class Lambda:
    def __init__(self, lam: float):
        self.lam = lam
    def update(self, C, eta_lam: float):
        self.lam = np.max(0, self.lam + eta_lam * C)
    def get_lam(self):
        return self.lam

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray

# Build (s, a, s') transitions from your offline trajectories
class KernelDataset(Dataset):
    def __init__(self, trajectories: List[TrajectoryDict], kernel_name: str):
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
         self.save_stats(kernel_name)
    
    def save_stats(self, kernel_name):
        stats_name =  str(kernel_name) + '_stats.pkl'
        stats_dir = f'./Transition_Kernel/{kernel_name}/Stats/'
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
    def __init__(self, trajs: List[TrajectoryDict], sigma: float, reward_name: str):
            
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
        self.save_stats(reward_name)
    
    def save_stats(self, reward_name):
        stats_name =  str(reward_name) + '_stats.pkl'
        stats_dir = f'./Rewards/{reward_name}/Stats/'
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
    
