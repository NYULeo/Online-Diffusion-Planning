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
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from Pretrain.Dataset import KitchenDataset, PointMazeDataset, get_env, Planner_Processor
from gymnasium.vector import AsyncVectorEnv
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Critic.nets import Critic
from Pretrain.Dataset import get_dataset
import json

def check_specific_dataset(dataset_name):
    if(dataset_name == 'kitchen'):
         return False
    elif(dataset_name == 'pointmaze'):
         return True

def spare_reward_prcocessor(rewards):
    Temp = []
    for i in range(1, len(rewards)):
        if(rewards[i] == rewards[i-1]+1):
            Temp.append(i)
    new_rewards = [0]*len(rewards)
    for i in range(len(rewards)):
        if(i in Temp):
            new_rewards[i] = 1
        else:
            new_rewards[i] = 0
    return np.array(new_rewards, dtype = np.float64) 

def reward_filter(obs, rews, goal):
    #target_goals = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5], [-2.5, 2.5]])
    for i in range(1, len(obs)):
        pos = obs[i][:2] 
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        #goal_coord = np.asarray(goal_coord, dtype=np.float32).reshape(-1)  
        dist = np.linalg.norm(pos - g) 
        if (dist < 0.5):
            rews[i-1] = 1
        else:
            rews[i-1] = 0
    return rews

def save_reward_model(reward_net, dataset_name, specific_dataset, step):
    reward_net.eval()
    name = getName(dataset_name, specific_dataset)
    net_dict = reward_net.state_dict()
    if(check_specific_dataset(dataset_name)):
          os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
          save_path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{name}_Reward_{str(step)}.pkl'
    else: 
          os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Models/', exist_ok=True)
          save_path = f'./Finetuning/Rewards/{dataset_name}/Models/{name}_Reward_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)

def save_kernel_model(kernel_net, dataset_name, specific_dataset, step, ensemble_idx):
    kernel_net.eval()
    name = getName(dataset_name, specific_dataset)
    net_dict = kernel_net.state_dict()
    if(check_specific_dataset(dataset_name)):
          os.makedirs(f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}', exist_ok=True)
          save_path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    else: 
          os.makedirs(f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}', exist_ok=True)
          save_path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    torch.save(net_dict, save_path)
    #print(f"Kernel model save to {name}_{str(step)}_{str(ensemble_idx)}.pkl")

def get_reward_model(dataset_name, specific_dataset, step):
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    name = getName(dataset_name, specific_dataset)
    if(check_specific_dataset(dataset_name)):
         path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{name}_Reward_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rewards/{dataset_name}/Models/{name}_Reward_{str(step)}.pkl'
    model_state_dict = torch.load(path, weights_only=True, map_location='cpu')
    return model_state_dict, obs_dim, act_dim

def get_reward_stats(dataset_name, specific_dataset, step):
    name = getName(dataset_name, specific_dataset)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/{name}_Reward_stats_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rewards/{dataset_name}/Stats/{name}_Reward_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats  

def get_kernel(dataset_name, specific_dataset, step):
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    name = getName(dataset_name, specific_dataset)
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
    name = getName(dataset_name, specific_dataset)
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

def save_critic(model, dataset_name, specific_dataset, step):
    model.eval()
    name = getName(dataset_name, specific_dataset)
    net_dict = model.state_dict()
    os.makedirs(f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
    save_path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)
    print(f"critic model save to {name}_{str(step)}.pkl")

def get_critic_model(dataset_name, specific_dataset, step):
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    if(dataset_name == 'pointmaze'):
         obs_dim = obs_dim - 2
    name = getName(dataset_name, specific_dataset)
    path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl'
    model_state_dict = torch.load(path, weights_only=True, map_location='cpu')
    return model_state_dict, obs_dim

def get_critic_stats(dataset_name, specific_dataset, step):
    name = getName(dataset_name, specific_dataset)
    path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/{name}_Critic_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats 

def save_trajs(trajs, env_name, specific_env, step):
    os.makedirs(f'./Finetuning/Rollouts/{env_name}/{specific_env}/', exist_ok=True)
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{str(step)}.pkl'
    with open(save_path, 'wb') as f:
         pickle.dump(trajs, f)
    print(f"trajectories saved")

def get_trajs(env_name, specific_env, step):
    path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{str(step)}.pkl'
    with open(path, 'rb') as f:
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

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray

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
            for t in range(len(acts)):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                data.append((s_t, a_t, s_tp1))
         self.data = data
         self.save_stats(dataset_name, specific_dataset, step)
    
    def save_stats(self, dataset_name, specific_dataset, step):
        name = getName(dataset_name, specific_dataset)
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
    def __init__(self, trajs: List[TrajectoryDict], sigma: float, dataset_name: str, specific_dataset: str, step: int, goal: Optional[np.array] = None, target_reward: Optional[float] = None):
            
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
        allowed_values = [0,1]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            acts = traj['actions']
            rews = traj['rewards']
            #rews = spare_reward_prcocessor(rews)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if( goal is not None):
                rews = reward_filter(obs, rews, goal)
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            rews = gaussian_filter1d(rews, sigma)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, step)
    
    def save_stats(self, dataset_name, specific_dataset, step):
        name = getName(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Reward_stats_{str(step)}.pkl'
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
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

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
        allowed_values = [0,1]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            rews = traj['rewards']
            #rews = spare_reward_prcocessor(rews)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if( goal is not None):
                rews = reward_filter(obs, rews, goal)
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            rews = gaussian_filter1d(rews, sigma)
            rews = self.reward_processor(rews, horizon, gamma)
            for t in range(len(obs)-horizon):
                obs_t = self.stats.norm_obs(obs[t])
                r_t   = rews[t]
                obs_next_t = self.stats.norm_obs(obs[t + horizon])
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

def train_reward(trajs: List[TrajectoryDict], dataset_name: str, batch_size, num_steps, lr, sigma, step, target_reward: Optional[float] = None, specific_dataset: Optional[str] = None, goal: Optional[np.array] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    print(f"Training reward approximator for {dataset_name}_{specific_dataset} Dataset") 
    dataset = RewardDataset(trajs, sigma, dataset_name, specific_dataset, step, goal, target_reward)
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8))
    reward_net = SimpleReward(obs_dim, act_dim).to(device)
    optimizer = optim.AdamW(reward_net.parameters(), lr = lr, weight_decay = 1e-4)
    total_loss = 0
    counter = 0
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
           optimizer.step()
           total_loss += loss.item()
           counter += 1
    save_reward_model(reward_net, dataset_name, specific_dataset, step)
    print(f"reward model saved")
           
def train_kernel(trajs: List[TrajectoryDict], dataset_name: str, specific_dataset: str,
                 batch_size=256, lr=1e-3, num_steps=10000,
                 ensemble_size=10, λ_reg=1e-3, num_hidden_layers=2, step: int = 0):
    # Prepare dataset / dataloader
    print(f"Training kernel for {dataset_name}_{specific_dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=8))
    # Create ensemble of models
    ensemble = [RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers).to(device) for _ in range(ensemble_size)]
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

    for idx, m in enumerate(ensemble):
         ckpt = copy.deepcopy(m).cpu()
         save_kernel_model(ckpt, dataset_name, specific_dataset, step, idx)
    print(f"Kernel model saved")

def train_critic(trajs: List[TrajectoryDict], dataset_name: str, specific_dataset: str, sigma: float, batch_size, num_steps, gamma, horizon, lr, tau, step: int, goal = None, target_reward = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #get information
    dataset = CriticDataset(trajs, sigma, dataset_name, specific_dataset, step, goal, target_reward, horizon, gamma)
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    
    if(dataset_name == 'pointmaze'):
         obs_dim = obs_dim - 2
    #prepare training
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True))
    critic = Critic(obs_dim).to(device)
    critic.train()
    target_critic = Critic(obs_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    optimizer = optim.Adam(critic.parameters(), lr = lr)

    print(f"Training critic for {dataset_name}-{specific_dataset}")
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, r, s_next = next(dataloader)
           s = s.to(device)
           r = r.to(device)
           s_next = s_next.to(device)

           # Compute target Q-values
           with torch.no_grad():
              q_next = target_critic(s_next)
              target = r + ( (gamma**horizon) * q_next)

           # Predicted V-values
           q_pred = critic(s)
           loss = ((q_pred - target) ** 2).mean()

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, step)
    print(f"critic model saved")

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
    if(env_name == 'antmaze'):
        return np.concatenate([
               s0['observation'],
               s0['achieved_goal']
           ])
    else:
        return s0['observation']

def rollout_parallel(env_name, specific_env, horizon = 32, steps_T = 50, num_karras = 10, eta = 0.8, episode_length = 4000, checkpoint_step = 1000000, num_envs=8, goal_cell = None, start_cells = None, device: torch.device = None, seed_base: int = 0):
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
     save_trajs(trajs, env_name, specific_env, checkpoint_step)
     #print(f"Average Normalized Score: {score:.2f}")
     return trajs, score, total_steps

def load_hyperparameters(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

