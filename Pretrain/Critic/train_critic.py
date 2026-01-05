import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from Dataset import get_dataset, get_env
from utils import set_seed, SAStats
from Critic.nets import Critic
import pickle
from scipy.ndimage import gaussian_filter1d
import os
from typing import Optional
from utils import cycle


def get_CriticName(env_name, specific_env):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High_Critic.pt'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium_Critic.pt'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed_Critic.pt'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
         if(specific_env == 'large'):
              return 'PointMaze_Large_Critic.pt'
         elif(specific_env == 'medium'):
              return 'PointMaze_Medium_Critic.pt'
         elif(specific_env == 'unmaze'):
              return 'PointMaze_Unmaze_Critic.pt'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: '{env_name}")

def reward_filter(obs, rews, goal):
    #target_goals = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5], [-2.5, 2.5]])
    target_goals = goal
    for i in range(1, len(obs)):
        goal_coord = np.floor(obs[i][:2]) + 0.5
        #goal_coord = np.round(goal_coord, 1)  
        if np.any(np.all(np.equal(goal_coord, target_goals), axis=1)):
            rews[i-1] = 1
        else:
            rews[i-1] = 0
    return rews

def save_critic(model, dataset_name, specific_dataset, step):
    model.eval()
    name = get_CriticName(dataset_name, specific_dataset)
    net_dict = model.state_dict()
    os.makedirs(f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
    save_path = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)
    print(f"critic model save to {name}.pkl")

def get_critic_model(dataset_name, specific_dataset, step):
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    name = get_CriticName(dataset_name, specific_dataset)
    path = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl'
    model_state_dict = torch.load(path, weights_only=True, map_location='cpu')
    return model_state_dict, obs_dim

def get_critic_stats(dataset_name, specific_dataset):
    name = get_CriticName(dataset_name, specific_dataset)
    path = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Stats/{name}_Critic_stats.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats 
"""
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(2-1)
"""
"""
class Critic_Processor():
     def __init__(self, dataset_name, speific_dataset):
          critic_name = get_CriticName(dataset_name, speific_dataset)
          stats_name = critic_name.replace('.pt', '_stats.pkl')
          with open(stats_name, 'rb') as f:
                self.stats = pickle.load(f)
    
     def preprocess(self, obs, act):
          obs = self.stats.norm_obs(obs)
          act = np.clip(act, -1.0, 1.0)
          return obs, act
"""
class CriticDataset(Dataset):
    def __init__(self, sigma: float, dataset_name: str, specific_dataset: str, goal: Optional[np.array] = None, target_reward: Optional[float] = None, horizon: int = 32, gamma: float = 0.99):
        # ----- gather raw obs/actions to fit stats -----
        data = get_dataset(dataset_name, specific_dataset)
        trajs = data.get_trajectories()
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
            #acts = traj['actions']  
            rews = traj['rewards']
            if( goal is not None):
                rews = reward_filter(obs, rews, goal)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
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
            
            """
            for t in range(len(obs)-1):
                obs_t = self.stats.norm_obs(obs[t+1])
                r_t   = rews[t]
                transitions.append((obs_t, r_t))
                Total += 1
            """
                
        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset)
    
    def save_stats(self, dataset_name, specific_dataset):
        name = get_CriticName(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Critic_stats.pkl'
        stats_dir = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.transitions)#

    def __getitem__(self, idx):
        s, r, s_next = self.transitions[idx]
        return (
            torch.tensor(s, dtype = torch.float32),
            torch.tensor(r, dtype = torch.float32),
            torch.tensor(s_next, dtype = torch.float32)
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
            #new_rews.append(np.sum(rews[t:]))
        return new_rews


def train_critic(dataset_name: str, specific_dataset: str, sigma: float, batch_size, num_steps, gamma, horizon, lr, tau, goal = None, target_reward = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device {device}")

    #get information
    dataset = CriticDataset(sigma, dataset_name, specific_dataset, goal, target_reward, horizon, gamma)
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
   
    #prepare training
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True))
    critic = Critic(obs_dim).to(device)
    target_critic = Critic(obs_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    optimizer = optim.Adam(critic.parameters(), lr = lr)
   
    print(f"Training critic for {dataset_name}-{specific_dataset}")
    for k in range(1, num_steps + 1):  # number of passes over dataset
           #s, r, s_next = next(dataloader)
           s, r, s_next = next(dataloader)
           #s, r = next(dataloader)
           s = s.to(device)
           r = r.to(device)
           s_next = s_next.to(device)

           # Compute target V-values
           with torch.no_grad():
              q_next = target_critic(s_next)
              target = r + ( (gamma**horizon) * q_next)
              #target = r 

           # Predicted V-values
           q_pred = critic(s)
           loss = ((q_pred - target) ** 2).mean()

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           

           """
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
            """
            
        
           if(k % 200 == 0):
                #target_critic.eval()
                #save_critic(target_critic, dataset_name, specific_dataset, k)
                critic.eval()
                save_critic(critic, dataset_name, specific_dataset, k)
                print(f"Checkpoint saved at step {k}")
                
    print(f"critic model saved")




if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    train_critic(dataset_name = 'pointmaze', 
                 specific_dataset = 'medium', 
                 sigma = 7.0, 
                 batch_size = 256, 
                 num_steps = 1000, 
                 gamma = 1.0, 
                 horizon = 32, 
                 lr = 1e-4, 
                 tau = 0.005,
                 goal = np.array([[-0.5, 0.5]], dtype = float),
                 target_reward = 1.0)





