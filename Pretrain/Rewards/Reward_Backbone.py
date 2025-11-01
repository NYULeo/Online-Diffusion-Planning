

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional
from Dataset import KitchenDataset, PointMazeDataset, get_dataset, get_env
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from Pretrain.utils import set_seed, SAStats
#from Critic.train_critic import get_CriticName
import torch.nn as nn
import pickle
from .nets import CategoricalReward, ScalarReward, gaussian_rewards
import os
from scipy.ndimage import gaussian_filter1d, convolve
from utils import cycle
import copy
from sympy import Predicate, factorint




def save_model(reward_net, reward_name, num_steps):
    reward_net.eval()
    net_dict = reward_net.state_dict()
    os.makedirs(f'./Pretrain/Rewards/{reward_name}/Models/', exist_ok=True)
    save_path = f'./Pretrain/Rewards/{reward_name}/Models/{reward_name}_{num_steps}.pkl'
    torch.save(net_dict, save_path)
    print(f"reward model save to {reward_name}_{num_steps}.pkl")

def load_model(reward_name, num_steps):
    load_path = f'./Pretrain/Rewards/{reward_name}/Models/{reward_name}_{num_steps}.pkl'
    #state_dict = torch.load(load_path, map_location='cpu')
    state_dict = torch.load(load_path, weights_only=True, map_location='cpu')
    return state_dict
"""
class Reward_Processor():
     def __init__(self, dataset_name, specific_dataset):
          critic_name = get_CriticName(dataset_name, specific_dataset)
          stats_name = critic_name.replace('.pt', '_stats.pkl')
          with open(stats_name, 'rb') as f:
                self.stats = pickle.load(f)
     
     def preprocess(self, obs, act):
          obs = self.stats.norm_obs(obs)
          act = self.stats.norm_act(act)
          return obs, act
"""
def Train_Dataset(dataset_name, specific_dataset: Optional[str] = None):
    if(dataset_name == 'kitchen'):
         data_1 = KitchenDataset('complete')
         data_2 = KitchenDataset('partial')
         data_3 = KitchenDataset('mixed')
         trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
         name = 'Kitchen_Reward'
         obs_dim = data_1.get_state_dim()
         act_dim = data_1.get_action_dim()
         return trajs, name, obs_dim, act_dim
     
    elif(dataset_name == 'pointmaze'):
         if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         elif(specific_dataset == 'large'):
              data = PointMazeDataset('large')
              name = '2DMaze_Reward_large'
         elif(specific_dataset == 'medium'):
              data = PointMazeDataset('medium')
              name = '2DMaze_Reward_medium'
         elif(specific_dataset == 'umaze'):
              data = PointMazeDataset('umaze')
              name = '2DMaze_Reward_umaze'
         else: 
              raise ValueError(f"Invalid dataset name: {specific_dataset}")
         obs_dim = data.get_state_dim()
         act_dim = data.get_action_dim()
         trajs = data.get_trajectories()
         return trajs, name, obs_dim, act_dim
    else:
         raise ValueError(f"Invalid dataset name: {dataset_name}")
         

class RewardDataset(Dataset):
    def __init__(self, trajs, sigma, reward_name, target_reward: Optional[float] = None):
            
        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        
        for traj in trajs:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]
        #act_all = np.concatenate(act_list, axis=0)  # [N, d_a]
        
        allowed_values = [0,1]
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
        
        transitions = []
        for traj in trajs:
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
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
        self.save_stats(reward_name)
    
    def save_stats(self, reward_name):
        stats_name =  str(reward_name) + '_stats.pkl'
        stats_dir = f'./Rewards/{reward_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )
    

def train_reward(dataset_name: str, batch_size, num_steps, lr, sigma, target_reward: Optional[float] = None, specific_dataset: Optional[str] = None):
    save_freq = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    print(f"Training reward approximator for {dataset_name} Dataset") 
    dataset = RewardDataset(trajs, sigma, reward_name, target_reward)
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8))
    
   
    #reward_net = Reward(obs_dim, act_dim).to(device)
    reward_net = ScalarReward(
        obs_dim,
        act_dim,
        hidden_units=1024,
        num_layers=5).to(device)

    optimizer = optim.Adam(reward_net.parameters(), lr = lr, weight_decay = 1e-5)
    total_loss = 0
    step = 0
    for i in range(num_steps):
           s, a, r = next(dataloader)
           s = s.to(device)
           a = a.to(device)
           r = r.to(device)
        
           # Predicted Reward
           optimizer.zero_grad()
           loss = reward_net.loss(s, a, r)  # r_batch in [0,1]
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
           step += 1

           if step % 100 == 0:
              avg_loss = total_loss / 100
              print(f"Step {step}, loss {avg_loss:.4f}")
              total_loss = 0

           if step % save_freq == 0:
              checkpoint = copy.deepcopy(reward_net)
              save_model(checkpoint, reward_name, step)
           


class test_dataset(Dataset):
    def __init__(self, trajs, sigma, Reward_name, target_reward: Optional[float] = None):
        stats_path = f'./Rewards/{Reward_name}/Stats/{Reward_name}_stats.pkl'
        with open(stats_path, 'rb') as f:
              self.stats = pickle.load(f)
        transitions = []
        allowed_values = [0,1]
        for traj in trajs:
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
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
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )
        
def test_Model(dataset_name, specific_dataset: Optional[str] = None, trajs: Optional[list] = None, sigma: float = 3, target_reward: Optional[float] = None, save_freq: int = 50, num_steps: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if(trajs is None): 
        train_Trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = RewardDataset(train_Trajs, sigma, reward_name, target_reward)
    else:
        _, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = test_dataset(trajs, sigma, reward_name, target_reward)
    print(f"Testing the reward model on {len(dataset)} samples")
    a = factorint(len(dataset))
    batch_size = int(np.min(list(a.keys())))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    num = save_freq
    while num <= num_steps:
         state_dict = load_model(reward_name, num)
         reward_net = ScalarReward(obs_dim, act_dim).to(device)
         reward_net.load_state_dict(state_dict)
         reward_net.eval()
         total_mean_loss = 0.0
         total_var = 0.0
         total_reward = 0.0
         for s, a, r in dataloader:
             s = s.to(device)
             a = a.to(device)
             r = r.to(device)
             pred = reward_net.predict(s, a)
             var = reward_net.variance(s, a)
             mean_loss = ((pred - r).abs()).mean()
             total_mean_loss += mean_loss.item()
             total_var += var.mean().item()
             total_reward += pred.mean().item()
         avg_mean_loss = total_mean_loss / len(dataloader)
         avg_var = total_var / len(dataloader)
         avg_reward = total_reward / len(dataloader)
         print(f"model {num}, Loss {avg_mean_loss:.4f}, Variance {avg_var:.4f}, Reward: {avg_reward:.4f}")

         num += save_freq

def get_pretrained_reward(dataset_name, checkpoints, specific_dataset: Optional[str] = None):
       _, name, obs_dim, act_dim  =  Train_Dataset(dataset_name, specific_dataset)
       reward_model_state_dict = load_model(name, checkpoints)
       return reward_model_state_dict, obs_dim, act_dim, name


def get_pretrained_reward_stats(Reward_name):
    stats_path = f'./Pretrain/Rewards/{Reward_name}/Stats/{Reward_name}_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats



def test_Single_Model(dataset_name, specific_dataset: Optional[str] = None, trajs: Optional[list] = None, sigma: float = 3, target_reward: Optional[float] = None, num: int = 10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if(trajs is None): 
        train_Trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = RewardDataset(train_Trajs, sigma, reward_name, target_reward)
    else:
        _, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = test_dataset(trajs, sigma, reward_name, target_reward)
    print(f"Testing the reward model on {len(dataset)} samples")
    a = factorint(len(dataset))
    batch_size = int(np.min(list(a.keys())))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    
    state_dict = load_model(reward_name, num)
    reward_net = ScalarReward(obs_dim, act_dim).to(device)
    reward_net.load_state_dict(state_dict)
    reward_net.eval()
    total_mean_loss = 0
    total_var = 0
    for s, a, r in dataloader:
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        mean = reward_net.predict(s, a)
        var = reward_net.variance(s, a)
        mean_loss = ((mean - r).abs()).mean()
        total_mean_loss += mean_loss.item()
        total_var += var.mean().item()
    avg_mean_loss = total_mean_loss / len(dataloader)
    avg_var = total_var / len(dataloader)
    print(f"model {num}, Loss {avg_mean_loss:.4f}, Variance {avg_var:.4f}")




    