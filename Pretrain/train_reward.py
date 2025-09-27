from typing import Optional
from Dataset import KitchenDataset, PointMazeDataset, get_dataset, get_env
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from utils import set_seed, SAStats
from Critic.train_critic import get_CriticName
import torch.nn as nn
import pickle
from Rewards.nets import CategoricalReward, ScalarReward, gaussian_rewards
import os
from scipy.ndimage import gaussian_filter1d, convolve

    

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
    def __init__(self, trajs, sigma, reward_name):
            
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
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
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
        stats_dir = './Rewards/Stats/'
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


def train_reward(dataset_name: str, batch_size, epochs, lr, sigma, specific_dataset: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    print(f"Training reward approximator for {dataset_name} Dataset") 
    dataset = RewardDataset(trajs, sigma, reward_name)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    
   
    #reward_net = Reward(obs_dim, act_dim).to(device)
    reward_net = ScalarReward(
        obs_dim,
        act_dim,
        hidden_units=1024,
        num_layers=5).to(device)

    optimizer = optim.Adam(reward_net.parameters(), lr = lr, weight_decay = 1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
           optimizer.zero_grad()
           s, a, r = batch
           s = s.to(device)
           a = a.to(device)
           r = r.to(device)
        
           # Predicted Reward
           loss = reward_net.loss(s, a, r)  # r_batch in [0,1]
           loss.backward()
           optimizer.step()
           total_loss += loss.item()

        if epoch % 10 == 0:
              print(f"Epoch {epoch}, loss {total_loss/len(dataloader):.4f}")
    
    reward_net.eval()
    os.makedirs('./Rewards/Models/', exist_ok=True)
    save_path = f'Rewards/Models/{reward_name}.pkl'
    torch.save(reward_net, save_path)
    print(f"reward model save to {reward_name}")



if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    train_reward(
    dataset_name = 'kitchen',  
    batch_size=1024, 
    epochs=50,   
    lr=1e-4,
    sigma=3)



