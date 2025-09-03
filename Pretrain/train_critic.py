import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from Dataset import get_dataset, get_env
from utils import set_seed, SAStats
import pickle

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

class CriticDataset(Dataset):
    def __init__(self, dataset_name, specific_dataset):
        data = get_dataset(dataset_name, specific_dataset)
        self.critic_name = get_CriticName(dataset_name, specific_dataset)
        self.trajs = data.get_trajectories()
        transitions = []
        
        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        for traj in self.trajs:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]
        
        
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean=obs_all.mean(axis=0)
        self.stats.obs_std =obs_all.std(axis=0)

        
        for traj in self.trajs:
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
            for t in range(len(acts)):
                s_t   = self.stats.norm_obs(obs[t])
                a_t   = self.stats.norm_act(acts[t])
                r_t   = rews[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])  if t < (len(acts)-1) else np.zeros_like(s_t)
                a_tp1 = self.stats.norm_act(acts[t+1]) if t < (len(acts)-1) else np.zeros_like(a_t)
                done_t = 1.0 if t == (len(acts)-1) else 0.0
                transitions.append((s_t, a_t, r_t, s_tp1, a_tp1, done_t))
        
        self.transitions = transitions
        self.save_stats()
    
    def save_stats(self):
        stats_name = self.critic_name.replace('.pt', '_stats.pkl')
        with open(stats_name, 'wb') as f:
              pickle.dump(self.stats, f)


    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r, s_next, a_next, d = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(a_next, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )
    


def train_critic(dataset_name: str, specific_dataset: str, batch_size, epochs, gamma, lr, tau):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    #get information
    model_name = get_CriticName(dataset_name, specific_dataset)
    dataset = CriticDataset(dataset_name, specific_dataset)
    env, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
   
    #prepare training
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    critic = Critic(obs_dim, act_dim).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    optimizer = optim.Adam(critic.parameters(), lr = lr)

    print(f"Training critic for {dataset_name}-{specific_dataset}")
    for epoch in range(epochs):  # number of passes over dataset
       total_loss = 0
       for batch in dataloader:
           s, a, r, s_next, a_next, d = batch
           s = s.to(device)
           a = a.to(device)
           r = r.to(device)
           s_next = s_next.to(device)
           a_next = a_next.to(device)
           d = d.to(device)

           # Compute target Q-values
           with torch.no_grad():
              q_next = target_critic(s_next, a_next)
              target = r + gamma * (1 - d) * q_next

           # Predicted Q-values
           q_pred = critic(s, a)
           loss = ((q_pred - target) ** 2).mean()

           optimizer.zero_grad()
           loss.backward()
           total_loss += loss.item()
           optimizer.step()

           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
       avg_loss = total_loss / len(dataloader)
       if epoch % 10 == 0:
              print(f"Epoch {epoch+10}, loss {avg_loss:.4f}")
    
    critic.eval()
    torch.save(critic.state_dict(), model_name) 
    print(f"critic model save to {model_name}")


if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    train_critic(
    dataset_name = 'kitchen', 
    specific_dataset = 'complete', 
    batch_size=1024, 
    epochs=50,  
    gamma=0.99, 
    lr=1e-3, 
    tau = 0.005)


