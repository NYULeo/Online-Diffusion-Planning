
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Backbone import UNet1D, SDETrainer
from Dataset import get_dataset, get_env
import pickle
from utils import SAStats, set_seed

def get_PlannerName(env_name, specific_env):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High_Planner.pt'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium_Planner.pt'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed_Planner.pt'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
         if(specific_env == 'large'):
              return 'PointMaze_Large_Planner.pt'
         elif(specific_env == 'medium'):
              return 'PointMaze_Medium_Planner.pt'
         elif(specific_env == 'unmaze'):
              return 'PointMaze_Unmaze_Planner.pt'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: '{env_name}")


class PlannerDataset(Dataset):
    def __init__(self, dataset_name, specific_dataset, horizon):
        data = get_dataset(dataset_name, specific_dataset)
        self.planner_name = get_PlannerName(dataset_name, specific_dataset)
        self.traj = data.get_trajectories()
        self.horizon = horizon
        self.windows = []

        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        for traj in self.traj:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]
        act_all = np.concatenate(act_list, axis=0)  # [N, d_a]

        #get stats
        self.stats = SAStats()
        self.stats.obs_mean=obs_all.mean(axis=0)
        self.stats.obs_std =obs_all.std(axis=0)
        self.stats.act_min =act_all.min(axis=0)
        self.stats.act_max =act_all.max(axis=0)

        # ----- build normalized sliding windows -----
        for traj in self.traj:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))

            # per-step normalize then concat [s_t, a_t]
            sa_pairs = []
            for t in range(L):
                s_norm = self.stats.norm_obs(obs[t])
                a_norm = self.stats.norm_act(acts[t])
                sa_pairs.append(np.concatenate([s_norm, a_norm], axis=0))

            # sliding horizon, then flatten to 1D
            for start in range(0, L - horizon + 1):
                segment = np.array(sa_pairs[start:start + horizon])  # [H, d_s+d_a]
                self.windows.append(torch.from_numpy(segment.flatten()).float())
        
        self.save_stats()

    def save_stats(self):
        stats_name = self.planner_name.replace('.pt', '_stats.pkl')
        with open(stats_name, 'wb') as f:
              pickle.dump(self.stats, f)
 

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
 

class Planner_Processor():
     def __init__(self, dataset_name, speific_dataset):
          Planner_name = get_PlannerName(dataset_name, speific_dataset)
          stats_name = Planner_name.replace('.pt', '_stats.pkl')
          with open(stats_name, 'rb') as f:
                self.stats = pickle.load(f)
    
     def preprocess(self, obs):
          obs = self.stats.norm_obs(obs)
          return obs
     
     def postprocess(self, act):
          act = self.stats.denorm_act(act)
          return  act
         

def train_planner(dataset_name, specific_dataset, horizon, batch_size, num_epochs, lr):  # pragma: no cover
    """Run a small example demonstrating model instantiation and training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env, state_dim, action_dim = get_env(dataset_name, specific_dataset)
    dataset = PlannerDataset(dataset_name, specific_dataset, horizon)
    model_name = dataset.planner_name
    

    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    model = UNet1D(input_dim = ((state_dim + action_dim) * horizon)).to(device)
    model.train()
    trainer = SDETrainer(model, device = device)
    optim = torch.optim.AdamW(model.parameters(), lr, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs)
    print(f"Training planner for {dataset_name}-{specific_dataset} Dataset]")
    for epoch in range(num_epochs):
       total_loss = 0
       num_batches = 0
       for sa0 in dataloader:
            optim.zero_grad()
            loss = trainer.train_step(sa0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            num_batches += 1
       scheduler.step()
       avg_loss = total_loss / num_batches
       print(f"Epoch {epoch}, avg_loss = {avg_loss:.4f}")
    model.eval()
    torch.save(model.state_dict(), model_name) 
    print(f"Planner model saved to {model_name}")
    return model



if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'kitchen'
     specific_dataset = 'complete'
     horizon = 32
     train_planner(dataset_name = dataset_name, specific_dataset = specific_dataset, horizon = horizon, batch_size = 128, num_epochs = 100, lr = 2e-4)
    

