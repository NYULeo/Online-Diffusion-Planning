from transition_kernel import TransitionKernel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from Dataset import KitchenDataset
from utils import *


# Build (s, a, s') transitions from your offline trajectories
class KernelDataset(Dataset):
    def __init__(self, trajectories):
        data = []
        for traj in trajectories:
            obs = np.asarray(traj['observations'])
            acts = np.asarray(traj['actions'])
            for t in range(len(acts)):
                s_t   = obs[t]
                a_t   = acts[t]
                s_tp1 = obs[t+1]
                data.append((s_t, a_t, s_tp1))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, s_next = self.data[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32)
        )

def train(dataset_name, batch_size, lr, epochs):
     # Prepare dataset and dataloader
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     if(dataset_name == 'kitchen'):
         kitchen_data_1 = KitchenDataset('complete')
         kitchen_data_2 = KitchenDataset('partial')
         kitchen_data_3 = KitchenDataset('mixed')
         trajectories = kitchen_data_1.get_trajectories() + kitchen_data_2.get_trajectories() + kitchen_data_3.get_trajectories()
     else:
         raise ValueError(f"Invalid dataset name: {dataset_name}")

     obs_dim = kitchen_data_1.get_state_dim()
     act_dim = kitchen_data_1.get_action_dim()
     dataset = KernelDataset(trajectories)
     loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

     # Create model and optimiser
     model = TransitionKernel(obs_dim, act_dim).to(device)
     optimiser = optim.Adam(model.parameters(), lr)

     #total probability before training
     total_prob = total_pro(trajectories, model)
     print(f"Total Probability Before Training: {total_prob:.4f}")
     # Training loop
     for epoch in range(epochs):
          total_nll = 0.0
          for s, a, s_next in loader:
               print(len(s))
               break
               s = s.to(device)
               a = a.to(device)
               s_next = s_next.to(device)

               mu, log_std = model(s, a)
               loss = model.gaussian_nll(s_next, mu, log_std)

               optimiser.zero_grad()
               loss.backward()
               optimiser.step()

               total_nll += loss.item() * len(s)
          break
          avg_nll = total_nll / len(dataset)
          if epoch % 10 == 0 or epoch == epochs - 1:
               print(f"Epoch {epoch}, negative log-likelihood: {avg_nll:.4f}")

     #total probability after training
     total_prob = total_pro(trajectories, model)
     print(f"Total Probability After Training: {total_prob:.4f}")

if __name__ == '__main__':  # pragma: no cover
    random.seed(1)
    train(dataset_name = 'kitchen', batch_size = 256, lr = 1e-3, epochs = 100)
