import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Dataset import KitchenDataset, PointMazeDataset
from utils import *
import random
import copy
# Define the Gaussian forward dynamics model: inputs (s, a), outputs mean and log_std of s'
class TransitionKernel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, obs_dim)
        self.log_std_head = nn.Linear(hidden_dim, obs_dim)
   
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mu = self.mean_head(h)
        log_std = self.log_std_head(h)
        # Clamp log_std to reasonable range for numerical stability
        log_std = torch.clamp(log_std, min=-10.0, max=2.0)
        return mu, log_std

    def gaussian_nll(self, x, mu, log_std):
        var = torch.exp(2 * log_std)
        nll = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((x - mu) ** 2) / var
        return nll.sum(dim=-1).mean()


def save_model(kernel_net, kernel_name, num_steps):
    kernel_net.eval()
    net_dict =  kernel_net.state_dict()
    os.makedirs(f'./Transition_Kernel/{kernel_name}/Models/', exist_ok=True)
    save_path = f'./Transition_Kernel/{kernel_name}/Models/{kernel_name}_{num_steps}.pkl'
    torch.save(net_dict, save_path)
    print(f"Kernel model save to {kernel_name}_{num_steps}.pkl")

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

def train_kernel(dataset_name, specific_dataset: Optional[str] = None, batch_size = 256, lr = 1e-3, num_steps = 10000):
     # Prepare dataset and dataloader
     save_freq = 100
     print(f"Training kernel for {dataset_name} Dataset")
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     if(dataset_name == 'kitchen'):
         data_1 = KitchenDataset('complete')
         data_2 = KitchenDataset('partial')
         data_3 = KitchenDataset('mixed')
         trajectories = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
         kernel_name = 'Kitchen_Kernel.pkl'

     elif(dataset_name == 'pointmaze'):
         if(specific_dataset is None):
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         data = PointMazeDataset(specific_dataset)
         trajectories = data.get_trajectories()
         kernel_name = '2DMaze_Kernel.pkl'
     else:
         raise ValueError(f"Invalid dataset name: {dataset_name}")

     obs_dim = data_1.get_state_dim()
     act_dim = data_1.get_action_dim()
     dataset = KernelDataset(trajectories)
     loader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8))

     # Create model and optimiser
     model = TransitionKernel(obs_dim, act_dim).to(device)
     optimiser = optim.Adam(model.parameters(), lr)

     #total probability before training
     total_prob = total_pro(trajectories, model)
     print(f"Total Probability Before Training: {total_prob:.4f}")
     # Training loop

     model.train()
     step = 0
     total_nll = 0.0
     for i in range(num_steps):
          s, a, s_next = next(loader)
          s = s.to(device)
          a = a.to(device)
          s_next = s_next.to(device)

          mu, log_std = model(s, a)
          loss = model.gaussian_nll(s_next, mu, log_std)

          optimiser.zero_grad()
          loss.backward()
          optimiser.step()
          total_nll += loss.item() 
          
          if step % 10 == 0:
              avg_loss = total_nll / 10
              print(f"Step {step}, loss {avg_loss:.4f}")
              total_loss = 0

          if step % save_freq == 0:
              checkpoint = copy.deepcopy(model)
              save_model(checkpoint, model, step)
         
     
     model.eval()
     #total probability after training
     total_prob = total_pro(trajectories, model)
     print(f"Total Probability After Training: {total_prob:.4f}")
     #torch.save(model, kernel_name)
     


if __name__ == '__main__':  # pragma: no cover
    random.seed(1)
    train_kernel(dataset_name = 'kitchen', batch_size = 256, lr = 1e-3, epochs = 100)


 