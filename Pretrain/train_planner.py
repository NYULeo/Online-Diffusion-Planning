
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Dataset import KitchenDataset, PointMazeDataset
from Backbone import UNet1D, SDETrainer
from utils import set_seed

class PlannerDataset(Dataset):
    """Return trajectories as sequences of concatenated state–action pairs."""

    def __init__(self, trajectories, horizon):
        self.traj = trajectories
        self.horizon = horizon
        self.windows = []
        for traj in trajectories:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs, acts = obs[:L], acts[:L]
            sa_pairs = []
            for i in range(L):
                 sa_pairs.append(np.concatenate([obs[i], acts[i]]))
            # Slide a window of length `horizon` along the trajectory
            for start in range(0, L - horizon + 1):
                element = sa_pairs[start:start + horizon]
                element = np.array(element).flatten()
                self.windows.append(torch.from_numpy(element).float())
            

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def train_planner(dataset_name, specific_dataset, batch_size, horizon, num_epochs, lr):  # pragma: no cover
    """Run a small example demonstrating model instantiation and training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(dataset_name == 'kitchen'):
         data = KitchenDataset(specific_dataset)
         if(specific_dataset == 'complete'):
              model_name = 'Kitchen_High_Planner.pt'
         elif(specific_dataset == 'partial'):
              model_name = 'Kitchen_medium_Planner.pt'
         else:
              model_name = 'Kitchen_Low_Planner.pt'
    elif(dataset_name == 'pointmaze'):
           data = PointMazeDataset(specific_dataset)
           if(specific_dataset == 'large'):
              model_name = '2DMaze_Large_Planner.pt'
           elif(specific_dataset == 'medium'):
              model_name = '2DMaze_medium_Planner.pt'
           else:
              model_name = '2DMaze_nnmaze_Planner.pt'
    else:
         raise ValueError(f"Invalid Dataset Name: {dataset_name}")
    
    print(f"Training planner for {dataset_name}-{specific_dataset} Dataset]")
    state_dim = data.get_state_dim()
    action_dim = data.get_action_dim()
    sa_dim = state_dim + action_dim
    trajectories = data.get_trajectories()
    dataset = PlannerDataset(trajectories, horizon)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


    model = UNet1D(input_dim = sa_dim * horizon, base_channels=32).to(device)
    model.train()
    trainer = SDETrainer(model, device = device)
    optim = torch.optim.Adam(model.parameters(), lr)
    for epoch in range(num_epochs):
       for sa0 in dataloader:
            optim.zero_grad()
            loss = trainer.train_step(sa0)
            loss.backward()
            optim.step()
       print(f"Epoch {epoch}, loss = {loss.item():.4f}")
    model.eval()
    return model
    
    #torch.save(model.state_dict(), model_name) 
    #print(f"Planner model saved to {model_name}")

"""
if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     train_planner(dataset_name = 'kitchen', specific_dataset = 'complete', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    
"""



#train_planner(dataset_name = 'kitchen', specific_dataset = 'complete', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
