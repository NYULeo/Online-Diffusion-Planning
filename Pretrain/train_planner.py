
from diffusion_transformer import DiffusionSDETrainer, ScoreModel
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Dataset import KitchenDataset
from diffusion_unet import UNet1D, SDETrainer
import random


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


def train(batch_size, horizon, num_epochs, lr):  # pragma: no cover
    """Run a small example demonstrating model instantiation and training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kitchen_data = KitchenDataset(name='complete')
    state_dim = kitchen_data.get_state_dim()
    action_dim = kitchen_data.get_action_dim()
    sa_dim = state_dim + action_dim
    trajectories = kitchen_data.get_trajectories()
    dataset = PlannerDataset(trajectories, horizon)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


    model = UNet1D(input_dim=feature_dim, base_channels=32).to(device)
    trainer = SDETrainer(model, device = device)
    optim = torch.optim.Adam(model.parameters(), lr)
    for epoch in range(num_epochs):
       for sa0 in dataloader:
            optim.zero_grad()
            loss = trainer.train_step(sa0)
            loss.backward()
            optim.step()
       print(f"Epoch {epoch}, loss = {loss.item():.4f}")



def example_usage():  # pragma: no cover
    """Demonstrate offline training with random data."""
    # Generate random state–action data of dimension 40
    B = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = 40
    x0 = torch.randn(B, feature_dim, device = device)
    model = UNet1D(input_dim=feature_dim, base_channels=32).to(device)
    trainer = SDETrainer(model, device = device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(10):
        opt.zero_grad()
        loss = trainer.train_step(x0)
        loss.backward()
        opt.step()
        print(f"step {step}, loss={loss.item():.4f}")


if __name__ == '__main__':  # pragma: no cover
    random.seed(1)
    example_usage()
"""
if __name__ == '__main__':  # pragma: no cover
    train(batch_size = 8, horizon = 32, num_epochs = 100, lr = 3e-4)
"""


            
