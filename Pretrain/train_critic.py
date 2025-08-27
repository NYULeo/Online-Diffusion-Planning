from Dataset import KitchenDataset
import random
from torch.utils.data import Dataset, DataLoader
from Critic import QNet
import torch
import torch.optim as optim
import numpy as np


class CriticDataset(Dataset):
    def __init__(self, trajs):
        transitions = []
        for traj in trajs:
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
            for t in range(len(acts)):
                s_t   = obs[t]
                a_t   = acts[t]
                r_t   = rews[t]
                s_tp1 = obs[t+1]  if t < (len(acts)-1) else np.zeros_like(s_t)
                a_tp1 = acts[t+1] if t < (len(acts)-1) else np.zeros_like(a_t)
                done_t = 1.0 if t == (len(acts)-1) else 0.0
                transitions.append((s_t, a_t, r_t, s_tp1, a_tp1, done_t))

        self.transitions = transitions

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

def train_critic(
    dataset_name: str,
    specific_dataset: str,
    batch_size=1024, 
    epochs=20, 
    gamma=0.99,
    lr=3e-4,
    tau = 0.005,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if(dataset_name == 'kitchen'):
         data =  KitchenDataset(specific_dataset)
    else:
         raise ValueError(f"Dataset {dataset_name} not found")
    trajectories = data.get_trajectories()
    dataset = CriticDataset(trajectories)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    
    obs_dim = data.get_state_dim()
    act_dim = data.get_action_dim()

    critic = QNet(obs_dim, act_dim).to(device)
    target_critic = QNet(obs_dim, act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    optimizer = optim.Adam(critic.parameters(), lr = lr)

    for epoch in range(epochs):  # number of passes over dataset
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
           optimizer.step()

           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)

       if epoch % 10 == 0:
              print(f"Epoch {epoch+10}, loss {loss.item():.4f}")

if __name__ == '__main__':  # pragma: no cover
    random.seed(1)
    train_critic(
    dataset_name = 'kitchen', 
    specific_dataset = 'mixed', 
    batch_size=1024, 
    epochs=50,  
    gamma=0.99, 
    lr=1e-3, 
    tau = 0.005)

