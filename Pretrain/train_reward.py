from Dataset import KitchenDataset, PointMazeDataset
import random
from torch.utils.data import Dataset, DataLoader
from Critic import QNet
import torch
import torch.optim as optim
import numpy as np



class RewardDataset(Dataset):
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
                transitions.append((s_t, a_t, r_t))

        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )


def train_reward(dataset_name: str, batch_size, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if(dataset_name == 'kitchen'):
         data_1 = KitchenDataset('complete')
         data_2 = KitchenDataset('partial')
         data_3 = KitchenDataset('mixed')
         trajectories = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
         reward_name = 'Kitchen_Reward.pkl'
     
    elif(dataset_name == 'pointmaze'):
         data_1 = PointMazeDataset('large')
         data_2 = PointMazeDataset('medium')
         data_3 = PointMazeDataset('umaze')
         trajectories = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
         reward_name = '2DMaze_Reward.pkl'
    else:
         raise ValueError(f"Invalid dataset name: {dataset_name}")
    print(f"Training reward approximator for {dataset_name} Dataset")

    dataset = RewardDataset(trajectories)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    
    obs_dim = data_1.get_state_dim()
    act_dim = data_1.get_action_dim()

    reward_net = QNet(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(reward_net.parameters(), lr = lr)

    for epoch in range(epochs):  # number of passes over dataset
       for batch in dataloader:
           s, a, r = batch
           s = s.to(device)
           a = a.to(device)
           r = r.to(device)
        
           # Predicted Reward
           q_pred = reward_net(s, a)
           loss = ((q_pred - r) ** 2).mean()

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


       if epoch % 10 == 0:
              print(f"Epoch {epoch+10}, loss {loss.item():.4f}")
    
    reward_net.eval()
    torch.save(reward_net, reward_name)
    print(f"reward model save to {reward_name}")



if __name__ == '__main__':  # pragma: no cover
    random.seed(1)
    train_reward(
    dataset_name = 'kitchen',  
    batch_size=1024, 
    epochs=50,   
    lr=1e-3)



