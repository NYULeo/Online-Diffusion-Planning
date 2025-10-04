import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Dataset import KitchenDataset, PointMazeDataset
from utils import *
import random
import copy
from .Kernel_Net import TransitionKernel
from sympy import factorint

# Define the Gaussian forward dynamics model: inputs (s, a), outputs mean and log_std of s'


def save_model(kernel_net, kernel_name, num_steps):
    kernel_net.eval()
    net_dict =  kernel_net.state_dict()
    os.makedirs(f'./Transition_Kernel/{kernel_name}/Models/', exist_ok=True)
    save_path = f'./Transition_Kernel/{kernel_name}/Models/{kernel_name}_{num_steps}.pkl'
    torch.save(net_dict, save_path)
    print(f"Kernel model save to {kernel_name}_{num_steps}.pkl")


def load_model(kernel_name, num_steps):
    load_path = f'./Transition_Kernel/{kernel_name}/Models/{kernel_name}_{num_steps}.pkl'
    state_dict = torch.load(load_path, map_location='cpu')
    return state_dict


def Train_Dataset(dataset_name, specific_dataset: Optional[str] = None):
    if(dataset_name == 'kitchen'):
         data_1 = KitchenDataset('complete')
         data_2 = KitchenDataset('partial')
         data_3 = KitchenDataset('mixed')
         trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
         name = 'Kitchen_Kernel'
         obs_dim = data_1.get_state_dim()
         act_dim = data_1.get_action_dim()
         return trajs, name, obs_dim, act_dim
     
    elif(dataset_name == 'pointmaze'):
         if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         elif(specific_dataset == 'large'):
              data = PointMazeDataset('large')
              name = '2DMaze_Kernel_large'
         elif(specific_dataset == 'medium'):
              data = PointMazeDataset('medium')
              name = '2DMaze_Kernel_medium'
         elif(specific_dataset == 'umaze'):
              data = PointMazeDataset('umaze')
              name = '2DMaze_Kernel_umaze'
         else: 
              raise ValueError(f"Invalid dataset name: {specific_dataset}")
         obs_dim = data.get_state_dim()
         act_dim = data.get_action_dim()
         trajs = data.get_trajectories()
         return trajs, name, obs_dim, act_dim

# Build (s, a, s') transitions from your offline trajectories
class KernelDataset(Dataset):
    def __init__(self, trajectories, kernel_name):
         obs_list, act_list = [], []
        
         for traj in trajectories:
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
         data = []
         for traj in trajectories:
            obs = np.asarray(traj['observations'])
            acts = np.asarray(traj['actions'])
            for t in range(len(acts)):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                data.append((s_t, a_t, s_tp1))
         self.data = data
         self.save_stats(kernel_name)
    
    def save_stats(self, kernel_name):
        stats_name =  str(kernel_name) + '_stats.pkl'
        stats_dir = f'./Transition_Kernel/{kernel_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, s_next = self.data[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32)
        )

class test_dataset(Dataset):
    def __init__(self, trajs, kernel_name):
        stats_path = f'./Transition_Kernel/{kernel_name}/Stats/{kernel_name}_stats.pkl'
        with open(stats_path, 'rb') as f:
              self.stats = pickle.load(f)
        transitions = []
        for traj in trajs:
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            for t in range(len(acts)):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                transitions.append((s_t, a_t, s_tp1))

        self.transitions = transitions
    
    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, s_next = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
        )
        

def train_kernel(dataset_name, specific_dataset: Optional[str] = None, batch_size = 256, lr = 1e-3, num_steps = 10000):
     # Prepare dataset and dataloader
     save_freq = 5000
     if(specific_dataset is None):
         print(f"Training kernel for {dataset_name} Dataset")
     else: 
         print(f"Training kernel for {dataset_name}_{specific_dataset} Dataset")
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print(f'Using device: {device}')
     trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
     dataset = KernelDataset(trajs, kernel_name)
     loader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8))

     # Create model and optimiser
     model = TransitionKernel(obs_dim, act_dim).to(device)
     optimiser = optim.Adam(model.parameters(), lr, weight_decay = 1e-5)

     #total probability before training
     total_prob = total_pro(trajs, model, device)
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
          step += 1
          
          if step % 10 == 0:
              avg_loss = total_nll / 10
              print(f"Step {step}, loss {avg_loss:.4f}")
              total_nll = 0.0

          if step % save_freq == 0:
              checkpoint = copy.deepcopy(model)
              save_model(checkpoint, kernel_name, step)
        
         
     #total probability after training
     model.eval()
     total_prob = total_pro(trajs, model, device)
     print(f"Total Probability After Training: {total_prob:.4f}")
     save_model(model, kernel_name, num_steps)
     

def test_Model(dataset_name, specific_dataset: Optional[str] = None, trajs: Optional[list] = None,  save_freq: int = 50, num_steps: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if(trajs is None): 
        train_Trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = KernelDataset(train_Trajs, kernel_name)
    else:
        _, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = test_dataset(trajs, kernel_name)
    print(f"Testing the reward model on {len(dataset)} samples")
    a = factorint(len(dataset))
    batch_size = int(np.min(list(a.keys())))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    num = save_freq
    while num <= num_steps:
         state_dict = load_model(kernel_name, num)
         kernel_net = TransitionKernel(obs_dim, act_dim).to(device)
         kernel_net.load_state_dict(state_dict)
         kernel_net.eval()
         total_loss = 0
         for s, a, s_next in dataloader:
             s_batch = s.to(device)
             a_batch = a.to(device)
             s_next_batch = s_next.to(device)
             for i in range(s.shape[0]):
                  s_i = s_batch[i].cpu().numpy()
                  a_i = a_batch[i].cpu().numpy()
                  s_next_i = s_next_batch[i].cpu().numpy()
                  prob = compute_log_prob(kernel_net, s_i, a_i, s_next_i, device)
                  total_loss += prob
         avg_loss = total_loss / len(dataset)
         print(f"model {num}, Loss {avg_loss:.4f}")
         num += save_freq
    
