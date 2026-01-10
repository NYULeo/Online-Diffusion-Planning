import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Dataset import KitchenDataset, PointMazeDataset
from .Kernel_Net import  RobustTransitionKernel
from sympy import factorint
import pickle
import os
from typing import Optional
import math
import copy
from Pretrain.utils import SAStats, cycle
import json

def check_specifc_dataset(dataset_name):
    if(dataset_name == 'kitchen'):
         return False
    elif(dataset_name == 'pointmaze'):
         return True

def getName(env_name, specific_env):
     if(env_name == 'kitchen'):
          return 'Kitchen'
     elif(env_name == 'pointmaze'):
          if specific_env == 'open_dense':
               return 'PointMaze_OpenDense'
          elif specific_env == 'umaze':
               return 'PointMaze_Umaze'
          elif specific_env == 'large_dense':
               return 'PointMaze_LargeDense'
          elif specific_env == 'medium':
               return 'PointMaze_Medium'
          elif specific_env == 'umaze_dense':
               return 'PointMaze_UmazeDense'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env == 'open':
               return 'PointMaze_Open'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'antmaze'):
          if specific_env == 'medium_play':
               return 'AntMaze_MediumPlay'
          elif specific_env == 'umaze_diverse':
               return 'AntMaze_UmazeDiverse'
          elif specific_env == 'large_diverse':
               return 'AntMaze_LargeDiverse'
          elif specific_env == 'large_play':
               return 'AntMaze_LargePlay'
          elif specific_env == 'medium_diverse':
               return 'AntMaze_MediumDiverse'
          elif specific_env == 'umaze':
               return 'AntMaze_Umaze'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def save_kernel_hyperparameters(dataset_name, batch_size, num_steps, lr, 
                                obs_dim, act_dim, kernel_name, optimizer, kernel_net, 
                                ensemble_size, λ_reg, specific_dataset: Optional[str] = None):
  
   
    os.makedirs(f"./Pretrain/Transition_Kernel/{kernel_name}/args/", exist_ok=True)
    filepath = f"./Pretrain/Transition_Kernel/{kernel_name}/args/hyperparameters.json"
    
    def convert_to_json_serializable(obj):
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj
    
    # Get optimizer info
    optimizer_type = type(optimizer).__name__
    optimizer_params = {
        'type': optimizer_type,
        'lr': lr,
        'weight_decay': optimizer.param_groups[0].get('weight_decay', 0)
    }
    
    # Get model architecture info
    model_info = {
        'model_type': type(kernel_net).__name__,
        'obs_dim': int(obs_dim),
        'act_dim': int(act_dim),
    }
    
    # Add model-specific parameters if available
    if hasattr(kernel_net, 'min_log_std'):
        model_info['min_log_std'] = float(kernel_net.min_log_std)
    if hasattr(kernel_net, 'max_log_std'):
        model_info['max_log_std'] = float(kernel_net.max_log_std)
    if hasattr(kernel_net, 'noise_floor'):
        model_info['noise_floor'] = float(kernel_net.noise_floor)
    
    # Compile all hyperparameters
    hyperparams = {
        'env_details': {
            'dataset_name': dataset_name,
            'specific_dataset': specific_dataset,
            'obs_dim': int(obs_dim),
            'act_dim': int(act_dim),
            'kernel_name': kernel_name,
        },
        'model_architecture': model_info,
        'training_hyperparameters': {
            'num_steps': num_steps,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer_params,
            'save_freq': 2000,  # Hardcoded in train_kernel
        },
        'ensemble_config': {
            'ensemble_size': int(ensemble_size),
            'λ_reg': float(λ_reg),
        }
    }
    
    # Handle numpy arrays, torch.device, and other non-JSON-serializable types
    hyperparams = convert_to_json_serializable(hyperparams)
    
    # Save with pretty printing (indent=4 makes it human-readable)
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=4, sort_keys=False)
    
    print(f"Kernel pretraining hyperparameters saved to {filepath}", flush=True)


# Define the Gaussian forward dynamics model: inputs (s, a), outputs mean and log_std of s'
"""
def compute_log_prob(model, s, a, s_next):
    with torch.no_grad():
        mu, log_std = model(s, a)
        sigma = torch.exp(log_std)
        D = mu.size(-1)
        # Compute log prob per dimension and sum
        log_prob = -0.5 * (((s_next - mu) / sigma) ** 2).sum(dim=-1)
        log_prob += -0.5 * (D * math.log(2 * math.pi) + 2 * log_std.sum(dim=-1))
    return log_prob.item()

"""
def save_model(kernel_net, kernel_name, num_steps, ensemble_idx):
    kernel_net.eval()
    net_dict = kernel_net.state_dict()
    os.makedirs(f'./Pretrain/Transition_Kernel/{kernel_name}/Models/{num_steps}', exist_ok=True)
    save_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Models/{num_steps}/{kernel_name}_{num_steps}_{ensemble_idx}.pkl'
    torch.save(net_dict, save_path)
    print(f"Kernel model save to {kernel_name}_{num_steps}_{ensemble_idx}.pkl")

def save_to_finetuning(kernel_net, dataset_name, ensemble_idx, specific_dataset: Optional[str] = None):
    kernel_net.eval()
    net_dict = kernel_net.state_dict()
    name = getName(dataset_name, specific_dataset)
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Kernels/{dataset_name}/Models/{str(0)}', exist_ok=True)
        save_path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(0)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(0)}', exist_ok=True)
        save_path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(0)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    torch.save(net_dict, save_path)
    print(f"kernel model save to {save_path}")

def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None):
    name = getName(dataset_name, specific_dataset)
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Kernels/{dataset_name}/Stats/', exist_ok=True)
        savepath = f'./Finetuning/Kernels/{dataset_name}/Stats/{name}_Kernel_stats_{str(0)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Stats/', exist_ok=True)
        savepath = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Stats/{name}_Kernel_stats_{str(0)}.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")

def count_files_in_folder(folder_path):
    """
    Count the number of files in a specific folder.
    Returns the count of files (excluding directories).
    """
    try:
        # Get all items in the folder
        items = os.listdir(folder_path)
        
        # Count only files (not directories)
        file_count = 0
        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                file_count += 1
        
        return file_count
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return 0
    except PermissionError:
        print(f"Permission denied to access '{folder_path}'.")
        return 0

def load_model(kernel_name, num_steps, ensemble_idx):
    load_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Models/{num_steps}/{kernel_name}_{num_steps}_{ensemble_idx}.pkl'
    #state_dict = torch.load(load_path, map_location='cpu')
    state_dict = torch.load(load_path, weights_only=True)
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
        stats_dir = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/'
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
        stats_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/{kernel_name}_stats.pkl'
        with open(stats_path, 'rb') as f:
              self.stats = pickle.load(f)
        transitions = []
        for traj in trajs:
            obs = np.asarray(traj['observations'])      
            acts = np.asarray(traj['actions'])
            if(len(obs) != len(acts)):
                 L = len(acts)
            else:
                 L = len(acts) - 1
            for t in range(L):
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
        
"""
def train_kernel(dataset_name, specific_dataset: Optional[str] = None, batch_size = 256, lr = 1e-3, num_steps = 10000):
     # Prepare dataset and dataloader
     save_freq = 2000
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
          
          if step % 500 == 0:
              avg_loss = total_nll / 500
              print(f"Step {step}, loss {avg_loss:.4f}")
              total_nll = 0.0

          if step % save_freq == 0:
              checkpoint = copy.deepcopy(model)
              save_model(checkpoint, kernel_name, step)
        
         
     #total probability after training
     model.eval()
     save_model(model, kernel_name, num_steps)
     

def test_Model(dataset_name, specific_dataset: Optional[str] = None, trajs: Optional[list] = None,  save_freq: int = 50, num_steps: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    Train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)  
    if(trajs is None):
         dataset = test_dataset(Train_trajs, kernel_name)
    else:
         dataset = test_dataset(trajs, kernel_name)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 8)
    num = save_freq 
    while num <= num_steps:
        state_dict = load_model(kernel_name, num)
        kernel_net = TransitionKernel(obs_dim, act_dim).to(device)
        kernel_net.load_state_dict(state_dict)
        kernel_net.eval()
        probs = []
        for s, a, s_next in dataloader:
             s = s.to(device)
             a = a.to(device)
             s_next = s_next.to(device)
             probs.append(compute_log_prob(kernel_net, s, a, s_next))
        mean_probs = np.mean(probs)
        min_probs = np.min(probs)
        print(f"Model {num}, mean_prob: {mean_probs:.4f}, min_prob {min_probs:.4f}")
        num += save_freq
        
"""

def train_kernel(dataset_name, specific_dataset: str = None,
                 batch_size=256, lr=1e-3, num_steps=10000,
                 ensemble_size=10, λ_reg=1e-3):
    # Prepare dataset / dataloader
    if specific_dataset is None:
        print(f"Training kernel for {dataset_name}")
    else:
        print(f"Training kernel for {dataset_name}_{specific_dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    dataset = KernelDataset(trajs, kernel_name)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=8))

    # Create ensemble of models
    ensemble = [RobustTransitionKernel(obs_dim, act_dim).to(device) for _ in range(ensemble_size)]
    optimizers = [optim.Adam(m.parameters(), lr, weight_decay=1e-5) for m in ensemble]


    # Save hyperparameters at the start of training
    save_kernel_hyperparameters(
        dataset_name, 
        batch_size, 
        num_steps, 
        lr,
        obs_dim,
        act_dim, 
        kernel_name, 
        optimizers[0],  # Use first optimizer as representative
        ensemble[0],    # Use first model as representative
        ensemble_size,
        λ_reg,
        specific_dataset=specific_dataset
    )
    if(check_specifc_dataset(dataset_name)):
        SD = specific_dataset
    else:
        SD = None
    step = 0
    total_loss = 0.0
    save_freq = 50000

    for step in range(1, num_steps + 1):
        s, a, s_next = next(loader)
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)

        # For each model in ensemble, compute loss
        losses = []
        mus = []
        log_stds = []
        for m in ensemble:
            mu, log_std = m(s, a)
            mus.append(mu)
            log_stds.append(log_std)
            loss = m.gaussian_nll(s_next, mu, log_std)
            losses.append(loss)
        # optional: variance‐disagreement inflation
        # compute mean of mus
        mus_stack = torch.stack(mus, dim=0)  # (K, B, obs_dim)
        mu_mean = mus_stack.mean(dim=0)      # (B, obs_dim)
        # disagreement = average squared deviation
        disagreement = ((mus_stack - mu_mean.unsqueeze(0)) ** 2).mean(dim=0) 
        disagreement_detached = disagreement.detach()
        # inflate each model’s loss by penalizing small variance in high disagreement dims
        for i, m in enumerate(ensemble):
            # log_std_i is log_stds[i]
            # penalize if log_std is too small relative to disagreement
            penalty = (disagreement_detached / (torch.exp(2 * log_stds[i]) + m.noise_floor)).sum(dim=-1).mean()
            losses[i] = losses[i] + λ_reg * penalty

        # Backprop & optimize each model
        for i, (m, opt) in enumerate(zip(ensemble, optimizers)):
            opt.zero_grad()
            losses[i].backward()
            opt.step()

        avg_loss = sum(losses).item() / ensemble_size
        total_loss += avg_loss

        if step % 500 == 0:
            print(f"Step {step}, avg_loss: {total_loss / 500:.6f}")
            total_loss = 0.0

        if step % save_freq == 0 or step == num_steps:
            # Save all ensemble members
            for idx, m in enumerate(ensemble):
                ckpt = copy.deepcopy(m).cpu()
                save_model(ckpt, kernel_name, step, idx)
            if(step == num_steps):
                for idx, m in enumerate(ensemble):
                    ckpt = copy.deepcopy(m).cpu()
                    save_to_finetuning(ckpt, dataset_name, idx, SD)
                 
    
    stats = get_pretrained_kernel_stats(kernel_name)
    save_stats_to_finetuning(stats, dataset_name, SD)
    # Return final ensemble
    return ensemble


def test_kernel(dataset_name, specific_dataset: str = None,
                trajs: list = None,
                save_freq: int = 50, num_steps: int = 500, ensemble_size = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    if trajs is None:
        dataset = test_dataset(train_trajs, kernel_name)
    else:
        dataset = test_dataset(trajs, kernel_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=8)

    # For each saved checkpoint / ensemble member
    step = save_freq
    while step <= num_steps:
        # Load ensemble members
        ensemble = []
        for idx in range(ensemble_size):
            state_dict = load_model(kernel_name, step, idx)
            m = RobustTransitionKernel(obs_dim, act_dim).to(device)
            m.load_state_dict(state_dict)
            m.eval()
            ensemble.append(m)

        # Compute log-probs over dataset
        all_lp = []
        #worst = (None, float("inf"), None)  # (idx, log_prob, (s, a, s_next))
        for i, (s, a, s_next) in enumerate(dataloader):
            s = s.to(device)
            a = a.to(device)
            s_next = s_next.to(device)

            # For each ensemble member, compute log_prob
            lps = []
            for m in ensemble:
                mu, log_std = m(s, a)
                lp = m.log_prob(s_next, mu, log_std).item()
                lps.append(lp)
            # You can take mean, or min over ensemble.
            lp_mean = sum(lps) / len(lps)
            all_lp.append(lp_mean)

            #if lp_mean < worst[1]:
             #   worst = (i, lp_mean, (s.cpu().numpy(), a.cpu().numpy(), s_next.cpu().numpy()))

        mean_lp = float(np.mean(all_lp))
        min_lp = float(np.min(all_lp))
        print(f"Checkpoint {step}: mean_log_prob = {mean_lp:.4f}, min_log_prob = {min_lp:.4f}")
        #print("Worst transition index", worst[0], "lp", worst[1])
        #print("Corresponding s, a, s_next:", worst[2])
        step += save_freq


def get_pretrained_kernel(dataset_name, checkpoints, specific_dataset: Optional[str] = None):
       _, name, obs_dim, act_dim  =  Train_Dataset(dataset_name, specific_dataset)
       path = f'./Pretrain/Transition_Kernel/{name}/Models/{checkpoints}'
       file_count = count_files_in_folder(path)
       kernel_state_dicts = []
       for i in range(file_count):
           kernel_state_dicts.append(load_model(name, checkpoints, i))
       return kernel_state_dicts, obs_dim, act_dim, name

def get_pretrained_kernel_stats(kernel_name):
     stats_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/{kernel_name}_stats.pkl'
     with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
     return stats









