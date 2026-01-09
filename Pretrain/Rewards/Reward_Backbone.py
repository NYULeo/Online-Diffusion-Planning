import sys
import os
from sympy.printing.rcode import reserved_words
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
from typing import Optional
from Dataset import KitchenDataset, PointMazeDataset, get_dataset, get_env
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from Pretrain.utils import set_seed, SAStats
import torch.nn as nn
import pickle
from Pretrain.Rewards.nets import Reward, MLPNetwork, ScalarReward, SimpleReward
import os
from scipy.ndimage import gaussian_filter1d, convolve
from Pretrain.utils import cycle
import copy
from sympy import Predicate, factorint
import torch.nn.functional as F
import numpy as np
import json

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
          elif specific_env== 'medium':
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

def save_reward_hyperparameters(dataset_name, batch_size, num_steps, lr, sigma, 
                                  obs_dim, act_dim, reward_name, optimizer, reward_net, filepath: Optional[str] = None, 
                                  specific_dataset: Optional[str] = None, target_reward: Optional[float] = None,
                                  goal: Optional[np.array] = None):
    if filepath is None:
        if specific_dataset is None:
            os.makedirs(f"./Pretrain/Rewards/{dataset_name}/args/", exist_ok=True)
            filepath = f"./Pretrain/Rewards/{dataset_name}/args/hyperparameters.json"
        else:
            os.makedirs(f"./Pretrain/Rewards/{dataset_name}/{specific_dataset}/args/", exist_ok=True)
            filepath = f"./Pretrain/Rewards/{dataset_name}/{specific_dataset}/args/hyperparameters.json"
    
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
        'model_type': type(reward_net).__name__,
        'obs_dim': int(obs_dim),
        'act_dim': int(act_dim),
    }
    
    # Add model-specific parameters if available
    if hasattr(reward_net, 'hidden_dim'):
        model_info['hidden_dim'] = int(reward_net.hidden_dim)
    if hasattr(reward_net, 'num_layers'):
        model_info['num_layers'] = int(reward_net.num_layers)
    if hasattr(reward_net, 'output_dim'):
        model_info['output_dim'] = int(reward_net.output_dim)
    
    # Compile all hyperparameters
    hyperparams = {
        'env_details': {
            'dataset_name': dataset_name,
            'specific_dataset': specific_dataset,
            'obs_dim': int(obs_dim),
            'act_dim': int(act_dim),
            'reward_name': reward_name,
        },
        'model_architecture': model_info,
        'training_hyperparameters': {
            'num_steps': num_steps,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer_params,
        },
        'reward_processing': {
            'sigma': sigma,
            'target_reward': target_reward,
            'goal': convert_to_json_serializable(goal),
        }
    }
    
    # Handle numpy arrays, torch.device, and other non-JSON-serializable types
    hyperparams = convert_to_json_serializable(hyperparams)
    
    # Save with pretty printing (indent=4 makes it human-readable)
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=4, sort_keys=False)
    
    print(f"Reward pretraining hyperparameters saved to {filepath}", flush=True)

def reward_filter(obs, rews, goal):
    #target_goals = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5], [-2.5, 2.5]])
    target_goals = goal
    for i in range(1, len(obs)):
        goal_coord = np.floor(obs[i][:2]) + 0.5
        #goal_coord = np.round(goal_coord, 1)  
        if np.any(np.all(np.equal(goal_coord, target_goals), axis=1)):
            rews[i-1] = 1
        else:
            rews[i-1] = 0
    return rews

def save_to_finetuning(reward_net, dataset_name, specific_dataset: Optional[str] = None):
    reward_net.eval()
    net_dict = reward_net.state_dict()
    name = getName(dataset_name, specific_dataset)
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Models/', exist_ok=True)
        save_path = f'./Finetuning/Rewards/{dataset_name}/Models/{name}_Reward_{str(0)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
        save_path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{name}_Reward_{str(0)}.pkl'
    torch.save(net_dict, save_path)
    print(f"reward model save to {save_path}")

def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None):
    name = getName(dataset_name, specific_dataset)
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Stats/', exist_ok=True)
        savepath = f'./Finetuning/Rewards/{dataset_name}/Stats/{name}_Reward_stats_{str(0)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/', exist_ok=True)
        savepath = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/{name}_Reward_stats_{str(0)}.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")

def save_model(reward_net, reward_name, num_steps):
    reward_net.eval()
    net_dict = reward_net.state_dict()
    os.makedirs(f'./Pretrain/Rewards/{reward_name}/Models/', exist_ok=True)
    save_path = f'./Pretrain/Rewards/{reward_name}/Models/{reward_name}_{num_steps}.pkl'
    print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)
    print(f"reward model save to {reward_name}_{num_steps}.pkl")

def load_model(reward_name, num_steps):
    load_path = f'./Pretrain/Rewards/{reward_name}/Models/{reward_name}_{num_steps}.pkl'
    #state_dict = torch.load(load_path, map_location='cpu')
    state_dict = torch.load(load_path, weights_only=True, map_location='cpu')
    return state_dict

def Train_Dataset(dataset_name, specific_dataset: Optional[str] = None):
    from Dataset import KitchenDataset, PointMazeDataset
    if(dataset_name == 'kitchen'):
         data_1 = KitchenDataset('complete')
         data_2 = KitchenDataset('partial')
         data_3 = KitchenDataset('mixed')
         trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
         name = 'Kitchen_Reward'
         obs_dim = data_1.get_state_dim()
         act_dim = data_1.get_action_dim()
         return trajs, name, obs_dim, act_dim
     
    elif(dataset_name == 'pointmaze'):
         if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         elif(specific_dataset == 'large'):
              data = PointMazeDataset('large')
              name = '2DMaze_Reward_large'
         elif(specific_dataset == 'medium'):
              data = PointMazeDataset('medium')
              name = '2DMaze_Reward_medium'
         elif(specific_dataset == 'umaze'):
              data = PointMazeDataset('umaze')
              name = '2DMaze_Reward_umaze'
         else: 
              raise ValueError(f"Invalid dataset name: {specific_dataset}")
         obs_dim = data.get_state_dim()
         act_dim = data.get_action_dim()
         trajs = data.get_trajectories()
         return trajs, name, obs_dim, act_dim
    else:
         raise ValueError(f"Invalid dataset name: {dataset_name}")
         

class RewardDataset(Dataset):
    def __init__(self, trajs, sigma, reward_name, target_reward: Optional[float] = None, goal: Optional[np.array] = None):
            
        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        for traj in trajs:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]
        
        allowed_values = [0,1]
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
        
        transitions = []
        Total = 0
        Zero = 0
        for traj in trajs:
            obs = np.asarray(traj['observations'])
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
            if(goal is not None):
                 rews = reward_filter(obs, rews, goal)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            
            rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))
                Total += 1
                if(r_t == 0.0):
                    Zero += 1
            
        self.transitions = transitions
        self.save_stats(reward_name)
    
    def save_stats(self, reward_name):
        stats_name =  str(reward_name) + '_stats.pkl'
        stats_dir = f'./Pretrain/Rewards/{reward_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )
    

def train_reward(dataset_name: str, batch_size, num_steps, save_freq, lr, sigma, target_reward: Optional[float] = None, specific_dataset: Optional[str] = None, goal: Optional[np.array] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    print(f"Training reward approximator for {dataset_name} Dataset") 
    dataset = RewardDataset(trajs, sigma, reward_name, target_reward, goal)
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8))
    reward_net = SimpleReward(obs_dim, act_dim).to(device)
    optimizer = optim.AdamW(reward_net.parameters(), lr = lr, weight_decay = 1e-4)
    save_reward_hyperparameters(
        dataset_name, 
        batch_size, 
        num_steps, 
        lr, 
        sigma,
        obs_dim,
        act_dim, 
        reward_name, 
        optimizer, 
        reward_net, 
        filepath = None,
        specific_dataset = specific_dataset, 
        target_reward = target_reward, 
        goal = goal
    )
    total_loss = 0
    step = 0
    for i in range(num_steps):
           s, a, r = next(dataloader)
           s = s.to(device)
           a = a.to(device)
           r = r.to(device)
        
           # Predicted Reward
           optimizer.zero_grad()
           #pred = reward_net(torch.cat([s, a], dim = 1))
           pred = reward_net(s, a)
           loss = F.mse_loss(pred, r)
           #loss = reward_net.loss(s, a, r)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
           step += 1

           if step % 100 == 0:
              avg_loss = total_loss / 100
              print(f"Step {step}, loss {avg_loss:.4f}")
              total_loss = 0

           if step % save_freq == 0:
              checkpoint = copy.deepcopy(reward_net)
              save_model(checkpoint, reward_name, step)
    save_to_finetuning(reward_net, dataset_name, specific_dataset)
    stats = get_pretrained_reward_stats(reward_name)
    save_stats_to_finetuning(stats, dataset_name, specific_dataset)

class test_dataset(Dataset):
    def __init__(self, trajs, sigma, Reward_name, target_reward: Optional[float] = None, goal: Optional[np.array] = None):
        self.stats = get_pretrained_reward_stats(Reward_name)
        transitions = []
        allowed_values = [0,1]
        for traj in trajs:
            obs = np.asarray(traj['observations'])        
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
            if(goal is not None):
                 rews = reward_filter(obs, rews, goal)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            rews = gaussian_filter1d(rews, sigma, mode = 'nearest', truncate = 200/sigma)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )
        
def test_Model(dataset_name, specific_dataset: Optional[str] = None, trajs: Optional[list] = None, sigma: float = 3, target_reward: Optional[float] = None, goal: Optional[np.array] = None, save_freq: int = 50, num_steps: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    print(f"Testing the reward model for {dataset_name} Dataset")
    print(f"Target reward: {target_reward}, Sigma: {sigma}")

    if(trajs is None): 
        train_Trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = RewardDataset(train_Trajs, sigma, reward_name, target_reward, goal)
    else:
        _, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = test_dataset(trajs, sigma, reward_name, target_reward, goal)
    print(f"Testing the reward model on {len(dataset)} samples")
    a = factorint(len(dataset))
    batch_size = int(np.min(list(a.keys())))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    num = save_freq
    while num <= num_steps:
         state_dict = load_model(reward_name, num)
         reward_net = SimpleReward(obs_dim, act_dim).to(device)
         #reward_net = DeepScaledReward(obs_dim, act_dim).to(device)
         #reward_net = Reward(obs_dim, act_dim).to(device)
         #reward_net = MLPNetwork(input_dim = obs_dim + act_dim, out_dim = 1, hidden_dims = [200, 200, 200, 200], act_fn = 'swish', out_act_fn = 'identity').to(device)
         reward_net.load_state_dict(state_dict)
         reward_net.eval()
         total_mean_loss = 0.0
         total_reward = 0.0
         for s, a, r in dataloader:
             s = s.to(device)
             a = a.to(device)
             r = r.to(device)
             #pred = reward_net(torch.cat([s, a], dim = 1))
             pred = reward_net(s, a)
             loss = F.mse_loss(pred, r)
             #loss = reward_net.loss(s, a, r)
             total_mean_loss += loss.item()
             total_reward += pred.mean().item()
             
         avg_mean_loss = total_mean_loss / len(dataloader)
         avg_reward = total_reward / len(dataloader)
         print(f"model {num}, Loss {avg_mean_loss:.4f}, Reward: {avg_reward:.4f}")
         num += save_freq

def get_pretrained_reward(dataset_name, checkpoints, specific_dataset: Optional[str] = None):
       _, name, obs_dim, act_dim  =  Train_Dataset(dataset_name, specific_dataset)
       reward_model_state_dict = load_model(name, checkpoints)
       return reward_model_state_dict, obs_dim, act_dim, name

def get_pretrained_reward_stats(Reward_name):
    stats_path = f'./Pretrain/Rewards/{Reward_name}/Stats/{Reward_name}_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats


'''
def test_Single_Model(dataset_name, specific_dataset: Optional[str] = None, trajs: Optional[list] = None, sigma: float = 3, target_reward: Optional[float] = None, num: int = 10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if(trajs is None): 
        train_Trajs, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = RewardDataset(train_Trajs, sigma, reward_name, target_reward)
    else:
        _, reward_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
        dataset = test_dataset(trajs, sigma, reward_name, target_reward)
    print(f"Testing the reward model on {len(dataset)} samples")
    a = factorint(len(dataset))
    batch_size = int(np.min(list(a.keys())))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    
    state_dict = load_model(reward_name, num)
    reward_net = ScalarReward(obs_dim, act_dim).to(device)
    reward_net.load_state_dict(state_dict)
    reward_net.eval()
    total_mean_loss = 0
    total_var = 0
    for s, a, r in dataloader:
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        mean = reward_net.predict(s, a)
        var = reward_net.variance(s, a)
        mean_loss = ((mean - r).abs()).mean()
        total_mean_loss += mean_loss.item()
        total_var += var.mean().item()
    avg_mean_loss = total_mean_loss / len(dataloader)
    avg_var = total_var / len(dataloader)
    print(f"model {num}, Loss {avg_mean_loss:.4f}, Variance {avg_var:.4f}")



def grad_norm(s, a, reward_net):
     s.requires_grad_(True)
     a.requires_grad_(True)
     pred = reward_net(s, a)
     
     # Compute gradients with respect to the full batch
     grad_outputs = torch.ones_like(pred)
     grads_s, grads_a = torch.autograd.grad(
         outputs=pred,
         inputs=(s, a),
         grad_outputs=grad_outputs,
         create_graph=False,
         retain_graph=False,
         allow_unused=True  # In case one input is not used
     )
     
     # Handle case where one input might not be used
     if grads_s is None:
         grads_s = torch.zeros_like(s)
     if grads_a is None:
         grads_a = torch.zeros_like(a)
     
     # Compute per-sample gradient norms
     grad_norms = torch.cat([grads_s, grads_a], dim=-1).norm(p=2, dim=-1)  # [batch_size]
     grad_norm_avg = grad_norms.mean().item()
     
     return pred, grad_norm_avg
'''

