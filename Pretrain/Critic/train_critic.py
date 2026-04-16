
from pathlib import Path
import copy
import pickle
from typing import Optional, List

import numpy as np
from sympy.integrals.meijerint import _rewrite_single
import torch
import torch.optim as optim
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader

from Finetuning.utils import TrajectoryDict, get_trajs, getName
from Pretrain.Dataset import get_dataset, get_env
from Pretrain.utils import set_seed, SAStats, cycle, ema_smooth
from Pretrain.Critic.nets import Critic, CriticEnsemble

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # Online-Diffusion-Planning/
PRETRAIN_DIR = PROJECT_ROOT / "Pretrain"
FINETUNE_DIR = PROJECT_ROOT / "Finetuning"

def spare_reward_prcocessor(rewards):
    Temp = []
    for i in range(1, len(rewards)):
        if(rewards[i] == rewards[i-1]+1):
            Temp.append(i)
    new_rewards = [0]*len(rewards)
    for i in range(len(rewards)):
        if(i in Temp):
            new_rewards[i] = 1.0
        else:
            new_rewards[i] = 0.0
    return np.array(new_rewards, dtype = np.float64) 
    #return np.array(new_rewards) 

def save_critic_hyperparameters(dataset_name, batch_size, num_steps, lr, sigma, alpha, 
                                obs_dim, critic_net, optimizer, gamma, horizon, tau,
                                specific_dataset: Optional[str] = None, 
                                target_reward: Optional[float] = None,
                                goal: Optional[np.array] = None):
    
    """
    os.makedirs(f"./Pretrain/Critic/{dataset_name}/{specific_dataset}/args/", exist_ok=True)
    filepath = f"./Pretrain/Critic/{dataset_name}/{specific_dataset}/args/hyperparameters.json"
    """
    args_dir = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "args"
    args_dir.mkdir(parents=True, exist_ok=True)
    filepath = args_dir / "hyperparameters.json"

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
        'model_type': type(critic_net).__name__,
        'obs_dim': int(obs_dim),
    }
    
    # Add model-specific parameters if available
    if hasattr(critic_net, 'hidden'):
        model_info['hidden_dim'] = int(critic_net.hidden)
    
    # Compile all hyperparameters
    hyperparams = {
        'env_details': {
            'dataset_name': dataset_name,
            'specific_dataset': specific_dataset,
            'obs_dim': int(obs_dim),
        },
        'model_architecture': model_info,
        'training_hyperparameters': {
            'num_steps': num_steps,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer_params,
        },
        'critic_config': {
            'gamma': float(gamma),
            'horizon': int(horizon),
            'tau': float(tau),
        },
        'reward_processing': {
            'sigma': float(sigma) if sigma is not None else None,
            'alpha': float(alpha) if alpha is not None else None,
            'target_reward': target_reward,
            'goal': convert_to_json_serializable(goal),
        }
    }
    
    # Handle numpy arrays, torch.device, and other non-JSON-serializable types
    hyperparams = convert_to_json_serializable(hyperparams)
    
    # Save with pretty printing (indent=4 makes it human-readable)
    import json
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=4, sort_keys=False)
    
    print(f"Critic pretraining hyperparameters saved to {filepath}", flush=True)


def get_CriticName(env_name, specific_env):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High_Critic.pt'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium_Critic.pt'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed_Critic.pt'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
         if(specific_env == 'large'):
              return 'PointMaze_Large_Critic.pt'
         elif(specific_env == 'medium'):
              return 'PointMaze_Medium_Critic.pt'
         elif(specific_env == 'unmaze'):
              return 'PointMaze_Unmaze_Critic.pt'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'cube'):
         if specific_env == 'single-play':
              return 'Cube_SinglePlay_Critic.pt'
         elif specific_env == 'single-noisy':
             return 'Cube_SingleNoisy_Critic.pt'
         elif specific_env == 'double-play':
             return 'Cube_DoublePlay_Critic.pt'
         elif specific_env == 'double-noisy':
             return 'Cube_DoubleNoisy_Critic.pt'
         elif specific_env == 'triple-play':
             return 'Cube_TriplePlay_Critic.pt'
         elif specific_env == 'triple-noisy':
             return 'Cube_TripleNoisy_Critic.pt'
         elif specific_env == 'quadruple-play':
             return 'Cube_QuadruplePlay_Critic.pt'
         elif specific_env == 'quadruple-noisy':
             return 'Cube_QuadrupleNoisy_Critic.pt'
         else:
             raise ValueError(f"Invalid cube dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

"""
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
"""


def reward_filter(obs, rews, goal):
    #target_goals = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5], [-2.5, 2.5]])
    for i in range(1, len(obs)):
        pos = obs[i][:2] 
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        #goal_coord = np.asarray(goal_coord, dtype=np.float32).reshape(-1)  
        dist = np.linalg.norm(pos - g) 
        if (dist < 0.5):
            rews[i-1] = 1.0
        else:
            rews[i-1] = 0.0
    return rews

"""
def obs_filter(obs):
    obs = obs[:, 2]
    return obs
"""

def save_critic(model, dataset_name, specific_dataset, step):
    model.eval()
    name = get_CriticName(dataset_name, specific_dataset)
    net_dict = model.state_dict()
    """
    os.makedirs(f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
    save_path = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl'
    """
    models_dir = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"{name}_Critic_{step}.pkl"
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    torch.save(net_dict, save_path)
    print(f"critic model save to {name}.pkl")

def save_to_finetuning(critic_net, dataset_name, specific_dataset):
    critic_net.eval()
    net_dict = critic_net.state_dict()
    name = getName(dataset_name, specific_dataset)
    """
    os.makedirs(f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
    save_path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(0)}.pkl'
    """
    ft_models_dir = FINETUNE_DIR / "Critics" / dataset_name / specific_dataset / "Models"
    ft_models_dir.mkdir(parents=True, exist_ok=True)
    save_path = ft_models_dir / f"{name}_Critic_0.pkl"
    torch.save(net_dict, save_path)
    print(f"critic model save to {save_path}")

"""
def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None):
    name = getName(dataset_name, specific_dataset)
    os.makedirs(f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/', exist_ok=True)
    savepath = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/{name}_Critic_stats_{str(0)}.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")
"""

def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None):
    name = getName(dataset_name, specific_dataset)
    ft_stats_dir = FINETUNE_DIR / "Critics" / dataset_name / specific_dataset / "Stats"
    ft_stats_dir.mkdir(parents=True, exist_ok=True)
    savepath = ft_stats_dir / f"{name}_Critic_stats_0.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")

"""
def get_critic_model(dataset_name, specific_dataset, step):
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    name = get_CriticName(dataset_name, specific_dataset)
    path = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl'
    model_state_dict = torch.load(path, weights_only=True, map_location='cpu')
    return model_state_dict, obs_dim
"""
def get_critic_model(dataset_name, specific_dataset, step):
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    name = get_CriticName(dataset_name, specific_dataset)
    path = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Models" / f"{name}_Critic_{step}.pkl"
    model_state_dict = torch.load(path, weights_only=True, map_location="cpu")
    return model_state_dict, obs_dim

"""
def get_critic_stats(dataset_name, specific_dataset):
    name = get_CriticName(dataset_name, specific_dataset)
    path = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Stats/{name}_Critic_stats.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats 
"""
def get_critic_stats(dataset_name, specific_dataset):
    name = get_CriticName(dataset_name, specific_dataset)
    path = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Stats" / f"{name}_Critic_stats.pkl"
    with open(path, "rb") as f:
        stats = pickle.load(f)
    return stats



class Critic_Test_Dataset(Dataset):
    def __init__(self, dataset_name: str, specific_dataset: str, trajs, sigma: Optional[float] = None, alpha: Optional[float] = None, goal: Optional[np.array] = None, target_reward: Optional[float] = None, horizon: int = 32, gamma: float = 0.99):
        # ----- gather raw obs/actions to fit stats -----
        """
        if(dataset_name == 'pointmaze'):
           trajs = copy.deepcopy(trajs)
           for traj in trajs:
                traj['observations'] = traj['observations'][:,:2]
        """
        
        self.stats = get_critic_stats(dataset_name, specific_dataset)
        allowed_values = [0, 1]

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards']
            rews = spare_reward_prcocessor(rews)
            if( goal is not None):
                rews = reward_filter(obs, rews, goal)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            
            if(alpha is not None):
                rews = ema_smooth(rews, alpha)
            elif(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)

            for t in range(len(obs)):
                 obs_t = self.stats.norm_obs(obs[t])
                 r_t   = np.sum(rews[t:])
                 transitions.append((obs_t, r_t))
        self.transitions = transitions
    
    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, r = self.transitions[idx]
        return (
            torch.tensor(s, dtype = torch.float32),
            torch.tensor(r, dtype = torch.float32)
        )
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

class CriticDataset(Dataset):
    def __init__(self, dataset_name: str, specific_dataset: str, trajs: List[TrajectoryDict], goal: Optional[np.array] = None, target_reward: Optional[float] = None, horizon: int = 32, gamma: float = 0.99, sigma: float = 7.0, alpha: Optional[float] = None):
        
        
        """
        if(dataset_name == 'pointmaze'):
           trajs = copy.deepcopy(trajs)
           for traj in trajs:
                traj['observations'] = traj['observations'][:,:2]
        """
        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards']
            rews = spare_reward_prcocessor(rews)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if( goal is not None):
                rews = reward_filter(obs, rews, goal)
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            if(alpha is not None):
                rews = ema_smooth(rews, alpha)
            elif(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            if(len(obs) > horizon):
               rews = self.reward_processor(rews, horizon, gamma)
               for t in range(len(obs)-horizon):
                   obs_t = self.stats.norm_obs(obs[t])
                   r_t   = rews[t]
                   obs_next_t = self.stats.norm_obs(obs[min(t+horizon, len(obs)-1)])
                   transitions.append((obs_t, r_t, obs_next_t))
           
        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset)
        
    """
    def save_stats(self, dataset_name, specific_dataset):
        name = get_CriticName(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Critic_stats.pkl'
        stats_dir = f'./Pretrain/Critic/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")
    """
    def save_stats(self, dataset_name, specific_dataset):
        name = get_CriticName(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Critic_stats.pkl'
        stats_dir = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        savepath = stats_dir / stats_name
        with open(savepath, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.transitions)#

    def __getitem__(self, idx):
        s, r, s_next = self.transitions[idx]
        return (
            torch.tensor(s, dtype = torch.float32),
            torch.tensor(r, dtype = torch.float32),
            torch.tensor(s_next, dtype = torch.float32)
        )
    
    def boost_signal(self, target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews
    
    def reward_processor(self, rews, horizon, gamma):
        new_rews = []
        for t in range(len(rews)):
            R = 0.0
            for i in range(t, min(t + horizon, len(rews))):
                R += (gamma**(i-t))*rews[i]
                #R += (gamma**(i-t+1))*rews[i]
            new_rews.append(R)
            #new_rews.append(np.sum(rews[t:]))
        return new_rews

def train_critic(dataset_name: str, specific_dataset: str, hidden_layers: int, hidden_dim: int, batch_size, num_steps, gamma, horizon, lr, tau, goal, sigma: Optional[float] = None, alpha: Optional[float] = None, target_reward = 1.0, trajs: List[TrajectoryDict] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CriticDataset(dataset_name, specific_dataset, trajs, goal, target_reward, horizon, gamma, sigma, alpha)
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)

    """
    #prepare training
    if(dataset_name == 'pointmaze'):
          obs_dim = obs_dim - 2
    """
    
    
    dataloader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True))
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic.train()
    target_critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    optimizer = optim.Adam(critic.parameters(), lr = lr)

    save_critic_hyperparameters(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            sigma=sigma,
            alpha=alpha,
            obs_dim=obs_dim,
            critic_net=critic,
            optimizer=optimizer,
            gamma=gamma,
            horizon=horizon,
            tau=tau,
            specific_dataset=specific_dataset,
            target_reward=target_reward,
            goal=goal)  
   
    print(f"Training critic for {dataset_name}-{specific_dataset}")
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, r, s_next = next(dataloader)
           s = s.to(device)
           r = r.to(device)
           s_next = s_next.to(device)

           # Compute target V-values
           with torch.no_grad():
              q_next = target_critic(s_next)
              target = r + ( (gamma**horizon) * q_next)

           # Predicted V-values
           q_pred = critic(s)
           loss = ((q_pred - target) ** 2).mean()

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
        
           if(k % 5000 == 0):
                target_critic.eval()
                save_critic(target_critic, dataset_name, specific_dataset, k)
                print(f"Checkpoint saved at step {k}")
    save_to_finetuning(target_critic, dataset_name, specific_dataset)
    stats = get_critic_stats(dataset_name, specific_dataset)
    save_stats_to_finetuning(stats, dataset_name, specific_dataset)
    print(f"critic model saved")


def test_critic(dataset_name: str, specific_dataset: str, hidden_layers: int, hidden_dim: int, checkpoint_step, gamma, horizon, goal = None,  sigma: Optional[float] = None, alpha: Optional[float] = None,  target_reward = 1.0, trajs: Optional[List[TrajectoryDict]] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Critic_Test_Dataset(dataset_name, specific_dataset, trajs, sigma, alpha, goal, target_reward, horizon, gamma)
    batch_size = 100
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, checkpoint_step)
    
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"Testing critic for {dataset_name}-{specific_dataset} at checkpoint step {checkpoint_step}")
    total_loss = 0.0
    for s, r in dataloader:
           s = s.to(device)
           r = r.to(device)
           pred = model(s)
           total_loss += ((pred - r)**2).mean().item()
    avg_loss = total_loss/len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")




"""
def train_critic(dataset_name: str, specific_dataset: str, hidden_layers: int, hidden_dim: int, sigma: float, batch_size, num_steps, gamma, horizon, lr, tau, goal, target_reward=1.0, trajs: List[TrajectoryDict] = None, num_heads: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CriticDataset(sigma, dataset_name, specific_dataset, trajs, goal, target_reward, horizon, gamma)
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)

    dataloader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))
    critic = CriticEnsemble(obs_dim, hidden_dim, hidden_layers, num_heads=num_heads).to(device)
    critic.train()
    target_critic = CriticEnsemble(obs_dim, hidden_dim, hidden_layers, num_heads=num_heads).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    optimizer = optim.Adam(critic.parameters(), lr=lr)

    save_critic_hyperparameters(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            sigma=sigma,
            obs_dim=obs_dim,
            critic_net=critic,
            optimizer=optimizer,
            gamma=gamma,
            horizon=horizon,
            tau=tau,
            specific_dataset=specific_dataset,
            target_reward=target_reward,
            goal=goal)

    print(f"Training critic ensemble (num_heads={num_heads}) for {dataset_name}-{specific_dataset}")
    for k in range(1, num_steps + 1):
           s, r, s_next = next(dataloader)
           s = s.to(device)
           r = r.to(device)
           s_next = s_next.to(device)

           with torch.no_grad():
              q_next = target_critic(s_next, aggregate="mean")
              target = r + ((gamma**horizon) * q_next)

           q_pred = critic(s, aggregate="mean")
           loss = ((q_pred - target) ** 2).mean()

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)

           if k % 5000 == 0:
                target_critic.eval()
                save_critic(target_critic, dataset_name, specific_dataset, k)
                print(f"Checkpoint saved at step {k}")
    save_to_finetuning(target_critic, dataset_name, specific_dataset)
    stats = get_critic_stats(dataset_name, specific_dataset)
    save_stats_to_finetuning(stats, dataset_name, specific_dataset)
    print("critic model saved")



def test_critic(dataset_name: str, specific_dataset: str, hidden_layers: int, hidden_dim: int, checkpoint_step, sigma, gamma, horizon, goal=None, target_reward=1.0, trajs: Optional[List[TrajectoryDict]] = None, num_heads: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Critic_Test_Dataset(sigma, dataset_name, specific_dataset, trajs, goal, target_reward, horizon, gamma)
    batch_size = 100
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, checkpoint_step)

    model = CriticEnsemble(obs_dim, hidden_dim, hidden_layers, num_heads=num_heads).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    print(f"Testing critic ensemble (num_heads={num_heads}) for {dataset_name}-{specific_dataset} at checkpoint step {checkpoint_step}")
    total_loss = 0.0
    for s, r in dataloader:
           s = s.to(device)
           r = r.to(device)
           pred = model(s, aggregate="mean")
           total_loss += ((pred - r) ** 2).mean().item()
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")

"""