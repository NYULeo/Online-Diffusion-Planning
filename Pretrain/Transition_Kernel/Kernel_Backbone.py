import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Online-Diffusion-Planning/
PRETRAIN_DIR = PROJECT_ROOT / "Pretrain"
FINETUNE_DIR = PROJECT_ROOT / "Finetuning"
from numpy.matlib import std
from scipy.stats import median_abs_deviation
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Pretrain.Dataset import (
    CubeDataset_Singletask, 
    OGPointmazeDataset, 
    OGPointmazeDataset_Singletask, 
    AntmazeDataset,
    AntmazeDataset_Singletask,
    HumanoidmazeDataset,
    HumanoidmazeDataset_Singletask,
    CubeDataset, 
    SceneDataset, 
    SceneDataset_Singletask,
    PuzzleDataset,
    PuzzleDataset_Singletask
)
from .Kernel_Net import  RobustTransitionKernel, MoGTransitionKernel
from sympy import factorint
import pickle
import os
from typing import Optional, List
import math
import copy

try:
    from Pretrain.utils import SAStats, cycle, check_device
except ModuleNotFoundError:
    from utils import SAStats, cycle, check_device
import json

def check_specific_dataset(dataset_name):
    if(dataset_name in ['kitchen', 'scene']):
         return False
    elif dataset_name in ['pointmaze', 'cube', 'ogpointmaze', 'puzzle', 'antmaze', 'humanoidmaze']:
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
          if specific_env == 'medium':
               return 'AntMaze_Medium'
          elif specific_env == 'large':
               return 'AntMaze_Large'
          elif specific_env == 'giant':
               return 'AntMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     
     elif(env_name == 'humanoidmaze'):
          if specific_env == 'medium':
                return 'HumanoidMaze_Medium'
          elif specific_env == 'large':
                return 'HumanoidMaze_Large'
          elif specific_env == 'giant':
                return 'HumanoidMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'cube'):
          if specific_env == 'single':
               return 'Cube_Single'
          elif specific_env == 'double':
               return 'Cube_Double'
          elif specific_env == 'triple':
               return 'Cube_Triple'
          elif specific_env == 'quadruple':
               return 'Cube_Quadruple'
          else:
               raise ValueError(f"Invalid cube dataset name: {specific_env}")
     
     elif(env_name == 'puzzle'):
          if specific_env == '3x3':
                return 'Puzzle_3x3'
          elif specific_env == '4x4':
                return 'Puzzle_4x4'
          elif specific_env == '4x5':
                return 'Puzzle_4x5'
          elif specific_env == '4x6':
                return 'Puzzle_4x6'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'scene'):
         return 'Scene'
             
     elif(env_name == 'ogpointmaze'):
          if specific_env == 'medium':
               return 'OG2DMaze_Medium'
          elif specific_env == 'large':
               return 'OG2DMaze_Large'
          elif specific_env == 'giant':
               return 'OG2DMaze_Giant'
          else:
               raise ValueError(f"Invalid ogpointmaze dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def save_kernel_hyperparameters(dataset_name, batch_size, num_steps, lr, 
                                obs_dim, act_dim, kernel_name, optimizer, kernel_net, 
                                ensemble_size, λ_reg, specific_dataset: Optional[str] = None):
    
   
    """
    os.makedirs(f"./Pretrain/Transition_Kernel/{kernel_name}/args/", exist_ok=True)
    filepath = f"./Pretrain/Transition_Kernel/{kernel_name}/args/hyperparameters.json"
    """
    args_dir = PRETRAIN_DIR / "Transition_Kernel" / kernel_name / "args"
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
    """
    os.makedirs(f'./Pretrain/Transition_Kernel/{kernel_name}/Models/{num_steps}', exist_ok=True)
    save_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Models/{num_steps}/{kernel_name}_{num_steps}_{ensemble_idx}.pkl'
    """
    models_dir = PRETRAIN_DIR / "Transition_Kernel" / kernel_name / "Models" / str(num_steps)
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"{kernel_name}_{num_steps}_{ensemble_idx}.pkl"
    torch.save(net_dict, save_path)
    print(f"Kernel model save to {kernel_name}_{num_steps}_{ensemble_idx}.pkl")

def save_to_finetuning(kernel_net, dataset_name, ensemble_idx, specific_dataset: Optional[str] = None):
    kernel_net.eval()
    net_dict = kernel_net.state_dict()
    name = getName(dataset_name, specific_dataset)
    """
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Kernels/{dataset_name}/Models/{str(0)}', exist_ok=True)
        save_path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(0)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(0)}', exist_ok=True)
        save_path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(0)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    """
    if specific_dataset is None:
         ft_models_dir = FINETUNE_DIR / "Kernels" / dataset_name / "Models" / "0"
    else:
         ft_models_dir = FINETUNE_DIR / "Kernels" / dataset_name / specific_dataset / "Models" / "0"
    ft_models_dir.mkdir(parents=True, exist_ok=True)
    save_path = ft_models_dir / f"{name}_Kernel_{ensemble_idx}.pkl"
    torch.save(net_dict, save_path)
    print(f"kernel model save to {save_path}")

"""
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
"""

def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None):
    name = getName(dataset_name, specific_dataset)
    if specific_dataset is None:
        ft_stats_dir = FINETUNE_DIR / "Kernels" / dataset_name / "Stats"
    else:
        ft_stats_dir = FINETUNE_DIR / "Kernels" / dataset_name / specific_dataset / "Stats"
    ft_stats_dir.mkdir(parents=True, exist_ok=True)
    savepath = ft_stats_dir / f"{name}_Kernel_stats_0.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")
   

def check_trajs_exit(env_name, specific_env, task_id, step):
    from pathlib import Path
    if(step is not None):
         path = Path(f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/Generated_trajs_Info_{step}.pkl')
    else:
         path = Path(f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{step}.pkl')
    if not path.exists():
        return None
    else:
        with path.open('rb') as f:
             trajs = pickle.load(f)
        return trajs
    
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

"""
def load_model(kernel_name, num_steps, ensemble_idx):
    load_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Models/{num_steps}/{kernel_name}_{num_steps}_{ensemble_idx}.pkl'
    #state_dict = torch.load(load_path, map_location='cpu')
    state_dict = torch.load(load_path, weights_only=True)
    return state_dict
"""
def load_model(kernel_name, num_steps, ensemble_idx):
    load_path = (
        PRETRAIN_DIR
        / "Transition_Kernel"
        / kernel_name
        / "Models"
        / str(num_steps)
        / f"{kernel_name}_{num_steps}_{ensemble_idx}.pkl"
    )
    state_dict = torch.load(load_path, weights_only=True)
    return state_dict

def Train_Dataset(dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
    if(dataset_name == 'cube'):
        if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif(specific_dataset == 'single'):
             data_1 = CubeDataset('single-play')
             data_2 = CubeDataset('single-noisy')
             if(task_id is not None):
                 data_3 = CubeDataset_Singletask('single-play', task_id)
                 data_4 = CubeDataset_Singletask('single-noisy', task_id)
             name = 'Cube_Kernel_single'
        elif(specific_dataset == 'double'):
             data_1 = CubeDataset('double-play')
             data_2 = CubeDataset('double-noisy')
             if(task_id is not None):
                 data_3 = CubeDataset_Singletask('double-play', task_id)
                 data_4 = CubeDataset_Singletask('double-noisy', task_id)
             name = 'Cube_Kernel_double'
        elif(specific_dataset == 'triple'):
             data_1 = CubeDataset('triple-play')
             data_2 = CubeDataset('triple-noisy')
             if(task_id is not None):
                 data_3 = CubeDataset_Singletask('triple-play', task_id)
                 data_4 = CubeDataset_Singletask('triple-noisy', task_id)
             name = 'Cube_Kernel_triple'
        elif(specific_dataset == 'quadruple'):
             data_1 = CubeDataset('quadruple-play')
             data_2 = CubeDataset('quadruple-noisy')
             if(task_id is not None):
                 data_3 = CubeDataset_Singletask('quadruple-play', task_id)
                 data_4 = CubeDataset_Singletask('quadruple-noisy', task_id)
             name = 'Cube_Kernel_quadruple'
        else: 
            raise ValueError(f"Invalid dataset name: {specific_dataset}")
        if(task_id is not None):
            trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories() + data_4.get_trajectories()
        else:
            trajs = data_1.get_trajectories() + data_2.get_trajectories()
        obs_dim = data_1.get_state_dim()
        act_dim = data_1.get_action_dim()
        return trajs, name, obs_dim, act_dim
    
    elif(dataset_name == 'puzzle'):
        if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif(specific_dataset == '3x3'):
             data_1 = PuzzleDataset('3x3-play')
             data_2 = PuzzleDataset('3x3-noisy')
             if(task_id is not None):
                 data_3 = PuzzleDataset_Singletask('3x3-play', task_id)
                 data_4 = PuzzleDataset_Singletask('3x3-noisy', task_id)
             name = 'Puzzle_Kernel_3x3'
        elif(specific_dataset == '4x4'):
             data_1 = PuzzleDataset('4x4-play')
             data_2 = PuzzleDataset('4x4-noisy')
             if(task_id is not None):
                 data_3 = PuzzleDataset_Singletask('4x4-play', task_id)
                 data_4 = PuzzleDataset_Singletask('4x4-noisy', task_id)
             name = 'Puzzle_Kernel_4x4'
        elif(specific_dataset == '4x5'):
             data_1 = PuzzleDataset('4x5-play')
             data_2 = PuzzleDataset('4x5-noisy')
             if(task_id is not None):
                 data_3 = PuzzleDataset_Singletask('4x5-play', task_id)
                 data_4 = PuzzleDataset_Singletask('4x5-noisy', task_id)
             name = 'Puzzle_Kernel_4x5'
        elif(specific_dataset == '4x6'):
             data_1 = PuzzleDataset('4x6-play')
             data_2 = PuzzleDataset('4x6-noisy')
             if(task_id is not None):
                 data_3 = PuzzleDataset_Singletask('4x6-play', task_id)
                 data_4 = PuzzleDataset_Singletask('4x6-noisy', task_id)
             name = 'Puzzle_Kernel_4x6'
        else: 
            raise ValueError(f"Invalid dataset name: {specific_dataset}")
        if(task_id is not None):
            trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories() + data_4.get_trajectories()
        else:
            trajs = data_1.get_trajectories() + data_2.get_trajectories()
        obs_dim = data_1.get_state_dim()
        act_dim = data_1.get_action_dim()
        return trajs, name, obs_dim, act_dim

    elif(dataset_name == 'scene'):
        data_1 = SceneDataset('play')
        data_2 = SceneDataset('noisy')
        if(task_id is not None):
            data_3 = SceneDataset_Singletask('play', task_id)
            data_4 = SceneDataset_Singletask('noisy', task_id)
        name = 'Scene_Kernel'
        if(task_id is not None):
            trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories() + data_4.get_trajectories()
        else:
            trajs = data_1.get_trajectories() + data_2.get_trajectories()
        obs_dim = data_1.get_state_dim()
        act_dim = data_1.get_action_dim()
        return trajs, name, obs_dim, act_dim

    elif(dataset_name == 'ogpointmaze'):
        if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif(specific_dataset == 'medium'):
             data_1 = OGPointmazeDataset('medium')
             if(task_id is not None):
                 data_2 = OGPointmazeDataset_Singletask('medium', task_id)
             name = 'OG2DMaze_Kernel_medium'
        elif(specific_dataset == 'large'):
             data_1 =  OGPointmazeDataset('large')
             if(task_id is not None):
                 data_2 = OGPointmazeDataset_Singletask('large', task_id)
             name = 'OG2DMaze_Kernel_large'
        elif(specific_dataset == 'giant'):
             data_1 = OGPointmazeDataset('giant')
             if(task_id is not None):
                 data_2 = OGPointmazeDataset_Singletask('giant', task_id)
             name = 'OG2DMaze_Kernel_giant'
        else: 
            raise ValueError(f"Invalid dataset name: {specific_dataset}")
        if(task_id is not None):
            trajs = data_1.get_trajectories() + data_2.get_trajectories() 
        else:
            trajs = data_1.get_trajectories()
        obs_dim = data_1.get_state_dim()
        act_dim = data_1.get_action_dim()
        return trajs, name, obs_dim, act_dim
    
    elif(dataset_name == 'antmaze'):
        if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif(specific_dataset == 'medium'):
             data_1 = AntmazeDataset('medium')
             if(task_id is not None):
                 data_2 = AntmazeDataset_Singletask('medium', task_id)
             name = 'AntMaze_Kernel_medium'
        elif(specific_dataset == 'large'):
             data_1 =  AntmazeDataset('large')
             if(task_id is not None):
                 data_2 = AntmazeDataset_Singletask('large', task_id)
             name = 'AntMaze_Kernel_large'
        elif(specific_dataset == 'giant'):
             data_1 = AntmazeDataset('giant')
             if(task_id is not None):
                 data_2 = AntmazeDataset_Singletask('giant', task_id)
             name = 'AntMaze_Kernel_giant'
        else: 
            raise ValueError(f"Invalid dataset name: {specific_dataset}")
        if(task_id is not None):
            trajs = data_1.get_trajectories() + data_2.get_trajectories() 
        else:
            trajs = data_1.get_trajectories()
        obs_dim = data_1.get_state_dim()
        act_dim = data_1.get_action_dim()
        return trajs, name, obs_dim, act_dim
    
    elif(dataset_name == 'humanoidmaze'):
        if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif(specific_dataset == 'medium'):
             data_1 = HumanoidmazeDataset('medium')
             if(task_id is not None):
                 data_2 = HumanoidmazeDataset_Singletask('medium', task_id)
             name = 'HumanoidMaze_Kernel_medium'
        elif(specific_dataset == 'large'):
             data_1 =  HumanoidmazeDataset('large')
             if(task_id is not None):
                 data_2 = HumanoidmazeDataset_Singletask('large', task_id)
             name = 'HumanoidMaze_Kernel_large'
        elif(specific_dataset == 'giant'):
             data_1 = HumanoidmazeDataset('giant')
             if(task_id is not None):
                 data_2 = HumanoidmazeDataset_Singletask('giant', task_id)
             name = 'HumanoidMaze_Kernel_giant'
        else: 
            raise ValueError(f"Invalid dataset name: {specific_dataset}")
        if(task_id is not None):
            trajs = data_1.get_trajectories() + data_2.get_trajectories() 
        else:
            trajs = data_1.get_trajectories()
        obs_dim = data_1.get_state_dim()
        act_dim = data_1.get_action_dim()
        return trajs, name, obs_dim, act_dim

    else:
        raise ValueError(f"Invalid Dataset Name: {dataset_name}")   
             
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
            L = min(len(acts), len(obs) - 1)
            for t in range(L):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                data.append((s_t, a_t, s_tp1))
         self.data = data
         self.save_stats(kernel_name)
    """
    def save_stats(self, kernel_name):
        stats_name =  str(kernel_name) + '_stats.pkl'
        stats_dir = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")
    """
    def save_stats(self, kernel_name):
       stats_name = f"{kernel_name}_stats.pkl"
       stats_dir = PRETRAIN_DIR / "Transition_Kernel" / kernel_name / "Stats"
       stats_dir.mkdir(parents=True, exist_ok=True)
       savepath = stats_dir / stats_name
       with open(savepath, "wb") as f:
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
        """
        stats_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/{kernel_name}_stats.pkl'
        with open(stats_path, 'rb') as f:
              self.stats = pickle.load(f)
        """
        stats_path = PRETRAIN_DIR / "Transition_Kernel" / kernel_name / "Stats" / f"{kernel_name}_stats.pkl"
        with open(stats_path, "rb") as f:
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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from itertools import cycle
from tqdm import tqdm


def train_mog_kernel(
    dataset_name: str,
    specific_dataset: str = None,
    task_id: Optional[int] = None,
    trajs: Optional[list] = None,
    batch_size: int = 256,
    lr: float = 1e-4,
    num_steps: int = 25000,
    save_freq: int = 2000,
    ensemble_size: int = 6,           # 5~8 recommended
    num_modes: int = 8,  
    num_hidden_layers: int = 3,             # 6~8 recommended for manipulation
    hidden_dim: int = 512,
    λ_reg: float = 2e-3,              # disagreement regularization
    noise_floor: float = 1e-6,
    device=None
):
    device = check_device()

    print(f"Training MoG Transition Kernel for {dataset_name}")
    if specific_dataset:
        print(f"  Specific dataset: {specific_dataset}")

    # Prepare dataset
    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id)
    if(trajs is not None):
          total_trajs = train_trajs + trajs
    else:
          total_trajs = train_trajs 
    dataset = KernelDataset(total_trajs, kernel_name)
    loader = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True,
                              pin_memory=True, num_workers=8, persistent_workers = True))
    
    # Create ensemble of MoG kernels
    ensemble = [
        MoGTransitionKernel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_modes = num_modes,
            num_hidden_layers = num_hidden_layers,
            hidden_dim = hidden_dim, 
            noise_floor = noise_floor
        ).to(device)
        for _ in range(ensemble_size)
    ]

    optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5) 
                  for m in ensemble]

    # Save hyperparameters (you may need to adjust this function for MoG)
    save_kernel_hyperparameters(
        dataset_name,
        batch_size,
        num_steps,
        lr,
        obs_dim,
        act_dim,
        kernel_name,
        optimizers[0],
        ensemble[0],
        ensemble_size,
        λ_reg,
        specific_dataset=specific_dataset
    )

    SD = specific_dataset if check_specific_dataset(dataset_name) else None

    step = 0
    total_loss = 0.0

    for step in tqdm(range(1, num_steps + 1), desc="Training MoG Kernel"):
        s, a, s_next = next(loader)
        s = s.to(device)
        a = a.to(device)
        s_next = s_next.to(device)

        losses = []

        for m in ensemble:
            mu, log_std, weights = m(s, a)
            loss = m.mog_nll(s_next, mu, log_std, weights)

            # === Optional: disagreement regularization ===
            # Average over modes for disagreement calculation
            mu_mean = mu.mean(dim=1)                    # (B, obs_dim)
            disagreement = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1).mean(dim=0)
            
            var = torch.exp(2 * log_std) + m.noise_floor
            penalty = (disagreement / (var.mean(dim=1) + 1e-6)).mean()
            
            loss = loss + λ_reg * penalty
            losses.append(loss)

        # Backprop
        for m, opt, loss in zip(ensemble, optimizers, losses):
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=5.0)
            opt.step()

        # Logging
        avg_loss = sum(loss.item() for loss in losses) / ensemble_size
        total_loss += avg_loss

        if step % 100 == 0:
            print(f"Step {step:6d} | Avg Loss: {total_loss/100:.6f}")
            total_loss = 0.0

        # Save checkpoints
        if step % save_freq == 0 or step == num_steps:
            for idx, m in enumerate(ensemble):
                ckpt = copy.deepcopy(m).cpu()
                save_model(ckpt, f"{kernel_name}", step, idx)

            if step == num_steps:
                for idx, m in enumerate(ensemble):
                    ckpt = copy.deepcopy(m).cpu()
                    save_to_finetuning(ckpt, dataset_name, idx, SD)

                stats = get_pretrained_kernel_stats(f"{kernel_name}")
                save_stats_to_finetuning(stats, dataset_name, SD)

    print("MoG Transition Kernel training completed!")
    return ensemble


def train_kernel(dataset_name, specific_dataset: str = None,
                 batch_size=256, lr=1e-3, num_steps=10000, save_freq = 200,
                 ensemble_size=10, hidden_layers = 2, hidden_dim = 256, λ_reg=1e-3, trajs: Optional[list] = None):
    # Prepare dataset / dataloader
    if specific_dataset is None:
        print(f"Training kernel for {dataset_name}")
    else:
        print(f"Training kernel for {dataset_name}_{specific_dataset}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = check_device()
    print("Using device:", device)
    if(trajs is None):
           trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    dataset = KernelDataset(trajs, kernel_name)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=8))

    # Create ensemble of models
    ensemble = [RobustTransitionKernel(obs_dim, act_dim, hidden_layers, hidden_dim).to(device) for _ in range(ensemble_size)]
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
    if(check_specific_dataset(dataset_name)):
        SD = specific_dataset
    else:
        SD = None
    step = 0
    total_loss = 0.0

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
                save_freq: int = 50, num_steps: int = 500, hidden_layers = 2, hidden_dim = 256, ensemble_size = 3, quantile = 0.999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = check_device()
    print("Using device:", device)

    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    if trajs is None:
        dataset = test_dataset(train_trajs, kernel_name)
    else:
        dataset = test_dataset(trajs, kernel_name)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    
    # For each saved checkpoint / ensemble member
    step = save_freq
    while step <= num_steps:
        # Load ensemble members
        ensemble = []
        for idx in range(ensemble_size):
            state_dict = load_model(kernel_name, step, idx)
            m = RobustTransitionKernel(obs_dim, act_dim, hidden_layers, hidden_dim).to(device)
            m.load_state_dict(state_dict)
            m.eval()
            ensemble.append(m)

        # Compute log-probs over dataset
        all_D2_total = []
        all_log_density = []
        #all_D_total = []
        count = 0
        #worst = (None, float("inf"), None)  # (idx, log_prob, (s, a, s_next))
        for i, (s, a, s_next) in enumerate(dataloader):
            s = s.to(device)
            a = a.to(device)
            s_next = s_next.to(device)

            #compute total mahalanobis distance
            with torch.no_grad():
                D2_total = compute_total_mahalanobis_score(ensemble, s, a, s_next)
                log_density = compute_log_density(ensemble, s, a, s_next)
            D2 = D2_total.detach().cpu().numpy()
            log_density = log_density.detach().cpu().numpy()
            all_D2_total.extend(D2)
            all_log_density.extend(log_density)
            count += 1
        
        print('Mahalanobis Distance')
        all_D2_total = np.array(all_D2_total)
        mean_D2_total = float(all_D2_total.mean())
        min_D2_total = float(all_D2_total.min())
        max_D2_total = float(all_D2_total.max())
        std_D2_total = float(all_D2_total.std())
        tau = float(np.quantile(all_D2_total, quantile))
        print(f"Checkpoint {step}")
        print(f"mean_D2_total = {mean_D2_total:.4f}")
        print(f"min_D2_total = {min_D2_total:.4f}")
        print(f"max_D2_total = {max_D2_total:.4f}")
        print(f"std_D2_total = {std_D2_total:.4f}")
        print(f"τ ({quantile*100:.0f}th percentile) : {tau:.4f}")
        
        print('Log Density')
        all_log_density = np.array(all_log_density)
        mean_log_density = float(all_log_density.mean())
        min_log_density = float(all_log_density.min())
        max_log_density = float(all_log_density.max())
        std_log_density = float(all_log_density.std())
        tau = float(np.quantile(all_log_density, 1 - quantile))
        print(f"Checkpoint {step}")
        print(f"mean_log_density = {mean_log_density:.4f}")
        print(f"min_log_density = {min_log_density:.4f}")
        print(f"max_log_density = {max_log_density:.4f}")
        print(f"std_log_density = {std_log_density:.4f}")
        print(f"τ ({(1-quantile)*100:.0f}th percentile) : {tau:.4f}")
        step += save_freq


def test_kernel_mog(dataset_name, specific_dataset: str = None, task_id: Optional[int] = None,
                trajs: list = None,
                save_freq: int = 50, num_steps: int = 500, num_hidden_layers = 2, hidden_dim = 256, ensemble_size = 3, num_modes = 9, noise_floor = 1e-6, quantile = 0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = check_device()
    print("Using device:", device)

    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id)
    if trajs is not None:
        total_trajs = train_trajs + trajs
    else:
        total_trajs = train_trajs
    dataset = test_dataset(total_trajs, kernel_name)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    
    # For each saved checkpoint / ensemble member
    step = save_freq
    while step <= num_steps:
        # Load ensemble members
        ensemble = []
        for idx in range(ensemble_size):
            state_dict = load_model(kernel_name, step, idx)
            m = MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, noise_floor).to(device)
            m.load_state_dict(state_dict)
            m.eval()
            ensemble.append(m)

        # Compute log-probs over dataset
        all_D2_total = []
        all_log_density = []
        #all_D_total = []
        count = 0
        #worst = (None, float("inf"), None)  # (idx, log_prob, (s, a, s_next))
        for i, (s, a, s_next) in enumerate(dataloader):
            s = s.to(device)
            a = a.to(device)
            s_next = s_next.to(device)

            #compute total mahalanobis distance
            with torch.no_grad():
                D2_total = compute_total_mahalanobis_score_mog(ensemble, s, a, s_next)
                log_density = compute_log_density_mog(ensemble, s, a, s_next)
            D2 = D2_total.detach().cpu().numpy()
            log_density = log_density.detach().cpu().numpy()
            all_D2_total.extend(D2)
            all_log_density.extend(log_density)
            count += 1
        
        print('Mahalanobis Distance')
        all_D2_total = np.array(all_D2_total)
        mean_D2_total = float(all_D2_total.mean())
        min_D2_total = float(all_D2_total.min())
        max_D2_total = float(all_D2_total.max())
        std_D2_total = float(all_D2_total.std())
        tau = float(np.quantile(all_D2_total, quantile))
        print(f"Checkpoint {step}")
        print(f"mean_D2_total = {mean_D2_total:.4f}")
        print(f"min_D2_total = {min_D2_total:.4f}")
        print(f"max_D2_total = {max_D2_total:.4f}")
        print(f"std_D2_total = {std_D2_total:.4f}")
        print(f"τ ({quantile*100:.0f}th percentile) : {tau:.4f}")
        
        print('Log Density')
        all_log_density = np.array(all_log_density)
        mean_log_density = float(all_log_density.mean())
        min_log_density = float(all_log_density.min())
        max_log_density = float(all_log_density.max())
        std_log_density = float(all_log_density.std())
        tau = float(np.quantile(all_log_density, 1 - quantile))
        print(f"Checkpoint {step}")
        print(f"mean_log_density = {mean_log_density:.4f}")
        print(f"min_log_density = {min_log_density:.4f}")
        print(f"max_log_density = {max_log_density:.4f}")
        print(f"std_log_density = {std_log_density:.4f}")
        print(f"τ ({(1-quantile)*100:.0f}th percentile) : {tau:.4f}")
        step += save_freq



def get_pretrained_kernel(dataset_name, checkpoints, specific_dataset: Optional[str] = None):
       _, name, obs_dim, act_dim  =  Train_Dataset(dataset_name, specific_dataset)
       """
       path = f'./Pretrain/Transition_Kernel/{name}/Models/{checkpoints}'
       file_count = count_files_in_folder(path)
       """
       path = PRETRAIN_DIR / "Transition_Kernel" / name / "Models" / str(checkpoints)
       file_count = count_files_in_folder(str(path))
       kernel_state_dicts = []
       for i in range(file_count):
           kernel_state_dicts.append(load_model(name, checkpoints, i))
       return kernel_state_dicts, obs_dim, act_dim, name

def get_pretrained_kernel_stats(kernel_name):
    """
     stats_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/{kernel_name}_stats.pkl'
     with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
     return stats
    """
    stats_path = PRETRAIN_DIR / "Transition_Kernel" / kernel_name / "Stats" / f"{kernel_name}_stats.pkl"
    with open(stats_path, "rb") as f:
          stats = pickle.load(f)
    return stats

"""
def compute_total_mahalanobis_score(
    kernels: List[RobustTransitionKernel],
    s: torch.Tensor,
    a: torch.Tensor,
    s_next: torch.Tensor,
) -> torch.Tensor:
    
    K = len(kernels)
    device = s.device

    # === 1. Vectorized ensemble forward (fastest when gradients are needed) ===
    # Stack all models into one batched forward
    def single_model_forward(k: RobustTransitionKernel):
        return k(s, a)   # returns (mu, log_std)

    # Use vmap (PyTorch ≥ 2.0) if available — this is the most efficient way
    if hasattr(torch, "vmap"):
        mus, log_stds = torch.vmap(single_model_forward)(kernels)   # (K, B, dim)
    else:
        # Fallback: manual loop (still fast and fully differentiable)
        mus = []
        log_stds = []
        for k in kernels:
            mu, log_std = k(s, a)
            mus.append(mu)
            log_stds.append(log_std)
        mus = torch.stack(mus, dim=0)
        log_stds = torch.stack(log_stds, dim=0)

    # === 2. Total predictive statistics (all in autograd graph) ===
    mu_total = mus.mean(dim=0)                                      # (B, obs_dim)

    var_aleatoric = (torch.exp(2 * log_stds) + kernels[0].noise_floor).mean(dim=0)

    var_epistemic = mus.var(dim=0, unbiased=False)                  # population variance

    var_total = var_aleatoric + var_epistemic
    var_total = torch.clamp(var_total, min=1e-8)

    # === 3. Mahalanobis distance (fully differentiable) ===
    residual = s_next - mu_total
    residual = torch.clamp(residual, -10.0, 10.0)

    D2_total = ((residual ** 2) / var_total).sum(dim=-1)            # (B,)

    return D2_total
"""

def compute_total_mahalanobis_score(kernels: List[RobustTransitionKernel], s, a, s_next):
    mus = []
    log_stds = []
    for kernel in kernels:
            mu, log_std = kernel(s, a)          # (B, obs_dim)
            mus.append(mu)
            log_stds.append(log_std)
    # Stack -> (K, B, obs_dim)
    mus = torch.stack(mus, dim=0)
    log_stds = torch.stack(log_stds, dim=0)
    # 1. Total mean
    mu_total = mus.mean(dim=0)                    # (B, obs_dim)
    # 2. Aleatoric variance (average predicted variance)
    var_aleatoric = (torch.exp(2 * log_stds) + kernels[0].noise_floor).mean(dim=0)
    # 3. Epistemic variance (disagreement of means)
    var_epistemic = mus.var(dim=0, unbiased=False)   # population variance (common in MBRL)
    # 4. Total variance
    var_total = var_aleatoric + var_epistemic
    var_total = torch.clamp(var_total, min=1e-8)
    # 5. Squared Mahalanobis Distance (Total Score)
    residual = s_next - mu_total
    #residual = torch.clamp(residual, -10.0, 10.0)   # stability
    D2_total = ((residual ** 2) / var_total).sum(dim=-1)   # (B,)
    return D2_total

def compute_log_density(kernels: List[RobustTransitionKernel], s, a, s_next):
    log_probs = []
    for kernel in kernels:
        mu, log_std = kernel(s, a)
        lp = kernel.log_prob(s_next, mu, log_std)
        log_probs.append(lp)
    #log_probs = torch.stack(log_probs, dim=0).mean(dim = 0)
    log_probs = torch.stack(log_probs, dim=0)
    log_density = torch.logsumexp(log_probs, dim=0) - math.log(len(kernels)) 
    return log_density
    #return log_probs

def compute_log_density_mog(kernels: List[MoGTransitionKernel], s, a, s_next):
    """Returns total log p(s'|s,a) under ensemble of MoGs"""
    all_log_probs = []
    
    for kernel in kernels:
        mu, log_std, weights = kernel(s, a)
        lp = kernel.log_prob(s_next, mu, log_std, weights)   # must use this method
        all_log_probs.append(lp)
    
    all_log_probs = torch.stack(all_log_probs, dim=0)            # (K_ens, B)
    
    # Proper ensemble logsumexp
    log_density = torch.logsumexp(all_log_probs, dim=0) - math.log(len(kernels))
    
    return log_density


def compute_total_mahalanobis_score_mog(
    kernels: list, 
    s: torch.Tensor, 
    a: torch.Tensor, 
    s_next: torch.Tensor
) -> torch.Tensor:
    """
    MoG-compatible Total Mahalanobis Distance.
    """
    K_ens = len(kernels)                    # number of ensemble members
    B = s.shape[0]
    
    mu_list = []
    var_list = []
    
    for kernel in kernels:
        mu, log_std, weights = kernel(s, a)           # mu: (B, K_modes, obs_dim)
                                                      # weights: (B, K_modes)
        
        K_modes = weights.shape[1]
        
        # === Mixture statistics for this model ===
        # Weighted mean
        mu_mix = torch.sum(weights.unsqueeze(-1) * mu, dim=1)          # (B, obs_dim)
        
        # Aleatoric variance: E[Var]
        var_ale = torch.exp(2 * log_std) + kernel.noise_floor          # (B, K_modes, obs_dim)
        var_ale_mix = torch.sum(weights.unsqueeze(-1) * var_ale, dim=1)  # (B, obs_dim)
        
        # Epistemic variance: Var[E]
        mu_centered = mu - mu_mix.unsqueeze(1)                         # (B, K_modes, obs_dim)
        var_epi_mix = torch.sum(weights.unsqueeze(-1) * (mu_centered ** 2), dim=1)
        
        var_mix = var_ale_mix + var_epi_mix
        var_mix = torch.clamp(var_mix, min=1e-6)
        
        mu_list.append(mu_mix)
        var_list.append(var_mix)
    
    # === Ensemble level ===
    mu_ensemble = torch.stack(mu_list, dim=0)           # (K_ens, B, obs_dim)
    var_ensemble = torch.stack(var_list, dim=0)         # (K_ens, B, obs_dim)
    
    mu_total = mu_ensemble.mean(dim=0)                  # (B, obs_dim)
    
    var_aleatoric = var_ensemble.mean(dim=0)
    var_epistemic = mu_ensemble.var(dim=0, unbiased=False)
    
    var_total = var_aleatoric + var_epistemic
    var_total = torch.clamp(var_total, min=1e-6)
    
    # === Mahalanobis ===
    residual = s_next - mu_total
    residual = torch.clamp(residual, -10.0, 10.0)
    
    D2_total = ((residual ** 2) / var_total).sum(dim=-1)   # (B,)
    
    return D2_total

