'''Reward backbone training/eval for ODP (JAX/Flax port, FQL-style).

Faithful torch->JAX conversion of the reward training/eval pipeline. Public API (class/function names,
call signatures, hyperparameters, magic numbers) is preserved exactly; only the framework internals
change. The reward nets (SimpleReward/EnsembleReward) are linen modules from Pretrain/Rewards/nets.py
and are built via `model_def.init(rng, *example)` + a TrainState, not torch `.to(device)`/`.parameters()`.
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
from typing import Optional

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax

from Dataset import CubeDataset_Singletask, KitchenDataset, PointMazeDataset, get_dataset, get_env, CubeDataset, OGPointmazeDataset_Singletask
import random
try:
    from Pretrain.utils import set_seed, SAStats, ema_smooth, cycle, check_device
except ModuleNotFoundError:
    from utils import set_seed, SAStats, ema_smooth, cycle, check_device
import pickle
try:
    from Pretrain.Rewards.nets import Reward, MLPNetwork, ScalarReward, SimpleReward, EnsembleReward
    from Pretrain.Dataset import TrajectoryDict
except ModuleNotFoundError:
    from Rewards.nets import Reward, MLPNetwork, ScalarReward, SimpleReward, EnsembleReward
    from Dataset import TrajectoryDict
import os
from scipy.ndimage import gaussian_filter1d, convolve
import copy
from sympy import Predicate, factorint
import json
from typing import TypedDict, List

# Shared port plumbing (mirrors fql).
from JAX_PORT.jax_utils import (
    MLP, ModuleDict, TrainState, nonpytree_field, default_init, ensemblize,
    target_update, save_agent, restore_agent, supply_rng,
)


def make_reward_increase(trajs) -> List[TrajectoryDict]:
     def separate_traj(traj) -> TrajectoryDict :
         new_trajs = []
         last_start = 0
         for i in range(1, len(traj['rewards'])):
            if(traj['rewards'][i] < traj['rewards'][i-1]):
                 new_traj = {
                     'observations': traj['observations'][last_start: i].copy(),
                     'actions': traj['actions'][last_start: i].copy(),
                     'rewards': traj['rewards'][last_start: i].copy(),
                 }
                 new_trajs.append(new_traj)
                 last_start = i
         if last_start < len(traj['rewards']):
             new_trajs.append({
                  'observations': traj['observations'][last_start:].copy(),
                  'actions': traj['actions'][last_start:].copy(),
                  'rewards': traj['rewards'][last_start:].copy(),
              })
         return new_trajs
     new_trajs = []
     for traj in trajs:
         new_trajs.extend(separate_traj(traj.copy()))
     return new_trajs

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray

def divide_trajs(trajs):
    success_trajs = []
    failed_trajs = []
    for traj in trajs:
        if(traj['rewards'][-1] == 1.0):
            success_trajs.append(traj)
        else:
            failed_trajs.append(traj)
    return success_trajs, failed_trajs

def drop_trajs(trajs, percentage):
    success_trajs, failed_trajs = divide_trajs(trajs)
    random.shuffle(failed_trajs)
    failed_trajs = failed_trajs[:int(len(failed_trajs) * percentage)]
    return success_trajs + failed_trajs

def check_specific_dataset(dataset_name):
    if(dataset_name == 'kitchen'):
         return False
    elif dataset_name in ['pointmaze', 'cube', 'ogpointmaze']:
        return True

def get_trajs(env_name, specific_env, step, task_id: Optional[int] = None):
    if(task_id is not None):
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{str(task_id)}/Generated_trajs_Info_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{str(step)}.pkl'
    with open(path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs

def getName(env_name, specific_env, task_id: Optional[int] = None):
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
     elif(env_name == 'cube'):
         if(task_id is None):
            raise ValueError('Task ID is required for cube dataset')
         if specific_env == 'single' or specific_env == 'single-play':
              return f'Cube_Single_Task{task_id}'
         elif specific_env == 'double'  or specific_env == 'double-play':
              return f'Cube_Double_Task{task_id}'
         elif specific_env == 'triple' or specific_env == 'triple-play':
              return f'Cube_Triple_Task{task_id}'
         elif specific_env == 'quadruple' or specific_env == 'quadruple-play':
              return f'Cube_Quadruple_Task{task_id}'
         else:
              raise ValueError(f"Invalid cube dataset name: {specific_env}")
     elif(env_name == 'ogpointmaze'):
         if(task_id is None):
            raise ValueError('Task ID is required for ogpointmaze dataset')
         if specific_env == 'medium':
              return f'OG2DMaze_Medium_Task{task_id}'
         elif specific_env == 'large'  :
              return f'OG2DMaze_Large_Task{task_id}'
         elif specific_env == 'giant':
              return  f'OG2DMaze_Giant_Task{task_id}'
         else:
              raise ValueError(f"Invalid ogpointmaze dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def get_reward_name(dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
    name = getName(dataset_name, specific_dataset, task_id)
    reward_name = f"{name}_Reward"
    return reward_name

def save_reward_hyperparameters(dataset_name, batch_size, num_steps, lr, sigma, alpha,
                                  obs_dim, act_dim, reward_name, optimizer, reward_net, filepath: Optional[str] = None,
                                  specific_dataset: Optional[str] = None, target_reward: Optional[float] = None,
                                  goal: Optional[np.array] = None, task_id: Optional[int] = None,
                                  pos_weight: Optional[float] = None,
                                  ensemble_size: Optional[int] = None):

    if filepath is None:
        os.makedirs(f"./Pretrain/Rewards/{reward_name}/args/", exist_ok=True)
        filepath = f"./Pretrain/Rewards/{reward_name}/args/hyperparameters.json"


    def convert_to_json_serializable(obj):
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
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

    # Get optimizer info. In the JAX port `optimizer` is a small dict of optax hyperparameters
    # ({'type': ..., 'weight_decay': ...}); fall back gracefully if a raw optax tx is passed.
    if isinstance(optimizer, dict):
        optimizer_type = optimizer.get('type', 'AdamW')
        weight_decay = optimizer.get('weight_decay', 0)
    else:
        optimizer_type = type(optimizer).__name__
        weight_decay = 0
    optimizer_params = {
        'type': optimizer_type,
        'lr': lr,
        'weight_decay': weight_decay,
    }

    # Prefer the explicit kwarg; otherwise infer from the model itself.
    if ensemble_size is None and hasattr(reward_net, 'ensemble_size'):
        ensemble_size = int(reward_net.ensemble_size)

    # Get model architecture info
    model_info = {
        'model_type': type(reward_net).__name__,
        'obs_dim': int(obs_dim),
        'act_dim': int(act_dim),
    }

    # Add model-specific parameters if available
    if hasattr(reward_net, 'hidden_dim'):
        model_info['hidden_dim'] = int(reward_net.hidden_dim)
    if hasattr(reward_net, 'hidden_layers'):
        model_info['hidden_layers'] = int(reward_net.hidden_layers)
    if hasattr(reward_net, 'num_layers'):
        model_info['num_layers'] = int(reward_net.num_layers)
    if hasattr(reward_net, 'output_dim'):
        model_info['output_dim'] = int(reward_net.output_dim)
    if ensemble_size is not None:
        model_info['ensemble_size'] = int(ensemble_size)

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
            'pos_weight': pos_weight,
            'ensemble_size': ensemble_size,
        },
        'reward_processing': {
            'sigma': float(sigma) if sigma is not None else None,
            'alpha': float(alpha) if alpha is not None else None,
            'target_reward': target_reward,
            'goal': convert_to_json_serializable(goal),
            'task_id': task_id
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
    for i in range(1, len(obs)):
        pos = obs[i][:2] 
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        #goal_coord = np.asarray(goal_coord, dtype=np.float32).reshape(-1)  
        dist = np.linalg.norm(pos - g) 
        if (dist < 0.5):
            rews[i-1] = 1
        else:
            rews[i-1] = 0
    return rews

def save_to_finetuning(reward_net, dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
    # `reward_net` is a flax params pytree (or TrainState) for the converted linen reward net.
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    net_dict = flax.serialization.to_state_dict(reward_net)
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Models/', exist_ok=True)
        save_path = f'./Finetuning/Rewards/{dataset_name}/Models/{reward_name}_{str(0)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
        save_path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{reward_name}_{str(0)}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(net_dict, f)
    print(f"reward model save to {save_path}")

def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
    #name = getName(dataset_name, specific_dataset)
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    if(specific_dataset is None):
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Stats/', exist_ok=True)
        savepath = f'./Finetuning/Rewards/{dataset_name}/Stats/{reward_name}_stats_{str(0)}.pkl'
    else:
        os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/', exist_ok=True)
        savepath = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/{reward_name}_stats_{str(0)}.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")

def save_model(reward_net, dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None, num_steps: int = 0):
    # `reward_net` is a flax params pytree (or TrainState) for the converted linen reward net.
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    net_dict = flax.serialization.to_state_dict(reward_net)
    os.makedirs(f'./Pretrain/Rewards/{reward_name}/Models/', exist_ok=True)
    save_path = f'./Pretrain/Rewards/{reward_name}/Models/{reward_name}_{num_steps}.pkl'
    print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    with open(save_path, 'wb') as f:
        pickle.dump(net_dict, f)
    print(f"reward model save to {reward_name}_{num_steps}.pkl")

def load_model( dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None, num_steps: int = 0):
    #load_path = f'./Pretrain/Rewards/{reward_name}/Models/{reward_name}_{num_steps}.pkl'
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    load_path = os.path.join(
        project_root,
        "Pretrain",
        "Rewards",
        reward_name,
        "Models",
        f"{reward_name}_{num_steps}.pkl",
    )
    # TODO(checkpoint-bridge): pre-port checkpoints were torch state_dicts written via torch.save; the
    # JAX port writes flax serialization state_dicts via pickle. Ingesting a legacy torch .pkl requires
    # remapping torch Linear/LayerNorm keys to the flax param tree (see CONVERSION_GUIDE.md §10).
    with open(load_path, 'rb') as f:
        state_dict = pickle.load(f)
    return state_dict

def check_trajs_exit(env_name, specific_env, task_id, step):
    from pathlib import Path
    if(task_id is not None):
         path = Path(f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/Generated_trajs_Info_{step}.pkl')
    else:
         path = Path(f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{step}.pkl')
    if not path.exists():
        return None
    else:
        with path.open('rb') as f:
             trajs = pickle.load(f)
        return trajs
    
def Train_Dataset(dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None, goal: Optional[np.array] = None, traj_length: Optional[int] = None):
    from Dataset import KitchenDataset, PointMazeDataset, CubeDataset
    if(dataset_name == 'kitchen'):
         data_1 = KitchenDataset('complete')
         data_2 = KitchenDataset('partial')
         data_3 = KitchenDataset('mixed')
         trajs = data_1.get_trajectories() + data_2.get_trajectories() + data_3.get_trajectories()
        # trajs = data_1.get_trajectories()
         name = 'Kitchen_Reward'
         obs_dim = data_1.get_state_dim()
         act_dim = data_1.get_action_dim()
         return trajs, name, obs_dim, act_dim
     
    elif(dataset_name == 'pointmaze'):
         if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         elif(specific_dataset == 'large'):
              data = PointMazeDataset('large', goal, mode = 'reward')
              name = '2DMaze_Reward_large'
         elif(specific_dataset == 'medium'):
              data = PointMazeDataset('medium', goal, mode = 'reward')
              name = '2DMaze_Reward_medium'
         elif(specific_dataset == 'umaze'):
              data = PointMazeDataset('umaze', goal, mode = 'reward')
              name = '2DMaze_Reward_umaze'
         else: 
              raise ValueError(f"Invalid dataset name: {specific_dataset}")
         obs_dim = data.get_state_dim()
         act_dim = data.get_action_dim()
         trajs = data.get_trajectories()
         return trajs, name, obs_dim, act_dim

    elif(dataset_name == 'ogpointmaze'):
         if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         elif(specific_dataset == 'medium'):
              data = OGPointmazeDataset_Singletask('medium', task_id, mode = 'reward')
              name = f'OG2DMaze_Reward_medium_task{task_id}'
         elif(specific_dataset == 'large'):
              data = OGPointmazeDataset_Singletask('large', task_id, mode = 'reward')
              name = f'OG2DMaze_Reward_large_task{task_id}'
         elif(specific_dataset == 'giant'):
              data = OGPointmazeDataset_Singletask('giant', task_id, mode = 'reward')
              name = f'OG2DMaze_Reward_giant_task{task_id}'
         else: 
              raise ValueError(f"Invalid dataset name: {specific_dataset}")
         obs_dim = data.get_state_dim()
         act_dim = data.get_action_dim()
         trajs = data.get_trajectories()
         #trajs = make_reward_increase(trajs)
         return trajs, name, obs_dim, act_dim

    elif(dataset_name == 'cube'):
         if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
         elif(specific_dataset == 'single'):
             data_1 = CubeDataset_Singletask('single-play', task_id, traj_length)
             data_2 = CubeDataset_Singletask('single-noisy', task_id, traj_length)
             name = f'Cube_Reward_single_task{task_id}'
         elif(specific_dataset == 'double'):
             data_1 = CubeDataset_Singletask('double-play', task_id, traj_length)
             data_2 = CubeDataset_Singletask('double-noisy', task_id, traj_length)
             name = f'Cube_Reward_double_task{task_id}'
         elif(specific_dataset == 'triple'):
             data_1 = CubeDataset_Singletask('triple-play', task_id, traj_length)
             data_2 = CubeDataset_Singletask('triple-noisy', task_id, traj_length)
             name = f'Cube_Reward_triple_task{task_id}'
         else: 
              raise ValueError(f"Invalid dataset name: {specific_dataset}")
         obs_dim = data_1.get_state_dim()
         act_dim = data_1.get_action_dim()
         trajs = data_1.get_trajectories() + data_2.get_trajectories()
         #trajs = make_reward_increase(trajs)
         return trajs, name, obs_dim, act_dim

    else:
         raise ValueError(f"Invalid dataset name: {dataset_name}")
         
def reward_filter_goals(trajs: List[TrajectoryDict], goal) -> List[TrajectoryDict]:
    def reward_filter2(traj: TrajectoryDict, goal) -> List[TrajectoryDict]:
        last_step = 1
        #i = 1
        new_trajs = []
        new_rews = [0]*len(traj['rewards'])
        traj['rewards'] = new_rews
        
        #while(i < len(traj['observations'])):
        for i in range(1, len(traj['observations'])):
          pos = traj['observations'][i][:2]
          g = np.asarray(goal, dtype=np.float32).reshape(-1)
          dist = np.linalg.norm(pos - g) 
          if(dist < 0.5):
              
              if((i - last_step) < 70):
                  #last_step = i+1
                  continue
              else:
                 rews = traj['rewards'][last_step:i-1]
                 rews[-1] = 1.0
                 new_trajs.append({'observations': traj['observations'][last_step:i-1], 'actions': traj['actions'][last_step:i-1], 'rewards': rews})
                 last_step = i+1
        return new_trajs
    
    new_trajs = []
    for traj in trajs:
        new_trajs.extend(reward_filter2(traj, goal))
    new_trajs2 = []
    for traj in new_trajs:
        new_trajs2.append({'observations':traj['observations'][-200:], 'actions': traj['actions'][-200:], 'rewards': traj['rewards'][-200:]})
    return new_trajs2

class RewardDataset:
    def __init__(self, trajs, reward_name, sigma: Optional[float] = None, alpha: Optional[float] = None, target_reward: Optional[float] = None):


        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        for traj in trajs:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis=0)  # [N, d_s]

        allowed_values = [0.0, 1.0]
        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8

        transitions = []
        for traj in trajs:
            obs = np.asarray(traj['observations'])
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            if(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            elif(alpha is not None):
                rews = ema_smooth(rews, alpha)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
        # Stacked numpy arrays for fql-style batched sampling (host-side numpy; §13).
        self._obs = np.asarray([t[0] for t in transitions], dtype=np.float32)
        self._act = np.asarray([t[1] for t in transitions], dtype=np.float32)
        self._rew = np.asarray([t[2] for t in transitions], dtype=np.float32)
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
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r = self.transitions[idx]
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(r, dtype=np.float32),
        )

    def sample(self, batch_size):
        # fql-style host-side numpy sampling (random index sampling via np.random; see CONVERSION_GUIDE §13).
        idx = np.random.randint(0, len(self._obs), size=batch_size)
        return self._obs[idx], self._act[idx], self._rew[idx]


def _reward_batches(dataset, batch_size):
    # Generator of (s, a, r) numpy batches; wrap with `cycle` to mimic the old infinite DataLoader.
    while True:
        yield dataset.sample(batch_size)


def _eval_batches(dataset, batch_size):
    # One shuffled full pass over the dataset in fixed-size numpy chunks (replaces eval DataLoader).
    n = len(dataset)
    order = np.random.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield dataset._obs[idx], dataset._act[idx], dataset._rew[idx]

def train_reward(dataset_name: str, hidden_layers: int, hidden_dim: int, batch_size, num_steps, save_freq, lr, min_lr, sigma: Optional[float] = None, alpha: Optional[float] = None, target_reward: Optional[float] = None, specific_dataset: Optional[str] = None, task_id: Optional[int] = None, goal: Optional[np.array] = None,traj_length: Optional[int] = None, trajs: Optional[list] = None, *, rng=None):
    # API-CHANGE: trailing keyword-only `rng=` added (linen param init is stochastic; see CONVERSION_GUIDE §8).
    device = check_device()
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    if(trajs is  None):
         trajs, _, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id, goal, traj_length)
    else:
         train_trajs, _, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id, goal, traj_length)
         trajs = trajs + train_trajs
    print(f"Training reward approximator for {dataset_name} Dataset")
    dataset = RewardDataset(trajs, reward_name, sigma, alpha, target_reward)
    dataloader = cycle(_reward_batches(dataset, batch_size))

    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    reward_net = SimpleReward(obs_dim, act_dim, hidden_dim, hidden_layers)
    ex_s = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_a = jnp.zeros((1, act_dim), dtype=jnp.float32)
    params = reward_net.init(init_rng, ex_s, ex_a)['params']
    # optax: clip_by_global_norm(1.0) chained with AdamW(cosine_decay_schedule, weight_decay=1e-4).
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    train_state = TrainState.create(reward_net, params, tx=tx)

    @jax.jit
    def update(state, s, a, r):
        def loss_fn(p):
            pred = state(s, a, params=p)
            loss = jnp.mean(optax.huber_loss(pred, r, delta=1.0))
            return loss, {'loss': loss}
        return state.apply_loss_fn(loss_fn)

    if(check_specific_dataset(dataset_name)):
        SD = specific_dataset
    else:
        SD = None

    total_loss = 0
    step = 0
    for i in range(num_steps):
           s, a, r = next(dataloader)
           s = jnp.asarray(s)
           a = jnp.asarray(a)
           r = jnp.asarray(r)

           # Predicted Reward (grad + clip + step via TrainState).
           train_state, info = update(train_state, s, a, r)
           total_loss += float(info['loss'])
           step += 1

           if step % 2000 == 0:
              avg_loss = total_loss / 2000
              print(f"Step {step}, loss {avg_loss:.4f}")
              try:
                  from wandb_logger import wlog
                  wlog({'reward/loss': avg_loss, 'reward/lr': float(schedule(train_state.step - 1))}, step=step)
              except Exception:
                  pass
              total_loss = 0

           if step % save_freq == 0:
              checkpoint = copy.deepcopy(train_state.params)
              save_model(checkpoint, dataset_name, specific_dataset, task_id, step)

    save_to_finetuning(train_state.params, dataset_name, SD, task_id)
    stats = get_pretrained_reward_stats(reward_name)
    save_stats_to_finetuning(stats, dataset_name, SD, task_id)

def train_reward_pos_weight(
    dataset_name: str,
    hidden_layers: int = 5,
    hidden_dim: int = 512,
    batch_size: int = 256,
    num_steps: int = 30000,
    save_freq: int = 5000,
    lr: float = 5e-5,                    # lowered
    min_lr: float = 1e-6,
    sigma: Optional[float] = None,
    alpha: Optional[float] = None,
    target_reward: Optional[float] = 10.0,   # lowered
    specific_dataset: Optional[str] = None,
    goal: Optional[np.array] = None,
    task_id: Optional[int] = None,
    traj_length: Optional[int] = None,
    trajs: Optional[list] = None,
    pos_weight: float = 25.0,                # tuned down a bit
    device=None,
    *, rng=None,
):
    # API-CHANGE: trailing keyword-only `rng=` added (linen param init is stochastic; see CONVERSION_GUIDE §8).
    device = check_device() if device is None else device
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)

    if trajs is None:
        trajs, _, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id, goal, traj_length)
    else:
        train_trajs, _, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id, goal, traj_length)
        trajs = trajs + train_trajs

    print(f"Training reward approximator for {dataset_name}-{specific_dataset} | pos_weight={pos_weight}")

    dataset = RewardDataset(trajs, reward_name, sigma, alpha, target_reward)
    dataloader = cycle(_reward_batches(dataset, batch_size))

    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    reward_net = SimpleReward(obs_dim, act_dim, hidden_dim, hidden_layers)
    ex_s = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_a = jnp.zeros((1, act_dim), dtype=jnp.float32)
    params = reward_net.init(init_rng, ex_s, ex_a)['params']
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    train_state = TrainState.create(reward_net, params, tx=tx)

    @jax.jit
    def update(state, s, a, r):
        def loss_fn(p):
            pred = state(s, a, params=p)                 # (B,)
            # === Weighted Loss (Improved) ===
            weights = jnp.where(r > 0.01, pos_weight, 1.0)
            loss = (weights * optax.huber_loss(pred, r, delta=1.0)).mean()
            # Positive regularization
            pos_reg = jnp.mean(jax.nn.relu(pred) ** 2) * 0.02   # lowered coefficient
            total_loss_val = loss + pos_reg
            return total_loss_val, {'loss': loss}
        return state.apply_loss_fn(loss_fn)

    if check_specific_dataset(dataset_name):
        SD = specific_dataset
    else:
        SD = None

    # Save hyperparameters. `optimizer` info is a plain dict in the JAX port (no torch param_groups).
    save_reward_hyperparameters(
        dataset_name, batch_size, num_steps, lr, sigma, alpha,
        obs_dim, act_dim, reward_name, {'type': 'AdamW', 'weight_decay': 1e-4}, reward_net,
        filepath=None, specific_dataset=specific_dataset,
        target_reward=target_reward, goal=goal, task_id=task_id,
        pos_weight=pos_weight
    )

    total_loss = 0.0
    for step in range(1, num_steps + 1):
        s, a, r = next(dataloader)
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        r = jnp.asarray(r)

        train_state, info = update(train_state, s, a, r)

        total_loss += float(info['loss'])

        if step % 2000 == 0:                                 # more frequent log
            avg_loss = total_loss / 2000
            pos_ratio = float((r > 0.01).mean())
            current_lr = float(schedule(train_state.step - 1))
            print(f"Step {step:6d} | Loss: {avg_loss:.6f} | "
                  f"Pos Ratio: {pos_ratio:.4f} | LR: {current_lr:.2e}")
            total_loss = 0.0

        if step % save_freq == 0 or step == num_steps:
            checkpoint = copy.deepcopy(train_state.params)
            save_model(checkpoint, dataset_name, specific_dataset, task_id, step)

    save_to_finetuning(train_state.params, dataset_name, SD, task_id)
    stats = get_pretrained_reward_stats(reward_name)
    save_stats_to_finetuning(stats, dataset_name, SD, task_id)
    print("Reward model training finished!")
    return train_state.params

def _bootstrap_per_member(s, a, r, ensemble_size, *, rng):
    # API-CHANGE: torch passed a `device`; the JAX port replaces it with a keyword-only `rng` (keyed
    # bootstrap index sampling via jax.random.randint; see CONVERSION_GUIDE §8).
    B = s.shape[0]
    idx = jax.random.randint(rng, (ensemble_size, B), 0, B)
    return s[idx], a[idx], r[idx]


def train_reward_ensemble(
    dataset_name: str,
    hidden_layers: int,
    hidden_dim: int,
    batch_size: int,
    num_steps: int,
    save_freq: int,
    lr: float,
    min_lr: float,
    ensemble_size: int = 5,
    bootstrap: bool = True,
    save_percentage: float = 0.0,
    sigma: Optional[float] = None,
    alpha: Optional[float] = None,
    target_reward: Optional[float] = None,
    specific_dataset: Optional[str] = None,
    task_id: Optional[int] = None,
    goal: Optional[np.ndarray] = None,
    traj_length: Optional[int] = None,
    trajs: Optional[list] = None,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 2000,
    *, rng=None,
):
    # API-CHANGE: trailing keyword-only `rng=` added (linen param init + keyed bootstrap sampling).
    device = check_device()
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    # --- build dataset
    if trajs is None:
        trajs, _, obs_dim, act_dim = Train_Dataset(
            dataset_name, specific_dataset, task_id, goal, traj_length
        )
    else:
        extra_trajs, _, obs_dim, act_dim = Train_Dataset(
            dataset_name, specific_dataset, task_id, goal, traj_length
        )
        trajs = trajs + extra_trajs
    trajs = drop_trajs(trajs, save_percentage)
    print(f"[ensemble:{ensemble_size}] training reward for "
          f"{dataset_name}/{specific_dataset} task={task_id}  "
          f"(obs_dim={obs_dim}, act_dim={act_dim})")
    dataset = RewardDataset(trajs, reward_name, sigma, alpha, target_reward)
    dataloader = cycle(_reward_batches(dataset, batch_size))
    # --- build model + optim
    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    reward_net = EnsembleReward(
        obs_dim, act_dim, hidden_dim, hidden_layers,
        ensemble_size=ensemble_size,
    )
    ex_s = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_a = jnp.zeros((1, act_dim), dtype=jnp.float32)
    params = reward_net.init(init_rng, ex_s, ex_a)['params']
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    # optax: optional clip_by_global_norm(grad_clip) chained with AdamW(schedule, weight_decay).
    if grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(schedule, weight_decay=weight_decay))
    else:
        tx = optax.adamw(schedule, weight_decay=weight_decay)
    train_state = TrainState.create(reward_net, params, tx=tx)

    @jax.jit
    def update(state, s_e, a_e, r_e):
        def loss_fn(p):
            pred_e = state(s_e, a_e, params=p)            # (E, B)
            loss = jnp.mean(optax.huber_loss(pred_e, r_e, delta=1.0))
            return loss, {'loss': loss}
        return state.apply_loss_fn(loss_fn)

    SD = specific_dataset if check_specific_dataset(dataset_name) else None
    # --- persist hyperparams (works because save_reward_hyperparameters reads
    # whatever attrs the model exposes; EnsembleReward exposes hidden_dim,
    # hidden_layers, ensemble_size). `optimizer` info is a plain dict in the JAX port.
    save_reward_hyperparameters(
        dataset_name, batch_size, num_steps, lr, sigma, alpha,
        obs_dim, act_dim, reward_name, {'type': 'AdamW', 'weight_decay': weight_decay}, reward_net,
        filepath=None,
        specific_dataset=specific_dataset,
        target_reward=target_reward,
        goal=goal,
        task_id=task_id,
        ensemble_size=ensemble_size,
    )
    # --- train loop
    running_loss = 0.0
    for step in range(1, num_steps + 1):
        s, a, r = next(dataloader)
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        r = jnp.asarray(r)
        if bootstrap and ensemble_size > 1:
            rng, boot_rng = jax.random.split(rng)
            s_e, a_e, r_e = _bootstrap_per_member(s, a, r, ensemble_size, rng=boot_rng)
        else:
            # diversity from random init only
            s_e = jnp.broadcast_to(s[None], (ensemble_size,) + s.shape)
            a_e = jnp.broadcast_to(a[None], (ensemble_size,) + a.shape)
            r_e = jnp.broadcast_to(r[None], (ensemble_size,) + r.shape)
        train_state, info = update(train_state, s_e, a_e, r_e)
        # mean over (E*B) ≡ mean of per-member SmoothL1 losses
        running_loss += float(info['loss'])
        if step % log_every == 0:
            # quick diagnostic: ensemble disagreement on the current batch (no grads; call w/o params=).
            shared_pred = train_state(s, a)            # (E, B)
            disagree = float(shared_pred.std(axis=0).mean())
            avg_loss = running_loss / log_every
            lr_now = float(schedule(train_state.step - 1))
            print(f"step {step:>7d} | loss {avg_loss:.4f} | "
                  f"ens_std {disagree:.4f} | lr {lr_now:.2e}")
            running_loss = 0.0
        if step % save_freq == 0:
            checkpoint = copy.deepcopy(train_state.params)
            save_model(checkpoint, dataset_name, specific_dataset, task_id, step)
    # final artifacts for finetuning
    save_to_finetuning(train_state.params, dataset_name, SD, task_id)
    stats = get_pretrained_reward_stats(reward_name)
    save_stats_to_finetuning(stats, dataset_name, SD, task_id)
    print("ensemble reward training finished.")
    return train_state.params

class test_dataset:
    def __init__(self, trajs, Reward_name, sigma: Optional[float] = None, alpha: Optional[float] = None, target_reward: Optional[float] = None, goal: Optional[np.array] = None):
        self.stats = get_pretrained_reward_stats(Reward_name)
        transitions = []
        allowed_values = [0,1]
        for traj in trajs:
            obs = np.asarray(traj['observations'])        
            acts = np.asarray(traj['actions'])
            rews = np.asarray(traj['rewards'])
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            if(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            elif(alpha is not None):
                rews = ema_smooth(rews, alpha)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
        # Stacked numpy arrays for fql-style batched evaluation (host-side numpy; §13).
        self._obs = np.asarray([t[0] for t in transitions], dtype=np.float32)
        self._act = np.asarray([t[1] for t in transitions], dtype=np.float32)
        self._rew = np.asarray([t[2] for t in transitions], dtype=np.float32)

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
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(r, dtype=np.float32),
        )

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self._obs), size=batch_size)
        return self._obs[idx], self._act[idx], self._rew[idx]

def test_Model_ensemble(
    dataset_name,
    hidden_layers: int,
    hidden_dim: int,
    ensemble_size: int = 5,
    specific_dataset: Optional[str] = None,
    trajs: Optional[list] = None,
    sigma: Optional[float] = None,
    alpha: Optional[float] = None,
    target_reward: Optional[float] = None,
    goal: Optional[np.array] = None,
    task_id: Optional[int] = None,
    traj_length: Optional[int] = 100,
    save_freq: int = 50,
    num_steps: int = 500,
    batch_size: int = 256,
    report_per_member: bool = True,
    *, rng=None,
):
    """Evaluate ensemble reward checkpoints saved under
    ./Pretrain/Rewards/<reward_name>/Models/<reward_name>_<step>.pkl

    For each checkpoint in [save_freq, 2*save_freq, ..., num_steps]:
      - Loads it into an EnsembleReward(ensemble_size=...).
      - Computes SmoothL1 loss of the ensemble mean vs. target.
      - Reports per-member loss + ensemble disagreement (std across members)
        as mean / max / 95th percentile.
    """
    # API-CHANGE: trailing keyword-only `rng=` added (linen param init template is stochastic).
    device = check_device()
    print(f"Using device {device}")
    print(f"Testing reward ensemble (size={ensemble_size}) for "
          f"{dataset_name}-{specific_dataset} Dataset")
    print(f"Target reward: {target_reward}, Sigma: {sigma}, Alpha: {alpha}")

    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)

    # ----- build eval dataset (same logic as test_Model) -----
    if trajs is None:
        train_trajs, _, obs_dim, act_dim = Train_Dataset(
            dataset_name, specific_dataset, task_id, goal, traj_length,
        )
        dataset = RewardDataset(train_trajs, reward_name, sigma, alpha, target_reward)
    else:
        train_trajs, _, obs_dim, act_dim = Train_Dataset(
            dataset_name, specific_dataset, task_id, goal, traj_length,
        )
        trajs = trajs + train_trajs
        dataset = test_dataset(trajs, reward_name, sigma, alpha, target_reward, goal)

    print(f"Testing on {len(dataset)} samples")

    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    reward_net = EnsembleReward(
        obs_dim, act_dim, hidden_dim, hidden_layers,
        ensemble_size=ensemble_size,
    )
    ex_s = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_a = jnp.zeros((1, act_dim), dtype=jnp.float32)
    template_params = reward_net.init(init_rng, ex_s, ex_a)['params']

    num = save_freq
    while num <= num_steps:
        state_dict = load_model(dataset_name, specific_dataset, task_id, num)

        # TODO(checkpoint-bridge): legacy torch state_dicts need key remapping; here we restore a flax
        # serialization state_dict produced by this port into the param template.
        params = flax.serialization.from_state_dict(template_params, state_dict)
        reward_state = TrainState.create(reward_net, params)

        total_mean_loss = 0.0
        total_reward_mean = 0.0
        total_member_loss = np.zeros(ensemble_size)
        all_means = []
        all_stds = []

        n_batches = 0
        for s, a, r in _eval_batches(dataset, batch_size):
            s = jnp.asarray(s)
            a = jnp.asarray(a)
            r = jnp.asarray(r)

            # (E, B) — single forward gets every member's prediction (no grads; call w/o params=).
            pred_e = reward_state(s, a)
            mean_pred = pred_e.mean(axis=0)          # (B,)
            std_pred = pred_e.std(axis=0)            # (B,)

            # Loss of the ensemble mean (what you'd use at inference)
            loss = jnp.mean(optax.huber_loss(mean_pred, r, delta=1.0))
            total_mean_loss += float(loss)
            total_reward_mean += float(mean_pred.mean())

            # Per-member losses
            if report_per_member:
                r_e = jnp.broadcast_to(r[None], pred_e.shape)
                per_member = optax.huber_loss(
                    pred_e, r_e, delta=1.0,
                ).mean(axis=1)                        # (E,)
                total_member_loss += np.asarray(per_member)

            all_means.append(np.asarray(mean_pred))
            all_stds.append(np.asarray(std_pred))
            n_batches += 1

        avg_mean_loss = total_mean_loss / n_batches
        avg_reward = total_reward_mean / n_batches
        means_np = np.concatenate(all_means)
        stds_np = np.concatenate(all_stds)

        print(f"\n--- checkpoint {num} ---")
        print(f"loss (ensemble mean):  {avg_mean_loss:.4f}")
        print(f"reward (mean of mean): {avg_reward:.4f}")
        print(f"mean reward:           {means_np.mean():.4f}")
        print(f"std  reward:           {means_np.std():.4f}")
        print(f"max  reward:           {means_np.max():.4f}")
        print(f"min  reward:           {means_np.min():.4f}")

        # Ensemble disagreement diagnostics
        print(f"ensemble std (mean):   {stds_np.mean():.4f}")
        print(f"ensemble std (max):    {stds_np.max():.4f}")
        print(f"ensemble std (p95):    {np.percentile(stds_np, 95):.4f}")

        if report_per_member:
            per_member_avg = (total_member_loss / n_batches)
            for i, l in enumerate(per_member_avg):
                print(f"  member {i}: loss {l:.4f}")
            print(f"  member-loss spread: "
                  f"{per_member_avg.max() - per_member_avg.min():.4f}")

        num += save_freq

def test_Model(dataset_name, hidden_layers: int, hidden_dim: int, specific_dataset: Optional[str] = None, trajs: Optional[list] = None, sigma: Optional[float] = None, alpha: Optional[float] = None, target_reward: Optional[float] = None, goal: Optional[np.array] = None, task_id: Optional[int] = None, traj_length: Optional[int] = 100, save_freq: int = 50, num_steps: int = 500, *, rng=None):
    # API-CHANGE: trailing keyword-only `rng=` added (linen param init template is stochastic).
    device = check_device()
    print(f"Using device {device}")
    print(f"Testing the reward model for {dataset_name}-{specific_dataset} Dataset")
    print(f"Target reward: {target_reward}, Sigma: {sigma}, Alpha: {alpha}")
    reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    if(trajs is None):
        train_trajs, _, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id, goal, traj_length)
        dataset = RewardDataset(train_trajs, reward_name, sigma, alpha, target_reward)
    else:
        train_trajs, _, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id, goal, traj_length)
        trajs = trajs + train_trajs
        dataset = test_dataset(trajs, reward_name, sigma, alpha, target_reward, goal)
    print(f"Testing the reward model on {len(dataset)} samples")
    a = factorint(len(dataset))
    batch_size = 256
    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    reward_net = SimpleReward(obs_dim, act_dim, hidden_dim, hidden_layers)
    ex_s = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_a = jnp.zeros((1, act_dim), dtype=jnp.float32)
    template_params = reward_net.init(init_rng, ex_s, ex_a)['params']
    num = save_freq
    while num <= num_steps:
         Rewards = []
         state_dict = load_model(dataset_name, specific_dataset, task_id, num)
         # TODO(checkpoint-bridge): restore the port's flax serialization state_dict into the template.
         params = flax.serialization.from_state_dict(template_params, state_dict)
         reward_state = TrainState.create(reward_net, params)
         total_mean_loss = 0.0
         total_reward = 0.0
         n_batches = 0
         for s, a, r in _eval_batches(dataset, batch_size):
             s = jnp.asarray(s)
             a = jnp.asarray(a)
             r = jnp.asarray(r)
             pred = reward_state(s, a)                  # no grads; call w/o params=
             loss = jnp.mean(optax.huber_loss(pred, r, delta=1.0))
             total_mean_loss += float(loss)
             total_reward += float(pred.mean())
             Rewards.extend(np.asarray(pred))
             n_batches += 1

         avg_mean_loss = total_mean_loss / n_batches
         avg_reward = total_reward / n_batches
         print(f"model {num}, Loss {avg_mean_loss:.4f}, Reward: {avg_reward:.4f}")

         Rewards = np.array(Rewards)
         mean_R = Rewards.mean()
         std_R = Rewards.std()
         max_R = Rewards.max()
         min_R = Rewards.min()
         print(f"mean reward: {mean_R:.4f}")
         print(f'std_reward: {std_R:.4f}')
         print(f"max_reward: {max_R:.4f}")
         print(f"min_reward: {min_R:.4f}")

         num += save_freq

def get_pretrained_reward(dataset_name, checkpoints, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
       _, _, obs_dim, act_dim  =  Train_Dataset(dataset_name, specific_dataset)
       reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
       reward_model_state_dict = load_model(reward_name, checkpoints)
       return reward_model_state_dict, obs_dim, act_dim, reward_name

def get_pretrained_reward_stats(reward_name):
    #stats_path = f'./Pretrain/Rewards/{Reward_name}/Stats/{Reward_name}_stats.pkl'
    stats_path = os.path.join(
        project_root,
        "Pretrain",
        "Rewards",
        reward_name,
        "Stats",
        f"{reward_name}_stats.pkl",
    )
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats

