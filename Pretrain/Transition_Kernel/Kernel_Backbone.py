'''Train / test the Gaussian and Mixture-of-Gaussians transition kernels (JAX/Flax port).'''
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Online-Diffusion-Planning/
PRETRAIN_DIR = PROJECT_ROOT / 'Pretrain'
FINETUNE_DIR = PROJECT_ROOT / 'Finetuning'
from scipy.stats import median_abs_deviation

import jax
import jax.numpy as jnp
import flax
import numpy as np
import optax

from Pretrain.Dataset import CubeDataset_Singletask, KitchenDataset, OGPointmazeDataset, OGPointmazeDataset_Singletask, PointMazeDataset, CubeDataset
from .Kernel_Net import RobustTransitionKernel, MoGTransitionKernel
from sympy import factorint
import pickle
import os
from typing import Optional, List
import math
import copy

# Shared port plumbing (mirrors fql).
from JAX_PORT.jax_utils import TrainState

try:
    from Pretrain.utils import SAStats, cycle, check_device
except ModuleNotFoundError:
    from utils import SAStats, cycle, check_device
import json

def check_specific_dataset(dataset_name):
    if(dataset_name == 'kitchen'):
         return False
    elif dataset_name in ['pointmaze', 'cube', 'ogpointmaze']:
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

     elif(env_name == 'ogpointmaze'):
          if specific_env == 'medium':
               return 'OG2DMaze_Medium'
          elif specific_env == 'large':
               return 'OG2DMaze_Large'
          elif specific_env == 'giant':
               return 'OG2DMaze_Giant'
          else:
               raise ValueError(f"Invalid cube dataset name: {specific_env}")
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

    # Get optimizer info. `optimizer` is now an optax GradientTransformation (no param_groups);
    # the weight decay is the constant folded into the chain at construction time (1e-5).
    optimizer_type = type(optimizer).__name__
    optimizer_params = {
        'type': optimizer_type,
        'lr': lr,
        'weight_decay': 1e-5,
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
def save_model(kernel_net, kernel_name, num_steps, ensemble_idx):
    # `kernel_net` is now the flax params pytree for one ensemble member (see §10).
    net_dict = flax.serialization.to_state_dict(kernel_net)
    models_dir = PRETRAIN_DIR / 'Transition_Kernel' / kernel_name / 'Models' / str(num_steps)
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f'{kernel_name}_{num_steps}_{ensemble_idx}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(net_dict, f)
    print(f'Kernel model save to {kernel_name}_{num_steps}_{ensemble_idx}.pkl')

def save_to_finetuning(kernel_net, dataset_name, ensemble_idx, specific_dataset: Optional[str] = None):
    # `kernel_net` is now the flax params pytree for one ensemble member (see §10).
    net_dict = flax.serialization.to_state_dict(kernel_net)
    name = getName(dataset_name, specific_dataset)
    if specific_dataset is None:
         ft_models_dir = FINETUNE_DIR / 'Kernels' / dataset_name / 'Models' / '0'
    else:
         ft_models_dir = FINETUNE_DIR / 'Kernels' / dataset_name / specific_dataset / 'Models' / '0'
    ft_models_dir.mkdir(parents=True, exist_ok=True)
    save_path = ft_models_dir / f'{name}_Kernel_{ensemble_idx}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(net_dict, f)
    print(f'kernel model save to {save_path}')

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

def load_model(kernel_name, num_steps, ensemble_idx):
    load_path = (
        PRETRAIN_DIR
        / 'Transition_Kernel'
        / kernel_name
        / 'Models'
        / str(num_steps)
        / f'{kernel_name}_{num_steps}_{ensemble_idx}.pkl'
    )
    with open(load_path, 'rb') as f:
        state_dict = pickle.load(f)
    return state_dict

def Train_Dataset(dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
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
     
    elif(dataset_name == 'cube'):
        
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

    elif(dataset_name == 'ogpointmaze'):
        
        if(specific_dataset is None): 
             raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif(specific_dataset == 'medium'):
             data_1 = OGPointmazeDataset('medium')
             if(task_id is not None):
                 data_2 = OGPointmazeDataset_Singletask('medium', task_id, mode = 'reward')
             name = 'OG2DMaze_Kernel_medium'
        elif(specific_dataset == 'large'):
             data_1 =  OGPointmazeDataset('large')
             if(task_id is not None):
                 data_2 = OGPointmazeDataset_Singletask('large', task_id, mode = 'reward')
             name = 'OG2DMaze_Kernel_large'
        elif(specific_dataset == 'giant'):
             data_1 = OGPointmazeDataset('giant')
             if(task_id is not None):
                 data_2 = OGPointmazeDataset_Singletask('giant', task_id, mode = 'reward')
             name = 'Cube_Kernel_giant'
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
class KernelDataset:
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
         # Stacked numpy arrays for fql-style batched sampling (host-side; np RNG for shuffling, §13).
         self.s = np.stack([d[0] for d in data], axis=0).astype(np.float32)
         self.a = np.stack([d[1] for d in data], axis=0).astype(np.float32)
         self.s_next = np.stack([d[2] for d in data], axis=0).astype(np.float32)
         self.save_stats(kernel_name)
    def save_stats(self, kernel_name):
       stats_name = f'{kernel_name}_stats.pkl'
       stats_dir = PRETRAIN_DIR / 'Transition_Kernel' / kernel_name / 'Stats'
       stats_dir.mkdir(parents=True, exist_ok=True)
       savepath = stats_dir / stats_name
       with open(savepath, 'wb') as f:
            pickle.dump(self.stats, f)
       print(f'saved stats to {savepath}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, s_next = self.data[idx]
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(s_next, dtype=np.float32),
        )

    def sample(self, batch_size):
        '''fql-style batched sampler: returns (s, a, s_next) numpy arrays (see §13).'''
        idxs = np.random.randint(0, len(self.s), size=batch_size)
        return self.s[idxs], self.a[idxs], self.s_next[idxs]

class test_dataset:
    def __init__(self, trajs, kernel_name):
        """
        stats_path = f'./Pretrain/Transition_Kernel/{kernel_name}/Stats/{kernel_name}_stats.pkl'
        with open(stats_path, 'rb') as f:
              self.stats = pickle.load(f)
        """
        stats_path = PRETRAIN_DIR / 'Transition_Kernel' / kernel_name / 'Stats' / f'{kernel_name}_stats.pkl'
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
        # Stacked numpy arrays for fql-style batched sampling (§13).
        self.s = np.stack([t[0] for t in transitions], axis=0).astype(np.float32)
        self.a = np.stack([t[1] for t in transitions], axis=0).astype(np.float32)
        self.s_next = np.stack([t[2] for t in transitions], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, s_next = self.transitions[idx]
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(s_next, dtype=np.float32),
        )

    def sample(self, batch_size):
        '''fql-style batched sampler: returns (s, a, s_next) numpy arrays (see §13).'''
        idxs = np.random.randint(0, len(self.s), size=batch_size)
        return self.s[idxs], self.a[idxs], self.s_next[idxs]


import copy
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
    device=None,
    *, rng=None,  # API-CHANGE/HIGH#3: threaded PRNG key (was implicitly stochastic via init/shuffle).
):
    device = check_device()

    print(f'Training MoG Transition Kernel for {dataset_name}')
    if specific_dataset:
        print(f'  Specific dataset: {specific_dataset}')

    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Prepare dataset
    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id)
    if(trajs is not None):
          total_trajs = train_trajs + trajs
    else:
          total_trajs = train_trajs
    dataset = KernelDataset(total_trajs, kernel_name)

    # Create ensemble of MoG kernels. MEDIUM FIX: construct with KEYWORDS so the 6th positional does
    # NOT bind to min_log_std; pass noise_floor by keyword (Kernel_Net.py field order is unchanged).
    model_defs = [
        MoGTransitionKernel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_modes=num_modes,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            noise_floor=noise_floor,
        )
        for _ in range(ensemble_size)
    ]

    # optax: grad-clip(5.0) chained before adamw(weight_decay=1e-5) (§5).
    def make_tx():
        return optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(lr, weight_decay=1e-5))

    example_s = jnp.asarray(dataset.s[:1])
    example_a = jnp.asarray(dataset.a[:1])
    train_states = []
    for model_def in model_defs:
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, example_s, example_a)['params']
        train_states.append(TrainState.create(model_def, params, tx=make_tx()))

    # Save hyperparameters (you may need to adjust this function for MoG)
    save_kernel_hyperparameters(
        dataset_name,
        batch_size,
        num_steps,
        lr,
        obs_dim,
        act_dim,
        kernel_name,
        train_states[0].tx,  # representative optax tx
        model_defs[0],       # representative model_def (linen module exposes the hyperparameter attrs)
        ensemble_size,
        λ_reg,
        specific_dataset=specific_dataset
    )

    SD = specific_dataset if check_specific_dataset(dataset_name) else None

    @jax.jit
    def mog_update(train_state, s, a, s_next):
        model_def = train_state.model_def

        def loss_fn(params):
            mu, log_std, weights = train_state(s, a, params=params)
            loss = model_def.apply({'params': params}, s_next, mu, log_std, weights, method=model_def.mog_nll)

            # === Optional: disagreement regularization ===
            # Average over modes for disagreement calculation
            mu_mean = mu.mean(axis=1)                    # (B, obs_dim)
            disagreement = ((mu - jnp.expand_dims(mu_mean, 1)) ** 2).mean(axis=1).mean(axis=0)

            var = jnp.exp(2 * log_std) + model_def.noise_floor
            penalty = (disagreement / (var.mean(axis=1) + 1e-6)).mean()

            loss = loss + λ_reg * penalty
            return loss, loss

        grads, loss = jax.grad(loss_fn, has_aux=True)(train_state.params)
        return train_state.apply_gradients(grads=grads), loss

    step = 0
    total_loss = 0.0

    for step in tqdm(range(1, num_steps + 1), desc='Training MoG Kernel'):
        s, a, s_next = dataset.sample(batch_size)
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        s_next = jnp.asarray(s_next)

        losses = []
        for i in range(ensemble_size):
            train_states[i], loss = mog_update(train_states[i], s, a, s_next)
            losses.append(loss)

        # Logging
        avg_loss = sum(float(loss) for loss in losses) / ensemble_size
        total_loss += avg_loss

        if step % 100 == 0:
            print(f'Step {step:6d} | Avg Loss: {total_loss/100:.6f}')
            try:
                from wandb_logger import wlog
                wlog({'kernel/avg_loss': total_loss / 100, 'kernel/step_loss': avg_loss}, step=step)
            except Exception:
                pass
            total_loss = 0.0

        # Save checkpoints
        if step % save_freq == 0 or step == num_steps:
            for idx, ts in enumerate(train_states):
                save_model(ts.params, f'{kernel_name}', step, idx)

            if step == num_steps:
                for idx, ts in enumerate(train_states):
                    save_to_finetuning(ts.params, dataset_name, idx, SD)

                stats = get_pretrained_kernel_stats(f'{kernel_name}')
                save_stats_to_finetuning(stats, dataset_name, SD)

    print('MoG Transition Kernel training completed!')
    # Return final ensemble as (model_def, train_state) pairs (§11: python list of independent models).
    return [(model_defs[i], train_states[i]) for i in range(ensemble_size)]




def train_kernel(dataset_name, specific_dataset: str = None,
                 batch_size=256, lr=1e-3, num_steps=10000, save_freq = 200,
                 ensemble_size=10, hidden_layers = 2, hidden_dim = 256, λ_reg=1e-3, trajs: Optional[list] = None,
                 *, rng=None):  # API-CHANGE/HIGH#3: threaded PRNG key (was implicitly stochastic).
    # Prepare dataset / dataloader
    if specific_dataset is None:
        print(f'Training kernel for {dataset_name}')
    else:
        print(f'Training kernel for {dataset_name}_{specific_dataset}')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = check_device()
    print('Using device:', device)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    if(trajs is None):
           trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    dataset = KernelDataset(trajs, kernel_name)

    # Create ensemble of models
    model_defs = [RobustTransitionKernel(obs_dim, act_dim, hidden_layers, hidden_dim) for _ in range(ensemble_size)]

    # optax: adamw with weight_decay=1e-5 (§5).
    def make_tx():
        return optax.adamw(lr, weight_decay=1e-5)

    example_s = jnp.asarray(dataset.s[:1])
    example_a = jnp.asarray(dataset.a[:1])
    train_states = []
    for model_def in model_defs:
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, example_s, example_a)['params']
        train_states.append(TrainState.create(model_def, params, tx=make_tx()))

    # Save hyperparameters at the start of training
    save_kernel_hyperparameters(
        dataset_name,
        batch_size,
        num_steps,
        lr,
        obs_dim,
        act_dim,
        kernel_name,
        train_states[0].tx,  # Use first optimizer (optax tx) as representative
        model_defs[0],       # Use first model_def as representative (exposes hyperparameter attrs)
        ensemble_size,
        λ_reg,
        specific_dataset=specific_dataset
    )
    if(check_specific_dataset(dataset_name)):
        SD = specific_dataset
    else:
        SD = None

    @jax.jit
    def forward_member(train_state, s, a):
        # No params= -> stored params, no gradient (used to build the detached disagreement, §6).
        return train_state(s, a)

    @jax.jit
    def kernel_update(train_state, s, a, s_next, disagreement_detached):
        model_def = train_state.model_def

        def loss_fn(params):
            mu, log_std = train_state(s, a, params=params)
            loss = model_def.apply({'params': params}, s_next, mu, log_std, method=model_def.gaussian_nll)
            # penalize if log_std is too small relative to disagreement
            penalty = (disagreement_detached / (jnp.exp(2 * log_std) + model_def.noise_floor)).sum(axis=-1).mean()
            loss = loss + λ_reg * penalty
            return loss, loss

        grads, loss = jax.grad(loss_fn, has_aux=True)(train_state.params)
        return train_state.apply_gradients(grads=grads), loss

    step = 0
    total_loss = 0.0

    for step in range(1, num_steps + 1):
        s, a, s_next = dataset.sample(batch_size)
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        s_next = jnp.asarray(s_next)

        # For each model in ensemble, forward (no grad) to build the disagreement statistic.
        mus = []
        for i in range(ensemble_size):
            mu, log_std = forward_member(train_states[i], s, a)
            mus.append(mu)
        # optional: variance-disagreement inflation
        # compute mean of mus
        mus_stack = jnp.stack(mus, axis=0)  # (K, B, obs_dim)
        mu_mean = mus_stack.mean(axis=0)    # (B, obs_dim)
        # disagreement = average squared deviation
        disagreement = ((mus_stack - jnp.expand_dims(mu_mean, 0)) ** 2).mean(axis=0)
        disagreement_detached = jax.lax.stop_gradient(disagreement)

        # Backprop & optimize each model (gaussian_nll + λ_reg * penalty with detached disagreement).
        losses = []
        for i in range(ensemble_size):
            train_states[i], loss = kernel_update(train_states[i], s, a, s_next, disagreement_detached)
            losses.append(loss)

        avg_loss = float(sum(losses)) / ensemble_size
        total_loss += avg_loss

        if step % 500 == 0:
            print(f'Step {step}, avg_loss: {total_loss / 500:.6f}')
            total_loss = 0.0

        if step % save_freq == 0 or step == num_steps:
            # Save all ensemble members
            for idx, ts in enumerate(train_states):
                save_model(ts.params, kernel_name, step, idx)
            if(step == num_steps):
                for idx, ts in enumerate(train_states):
                    save_to_finetuning(ts.params, dataset_name, idx, SD)


    stats = get_pretrained_kernel_stats(kernel_name)
    save_stats_to_finetuning(stats, dataset_name, SD)
    # Return final ensemble as (model_def, train_state) pairs (§11: python list of independent models).
    return [(model_defs[i], train_states[i]) for i in range(ensemble_size)]


def test_kernel(dataset_name, specific_dataset: str = None,
                trajs: list = None,
                save_freq: int = 50, num_steps: int = 500, hidden_layers = 2, hidden_dim = 256, ensemble_size = 3, quantile = 0.999,
                *, rng=None):  # API-CHANGE/HIGH#3: threaded PRNG key (was implicitly stochastic).
    device = check_device()
    print('Using device:', device)
    if rng is None:
        rng = jax.random.PRNGKey(0)

    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset)
    if trajs is None:
        dataset = test_dataset(train_trajs, kernel_name)
    else:
        dataset = test_dataset(trajs, kernel_name)

    # Build a params template once for restoring flax state dicts (init values overwritten on restore).
    model_def = RobustTransitionKernel(obs_dim, act_dim, hidden_layers, hidden_dim)
    rng, init_rng = jax.random.split(rng)
    params_template = model_def.init(init_rng, jnp.asarray(dataset.s[:1]), jnp.asarray(dataset.a[:1]))['params']

    # For each saved checkpoint / ensemble member
    step = save_freq
    while step <= num_steps:
        # Load ensemble members as (model_def, params) tuples (§11).
        ensemble = []
        for idx in range(ensemble_size):
            state_dict = load_model(kernel_name, step, idx)
            params = flax.serialization.from_state_dict(params_template, state_dict)
            ensemble.append((model_def, params))

        # Compute log-probs over dataset (single full pass in fixed-size batches; order-invariant stats).
        all_D2_total = []
        all_log_density = []
        #all_D_total = []
        count = 0
        #worst = (None, float("inf"), None)  # (idx, log_prob, (s, a, s_next))
        n = len(dataset)
        for start in range(0, n, 256):
            s = jnp.asarray(dataset.s[start:start + 256])
            a = jnp.asarray(dataset.a[start:start + 256])
            s_next = jnp.asarray(dataset.s_next[start:start + 256])

            #compute total mahalanobis distance (no grad: pure inference, §6)
            D2_total = compute_total_mahalanobis_score(ensemble, s, a, s_next)
            log_density = compute_log_density(ensemble, s, a, s_next)
            D2 = np.asarray(D2_total)
            log_density = np.asarray(log_density)
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
        print(f'Checkpoint {step}')
        print(f'mean_D2_total = {mean_D2_total:.4f}')
        print(f'min_D2_total = {min_D2_total:.4f}')
        print(f'max_D2_total = {max_D2_total:.4f}')
        print(f'std_D2_total = {std_D2_total:.4f}')
        print(f'τ ({quantile*100:.0f}th percentile) : {tau:.4f}')

        print('Log Density')
        all_log_density = np.array(all_log_density)
        mean_log_density = float(all_log_density.mean())
        min_log_density = float(all_log_density.min())
        max_log_density = float(all_log_density.max())
        std_log_density = float(all_log_density.std())
        tau = float(np.quantile(all_log_density, 1 - quantile))
        print(f'Checkpoint {step}')
        print(f'mean_log_density = {mean_log_density:.4f}')
        print(f'min_log_density = {min_log_density:.4f}')
        print(f'max_log_density = {max_log_density:.4f}')
        print(f'std_log_density = {std_log_density:.4f}')
        print(f'τ ({(1-quantile)*100:.0f}th percentile) : {tau:.4f}')
        step += save_freq


def test_kernel_mog(dataset_name, specific_dataset: str = None, task_id: Optional[int] = None,
                trajs: list = None,
                save_freq: int = 50, num_steps: int = 500, num_hidden_layers = 2, hidden_dim = 256, ensemble_size = 3, num_modes = 9, noise_floor = 1e-6, quantile = 0.95,
                *, rng=None):  # API-CHANGE/HIGH#3: threaded PRNG key (was implicitly stochastic).
    device = check_device()
    print('Using device:', device)
    if rng is None:
        rng = jax.random.PRNGKey(0)

    train_trajs, kernel_name, obs_dim, act_dim = Train_Dataset(dataset_name, specific_dataset, task_id)
    if trajs is not None:
        total_trajs = train_trajs + trajs
    else:
        total_trajs = train_trajs
    dataset = test_dataset(total_trajs, kernel_name)

    # MEDIUM FIX: construct with KEYWORD noise_floor so the 6th positional does NOT bind to min_log_std.
    model_def = MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, noise_floor=noise_floor)
    rng, init_rng = jax.random.split(rng)
    params_template = model_def.init(init_rng, jnp.asarray(dataset.s[:1]), jnp.asarray(dataset.a[:1]))['params']

    # For each saved checkpoint / ensemble member
    step = save_freq
    while step <= num_steps:
        # Load ensemble members as (model_def, params) tuples (§11).
        ensemble = []
        for idx in range(ensemble_size):
            state_dict = load_model(kernel_name, step, idx)
            params = flax.serialization.from_state_dict(params_template, state_dict)
            ensemble.append((model_def, params))

        # Compute log-probs over dataset (single full pass in fixed-size batches; order-invariant stats).
        all_D2_total = []
        all_log_density = []
        #all_D_total = []
        count = 0
        #worst = (None, float("inf"), None)  # (idx, log_prob, (s, a, s_next))
        n = len(dataset)
        for start in range(0, n, 256):
            s = jnp.asarray(dataset.s[start:start + 256])
            a = jnp.asarray(dataset.a[start:start + 256])
            s_next = jnp.asarray(dataset.s_next[start:start + 256])

            #compute total mahalanobis distance (no grad: pure inference, §6)
            D2_total = compute_total_mahalanobis_score_mog(ensemble, s, a, s_next)
            log_density = compute_log_density_mog(ensemble, s, a, s_next)
            D2 = np.asarray(D2_total)
            log_density = np.asarray(log_density)
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
        print(f'Checkpoint {step}')
        print(f'mean_D2_total = {mean_D2_total:.4f}')
        print(f'min_D2_total = {min_D2_total:.4f}')
        print(f'max_D2_total = {max_D2_total:.4f}')
        print(f'std_D2_total = {std_D2_total:.4f}')
        print(f'τ ({quantile*100:.0f}th percentile) : {tau:.4f}')

        print('Log Density')
        all_log_density = np.array(all_log_density)
        mean_log_density = float(all_log_density.mean())
        min_log_density = float(all_log_density.min())
        max_log_density = float(all_log_density.max())
        std_log_density = float(all_log_density.std())
        tau = float(np.quantile(all_log_density, 1 - quantile))
        print(f'Checkpoint {step}')
        print(f'mean_log_density = {mean_log_density:.4f}')
        print(f'min_log_density = {min_log_density:.4f}')
        print(f'max_log_density = {max_log_density:.4f}')
        print(f'std_log_density = {std_log_density:.4f}')
        print(f'τ ({(1-quantile)*100:.0f}th percentile) : {tau:.4f}')
        step += save_freq



def get_pretrained_kernel(dataset_name, checkpoints, specific_dataset: Optional[str] = None):
       _, name, obs_dim, act_dim  =  Train_Dataset(dataset_name, specific_dataset)
       path = PRETRAIN_DIR / 'Transition_Kernel' / name / 'Models' / str(checkpoints)
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
    stats_path = PRETRAIN_DIR / 'Transition_Kernel' / kernel_name / 'Stats' / f'{kernel_name}_stats.pkl'
    with open(stats_path, 'rb') as f:
          stats = pickle.load(f)
    return stats


def compute_total_mahalanobis_score(kernels: List[RobustTransitionKernel], s, a, s_next):
    # §11: `kernels` is a python list of (model_def, params) for independently-loaded kernels.
    mus = []
    log_stds = []
    for model_def, params in kernels:
            mu, log_std = model_def.apply({'params': params}, s, a)  # (B, obs_dim)
            mus.append(mu)
            log_stds.append(log_std)
    # Stack -> (K, B, obs_dim)
    mus = jnp.stack(mus, axis=0)
    log_stds = jnp.stack(log_stds, axis=0)
    # 1. Total mean
    mu_total = mus.mean(axis=0)                    # (B, obs_dim)
    # 2. Aleatoric variance (average predicted variance)
    var_aleatoric = (jnp.exp(2 * log_stds) + kernels[0][0].noise_floor).mean(axis=0)
    # 3. Epistemic variance (disagreement of means)
    var_epistemic = mus.var(axis=0)   # population variance (common in MBRL)
    # 4. Total variance
    var_total = var_aleatoric + var_epistemic
    var_total = jnp.clip(var_total, a_min=1e-8)
    # 5. Squared Mahalanobis Distance (Total Score)
    residual = s_next - mu_total
    #residual = jnp.clip(residual, -10.0, 10.0)   # stability
    D2_total = ((residual ** 2) / var_total).sum(axis=-1)   # (B,)
    return D2_total

def compute_log_density(kernels: List[RobustTransitionKernel], s, a, s_next):
    # §11: `kernels` is a python list of (model_def, params).
    log_probs = []
    for model_def, params in kernels:
        mu, log_std = model_def.apply({'params': params}, s, a)
        lp = model_def.apply({'params': params}, s_next, mu, log_std, method=model_def.log_prob)
        log_probs.append(lp)
    #log_probs = jnp.stack(log_probs, axis=0).mean(axis=0)
    log_probs = jnp.stack(log_probs, axis=0)
    log_density = jax.scipy.special.logsumexp(log_probs, axis=0) - math.log(len(kernels))
    return log_density
    #return log_probs

def compute_log_density_mog(kernels: List[MoGTransitionKernel], s, a, s_next):
    """Returns total log p(s'|s,a) under ensemble of MoGs"""
    # §11: `kernels` is a python list of (model_def, params).
    all_log_probs = []

    for model_def, params in kernels:
        mu, log_std, weights = model_def.apply({'params': params}, s, a)
        lp = model_def.apply({'params': params}, s_next, mu, log_std, weights, method=model_def.log_prob)
        all_log_probs.append(lp)

    all_log_probs = jnp.stack(all_log_probs, axis=0)            # (K_ens, B)

    # Proper ensemble logsumexp
    log_density = jax.scipy.special.logsumexp(all_log_probs, axis=0) - math.log(len(kernels))

    return log_density


def compute_total_mahalanobis_score_mog(
    kernels: list,
    s,
    a,
    s_next,
):
    """
    MoG-compatible Total Mahalanobis Distance.
    """
    # §11: `kernels` is a python list of (model_def, params).
    K_ens = len(kernels)                    # number of ensemble members
    B = s.shape[0]

    mu_list = []
    var_list = []

    for model_def, params in kernels:
        mu, log_std, weights = model_def.apply({'params': params}, s, a)  # mu: (B, K_modes, obs_dim)
                                                      # weights: (B, K_modes)

        K_modes = weights.shape[1]

        # === Mixture statistics for this model ===
        # Weighted mean
        mu_mix = jnp.sum(jnp.expand_dims(weights, -1) * mu, axis=1)          # (B, obs_dim)

        # Aleatoric variance: E[Var]
        var_ale = jnp.exp(2 * log_std) + model_def.noise_floor          # (B, K_modes, obs_dim)
        var_ale_mix = jnp.sum(jnp.expand_dims(weights, -1) * var_ale, axis=1)  # (B, obs_dim)

        # Epistemic variance: Var[E]
        mu_centered = mu - jnp.expand_dims(mu_mix, 1)                         # (B, K_modes, obs_dim)
        var_epi_mix = jnp.sum(jnp.expand_dims(weights, -1) * (mu_centered ** 2), axis=1)

        var_mix = var_ale_mix + var_epi_mix
        var_mix = jnp.clip(var_mix, a_min=1e-6)

        mu_list.append(mu_mix)
        var_list.append(var_mix)

    # === Ensemble level ===
    mu_ensemble = jnp.stack(mu_list, axis=0)           # (K_ens, B, obs_dim)
    var_ensemble = jnp.stack(var_list, axis=0)         # (K_ens, B, obs_dim)

    mu_total = mu_ensemble.mean(axis=0)                  # (B, obs_dim)

    var_aleatoric = var_ensemble.mean(axis=0)
    var_epistemic = mu_ensemble.var(axis=0)

    var_total = var_aleatoric + var_epistemic
    var_total = jnp.clip(var_total, a_min=1e-6)

    # === Mahalanobis ===
    residual = s_next - mu_total
    residual = jnp.clip(residual, -10.0, 10.0)

    D2_total = ((residual ** 2) / var_total).sum(axis=-1)   # (B,)

    return D2_total

