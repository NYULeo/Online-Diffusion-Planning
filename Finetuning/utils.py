'''Finetuning hub for ODP: datasets, reward/kernel/critic trainers, planner rollouts, schedulers.

JAX/Flax (FQL-style) port of the original torch module. Training loops use optax + jax_utils.TrainState
(grads via apply_loss_fn / jax.grad), EMA/target nets use jax_utils.target_update, datasets store numpy
arrays and expose fql-style sample(), and the diffusion rollouts thread an explicit rng. Functions that
ingest pre-trained torch checkpoints (get_planner / get_reward_model / get_kernel / get_critic_model and
the save_* twins) keep their signatures and carry a `# TODO(checkpoint-bridge)` describing the torch
state_dict -> flax param-tree remap (Dense weight (out,in) -> kernel (in,out).T).
'''
import sys
import os

#from Finetuning.heatmap_plot import critic_heatmap
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import numpy as np
import os
import pickle
from Pretrain.utils import SAStats
from scipy.ndimage import gaussian_filter1d
from typing import TypedDict, List
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from Pretrain.Dataset import get_PlannerName
from typing import Tuple, Dict
from Pretrain.Transition_Kernel.Kernel_Backbone import count_files_in_folder
import copy
from Pretrain.Rewards.nets import SimpleReward, EnsembleReward
from Pretrain.Transition_Kernel.Kernel_Net import MoGTransitionKernel, RobustTransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_total_mahalanobis_score, compute_log_density_mog, compute_log_density, compute_total_mahalanobis_score_mog
from Pretrain.Dataset import KitchenDataset, PointMazeDataset, get_env, get_dataset, Planner_Processor
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Critic.nets import Critic
from Pretrain.Dataset import get_dataset
import json
import random

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

from flax_utils import TrainState, target_update



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

def _bootstrap_per_member(s, a, r, ensemble_size, device):
    # Host-side data bootstrap (numpy RNG for data shuffling per CONVERSION_GUIDE.md §13).
    B = s.shape[0]
    idx = np.random.randint(0, B, size=(ensemble_size, B))
    return s[idx], a[idx], r[idx]

def check_specific_dataset(dataset_name):
    if(dataset_name == 'kitchen'):
         return False
    elif dataset_name in ['pointmaze', 'cube', 'ogpointmaze']:
        return True

def reward_name_converter(specific_dataset):
    if(specific_dataset == 'single-play' or specific_dataset == 'single-noise'):
        return 'single'
    elif(specific_dataset == 'double-play' or specific_dataset == 'double-noise'):
        return 'double'
    elif(specific_dataset == 'triple-play' or specific_dataset == 'triple-noise'):
        return 'triple'
    elif(specific_dataset == 'quadruple-play' or specific_dataset == 'quadruple-noise'):
        return 'quadruple'
    else:
        return specific_dataset

def reward_processor(rewards, name: str):
    def spare_reward_processor(rewards):
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

    def ogbench_reward_processor(rewards):
         if(not isinstance(rewards, np.ndarray)):
              rewards = np.array(rewards)
         Min = np.min(rewards)
         dist = 0 - Min
         rews = rewards + dist
         return rews
    
    if(name in ('cube', 'ogpointmaze', 'antmaze', 'humanoidmaze', 'puzzle', 'scene')):
         return ogbench_reward_processor(rewards)
    else:
         return spare_reward_processor(rewards)

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
              if((i - last_step) < 3):
                  last_step = i+1
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
    return new_trajs
   
def save_reward_model(reward_net, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    # TODO(checkpoint-bridge): `reward_net` is now a (model_def, params) / TrainState pair (was a torch
    # nn.Module). We persist `flax.serialization.to_state_dict(params)` via pickle, keeping the exact path
    # layout so the rest of the pipeline finds the checkpoint. The torch->flax Dense remap
    # (weight (out,in) -> kernel (in,out).T, bias->bias, LayerNorm weight->scale) is documented in get_reward_model.
    net_dict = flax.serialization.to_state_dict(reward_net)
    specific_dataset = reward_name_converter(specific_dataset)
    #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
    if(check_specific_dataset(dataset_name)):
          os.makedirs(f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
          save_path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{reward_name}_Reward_{str(step)}.pkl'
    else:
          os.makedirs(f'./Finetuning/Rewards/{dataset_name}/Models/', exist_ok=True)
          save_path = f'./Finetuning/Rewards/{dataset_name}/Models/{reward_name}_Reward_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    with open(save_path, 'wb') as f:
          pickle.dump(net_dict, f)

def save_kernel_model(kernel_net, dataset_name, specific_dataset, step, ensemble_idx):
    # TODO(checkpoint-bridge): `kernel_net` is now (model_def, params)/params (was a torch nn.Module). Persist
    # via flax.serialization + pickle, keeping the path layout; torch->flax Dense remap documented in get_kernel.
    specific_dataset = reward_name_converter(specific_dataset)
    name = getName2(dataset_name, specific_dataset)
    net_dict = flax.serialization.to_state_dict(kernel_net)
    if(check_specific_dataset(dataset_name)):
          os.makedirs(f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}', exist_ok=True)
          save_path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    else:
          os.makedirs(f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}', exist_ok=True)
          save_path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}/{name}_Kernel_{str(ensemble_idx)}.pkl'
    with open(save_path, 'wb') as f:
          pickle.dump(net_dict, f)
    #print(f"Kernel model save to {name}_{str(step)}_{str(ensemble_idx)}.pkl")

def get_reward_model(dataset_name, specific_dataset, step, task_id: Optional[int] = None):
    # TODO(checkpoint-bridge): legacy checkpoints are torch state_dicts saved via torch.save. To ingest a
    # torch .pkl into the flax SimpleReward param tree, remap each Dense: torch Linear `weight` (out,in) ->
    # flax `kernel` = weight.T (in,out); `bias` -> `bias`; LayerNorm `weight` -> `scale`, `bias` -> `bias`.
    # New checkpoints are saved as flax.serialization state dicts (pickle); load them directly here.
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    specific_dataset = reward_name_converter(specific_dataset)
    #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Models/{reward_name}_Reward_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rewards/{dataset_name}/Models/{reward_name}_Reward_{str(step)}.pkl'
    with open(path, 'rb') as f:
        model_state_dict = pickle.load(f)
    return model_state_dict, obs_dim, act_dim

def get_reward_stats(dataset_name, specific_dataset, step, task_id: Optional[int] = None):
    specific_dataset = reward_name_converter(specific_dataset)
    #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
    reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/{reward_name}_Reward_stats_{str(step)}.pkl'
        
    else:
        path = f'./Finetuning/Rewards/{dataset_name}/Stats/{reward_name}_Reward_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats  

def get_kernel(dataset_name, specific_dataset, step):
    # TODO(checkpoint-bridge): returns a python list of kernel state dicts (independently-loaded models,
    # NOT a vmapped ensemble — keep as a list). Legacy torch state_dicts need the per-Dense remap
    # (weight (out,in) -> kernel (in,out).T, bias->bias, LayerNorm weight->scale); new ones are flax pickles.
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    specific_dataset = reward_name_converter(specific_dataset)
    name = getName2(dataset_name, specific_dataset)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}'
    else:
        path = f'./Finetuning/Kernels/{dataset_name}/Models/{str(step)}'
    file_count = count_files_in_folder(path)
    kernel_state_dicts = []
    for i in range(file_count):
        if(check_specific_dataset(dataset_name)):
            dir = f"./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Models/{str(step)}/{name}_Kernel_{str(i)}.pkl"
        else:
            dir = f"./Finetuning/Kernels/{dataset_name}/Models/{str(step)}/{name}_Kernel_{str(i)}.pkl"
        with open(dir, 'rb') as f:
            kernel_state_dicts.append(pickle.load(f))
    return kernel_state_dicts, obs_dim, act_dim

def get_kernel_stats(dataset_name, specific_dataset, step):
    specific_dataset = reward_name_converter(specific_dataset)
    name = getName2(dataset_name, specific_dataset)
    if(check_specific_dataset(dataset_name)):
        path = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Stats/{name}_Kernel_stats_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Kernels/{dataset_name}/Stats/{name}_Kernel_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats

def save_planner(model, dataset_name, specific_dataset, step: int,
                 task_id: Optional[int] = None):              # NEW arg
    # TODO(checkpoint-bridge): `model` is now (model_def, params)/params for the flax DiT1d (was a torch
    # nn.Module). The 'ema' field holds flax.serialization.to_state_dict(params); the torch->flax remap
    # (Dense weight (out,in)->kernel (in,out).T, bias->bias, LayerNorm weight->scale) is documented in get_planner.
    # `model` may be a TrainState (the finetuner passes self.ema_model) or a raw param tree. Store the
    # PARAM TREE only — get_planner's consumers load 'ema' into a DiT1d param tree, and SDETrainer saves
    # ema params the same way. Saving a full TrainState state-dict (step/params/opt_state) would break the load.
    ema_params = model.params if hasattr(model, 'params') else model
    data = {
        'dataset_name': dataset_name,
        'specific_dataset': specific_dataset,
        'task_id': task_id,                                   # NEW field
        'step': step,
        'ema': flax.serialization.to_state_dict(ema_params),
    }
    base = getName(dataset_name, specific_dataset)
    tid  = f"_task{task_id}" if task_id is not None else ""
    fname = f"{base}{tid}_Planner_{step}.pt"
    dir   = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}"
    os.makedirs(dir, exist_ok=True)
    savepath = f"{dir}/{fname}"
    with open(savepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"saved model to {savepath}")

def get_planner(dataset_name, specific_dataset, step,
                task_id: Optional[int] = None):               # NEW arg
    # TODO(checkpoint-bridge): returns the planner 'ema' params. Legacy files are torch.save dicts whose
    # 'ema' is a torch state_dict; ingest into the flax DiT1d param tree with the per-Dense remap
    # (weight (out,in)->kernel (in,out).T, bias->bias, LayerNorm weight->scale). New files are pickled flax dicts.
    base = getName(dataset_name, specific_dataset)
    tid  = f"_task{task_id}" if task_id is not None else ""
    path = f"./Finetuning/Planners/{dataset_name}/{specific_dataset}/{base}{tid}_Planner_{step}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, 'rb') as f:
        ema = pickle.load(f)['ema']
    # Tolerate checkpoints that stored a full TrainState state-dict ({'step','params','opt_state'})
    # instead of the bare param tree: return just the params so consumers get a DiT1d param tree.
    if isinstance(ema, dict) and 'params' in ema and 'step' in ema:
        ema = ema['params']
    return ema

def save_critic(model, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    # TODO(checkpoint-bridge): `model` is now (model_def, params)/params for the flax Critic (was a torch
    # nn.Module). Persist flax.serialization.to_state_dict; the torch->flax Dense remap is documented in get_critic_model.
    critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
    net_dict = flax.serialization.to_state_dict(model)
    os.makedirs(f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/', exist_ok=True)
    save_path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{critic_name}_Critic_{str(step)}.pkl'
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    with open(save_path, 'wb') as f:
        pickle.dump(net_dict, f)
    print(f"critic model save to {critic_name}_{str(step)}.pkl")

def get_critic_model(dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    # TODO(checkpoint-bridge): legacy critic checkpoints are torch state_dicts; ingest into the flax Critic
    # param tree with the per-Dense remap (weight (out,in)->kernel (in,out).T, bias->bias, LayerNorm
    # weight->scale). New checkpoints are pickled flax.serialization state dicts; load directly.
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{critic_name}_Critic_{str(step)}.pkl'
    with open(path, 'rb') as f:
        model_state_dict = pickle.load(f)
    return model_state_dict, obs_dim

def get_critic_stats(dataset_name, specific_dataset, task_id: Optional[int] = None,  step: int = 0) -> SAStats:
    critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/{critic_name}_Critic_stats_{str(step)}.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats 

def save_trajs(trajs, env_name, specific_env, step, task_id: Optional[int] = None):
    if(task_id is not None):
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{str(task_id)}'
    else:
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/'
    os.makedirs(path, exist_ok=True)
    save_path =  f'{path}/Generated_trajs_Info_{str(step)}.pkl'
    with open(save_path, 'wb') as f:
         pickle.dump(trajs, f)
    print(f"trajectories saved")

def get_trajs(env_name, specific_env, step, task_id: Optional[int] = None):
    if(task_id is not None):
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{str(task_id)}/Generated_trajs_Info_{str(step)}.pkl'
    else:
        path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{str(step)}.pkl'
    with open(path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs

def save_success_trajs_for_reward(trajs, env_name, specific_env, task_id, step):
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/trajs_task{task_id}_success_{step}.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajs, f)
    print("trajectories saved")

def load_success_trajs(env_name, specific_env, task_id, step):
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/trajs_task{task_id}_success_{step}.pkl'
    with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs

class Lambda:
    def __init__(self, lam: float, beta: float, eta_lam: float):
        self.lam = lam
        self.base_lam = lam
        self.beta = beta
        self.eta_lam = eta_lam
    
    def update(self, C):
        self.lam = np.maximum(self.base_lam, self.lam + (self.eta_lam * C))
        self.lam = np.clip(self.lam, 0.0, 5.0)
    
    def set_lam(self, lam: float):
        self.lam = lam

    def get_lam(self):
        return self.lam

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))

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
     
     elif(env_name == 'cube'):
          if specific_env == 'single-play':
                return 'Cube_SinglePlay'
          elif specific_env == 'single-noisy':
                return 'Cube_SingleNoisy'
          elif specific_env == 'double-play':
                return 'Cube_DoublePlay'
          elif specific_env == 'double-noisy':
                return 'Cube_DoubleNoisy'
          elif specific_env == 'triple-play':
                return 'Cube_TriplePlay'
          elif specific_env == 'triple-noisy':
                return 'Cube_TripleNoisy'
          elif specific_env == 'quadruple-play':
                return 'Cube_QuadruplePlay'
          elif specific_env == 'quadruple-noisy':
                return 'Cube_QuadrupleNoisy'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
          if specific_env == 'medium':
                return 'OG2DMaze_Medium'
          elif specific_env == 'large':
                return 'OG2DMaze_Large'
          elif specific_env == 'giant':
                return 'OG2DMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def getName2(env_name, specific_env):
     if(env_name == 'kitchen'):
          return 'Kitchen'

     elif(env_name == 'pointmaze'):
          if specific_env == 'umaze':
               return 'PointMaze_Umaze'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env== 'medium':
               return 'PointMaze_Medium'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")

     elif(env_name == 'antmaze'):
          if specific_env == 'medium':
               return 'AntMaze_Medium'
          elif specific_env == 'large':
               return 'AntMaze_Large'
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
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
          if specific_env == 'medium':
                return 'OG2DMaze_Medium'
          elif specific_env == 'large':
                return 'OG2DMaze_Large'
          elif specific_env == 'giant':
                return 'OG2DMaze_Giant'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def get_CriticName(env_name, specific_env, task_id: Optional[int] = None):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
         if(specific_env == 'large'):
              return 'PointMaze_Large'
         elif(specific_env == 'medium'):
              return 'PointMaze_Medium'
         elif(specific_env == 'unmaze'):
              return 'PointMaze_Unmaze'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")

     elif(env_name == 'cube'):
         if specific_env == 'single-play':
              return f'Cube_SinglePlay_task{task_id}'
         elif specific_env == 'single-noisy':
             return f'Cube_SingleNoisy_task{task_id}'
         elif specific_env == 'double-play':
             return f'Cube_DoublePlay_task{task_id}'
         elif specific_env == 'double-noisy':
             return f'Cube_DoubleNoisy_task{task_id}'
         elif specific_env == 'triple-play':
             return f'Cube_TriplePlay_task{task_id}'
         elif specific_env == 'triple-noisy':
             return f'Cube_TripleNoisy_task{task_id}'
         elif specific_env == 'quadruple-play':
             return f'Cube_QuadruplePlay_task{task_id}'
         elif specific_env == 'quadruple-noisy':
             return f'Cube_QuadrupleNoisy_task{task_id}'
         else:
             raise ValueError(f"Invalid cube dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
         if(task_id is None):
              raise ValueError('Task ID is required for cube dataset')
         if(specific_env == 'medium'):
              return f'OG2DMaze_Medium_task{task_id}'
         elif(specific_env == 'large'):
              return f'OG2DMaze_Large_task{task_id}'
         elif(specific_env == 'giant'):
              return f'OG2DMaze_Giant_task{task_id}'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def get_RewardName(env_name, specific_env, task_id: Optional[int] = None):
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
          elif specific_env == 'medium_dense':
               return 'PointMaze_MediumDense'
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
            raise ValueError('Task ID is required for cube dataset')
         if(specific_env == 'medium'):
              return f'OG2DMaze_Medium_Task{task_id}'
         elif(specific_env == 'large'):
              return f'OG2DMaze_Large_Task{task_id}'
         elif(specific_env == 'giant'):
              return f'OG2DMaze_Giant_Task{task_id}'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

class KernelDataset:
    def __init__(self, trajectories: List[TrajectoryDict], dataset_name: str, specific_dataset: str, step: int):
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
            obs = traj['observations']
            acts = traj['actions']
            for t in range(len(obs)-1):
                s_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                s_tp1 = self.stats.norm_obs(obs[t+1])
                data.append((s_t, a_t, s_tp1))
         self.data = data
         self.save_stats(dataset_name, specific_dataset, step)

    def save_stats(self, dataset_name, specific_dataset, step):
        specific_dataset = reward_name_converter(specific_dataset)
        name = getName2(dataset_name, specific_dataset)
        stats_name =  str(name) + f'_Kernel_stats_{str(step)}.pkl'
        if(check_specific_dataset(dataset_name)):
             stats_dir = f'./Finetuning/Kernels/{dataset_name}/{specific_dataset}/Stats/'
        else:
             stats_dir = f'./Finetuning/Kernels/{dataset_name}/Stats/'
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
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(s_next, dtype=np.float32),
        )

    def sample(self, batch_size):
        # fql-style host-side sampling: returns numpy batch (s, a, s_next).
        idxs = np.random.randint(0, len(self.data), size=batch_size)
        s = np.stack([np.asarray(self.data[i][0], dtype=np.float32) for i in idxs], axis=0)
        a = np.stack([np.asarray(self.data[i][1], dtype=np.float32) for i in idxs], axis=0)
        s_next = np.stack([np.asarray(self.data[i][2], dtype=np.float32) for i in idxs], axis=0)
        return s, a, s_next

class RewardDataset:
    def __init__(self, trajs: List[TrajectoryDict], sigma: float, dataset_name: str, specific_dataset: str, step: int, goal: Optional[np.array] = None, target_reward: Optional[float] = None, task_id: Optional[int] = None):
            
        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        
        for traj in trajs:
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
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            acts = traj['actions']
            rews = traj['rewards']
            """
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            """
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            rews = gaussian_filter1d(rews, sigma)
            for t in range(len(acts)):
                obs_t = self.stats.norm_obs(obs[t])
                a_t   = acts[t]
                r_t   = rews[t]
                transitions.append((obs_t, a_t, r_t))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, step)
    
    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = 0, step = 0):
        specific_dataset = reward_name_converter(specific_dataset)
        #reward_name = get_reward_name(dataset_name, specific_dataset, task_id)
        reward_name = get_RewardName(dataset_name, specific_dataset, task_id)
        stats_name =  str(reward_name) + f'_Reward_stats_{str(step)}.pkl'
        if(check_specific_dataset(dataset_name)):
            stats_dir = f'./Finetuning/Rewards/{dataset_name}/{specific_dataset}/Stats/'
        else:
            stats_dir = f'./Finetuning/Rewards/{dataset_name}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

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
        # fql-style host-side sampling: returns numpy batch (s, a, r).
        idxs = np.random.randint(0, len(self.transitions), size=batch_size)
        s = np.stack([np.asarray(self.transitions[i][0], dtype=np.float32) for i in idxs], axis=0)
        a = np.stack([np.asarray(self.transitions[i][1], dtype=np.float32) for i in idxs], axis=0)
        r = np.stack([np.asarray(self.transitions[i][2], dtype=np.float32) for i in idxs], axis=0)
        return s, a, r

    def boost_signal(self, target_reward, rews):
         rews = np.asarray(rews, dtype=np.float64).copy()
         rews = rews * target_reward
         return rews

def train_reward(trajs: List[TrajectoryDict],
                 dataset_name: str,
                 hidden_layers: int,
                 hidden_dim: int,
                 batch_size,
                 num_steps,
                 lr, min_lr, sigma,
                 step,
                 target_reward: Optional[float] = None,
                 specific_dataset: Optional[str] = None,
                 goal: Optional[np.array] = None,
                 task_id: Optional[int] = None,
                 *, rng=None):  # API-CHANGE: rng= threaded for param init (was implicitly stochastic)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    print(f"Training reward approximator for {dataset_name}_{specific_dataset} Dataset")
    dataset = RewardDataset(trajs, sigma, dataset_name, specific_dataset, step, goal, target_reward, task_id)
    reward_net = SimpleReward(obs_dim, act_dim, hidden_dim, hidden_layers)
    # CosineAnnealingLR(T_max=num_steps, eta_min=min_lr) folded into the optax schedule (reads opt_state.count).
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.adamw(schedule, weight_decay=1e-4)
    s0, a0, _ = dataset.sample(batch_size)
    rng, init_rng = jax.random.split(rng)
    params = reward_net.init(init_rng, jnp.asarray(s0), jnp.asarray(a0))['params']
    train_state = TrainState.create(reward_net, params, tx=tx)
    total_loss = 0
    counter = 0

    @jax.jit
    def _update(train_state, s, a, r):
        def loss_fn(params):
            pred = train_state(s, a, params=params)
            #loss = jnp.mean((pred - r) ** 2)
            loss = jnp.mean(optax.huber_loss(pred, r, delta=1.0))
            return loss, {'loss': loss}
        return train_state.apply_loss_fn(loss_fn)

    for i in range(num_steps):
           s, a, r = dataset.sample(batch_size)
           s = jnp.asarray(s)
           a = jnp.asarray(a)
           r = jnp.asarray(r)

           # Predicted Reward + gradient step (apply_loss_fn handles grad/step/sched)
           train_state, info = _update(train_state, s, a, r)
           total_loss += float(info['loss'])
           counter += 1
    save_reward_model(train_state.params, dataset_name, specific_dataset, task_id, step)
    print(f"reward model saved")

def train_reward_ensemble(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    hidden_layers: int,
    hidden_dim: int,
    batch_size: int,
    num_steps: int,
    lr: float,
    min_lr: float,
    ensemble_size: int = 5,
    bootstrap: bool = True,
    save_percentage: float = 0.0,
    sigma: Optional[float] = None,
    step: int = 0,
    target_reward: Optional[float] = None,
    specific_dataset: Optional[str] = None,
    goal: Optional[np.ndarray] = None,
    task_id: Optional[int] = None,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    *, rng=None,  # API-CHANGE: rng= threaded for param init (was implicitly stochastic)
):

    if rng is None:
        rng = jax.random.PRNGKey(0)
    trajs = drop_trajs(trajs, save_percentage)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
    dataset = RewardDataset(trajs, sigma, dataset_name, specific_dataset, step, goal, target_reward, task_id)
    # --- build model + optim
    reward_net = EnsembleReward(
        obs_dim, act_dim, hidden_dim, hidden_layers,
        ensemble_size=ensemble_size,
    )
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    if grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(schedule, weight_decay=weight_decay))
    else:
        tx = optax.adamw(schedule, weight_decay=weight_decay)
    s0, a0, _ = dataset.sample(batch_size)
    s0_e = np.broadcast_to(s0[None], (ensemble_size, *s0.shape))
    a0_e = np.broadcast_to(a0[None], (ensemble_size, *a0.shape))
    rng, init_rng = jax.random.split(rng)
    params = reward_net.init(init_rng, jnp.asarray(s0_e), jnp.asarray(a0_e))['params']
    train_state = TrainState.create(reward_net, params, tx=tx)

    @jax.jit
    def _update(train_state, s_e, a_e, r_e):
        def loss_fn(params):
            pred_e = train_state(s_e, a_e, params=params)            # (E, B)
            loss = jnp.mean(optax.huber_loss(pred_e, r_e, delta=1.0))
            return loss, {'loss': loss}
        return train_state.apply_loss_fn(loss_fn)

    running_loss = 0.0
    for step in range(1, num_steps + 1):
        s, a, r = dataset.sample(batch_size)
        if bootstrap and ensemble_size > 1:
            s_e, a_e, r_e = _bootstrap_per_member(s, a, r, ensemble_size, None)
        else:
            # diversity from random init only
            s_e = np.broadcast_to(s[None], (ensemble_size, *s.shape))
            a_e = np.broadcast_to(a[None], (ensemble_size, *a.shape))
            r_e = np.broadcast_to(r[None], (ensemble_size, *r.shape))
        # mean over (E*B) ≡ mean of per-member SmoothL1 losses
        """
        per_elem = optax.huber_loss(pred_e, r_e, delta=1.0)
        positive_weight = 50.0                       # try 8.0 ~ 30.0
        weights = jnp.where(r_e > 0, positive_weight, 1.0)
        loss = (weights * per_elem).mean()
        """
        train_state, info = _update(train_state, jnp.asarray(s_e), jnp.asarray(a_e), jnp.asarray(r_e))
        running_loss += float(info['loss'])
    save_reward_model(train_state.params, dataset_name, specific_dataset, task_id, step)
    print(f"reward model saved")

def train_kernel(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    batch_size=256,
    lr=1e-3,
    num_steps=10000,
    ensemble_size=10,
    λ_reg=1e-3,
    num_hidden_layers=2,
    hidden_dim=256,
    step: int = 0,
    constraint_type: str = "mahalanobis",
    quantile: float = 0.95,
    x_generated_plans: Optional[list] = None,
    accelerator=None,
    *, rng=None,  # API-CHANGE: rng= threaded for per-member param init (was implicitly stochastic)
):

    if accelerator is not None and accelerator.is_main_process:
          print(f"Training kernel for {dataset_name}_{specific_dataset}")
    if rng is None:
        rng = jax.random.PRNGKey(0)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # distributed role info
    if accelerator is None:
        is_main, rank, world = True, 0, 1
    else:
        is_main = accelerator.is_main_process
        rank = accelerator.process_index
        world = accelerator.num_processes

    # normalize constraint string
    ctype = "log_prob" if constraint_type in ("log_prob", "log_density") else "mahalanobis"

    # -----------------------------
    # Phase A: train on main only
    # -----------------------------
    ensemble = None
    if is_main:
        dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
        # Independently-checkpointed kernels stay a python list of (model_def, TrainState) per §11.
        ensemble = [
            RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers, hidden_dim)
            for _ in range(ensemble_size)
        ]
        s0, a0, _ = dataset.sample(batch_size)
        train_states = []
        for m in ensemble:
            rng, init_rng = jax.random.split(rng)
            params = m.init(init_rng, jnp.asarray(s0), jnp.asarray(a0))['params']
            train_states.append(TrainState.create(m, params, tx=optax.adamw(lr, weight_decay=1e-5)))

        noise_floors = [ts.model_def.noise_floor for ts in train_states]

        @jax.jit
        def _update(train_states, s, a, s_next):
            # forward all members (stored params, no grad) to get disagreement target
            mus = [ts(s, a)[0] for ts in train_states]
            mus_stack = jnp.stack(mus, axis=0)
            mu_mean = mus_stack.mean(axis=0)
            disagreement = jax.lax.stop_gradient(((mus_stack - mu_mean[None]) ** 2).mean(axis=0))

            new_states = []
            infos = []
            for i, ts in enumerate(train_states):
                def loss_fn(params, ts=ts, i=i):
                    mu, log_std = ts(s, a, params=params)
                    nll = ts(s_next, mu, log_std, params=params, method='gaussian_nll')
                    penalty = (disagreement / (jnp.exp(2 * log_std) + noise_floors[i])).sum(axis=-1).mean()
                    loss = nll + λ_reg * penalty
                    return loss, {'loss': loss}
                new_ts, info = ts.apply_loss_fn(loss_fn)
                new_states.append(new_ts)
                infos.append(info)
            return new_states, infos

        for _ in range(1, num_steps + 1):
            s, a, s_next = dataset.sample(batch_size)
            train_states, _ = _update(train_states, jnp.asarray(s), jnp.asarray(a), jnp.asarray(s_next))

        # save trained kernels for all ranks to load
        for idx, ts in enumerate(train_states):
            save_kernel_model(ts.params, dataset_name, specific_dataset, step, idx)
        print("Kernel model saved")

    if accelerator is not None:
        accelerator.wait_for_everyone()

    # ----------------------------------------
    # Phase B: threshold by all GPUs in parallel
    # ----------------------------------------
    threshold = None
    if x_generated_plans is not None:
        # every rank loads saved kernels
        kernel_state_dicts, _, _ = get_kernel(dataset_name, specific_dataset, step)
        kernel_stats = get_kernel_stats(dataset_name, specific_dataset, step)
        # TODO(checkpoint-bridge): rebuild each kernel as a TrainState; legacy torch state_dicts need the
        # per-Dense remap (weight (out,in)->kernel (in,out).T) before from_state_dict. We init a template and
        # restore the saved flax params. Kernels stay a python list (independently-loaded models, §11).
        eval_ensemble = []
        for sd in kernel_state_dicts:
            m = RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers, hidden_dim)
            s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
            a_ex = jnp.zeros((1, act_dim), dtype=jnp.float32)
            rng, init_rng = jax.random.split(rng)
            params = m.init(init_rng, s_ex, a_ex)['params']
            ts = TrainState.create(m, params, tx=None)
            ts = flax.serialization.from_state_dict(ts, sd) if isinstance(sd, dict) and 'params' in sd else ts.replace(params=sd)
            eval_ensemble.append(ts)

        # shard plans across ranks
        local_plans = x_generated_plans[rank::world]
        local_values = []
        for x in local_plans:
            for j in range(1, len(x) - 1):
                obs = jnp.asarray(kernel_stats.norm_obs(x[j, :obs_dim].copy()), dtype=jnp.float32)[None]
                act = jnp.asarray(x[j, obs_dim:obs_dim + act_dim].copy(), dtype=jnp.float32)[None]
                s_next = jnp.asarray(kernel_stats.norm_obs(x[j + 1, :obs_dim].copy()), dtype=jnp.float32)[None]

                if ctype == "log_prob":
                    v = float(compute_log_density(eval_ensemble, obs, act, s_next))
                else:
                    v = float(compute_total_mahalanobis_score(eval_ensemble, obs, act, s_next))
                local_values.append(v)

        # gather local values from all ranks
        if accelerator is not None:
            gathered = accelerator.gather_for_metrics([local_values], use_gather_object=True)
        else:
            gathered = [local_values]

        if is_main:
            values = []
            for chunk in gathered:
                values.extend(chunk)
            if ctype == "log_prob":
                threshold = float(np.quantile(values, 1 - quantile))
            else:
                threshold = float(np.quantile(values, quantile))
            print(f"New Threshold for {ctype}: {threshold}")
        else:
            threshold = 0.0

        # broadcast scalar threshold: torch.distributed removed. All ranks already hold the gathered values,
        # so every rank recomputes the same scalar from `gathered` to stay in sync (no cross-process op needed).
        if accelerator is not None and not is_main and gathered is not None:
            values = []
            for chunk in gathered:
                values.extend(chunk)
            if len(values) > 0:
                if ctype == "log_prob":
                    threshold = float(np.quantile(values, 1 - quantile))
                else:
                    threshold = float(np.quantile(values, quantile))

    if accelerator is not None:
        accelerator.wait_for_everyone()

    return threshold

def train_kernel_mog(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    batch_size=256,
    lr=1e-3,
    num_steps=10000,
    ensemble_size=10,
    λ_reg=1e-3,
    num_modes: Optional[int] = 8,
    num_hidden_layers=2,
    hidden_dim=256,
    kernel_noise_floor: Optional[float] = 1e-4,
    step: int = 0,
    constraint_type: str = "mahalanobis",
    quantile: float = 0.95,
    x_generated_plans: Optional[List] = None,
    accelerator=None,
    *, rng=None,  # API-CHANGE: rng= threaded for per-member param init (was implicitly stochastic)
):
    if accelerator is not None and accelerator.is_main_process:
          print(f"Training kernel for {dataset_name}_{specific_dataset}")
    if rng is None:
        rng = jax.random.PRNGKey(0)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    if accelerator is None:
        is_main, rank, world = True, 0, 1
    else:
        is_main = accelerator.is_main_process
        rank = accelerator.process_index
        world = accelerator.num_processes

    ctype = "log_prob" if constraint_type in ("log_prob", "log_density") else "mahalanobis"

    # -----------------------------
    # Phase A: train on main only
    # -----------------------------
    if is_main:
        dataset = KernelDataset(trajs, dataset_name, specific_dataset, step)
        # Independently-checkpointed kernels stay a python list of TrainStates per §11.
        ensemble = [
            MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, kernel_noise_floor)
            for _ in range(ensemble_size)
        ]
        s0, a0, _ = dataset.sample(batch_size)
        train_states = []
        for m in ensemble:
            rng, init_rng = jax.random.split(rng)
            params = m.init(init_rng, jnp.asarray(s0), jnp.asarray(a0))['params']
            tx = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(lr, weight_decay=1e-5))
            train_states.append(TrainState.create(m, params, tx=tx))

        noise_floors = [ts.model_def.noise_floor for ts in train_states]

        @jax.jit
        def _update(train_states, s, a, s_next):
            new_states = []
            infos = []
            for i, ts in enumerate(train_states):
                def loss_fn(params, ts=ts, i=i):
                    mu, log_std, weights = ts(s, a, params=params)
                    loss = ts(s_next, mu, log_std, weights, params=params, method='mog_nll')

                    mu_mean = mu.mean(axis=1)
                    disagreement = ((mu - mu_mean[:, None]) ** 2).mean(axis=1).mean(axis=0)
                    var = jnp.exp(2 * log_std) + noise_floors[i]
                    penalty = (disagreement / (var.mean(axis=1) + 1e-6)).mean()
                    total = loss + λ_reg * penalty
                    return total, {'loss': total}
                new_ts, info = ts.apply_loss_fn(loss_fn)
                new_states.append(new_ts)
                infos.append(info)
            return new_states, infos

        for _ in range(1, num_steps + 1):
            s, a, s_next = dataset.sample(batch_size)
            train_states, _ = _update(train_states, jnp.asarray(s), jnp.asarray(a), jnp.asarray(s_next))

        for idx, ts in enumerate(train_states):
            save_kernel_model(ts.params, dataset_name, specific_dataset, step, idx)
        print("Kernel model saved")

    if accelerator is not None:
        accelerator.wait_for_everyone()

    # ----------------------------------------
    # Phase B: threshold by all GPUs in parallel
    # ----------------------------------------
    threshold = None
    if x_generated_plans is not None:
        kernel_state_dicts, _, _ = get_kernel(dataset_name, specific_dataset, step)
        kernel_stats = get_kernel_stats(dataset_name, specific_dataset, step)
        # TODO(checkpoint-bridge): rebuild each MoG kernel as a TrainState; legacy torch state_dicts need the
        # per-Dense remap (weight (out,in)->kernel (in,out).T) before from_state_dict. Kernels stay a list (§11).
        eval_ensemble = []
        for sd in kernel_state_dicts:
            m = MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, kernel_noise_floor)
            s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
            a_ex = jnp.zeros((1, act_dim), dtype=jnp.float32)
            rng, init_rng = jax.random.split(rng)
            params = m.init(init_rng, s_ex, a_ex)['params']
            ts = TrainState.create(m, params, tx=None)
            ts = flax.serialization.from_state_dict(ts, sd) if isinstance(sd, dict) and 'params' in sd else ts.replace(params=sd)
            eval_ensemble.append(ts)

        local_plans = x_generated_plans[rank::world]
        local_values = []
        for x in local_plans:
            for j in range(1, len(x) - 1):
                obs = jnp.asarray(kernel_stats.norm_obs(x[j, :obs_dim].copy()), dtype=jnp.float32)[None]
                act = jnp.asarray(x[j, obs_dim:obs_dim + act_dim].copy(), dtype=jnp.float32)[None]
                s_next = jnp.asarray(kernel_stats.norm_obs(x[j + 1, :obs_dim].copy()), dtype=jnp.float32)[None]

                if ctype == "log_prob":
                    v = float(compute_log_density_mog(eval_ensemble, obs, act, s_next))
                else:
                    v = float(compute_total_mahalanobis_score_mog(eval_ensemble, obs, act, s_next))
                local_values.append(v)

        if accelerator is not None:
            gathered = accelerator.gather_for_metrics([local_values], use_gather_object=True)
        else:
            gathered = [local_values]

        if is_main:
            values = []
            for chunk in gathered:
                values.extend(chunk)
            if ctype == "log_prob":
                threshold = float(np.quantile(values, 1 - quantile))
            else:
                threshold = float(np.quantile(values, quantile))
            print(f"New Threshold for {ctype}: {threshold}")
        else:
            threshold = 0.0

        # broadcast scalar threshold: torch.distributed removed. Non-main ranks recompute from gathered values.
        if accelerator is not None and not is_main and gathered is not None:
            values = []
            for chunk in gathered:
                values.extend(chunk)
            if len(values) > 0:
                if ctype == "log_prob":
                    threshold = float(np.quantile(values, 1 - quantile))
                else:
                    threshold = float(np.quantile(values, quantile))

    if accelerator is not None:
        accelerator.wait_for_everyone()

    return threshold

def compute_threshold_mog(kernels, kernel_stats, obs_dim, act_dim, x, constraint_type: str = 'log_prob', quantile: float = 0.999, device: str = 'cuda'):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    values = []
    for i in range(len(x)):
       for j in range(1, len(x[i])-1):
           obs = jnp.asarray(kernel_stats.norm_obs(x[i][j, :obs_dim].copy()), dtype=jnp.float32)[None]
           act = jnp.asarray(x[i][j, obs_dim:(obs_dim+act_dim)].copy(), dtype=jnp.float32)[None]
           s_next = jnp.asarray(kernel_stats.norm_obs(x[i][j+1, :obs_dim].copy()), dtype=jnp.float32)[None]
           if(constraint_type == 'log_prob'):
               value = float(compute_log_density_mog(kernels, obs, act, s_next))
           else:
               value = float(compute_total_mahalanobis_score_mog(kernels, obs, act, s_next))
           values.append(value)
    if(constraint_type == 'log_prob'):
         threshold = np.quantile(values, (1 - quantile))
    elif(constraint_type == 'mahalanobis'):
         threshold = np.quantile(values, quantile)
    else:
         raise ValueError(f"Invalid constraint type: {constraint_type}")
    return threshold

def compute_threshold(kernels, kernel_stats, obs_dim, act_dim, x, constraint_type: str = 'log_prob', quantile: float = 0.999, device: str = 'cuda'):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    values = []
    for i in range(len(x)):
       for j in range(1, len(x[i])-1):
           obs = jnp.asarray(kernel_stats.norm_obs(x[i][j, :obs_dim].copy()), dtype=jnp.float32)[None]
           act = jnp.asarray(x[i][j, obs_dim:(obs_dim+act_dim)].copy(), dtype=jnp.float32)[None]
           s_next = jnp.asarray(kernel_stats.norm_obs(x[i][j+1, :obs_dim].copy()), dtype=jnp.float32)[None]
           if(constraint_type == 'log_prob'):
                value = float(compute_log_density(kernels, obs, act, s_next))
           else:
                value = float(compute_total_mahalanobis_score(kernels, obs, act, s_next))
           values.append(value)
    if(constraint_type == 'log_prob'):
         threshold = np.quantile(values, (1 - quantile))
    elif(constraint_type == 'mahalanobis'):
         threshold = np.quantile(values, quantile)
    else:
         raise ValueError(f"Invalid constraint type: {constraint_type}")
    return threshold
     
def check_Critic(dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = f"./Finetuning/Critics/{dataset_name}/{specific_dataset}/Models/{name}_Critic_{str(step)}.pkl"
    if(os.path.exists(path)):
        return True
    else:
        return False
         
def update_critic_stats(dataset_name, specific_dataset, new_stats: SAStats, task_id: Optional[int] = None, old_step: int = 0, momentum: float = 0.005) -> SAStats:
    old_stats = get_critic_stats(dataset_name, specific_dataset, task_id = task_id, step = old_step)
    stats = SAStats()
    stats.obs_mean = ((1 - momentum) * old_stats.obs_mean) + (momentum * new_stats.obs_mean)
    stats.obs_std = ((1 - momentum) * old_stats.obs_std) + (momentum * new_stats.obs_std)
    return stats

def get_new_critic_stats(trajs: List[TrajectoryDict]) -> SAStats:
    obs_all = []
    for traj in trajs:
        obs_all.append(traj['observations'])
    obs_all = np.concatenate(obs_all, axis = 0)
    stats = SAStats()
    stats.obs_mean = obs_all.mean(axis=0)
    stats.obs_std = obs_all.std(axis=0)+ 1e-8
    return stats

class Critic_Buffer():
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       trajs:  List[TrajectoryDict],
                       sigma: float,
                       target_reward: Optional[float] = None, 
                       horizon: int = 32,
                       gamma: float = 0.99,
                       lam: float = 0.95,
                       task_id: Optional[int] = None,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005):
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.data = CriticDataset(dataset_name,
                                  specific_dataset, 
                                  trajs, 
                                  sigma,
                                  target_reward,
                                  horizon,
                                  task_id, 
                                  old_step,  
                                  new_step, 
                                  momentum)
       
     
    def obtain_training_data(self, target_critic, batch_size: int, device: str):
        obs_chunks, rews_chunks = self.data.sample(batch_size)   # (B, T, dim), (B, T)
        obs_chunks = jnp.asarray(obs_chunks)
        rews_chunks = jnp.asarray(rews_chunks)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]

        # target_critic is a frozen TrainState; calling it without params= stops gradients (== torch no_grad).
        values = target_critic(obs_chunks)            # (B, T)

        deltas = (
              rews_chunks[:, :-1]
              + self.gamma * values[:, 1:]
               - values[:, :-1]
          )                                             # (B, T-1)

        advantages = jnp.zeros((B, T - 1))
        last_adv = jnp.zeros((B,))
        for t in reversed(range(T - 1)):
            last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
            advantages = advantages.at[:, t].set(last_adv)

        value_targets = values[:, 0] + advantages[:, 0]   # (B,)

        return obs_chunks[:, 0], value_targets

class CriticDataset:
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       trajs: List[TrajectoryDict],
                       sigma: float,
                       target_reward: Optional[float] = None,
                       horizon: int = 32,
                       task_id: Optional[int] = None, 
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005):
        # ----- gather raw obs/actions to fit stats -----
        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0)+ 1e-8
        if(old_step is not None):
             self.stats = update_critic_stats(dataset_name, specific_dataset, stats, task_id, old_step, momentum)
        else:
             self.stats = stats
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']      
            rews = traj['rewards']
            """
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")
            """
            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            if(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            if len(obs) < horizon:
                continue 
            for t in range(len(obs) - horizon):
                 obs_chunk = self.stats.norm_obs(obs[t : t + horizon]).astype(np.float32)
                 rews_chunk = rews[t: min(t+horizon, len(rews))]
                 transitions.append((obs_chunk, rews_chunk))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, new_step)
    
    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __getitem__(self, idx):
        obs_chunk, rews_chunk = self.transitions[idx]
        return (
            np.asarray(obs_chunk, dtype=np.float32),
            np.asarray(rews_chunk, dtype=np.float32),
        )
    def __len__(self):
        return len(self.transitions)

    def sample(self, batch_size):
        # fql-style host-side sampling: returns numpy batch (obs_chunks, rews_chunks).
        idxs = np.random.randint(0, len(self.transitions), size=batch_size)
        obs = np.stack([np.asarray(self.transitions[i][0], dtype=np.float32) for i in idxs], axis=0)
        rews = np.stack([np.asarray(self.transitions[i][1], dtype=np.float32) for i in idxs], axis=0)
        return obs, rews

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

def train_critic(trajs: List[TrajectoryDict],
                 dataset_name: str,
                 specific_dataset: str,
                 hidden_layers: int,
                 hidden_dim: int,
                 sigma: float,
                 batch_size,
                 num_steps,
                 gamma, lam, horizon,
                 lr,
                 min_lr,
                 tau,
                 old_step: Optional[int] = None,
                 new_step: int = 0,
                 momentum: float = 0.005,
                 target_reward = 1.0,
                 task_id: Optional[int] = None,
                 *, rng=None):  # API-CHANGE: rng= threaded for param init (was implicitly stochastic)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    rng, init_rng = jax.random.split(rng)
    params = critic.init(init_rng, s_ex)['params']
    if(old_step is not None):
        # TODO(checkpoint-bridge): get_critic_model returns the saved flax param tree (new ckpts) or a torch
        # state_dict (legacy) needing the per-Dense remap (weight (out,in)->kernel (in,out).T). For new flax
        # checkpoints the returned dict IS the params; restore into the template via from_state_dict.
        critic_state_dict, _ = get_critic_model(dataset_name, specific_dataset, task_id = task_id, step = old_step)
        params = flax.serialization.from_state_dict(params, critic_state_dict)
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-2))
    train_state = TrainState.create(critic, params, tx=tx)
    # target network: a frozen TrainState (no optimizer) updated via Polyak (target_update).
    target_state = TrainState.create(critic, copy.deepcopy(params), tx=None)
    buffer = Critic_Buffer(
            dataset_name=dataset_name,
            specific_dataset=specific_dataset,
            trajs=trajs,
            sigma=sigma,
            target_reward=target_reward,
            horizon=horizon,
            gamma=gamma,
            lam=lam,
            task_id=task_id,
            old_step=old_step,
            new_step=new_step,
            momentum=momentum)
    print(f"Training critic for {dataset_name}-{specific_dataset}")

    @jax.jit
    def _update(train_state, target_state, s, target_value):
        def loss_fn(params):
            q_pred = train_state(s, params=params)
            loss = jnp.mean(optax.huber_loss(q_pred, target_value, delta=1.0))
            #loss = jnp.mean((q_pred - target_value) ** 2)
            return loss, {'loss': loss}
        new_state, info = train_state.apply_loss_fn(loss_fn)
        # Soft update target network: tgt = (1 - tau) * tgt + tau * online == target_update(online, tgt, tau).
        new_target_params = target_update(new_state.params, target_state.params, tau)
        return new_state, target_state.replace(params=new_target_params), info

    total_loss = 0.0
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, target_value = buffer.obtain_training_data(target_state, batch_size, None)
           s = jnp.asarray(s)
           target_value = jnp.asarray(target_value)

           train_state, target_state, info = _update(train_state, target_state, s, target_value)
           total_loss += float(info['loss'])

           if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/200}")
                total_loss = 0.0
    save_critic(target_state.params, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")

class Critic_Test_Dataset:
    def __init__(self,
                 dataset_name: str,
                 specific_dataset: str,
                 checkpoint_step: int,
                 trajs: List[TrajectoryDict],
                 sigma: Optional[float] = None,
                 task_id: Optional[int] = None,
                 target_reward: Optional[float] = None,
                 horizon: int = 32,
                 gamma: float = 0.99):
        self.stats = get_critic_stats(dataset_name, specific_dataset, task_id, checkpoint_step)
        self.horizon = horizon
        self.gamma = gamma

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards'].copy()

            if target_reward is not None:
                rews = self.boost_signal(target_reward, rews)
            if sigma is not None:
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate=200/sigma)

            for t in range(len(obs) - horizon):
                obs_t = self.stats.norm_obs(obs[t])
                rews_chunk = rews[t : t + horizon]
                transitions.append((obs_t, rews_chunk))

        self.transitions = transitions
        print(f"Test dataset created: {len(self.transitions)} samples")

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        obs_t, rews_chunk = self.transitions[idx]
        return (
            np.asarray(obs_t, dtype=np.float32),
            np.asarray(rews_chunk, dtype=np.float32),
        )

    def iterate(self, batch_size):
        # fql-style deterministic, ordered iteration (replaces DataLoader(shuffle=False, drop_last=False)).
        for start in range(0, len(self.transitions), batch_size):
            chunk = self.transitions[start:start + batch_size]
            obs = np.stack([np.asarray(c[0], dtype=np.float32) for c in chunk], axis=0)
            rews = np.stack([np.asarray(c[1], dtype=np.float32) for c in chunk], axis=0)
            yield obs, rews

def test_critic(dataset_name: str,
                specific_dataset: str,
                hidden_layers: int,
                hidden_dim: int,
                checkpoint_step: int,
                gamma: float = 0.99,
                horizon: int = 32,
                sigma: Optional[float] = None,
                target_reward: float = 10.0,      # ← must match reward model
                trajs: List[TrajectoryDict] = None,
                task_id: Optional[int] = None,
                *, rng=None):  # API-CHANGE: rng= threaded for param-template init (was implicitly stochastic)
    if rng is None:
        rng = jax.random.PRNGKey(0)

    dataset = Critic_Test_Dataset(
        dataset_name, specific_dataset, checkpoint_step, trajs,
        sigma, task_id, target_reward, horizon, gamma
    )

    # Load model
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, checkpoint_step)
    model = Critic(obs_dim, hidden_dim, hidden_layers)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros((1, obs_dim), dtype=jnp.float32))['params']
    # TODO(checkpoint-bridge): restore saved flax params (new ckpt) / torch-remapped (legacy) into template.
    params = flax.serialization.from_state_dict(params, model_state_dict)
    model_state = TrainState.create(model, params, tx=None)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    print(f"Testing critic at checkpoint {checkpoint_step}...")

    # frozen TrainState: call without params= to stop gradients (== torch no_grad / eval).
    for s, rews_chunk in dataset.iterate(256):               # s: (B,), rews_chunk: (B, horizon)
        s = jnp.asarray(s)
        rews_chunk = jnp.asarray(rews_chunk)

        pred = jnp.squeeze(model_state(s), axis=-1)          # (B,)  ← normalized V(s)

        # Compute raw n-step return
        gamma_pow = jnp.asarray([gamma ** i for i in range(horizon)], dtype=jnp.float32)
        raw_target = (gamma_pow[None] * rews_chunk).sum(axis=1)

        # === Normalize target (CRITICAL) ===
        tgt_mean = raw_target.mean()
        tgt_std = raw_target.std() + 1e-8
        target = (raw_target - tgt_mean) / tgt_std

        loss = jnp.mean(optax.huber_loss(pred, target, delta=1.0))
        total_loss += float(loss) * s.shape[0]

        all_preds.extend(np.asarray(pred))
        all_targets.extend(np.asarray(target))

    avg_loss = total_loss / len(dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

    print(f"Test Results (Checkpoint {checkpoint_step}):")
    print(f"  Smooth L1 Loss : {avg_loss:.4f}")
    print(f"  MAE            : {mae:.4f}")
    print(f"  Mean Pred      : {np.mean(all_preds):.3f}")
    print(f"  Mean Target    : {np.mean(all_targets):.3f}")
    print(f"  Pred Std       : {np.std(all_preds):.3f}")

    return avg_loss, mae

def traj_cutoff(trajs, length):
    new_trajs = []
    for traj in trajs:
        L = len(traj['observations'])
        if(L > (length + 1)):
             index_obs = L - (length + 1)
             index_acts = L - length
             index_rews = L - length
             traj['observations'] = traj['observations'][index_obs:]
             traj['actions'] = traj['actions'][index_acts:]
             traj['rewards'] = traj['rewards'][index_rews:]
        new_trajs.append(traj)
    return new_trajs

def get_success_trajs(trajs):
    success_trajs = []
    for traj in trajs:
        if(traj['rewards'][-1] == 1.0):
            success_trajs.append(traj)
    return success_trajs

class PlannerDataset:
    def __init__(self, trajs: List[TrajectoryDict], horizon: int, dataset_name: str, specific_dataset: str, task_id: Optional[int] = None, cutoff_length: Optional[int] = None):
        self.trajs = copy.deepcopy(trajs)
        if(cutoff_length is not None):
            self.trajs = traj_cutoff(self.trajs, cutoff_length)
        print(f"total steps for Finetuning: {np.sum([len(traj['observations']) for traj in self.trajs])}")
        self.conditions = []
        self.horizon = horizon
        self.planner_processor = Planner_Processor(dataset_name, specific_dataset, task_id)
        for traj in self.trajs:
            obs = traj['observations']
            for t in range(len(obs)):
                s_norm = self.planner_processor.preprocess(obs[t])
                s_norm = np.asarray(s_norm, dtype=np.float32)
                self.conditions.append(s_norm)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        return self.conditions[idx]

    def sample(self, batch_size):
        # fql-style host-side sampling: returns a numpy batch of planner-normalized conditions.
        idxs = np.random.randint(0, len(self.conditions), size=batch_size)
        return np.stack([np.asarray(self.conditions[i], dtype=np.float32) for i in idxs], axis=0)

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        # API-CHANGE: returns the updated EMA param pytree (JAX params are immutable, no in-place mutation).
        # ma = beta * ma + (1 - beta) * current == target_update(current, ma, tau=1 - beta) (guide §5).
        return target_update(current_model, ma_model, 1 - self.beta)

    def update_average(self, old, new):
        if old is None:
            return new
        return jax.tree_util.tree_map(lambda o, n: o * self.beta + (1 - self.beta) * n, old, new)

class RewardTracker:
    """Track and plot rewards during finetuning (mirrors LossTracker API)."""

    def __init__(self, save_dir: str = "./logs/"):
        self.save_dir = save_dir
        self.steps = []
        self.rewards = []
        #self.learning_rates = []
        self.constraints = []
        os.makedirs(save_dir, exist_ok=True)

    def log_reward(self, step: int, reward: float, constraint: Optional[float] = None):
        self.steps.append(step)
        self.rewards.append(reward)
        if constraint is not None:
            self.constraints.append(constraint)

    def save_logs(self, filename: str = "reward_logs.pkl"):
        data = {
            'steps': self.steps,
            'rewards': self.rewards,
            'constraints': self.constraints
        }
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Reward logs saved to {save_path}")

    def plot_reward_curve(self,
                          save_path: Optional[str] = None,
                          title: str = "Finetuning Reward Curve",
                          show_constraint: bool = False,
                          smooth_window: int = 50):
        if not self.rewards:
            print("No reward data to plot!")
            return

        sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor':'black'})
        plt.rcParams.update({'font.size': 14})

        okabe_ito = ["#D55E00","#000000", "#E69F00", "#56B4E9", "#009E73",
                       "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FF0000"]
        raw_color    = okabe_ito[3]   
        smooth_color = okabe_ito[4] 
        constraint_color     = okabe_ito[9]  

        fig, ax1 = plt.subplots(figsize=(12, 8))
        steps = np.array(self.steps)
        rewards = np.array(self.rewards)


         # Plot smoothed if possible
        if len(rewards) > smooth_window and smooth_window > 1:
            smoothed = self._smooth_curve(rewards, smooth_window)
            # only plot where valid (not nan)
            valid_idx = ~np.isnan(smoothed)
            ax1.plot(steps[valid_idx], smoothed[valid_idx],
                     color=smooth_color, linewidth=2.5,
                     label=f'Smoothed Reward (window={smooth_window})')
        
        
        ax1.plot(steps, rewards, alpha=0.3, color=raw_color, linewidth=1.0, label='Raw Reward')
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12, color=raw_color)
        ax1.tick_params(axis='y', labelcolor=raw_color)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fancybox=True, fontsize=12)
        sns.despine()

        if show_constraint and self.constraints:
            ax2 = ax1.twinx()
            C_vals = np.array(self.constraints)
            ax2.plot(steps[:len(C_vals)], C_vals, color=constraint_color, alpha=0.7, linewidth=1.5, label='Constraint')
            ax2.set_ylabel('Constraint', fontsize=12, color=constraint_color)
            ax2.tick_params(axis='y', labelcolor=constraint_color)
            ax2.legend(loc='upper right')
        
        sns.despine()
        #plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

       
        if save_path is None:
            save_path = os.path.join(self.save_dir, "reward_curve.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reward curve saved to {save_path}")
        plt.show()
        return fig


    def _smooth_curve(self, data: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        padded = np.full_like(data, np.nan)
        padded[window-1:] = smoothed
        return padded

def karras_beta_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device: str
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns: t_grid, beta_grid, sigma_grid
    beta(t) computed from VP-SDE marginals using Karras timesteps.
    """
    t = jnp.linspace(1.0, 0.0, num_steps + 1)
    sigma_k = sigma_min * (sigma_max / sigma_min) ** t
    alpha = 1.0 / jnp.sqrt(1.0 + sigma_k**2)
    sigma = sigma_k * alpha

    # Compute β(t) from dσ²/dt = β(t) * σ²(t)
    # From VP-SDE: dσ²/dt = β(t) * (1 - σ²(t))
    # But we use numerical diff for stability

    sigma_sq = sigma**2
    d_sigma_sq = jnp.diff(sigma_sq, axis=0)
    dt = jnp.diff(t, axis=0)
    beta = d_sigma_sq / (1 - sigma_sq[:-1]) / dt
    beta = jnp.concatenate([beta, beta[-1][None]])  # pad last

    return t, beta, sigma

def clip_actions(x: jnp.ndarray, d_s: int) -> jnp.ndarray:
    actions = jnp.clip(x[..., d_s:], -1.0, 1.0)
    x = x.at[..., d_s:].set(actions)
    return x

def get_normalized_score(trajs, expert_score: Optional[float] = None):
    total = 0.0
    for i in range(len(trajs)):
        temp = 0.0
        for j in range(len(trajs[i]['rewards'])):
            #if(trajs[i]['rewards'][j] == 1):
            temp += (0.99**j) * trajs[i]['rewards'][j]
        total += temp
    avg_discounted_return = total / len(trajs)
    # 5. Compute normalized score
    normalized_score = 100 * avg_discounted_return 
    if(expert_score is not None):
        normalized_score = 100 * (normalized_score / expert_score)
    #print(f"Normalized Score: {normalized_score:.2f}")
    return normalized_score

def get_expert_score(dataset_name):
    if(dataset_name == 'kitchen'):
         data = get_dataset(dataset_name, 'complete')
         trajs = data.get_trajectories()
         score = get_normalized_score(trajs, None)
         return score
    else:
         return None

def get_current_state(s0, env_name):
    if env_name == 'antmaze':
        return np.concatenate([
            s0['observation'],
            s0['achieved_goal']
        ])
    if isinstance(s0, dict):
        return s0['observation']
    return s0

def load_hyperparameters(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

def rollout_parallel(
    env_name,
    specific_env,
    horizon = 32,
    steps_T = 50,
    num_karras = 10,
    eta = 0.8,
    episode_length = 4000,
    checkpoint_step = 1000000,
    num_envs = 8,
    goal_cell = None,
    start_cells = None,
    device: str = None,
    seed_base: int = 0,
    *, rng=None):  # API-CHANGE: rng= threaded for the diffusion sampler (was implicitly stochastic)
     #print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     #print(f"Running {num_envs} environments in parallel")
     trajs = []
     #print(f"Using device {device}")

     # Uses Accelerate's RANK env var (automatically set in DDP)
     rank = int(os.environ.get("RANK", 0))
     np.random.seed(12345 + rank + seed_base)
     if rng is None:
          rng = jax.random.PRNGKey(12345 + rank + seed_base)

     # Create environment factory function
     _, d_s, d_a = get_env(env_name, specific_env)
     def make_env():
         env, _, _ = get_env(env_name, specific_env)
         return env

     # Create vectorized environment
     vec_env = SyncVectorEnv([make_env for _ in range(num_envs)])
     #maze = env.unwrapped.maze  # Access the internal Maze object
     #maze_map = maze.maze_map
     #rows, cols = len(maze_map), len(maze_map[0])

     # Get Planner
     state_dict = get_planner(env_name, specific_env, checkpoint_step)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier")
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier")
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     # TODO(checkpoint-bridge): restore planner params (new flax ckpt / torch-remapped legacy) into a frozen
     # TrainState; the diffusion sampler calls it without params= (== torch eval / no_grad).
     rng, init_rng = jax.random.split(rng)
     params = model.init(init_rng, jnp.zeros((1, horizon, d_s + d_a)), jnp.zeros((1,)))['params']
     params = flax.serialization.from_state_dict(params, state_dict)
     model = TrainState.create(model, params, tx=None)

     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     # <<< MODIFIED: Unique env reset seeds per process to prevent identical trajectories across GPUs
     reset_seeds = list(range(seed_base, seed_base + num_envs))
     
     """
     if(goal_cell is not None):
         maze = env.unwrapped.maze  # Access the internal Maze object
         maze_map = maze.maze_map
         rows, cols = len(maze_map), len(maze_map[0])
         free_cells = []
         for row in range(rows):
             for col in range(cols):
                 if maze_map[row][col] != 1:  # 1 = wall; others are free/open
                       free_cells.append(np.array([row, col]))
         free_cells = np.array(free_cells)
         
         
         free_cells = np.array([[6,6], [1,1], [1,6], [3,2], [5,4], [3,4], [4,1], [4,6], [2,4], [2,1]])
         selected_indices = np.random.choice(len(free_cells), size=4, replace=False)
         selected_free_cells = free_cells[selected_indices]
         
         selected_free_cells = np.array([[6,6], [5,4], [2,4], [2,1]])
         start_cells = []
         for i in range(len(selected_free_cells)):
             if(np.array_equal(selected_free_cells[i], goal_cell)):
                 continue
             else:
                 start_cells.append(selected_free_cells[i].copy())
         start_cells = np.array(start_cells)
     else:
         start_cells = [None]
     """
     total_steps = 0
     for start_cell in start_cells:
       # Reset all environments
       #seeds = list(range(num_envs)) 
       opt = {}
       if goal_cell is not None:
             opt["goal_cell"] = goal_cell.copy()
       else:
             opt['goal_cell'] = None
       if start_cell is not None:
             opt["reset_cell"] = start_cell.copy()
       else:
             opt['reset_cell'] = None
       s0_vec = vec_env.reset(seed = reset_seeds, options=[opt for _ in range(num_envs)])
       current_states = s0_vec[0]['observation']
     
       # Store trajectories for each environment
       all_rewards = [0.0 for _ in range(num_envs)]
       done_envs = [False for _ in range(num_envs)]
       observations = [[] for _ in range(num_envs)]
       acts = [[] for _ in range(num_envs)]
       rewards = [[] for _ in range(num_envs)]
       for env_idx in range(num_envs):
          observations[env_idx].append(current_states[env_idx].copy())
     
       for i in range(episode_length):
          actions = np.zeros((num_envs, d_a))
         
          # Generate actions for each environment
          for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             current_state = current_states[env_idx]
             current_state_norm = planner_processor.preprocess(current_state)
             rng, sub = jax.random.split(rng)
             x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=sub)
             action = x[0, d_s:(d_s+d_a)].copy()
             actions[env_idx] = action
         
          # Step all environments at once
          obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
         
          # Update trajectories
          for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             
             observations[env_idx].append(obs_vec['observation'][env_idx].copy())
             acts[env_idx].append(actions[env_idx].copy())
             rewards[env_idx].append(rewards_vec[env_idx])
             all_rewards[env_idx] += rewards_vec[env_idx]
             
             current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
             if terminated_vec[env_idx] or truncated_vec[env_idx]:
                 done_envs[env_idx] = True
                 #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
          # Check if all environments are done
          if all(done_envs):
             #print("All environments completed!")
             break
     
       # Find the trajectory with the maximum reward
       for env_idx in range(num_envs):
          total_steps += (len(observations[env_idx]) - 1)
          trajs.append({
              'observations': np.asarray(observations[env_idx].copy()),
              'actions': np.asarray(acts[env_idx].copy()),
              'rewards': np.asarray(reward_processor(rewards[env_idx].copy(), env_name))
          })
        
     vec_env.close()
     if(goal_cell is None):
          expert_score = get_expert_score(env_name)
          score = get_normalized_score(trajs, expert_score)
     else:
          score = get_normalized_score(trajs)
     #save_trajs(trajs, env_name, specific_env, checkpoint_step)
     #print(f"Average Normalized Score: {score:.2f}")
     return trajs, score, total_steps

def rollout_parallel2(
     env_name,
     specific_env,
     horizon = 32,
     steps_T = 50,
     num_karras = 10,
     eta = 0.8,
     episode_length = 4000,
     checkpoint_step = 1000000,
     num_envs = 8,
     goal_cell: Optional[np.ndarray] = None,
     start_cells: Optional[List[np.ndarray]] = None,
     task_id: Optional[int] = None,
     device: str = None,
     seed_base: int = 0,
     continual_rollout = False,
     chunk_size = 5,
     *, rng=None):  # API-CHANGE: rng= threaded for the diffusion sampler (was implicitly stochastic)
     #print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     #print(f"Running {num_envs} environments in parallel")
     trajs = []
     #print(f"Using device {device}")

     # Uses Accelerate's RANK env var (automatically set in DDP)
     rank = int(os.environ.get("RANK", 0))
     np.random.seed(12345 + rank + seed_base)
     if rng is None:
          rng = jax.random.PRNGKey(12345 + rank + seed_base)

     # Create environment factory function
     _, d_s, d_a = get_env(env_name, specific_env, task_id = task_id)
     def make_env():
         env, _, _ = get_env(env_name, specific_env, task_id = task_id)
         return env
     
     # Create vectorized environment
     vec_env = SyncVectorEnv([make_env for _ in range(num_envs)])
     #maze = env.unwrapped.maze  # Access the internal Maze object
     #maze_map = maze.maze_map
     #rows, cols = len(maze_map), len(maze_map[0])
    
     # Get Planner
     state_dict = get_planner(env_name, specific_env, checkpoint_step, task_id)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier")
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier")
     elif(env_name == 'antmaze'):
         model = DiT1d(in_dim = d_s, emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif env_name == 'cube':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier")
     elif env_name == 'ogpointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier")
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     # TODO(checkpoint-bridge): restore planner params (new flax ckpt / torch-remapped legacy) into a frozen
     # TrainState; the diffusion sampler calls it without params= (== torch eval / no_grad).
     in_dim = d_s if env_name == 'antmaze' else (d_s + d_a)
     rng, init_rng = jax.random.split(rng)
     params = model.init(init_rng, jnp.zeros((1, horizon, in_dim)), jnp.zeros((1,)))['params']
     params = flax.serialization.from_state_dict(params, state_dict)
     model = TrainState.create(model, params, tx=None)
     
     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env, task_id)
     
     # <<< MODIFIED: Unique env reset seeds per process to prevent identical trajectories across GPUs
     reset_seeds = list(range(seed_base, seed_base + num_envs))
     
    
     total_steps = 0
     if (start_cells is not None):
      for start_cell in start_cells:
         # Reset all environments
         #seeds = list(range(num_envs)) 
        opt = {}
        if goal_cell is not None:
             opt["goal_cell"] = goal_cell.copy()
        else:
             opt['goal_cell'] = None
        if start_cell is not None:
             opt["reset_cell"] = start_cell.copy()
        else:
             opt['reset_cell'] = None
        
        s0_vec = vec_env.reset(seed = reset_seeds, options=[opt for _ in range(num_envs)])
        current_states = s0_vec[0]['observation']
     
        # Store trajectories for each environment
        all_rewards = [0.0 for _ in range(num_envs)]
        done_envs = [False for _ in range(num_envs)]
        observations = [[] for _ in range(num_envs)]
        acts = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        Temp_acts = [[] for _ in range(num_envs)]
        for env_idx in range(num_envs):
            observations[env_idx].append(current_states[env_idx].copy())
     
        for i in range(episode_length):
            actions = np.zeros((num_envs, d_a))
         
          # Generate actions for each environment
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
               if(continual_rollout):
                   if(len(Temp_acts[env_idx]) == 0):
                      current_state = current_states[env_idx]
                      current_state_norm = planner_processor.preprocess(current_state)
                      rng, sub = jax.random.split(rng)
                      x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=sub)
                      for k in range(len(x)):
                          Temp_acts[env_idx].append(x[k, d_s:(d_s+d_a)].copy())

                   actions[env_idx] = Temp_acts[env_idx][0].copy()
                   Temp_acts[env_idx] = Temp_acts[env_idx][1:].copy()
               else:
                   current_state = current_states[env_idx]
                   current_state_norm = planner_processor.preprocess(current_state)
                   rng, sub = jax.random.split(rng)
                   x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=sub)
                   action = x[0, d_s:(d_s+d_a)].copy()
                   actions[env_idx] = action

            # Step all environments at once
            obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)

             # Update trajectories
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
             
               observations[env_idx].append(obs_vec['observation'][env_idx].copy())
               acts[env_idx].append(actions[env_idx].copy())
               rewards[env_idx].append(rewards_vec[env_idx])
               all_rewards[env_idx] += rewards_vec[env_idx]
             
               current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
               if terminated_vec[env_idx] or truncated_vec[env_idx]:
                   done_envs[env_idx] = True
                   #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
             # Check if all environments are done
            if all(done_envs):
                #print("All environments completed!")
                break
        
        for env_idx in range(num_envs):
                   total_steps += (len(observations[env_idx]) - 1)
                   trajs.append({
                      'observations': np.asarray(observations[env_idx].copy()),
                      'actions': np.asarray(acts[env_idx].copy()),
                      'rewards': np.asarray(rewards[env_idx].copy())
         }) 
     else:
        opt =  {"task_id": task_id}
        #s0_vec = vec_env.reset(seed = reset_seeds, options=[opt for _ in range(num_envs)])
        #current_states = s0_vec[0]['observation']
        obs0, _ = vec_env.reset(seed=reset_seeds, options=[opt for _ in range(num_envs)])
        #obs0, _ = vec_env.reset(seed=reset_seeds)
        if isinstance(obs0, dict):
              current_states = obs0['observation']
        else:
              current_states = obs0
     
        # Store trajectories for each environment
        all_rewards = [0.0 for _ in range(num_envs)]
        done_envs = [False for _ in range(num_envs)]
        observations = [[] for _ in range(num_envs)]
        acts = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        Temp_acts = [[] for _ in range(num_envs)]
        for env_idx in range(num_envs):
            observations[env_idx].append(current_states[env_idx].copy())
     
        for i in range(episode_length):
            actions = np.zeros((num_envs, d_a))
         
            # Generate actions for each environment
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
               if(continual_rollout):
                   if(len(Temp_acts[env_idx]) == 0):
                      current_state = current_states[env_idx]
                      current_state_norm = planner_processor.preprocess(current_state)
                      rng, sub = jax.random.split(rng)
                      x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=sub)
                      for k in range(chunk_size):
                          Temp_acts[env_idx].append(x[k, d_s:(d_s+d_a)].copy())

                   actions[env_idx] = Temp_acts[env_idx][0].copy()
                   Temp_acts[env_idx] = Temp_acts[env_idx][1:].copy()
               else:
                   current_state = current_states[env_idx]
                   current_state_norm = planner_processor.preprocess(current_state)
                   rng, sub = jax.random.split(rng)
                   x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=sub)
                   action = x[0, d_s:(d_s+d_a)].copy()
                   actions[env_idx] = action
         
            # Step all environments at once
            obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
            obs_batch = obs_vec['observation'] if isinstance(obs_vec, dict) else obs_vec
         
             # Update trajectories
            for env_idx in range(num_envs):
               if done_envs[env_idx]:
                   continue
             
               observations[env_idx].append(obs_batch[env_idx].copy())
               acts[env_idx].append(actions[env_idx].copy())
               rewards[env_idx].append(rewards_vec[env_idx])
               all_rewards[env_idx] += rewards_vec[env_idx]
             
               current_states[env_idx] = obs_batch[env_idx].copy()
             
               if terminated_vec[env_idx] or truncated_vec[env_idx]:
                   done_envs[env_idx] = True
                   #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
             # Check if all environments are done
            if all(done_envs):
                    #print("All environments completed!")
                    break
     
            # Find the trajectory with the maximum reward
        for env_idx in range(num_envs):
                   total_steps += (len(observations[env_idx]) - 1)
                   trajs.append({
                      'observations': np.asarray(observations[env_idx].copy()),
                      'actions': np.asarray(acts[env_idx].copy()),
                      'rewards': np.asarray(reward_processor(rewards[env_idx].copy(), env_name))
        })     
     vec_env.close()
     success_rate = check_success_rate(trajs)
     print(f"success rate: {success_rate:.2f}")
     if(goal_cell is None):
            expert_score = get_expert_score(env_name)
            score = get_normalized_score(trajs, expert_score)
     else:
            score = get_normalized_score(trajs)
     #save_trajs(trajs, env_name, specific_env, checkpoint_step)
     #print(f"Average Normalized Score: {score:.2f}")
     return trajs, score, success_rate, total_steps

def rollout_parallel3(
    env_name, specific_env,
    horizon=32, steps_T=50, num_karras=10, eta=0.8,
    episode_length=4000, checkpoint_step=1000000,
    num_envs=8,
    goal_cell: Optional[np.ndarray] = None,
    start_cells: Optional[List[np.ndarray]] = None,
    task_id: Optional[int] = None,
    device: str = None,
    seed_base: int = 0,
    continual_rollout=False,
    chunk_size=10,          # currently unused
    *, rng=None,  # API-CHANGE: rng= threaded for the diffusion sampler (was implicitly stochastic)
):
    trajs = []
    total_steps = 0

    # Seeding
    rank = int(os.environ.get("RANK", 0))
    np.random.seed(12345 + rank + seed_base)
    if rng is None:
        rng = jax.random.PRNGKey(12345 + rank + seed_base)

    # Environment & Vector Env
    _, d_s, d_a = get_env(env_name, specific_env, task_id = task_id)

    def make_env():
        env, _, _ = get_env(env_name, specific_env, task_id = task_id)
        return env

    vec_env = SyncVectorEnv([make_env for _ in range(num_envs)])

    # Load model
    state_dict = get_planner(env_name, specific_env, checkpoint_step, task_id)

    if env_name in ['kitchen', 'pointmaze', 'cube']:
        model = DiT1d(
            in_dim=(d_s + d_a), emb_dim=128, d_model=256,
            n_heads=256//64, depth=2, timestep_emb_type="fourier"
        )
    elif env_name == 'antmaze':
        model = DiT1d(
            in_dim=d_s, emb_dim=128, d_model=256,
            n_heads=256//64, depth=2, timestep_emb_type="fourier"
        )
    else:
        raise ValueError(f"Invalid Environment: {env_name}")

    # TODO(checkpoint-bridge): restore planner params (new flax ckpt / torch-remapped legacy) into a frozen
    # TrainState; the diffusion sampler calls it without params= (== torch eval / no_grad).
    in_dim = d_s if env_name == 'antmaze' else (d_s + d_a)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros((1, horizon, in_dim)), jnp.zeros((1,)))['params']
    params = flax.serialization.from_state_dict(params, state_dict)
    model = TrainState.create(model, params, tx=None)

    planner_processor = Planner_Processor(env_name, specific_env)
    reset_seeds = list(range(seed_base, seed_base + num_envs))

    def run_rollout(options_list: Optional[dict] = None):
        """Helper to run one batch of environments (avoids duplication)."""
        nonlocal total_steps, rng
        
        if(options_list is not None):
             obs, info = vec_env.reset(seed=reset_seeds, options=options_list)
             #obs, info = vec_env.reset(options=options_list)
        else:
             obs, info = vec_env.reset(seed=reset_seeds)
             #obs, info = vec_env.reset()
        
        
        if isinstance(obs, dict):
            current_states = obs['observation']
        else:
            current_states = obs

        all_rewards = [0.0] * num_envs
        done_envs = [False] * num_envs
        observations = [[] for _ in range(num_envs)]
        acts = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        Temp_acts = [[] for _ in range(num_envs)]

        for env_idx in range(num_envs):
            observations[env_idx].append(current_states[env_idx].copy())

        for i in range(episode_length):
            actions = np.zeros((num_envs, d_a))

            for env_idx in range(num_envs):
                if done_envs[env_idx]:
                    continue

                current_state = current_states[env_idx]
                current_state_norm = planner_processor.preprocess(current_state)

                if continual_rollout and len(Temp_acts[env_idx]) > 0:
                    action = Temp_acts[env_idx].pop(0)
                else:
                    rng, sub = jax.random.split(rng)
                    x = sample_euler_karras(
                        current_state_norm, model, d_s, d_a,
                        horizon, steps_T, num_karras, eta, device, rng=sub
                    )
                    if continual_rollout:
                        Temp_acts[env_idx] = [x[k, d_s:(d_s + d_a)].copy() 
                                              for k in range(chunk_size)]
                        action = Temp_acts[env_idx].pop(0)
                    else:
                        action = x[0, d_s:(d_s + d_a)].copy()

                actions[env_idx] = action

            # Step
            obs_vec, rewards_vec, terminated_vec, truncated_vec, _ = vec_env.step(actions)

            # Consistent observation extraction
            obs_batch = obs_vec['observation'] if isinstance(obs_vec, dict) else obs_vec

            # Update
            for env_idx in range(num_envs):
                if done_envs[env_idx]:
                    continue

                observations[env_idx].append(obs_batch[env_idx].copy())
                acts[env_idx].append(actions[env_idx].copy())
                rewards[env_idx].append(rewards_vec[env_idx])
                all_rewards[env_idx] += rewards_vec[env_idx]
                current_states[env_idx] = obs_batch[env_idx].copy()

                if terminated_vec[env_idx] or truncated_vec[env_idx]:
                    done_envs[env_idx] = True

            if all(done_envs):
                break

        # Append finished trajectories
        for env_idx in range(num_envs):
            total_steps += len(observations[env_idx]) - 1
            trajs.append({
                'observations': np.asarray(observations[env_idx][:-1]),
                'actions': np.asarray(acts[env_idx]),
                'rewards': np.asarray(reward_processor(rewards[env_idx].copy(), env_name))  # FIXED
            })

    # ====================== Main Logic ======================
    if start_cells is not None and len(start_cells) > 0:
        for start_cell in start_cells:
            opt = {
                "goal_cell": goal_cell.copy() if goal_cell is not None else None,
                "reset_cell": start_cell.copy() if start_cell is not None else None,
            }
           
            run_rollout([opt] * num_envs)
    else:
        run_rollout()

    vec_env.close()

    valid, success_rate = checktrajs(trajs)
    print(f"valid: {valid}, success rate: {success_rate:.2f}")

    if goal_cell is None:
        expert_score = get_expert_score(env_name)
        score = get_normalized_score(trajs, expert_score)
    else:
        score = get_normalized_score(trajs)

    return trajs, score, success_rate, total_steps

import math
from dataclasses import dataclass
@dataclass
class AlphaSchedulerConfig:
    alpha_start: float
    alpha_end: float
    total_steps: int
    decay: bool = True
    
class AlphaScheduler:
    def __init__(self, config: AlphaSchedulerConfig):
        self.alpha_start = config.alpha_start
        self.alpha_end = config.alpha_end
        self.total_steps = config.total_steps
        self.decay_rate = self.total_steps / 10.0
        self.current_step = 1
        self.current_alpha = self.alpha_start
        self.decay = config.decay
    
    def step_alpha(self):
        if self.decay:
           if self.current_step > self.total_steps:
              self.current_alpha = self.alpha_end
           else:
              alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * math.exp(-self.current_step / self.decay_rate)
              self.current_alpha = max(alpha, self.alpha_end)
              self.current_step += 1
        else:
           self.current_alpha = self.alpha_start

    def get_alpha(self):
        return self.current_alpha

def checktrajs(trajs):
    success = 0
    for i in range(len(trajs)):
        Dict = {1: 0, 0: 0}
        for j in range(len(trajs[i]['rewards'])):
            if(trajs[i]['rewards'][j] not in Dict.keys()):
                Dict[int(trajs[i]['rewards'][j])] = 1
            else:
                Dict[int(trajs[i]['rewards'][j])] += 1
        if(1 in Dict.keys()):
           if(Dict[1] > 1):
              return False, 0
        if(len(list(Dict.keys())) > 2):
            return False, 0
        success += Dict[1]
    success_rate = success / len(trajs)
    return True, success_rate

def check_success_rate(trajs: List[TrajectoryDict]):
    success = 0
    for traj in trajs:
        if(traj['rewards'][-1] == 1.0):
            success += 1
    return success / len(trajs)
 
def check_device():
    device = jax.default_backend()
    if device == 'gpu':
        print("✅ Using GPU backend")
    elif device == 'tpu':
        print("✅ Using TPU backend")
    else:
        print("⚠️  Falling back to CPU (no GPU acceleration)")
    return device
      
def compute_threshold_mahalanobis(kernels, dataloader, quantile):
    all_D2_total = []
    for i, (s, a, s_next) in enumerate(dataloader):
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        s_next = jnp.asarray(s_next)
        #compute total mahalanobis distance
        # kernels is a list of (model_def, params); calling apply does not flow gradients (== torch no_grad).
        D2_total = compute_total_mahalanobis_score(kernels, s, a, s_next)
        all_D2_total.extend(np.asarray(D2_total))

    all_D2_total = np.array(all_D2_total)
    mean_D2_total = float(all_D2_total.mean())
    min_D2_total = float(all_D2_total.min())
    max_D2_total = float(all_D2_total.max())
    var_D2_total = float(all_D2_total.var())
    tau = float(np.quantile(all_D2_total, quantile))
    print(f"mean_D2_total = {mean_D2_total:.4f}")
    print(f"min_D2_total = {min_D2_total:.4f}")
    print(f"max_D2_total = {max_D2_total:.4f}")
    print(f"variance_D2_total = {var_D2_total:.4f}")
    print(f"τ ({quantile*100:.0f}th percentile) : {tau:.4f}")
    return tau

def compute_threshold_mahalanobis_mog(kernels, dataloader, quantile, device):
    chunks = []
    for s, a, s_next in dataloader:
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        s_next = jnp.asarray(s_next)
        d2 = compute_total_mahalanobis_score_mog(kernels, s, a, s_next)
        chunks.append(np.asarray(d2, dtype=np.float32))
    all_vals = np.concatenate(chunks, axis=0)
    tau = float(np.quantile(all_vals, quantile))
    print(f"mean_D2_total = {float(all_vals.mean()):.4f}")
    print(f"min_D2_total = {float(all_vals.min()):.4f}")
    print(f"max_D2_total = {float(all_vals.max()):.4f}")
    print(f"variance_D2_total = {float(all_vals.var()):.4f}")
    print(f"τ ({quantile*100:.0f}th percentile) : {tau:.4f}")
    return tau

def compute_threshold_log_prob(kernels, dataloader, quantile):
    all_log_density_total = []
    for i, (s, a, s_next) in enumerate(dataloader):
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        s_next = jnp.asarray(s_next)
        #compute total mahalanobis distance
        # kernels is a list of (model_def, params); calling apply does not flow gradients (== torch no_grad).
        log_density_total = compute_log_density(kernels, s, a, s_next)
        all_log_density_total.extend(np.asarray(log_density_total))

    all_log_density_total = np.array(all_log_density_total)
    mean_log_density_total = float(all_log_density_total.mean())
    min_log_density_total = float(all_log_density_total.min())
    max_log_density_total = float(all_log_density_total.max())
    var_log_density_total = float(all_log_density_total.var())
    tau = float(np.quantile(all_log_density_total, 1 - quantile))
    print(f"mean_D2_total = {mean_log_density_total:.4f}")
    print(f"min_D2_total = {min_log_density_total:.4f}")
    print(f"max_D2_total = {max_log_density_total:.4f}")
    print(f"variance_D2_total = {var_log_density_total:.4f}")
    print(f"τ ({(1 - quantile)*100:.0f}th percentile) : {tau:.4f}")
    return tau

def compute_threshold_log_prob_mog(kernels, dataloader, quantile, device):
    chunks = []
    for s, a, s_next in dataloader:
        s = jnp.asarray(s)
        a = jnp.asarray(a)
        s_next = jnp.asarray(s_next)
        lp = compute_log_density_mog(kernels, s, a, s_next)
        chunks.append(np.asarray(lp, dtype=np.float32))
    all_vals = np.concatenate(chunks, axis=0)
    tau = float(np.quantile(all_vals, 1.0 - quantile))
    print(f"mean_log_density_total = {float(all_vals.mean()):.4f}")
    print(f"min_log_density_total = {float(all_vals.min()):.4f}")
    print(f"max_log_density_total = {float(all_vals.max()):.4f}")
    print(f"variance_log_density_total = {float(all_vals.var()):.4f}")
    print(f"τ ({(1-quantile)*100:.0f}th percentile) : {tau:.4f}")
    return tau

def train_critic_with_planner(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 1000,
    *, rng=None,  # API-CHANGE: rng= threaded for param init + diffusion sampler (was implicitly stochastic)
):
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def _generate_plans_batch(
           s0_planner_norm: np.ndarray,   # (B, d_s) in planner-normalized space
           planner,
           d_s: int, d_a: int, horizon: int,
           steps_T: int, num_karras: int, eta: float,
           device: str,
           *, rng,
    ):
        # planner is a frozen TrainState; sample_euler_karras calls it without params= (== torch no_grad).
        plans = []
        for s0 in s0_planner_norm:
           rng, sub = jax.random.split(rng)
           x = sample_euler_karras(
               s0, planner, d_s, d_a, horizon,
               num_steps=steps_T, num_karras=num_karras, eta=eta, device=device, rng=sub,
           )
           plans.append(x)
        return jnp.asarray(np.stack(plans, axis=0), dtype=jnp.float32)

    device = check_device()
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # ------------------------------------------------------------------ critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    rng, init_rng = jax.random.split(rng)
    params = critic.init(init_rng, s_ex)['params']
    # TODO(checkpoint-bridge): get_critic_model returns the saved flax param tree (new ckpts) or a torch
    # state_dict (legacy) needing the per-Dense remap (weight (out,in)->kernel (in,out).T).
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=0,
    )
    params = flax.serialization.from_state_dict(params, critic_state)
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    critic_state_train = TrainState.create(critic, params, tx=tx)

    # target network: a frozen TrainState (no optimizer) updated via Polyak (target_update).
    target_critic = TrainState.create(critic, copy.deepcopy(params), tx=None)

    # ----------------------------------------------------------------- planner
    planner_def = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=2, timestep_emb_type="fourier",
    )
    rng, init_rng = jax.random.split(rng)
    planner_params = planner_def.init(
        init_rng, jnp.zeros((1, horizon, obs_dim + act_dim)), jnp.zeros((1,))
    )['params']
    # TODO(checkpoint-bridge): restore planner params (new flax ckpt / torch-remapped legacy) into a frozen
    # TrainState; the diffusion sampler calls it without params= (== torch eval / no_grad).
    planner_params = flax.serialization.from_state_dict(
        planner_params, get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner = TrainState.create(planner_def, planner_params, tx=None)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = jnp.asarray(planner_proc.stats.obs_mean, dtype=jnp.float32)
    planner_std  = jnp.asarray(
        np.maximum(planner_proc.stats.obs_std, 1e-3), dtype=jnp.float32,
    )

    # ----------------------------------------------------------- reward model
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_def = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    rng, init_rng = jax.random.split(rng)
    reward_params = reward_def.init(
        init_rng, jnp.zeros((1, obs_dim), dtype=jnp.float32), jnp.zeros((1, act_dim), dtype=jnp.float32)
    )['params']
    # TODO(checkpoint-bridge): restore reward params (new flax ckpt / torch-remapped legacy) into a frozen
    # TrainState; called without params= (== torch eval / no_grad).
    reward_params = flax.serialization.from_state_dict(reward_params, reward_state)
    reward_net = TrainState.create(reward_def, reward_params, tx=None)

    reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
    r_mean = jnp.asarray(reward_stat.obs_mean, dtype=jnp.float32)
    r_std  = jnp.asarray(np.maximum(reward_stat.obs_std, 1e-3), dtype=jnp.float32)

    # ----------------------------------- critic stats: load once, never save
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset,
        task_id=task_id, step=old_critic_checkpoint,
    )
    c_mean = jnp.asarray(critic_stat.obs_mean, dtype=jnp.float32)
    c_std  = jnp.asarray(np.maximum(critic_stat.obs_std, 1e-3), dtype=jnp.float32)

    # ---------------------------------------------------- starting-state pool
    s0_pool = np.concatenate([t['observations'] for t in trajs], axis=0).astype(np.float32)

    n = horizon - 1
    gamma_pow_t = jnp.asarray(
        [gamma ** t for t in range(n)], dtype=jnp.float32,
    )                                                                       # (n,)
    gamma_n = gamma ** n

    @jax.jit
    def _update(critic_state_train, target_critic, s0_critic, target_value):
        def loss_fn(params):
            v_pred = critic_state_train(s0_critic, params=params)            # (B,)
            #loss   = jnp.mean((v_pred - target_value) ** 2)
            loss = jnp.mean(optax.huber_loss(v_pred, target_value, delta=1.0))
            return loss, {'loss': loss}
        new_state, info = critic_state_train.apply_loss_fn(loss_fn)
        # Polyak target update: tgt = (1 - tau) * tgt + tau * online == target_update(online, tgt, tau).
        new_target_params = target_update(new_state.params, target_critic.params, tau)
        return new_state, target_critic.replace(params=new_target_params), info

    running = 0.0

    for k in range(1, num_steps + 1):
        # 1) sample raw start states
        idx    = np.random.randint(0, len(s0_pool), size=batch_size)
        s0_raw = s0_pool[idx]                                                # (B, d_s)

        # 2) plan with the diffusion planner
        s0_p  = np.stack([planner_proc.preprocess(o) for o in s0_raw])
        rng, sub = jax.random.split(rng)
        plans = _generate_plans_batch(
            s0_p, planner, obs_dim, act_dim, horizon,
            steps_T, num_karras, eta, device, rng=sub,
        )                                                                    # (B, H, d_s+d_a)

        # 3) recover RAW states from planner-norm; actions are already raw
        s_planner = plans[..., :obs_dim]                                     # (B, H, d_s)
        actions   = plans[..., obs_dim:]                                     # (B, H, d_a)
        s_raw     = s_planner * planner_std + planner_mean                   # (B, H, d_s)

        # 4) reward model: r̂(s_t, a_t) for t = 0..n-1
        # reward_net is a frozen TrainState; called without params= (== torch no_grad).
        B, H, _ = s_raw.shape
        s_for_r = (s_raw[:, :n] - r_mean) / r_std
        r_hat   = reward_net(
            s_for_r.reshape(B * n, -1),
            actions[:, :n].reshape(B * n, -1),
        ).reshape(B, n)                                                      # (B, n)

        # 5) discounted return + bootstrapped target value
        disc_return  = (gamma_pow_t[None] * r_hat).sum(axis=1)              # (B,)
        s_n_critic   = (s_raw[:, n] - c_mean) / c_std                       # (B, d_s)
        v_bootstrap  = target_critic(s_n_critic)                            # (B,)
        target_value = disc_return + gamma_n * v_bootstrap                   # (B,)

        # 6) input for V_β(s_0)
        s0_critic = (s_raw[:, 0] - c_mean) / c_std                          # (B, d_s)

        # 7) gradient step on V_β + Polyak target update
        critic_state_train, target_critic, info = _update(
            critic_state_train, target_critic, s0_critic, target_value
        )

        running += float(info['loss'])
        if k % log_every == 0:
            print(f"  step {k:>6}/{num_steps}   loss = {running / log_every:.4f}")
            running = 0.0

    save_critic(target_critic.params, dataset_name, specific_dataset, task_id, new_step)
    print("critic saved.")

class CriticDataset_Reward:
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs: List[TrajectoryDict],
                       horizon: int = 32,
                       old_step: Optional[int] = None,
                       new_step: int = 0,
                       momentum: float = 0.005,
                       task_id: Optional[int] = None,
                       *, rng=None):  # API-CHANGE: rng= threaded for reward param-template init
        if rng is None:
            rng = jax.random.PRNGKey(0)
        # ----- gather raw obs/actions to fit stats -----

        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)

        #get stats
        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0)+ 1e-8
        if(old_step is not None):
             self.stats = update_critic_stats(dataset_name, specific_dataset, stats, task_id, old_step, momentum)
        else:
             self.stats = stats

        device = check_device()
        _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
        reward_state, _, _ = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )
        reward_def = SimpleReward(
            obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
        )
        rng, init_rng = jax.random.split(rng)
        reward_params = reward_def.init(
            init_rng, jnp.zeros((1, obs_dim), dtype=jnp.float32), jnp.zeros((1, act_dim), dtype=jnp.float32)
        )['params']
        # TODO(checkpoint-bridge): restore reward params (new flax ckpt / torch-remapped legacy) into a frozen
        # TrainState; called without params= (== torch eval / no_grad).
        reward_params = flax.serialization.from_state_dict(reward_params, reward_state)
        reward_net = TrainState.create(reward_def, reward_params, tx=None)
        reward_stat = get_reward_stats(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )

        transitions = []

        for traj in trajs:
            obs = traj['observations']
            acts = traj['actions']
            T_traj = min(len(obs), len(acts))

            if T_traj < horizon:
                continue

            obs_for_r = reward_stat.norm_obs(obs[:T_traj]).astype(np.float32)
            s_t = jnp.asarray(obs_for_r, dtype=jnp.float32)
            a_t = jnp.asarray(acts[:T_traj], dtype=jnp.float32)
            rews = np.asarray(reward_net(s_t, a_t)).astype(np.float32)   # (T_traj,)
            # Scale down predicted rewards from reward model
            rews = np.clip(rews, -20.0, 20.0)      # adjust bounds if needed
            rews = rews / 5.0                      # or use a running std

            for t in range(len(obs) - horizon):
                 obs_chunk = self.stats.norm_obs(obs[t : t + horizon]).astype(np.float32)
                 rews_chunk = rews[t: min(t+horizon, len(rews))]
                 transitions.append((obs_chunk, rews_chunk))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, new_step)

    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __getitem__(self, idx):
        obs_chunk, rews_chunk = self.transitions[idx]
        return (
            np.asarray(obs_chunk, dtype=np.float32),
            np.asarray(rews_chunk, dtype=np.float32),
        )
    def __len__(self):
        return len(self.transitions)

    def sample(self, batch_size):
        # fql-style host-side sampling: returns numpy batch (obs_chunks, rews_chunks).
        idxs = np.random.randint(0, len(self.transitions), size=batch_size)
        obs = np.stack([np.asarray(self.transitions[i][0], dtype=np.float32) for i in idxs], axis=0)
        rews = np.stack([np.asarray(self.transitions[i][1], dtype=np.float32) for i in idxs], axis=0)
        return obs, rews

class Critic_Buffer_Reward():
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs:  List[TrajectoryDict],
                       horizon: int = 32,
                       gamma: float = 0.99,
                       lam: float = 0.95,
                       task_id: Optional[int] = None,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005):
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.data = CriticDataset_Reward(
            dataset_name         = dataset_name,
            specific_dataset     = specific_dataset,
            reward_hidden_layers = reward_hidden_layers,
            reward_hidden_dim    = reward_hidden_dim,
            reward_checkpoint    = reward_checkpoint,
            trajs                = trajs,
            horizon              = horizon,
            old_step             = old_step,
            new_step             = new_step,
            momentum             = momentum,
            task_id              = task_id,
        )
       
     
    def obtain_training_data(self, target_critic, batch_size: int, device: str):
        obs_chunks, rews_chunks = self.data.sample(batch_size)   # (B, T, dim), (B, T)
        obs_chunks = jnp.asarray(obs_chunks)
        rews_chunks = jnp.asarray(rews_chunks)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]

        # target_critic is a frozen TrainState; calling it without params= stops gradients (== torch no_grad).
        values = target_critic(obs_chunks)            # (B, T)

        deltas = (
              rews_chunks[:, :-1]
              + self.gamma * values[:, 1:]
               - values[:, :-1]
          )                                             # (B, T-1)

        advantages = jnp.zeros((B, T - 1))
        last_adv = jnp.zeros((B,))
        for t in reversed(range(T - 1)):
            last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
            advantages = advantages.at[:, t].set(last_adv)

        #value_targets = values[:, 0] + advantages[:, 0]   # (B,)
        values = target_critic(obs_chunks)                      # (B, T)
        deltas = (
              rews_chunks[:, :-1]
              + self.gamma * values[:, 1:]
              - values[:, :-1]
        )                                                       # (B, T-1)

        # GAE advantages
        advantages = jnp.zeros_like(deltas)
        last_adv = jnp.zeros((B,))
        for t in reversed(range(deltas.shape[1])):
            last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
            advantages = advantages.at[:, t].set(last_adv)

        # === ADD NORMALIZATION HERE ===
        value_targets = values[:, 0] + advantages[:, 0]         # raw targets

        # Normalize advantages and targets (running stats or batch stats)
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        tgt_mean = value_targets.mean()
        tgt_std  = value_targets.std() + 1e-8
        value_targets = (value_targets - tgt_mean) / tgt_std
        # =================================

        return obs_chunks[:, 0], value_targets

def train_critic_with_reward(trajs: List[TrajectoryDict], 
                 dataset_name: str, 
                 specific_dataset: str, 
                 reward_hidden_layers: int,
                 reward_hidden_dim: int,
                 reward_checkpoint: int,
                 critic_hidden_layers: int, 
                 critic_hidden_dim: int, 
                 batch_size, 
                 num_steps, 
                 gamma, lam, horizon, 
                 lr, 
                 min_lr, 
                 tau, 
                 old_step: Optional[int] = None,
                 new_step: int = 0,
                 momentum: float = 0.005,
                 task_id: Optional[int] = None,
                 *, rng=None):  # API-CHANGE: rng= threaded for param init (was implicitly stochastic)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    device = check_device()
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers)
    s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    rng, init_rng = jax.random.split(rng)
    params = critic.init(init_rng, s_ex)['params']
    if(old_step is not None):
        # TODO(checkpoint-bridge): get_critic_model returns the saved flax param tree (new ckpts) or a torch
        # state_dict (legacy) needing the per-Dense remap (weight (out,in)->kernel (in,out).T).
        critic_state_dict, _ = get_critic_model(dataset_name, specific_dataset, task_id = task_id, step = old_step)
        params = flax.serialization.from_state_dict(params, critic_state_dict)
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-2))
    train_state = TrainState.create(critic, params, tx=tx)
    # target network: a frozen TrainState (no optimizer) updated via Polyak (target_update).
    target_state = TrainState.create(critic, copy.deepcopy(params), tx=None)
    buffer = Critic_Buffer_Reward(
                       dataset_name,
                       specific_dataset,
                       reward_hidden_layers,
                       reward_hidden_dim,
                       reward_checkpoint,
                       trajs,
                       horizon,
                       gamma,
                       lam,
                       task_id,
                       old_step,
                       new_step,
                       momentum)
    print(f"Training critic for {dataset_name}-{specific_dataset}")

    @jax.jit
    def _update(train_state, target_state, s, target_value):
        def loss_fn(params):
            q_pred = train_state(s, params=params)
            loss = jnp.mean(optax.huber_loss(q_pred, target_value, delta=1.0))
            #loss = jnp.mean((q_pred - target_value) ** 2)
            return loss, {'loss': loss}
        new_state, info = train_state.apply_loss_fn(loss_fn)
        # Soft update target network: tgt = (1 - tau) * tgt + tau * online == target_update(online, tgt, tau).
        new_target_params = target_update(new_state.params, target_state.params, tau)
        return new_state, target_state.replace(params=new_target_params), info

    total_loss = 0.0
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, target_value = buffer.obtain_training_data(target_state, batch_size, device)
           s = jnp.asarray(s)
           target_value = jnp.asarray(target_value)

           train_state, target_state, info = _update(train_state, target_state, s, target_value)
           total_loss += float(info['loss'])

           if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/1000}")
                total_loss = 0.0
    save_critic(target_state.params, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")

@dataclass
class KernelConfig:
    """Everything needed to load the kernel ensemble + run the feasibility filter."""
    checkpoint: int                                # which kernel checkpoint to load
    type_kernel: str = 'robust'                    # 'robust' or 'mog'
    num_hidden_layers: int = 2                     # must match training
    hidden_dim: int = 256                          # must match training
    num_modes: int = 8                             # mog only
    noise_floor: Optional[float] = 1e-4            # mog only
    min_log_prob: float = -10.0                    # feasibility threshold
    oversample: int = 4  

def train_critic_with_planner2(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 1,
    *, rng=None,  # API-CHANGE: rng= threaded for param init + diffusion sampler (was implicitly stochastic)
):
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: str,
        *, rng,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )

        # TODO(checkpoint-bridge): rebuild each kernel as a (model_def, params) pair; legacy torch state_dicts
        # need the per-Dense remap (weight (out,in)->kernel (in,out).T) before from_state_dict. Kernels stay a
        # python list of independently-loaded models (§11), called via model_def.apply (no grad flows).
        s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        a_ex = jnp.zeros((1, act_dim), dtype=jnp.float32)
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                )
                rng, init_rng = jax.random.split(rng)
                k_params = k_net.init(init_rng, s_ex, a_ex)['params']
                k_params = flax.serialization.from_state_dict(k_params, sd)
                kernels.append((k_net, k_params))
        else:  # 'mog'
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                )
                rng, init_rng = jax.random.split(rng)
                k_params = k_net.init(init_rng, s_ex, a_ex)['params']
                k_params = flax.serialization.from_state_dict(k_params, sd)
                kernels.append((k_net, k_params))

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = jnp.asarray(kernel_stat.obs_mean, dtype=jnp.float32)
        k_std = jnp.asarray(
            np.maximum(kernel_stat.obs_std, 1e-3), dtype=jnp.float32,
        )
        return kernels, k_mean, k_std

    def is_plan_feasible(
        s_raw_plan,                         # (H, d_s)
        a_raw_plan,                         # (H, d_a)
        kernels,
        k_mean,                             # (d_s,)
        k_std,                              # (d_s,)
        kernel_config: KernelConfig,
        device:        str,
    ) -> bool:
        # kernels is a list of (model_def, params); calling apply does not flow gradients (== torch no_grad).
        s_k   = (s_raw_plan - k_mean) / k_std
        s_t   = s_k[:-1]
        a_t   = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = jnp.zeros(s_t.shape[0])
            for model_def, params in kernels:
                mu, log_std = model_def.apply({'params': params}, s_t, a_t)
                lp = model_def.apply({'params': params}, s_tp1, mu, log_std, method=model_def.log_prob)
                total = total + lp
            avg_lp = total / len(kernels)
        else:  # 'mog'
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all())

    def _generate_feasible_plans(
        s0_pool:        np.ndarray,
        planner,
        planner_proc:   Planner_Processor,
        planner_mean,
        planner_std,
        kernels,
        k_mean,
        k_std,
        kernel_config:  KernelConfig,
        obs_dim:        int,
        act_dim:        int,
        horizon:        int,
        steps_T:        int,
        num_karras:     int,
        eta:            float,
        batch_size:     int,
        device:         str,
        *, rng,
    ):
        accepted_plans = []
        accepted_s0    = []
        max_attempts   = kernel_config.oversample * batch_size
        attempts       = 0

        while len(accepted_plans) < batch_size and attempts < max_attempts:
            idx    = np.random.randint(0, len(s0_pool))
            s0_raw = s0_pool[idx]
            s0_p   = planner_proc.preprocess(s0_raw)
            # planner is a frozen TrainState; sample_euler_karras calls it without params= (== torch no_grad).
            rng, sub = jax.random.split(rng)
            x      = sample_euler_karras(
                s0_p, planner, obs_dim, act_dim, horizon,
                num_steps=steps_T, num_karras=num_karras,
                eta=eta, device=device, rng=sub,
            )

            x_t       = jnp.asarray(x, dtype=jnp.float32)
            s_planner = x_t[..., :obs_dim]
            a_raw     = x_t[..., obs_dim:]
            s_raw_pl  = s_planner * planner_std + planner_mean

            if is_plan_feasible(
                s_raw_plan    = s_raw_pl,
                a_raw_plan    = a_raw,
                kernels       = kernels,
                k_mean        = k_mean,
                k_std         = k_std,
                kernel_config = kernel_config,
                device        = device,
            ):
                accepted_plans.append(x_t)
                accepted_s0.append(s0_raw)
            attempts += 1

        if len(accepted_plans) == 0:
            raise RuntimeError(
                f"No feasible plans found after {attempts} attempts. "
                f"Lower `kernel_config.min_log_prob` or raise "
                f"`kernel_config.oversample`."
            )
        if len(accepted_plans) < batch_size:
            print(f"[Critic-Online] only {len(accepted_plans)}/{batch_size} "
                  f"feasible plans after {attempts} attempts "
                  f"(min_log_prob={kernel_config.min_log_prob}); proceeding")

        plans      = jnp.stack(accepted_plans, axis=0)
        s0_raw_acc = np.stack(accepted_s0, axis=0)
        return plans, s0_raw_acc

    # ------------------------------------------------------------------ setup
    device = check_device()
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # ------------------------------------------------------------------ critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    s_ex = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    rng, init_rng = jax.random.split(rng)
    params = critic.init(init_rng, s_ex)['params']
    # TODO(checkpoint-bridge): get_critic_model returns the saved flax param tree (new ckpts) or a torch
    # state_dict (legacy) needing the per-Dense remap (weight (out,in)->kernel (in,out).T).
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
    )
    params = flax.serialization.from_state_dict(params, critic_state)
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-2))
    critic_state_train = TrainState.create(critic, params, tx=tx)

    # target network: a frozen TrainState (no optimizer) updated via Polyak (target_update).
    target_critic = TrainState.create(critic, copy.deepcopy(params), tx=None)

    # ----------------------------------------------------------------- planner
    planner_def = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=2, timestep_emb_type="fourier",
    )
    rng, init_rng = jax.random.split(rng)
    planner_params = planner_def.init(
        init_rng, jnp.zeros((1, horizon, obs_dim + act_dim)), jnp.zeros((1,))
    )['params']
    # TODO(checkpoint-bridge): restore planner params (new flax ckpt / torch-remapped legacy) into a frozen
    # TrainState; the diffusion sampler calls it without params= (== torch eval / no_grad).
    planner_params = flax.serialization.from_state_dict(
        planner_params, get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner = TrainState.create(planner_def, planner_params, tx=None)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = jnp.asarray(planner_proc.stats.obs_mean, dtype=jnp.float32)
    planner_std  = jnp.asarray(
        np.maximum(planner_proc.stats.obs_std, 1e-3), dtype=jnp.float32,
    )

    # ----------------------------------------------------------- reward model
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_def = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    rng, init_rng = jax.random.split(rng)
    reward_params = reward_def.init(
        init_rng, jnp.zeros((1, obs_dim), dtype=jnp.float32), jnp.zeros((1, act_dim), dtype=jnp.float32)
    )['params']
    # TODO(checkpoint-bridge): restore reward params (new flax ckpt / torch-remapped legacy) into a frozen
    # TrainState; called without params= (== torch eval / no_grad).
    reward_params = flax.serialization.from_state_dict(reward_params, reward_state)
    reward_net = TrainState.create(reward_def, reward_params, tx=None)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = jnp.asarray(reward_stat.obs_mean, dtype=jnp.float32)
    r_std  = jnp.asarray(
        np.maximum(reward_stat.obs_std, 1e-3), dtype=jnp.float32,
    )

    # ------------------------------------------------------------------ kernel
    rng, kern_rng = jax.random.split(rng)
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config,
        obs_dim, act_dim, device, rng=kern_rng,
    )

    # ----------------------------------- critic stats: load once, never save
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset,
        task_id=task_id, step=0,
    )
    c_mean = jnp.asarray(critic_stat.obs_mean, dtype=jnp.float32)
    c_std  = jnp.asarray(
        np.maximum(critic_stat.obs_std, 1e-3), dtype=jnp.float32,
    )

    # ---------------------------------------------------- starting-state pool
    s0_pool = np.concatenate(
        [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)

    # === NEW: Running stats for targets ===
    running_tgt_mean = jnp.zeros(1)
    running_tgt_std  = jnp.ones(1)
    alpha = 0.99   # momentum
    # ======================================

    n = horizon - 1
    gamma_pow_t = jnp.asarray(
        [gamma ** t for t in range(n)], dtype=jnp.float32,
    )
    gamma_n = gamma ** n

    @jax.jit
    def _update(critic_state_train, target_critic, s0_critic, normalized_target):
        def loss_fn(params):
            v_pred = critic_state_train(s0_critic, params=params)              # (B',)
            loss   = jnp.mean(optax.huber_loss(v_pred, normalized_target, delta=1.0))
            return loss, {'loss': loss}
        new_state, info = critic_state_train.apply_loss_fn(loss_fn)
        # Polyak target update: tgt = (1 - tau) * tgt + tau * online == target_update(online, tgt, tau).
        new_target_params = target_update(new_state.params, target_critic.params, tau)
        return new_state, target_critic.replace(params=new_target_params), info

    running = 0.0

    for k in range(1, num_steps + 1):
        # 1) sample feasible plans (handles s_0 sampling internally)
        rng, sub = jax.random.split(rng)
        plans, _ = _generate_feasible_plans(
            s0_pool       = s0_pool,
            planner       = planner,
            planner_proc  = planner_proc,
            planner_mean  = planner_mean,
            planner_std   = planner_std,
            kernels       = kernels,
            k_mean        = k_mean,
            k_std         = k_std,
            kernel_config = kernel_config,
            obs_dim       = obs_dim,
            act_dim       = act_dim,
            horizon       = horizon,
            steps_T       = steps_T,
            num_karras    = num_karras,
            eta           = eta,
            batch_size    = batch_size,
            device        = device,
            rng           = sub,
        )                                                                     # (B', H, d_s+d_a)

        # 2) split planner output: states (planner-norm) and raw actions
        s_planner = plans[..., :obs_dim]                                      # (B', H, d_s)
        actions   = plans[..., obs_dim:]                                      # (B', H, d_a)
        s_raw     = s_planner * planner_std + planner_mean                    # (B', H, d_s)

        # 3) reward model: r̂(s_t, a_t) for t = 0..n-1
        # reward_net is a frozen TrainState; called without params= (== torch no_grad).
        B, H, _ = s_raw.shape
        s_for_r = (s_raw[:, :n] - r_mean) / r_std
        r_hat   = reward_net(
            s_for_r.reshape(B * n, -1),
            actions[:, :n].reshape(B * n, -1),
        ).reshape(B, n)

        # NEW: Strong scaling
        r_hat = jnp.clip(r_hat, -10.0, 10.0)
        r_hat = r_hat / 5.0                                                     # (B', n)

        # 4) discounted return + bootstrapped target value
        disc_return  = (gamma_pow_t[None] * r_hat).sum(axis=1)                # (B',)
        s_n_critic   = (s_raw[:, n] - c_mean) / c_std                         # (B', d_s)
        v_bootstrap  = target_critic(s_n_critic)                             # (B',)
        target_value = disc_return + gamma_n * v_bootstrap                    # (B',)

        # === NEW: Running normalization ===
        batch_mean = target_value.mean()
        batch_std  = target_value.std() + 1e-8

        running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
        running_tgt_std  = alpha * running_tgt_std  + (1 - alpha) * batch_std

        normalized_target = (target_value - running_tgt_mean) / running_tgt_std
        # =================================

        # 5) input for V_β(s_0)
        s0_critic = (s_raw[:, 0] - c_mean) / c_std                            # (B', d_s)

        # 6) gradient step on V_β + 7) Polyak target update
        critic_state_train, target_critic, info = _update(
            critic_state_train, target_critic, s0_critic, normalized_target
        )

        running += float(info['loss'])
        """
        if k % log_every == 0:
            print(f"  step {k:>6}/{num_steps}   loss = {running / log_every:.4f}")
            running = 0.0
        """

    save_critic(target_critic.params, dataset_name, specific_dataset, task_id, new_step)
    print("critic saved.")

