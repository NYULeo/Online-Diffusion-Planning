from optparse import Option
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "Pretrain"
import numpy as np
import minari
#import mediapy as media
import warnings
import gymnasium as gym
import gymnasium_robotics
import ogbench
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from torch.utils.data import Dataset
import pickle
try:
    from Pretrain.utils import SAStats
except ModuleNotFoundError:
    from utils import SAStats
import os
from typing import Optional, List, Dict, TypedDict
import numpy as np

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray

"""
def determine_stride(dataset_name, specific_dataset):
     if(dataset_name == 'antmaze'):
          return True
     else:
          return False
"""



#-------------------------------------------------------------------------------------#
#------------------------------------- Dataset ---------------------------------------#
#-------------------------------------------------------------------------------------#
def get_env(env_name, specific_env, render_mode = None, task_id: Optional[int] = None, goal: Optional[np.array] = None, episode_length: Optional[int] = None):
    data = get_dataset(env_name, specific_env, task_id, goal, episode_length)
    env = data.get_env(render_mode)
    d_s = data.get_state_dim()
    d_a = data.get_action_dim()
    return  env, d_s, d_a

def merger(traj_1, traj_2):
     states_1 = traj_1['observations']
     states_2 = traj_2['observations']
     if (np.array_equal(states_1[len(states_1)-1], states_2[0])):
          trajectory = {
                      'observations': np.concatenate((traj_1['observations'], traj_2['observations'][1:])),
                      'actions': np.concatenate((traj_1['actions'], traj_2['actions'])),
                      'rewards': np.concatenate((traj_1['rewards'], traj_2['rewards']))
                    }
          return trajectory
     else:
          return None   

def sparse_reward_processor(rewards):
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

    def mode_reward_processor(rewards):
        new_rews = [0]*len(rewards)
        for i in range(1, len(rewards)):
             if(rewards[i] == rewards[i-1]+1):
                new_rews[i] = 1.0
        return np.array(new_rews, dtype = np.float64)


    if(name in ('cube', 'ogpointmaze', 'antmaze', 'humanoidmaze', 'puzzle', 'scene')):
         rewards = ogbench_reward_processor(rewards)
         #rewards = mode_reward_processor(rewards)
         return rewards
    else:
         return spare_reward_processor(rewards)
    
def reward_processor_2(rewards):
     new_rewards = [0]*len(rewards)
     if(rewards[-1] == 0.0):
         new_rewards[-1] = 1.0
     return np.array(new_rewards, dtype = np.float64)

def get_dataset(name: str, 
                specific_name: str, 
                task_id: Optional[int] = None, 
                goal: Optional[np.array] = None, 
                traj_length: Optional[int] = None, 
                mode: Optional[str] = None):
       if(name == 'ogpointmaze'):
             if(task_id is None):
                 return OGPointmazeDataset(specific_name)
             else:
                 return OGPointmazeDataset_Singletask(specific_name, task_id, traj_length, mode)
       if(name == 'antmaze'):
             if(task_id is None):
                 return AntmazeDataset(specific_name)
             else:
                 return AntmazeDataset_Singletask(specific_name, task_id, traj_length, mode)
       if(name == 'humanoidmaze'):
             if(task_id is None):
                 return HumanoidmazeDataset(specific_name)
             else:
                 return HumanoidmazeDataset_Singletask(specific_name, task_id, traj_length, mode)
       elif(name == 'cube'):
            if(task_id is None):
                return CubeDataset(specific_name)
            else:
                return CubeDataset_Singletask(specific_name, task_id, traj_length, mode)
       elif name == "scene":
            if task_id is None:
                return SceneDataset(specific_name)
            else:
                return SceneDataset_Singletask(specific_name, task_id, traj_length, mode)
       elif name == "puzzle":
            if task_id is None:
                return PuzzleDataset(specific_name)
            else:
                return PuzzleDataset_Singletask(specific_name, task_id, traj_length, mode)
       else:
            raise ValueError(f"Invalid Dataset name: {name}")   

class OGPointmazeDataset:
    def __init__(self, name: str):
        
        self.name = name

        name_to_id = {
            "medium": f"pointmaze-medium-navigate-v0",
            "large": f"pointmaze-large-navigate-v0",
            "giant": f"pointmaze-giant-navigate-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]
        
        
        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )
       
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
       
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                }

                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class OGPointmazeDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None, mode: Optional[str] = 'reward'):
        
        self.name = name
        self.traj_length = traj_length
        self.mode = mode
        
        name_to_id = {
            "medium": f"pointmaze-medium-navigate-singletask-task{task_id}-v0",
            "large": f"pointmaze-large-navigate-singletask-task{task_id}-v0",
            "giant": f"pointmaze-giant-navigate-singletask-task{task_id}-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )
    """
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        if(self.mode == 'critic'):
           for i in range(N):
            # End of a natural episode (terminal or dataset end)
               if self.dataset["rewards"][i] == 0 or self.dataset['terminals'][i] == 1:
                     obs_slice = self.dataset["observations"][last_start : i]
                     act_slice = self.dataset["actions"][last_start : i]
                     rews = np.zeros(len(act_slice))
                     L = len(obs_slice)
                     if(self.traj_length is not None):
                           index = L - self.traj_length
                           if(index < 0):
                                index = 0
                     else:
                            index =  0
                
                    
                     if len(act_slice) < 3:
                          last_start = i + 1
                          continue
                     

                     if(self.dataset['rewards'][i] == 0):
                         rews[-1] = 1.0
                         trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           'rewards': rews[index:]
                          }
                         
                         trajectories.append(trajectory)
                         last_start = i + 1
                     else:
                         last_start = i + 1
            
        elif(self.mode == 'reward'):
             rews = np.zeros(len(self.dataset['rewards']))
             for i in range(N):
               # End of a natural episode (terminal or dataset end)
                if(self.dataset['rewards'][i] == 0):
                    rews[i-1] = 1.0
                if self.dataset["terminals"][i] == 1 or i == N - 1:
                    obs_slice = self.dataset["observations"][last_start : i]
                    act_slice = self.dataset["actions"][last_start : i]
                    rews_slice = rews[last_start : i]
                    if len(act_slice) < 10:
                       last_start = i + 1
                       continue

                    trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                        "rewards": rews_slice,
                    }

                    trajectories.append(trajectory)
                    last_start = i + 1
        else:
              raise ValueError(f"Invalid Mode: {self.mode}")
        return trajectories
    """

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
       
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        rewards = reward_processor(self.dataset['rewards'].copy(), 'ogpointmaze')
        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            #if self.dataset['terminals'][i] == 1 or self.dataset['rewards'][i] == 0:
            if self.dataset['terminals'][i] == 1:
                     obs_slice = self.dataset["observations"][last_start : i+1].copy()
                     act_slice = self.dataset["actions"][last_start : i].copy()
                     rews = rewards[last_start: i].copy()
                     masks = self.dataset['masks'][last_start : i].copy()
                     
            
                     L = len(obs_slice)
                     if(self.traj_length is not None):
                           index = L - self.traj_length
                           if(index < 0):
                                index = 0
                     else:
                            index =  0
                
                     
                     if len(act_slice) < 10:
                          last_start = i + 1
                          continue

                     """
                     if(self.mode == 'reward'):
                        if(sum(rews) == 0):
                            last_start = i + 1
                            continue 
                     """
                         
                     trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           "rewards":  rews[index:],
                           'masks': masks[index:]
                     }
                         
                     trajectories.append(trajectory)
                     last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode = render_mode)
        return env

class AntmazeDataset:
    def __init__(self, name: str):
        
        self.name = name

        name_to_id = {
            "medium": f"antmaze-medium-navigate-v0",
            "large": f"antmaze-large-navigate-v0",
            "giant": f"antmaze-giant-navigate-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]
        
        
        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )
       
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
       
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                }

                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class AntmazeDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None, mode: Optional[str] = 'reward'):
        
        self.name = name
        self.traj_length = traj_length
        self.mode = mode
        
        name_to_id = {
            "medium": f"antmaze-medium-navigate-singletask-task{task_id}-v0",
            "large": f"antmaze-large-navigate-singletask-task{task_id}-v0",
            "giant": f"antmaze-giant-navigate-singletask-task{task_id}-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )
 
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        rewards = reward_processor(self.dataset['rewards'].copy(), 'antmaze')
        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            #if self.dataset['terminals'][i] == 1 or self.dataset['rewards'][i] == 0:
            if self.dataset['terminals'][i] == 1:
                     obs_slice = self.dataset["observations"][last_start : i+1].copy()
                     act_slice = self.dataset["actions"][last_start : i].copy()
                     rews = rewards[last_start: i].copy()
                     masks = self.dataset['masks'][last_start : i].copy()
                     
            
                     L = len(obs_slice)
                     if(self.traj_length is not None):
                           index = L - self.traj_length
                           if(index < 0):
                                index = 0
                     else:
                            index =  0
                
                     
                     if len(act_slice) < 10:
                          last_start = i + 1
                          continue

                     """
                     if(self.mode == 'reward'):
                        if(sum(rews) == 0):
                            last_start = i + 1
                            continue 
                     """
                         
                     trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           "rewards":  rews[index:],
                           'masks': masks[index:]
                     }
                         
                     trajectories.append(trajectory)
                     last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode = render_mode)
        return env

class HumanoidmazeDataset:
    def __init__(self, name: str):
        
        self.name = name

        name_to_id = {
            "medium": f"humanoidmaze-medium-navigate-v0",
            "large": f"humanoidmaze-large-navigate-v0",
            "giant": f"humanoidmaze-giant-navigate-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]
        
        
        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )
       
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
       
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                }

                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class HumanoidmazeDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None, mode: Optional[str] = 'reward'):
        
        self.name = name
        self.traj_length = traj_length
        self.mode = mode
        
        name_to_id = {
            "medium": f"humanoidmaze-medium-navigate-singletask-task{task_id}-v0",
            "large": f"humanoidmaze-large-navigate-singletask-task{task_id}-v0",
            "giant": f"humanoidmaze-giant-navigate-singletask-task{task_id}-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )
 
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        rewards = reward_processor(self.dataset['rewards'].copy(), 'humanoidmaze')
        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            #if self.dataset['terminals'][i] == 1 or self.dataset['rewards'][i] == 0:
            if self.dataset['terminals'][i] == 1:
                     obs_slice = self.dataset["observations"][last_start : i+1].copy()
                     act_slice = self.dataset["actions"][last_start : i].copy()
                     rews = rewards[last_start: i].copy()
                     masks = self.dataset['masks'][last_start : i].copy()
                     
            
                     L = len(obs_slice)
                     if(self.traj_length is not None):
                           index = L - self.traj_length
                           if(index < 0):
                                index = 0
                     else:
                            index =  0
                
                     
                     if len(act_slice) < 10:
                          last_start = i + 1
                          continue

                     """
                     if(self.mode == 'reward'):
                        if(sum(rews) == 0):
                            last_start = i + 1
                            continue 
                     """
                         
                     trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           "rewards":  rews[index:],
                           "masks": masks[index:]
                     }
                         
                     trajectories.append(trajectory)
                     last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode = render_mode)
        return env

class CubeDataset:
    def __init__(self, name: str, task_id: Optional[int] = None):
        
        self.name = name
        if(task_id is None):
           name_to_id = {
            "single-play": "cube-single-play-v0",
            "single-noisy": "cube-single-noisy-v0",
            "double-play": "cube-double-play-v0",
            "double-noisy": "cube-double-noisy-v0",
            "triple-play": "cube-triple-play-v0",
            "triple-noisy": "cube-triple-noisy-v0",
            "quadruple-play": "cube-quadruple-play-v0",
            "quadruple-noisy": "cube-quadruple-noisy-v0",
           }
        else:
             name_to_id = {
            "single-play": f"cube-single-play-singletask-task{task_id}-v0",
            "single-noisy": f"cube-single-noisy-singletask-task{task_id}-v0",
            "double-play": f"cube-double-play-singletask-task{task_id}-v0",
            "double-noisy": f"cube-double-noisy-singletask-task{task_id}-v0",
            "triple-play": f"cube-triple-play-singletask-task{task_id}-v0",
            "triple-noisy": f"cube-triple-noisy-singletask-task{task_id}-v0",
            "quadruple-play": f"cube-quadruple-play-singletask-task{task_id}-v0",
            "quadruple-noisy": f"cube-quadruple-noisy-singletask-task{task_id}-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]
        
        
        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )


    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
       
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                }

                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class CubeDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None, mode: Optional[str] = None):
        
        self.name = name
        self.traj_length = traj_length
        self.mode = mode
        name_to_id = {
            "single-play": f"cube-single-play-singletask-task{task_id}-v0",
            "single-noisy": f"cube-single-noisy-singletask-task{task_id}-v0",
            "double-play": f"cube-double-play-singletask-task{task_id}-v0",
            "double-noisy": f"cube-double-noisy-singletask-task{task_id}-v0",
            "triple-play": f"cube-triple-play-singletask-task{task_id}-v0",
            "triple-noisy": f"cube-triple-noisy-singletask-task{task_id}-v0",
            "quadruple-play": f"cube-quadruple-play-singletask-task{task_id}-v0",
            "quadruple-noisy": f"cube-quadruple-noisy-singletask-task{task_id}-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
                 self.dataset_id, render_mode="rgb_array"
            )

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
       
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        rewards = reward_processor(self.dataset['rewards'].copy(), 'cube')
        #rewards =  reward_processor_2(self.dataset['rewards'].copy())
        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            #if self.dataset['terminals'][i] == 1 or self.dataset['rewards'][i] == 0:
            if self.dataset['terminals'][i] == 1:
                     obs_slice = self.dataset["observations"][last_start : i+1].copy()
                     act_slice = self.dataset["actions"][last_start : i].copy()
                     rews = rewards[last_start: i].copy()
                     masks = self.dataset['masks'][last_start : i].copy()
                     
            
                     L = len(obs_slice)
                     if(self.traj_length is not None):
                           index = L - self.traj_length
                           if(index < 0):
                                index = 0
                     else:
                            index =  0
                
                     
                     if len(act_slice) < 10:
                          last_start = i + 1
                          continue

                     """
                     if(self.mode == 'reward'):
                        if(sum(rews) == 0):
                            last_start = i + 1
                            continue 
                     """
                         
                     trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           #"rewards":  reward_processor_2(rews[index:].copy())
                           "rewards":  rews[index:],
                           "masks": masks[index:]
                     }
                         
                     trajectories.append(trajectory)
                     last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class SceneDataset:
    def __init__(self, name: str):
        self.name = name
        name_to_id = {
            "play": "scene-play-v0",
            "noisy": "scene-noisy-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
            self.dataset_id, render_mode="rgb_array"
        )

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                trajectories.append({
                    "observations": obs_slice,
                    "actions": act_slice,
                })
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class SceneDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None, mode: Optional[str] = None):
        self.name = name
        self.traj_length = traj_length
        self.mode = mode
        name_to_id = {
            "play": f"scene-play-singletask-task{task_id}-v0",
            'noisy': f"scene-noisy-singletask-task{task_id}-v0"
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
            self.dataset_id, render_mode="rgb_array"
        )

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        rewards = reward_processor(self.dataset["rewards"].copy(), "scene")
        for i in range(N):
            # End of a natural episode (terminal or success)
            #if self.dataset["terminals"][i] == 1 or self.dataset["rewards"][i] == 0:
            if self.dataset["terminals"][i] == 1 :
                obs_slice = self.dataset["observations"][last_start:i+1].copy()
                act_slice = self.dataset["actions"][last_start:i].copy()
                rews = rewards[last_start : i].copy()
                masks = self.dataset['masks'][last_start : i].copy()

                L = len(obs_slice)
                if self.traj_length is not None:
                    index = L - self.traj_length
                    if index < 0:
                        index = 0
                else:
                    index = 0
                
               
                if len(act_slice) < 10:
                    last_start = i + 1
                    continue
                

                trajectory = {
                    "observations": obs_slice[index:],
                    "actions": act_slice[index:],
                    "rewards": rews[index:],
                    "masks": masks[index:]
                }
                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class PuzzleDataset:
    def __init__(self, name: str):
        self.name = name
        name_to_id = {
            "3x3-play": "puzzle-3x3-play-v0",
            "3x3-noisy": "puzzle-3x3-noisy-v0",
            "4x4-play": "puzzle-4x4-play-v0",
            "4x4-noisy": "puzzle-4x4-noisy-v0",
            "4x5-play": "puzzle-4x5-play-v0",
            "4x5-noisy": "puzzle-4x5-noisy-v0",
            "4x6-play": "puzzle-4x6-play-v0",
            "4x6-noisy": "puzzle-4x6-noisy-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
            self.dataset_id, render_mode="rgb_array"
        )

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                trajectories.append({
                    "observations": obs_slice,
                    "actions": act_slice,
                })
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env

class PuzzleDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None, mode: Optional[str] = None):
        self.name = name
        self.traj_length = traj_length
        self.mode = mode
        name_to_id = {
            "3x3-play": f"puzzle-3x3-play-singletask-task{task_id}-v0",
            "3x3-noisy": f"puzzle-3x3-noisy-singletask-task{task_id}-v0",
            "4x4-play": f"puzzle-4x4-play-singletask-task{task_id}-v0",
            "4x4-noisy": f"puzzle-4x4-noisy-singletask-task{task_id}-v0",
            "4x5-play": f"puzzle-4x5-play-singletask-task{task_id}-v0",
            "4x5-noisy": f"puzzle-4x5-noisy-singletask-task{task_id}-v0",
            "4x6-play": f"puzzle-4x6-play-singletask-task{task_id}-v0",
            "4x6-noisy": f"puzzle-4x6-noisy-singletask-task{task_id}-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
            self.dataset_id, render_mode="rgb_array"
        )

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])
        rewards = reward_processor(self.dataset["rewards"].copy(), "puzzle")
        for i in range(N):
            # End of a natural episode (terminal or success)
            #if self.dataset["terminals"][i] == 1 or self.dataset["rewards"][i] == 0:
            if self.dataset["terminals"][i] == 1:
                obs_slice = self.dataset["observations"][last_start:i+1].copy()
                act_slice = self.dataset["actions"][last_start:i].copy()
                rews = rewards[last_start : i].copy()
                masks = self.dataset['masks'][last_start : i].copy()

                L = len(obs_slice)
                if self.traj_length is not None:
                    index = L - self.traj_length
                    if index < 0:
                        index = 0
                else:
                    index = 0
                
               
                if len(act_slice) < 10:
                    last_start = i + 1
                    continue
                

                trajectory = {
                    "observations": obs_slice[index:],
                    "actions": act_slice[index:],
                    "rewards": rews[index:],
                    "masks": masks[index:]
                }
                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env




#-------------------------------------------------------------------------------------#
#---------------------------------- Planner Dataset ----------------------------------#
#-------------------------------------------------------------------------------------#
def _get_planner_base(env_name, specific_env):
    if env_name == 'kitchen':
        if specific_env == 'complete':
            return 'Kitchen_High'
        elif specific_env == 'partial':
            return 'Kitchen_Medium'
        elif specific_env == 'mixed':
            return 'Kitchen_Mixed'
        else:
            raise ValueError(f"Invalid specific environment: {specific_env}")

    elif env_name == 'pointmaze':
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
    
    elif env_name == 'ogpointmaze':
        if specific_env == 'medium':
            return 'OG2DMaze_Medium'
        elif specific_env == 'large':
            return 'OG2DMaze_Large'
        elif specific_env == 'giant':
            return 'OG2DMaze_Giant'
        else:
            raise ValueError(f"Invalid dataset name: {specific_env}")

    elif env_name == 'antmaze':
        if specific_env == 'medium':
            return 'AntMaze_Medium'
        elif specific_env == 'large':
            return 'AntMaze_Large'
        elif specific_env == 'giant':
            return 'AntMaze_Giant'
        else:
            raise ValueError(f"Invalid Dataset name: {specific_env}")
    
    elif env_name == 'humanoidmaze':
        if specific_env == 'medium':
            return 'HumanoidMaze_Medium'
        elif specific_env == 'large':
            return 'HumanoidMaze_Large'
        elif specific_env == 'giant':
            return 'HumanoidMaze_Giant'
        else:
            raise ValueError(f"Invalid Dataset name: {specific_env}")

    elif env_name == 'cube':
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
            raise ValueError(f"Invalid dataset name: {specific_env}")

    elif env_name == 'scene':
        if specific_env == 'play':
            return 'Scene_Play'
        elif specific_env == 'noisy':
            return 'Scene_Noisy'
        else:
            raise ValueError(f"Invalid dataset name: {specific_env}")
    
    elif env_name == 'puzzle':
        if specific_env == '3x3-play':
            return 'Puzzle_3x3Play'
        elif specific_env == '3x3-noisy':
            return 'Puzzle_3x3Noisy'
        elif specific_env == '4x4-play':
            return 'Puzzle_4x4Play'
        elif specific_env == '4x4-noisy':
            return 'Puzzle_4x4Noisy'
        elif specific_env == '4x5-play':
            return 'Puzzle_4x5Play'
        elif specific_env == '4x5-noisy':
            return 'Puzzle_4x5Noisy'
        elif specific_env == '4x6-play':
            return 'Puzzle_4x6Play'
        elif specific_env == '4x6-noisy':
            return 'Puzzle_4x6Noisy'
        else:
            raise ValueError(f"Invalid dataset name: {specific_env}")

    else:
        raise ValueError(f"Invalid environment name: {env_name}")

def get_PlannerName(env_name, specific_env, task_id=None):
    """Returns the planner *stem* (no step, no extension).

    With task_id=None (default), the result matches the pre-existing
    behavior exactly, e.g. 'Cube_SinglePlay_Planner'.

    With task_id given, '_task{id}' is inserted before '_Planner', e.g.
    'Cube_SinglePlay_task4_Planner'.
    """
    base = _get_planner_base(env_name, specific_env)
    tid  = f"_task{task_id}" if task_id is not None else ""
    return f"{base}{tid}_Planner"

class PlannerDataset(Dataset):
    def __init__(self, dataset_name, specific_dataset, task_id, horizon, state_dim, action_dim, stride: Optional[int] = 1):
        data = get_dataset(dataset_name, specific_dataset, task_id)
        self.planner_name = get_PlannerName(dataset_name, specific_dataset, task_id)
        self.traj = data.get_trajectories()
        self.horizon = horizon
        self.windows = []
        self.conditions = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        """
        if(determine_stride(dataset_name, specific_dataset)):
           self.stride = stride
        else:
           self.stride = 1
        """
        self.stride = stride
        
        

        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        for traj in self.traj:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))
            obs_list.append(obs[:L])
            act_list.append(acts[:L])
        obs_all = np.concatenate(obs_list, axis = 0, dtype = np.float32)  # [N, d_s]

        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)
       

        # ----- build normalized sliding windows -----
        for traj in self.traj:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))     
            # per-step normalize then concat [s_t, a_t]
            sa_pairs = []
            if(self.stride == 1):
                for t in range(L):
                    s_norm = self.stats.norm_obs(obs[t])
                    #a_norm = self.stats.norm_act(acts[t])
                    a_norm = acts[t]
                    sa_pairs.append(np.concatenate([s_norm, a_norm], axis=0))
            else:
                 for t in range(L):
                    s_norm = self.stats.norm_obs(obs[t])
                    sa_pairs.append(s_norm)

            if(self.stride == 1):
              # sliding horizon, then flatten to 1D
              for start in range(0, L - horizon + 1):
                  segment = np.array(sa_pairs[start : start + horizon])  # [H, d_s+d_a]
                  self.windows.append(torch.from_numpy(segment).float())
                  self.conditions.append(torch.from_numpy(sa_pairs[start][:self.state_dim]).float())
            else:
                max_start = L - ((horizon - 1) * self.stride)
                if max_start <= 0:
                    continue
                for start in range(0, max_start):
                    idxs = start + (self.stride * np.arange(horizon))
                    segment = np.array([sa_pairs[i] for i in idxs])  # [H, d_s]
                    self.windows.append(torch.from_numpy(segment).float())
                    self.conditions.append(torch.from_numpy(sa_pairs[start]).float())
                
        
        self.save_stats(dataset_name, specific_dataset)
    
    """
    def save_stats(self, dataset_name, specific_dataset):
     
        stats_name =  str(self.planner_name) + '_stats.pkl'
        stats_dir = f'./Pretrain/Planners/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")
    """

    def save_stats(self, dataset_name, specific_dataset):
       stats_dir = PRETRAIN_DIR / "Planners" / dataset_name / specific_dataset / "Stats"
       stats_dir.mkdir(parents=True, exist_ok=True)
       stats_name = str(self.planner_name) + "_stats.pkl"
       savepath = stats_dir / stats_name
       with open(savepath, "wb") as f:
          pickle.dump(self.stats, f)
       print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.conditions[idx]
"""
class PlannerDataset_debug(Dataset):
    def __init__(self, dataset_name, specific_dataset, horizon, index):
        data = get_dataset(dataset_name, specific_dataset)
        self.planner_name = get_PlannerName(dataset_name, specific_dataset)
        self.traj = data.get_trajectories()
        self.horizon = horizon
        self.windows = []


        # ----- gather raw obs/actions to fit stats -----
        obs_list, act_list = [], []
        obs, acts = self.traj[index]['observations'], self.traj[index]['actions']
        L = min(len(obs), len(acts))
        obs_list.append(obs[:L])
        act_list.append(acts[:L])
        
        obs_all = np.concatenate(obs_list, axis = 0)  # [N, d_s]
        act_all = np.concatenate(act_list, axis = 0)
        total = [obs_all, act_all]
        with open('total.pkl', 'wb') as f:
            pickle.dump(total, f)


        #get stats
        self.stats = SAStats()
        self.stats.obs_mean=obs_all.mean(axis=0)
        self.stats.obs_std =obs_all.std(axis=0)
        self.save_stats()

        # ----- build normalized sliding windows -----
        #for traj in self.traj:
        obs, acts = self.traj[index]['observations'], self.traj[index]['actions']
        L = min(len(obs), len(acts))     
        sa_pairs = []
        for t in range(L):
             s_norm = self.stats.norm_obs(obs[t])
             a_norm = self.stats.norm_act(acts[t])
             sa_pairs.append(np.concatenate([s_norm, a_norm], axis=0))
        
        # sliding horizon, then flatten to 1D
        for start in range(0, L - horizon + 1):
                segment = np.array(sa_pairs[start:start + horizon])  # [H, d_s+d_a]
                self.windows.append(torch.from_numpy(segment).float())
            

    def save_stats(self):
        stats_name = self.planner_name.replace('.pt', '_stats.pkl')
        with open(stats_name, 'wb') as f:
              pickle.dump(self.stats, f)

 
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
"""
class Planner_Processor():
     """
     def __init__(self, dataset_name, specific_dataset):
          Planner_name = get_PlannerName(dataset_name, specific_dataset)
          stats_name = Planner_name + '_stats.pkl'  # Remove .pt replacement since Planner_name doesn't have .pt
          stats_dir = f'./Pretrain/Planners/{dataset_name}/{specific_dataset}/Stats/'
          stats_path = os.path.join(stats_dir, stats_name)
          

          # Check if stats file exists
          if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

          with open(stats_path, 'rb') as f:
              self.stats = pickle.load(f)
     """
     def __init__(self, dataset_name, specific_dataset, task_id: Optional[int] = None):
          stats_dir = PRETRAIN_DIR / "Planners" / dataset_name / specific_dataset / "Stats"
          planner_name = get_PlannerName(dataset_name, specific_dataset, task_id)
          stats_name = str(planner_name) + "_stats.pkl"
          stats_path = stats_dir / stats_name
          if not stats_path.exists():
               raise FileNotFoundError(f"Stats file not found: {stats_path}")
          with open(stats_path, "rb") as f:
               self.stats = pickle.load(f)
    
     def preprocess(self, obs):
          obs = self.stats.norm_obs(obs)
          return obs
     
     def norm_act(self, act):
          act = self.stats.norm_act(act)
          return act
     
     def postprocess(self, act):
          act = self.stats.denorm_act(act)
          return  act

class PlannerDataset_Rollout(Dataset):
    def __init__(self, dataset_name, specific_dataset, specific_train_dataset, horizon, state_dim, action_dim):
        data = get_dataset(dataset_name, specific_dataset)
        self.traj = data.get_trajectories()
        self.horizon = horizon
        self.windows = []
        self.conditions = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planner_processor = Planner_Processor(dataset_name, specific_train_dataset)
        
        # ----- build normalized sliding windows -----
        for traj in self.traj:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))     
            # per-step normalize then concat [s_t, a_t]
            sa_pairs = []
            for t in range(L):
                s_norm = self.planner_processor.preprocess(obs[t])
                #a_norm = self.planner_processor.norm_act(acts[t])
                a_norm = acts[t]
                sa_pairs.append(np.concatenate([s_norm, a_norm], axis=0))
               
            # sliding horizon, then flatten to 1D
            for start in range(0, L - horizon + 1):
                segment = np.array(sa_pairs[start : start + horizon])  # [H, d_s+d_a]
                self.windows.append(torch.from_numpy(segment).float())
                self.conditions.append(torch.from_numpy(sa_pairs[start][:self.state_dim]).float())


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.conditions[idx]










