from optparse import Option
from pathlib import Path

from sympy import true
from sympy.vector.coordsysrect import Str
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "Pretrain"
import numpy as np
import minari
from sympy.core import I
#import mediapy as media
import warnings
import gymnasium as gym
import gymnasium_robotics
import ogbench
warnings.filterwarnings("ignore", category=UserWarning)
from collections import namedtuple
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
from itertools import permutations

class TrajectoryDict(TypedDict):
    observations: np.ndarray
    actions: np.ndarray  
    rewards: np.ndarray


def determine_stride(dataset_name, specific_dataset):
     if(dataset_name == 'antmaze'):
          return True
     else:
          return False

#-------------------------------------------------------------------------------------#
#------------------------------------- Dataset ---------------------------------------#
#-------------------------------------------------------------------------------------#
def get_env(env_name, specific_env, render_mode = None, task_id: Optional[int] = None, goal: Optional[np.array] = None, episode_length: Optional[int] = None):
    data = get_dataset(env_name, specific_env, task_id, goal, episode_length)
    env = data.get_env(render_mode)
    d_s = data.get_state_dim()
    d_a = data.get_action_dim()
    if(env_name == 'antmaze'):
        d_s = d_s + 2
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
    
    if(name in ('cube', 'ogpointmaze', 'antmaze', 'humanoidmaze', 'puzzle', 'scene')):
         return ogbench_reward_processor(rewards)
    else:
         return spare_reward_processor(rewards)
    
def get_dataset(name: str, specific_name: str, task_id: Optional[int] = None, goal: Optional[np.array] = None, traj_length: Optional[int] = None, mode: Optional[str] = None):
       if(name == 'kitchen'):
            return KitchenDataset(specific_name)
       elif(name == 'pointmaze'):
            return PointMazeDataset(specific_name, goal, mode)
       elif(name == 'antmaze'): 
            return AntMazeDataset(specific_name)
       elif(name == 'ogpointmaze'):
             if(task_id is None):
                 return OGPointmazeDataset(specific_name)
             else:
                 return OGPointmazeDataset_Singletask(specific_name, task_id, traj_length, mode)
       elif(name == 'cube'):
            if(task_id is None):
                return CubeDataset(specific_name)
            else:
                return CubeDataset_Singletask(specific_name, task_id, traj_length)
       
       else:
            raise ValueError(f"Invalid Dataset name: {name}")     

class KitchenDataset():
     def __init__(self, name: str):
          if name == 'partial':
              self.dataset = minari.load_dataset('D4RL/kitchen/partial-v2',  download=True)
          elif name == 'complete':
              self.dataset = minari.load_dataset('D4RL/kitchen/complete-v2',  download=True)
          elif name == 'mixed':
              self.dataset = minari.load_dataset('D4RL/kitchen/mixed-v2',  download=True) 
          else:
              raise ValueError(f"Invalid Dataset name: {name}")
          
     def get_trajectories(self):
          trajectories = []
          for episode in self.dataset.iterate_episodes():
              observations = episode.observations['observation']
              actions = episode.actions
              rewards = episode.rewards
              terminated = episode.terminations
              truncated = episode.truncations
              done_seq = np.logical_or(terminated, truncated)
              
              for i in range(len(actions)):
                   if(done_seq[i]):
                        observations = observations[:i+2]
                        actions = actions[:i+1]
                        rewards = rewards[:i+1]
                        break
              
              
              if(len(actions) < 10):
                  continue
              else: 
                 new_rewards = self.spare_reward_kitchen(rewards)
                 if(not self.reward_checker(rewards, new_rewards)):
                       print('No')
                 trajectory = {
                      'observations': observations,
                      'actions': actions,
                      'rewards': new_rewards
                    }
                 """
                 if(len(trajectories) != 0):
                      Temp = merger(trajectories[len(trajectories)-1], trajectory)
                      if(Temp is not None):
                            trajectories.pop()
                            trajectories.append(Temp)
                      else:
                            trajectories.append(trajectory)
                 else:
                      trajectories.append(trajectory)
                 """
                 trajectories.append(trajectory)

          return trajectories  
     
     def reward_checker(self, rewards, new_rewards):
         if(len(rewards) != len(new_rewards)):
               return False
         for i in range(1, len(rewards)):
              if(rewards[i] == rewards[i-1]+1):
                  if(new_rewards[i] !=1):
                      return False
              else:
                  if(new_rewards[i] != 0):
                      return False
         return True

     def spare_reward_kitchen(self, rewards):
         Temp = []
         for i in range(1, len(rewards)):
            if(rewards[i] == rewards[i-1]+1):
                  Temp.append(i)
         new_rewards = [0]*len(rewards)
         for i in range(len(rewards)):
             if(i in Temp):
                  new_rewards[i] = 1
             else:
                  new_rewards[i] = 0
         return np.array(new_rewards, dtype = np.float64) 
     
     def get_state_dim(self):
          return self.dataset._observation_space['observation'].shape[0]
    
     def get_action_dim(self):
          return self.dataset._action_space.shape[0]
    
     def get_env(self, render_mode):
          # Use headless mode for servers without display capabilities
          return self.dataset.recover_environment(render_mode = render_mode)
          #env_spec = self.dataset.spec.env_spec
          #return gym.make(env_spec, render_mode='rgb_array')
          #return gym.make(env_spec, render_mode = None)
     
     def get_ref_max_score(self):
          return self.dataset.storage.metadata.get('ref_max_score')

     def get_ref_min_score(self):
          return self.dataset.storage.metadata.get('ref_min_score')
     
     def get_total_steps(self):
          return self.dataset.total_steps
     
class PointMazeDataset():
     def __init__(self, name: str, goal: Optional[np.array] = None, mode: Optional[str] = None):
          self.name = name
          if(mode is not None):
              self.mode = mode
          else:
              self.mode = 'reward'
          if(goal is not None):
               self.goal = goal
          else:
               self.goal = None
          if name == 'open_dense':
               self.dataset = minari.load_dataset('D4RL/pointmaze/open-dense-v2', download = True)
          elif name == 'umaze':
               self.dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download = True)
          elif name == 'large_dense':
               self.dataset = minari.load_dataset('D4RL/pointmaze/large-dense-v2', download = True)
          elif name == 'medium':
               self.dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download = True)
          elif name == 'medium_dense':
               self.dataset = minari.load_dataset('D4RL/pointmaze/medium-dense-v2', download = True)
          elif name == 'umaze_dense':
               self.dataset = minari.load_dataset('D4RL/pointmaze/umaze-dense-v2', download = True)
          elif name == 'large':
               self.dataset = minari.load_dataset('D4RL/pointmaze/large-v2', download = True)
          elif name == 'open':
               self.dataset = minari.load_dataset('D4RL/pointmaze/open-v2', download = True)
          else:
              raise ValueError(f"Invalid Dataset name: {name}")
          
     def get_trajectories(self):
          trajectories = []
          for episode in self.dataset.iterate_episodes():
              observations = episode.observations['observation']
              actions = episode.actions
              rewards = episode.rewards
              terminated = episode.terminations
              truncated = episode.truncations
              done_seq = np.logical_or(terminated, truncated)
              
              for i in range(len(actions)):
                   if(done_seq[i]):
                        observations = observations[:i+2]
                        actions = actions[:i+1]
                        rewards = rewards[:i+1]
                        break
              
              if(len(actions) < 10):
                    continue
              else: 
                    trajectory = {
                        'observations': observations,
                        'actions': actions,
                        'rewards': rewards
                      }
                    
                    if(self.mode != 'critic'):
                       if(len(trajectories) != 0):
                           Temp = merger(trajectories[len(trajectories)-1], trajectory)
                           if(Temp is not None):
                               trajectories.pop()
                               trajectories.append(Temp)
                           else:
                               trajectories.append(trajectory)
                       else:
                           trajectories.append(trajectory)
                    else:
                        trajectories.append(trajectory)
          
          if (self.goal is not None):
               if(self.mode == 'critic'):
                   trajectories = self.reward_filter_goals(trajectories, self.goal)
               elif(self.mode == 'reward'):
                   trajectories = self.reward_filter(trajectories, self.goal)
               else:
                   raise ValueError(f"Invalid mode: {self.mode}")
          return trajectories
     
     def get_state_dim(self):
          return self.dataset._observation_space['observation'].shape[0]
    
     def get_action_dim(self):
          return self.dataset._action_space.shape[0]
    
     def reward_filter_goals(self, trajs: List[TrajectoryDict], goal) -> List[TrajectoryDict]:
        def reward_filter2(traj: TrajectoryDict, goal) -> List[TrajectoryDict]:
          last_step = 1
          new_trajs = []
          new_rews = [0]*len(traj['rewards'])
          traj['rewards'] = new_rews
        
          for i in range(1, len(traj['observations'])):
              pos = traj['observations'][i][:2]
              g = np.asarray(goal, dtype=np.float32).reshape(-1)
              dist = np.linalg.norm(pos - g) 
              if(dist < 0.5):
                if((i - last_step) < 10):
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
     
     def reward_filter(self, trajs: List[TrajectoryDict], goal) -> List[TrajectoryDict]:
          new_trajs = []
          for traj in trajs:
               new_rews = [0]*len(traj['rewards'])
               traj['rewards'] = new_rews
               for i in range(1, len(traj['observations'])):
                     pos = traj['observations'][i][:2] 
                     g = np.asarray(goal, dtype=np.float32).reshape(-1)
                     #goal_coord = np.asarray(goal_coord, dtype=np.float32).reshape(-1)  
                     dist = np.linalg.norm(pos - g) 
                     if (dist < 0.5):
                           traj['rewards'][i-1] = 1
                     else:
                           traj['rewards'][i-1] = 0
               new_trajs.append(traj)
          return new_trajs

     def get_env(self, render_mode):
          
          gym.register_envs(gymnasium_robotics)

          if(self.name == 'medium'):
              env = gym.make('PointMaze_Medium-v3', max_episode_steps = 600, render_mode = render_mode, continuing_task=False)
          elif(self.name == 'large'):
              env = gym.make('PointMaze_Large-v3', max_episode_steps = 800, render_mode = render_mode, continuing_task=False)
          elif(self.name == 'umaze'):
              env = gym.make('PointMaze_Umaze-v3', max_episode_steps = 600, render_mode = render_mode, continuing_task=False)
          else:
              raise ValueError(f'Invalid dataset name')
          return env
          
        
          
          #return self.dataset.recover_environment(render_mode = 'rgb_array', continuing_task=True, reset_target=False, eval_env=True)

     def get_ref_max_score(self):
          return self.dataset.storage.metadata.get('ref_max_score')

     def get_ref_min_score(self):
          return self.dataset.storage.metadata.get('ref_min_score')

     def get_total_steps(self):
          return self.dataset.total_steps

class AntMazeDataset():
     def __init__(self, name: str):
          self.name = name
          if name == 'medium_play':
              self.dataset = minari.load_dataset('D4RL/antmaze/medium-play-v1', download=True)
          elif name == 'umaze_diverse':
              self.dataset = minari.load_dataset('D4RL/antmaze/umaze-diverse-v1', download=True)
          elif name == 'large_diverse':
              self.dataset = minari.load_dataset('D4RL/antmaze/large-diverse-v1', download=True)
          elif name == 'large_play':
              self.dataset = minari.load_dataset('D4RL/antmaze/large-play-v1', download=True)
          elif name == 'medium_diverse':
              self.dataset = minari.load_dataset('D4RL/antmaze/medium-diverse-v1', download=True)
          elif name == 'umaze':
              self.dataset = minari.load_dataset('D4RL/antmaze/umaze-v1', download=True)
          else:
              raise ValueError(f"Invalid Dataset name: {name}")
          
     def get_trajectories(self):
          trajectories = []
          for episode in self.dataset.iterate_episodes():
              observations = episode.observations['observation']
              positions = episode.observations['achieved_goal']
              observations = np.concatenate([positions, observations], axis = 1)
              actions = episode.actions
              rewards = episode.rewards
              terminated = episode.terminations
              truncated = episode.truncations
              done_seq = np.logical_or(terminated, truncated)
              
              for i in range(len(actions)):
                   if(done_seq[i]):
                        observations = observations[:i+2]
                        actions = actions[:i+1]
                        rewards = rewards[:i+1]
                        break
              
              if(len(actions) < 10):
                  continue
              else: 
                 trajectory = {
                      'observations': observations,
                      'actions': actions,
                      'rewards': rewards
                    }
                 if(len(trajectories) != 0):
                      Temp = merger(trajectories[len(trajectories)-1], trajectory)
                      if(Temp is not None):
                            trajectories.pop()
                            trajectories.append(Temp)
                      else:
                            trajectories.append(trajectory)
                 else:
                      trajectories.append(trajectory)

          return trajectories
     
     def get_state_dim(self):
          return self.dataset._observation_space['observation'].shape[0]
    
     def get_action_dim(self):
          return self.dataset._action_space.shape[0]
    
     def get_env(self, render_mode):
          
          gym.register_envs(gymnasium_robotics)
          if self.name in ['umaze', 'umaze_diverse']:
              env = gym.make('AntMaze_UMaze-v4', max_episode_steps=1000, render_mode=render_mode, continuing_task=False)
          elif self.name in ['medium_play', 'medium_diverse']:
              env = gym.make('AntMaze_Medium-v4', max_episode_steps=1000, render_mode=render_mode, continuing_task=False)
          elif self.name in ['large_play', 'large_diverse']:
              env = gym.make('AntMaze_Large-v4', max_episode_steps=1000, render_mode=render_mode, continuing_task=False)
          else:
              raise ValueError(f'Invalid dataset name')
          return env
          
          #return self.dataset.recover_environment(render_mode = render_mode)
         
     def get_ref_max_score(self):
          return self.dataset.storage.metadata.get('ref_max_score')

     def get_ref_min_score(self):
          return self.dataset.storage.metadata.get('ref_min_score')

     def get_total_steps(self):
          return self.dataset.total_steps

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

"""
class CubeDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None):
        
        self.name = name
        self.traj_length = traj_length
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
                
                     
                     if len(act_slice) < 10:
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
                         
                         trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           'rewards': rews[index:]
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
"""

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
        if(mode is not None):
              self.mode = mode
        else:
              self.mode = 'reward'
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

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode = render_mode, max_episode_steps = 2000)
        return env

class CubeDataset_Singletask:
    def __init__(self, name: str, task_id, traj_length: Optional[int] = None):
        
        self.name = name
        self.traj_length = traj_length
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
        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            if self.dataset['terminals'][i] == 1 or self.dataset['rewards'][i] == 0:
                     obs_slice = self.dataset["observations"][last_start : i].copy()
                     act_slice = self.dataset["actions"][last_start : i].copy()
                     #rews = sparse_reward_processor(self.dataset['rewards'][last_start+1: i+1].copy())
                     rews = rewards[last_start+1: i+1].copy()
                     
            
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
                     
                     if(sum(rews) == 0):
                          last_start = i + 1
                          continue 
                    
                         
                     trajectory = {
                           "observations": obs_slice[index:],
                           "actions": act_slice[index:],
                           "rewards": rews[index:]
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

"""
def get_PlannerName(env_name, specific_env):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High_Planner'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium_Planner'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed_Planner'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
          if specific_env == 'open_dense':
               return 'PointMaze_OpenDense_Planner'
          elif specific_env == 'umaze':
               return 'PointMaze_Umaze_Planner'
          elif specific_env == 'large_dense':
               return 'PointMaze_LargeDense_Planner'
          elif specific_env== 'medium':
               return 'PointMaze_Medium_Planner'
          elif specific_env == 'umaze_dense':
               return 'PointMaze_UmazeDense_Planner'
          elif specific_env == 'large':
               return 'PointMaze_Large_Planner'
          elif specific_env == 'open':
               return 'PointMaze_Open_Planner'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'antmaze'):
          if specific_env == 'medium_play':
               return 'AntMaze_MediumPlay_Planner'
          elif specific_env == 'umaze_diverse':
               return 'AntMaze_UmazeDiverse_Planner'
          elif specific_env == 'large_diverse':
               return 'AntMaze_LargeDiverse_Planner'
          elif specific_env == 'large_play':
               return 'AntMaze_LargePlay_Planner'
          elif specific_env == 'medium_diverse':
               return 'AntMaze_MediumDiverse_Planner'
          elif specific_env == 'umaze':
               return 'AntMaze_Umaze_Planner'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")

     elif(env_name == 'cube'):
         if specific_env == 'single-play':
              return 'Cube_SinglePlay_Planner'
         elif specific_env == 'single-noisy':
              return 'Cube_SingleNoisy_Planner'
         elif specific_env == 'double-play':
              return 'Cube_DoublePlay_Planner'
         elif specific_env == 'double-noisy':
              return 'Cube_DoubleNoisy_Planner'
         elif specific_env == 'triple-play':
              return 'Cube_TriplePlay_Planner'
         elif specific_env == 'triple-noisy':
              return 'Cube_TripleNoisy_Planner'
         elif specific_env == 'quadruple-play':
              return 'Cube_QuadruplePlay_Planner'
         elif specific_env == 'quadruple-noisy':
              return 'Cube_QuadrupleNoisy_Planner'
         else:
              raise ValueError(f"Invalid cube dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
         if specific_env == 'medium':
              return 'OG2DMaze_Medium_Planner'
         elif specific_env == 'large':
              return 'OG2DMaze_Large_Planner'
         elif specific_env == 'giant':
              return 'OG2DMaze_Giant_Planner'
         else:
              raise ValueError(f"Invalid ogpointmaze dataset name: {specific_env}")

     else:
         raise ValueError(f"Invalid environment name: {env_name}")
"""


def _get_planner_base(env_name, specific_env):
    """Return the env-specific stem, without '_Planner' and without task_id.

    e.g. ('cube', 'single-play') -> 'Cube_SinglePlay'
    """
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

    elif env_name == 'antmaze':
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
            raise ValueError(f"Invalid cube dataset name: {specific_env}")

    elif env_name == 'ogpointmaze':
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
        if(determine_stride(dataset_name, specific_dataset)):
           self.stride = stride
        else:
           self.stride = 1
        
        

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




#env, d_s, d_a = get_env('antmaze', 'medium_play')
#data = PlannerDataset('antmaze', 'medium_play', 40, d_s, d_a)
"""
vectors = []
data = get_dataset('kitchen', 'partial')
trajs = data.get_trajectories()
for traj in trajs:
     for i in range(len(traj['rewards'])):
         if(traj['rewards'][i] == 1):
             vectors.append(traj['observations'][i])
vectors = np.array(vectors)



import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def verify_four_clusters(vectors):
   
    vectors = np.array(vectors)
    
    # Fit K-means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(vectors)
    
    # Compute distances to assigned cluster centers
    centers = kmeans.cluster_centers_
    distances = cdist(vectors, centers, metric='euclidean')
    min_distances = distances[np.arange(len(vectors)), assignments]
    
    # Inertia (within-cluster sum of squares)
    inertia = kmeans.inertia_
    
    stats = {
        'inertia': inertia,
        'mean_distance_to_center': np.mean(min_distances),
        'max_distance_to_center': np.max(min_distances),
        'cluster_counts': np.bincount(assignments, minlength=4),
        'cluster_percentages': np.bincount(assignments, minlength=4) / len(vectors) * 100,
        'cluster_centers': centers
    }
    
    return kmeans, assignments, centers, stats


# Usage:
kmeans, assignments, centers, stats = verify_four_clusters(vectors)


import sklearn
print(f"Cluster distribution: {stats['cluster_counts']}")
print(f"Cluster percentages: {stats['cluster_percentages']}")
print(f"Inertia (within-cluster sum of squares): {stats['inertia']:.4f}")
print(f"Mean distance to center: {stats['mean_distance_to_center']:.4f}")
print(f"Max distance to center: {stats['max_distance_to_center']:.4f}")
print(f"\nCluster centers:\n{stats['cluster_centers']}")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def visualize_clusters(vectors, assignments, cluster_centers, title_prefix=""):
   
    vectors = np.array(vectors)
    n_clusters = len(cluster_centers)
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_clusters))
    
    # 1. 2D PCA plot
    pca_2d = PCA(n_components=2)
    vectors_2d = pca_2d.fit_transform(vectors)
    centers_2d = pca_2d.transform(cluster_centers)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 2D scatter
    ax = axes[0, 0]
    for i in range(n_clusters):
        mask = assignments == i
        ax.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                  c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
        ax.scatter(centers_2d[i, 0], centers_2d[i, 1], 
                  c=[colors[i]], marker='x', s=200, linewidths=3, 
                  edgecolors='black')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    ax.set_title(f'{title_prefix}2D PCA Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance histogram
    ax = axes[0, 1]
    from scipy.spatial.distance import cdist
    distances = cdist(vectors, cluster_centers, metric='euclidean')
    min_distances = distances[np.arange(len(vectors)), assignments]
    ax.hist(min_distances, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(min_distances), color='red', linestyle='--', 
               label=f'Mean: {np.mean(min_distances):.4f}')
    ax.set_xlabel('Distance to Cluster Center')
    ax.set_ylabel('Frequency')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot by cluster
    ax = axes[1, 0]
    cluster_distances = [min_distances[assignments == i] for i in range(n_clusters)]
    bp = ax.boxplot(cluster_distances, labels=[f'C{i}' for i in range(n_clusters)])
    ax.set_ylabel('Distance to Center')
    ax.set_title('Distance by Cluster')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Distance matrix
    ax = axes[1, 1]
    from scipy.spatial.distance import pdist, squareform
    center_distances = squareform(pdist(cluster_centers, metric='euclidean'))
    im = ax.imshow(center_distances, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Distance')
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax.set_yticklabels([f'C{i}' for i in range(n_clusters)])
    ax.set_title('Inter-Cluster Distances')
    for i in range(n_clusters):
        for j in range(n_clusters):
            text = ax.text(j, i, f'{center_distances[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if center_distances[i, j] > center_distances.max()/2 else "black")
    
    plt.tight_layout()
    plt.show()

# Usage:
visualize_clusters(vectors, assignments, stats['cluster_centers'], "Kitchen Rewards: ")

"""

"""Count training windows for horizon 32 vs 70 on pointmaze large."""

