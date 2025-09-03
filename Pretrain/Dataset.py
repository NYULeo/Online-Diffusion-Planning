import numpy as np
import minari
from sympy.core import I
import mediapy as media
import warnings
import gymnasium as gym
warnings.filterwarnings("ignore", category=UserWarning)


def get_env(env_name, specific_env):
    data = get_dataset(env_name, specific_env)
    #env = data.get_env()
    d_s = data.get_state_dim()
    d_a = data.get_action_dim()
    #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = 'rgb_array')
    return  d_s, d_a


def get_dataset(name: str, specific_name: str):
       if(name == 'kitchen'):
             if specific_name == 'partial':
                  return KitchenDataset('partial')
             elif specific_name == 'complete':
                  return KitchenDataset('complete')
             elif specific_name == 'mixed':
                  return KitchenDataset('mixed')
             else:
                  raise ValueError(f"Invalid Dataset name: {specific_name}")
       elif(name == 'PointMaze'):
            if specific_name == 'large':
                  return PointMazeDataset('large')
            elif specific_name== 'medium':
                  return PointMazeDataset('medium')
            elif specific_name == 'umaze':
                  return PointMazeDataset('unmaze')
            else:
                  raise ValueError(f"Invalid Dataset name: {specific_name}")
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
              
              trajectory = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards
              }
           
              trajectories.append(trajectory)
          
          return trajectories
     
     def get_state_dim(self):
          return self.dataset._observation_space['observation'].shape[0]
    
     def get_action_dim(self):
          return self.dataset._action_space.shape[0]
    
     def get_env(self):
          return self.dataset.recover_environment(render_mode = 'rgb_array')

     def get_total_steps(self):
          return self.dataset.total_steps
     
class PointMazeDataset():
     def __init__(self, name: str):
          if name == 'large':
              self.dataset = minari.load_dataset('D4RL/pointmaze/large-v2', download = True)
          elif name == 'medium':
              self.dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download = True)
          elif name == 'umaze':
              self.dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download = True)
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
              
              trajectory = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards
              }
           
              trajectories.append(trajectory)
          
          return trajectories
     
     def get_state_dim(self):
          return self.dataset._observation_space['observation'].shape[0]
    
     def get_action_dim(self):
          return self.dataset._action_space.shape[0]
    
     def get_env(self):
          return self.dataset.recover_environment(render_mode = 'rgb_array')

     def get_total_steps(self):
          return self.dataset.total_steps
     

