import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import minari



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
          return self.dataset.recover_environment()

     def get_total_steps(self):
          return self.dataset.total_steps
     


