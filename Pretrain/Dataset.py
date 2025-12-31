import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import minari
from sympy.core import I
#import mediapy as media
import warnings
import gymnasium as gym
import gymnasium_robotics
warnings.filterwarnings("ignore", category=UserWarning)
from collections import namedtuple
import torch
from torch.utils.data import Dataset
import pickle
from Pretrain.utils import SAStats
import os




#-------------------------------------------------------------------------------------#
#------------------------------------- Dataset ---------------------------------------#
#-------------------------------------------------------------------------------------#
def get_env(env_name, specific_env, render_mode = None):
    data = get_dataset(env_name, specific_env)
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

def get_dataset(name: str, specific_name: str):
       if(name == 'kitchen'):
            return KitchenDataset(specific_name)
       elif(name == 'pointmaze'):
            return PointMazeDataset(specific_name)
       elif(name == 'antmaze'): 
            return AntMazeDataset(specific_name)
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
     def __init__(self, name: str):
          self.name = name
          if name == 'open_dense':
               self.dataset = minari.load_dataset('D4RL/pointmaze/open-dense-v2', download = True)
          elif name == 'umaze':
               self.dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download = True)
          elif name == 'large_dense':
               self.dataset = minari.load_dataset('D4RL/pointmaze/large-dense-v2', download = True)
          elif name == 'medium':
               self.dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download = True)
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
              

              """
              for i in range(len(actions)):
                   if(done_seq[i]):
                        observations = observations[:i+2]
                        actions = actions[:i+1]
                        rewards = rewards[:i+1]
                        break
              """ 
              
              if(len(actions) < 10):
                  continue
              else: 
                    trajectory = {
                        'observations': observations,
                        'actions': actions,
                        'rewards': rewards
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
          
     
     def get_state_dim(self):
          return self.dataset._observation_space['observation'].shape[0]
    
     def get_action_dim(self):
          return self.dataset._action_space.shape[0]
    
     def get_env(self, render_mode):
          # Use headless mode for servers without display capabilities
          
          gym.register_envs(gymnasium_robotics)
          if(self.name == 'medium'):
              env = gym.make('PointMaze_Medium-v3', max_episode_steps = 1000, render_mode = render_mode, continuing_task=False)
          elif(self.name == 'large'):
              env = gym.make('PointMaze_Large-v3', max_episode_steps = 1000, render_mode = render_mode, continuing_task=False)
          elif(self.name == 'umaze'):
              env = gym.make('PointMaze_Umaze-v3', max_episode_steps = 1000, render_mode = render_mode, continuing_task=False)
          else:
              raise ValueError(f'Invalid dataset name')
          return env
          
          #return self.dataset.recover_environment(render_mode = render_mode)
          #return self.dataset.recover_environment(render_mode = 'rgb_array')
     
     def get_ref_max_score(self):
          return self.dataset.storage.metadata.get('ref_max_score')

     def get_ref_min_score(self):
          return self.dataset.storage.metadata.get('ref_min_score')

     def get_total_steps(self):
          return self.dataset.total_steps

class AntMazeDataset():
     def __init__(self, name: str):
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
          # Use headless mode for servers without display capabilities
          return self.dataset.recover_environment(render_mode = render_mode)
          #return self.dataset.recover_environment(render_mode = 'rgb_array')
          #env_spec = self.dataset.spec.env_spec
          #return gym.make(env_spec, render_mode='rgb_array')
          #return gym.make(env_spec, render_mode = None)
          
     def get_ref_max_score(self):
          return self.dataset.storage.metadata.get('ref_max_score')

     def get_ref_min_score(self):
          return self.dataset.storage.metadata.get('ref_min_score')

     def get_total_steps(self):
          return self.dataset.total_steps



#-------------------------------------------------------------------------------------#
#---------------------------------- Planner Dataset ----------------------------------#
#-------------------------------------------------------------------------------------#
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
     else:
         raise ValueError(f"Invalid environment name: '{env_name}")

class PlannerDataset(Dataset):
    def __init__(self, dataset_name, specific_dataset, horizon, state_dim, action_dim):
        data = get_dataset(dataset_name, specific_dataset)
        self.planner_name = get_PlannerName(dataset_name, specific_dataset)
        self.traj = data.get_trajectories()
        self.horizon = horizon
        self.windows = []
        self.conditions = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        

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
        self.stats.obs_mean=obs_all.mean(axis=0)
        self.stats.obs_std =obs_all.std(axis=0)
       

        # ----- build normalized sliding windows -----
        for traj in self.traj:
            obs, acts = traj['observations'], traj['actions']
            L = min(len(obs), len(acts))     
            # per-step normalize then concat [s_t, a_t]
            sa_pairs = []
            for t in range(L):
                s_norm = self.stats.norm_obs(obs[t])
                #a_norm = self.stats.norm_act(acts[t])
                a_norm = acts[t]
                sa_pairs.append(np.concatenate([s_norm, a_norm], axis=0))
               
            
            # sliding horizon, then flatten to 1D
            for start in range(0, L - horizon + 1):
                segment = np.array(sa_pairs[start : start + horizon])  # [H, d_s+d_a]
                self.windows.append(torch.from_numpy(segment).float())
                self.conditions.append(torch.from_numpy(sa_pairs[start][:self.state_dim]).float())
            
        self.save_stats()

    def save_stats(self):
        stats_name =  str(self.planner_name) + '_stats.pkl'
        stats_dir = f'.Planners/{self.dataset_name}/{self.specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
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


vectors = []
data = get_dataset('kitchen', 'complete')
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



print(f"Cluster distribution: {stats['cluster_counts']}")
print(f"Cluster percentages: {stats['cluster_percentages']}")
print(f"Inertia (within-cluster sum of squares): {stats['inertia']:.4f}")
print(f"Mean distance to center: {stats['mean_distance_to_center']:.4f}")
print(f"Max distance to center: {stats['max_distance_to_center']:.4f}")
print(f"\nCluster centers:\n{stats['cluster_centers']}")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
def visualize_clusters(vectors, assignments, cluster_centers, title_prefix=""):
    """Comprehensive clustering visualization"""
    vectors = np.array(vectors)
    n_clusters = len(cluster_centers)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
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