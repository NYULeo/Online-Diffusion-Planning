#from pstats import StatsProfile
import sys
import os

from torch.distributed import batch_isend_irecv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import numpy as np
import torch
from loguru import logger as log
import minari
from scipy.ndimage import gaussian_filter1d,  gaussian_filter

from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any
import torch.nn as nn
from Dataset import get_dataset, get_dataset
import gymnasium as gym
import torch
import math
from utils import set_seed
from Dataset import get_env, get_dataset
import gymnasium_robotics
import mediapy as media
from collections import namedtuple

from Planners.Backbone.utils import get_pretrained_planner
from torch.utils.data import DataLoader
from Dataset import PlannerDataset
from Rewards.nets import gaussian_rewards
import scipy
import scipy.ndimage
from sympy import factorint
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from Planners.Backbone.Dit import DiT1d
from Planners.Backbone.utils import compute_dot_alpha_beta
from Planners.Backbone.Sampler import sample_reverse_sde
from Dataset import Planner_Processor
import torch.nn.functional as F
from Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
from Dataset import get_dataset
from Rewards.nets import Reward
from Finetuning.traj_reward import TotalReward
from Rewards.Reward_Backbone import Train_Dataset, RewardDataset
import random
from Critic.train_critic import get_critic_model, get_critic_stats
from Critic.nets import Critic
try:
    import matplotlib
    #matplotlib.use('Agg')  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be skipped.")
    MATPLOTLIB_AVAILABLE = False
    plt = None



def plot_function(func, x_range=(-10, 10), num_points=1000, title="Function Plot", xlabel="x", ylabel="f(x)"):
    
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))


"""
from Rewards.Reward_Backbone import RewardDataset, Train_Dataset
trajs, name, obs_dim, act_dim = Train_Dataset('pointmaze', 'medium')
dataset = RewardDataset(trajs, 7.0, name, 1.0)
cords = dataset.transitions
coordinates = []
for cord in cords:
    coordinates.append(cord[0])
coordinates = np.array(coordinates)
coordinates = coordinates[:, :2]
plt.figure(figsize=(10, 8))
plt.hexbin(coordinates[:, 0], coordinates[:, 1], gridsize=50, cmap='viridis', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Hexbin Heatmap of Coordinates')
plt.show()
"""








"""
save_path = f'./Rollouts/{'pointmaze'}/{'medium'}/Generated_trajs_Info.pkl'
with open(save_path, 'rb') as f:
    data = pickle.load(f)
gen_trajs = data['trajs']


data_complete = get_dataset('pointmaze', 'medium')
trajs_complete = data_complete.get_trajectories()


reward_model_state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', 44000, 'medium')
reward_model = Reward(obs_dim, act_dim)
reward_model.load_state_dict(reward_model_state_dict)
reward_model.eval()
stats = get_pretrained_reward_stats(name)




total = 0.0
for i in range(len(gen_trajs)):
     traj = gen_trajs[i]
     traj_reward = 0.0
     Grad_sum = 0.0
     for j in range(len(traj['actions'])):
          obs = traj['observations'][j].copy()
          action = traj['actions'][j].copy()
          obs_norm = stats.norm_obs(obs)
          action_norm = action
          obs_norm = torch.tensor(obs_norm, dtype = torch.float32, requires_grad = True).unsqueeze(0)
          action_norm = torch.tensor(action_norm, dtype = torch.float32, requires_grad = True).unsqueeze(0)
          pred =   (100000/1024) *reward_model(obs_norm, action_norm)
          grad = torch.autograd.grad(
                 outputs=pred,
                 inputs=(obs_norm, action_norm),
                 grad_outputs=torch.ones_like(pred),
                 create_graph=False,
                 retain_graph=False)
          grad_obs = grad[0].squeeze(0)
          grad_action = grad[1].squeeze(0)
          Grad_sum += grad_obs.norm().item() + grad_action.norm().item()
          traj_reward += pred.item()
     print(f"Grad_sum: {Grad_sum / len(traj['actions'])}")
     traj_reward = traj_reward / len(traj['actions'])
     #print(f"Traj {i} reward: {traj_reward}")
     total += traj_reward
     
total = total / len(gen_trajs)
print(f"Complete Total reward: {total}")





import seaborn as sns
import matplotlib.colors as mcolors
import colorsys

def _lighten_color(color: str, amount: float=0.5) -> str:
        
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        rgb = mcolors.to_rgb(c)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # l is lightness, we increase it toward 1
        new_l = 1 - amount * (1 - l)
        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
        return mcolors.to_hex(new_rgb)


def smooth_curve(data: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        padded = np.full_like(data, np.nan)
        padded[window-1:] = smoothed
        return padded

def plot_reward_curve(
                          title: str = "Finetuning Reward Curve",
                          show_lr: bool = False,
                          smooth_window: int = 50):
        

        sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor':'black'})
        plt.rcParams.update({'font.size': 14})

        okabe_ito = ["#D55E00","#000000", "#E69F00", "#56B4E9", "#009E73",
                       "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FF0000"]
        raw_color    = okabe_ito[3]   
        smooth_color = okabe_ito[4] 
        lr_color     = okabe_ito[9]  # yellow (for learning rate curve)

        fig, ax1 = plt.subplots(figsize=(12, 8))
        steps = np.arange(100)
        rewards = np.random.randn(len(steps))


         # Plot smoothed if possible
        if len(rewards) > smooth_window and smooth_window > 1:
            smoothed = smooth_curve(rewards, smooth_window)
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
        plt.show()











        
        if show_lr and self.learning_rates:
            ax2 = ax1.twinx()
            lr_vals = np.array(self.learning_rates)
            ax2.plot(steps[:len(lr_vals)], lr_vals, color='green', alpha=0.7, linewidth=1.5, label='Learning Rate')
            ax2.set_ylabel('Learning Rate', fontsize=12, color=lr_color)
            ax2.tick_params(axis='y', labelcolor=lr_color)
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
        

from Dataset import get_dataset
with open('./Pretrain/Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
     trajs_info = pickle.load(f)
trajs = trajs_info['trajs']
rewards = []
for traj in trajs:
      Temp = np.array(traj['rewards'])
      R = Temp.sum()
      rewards.append(R)
rewards = np.array(rewards)
#print(rewards)


"""






"""
import minari
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.step_data import StepData
from minari.storage.local import delete_dataset
import gymnasium as gym
import numpy as np


with open('./Pretrain/Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
     trajs_info = pickle.load(f)
trajs = trajs_info['trajs']



data = get_dataset('pointmaze', 'medium')
env = data.get_env(render_mode = None)
episodes = []
dat = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
ref_min = dat.storage.metadata.get('ref_min_score')
ref_max = dat.storage.metadata.get('ref_max_score')

for traj_idx, traj in enumerate(trajs):
    obs_seq = np.asarray(traj['observations'])
    act_seq = np.asarray(traj['actions'])
    rew_seq = np.asarray(traj['rewards'])

    # Assume episode ends at last step; adjust if you know real truncations.
    terminated_flags = np.zeros(len(rew_seq), dtype=bool)
    truncated_flags = np.zeros(len(rew_seq), dtype=bool)
    terminated_flags[-1] = True

    buffer = EpisodeBuffer()
    for t in range(len(rew_seq)):
        step = StepData(
               observation=obs_seq[t],
               action=act_seq[t],
               reward=float(rew_seq[t]),
               terminated=bool(terminated_flags[t]),
               truncated=bool(truncated_flags[t]),
               info={}
        )
        buffer = buffer.add_step_data(step)
    episodes.append(buffer)




dataset = minari.create_dataset_from_buffers(
    dataset_id = gen_dataset_id(None, 'my-rollout', 0),
    buffer = episodes,
    env = env,
    ref_min_score = ref_min,
    ref_max_score = ref_max,
)

raw_returns = np.array([np.sum(traj['rewards']) for traj in trajs])        # undiscounted
# OR discounted (what 99% of papers report):
gamma = 0.99
disc_returns = np.array([
    sum(r * (gamma ** i) for i, r in enumerate(traj['rewards']))
    for traj in trajs
])

print(raw_returns)
normalized_scores = minari.get_normalized_score(dataset, disc_returns.mean())
print(normalized_scores)
delete_dataset('my-rollout-v0')

print(ref_min, ref_max)
"""




"""
import minari
import numpy as np
import pickle
from Dataset import get_dataset


def get_normalized_score(trajs, env_name, specific_env):
    # 2. Get official references from Minari
    data = get_dataset(env_name, specific_env)
    ref_min = data.get_ref_min_score()
    ref_max = data.get_ref_max_score()
   
    # 3. Count how many goals your agent reaches on average
    avg_goals = np.mean([np.sum(traj['rewards']) for traj in trajs])
    print(f"Average goals reached per episode: {avg_goals:.2f}")

    # 4. Convert to correct discounted return (4000-step episodes)
    avg_discounted_return = avg_goals * 66.8   # This is the only magic number you need

    # 5. Compute normalized score
    normalized_score = 100 * (avg_discounted_return - ref_min) / (ref_max - ref_min) 

    # Final result
    print(f"Normalized score (pointmaze/medium-v2): {normalized_score:.2f}")
"""



import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

import numpy as np
import torch
import matplotlib.pyplot as plt
import minari
import gymnasium_robotics
import gymnasium as gym
from typing import Optional

def plot_critic_heatmap(
    dataset_name: str = 'pointmaze',
    specific_dataset: str = 'medium',
    step: int = 1000,
    resolution: int = 256,
    batch_size: int = 16384,
    max_goals_to_plot: int = 20,
    grid_margin: float = 0.5,
    output_dir: Optional[str] = None,
    goal: Optional[np.ndarray] = None
):
    """
    Plot a heatmap of critic values for the pointmaze environment.
    
    Args:
        dataset_name: Name of the dataset ('pointmaze')
        specific_dataset: Specific dataset variant ('medium', 'large', 'umaze')
        step: Checkpoint step to load
        resolution: Grid resolution for the heatmap (default: 256)
        batch_size: Batch size for processing (default: 16384)
        max_goals_to_plot: Maximum number of goals to plot (default: 20)
        grid_margin: Extra padding around observed positions (default: 0.5)
        output_dir: Directory to save the heatmap (default: './reward_map')
        goal: Optional specific goal to plot. If None, extracts goals from dataset
    """
    from Pretrain.Critic.train_critic import get_critic_model, get_critic_stats
    from Pretrain.Critic.nets import Critic
    
    output_file = f"critic_{step}_heatmap.png"
    if output_dir is None:
        output_dir = os.path.join(project_root, "reward_map")
    
    print(f'Plotting the critic heatmap for checkpoint: {step}')
    
    # ================== Load Environment ==================
    if specific_dataset == 'medium':
        dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
    elif specific_dataset == 'large':
        dataset = minari.load_dataset('D4RL/pointmaze/large-v2', download=True)
    elif specific_dataset == 'umaze':
        dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download=True)
    else:
        raise ValueError(f"Unsupported specific_dataset: {specific_dataset}")
    
    env = dataset.recover_environment().unwrapped
    
    # ================== Extract Goals ==================
    if goal is not None:
        # Use provided goal
        GOALS = np.array([goal]) if goal.ndim == 1 else goal
        if GOALS.ndim == 1:
            GOALS = GOALS.reshape(1, -1)
    else:
        # Extract goals from dataset
        all_goals = set()
        pos_min = np.array([np.inf, np.inf], dtype=np.float32)
        pos_max = np.array([-np.inf, -np.inf], dtype=np.float32)
        
        t = 0
        while t < 20:
            obs, info = env.reset(seed=t)
            goal_extracted = env.generate_target_goal()
            
            if isinstance(goal_extracted, np.ndarray):
                goal_2d = goal_extracted[:2] if len(goal_extracted) >= 2 else goal_extracted
                goal_rounded = tuple(np.round(goal_2d, 2))
            else:
                goal_rounded = tuple(np.round(goal_extracted[:2], 2))
            
            all_goals.add(goal_rounded)
            
            # Update position bounds
            obs_dict, info = env.reset(seed=t)
            obs = obs_dict['observation']
            if len(obs) >= 2:
                positions = obs[:2]
                pos_min = np.minimum(pos_min, positions)
                pos_max = np.maximum(pos_max, positions)
            
            t += 1
        
        if len(all_goals) == 0:
            raise ValueError("No goals found in dataset!")
        
        goal_list = sorted(all_goals)
        if len(goal_list) > max_goals_to_plot:
            print(f"Found {len(goal_list)} unique goals, limiting to first {max_goals_to_plot}.")
            goal_list = goal_list[:max_goals_to_plot]
        
        GOALS = np.array(goal_list)
        
        # Determine plotting bounds from observations
        if np.isinf(pos_min).any() or np.isinf(pos_max).any():
            if hasattr(env, 'maze') and hasattr(env.maze, 'maze_map'):
                map_height = env.maze.map_length
                map_width = env.maze.map_width
                cell = env.maze.maze_size_scaling
                pos_min = np.array([-map_width / 2.0 * cell, -map_height / 2.0 * cell], dtype=np.float32)
                pos_max = np.array([map_width / 2.0 * cell, map_height / 2.0 * cell], dtype=np.float32)
            else:
                raise ValueError("Could not determine position bounds from dataset or maze.")
    
    # Set grid bounds
    if goal is not None:
        # If single goal provided, use default bounds or extract from env
        if hasattr(env, 'maze') and hasattr(env.maze, 'maze_map'):
            map_height = env.maze.map_length
            map_width = env.maze.map_width
            cell = env.maze.maze_size_scaling
            pos_min = np.array([-map_width / 2.0 * cell, -map_height / 2.0 * cell], dtype=np.float32)
            pos_max = np.array([map_width / 2.0 * cell, map_height / 2.0 * cell], dtype=np.float32)
        else:
            pos_min = np.array([-4.0, -4.0], dtype=np.float32)
            pos_max = np.array([4.0, 4.0], dtype=np.float32)
    
    grid_min = (pos_min - grid_margin).astype(np.float32)
    grid_max = (pos_max + grid_margin).astype(np.float32)
    
    default_start = np.array([-1.5, -0.5], dtype=np.float32)
    start_pos = default_start
    
    # ================== Load Critic Model ==================
    print(f"Loading critic model (step {step})...")
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, step)
    stats = get_critic_stats(dataset_name, specific_dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = Critic(obs_dim).to(device)
    critic.load_state_dict(model_state_dict)
    critic.eval()
    
    print(f"Critic loaded. Observation dimension: {obs_dim}")
    
    # ================== Create Grid ==================
    print(f"Creating {resolution}x{resolution} grid...")
    x = np.linspace(grid_min[0], grid_max[0], resolution)
    y = np.linspace(grid_min[1], grid_max[1], resolution)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # ================== Evaluate Critic for All Goals ==================
    print("Evaluating critic for all goals...")
    value_maps_per_goal = []
    
    for goal_idx, goal_pos in enumerate(GOALS):
        print(f"Processing goal {goal_idx+1}/{len(GOALS)}: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]")
        
        # Create observations: [x, y, goal_x, goal_y]
        obs_base = np.stack([
            X.ravel(),
            Y.ravel(),
            np.full(resolution**2, goal_pos[0]),
            np.full(resolution**2, goal_pos[1])
        ], axis=1).astype(np.float32)
        
        value_map_goal = np.zeros(resolution**2, dtype=np.float32)
        
        with torch.no_grad():
            for start in range(0, len(obs_base), batch_size):
                end = min(start + batch_size, len(obs_base))
                batch_obs = obs_base[start:end]
                
                # Normalize observations
                obs_norm = stats.norm_obs(batch_obs)
                obs_tensor = torch.from_numpy(obs_norm).float().to(device)
                
                # Compute critic values
                values = critic(obs_tensor).cpu().numpy()
                value_map_goal[start:end] = values
        
        value_maps_per_goal.append(value_map_goal.reshape(resolution, resolution))
    
    # Aggregate value maps (take maximum across all goals for each position)
    value_map = np.stack(value_maps_per_goal, axis=0).max(axis=0)
    
    # ================== Plot Heatmap ==================
    print("Creating heatmap...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot heatmap
    im = ax.imshow(
        value_map,
        extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
        origin='lower',
        cmap='RdYlBu_r',
        interpolation='bilinear'
    )
    plt.colorbar(im, ax=ax, label='Critic Value (V)')
    
    # Draw walls
    print("Drawing maze walls...")
    if hasattr(env.maze, 'walls'):
        for wall in env.maze.walls:
            (x0, y0), (x1, y1) = wall
            ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3, zorder=10)
    else:
        # Fallback: use maze_map if walls attribute doesn't exist
        maze_map = env.maze.maze_map
        map_height = env.maze.map_length
        map_width = env.maze.map_width
        cell_size = env.maze.maze_size_scaling
        
        for row in range(map_height):
            for col in range(map_width):
                if maze_map[row][col] == 1:  # This is a wall cell
                    try:
                        if hasattr(env.maze, 'cell_rowcol_to_xy'):
                            cell_center = env.maze.cell_rowcol_to_xy(row, col)
                            x_center, y_center = float(cell_center[0]), float(cell_center[1])
                        else:
                            raise AttributeError("Method not available")
                    except:
                        x_center = (col - map_width / 2.0 + 0.5) * cell_size
                        y_center = (map_height / 2.0 - row - 0.5) * cell_size
                    
                    half_cell = cell_size / 2.0
                    corners = [
                        [x_center - half_cell, y_center - half_cell],
                        [x_center + half_cell, y_center - half_cell],
                        [x_center + half_cell, y_center + half_cell],
                        [x_center - half_cell, y_center + half_cell],
                        [x_center - half_cell, y_center - half_cell]
                    ]
                    corners = np.array(corners)
                    ax.plot(corners[:, 0], corners[:, 1], 'k-', linewidth=2, zorder=10)
    
    # Mark start position
    ax.plot(
        start_pos[0],
        start_pos[1],
        'go',
        markersize=15,
        label='Start',
        markeredgecolor='black',
        markeredgewidth=2,
        zorder=20
    )
    
    # Mark all goal positions
    print(f"Plotting {len(GOALS)} goals...")
    for i, goal_pos in enumerate(GOALS):
        ax.plot(goal_pos[0], goal_pos[1], 'y*', markersize=20, 
                markeredgecolor='black', markeredgewidth=1, zorder=20)
        ax.text(goal_pos[0] + 0.3, goal_pos[1] + 0.3, f'G{i+1}', 
                fontsize=10, color='black', weight='bold', zorder=21,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('X position', fontsize=12)
    ax.set_ylabel('Y position', fontsize=12)
    ax.set_title(f'Critic Value Heatmap (Step {step}) - All {len(GOALS)} Goals', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_file)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Critic heatmap saved to {save_path}")
    
    plt.close()
    
    print(f"Final value_map stats: min={value_map.min():.4f}, max={value_map.max():.4f}, mean={value_map.mean():.4f}")
    
    return value_map, save_path


if __name__ == '__main__':
    # Example usage
    step = 200
    while(step <= 1000):
         np.random.seed(0)
         random.seed(0)
         plot_critic_heatmap(
             dataset_name='pointmaze',
             specific_dataset='medium',
             step=1000,
             goal=np.array([[-2.5, -2.5]], dtype=float)  # Optional: specific goal
         )
         step += 200
    print('Done')





