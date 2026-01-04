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


def plot_critic_heatmap(STEP, agg_method='max', highlight_negatives=True):
    """
    Plot a heatmap of critic values for the pointmaze environment.
    Similar structure to the reward heatmap function.
    
    Args:
        STEP: Checkpoint step to load
        agg_method: 'max', 'min', or 'mean' for aggregating across goals
        highlight_negatives: Whether to show a negative zoom panel
    """
    from Pretrain.Critic.train_critic import get_critic_model, get_critic_stats
    from Pretrain.Critic.nets import Critic
    
    # ================== Configuration ==================
    RESOLUTION = 256                # Grid resolution
    BATCH_SIZE = 16384              # Batch size for efficient processing
    MAX_GOALS_TO_PLOT = 20           # Plot only the first few unique goals
    GRID_MARGIN = 0.5               # Extra padding around observed positions
    OUTPUT_FILE = f"critic_{STEP}_heatmap.png"

    print(f'Plotting the critic heatmap for checkpoint: {STEP}')
    
    # ================== Load Environment ==================
    dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
    env = dataset.recover_environment().unwrapped

    # ================== Extract All Unique Goals from Dataset ==================
    all_goals = set()
    pos_min = np.array([np.inf, np.inf], dtype=np.float32)
    pos_max = np.array([-np.inf, -np.inf], dtype=np.float32)
    first_start = None

    # Fixed goal extraction loop
    t = 0
    while t < 20:
        obs, info = env.reset(seed=t)
        goal = env.generate_target_goal()
        
        # Convert to tuple and round to avoid floating point precision issues
        if isinstance(goal, np.ndarray):
            goal_2d = goal[:2] if len(goal) >= 2 else goal
            goal_rounded = tuple(np.round(goal_2d, 2))
        else:
            goal_rounded = tuple(np.round(goal[:2], 2))
        
        all_goals.add(goal_rounded)
        
        # Update position bounds from observation
        obs_dict, info = env.reset(seed=t)
        obs = obs_dict['observation']  # extract the numpy array
        if len(obs) >= 2:
            positions = obs[:2]  # Current position
            pos_min = np.minimum(pos_min, positions)
            pos_max = np.maximum(pos_max, positions)
            if first_start is None:
                first_start = positions
        
        t += 1
    
    # Convert to numpy array (limit to a few goals for clarity)
    if len(all_goals) == 0:
        raise ValueError("No goals found in dataset! Check dataset structure.")

    goal_list = sorted(all_goals)
    if len(goal_list) > MAX_GOALS_TO_PLOT:
        print(f"Found {len(goal_list)} unique goals, limiting to first {MAX_GOALS_TO_PLOT}.")
        goal_list = goal_list[:MAX_GOALS_TO_PLOT]

    GOALS = np.array(goal_list)
    print(f"Found {len(GOALS)} unique goals.")

    # Determine plotting bounds based on observed positions
    if np.isinf(pos_min).any() or np.isinf(pos_max).any():
        if hasattr(env, 'maze') and hasattr(env.maze, 'maze_map'):
            map_height = env.maze.map_length
            map_width = env.maze.map_width
            cell = env.maze.maze_size_scaling
            pos_min = np.array([-map_width / 2.0 * cell, -map_height / 2.0 * cell], dtype=np.float32)
            pos_max = np.array([map_width / 2.0 * cell, map_height / 2.0 * cell], dtype=np.float32)
        else:
            raise ValueError("Could not determine position bounds from dataset or maze.")

    # Expand bounds a bit so the heatmap includes some context outside trajectories
    grid_min = (pos_min - GRID_MARGIN).astype(np.float32)
    grid_max = (pos_max + GRID_MARGIN).astype(np.float32)
    
    # Determine start position
    default_start = np.array([-1.5, -0.5], dtype=np.float32)
    start_pos = default_start

    # ================== Load Critic Model ==================
    print(f"Loading critic model (step {STEP})...")
    model_state_dict, obs_dim = get_critic_model('pointmaze', 'medium', STEP)
    stats = get_critic_stats('pointmaze', 'medium')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = Critic(obs_dim).to(device)
    critic.load_state_dict(model_state_dict)
    critic.eval()
    
    print(f"Critic loaded. Observation dimension: {obs_dim}")

    # ================== Create Grid ==================
    x = np.linspace(grid_min[0], grid_max[0], RESOLUTION)
    y = np.linspace(grid_min[1], grid_max[1], RESOLUTION)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # ================== Evaluate Critic for All Goals ==================
    value_maps_per_goal = []
    gradnorm_maps_per_goal = []

    # Prepare normalization tensors
    obs_mean_t = torch.as_tensor(stats.obs_mean, dtype=torch.float32)
    obs_std_t = torch.as_tensor(np.maximum(stats.obs_std, getattr(stats, "std_floor", 1e-3)), dtype=torch.float32)

    for goal_idx, goal in enumerate(GOALS):
        print(f"Processing goal {goal_idx+1}/{len(GOALS)}: [{goal[0]:.2f}, {goal[1]:.2f}]")
    
        # Create observations: [x, y, goal_x, goal_y]
        obs_base = np.stack([
            X.ravel(),
            Y.ravel(),
            np.full(RESOLUTION**2, goal[0]),
            np.full(RESOLUTION**2, goal[1])
        ], axis=1).astype(np.float32)
    
        value_map_goal = np.zeros(RESOLUTION**2, dtype=np.float32)
        gradnorm_map_goal = np.full(RESOLUTION**2, np.nan, dtype=np.float32)

        for start in range(0, len(obs_base), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(obs_base))

            batch_obs = torch.from_numpy(obs_base[start:end]).float()
            batch_obs.requires_grad_(True)  # gradients w.r.t. [x,y,goal_x,goal_y]

            # Differentiable normalization
            obs_norm = (batch_obs - obs_mean_t) / obs_std_t

            # Compute critic values
            values = critic(obs_norm.to(device))  # [B]
            value_map_goal[start:end] = values.detach().cpu().numpy()

            # Compute gradient norm w.r.t. position (x, y) only
            grads = torch.autograd.grad(values.sum(), batch_obs, create_graph=False)[0]  # [B,4]
            gradnorm = torch.norm(grads[:, :2], dim=1)  # only ∇ w.r.t (x,y)
            gradnorm_map_goal[start:end] = gradnorm.detach().cpu().numpy()
        
        value_maps_per_goal.append(value_map_goal.reshape(RESOLUTION, RESOLUTION))
        gradnorm_maps_per_goal.append(gradnorm_map_goal.reshape(RESOLUTION, RESOLUTION))

    # Aggregate value maps (customizable: 'max', 'min', or 'mean')
    print(f"Aggregating with method: {agg_method}")
    if agg_method == 'max':
        value_map = np.stack(value_maps_per_goal, axis=0).max(axis=0)
        gradnorm_map = np.stack(gradnorm_maps_per_goal, axis=0).max(axis=0)
    elif agg_method == 'min':
        value_map = np.stack(value_maps_per_goal, axis=0).min(axis=0)
        gradnorm_map = np.stack(gradnorm_maps_per_goal, axis=0).min(axis=0)
    elif agg_method == 'mean':
        value_map = np.stack(value_maps_per_goal, axis=0).mean(axis=0)
        gradnorm_map = np.stack(gradnorm_maps_per_goal, axis=0).mean(axis=0)
    else:
        raise ValueError("agg_method must be 'max', 'min', or 'mean'")

    # Debug print (enhanced)
    neg_count = (value_map < 0).sum()
    neg_pct = 100 * neg_count / value_map.size
    print(f"Final value_map stats: min={value_map.min():.4f}, max={value_map.max():.4f}, mean={value_map.mean():.4f}")
    
    # Value stats over the entire grid
    vm_finite = np.isfinite(value_map)
    vm_mean = value_map[vm_finite].mean()
    vm_var = value_map[vm_finite].var()
    vm_std = value_map[vm_finite].std()

    print(f"Final value_map stats: min={value_map[vm_finite].min():.4f}, "
          f"max={value_map[vm_finite].max():.4f}, mean={vm_mean:.4f}, "
          f"var={vm_var:.6f}, std={vm_std:.6f}, N={vm_finite.sum()}")

    # Grad-norm stats
    gn_finite = np.isfinite(gradnorm_map)
    gn_mean = gradnorm_map[gn_finite].mean()
    gn_var = gradnorm_map[gn_finite].var()
    gn_std = gradnorm_map[gn_finite].std()
    print(f"Final gradnorm_map stats: min={gradnorm_map[gn_finite].min():.6f}, "
          f"max={gradnorm_map[gn_finite].max():.6f}, mean={gn_mean:.6f}, "
          f"var={gn_var:.6f}, std={gn_std:.6f}, N={gn_finite.sum()}")

    print(f"Negative positions: {neg_count} / {value_map.size} ({neg_pct:.1f}%)")
    if neg_pct < 1.0:
        print("Warning: Negatives are sparse (<1%). Try agg_method='mean' or 'min' to reveal more.")

    # Find and print top 5 most negative positions (world coords)
    neg_mask = value_map < 0
    if neg_mask.any():
        neg_indices = np.argwhere(neg_mask)
        sorted_neg = neg_indices[np.argsort(value_map[neg_mask])[::-1]]  # Most negative first
        print("Top 5 most negative positions (x, y, value):")
        for i in range(min(5, len(sorted_neg))):
            row, col = sorted_neg[i]
            print(f"  {i+1}: ({x[col]:.2f}, {y[row]:.2f}) = {value_map[row, col]:.4f}")
    else:
        print("No negative positions found—check model outputs.")

    # ================== Plot Heatmap ==================
    if MATPLOTLIB_AVAILABLE:
        print("Creating heatmap...")
        
        if highlight_negatives and neg_mask.any():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            axes = [ax1, ax2, ax3]
            titles = [
                f'Full Range (Step {STEP}) - Critic Value ({agg_method} agg)',
                f'Negative Zoom (Step {STEP}) - Critic Value',
                f'Grad-Norm (Step {STEP}) - ||∇_(x,y) V|| ({agg_method} agg)',
            ]
        else:
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 10))
            axes = [ax1, ax3]
            titles = [
                f'Critic Value Heatmap (Step {STEP}) - Value ({agg_method} agg)',
                f'Grad-Norm Heatmap (Step {STEP}) - ||∇_(x,y) V|| ({agg_method} agg)',
            ]

        data_min, data_max = value_map.min(), value_map.max()
        finite = np.isfinite(gradnorm_map)
        grad_vmax = np.percentile(gradnorm_map[finite], 99) if finite.any() else 1.0

        for idx, ax in enumerate(axes):
            is_grad_panel = (idx == len(axes) - 1)  # last axis is gradnorm

            if is_grad_panel:
                data = gradnorm_map
                vmin, vmax = 0.0, grad_vmax
                cmap = "viridis"
                cbar_label = r"||∇_{x,y} V||"
            else:
                data = value_map
                cbar_label = "Critic Value (V)"
                if idx == 0:  # full value
                    vmin, vmax = data_min, data_max
                    cmap = "coolwarm"
                else:  # negative zoom value
                    zoom_min, zoom_max = -0.1, 0.2
                    vmin = max(data_min, zoom_min)
                    vmax = min(0.0, zoom_max) if data_max > 0 else data_max
                    cmap = "Blues_r"

            im = ax.imshow(
                data,
                extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
                origin="lower",
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                interpolation="bilinear",
            )
            plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

            # Only overlay value=0 contour on value panels (not gradnorm)
            if (not is_grad_panel) and neg_mask.any():
                ax.contour(X, Y, value_map, levels=[0.0], colors="red",
                          linestyles="--", linewidths=2, alpha=0.7)
            
            # Draw walls using env.maze.walls (more accurate than maze_map)
            print("Drawing maze walls...")
            if hasattr(env.maze, 'walls'):
                for wall in env.maze.walls:
                    (x0, y0), (x1, y1) = wall
                    ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3, zorder=10)
            else:
                # Fallback: use maze_map if walls attribute doesn't exist
                print("  Using maze_map fallback...")
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
                zorder=20)

            # Mark all goal positions
            print(f"Plotting {len(GOALS)} goals...")
            for i, goal in enumerate(GOALS):
                ax.plot(goal[0], goal[1], 'y*', markersize=20 if idx==0 else 10, 
                        markeredgecolor='black', markeredgewidth=1, zorder=20)
                # Add goal number label (only on full plot)
                if idx == 0:
                    ax.text(goal[0] + 0.3, goal[1] + 0.3, f'G{i+1}', 
                            fontsize=10, color='black', weight='bold', zorder=21,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            ax.set_xlabel('X position', fontsize=12)
            ax.set_ylabel('Y position', fontsize=12)
            ax.set_title(titles[idx], fontsize=14, fontweight='bold')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(grid_min[0], grid_max[0])
            ax.set_ylim(grid_min[1], grid_max[1])

        plt.tight_layout()
        
        # Ensure we save to the project root directory
        reward_dir = os.path.join(project_root, "reward_map")
        os.makedirs(reward_dir, exist_ok=True)
        save_path = os.path.join(reward_dir, OUTPUT_FILE) if not os.path.isabs(OUTPUT_FILE) else OUTPUT_FILE
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Critic heatmap saved to {save_path}")
        
        # Close figure to free memory
        plt.close()
        
    else:
        print("Skipping plotting due to matplotlib import error.")
        print(f"Value map statistics:")
        print(f"  Min: {value_map.min():.4f}, Max: {value_map.max():.4f}, Mean: {value_map.mean():.4f}")
        print(f"  Shape: {value_map.shape}")
        # Save raw data as numpy array instead
        npy_path = os.path.join(project_root, OUTPUT_FILE.replace('.png', '.npy'))
        np.save(npy_path, value_map)
        print(f"Value map saved as numpy array to {npy_path}")


def heatmap(STEP, agg_method='max', highlight_negatives=True):
    # ================== Configuration ==================
    RESOLUTION = 256                # Grid resolution (256x256 is fast and looks good)
    BATCH_SIZE = 16384              # Batch size for efficient processing
    MAX_GOALS_TO_PLOT = 20           # Plot only the first few unique goals
    GRID_MARGIN = 0.5               # Extra padding around observed positions for plotting
    OUTPUT_FILE = f"{STEP}_heatmap.png"

    print(f'Ploting the heatmap for checkpoint: {STEP}')
    
    # ================== Load Environment ==================
    dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
    env = dataset.recover_environment().unwrapped  # Unwrap to access maze attribute

    # ================== Extract All Unique Goals from Dataset ==================
    all_goals = set()
    pos_min = np.array([np.inf, np.inf], dtype=np.float32)
    pos_max = np.array([-np.inf, -np.inf], dtype=np.float32)
    first_start = None

    # Fixed goal extraction loop
    t = 0
    while t < 20:
        obs, info = env.reset(seed=t)
        goal = env.generate_target_goal()
        
        # Convert to tuple and round to avoid floating point precision issues
        if isinstance(goal, np.ndarray):
            goal_2d = goal[:2] if len(goal) >= 2 else goal
            goal_rounded = tuple(np.round(goal_2d, 2))
        else:
            goal_rounded = tuple(np.round(goal[:2], 2))
        
        all_goals.add(goal_rounded)
        
        # Update position bounds from observation
        obs_dict, info = env.reset(seed=t)
        obs = obs_dict['observation']  # extract the numpy array
        if len(obs) >= 2:
            positions = obs[:2]  # Current position
            pos_min = np.minimum(pos_min, positions)
            pos_max = np.maximum(pos_max, positions)
            if first_start is None:
                first_start = positions
        
        t += 1
    
    # Convert to numpy array (limit to a few goals for clarity)
    if len(all_goals) == 0:
        raise ValueError("No goals found in dataset! Check dataset structure.")

    goal_list = sorted(all_goals)
    if len(goal_list) > MAX_GOALS_TO_PLOT:
        print(f"Found {len(goal_list)} unique goals, limiting to first {MAX_GOALS_TO_PLOT}.")
        goal_list = goal_list[:MAX_GOALS_TO_PLOT]

    GOALS = np.array(goal_list)
    print(f"Found {len(GOALS)} unique goals.")

    # Determine plotting bounds based on observed positions
    if np.isinf(pos_min).any() or np.isinf(pos_max).any():
        if hasattr(env, 'maze') and hasattr(env.maze, 'maze_map'):
            map_height = env.maze.map_length
            map_width = env.maze.map_width
            cell = env.maze.maze_size_scaling
            pos_min = np.array([-map_width / 2.0 * cell, -map_height / 2.0 * cell], dtype=np.float32)
            pos_max = np.array([map_width / 2.0 * cell, map_height / 2.0 * cell], dtype=np.float32)
        else:
            raise ValueError("Could not determine position bounds from dataset or maze.")

    # Expand bounds a bit so the heatmap includes some context outside trajectories
    grid_min = (pos_min - GRID_MARGIN).astype(np.float32)
    grid_max = (pos_max + GRID_MARGIN).astype(np.float32)
    
    # Determine start position
    default_start = np.array([-1.5, -0.5], dtype=np.float32)  # choose your constant
    start_pos = default_start

    # ================== Load Reward Model ==================

    model_state_dict, obs_dim = get_critic_model('pointmaze', 'medium', STEP)
    stats = get_critic_stats('pointmaze', 'medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = Critic(obs_dim).to(device)
    critic.load_state_dict(model_state_dict)
    critic.eval()

    # ================== Create Grid ==================
    x = np.linspace(grid_min[0], grid_max[0], RESOLUTION)
    y = np.linspace(grid_min[1], grid_max[1], RESOLUTION)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # ================== Evaluate Rewards for All Goals ==================
    # Use action candidates
    n_actions = 5  # Sample 5x5 = 25 actions
    actions = np.linspace(-1.0, 1.0, n_actions)
    action_grid = np.array([[ax, ay] for ax in actions for ay in actions]).astype(np.float32)
    acts_t = torch.from_numpy(action_grid)

    # Find the index of the zero action [0.0, 0.0] for using original reward values
    zero_action_idx = None
    for i, act in enumerate(action_grid):
        if np.allclose(act, [0.0, 0.0], atol=1e-6):
            zero_action_idx = i
            break
    
    if zero_action_idx is None:
        # Fallback: use middle action if zero doesn't exist
        zero_action_idx = len(action_grid) // 2
        print(f"Warning: Zero action not found, using action index {zero_action_idx}: {action_grid[zero_action_idx]}")

    # Store reward maps for each goal
    reward_maps_per_goal = []
    gradnorm_maps_per_goal = []

    for goal_idx, goal in enumerate(GOALS):
        print(f"Processing goal {goal_idx+1}/{len(GOALS)}: [{goal[0]:.2f}, {goal[1]:.2f}]")
    
        # Create observations: [x, y, goal_x, goal_y]
        obs_base = np.stack([
            X.ravel(),
            Y.ravel(),
            np.full(RESOLUTION**2, goal[0]),
            np.full(RESOLUTION**2, goal[1])
        ], axis=1).astype(np.float32)
    
        reward_map_goal = np.full(RESOLUTION**2, -1e10, dtype=np.float32)
       
        #Replace 
        """
        with torch.no_grad():
            for start in range(0, len(obs_base), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(obs_base))
                batch_obs = torch.from_numpy(obs_base[start:end])
            
                # Repeat actions for each position
                obs_rep = batch_obs.unsqueeze(1).repeat(1, len(acts_t), 1).reshape(-1, 4)
                act_rep = acts_t.unsqueeze(0).repeat(end-start, 1, 1).reshape(-1, 2)
            
                # Normalize
                obs_norm = stats.norm_obs(obs_rep)
                obs_norm = obs_norm.float()
                act_rep = act_rep.float()
            
                # Compute rewards
                r = model(obs_norm, act_rep).cpu().numpy().reshape(end-start, -1)
                reward_map_goal[start:end] = r[:, zero_action_idx]
        """
        obs_mean_t = torch.as_tensor(stats.obs_mean, dtype=torch.float32)
        obs_std_t = torch.as_tensor(np.maximum(stats.obs_std, getattr(stats, "std_floor", 1e-3)), dtype=torch.float32)

        gradnorm_map_goal = np.full(RESOLUTION**2, np.nan, dtype=np.float32)

        for start in range(0, len(obs_base), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(obs_base))

            batch_obs = torch.from_numpy(obs_base[start:end]).float()
            batch_obs.requires_grad_(True)  # gradients w.r.t. [x,y,goal_x,goal_y]

            act = torch.from_numpy(action_grid[zero_action_idx]).float()
            act_rep = act.unsqueeze(0).repeat(end - start, 1)  # [B,2]

            # differentiable normalization (DON’T use stats.norm_obs here)
            obs_norm = (batch_obs - obs_mean_t) / obs_std_t

            r = critic(obs_norm)  # [B]
            reward_map_goal[start:end] = r.detach().cpu().numpy()

            grads = torch.autograd.grad(r.sum(), batch_obs, create_graph=False)[0]  # [B,4]
            gradnorm = torch.norm(grads[:, :2], dim=1)  # only ∇ w.r.t (x,y)
            gradnorm_map_goal[start:end] = gradnorm.detach().cpu().numpy()
        reward_maps_per_goal.append(reward_map_goal.reshape(RESOLUTION, RESOLUTION))






        gradnorm_maps_per_goal.append(gradnorm_map_goal.reshape(RESOLUTION, RESOLUTION))

    # Aggregate reward maps (customizable: 'max', 'min', or 'mean')
    print(f"Aggregating with method: {agg_method}")
    if agg_method == 'max':
        reward_map = np.stack(reward_maps_per_goal, axis=0).max(axis=0)
    elif agg_method == 'min':
        reward_map = np.stack(reward_maps_per_goal, axis=0).min(axis=0)
    elif agg_method == 'mean':
        reward_map = np.stack(reward_maps_per_goal, axis=0).mean(axis=0)
    else:
        raise ValueError("agg_method must be 'max', 'min', or 'mean'")



    #Addition
    if agg_method == 'max':
         gradnorm_map = np.stack(gradnorm_maps_per_goal, axis=0).max(axis=0)
    elif agg_method == 'min':
         gradnorm_map = np.stack(gradnorm_maps_per_goal, axis=0).min(axis=0)
    elif agg_method == 'mean':
         gradnorm_map = np.stack(gradnorm_maps_per_goal, axis=0).mean(axis=0)











    # Debug print (enhanced)
    neg_count = (reward_map < 0).sum()
    neg_pct = 100 * neg_count / reward_map.size
    print(f"Final reward_map stats: min={reward_map.min():.4f}, max={reward_map.max():.4f}, mean={reward_map.mean():.4f}")
    
    # Reward stats over the entire grid (all positions after aggregation)
    rm_finite = np.isfinite(reward_map)
    rm_mean = reward_map[rm_finite].mean()
    rm_var  = reward_map[rm_finite].var()      # population variance (ddof=0)
    rm_std  = reward_map[rm_finite].std()

    print(f"Final reward_map stats: min={reward_map[rm_finite].min():.4f}, "
      f"max={reward_map[rm_finite].max():.4f}, mean={rm_mean:.4f}, "
      f"var={rm_var:.6f}, std={rm_std:.6f}, N={rm_finite.sum()}")

    # (Optional) grad-norm stats too
    gn_finite = np.isfinite(gradnorm_map)
    gn_mean = gradnorm_map[gn_finite].mean()
    gn_var  = gradnorm_map[gn_finite].var()
    gn_std  = gradnorm_map[gn_finite].std()
    print(f"Final gradnorm_map stats: min={gradnorm_map[gn_finite].min():.6f}, "
      f"max={gradnorm_map[gn_finite].max():.6f}, mean={gn_mean:.6f}, "
      f"var={gn_var:.6f}, std={gn_std:.6f}, N={gn_finite.sum()}")






    print(f"Negative positions: {neg_count} / {reward_map.size} ({neg_pct:.1f}%)")
    if neg_pct < 1.0:
        print("Warning: Negatives are sparse (<1%). Try agg_method='mean' or 'min' to reveal more.")

    # Find and print top 5 most negative positions (world coords)
    neg_mask = reward_map < 0
    if neg_mask.any():
        neg_indices = np.argwhere(neg_mask)
        sorted_neg = neg_indices[np.argsort(reward_map[neg_mask])[::-1]]  # Most negative first
        print("Top 5 most negative positions (x, y, reward):")
        for i in range(min(5, len(sorted_neg))):
            row, col = sorted_neg[i]
            print(f"  {i+1}: ({x[col]:.2f}, {y[row]:.2f}) = {reward_map[row, col]:.4f}")
    else:
        print("No negative positions found—check model outputs.")

    # ================== Plot Heatmap ==================
    if MATPLOTLIB_AVAILABLE:
        print("Creating heatmap...")
        
        #Replacement
        """
        if highlight_negatives and neg_mask.any():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            axes = [ax1, ax2]
            titles = [f'Full Range (Step {STEP}) - All {len(GOALS)} Goals ({agg_method} agg)',
                      f'Negative Zoom (Step {STEP}) - Low Rewards']
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
            axes = [ax]
            titles = [f'Reward Heatmap (Step {STEP}) - All {len(GOALS)} Goals ({agg_method} agg)']
        """
        if highlight_negatives and neg_mask.any():
             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
             axes = [ax1, ax2, ax3]
             titles = [
                  f'Full Range (Step {STEP}) - Reward ({agg_method} agg)',
                  f'Negative Zoom (Step {STEP}) - Reward',
                  f'Grad-Norm (Step {STEP}) - ||∇_(x,y) r|| ({agg_method} agg)',
             ]
        else:
             fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 10))
             axes = [ax1, ax3]
             titles = [
                    f'Reward Heatmap (Step {STEP}) - Reward ({agg_method} agg)',
                    f'Grad-Norm Heatmap (Step {STEP}) - ||∇_(x,y) r|| ({agg_method} agg)',
             ]


        #Replacement
        """
        data_min, data_max = reward_map.min(), reward_map.max()
        
        for idx, ax in enumerate(axes):
            # Auto-scale for full view; manual zoom for negatives
            if idx == 0:  # Full
                vmin, vmax = data_min, data_max
                cmap = 'coolwarm'
            else:  # Negative zoom
                zoom_min, zoom_max = -0.1, 0.2  # Adjust as needed based on your min
                vmin, vmax = max(data_min, zoom_min), min(0.0, zoom_max) if data_max > 0 else (data_min, data_max)
                cmap = 'Blues_r'  # Inverted blue for emphasis on negatives

            # Plot heatmap
            im = ax.imshow(
                reward_map,
                extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
                origin='lower',
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Reward', shrink=0.8)

            # Overlay negative contours (if any)
            if neg_mask.any():
                ax.contour(X, Y, reward_map, levels=[0.0], colors='red', linestyles='--', linewidths=2, alpha=0.7)
            """
        data_min, data_max = reward_map.min(), reward_map.max()
        finite = np.isfinite(gradnorm_map)
        grad_vmax = np.percentile(gradnorm_map[finite], 99) if finite.any() else 1.0

        for idx, ax in enumerate(axes):
            is_grad_panel = (idx == len(axes) - 1)  # last axis is gradnorm

            if is_grad_panel:
               data = gradnorm_map
               vmin, vmax = 0.0, grad_vmax
               cmap = "viridis"
               cbar_label = r"||∇_{x,y} reward||"
            else:
               data = reward_map
               cbar_label = "Reward"
               if idx == 0:  # full reward
                    vmin, vmax = data_min, data_max
                    cmap = "coolwarm"
               else:  # negative zoom reward
                    zoom_min, zoom_max = -0.1, 0.2
                    vmin = max(data_min, zoom_min)
                    vmax = min(0.0, zoom_max) if data_max > 0 else data_max
                    cmap = "Blues_r"

            im = ax.imshow(
              data,
              extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
              origin="lower",
              cmap=cmap,
              vmin=vmin, vmax=vmax,
              interpolation="bilinear",
            )
            plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

            # Only overlay reward=0 contour on reward panels (not gradnorm)
            if (not is_grad_panel) and neg_mask.any():
               ax.contour(X, Y, reward_map, levels=[0.0], colors="red",
                          linestyles="--", linewidths=2, alpha=0.7)
            # Draw walls using env.maze.walls (more accurate than maze_map)
            print("Drawing maze walls...")
            if hasattr(env.maze, 'walls'):
                 for wall in env.maze.walls:
                     (x0, y0), (x1, y1) = wall
                     ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3, zorder=10)
            else:
                # Fallback: use maze_map if walls attribute doesn't exist
                print("  Using maze_map fallback...")
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
                zorder=20)

            # Mark all goal positions
            print(f"Plotting {len(GOALS)} goals...")
            for i, goal in enumerate(GOALS):
                ax.plot(goal[0], goal[1], 'y*', markersize=20 if idx==0 else 10, 
                        markeredgecolor='black', markeredgewidth=1, zorder=20)
                # Add goal number label (only on full plot)
                if idx == 0:
                    ax.text(goal[0] + 0.3, goal[1] + 0.3, f'G{i+1}', 
                            fontsize=10, color='black', weight='bold', zorder=21,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            ax.set_xlabel('X position', fontsize=12)
            ax.set_ylabel('Y position', fontsize=12)
            ax.set_title(titles[idx], fontsize=14, fontweight='bold')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(grid_min[0], grid_max[0])
            ax.set_ylim(grid_min[1], grid_max[1])

        plt.tight_layout()
        
        # Ensure we save to the project root directory
        reward_dir = os.path.join(project_root, "reward_map")
        os.makedirs(reward_dir, exist_ok=True)
        save_path = os.path.join(reward_dir, OUTPUT_FILE) if not os.path.isabs(OUTPUT_FILE) else OUTPUT_FILE
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
        
        # Close figure to free memory
        plt.close()
        
    else:
        print("Skipping plotting due to matplotlib import error.")
        print(f"Reward map statistics:")
        print(f"  Min: {reward_map.min():.4f}, Max: {reward_map.max():.4f}, Mean: {reward_map.mean():.4f}")
        print(f"  Shape: {reward_map.shape}")
        # Save raw data as numpy array instead
        npy_path = os.path.join(project_root, OUTPUT_FILE.replace('.png', '.npy'))
        np.save(npy_path, reward_map)
        print(f"Reward map saved as numpy array to {npy_path}")




if __name__ == '__main__':
    # Example usage
    step = 200
    while(step <= 1000):
         np.random.seed(0)
         random.seed(0)
         #plot_critic_heatmap(step, agg_method='mean', highlight_negatives=True)
         heatmap(step, agg_method='mean', highlight_negatives = True)
         step += 200
    print('Done')





