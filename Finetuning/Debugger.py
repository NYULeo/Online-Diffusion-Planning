from math import cos
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)



# save_reward_heatmap_fast.py
# Works even when matplotlib/numpy is broken on the cluster
# Only requires: torch, numpy, imageio, minari, your reward code

import os
import numpy as np
import torch
import imageio.v3 as imageio
import minari
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
from Pretrain.Dataset import get_dataset
from typing import List
from utils import karras_beta_schedule
from Pretrain.Planners.Backbone.utils import cosine_beta




import numpy as np
import torch
try:
    import matplotlib
    #matplotlib.use('Agg')  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be skipped.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
import minari
from Pretrain.Rewards.nets import SimpleReward, Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
import random
from torch.utils.data import DistributedSampler, DataLoader
from utils import PlannerDataset
from utils import cycle
import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



def heatmap(STEP):
   # ================== Configuration ==================
   # ================== Configuration ==================
   #STEP = 100000                   # Checkpoint step to load
   RESOLUTION = 256                # Grid resolution (256x256 is fast and looks good)
   BATCH_SIZE = 16384              # Batch size for efficient processing
   MAX_GOALS_TO_PLOT = 20           # Plot only the first few unique goals
   GRID_MARGIN = 0.5               # Extra padding around observed positions for plotting
   OUTPUT_FILE = f"{STEP}_heatmap.png"

   # ================== Load Environment ==================
   #print("Loading environment...")
   print(f'Ploting the heatmap for checkpoint: {STEP}')
   dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
   env = dataset.recover_environment().unwrapped  # Unwrap to access maze attribute

   # ================== Extract All Unique Goals from Dataset ==================
   #print("Extracting all unique goals and position bounds from dataset...")
   all_goals = set()
   episode_count = 0
   max_episodes = 200  # Check first 200 episodes to find all goals
   pos_min = np.array([np.inf, np.inf], dtype=np.float32)
   pos_max = np.array([-np.inf, -np.inf], dtype=np.float32)
   first_start = None


   # Lines 89-94 - Fixed version
   t = 0
   while(t < 20):
    obs, info = env.reset(seed = t)
    goal = env.generate_target_goal()
    
    # Convert to tuple and round to avoid floating point precision issues
    # Handle both 2D arrays and longer arrays
    if isinstance(goal, np.ndarray):
        goal_2d = goal[:2] if len(goal) >= 2 else goal
        goal_rounded = tuple(np.round(goal_2d, 2))
    else:
        goal_rounded = tuple(np.round(goal[:2], 2))
    
    all_goals.add(goal_rounded)
    
    # Also update position bounds from observation
    obs_dict, info = env.reset(seed = t)
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
   #print(f"Found {len(GOALS)} unique goals:")
   #for i, goal in enumerate(GOALS):
       #print(f"  Goal {i+1}: [{goal[0]:.2f}, {goal[1]:.2f}]")

   # Determine plotting bounds based on observed positions
   if np.isinf(pos_min).any() or np.isinf(pos_max).any():
       raise ValueError("Could not determine position bounds from dataset.")


   # Determine plotting bounds based on observed positions
   if np.isinf(pos_min).any() or np.isinf(pos_max).any():
      if hasattr(env, 'maze') and hasattr(env.maze, 'maze_map'):
          map_height = env.maze.map_length
          map_width = env.maze.map_width
          cell = env.maze.maze_size_scaling
          pos_min = np.array([-map_width / 2.0 * cell, -map_height / 2.0 * cell], dtype=np.float32)
          pos_max = np.array([ map_width / 2.0 * cell,  map_height / 2.0 * cell], dtype=np.float32)
      else:
          raise ValueError("Could not determine position bounds from dataset or maze.")

   grid_min = (pos_min - GRID_MARGIN).astype(np.float32)
   grid_max = (pos_max + GRID_MARGIN).astype(np.float32)



   # Expand bounds a bit so the heatmap includes some context outside trajectories
   grid_min = (pos_min - GRID_MARGIN).astype(np.float32)
   grid_max = (pos_max + GRID_MARGIN).astype(np.float32)
    
   # Determine start position
 
   default_start = np.array([-1.5, -0.5], dtype=np.float32)  # choose your constant
   start_pos = default_start

   # ================== Load Reward Model ==================
   #print(f"Loading reward model (step {STEP})...")
   state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', STEP, 'medium')
   model = SimpleReward(obs_dim, act_dim)
   #model = Reward(obs_dim, act_dim)
   model.load_state_dict(state_dict)
   model.eval()
   stats = get_pretrained_reward_stats(name)

   # ================== Create Grid ==================
   #print(f"Creating {RESOLUTION}x{RESOLUTION} grid within observed bounds...")
   x = np.linspace(grid_min[0], grid_max[0], RESOLUTION)
   y = np.linspace(grid_min[1], grid_max[1], RESOLUTION)
   X, Y = np.meshgrid(x, y, indexing='xy')

   # ================== Evaluate Rewards for All Goals ==================
   #print("Evaluating rewards for all goals...")
   # Use action candidates
   n_actions = 5  # Sample 5x5 = 25 actions
   actions = np.linspace(-1.0, 1.0, n_actions)
   action_grid = np.array([[ax, ay] for ax in actions for ay in actions]).astype(np.float32)
   acts_t = torch.from_numpy(action_grid)

   #  Store reward maps for each goal
   reward_maps_per_goal = []

   for goal_idx, goal in enumerate(GOALS):
      #print(f"\nProcessing goal {goal_idx+1}/{len(GOALS)}: [{goal[0]:.2f}, {goal[1]:.2f}]")
    
     # Create observations: [x, y, goal_x, goal_y]
      obs_base = np.stack([
         X.ravel(),
         Y.ravel(),
         np.full(RESOLUTION**2, goal[0]),
         np.full(RESOLUTION**2, goal[1])
      ], axis=1).astype(np.float32)
    
      reward_map_goal = np.full(RESOLUTION**2, -1e10, dtype=np.float32)
    
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
             reward_map_goal[start:end] = r.max(axis=1)
            
             #if start % (BATCH_SIZE*10) == 0:
                #print(f"  → {start}/{len(obs_base)}")
    
      reward_maps_per_goal.append(reward_map_goal.reshape(RESOLUTION, RESOLUTION))


   # Aggregate reward maps (take maximum across all goals for each position)
   #print("\nAggregating reward maps across all goals...")
   reward_map = np.stack(reward_maps_per_goal, axis=0).max(axis=0)

   # ================== Plot Heatmap ==================
   if MATPLOTLIB_AVAILABLE:
    #print("Creating heatmap...")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot heatmap
    im = ax.imshow(
        reward_map,
        extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
        origin='lower',
                   cmap='RdYlBu_r', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Reward')

    # Draw walls using env.maze.walls (more accurate than maze_map)
    #print("Drawing maze walls...")
    if hasattr(env.maze, 'walls'):
        for wall in env.maze.walls:
            (x0, y0), (x1, y1) = wall
            ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3, zorder=10)
    else:
        # Fallback: use maze_map if walls attribute doesn't exist
        #print("  Using maze_map fallback...")
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
        ax.plot(goal[0], goal[1], 'y*', markersize=20, 
                markeredgecolor='black', markeredgewidth=1, zorder=20)
        # Add goal number label
        ax.text(goal[0] + 0.3, goal[1] + 0.3, f'G{i+1}', 
                fontsize=10, color='black', weight='bold', zorder=21,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('X position', fontsize=12)
    ax.set_ylabel('Y position', fontsize=12)
    ax.set_title(f'Reward Heatmap (Step {STEP}) - All {len(GOALS)} Goals', fontsize=14, fontweight='bold')
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
    #print(f"Heatmap saved to {save_path}")
    
    # Close figure to free memory (good practice, especially on servers)
    plt.close()
    # Note: plt.show() is not called - script runs headless and only saves to file
   else:
      print("Skipping plotting due to matplotlib import error.")
      print(f"Reward map statistics:")
      print(f"  Min: {reward_map.min():.4f}, Max: {reward_map.max():.4f}, Mean: {reward_map.mean():.4f}")
      print(f"  Shape: {reward_map.shape}")
    # Save raw data as numpy array instead
      npy_path = os.path.join(project_root, OUTPUT_FILE.replace('.png', '.npy'))
      np.save(npy_path, reward_map)
      print(f"Reward map saved as numpy array to {npy_path}")

"""

if __name__ == '__main__':
   
   step = 20000
   while(step <= 200000):
       np.random.seed(0)
       random.seed(0)
       torch.manual_seed(0) 
       #print(f"Ploting the heatmap for checkpoint {step}")
       heatmap(step)
       step += 20000
   print('Done')

"""

"""
def Initialize_Buffer():
        Buffer = []
        dataset = get_dataset('pointmaze', 'medium')
        trajs = dataset.get_trajectories()
        Buffer.extend(trajs)
        return Buffer
 

Buffer = Initialize_Buffer()
PlannerDataset = PlannerDataset(Buffer, 32, 'pointmaze', 'medium')
#sampler = DistributedSampler(PlannerDataset, shuffle=True, drop_last=True)
dataloader = DataLoader(PlannerDataset, 12,  shuffle = True,  drop_last = True)
dataloader = cycle(dataloader)
t = 0
coordinates = []
while (t<500):
   conds = next(dataloader)
   for cond in conds:
       coordinates.append(cond[:2].numpy())
   t += 1
"""







"""
save_path = f'./Finetuning/Initial_Conds_950.pkl'
with open(save_path, 'rb') as f:
    coordinates = pickle.load(f)

coordinates = np.array(coordinates)
plt.figure(figsize=(10, 8))
plt.hexbin(coordinates[:, 0], coordinates[:, 1], gridsize=50, cmap='viridis', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Hexbin Heatmap of Coordinates')
plt.show()
"""









"""

def plot_reward_curve(steps: List, rewards: List, constraints: List,
                      title: str = "Finetuning Reward Curve"):
        if not rewards:
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
        steps = np.array(steps)
        rewards = np.array(rewards)

        smooth_window_reward = 60
        smoothed = _smooth_curve(rewards, smooth_window_reward)
        valid_idx = ~np.isnan(smoothed)
        ax1.plot(steps[valid_idx], smoothed[valid_idx],
                     color=smooth_color, linewidth=3.0,
                     label=f'Smoothed Reward (window={smooth_window_reward})')
        ax1.plot(steps, rewards, alpha=0.3, color=raw_color, linewidth=1.0, label='Raw Reward')
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12, color=raw_color)
        ax1.tick_params(axis='y', labelcolor=raw_color)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fancybox=True, fontsize=12)
        sns.despine()

        
        ax2 = ax1.twinx()
        C_vals = np.array(constraints)
        smooth_window_constraint = 50
        smoothed = _smooth_curve(constraints, smooth_window_constraint)
        valid_idx = ~np.isnan(smoothed)
        ax2.plot(steps[valid_idx], smoothed[valid_idx],
                     color=constraint_color, linewidth=2.0,
                     label=f'Smoothed Constraint (window={smooth_window_constraint})')
        #ax2.plot(steps[:len(C_vals)], C_vals, color=constraint_color, alpha=0.7, linewidth=1.5, label='Constraint')
        ax2.set_ylabel('Constraint', fontsize=12, color=constraint_color)
        ax2.tick_params(axis='y', labelcolor=constraint_color)
        ax2.legend(loc='upper right')
        sns.despine()
        
        


        plt.tight_layout()
        plt.show()
        return fig

def _smooth_curve(data: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        padded = np.full_like(data, np.nan)
        padded[window-1:] = smoothed
        return padded

save_path = f'./Finetuning/PointMaze_Medium_Planner_finetune_reward_logs.pkl'
with open(save_path, 'rb') as f:
    data = pickle.load(f)

steps = data['steps'][:450]
rewards = data['rewards'][:450]
constraints = data['constraints'][:450]

plot_reward_curve(steps, rewards, constraints)
"""

