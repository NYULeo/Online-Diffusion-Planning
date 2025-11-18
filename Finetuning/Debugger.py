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
from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats

"""
# ================== Config ==================
DATASET_NAME = 'D4RL/pointmaze/medium-v2'
STEP = 44000
GOAL = np.array([9.0, 9.0])
RESOLUTION = 512                     # 512×512 looks gorgeous and still fast
OUTPUT_PNG = f"reward_heatmap_medium_step{STEP}_goal{GOAL[0]}_{GOAL[1]}.png"
BATCH_SIZE = 16384                   # safe even on 8GB GPU / 16GB RAM

print(f"Loading dataset {DATASET_NAME}...")
dataset = minari.load_dataset(DATASET_NAME, download=True)
env = dataset.recover_environment().unwrapped

print("Loading reward model...")
state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', STEP, 'medium')
model = Reward(obs_dim, act_dim)
model.load_state_dict(state_dict)
model.eval()
stats = get_pretrained_reward_stats(name)

# ================== Grid ==================
x = np.linspace(-1, 11, RESOLUTION)
y = np.linspace(-1, 11, RESOLUTION)
X, Y = np.meshgrid(x, y, indexing='xy')

obs_base = np.stack([
    X.ravel(),
    Y.ravel(),
    np.full(RESOLUTION**2, GOAL[0]),
    np.full(RESOLUTION**2, GOAL[1])
], axis=1).astype(np.float32)

# ================== Action candidates ==================
n_act = 25
acts = np.linspace(-1.0, 1.0, n_act)
act_grid_x, act_grid_y = np.meshgrid(acts, acts)
candidate_acts = np.stack([act_grid_x.ravel(), act_grid_y.ravel()], axis=1).astype(np.float32)
acts_t = torch.from_numpy(candidate_acts)

print(f"Evaluating {RESOLUTION}x{RESOLUTION} grid (max over {n_act}x{n_act} actions)...")

best = np.full(RESOLUTION**2, -1e10, dtype=np.float32)

with torch.no_grad():
    for start in range(0, len(obs_base), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(obs_base))
        batch_obs = torch.from_numpy(obs_base[start:end])

        # repeat actions
        obs_rep = batch_obs.unsqueeze(1).repeat(1, len(acts_t), 1).reshape(-1, 4)
        act_rep = acts_t.unsqueeze(0).repeat(end-start, 1, 1).reshape(-1, 2)

        obs_rep = stats.norm_obs(obs_rep)
        obs_rep = obs_rep.float()
        act_rep = act_rep.float()
        r = model(obs_rep, act_rep).cpu().numpy().reshape(end-start, -1)
        best[start:end] = r.max(axis=1)

        if start % (BATCH_SIZE*10) == 0:
            print(f"  → {start}/{len(obs_base)}")

reward_map = best.reshape(RESOLUTION, RESOLUTION)

# ================== Simple colormap (no matplotlib needed) ==================
def colormap(value, vmin=None, vmax=None):
    if vmin is None: vmin = reward_map.min()
    if vmax is None: vmax = reward_map.max()
    norm = (value - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)

    # Blue → White → Red
    r = np.where(norm < 0.5, 0.0, (norm - 0.5)*2)
    g = np.where(norm < 0.5, norm*2, 1 - (norm - 0.5)*2)
    b = 1 - norm*0.8
    return np.stack([r, g, b, np.ones_like(r)], axis=-1)  # RGBA

img = (colormap(reward_map) * 255).astype(np.uint8)

# ================== Draw walls & markers directly on image ==================
def world_to_pixel(x, y):
    px = ((x - (-1)) / 12 * (RESOLUTION-1)).astype(int)
    py = ((y - (-1)) / 12 * (RESOLUTION-1)).astype(int)
    return np.clip(px, 0, RESOLUTION-1), np.clip(py, 0, RESOLUTION-1)

# black walls
for wall in env.maze.walls:
    (x0,y0), (x1,y1) = wall
    px0, py0 = world_to_pixel(x0, y0)
    px1, py1 = world_to_pixel(x1, y1)
    # simple thick line
    for t in range(-8, 9):
        for s in range(-8, 9):
            if abs(t) + abs(s) < 12:
                img[py0 + t, px0 + s] = [0, 0, 0, 255]
                img[py1 + t, px1 + s] = [0, 0, 0, 255]

# start (green circle)
sx, sy = world_to_pixel(1.0, 1.0)
for dx in range(-25, 26):
    for dy in range(-25, 26):
        if dx*dx + dy*dy < 25**2:
            y, x = sy + dy, sx + dx
            if 0 <= y < RESOLUTION and 0 <= x < RESOLUTION:
                img[y, x] = [0, 255, 0, 255]

# goal (yellow star)
gx, gy = world_to_pixel(GOAL[0], GOAL[1])
for dx in range(-35, 36):
    for dy in range(-35, 36):
        if dx*dx + dy*dy < 35**2:
            y, x = gy + dy, gx + dx
            if 0 <= y < RESOLUTION and 0 <= x < RESOLUTION:
                img[y, x] = [255, 255, 0, 255]

print(f"Saving {OUTPUT_PNG} ...")
imageio.imwrite(OUTPUT_PNG, img)
print("Done! Heatmap saved with walls + start + goal.")

"""


import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be skipped.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
import minari
from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats


# ================== Configuration ==================
STEP = 44000                    # Checkpoint step to load
RESOLUTION = 256                # Grid resolution (256x256 is fast and looks good)
BATCH_SIZE = 16384              # Batch size for efficient processing
OUTPUT_FILE = f"reward_heatmap_step{STEP}_all_goals.png"

# ================== Load Environment ==================
print("Loading environment...")
dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
env = dataset.recover_environment().unwrapped  # Unwrap to access maze attribute

# ================== Extract All Unique Goals from Dataset ==================
print("Extracting all unique goals from dataset...")
all_goals = set()
episode_count = 0
max_episodes = 200  # Check first 200 episodes to find all goals

for episode in dataset:
    # Get goal from observations (goals are in obs[2:4] for pointmaze)
    obs = episode.observations
    if len(obs) > 0:
        # Goals are typically in the observation space [x, y, goal_x, goal_y]
        goals = obs[:, 2:4]  # Extract goal positions
        unique_goals_episode = np.unique(goals, axis=0)
        for goal in unique_goals_episode:
            # Round to 2 decimal places to avoid floating point precision issues
            goal_rounded = (round(goal[0], 2), round(goal[1], 2))
            all_goals.add(goal_rounded)
    
    episode_count += 1
    # Limit to first max_episodes for speed (should capture all goals)
    if episode_count >= max_episodes:
        break

# Convert to numpy array
if len(all_goals) == 0:
    raise ValueError("No goals found in dataset! Check dataset structure.")
    
GOALS = np.array(list(all_goals))
print(f"Found {len(GOALS)} unique goals:")
for i, goal in enumerate(GOALS):
    print(f"  Goal {i+1}: [{goal[0]:.2f}, {goal[1]:.2f}]")

# ================== Load Reward Model ==================
print(f"Loading reward model (step {STEP})...")
state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', STEP, 'medium')
model = Reward(obs_dim, act_dim)
model.load_state_dict(state_dict)
model.eval()
stats = get_pretrained_reward_stats(name)

# ================== Create Grid ==================
print(f"Creating {RESOLUTION}x{RESOLUTION} grid...")
x = np.linspace(-1, 11, RESOLUTION)
y = np.linspace(-1, 11, RESOLUTION)
X, Y = np.meshgrid(x, y, indexing='xy')

# ================== Evaluate Rewards for All Goals ==================
print("Evaluating rewards for all goals...")
# Use action candidates
n_actions = 5  # Sample 5x5 = 25 actions
actions = np.linspace(-1.0, 1.0, n_actions)
action_grid = np.array([[ax, ay] for ax in actions for ay in actions]).astype(np.float32)
acts_t = torch.from_numpy(action_grid)

# Store reward maps for each goal
reward_maps_per_goal = []

for goal_idx, goal in enumerate(GOALS):
    print(f"\nProcessing goal {goal_idx+1}/{len(GOALS)}: [{goal[0]:.2f}, {goal[1]:.2f}]")
    
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
            
            if start % (BATCH_SIZE*10) == 0:
                print(f"  → {start}/{len(obs_base)}")
    
    reward_maps_per_goal.append(reward_map_goal.reshape(RESOLUTION, RESOLUTION))







# Aggregate reward maps (take maximum across all goals for each position)
print("\nAggregating reward maps across all goals...")
reward_map = np.stack(reward_maps_per_goal, axis=0).max(axis=0)

# ================== Plot Heatmap ==================
if MATPLOTLIB_AVAILABLE:
    print("Creating heatmap...")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot heatmap
    im = ax.imshow(reward_map, extent=[-1, 11, -1, 11], origin='lower', 
                   cmap='RdYlBu_r', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Reward')

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
    ax.plot(1.0, 1.0, 'go', markersize=15, label='Start', markeredgecolor='black', 
            markeredgewidth=2, zorder=20)

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

    plt.tight_layout()
    
    # Ensure we save to the project root directory
    save_path = os.path.join(project_root, OUTPUT_FILE) if not os.path.isabs(OUTPUT_FILE) else OUTPUT_FILE
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    
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



