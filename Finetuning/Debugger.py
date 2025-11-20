import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

import numpy as np
import torch
import imageio.v3 as imageio
import minari

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}).")
    MATPLOTLIB_AVAILABLE = False
    plt = None

from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats

# === Configuration ===
STEP = 44000
RESOLUTION = 256
BATCH_SIZE = 16384
MAX_GOALS_TO_PLOT = 10
GRID_MARGIN = 0.5
OUTPUT_FILE = f"reward_heatmap_step{STEP}_all_goals.png"

# === Load Dataset / Environment ===
print("Loading environment...")
dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
env = dataset.recover_environment().unwrapped

# === Extract Goals and Position Bounds ===
print("Extracting unique goals and bounds...")
all_goals = set()
episode_count = 0
max_episodes = 200

pos_min = np.array([np.inf, np.inf], dtype=np.float32)
pos_max = np.array([-np.inf, -np.inf], dtype=np.float32)
first_start = None

for ep in dataset:
    obs = ep.observations['observation']
    if len(obs) > 0:
        positions = obs[:, :2]
        pos_min = np.minimum(pos_min, positions.min(axis=0))
        pos_max = np.maximum(pos_max, positions.max(axis=0))
        if first_start is None:
            first_start = positions[0]

        goals = obs[:, 2:4]
        uniq = np.unique(goals, axis=0)
        for g in uniq:
            # Round to avoid FP issues
            all_goals.add((round(float(g[0]), 3), round(float(g[1]), 3)))
    episode_count += 1
    if episode_count >= max_episodes:
        break

if not all_goals:
    raise RuntimeError("No goals found in dataset!")

goal_list = sorted(all_goals)
if len(goal_list) > MAX_GOALS_TO_PLOT:
    print(f"Trimming {len(goal_list)} goals to first {MAX_GOALS_TO_PLOT}")
    goal_list = goal_list[:MAX_GOALS_TO_PLOT]
GOALS = np.array(goal_list, dtype=np.float32)

print("Goals to plot:")
for i, g in enumerate(GOALS):
    print(f" {i}: {g}")

# Expand bounds for plotting
grid_min = pos_min - GRID_MARGIN
grid_max = pos_max + GRID_MARGIN

# Determine start
if hasattr(env.maze, 'start_pos'):
    start_pos = np.array(env.maze.start_pos[:2], dtype=np.float32)
elif first_start is not None:
    start_pos = np.array(first_start, dtype=np.float32)
else:
    start_pos = np.array([0.0, 0.0], dtype=np.float32)

# === Load Reward Model ===
print(f"Loading reward at step {STEP}")
state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', STEP, 'medium')
model = Reward(obs_dim, act_dim)
model.load_state_dict(state_dict)
model.eval()
stats = get_pretrained_reward_stats(name)

# === Make Grid ===
print(f"Creating grid of size {RESOLUTION}x{RESOLUTION}")
xs = np.linspace(grid_min[0], grid_max[0], RESOLUTION)
ys = np.linspace(grid_min[1], grid_max[1], RESOLUTION)
X, Y = np.meshgrid(xs, ys, indexing='xy')

# === Evaluate Reward Maps ===
print("Evaluating reward for each goal …")
n_actions = 5
actions = np.linspace(-1.0, 1.0, n_actions)
action_grid = np.array([[ax, ay] for ax in actions for ay in actions], dtype=np.float32)
acts_t = torch.from_numpy(action_grid)

reward_maps = []

for i, goal in enumerate(GOALS):
    print(f" Goal {i} at {goal}")
    # Build observations: [x, y, goal_x, goal_y]
    obs_base = np.stack([
        X.ravel(),
        Y.ravel(),
        np.full(X.size, goal[0], dtype=np.float32),
        np.full(X.size, goal[1], dtype=np.float32),
    ], axis=1)

    reward_flat = np.full(X.size, -np.inf, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, obs_base.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, obs_base.shape[0])
            batch_obs = torch.from_numpy(obs_base[start:end])
            # replicate for actions
            obs_rep = batch_obs.unsqueeze(1).repeat(1, len(acts_t), 1).reshape(-1, obs_base.shape[1])
            act_rep = acts_t.unsqueeze(0).repeat(end - start, 1, 1).reshape(-1, act_dim)

            obs_norm = stats.norm_obs(obs_rep.float())
            act_norm = act_rep.float()

            r = model(obs_norm, act_norm).cpu().numpy().reshape(end - start, -1)
            # max over actions
            reward_flat[start:end] = r.max(axis=1)

    reward_map = reward_flat.reshape(RESOLUTION, RESOLUTION)
    reward_maps.append(reward_map)

# Aggregate over goals
print("Aggregating reward maps …")
stacked = np.stack(reward_maps, axis=0)
reward_map = np.max(stacked, axis=0)

# === Plot Heatmap ===
if MATPLOTLIB_AVAILABLE:
    print("Plotting …")
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        reward_map,
        extent=(grid_min[0], grid_max[0], grid_min[1], grid_max[1]),
        origin='lower',
        cmap='RdYlBu_r',
        interpolation='bilinear'
    )
    plt.colorbar(im, ax=ax, label='Reward')

    # Draw walls from maze_map
    maze = env.unwrapped.maze
    if hasattr(maze, 'maze_map'):
        maze_map = np.array(maze.maze_map)
        rows, cols = maze_map.shape
        # compute cell centers
        cell_size = getattr(maze, 'maze_size_scaling', 1.0)
        for r in range(rows):
            for c in range(cols):
                if maze_map[r, c] == 1:
                    x_center = (c + 0.5) * cell_size
                    # note: row 0 is top or bottom? According to docs, row 0 is top of list → invert
                    y_center = (rows - 1 - r + 0.5) * cell_size
                    half = cell_size / 2.0
                    square = [
                        (x_center - half, y_center - half),
                        (x_center + half, y_center - half),
                        (x_center + half, y_center + half),
                        (x_center - half, y_center + half),
                        (x_center - half, y_center - half),
                    ]
                    xsq, ysq = zip(*square)
                    ax.plot(xsq, ysq, 'k-', linewidth=2, zorder=5)
    else:
        print("Warning: maze_map not found on env.maze, can't draw walls.")

    # Plot start
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=12,
            markeredgecolor='black', markeredgewidth=2, label='Start', zorder=10)

    # Plot goals
    for i, goal in enumerate(GOALS):
        ax.plot(goal[0], goal[1], 'y*', markersize=15,
                markeredgecolor='black', markeredgewidth=1, zorder=10)
        # label with offset to avoid overlap
        ax.text(goal[0] + 0.1, goal[1] + 0.1, f"G{i}", fontsize=10,
                color='black', weight='bold', zorder=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Reward Heatmap (step {STEP}, {len(GOALS)} goals)")
    ax.set_aspect('equal')
    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(project_root, OUTPUT_FILE)
    plt.savefig(save_path, dpi=150)
    print("Saved heatmap to:", save_path)
    plt.close()
else:
    print("Matplotlib not available — saving reward map as numpy array.")
    npy_path = os.path.join(project_root, OUTPUT_FILE.replace('.png', '.npy'))
    np.save(npy_path, reward_map)
    print("Saved numpy to:", npy_path)
