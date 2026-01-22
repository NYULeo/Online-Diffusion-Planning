from math import cos
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
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
import numpy as np
import matplotlib.pyplot as plt
import os
# Assuming project_root, MATPLOTLIB_AVAILABLE, get_pretrained_reward, etc., are defined elsewhere

def heatmap(STEP, agg_method='max', highlight_negatives=True):
    # ================== Configuration ==================
    RESOLUTION = 256           # Grid resolution (256x256 is fast and looks good)
    BATCH_SIZE = 16384              # Batch size for efficient processing
    MAX_GOALS_TO_PLOT = 3          # Plot only the first few unique goals
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
            """
            map_height = env.maze.map_length
            map_width = env.maze.map_width
            cell = env.maze.maze_size_scaling
            pos_min = np.array([-map_width / 2.0 * cell, -map_height / 2.0 * cell], dtype=np.float32)
            pos_max = np.array([map_width / 2.0 * cell, map_height / 2.0 * cell], dtype=np.float32)
            """
            map_height = env.maze.map_length
            map_width = env.maze.map_width
            cell = env.maze.maze_size_scaling
            pos_min = np.array([-map_width/2*cell, -map_height/2*cell])
            pos_max = np.array([ map_width/2*cell,  map_height/2*cell])
        else:
            raise ValueError("Could not determine position bounds from dataset or maze.")

    # Expand bounds a bit so the heatmap includes some context outside trajectories
    grid_min = (pos_min - GRID_MARGIN).astype(np.float32)
    grid_max = (pos_max + GRID_MARGIN).astype(np.float32)
    
    # Determine start position
    default_start = np.array([-1.5, -0.5], dtype=np.float32)  # choose your constant
    start_pos = default_start

    # ================== Load Reward Model ==================
    state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', STEP, 'medium')
    model = SimpleReward(obs_dim, act_dim)
    model.load_state_dict(state_dict)
    model.eval()
    stats = get_pretrained_reward_stats(name)

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

            r = model(obs_norm, act_rep)  # [B]
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

def plot_reward_heatmap_large(
    step=200,
    dataset_id="D4RL/pointmaze/large-v2",
    resolution=128,
    batch_size=8192,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import minari
    import torch
    from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
    from Pretrain.Rewards.nets import SimpleReward

    dataset = minari.load_dataset(dataset_id, download=True)
    env = dataset.recover_environment().unwrapped

    # Full maze bounds
    H = env.maze.map_length
    W = env.maze.map_width
    cell = env.maze.maze_size_scaling
    grid_min = np.array([-W / 2 * cell, -H / 2 * cell], dtype=np.float32)
    grid_max = np.array([ W / 2 * cell,  H / 2 * cell], dtype=np.float32)

    # Grid
    x = np.linspace(grid_min[0], grid_max[0], resolution)
    y = np.linspace(grid_min[1], grid_max[1], resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Reward model
    state_dict, obs_dim, act_dim, name = get_pretrained_reward("pointmaze", step, "large")
    model = SimpleReward(obs_dim, act_dim)
    model.load_state_dict(state_dict)
    model.eval()
    stats = get_pretrained_reward_stats(name)

    # Single goal (from env)
    obs_dict, _ = env.reset(seed=0)
    goal = env.generate_target_goal()
    goal = np.array(goal[:2], dtype=np.float32)

    # Evaluate reward at zero action
    obs_base = np.stack(
        [
            X.ravel(),
            Y.ravel(),
            np.full(resolution**2, goal[0]),
            np.full(resolution**2, goal[1]),
        ],
        axis=1,
    ).astype(np.float32)

    reward_map = np.full(resolution**2, -1e10, dtype=np.float32)

    obs_mean_t = torch.as_tensor(stats.obs_mean, dtype=torch.float32)
    obs_std_t = torch.as_tensor(
        np.maximum(stats.obs_std, getattr(stats, "std_floor", 1e-3)),
        dtype=torch.float32,
    )

    with torch.no_grad():
        for start in range(0, len(obs_base), batch_size):
            end = min(start + batch_size, len(obs_base))
            batch_obs = torch.from_numpy(obs_base[start:end]).float()
            act_rep = torch.zeros((end - start, 2), dtype=torch.float32)

            obs_norm = (batch_obs - obs_mean_t) / obs_std_t
            r = model(obs_norm, act_rep)
            reward_map[start:end] = r.cpu().numpy()

    reward_map = reward_map.reshape(resolution, resolution)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        reward_map,
        extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
        origin="lower",
        cmap="coolwarm",
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Reward", shrink=0.8)

    pad = 1.0
    xmin, xmax = (-W/2 + pad) * cell, (W/2 - pad) * cell
    ymin, ymax = (-H/2 + pad) * cell, (H/2 - pad) * cell
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Draw walls
    maze_map = env.maze.maze_map
    for row in range(H):
        for col in range(W):
            if row == 0 or row == H - 1 or col == 0 or col == W - 1:
                continue
            if maze_map[row][col] == 1:
                x0 = (col - W / 2.0 + 0.5) * cell
                y0 = (H / 2.0 - row - 0.5) * cell
                half = cell / 2.0
                square = np.array(
                    [
                        [x0 - half, y0 - half],
                        [x0 + half, y0 - half],
                        [x0 + half, y0 + half],
                        [x0 - half, y0 + half],
                        [x0 - half, y0 - half],
                    ]
                )
                ax.plot(square[:, 0], square[:, 1], "k-", linewidth=1)

    ax.set_aspect("equal")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Reward Heatmap (Step {step})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to repo-root reward_map/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reward_dir = os.path.join(project_root, "reward_map")
    os.makedirs(reward_dir, exist_ok=True)
    save_path = os.path.join(reward_dir, f"reward_heatmap_large_step_{step}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {save_path}")
    plt.close(fig)



if __name__ == '__main__':
    # Example usage
    step = 500
    while(step <= 2000):
         np.random.seed(0)
         random.seed(0)
         heatmap(step)
         step += 500
    print('Done')







"""
import numpy as np
import matplotlib.pyplot as plt
import minari


dataset_id = "D4RL/antmaze/large-play-v1"
dataset_id = 'D4RL/pointmaze/large-v2'
dataset = minari.load_dataset(dataset_id, download=True)
env = dataset.recover_environment().unwrapped

maze_map = env.maze.maze_map
H = env.maze.map_length
W = env.maze.map_width
cell = env.maze.maze_size_scaling

fig, ax = plt.subplots(figsize=(10, 7))

# Draw walls, but skip outer border cells
for row in range(H):
    for col in range(W):
        if row == 0 or row == H-1 or col == 0 or col == W-1:
            continue  # drop the outer frame
        if maze_map[row][col] == 1:
            x = (col - W/2.0 + 0.5) * cell
            y = (H/2.0 - row - 0.5) * cell
            half = cell / 2.0
            square = np.array([
                [x - half, y - half],
                [x + half, y - half],
                [x + half, y + half],
                [x - half, y + half],
                [x - half, y - half],
            ])
            ax.plot(square[:, 0], square[:, 1], "k-", linewidth=1)

# Origin
ax.plot(0, 0, "ro", label="origin")

pad = 1.0
xmin, xmax = (-W/2 + pad) * cell, (W/2 - pad) * cell
ymin, ymax = (-H/2 + pad) * cell, (H/2 - pad) * cell

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("PointMaze Large Coordinate System")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

"""

"""
import minari

dataset_id = "D4RL/antmaze/large-play-v1"
#dataset_id = "D4RL/pointmaze/large-v2"
dataset = minari.load_dataset(dataset_id, download=True)
env = dataset.recover_environment().unwrapped
print("distance_threshold:", getattr(env, "distance_threshold", None))

print("obs space:", env.observation_space)
print("action space:", env.action_space)
count = 0
#ep = next(dataset.iterate_episodes())
goal = np.array([15.0, -13.0], dtype = np.float32)
for ep in dataset.iterate_episodes():
    #print("observation keys:", ep.observations.keys())
    #print("Desired goals:", ep.observations['desired_goal'][0])
    for i in range(len(ep.observations['achieved_goal'])):
         print("Acheived goals:", ep.observations['achieved_goal'][i])
         print(type(ep.observations['achieved_goal'][i]))
    #print(len(ep.observations['achieved_goal']))
    break
   
    
    #print("observation shape:", ep.observations["observation"].shape)
    #print('rewards sum:', ep.rewards[888])
    
    for i in range(len(ep.observations['achieved_goal'])):
        if np.allclose(ep.observations['achieved_goal'][i], goal, atol = 0.5):
            count += 1
"""







