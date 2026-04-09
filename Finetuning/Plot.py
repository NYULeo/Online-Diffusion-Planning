import numpy as np
import matplotlib.pyplot as plt





def plot_lines(data_list, labels=None, colors=None, markers=None, 
               xlabel='X', ylabel='Y', title='Plot', 
               save_path=None, xlim=None, ylim=None):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Default styling matching the image
    if colors is None:
        colors = ['#56B4E9',   # Light blue
                 '#CC79A7',  # Pink/magenta
                 '#0072B2']   # Darker blue
    
    if markers is None:
        markers = ['o', 's', 'D']  # Circle, square, diamond
    
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(data_list))]
    
    for idx, data in enumerate(data_list):
        # Convert list of [x, y] pairs to numpy arrays
        data = np.array(data)
        x = data[:, 0]
        y = data[:, 1]
        
        # Plot line with markers
        ax.plot(x, y, 
               color=colors[idx % len(colors)], 
               marker=markers[idx % len(markers)],
               markersize=6,
               linewidth=2,
               label=labels[idx],
               markeredgecolor='white' if markers[idx % len(markers)] == 'o' else colors[idx % len(colors)],
               markeredgewidth=0.5 if markers[idx % len(markers)] == 'o' else 1,
               markerfacecolor=colors[idx % len(colors)],
               fillstyle='full' if markers[idx % len(markers)] == 'o' else 'none')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


# Or manually create from the extracted values (x-axis divided by 100):
# Each round has 2400 env steps, divided by 100 = 24

"""
plot_lines(
    [data],
    labels=['Normalized Score'],
    title='Normalized Score vs Environment Steps',
    xlabel='Env Steps (x100)',
    ylabel='Normalized Score'
)
"""
"""Count training windows for horizon 32 vs 70 on pointmaze large."""

import os
import numpy as np
import ogbench as og
import mediapy as media


"""
dataset_name = "cube-single-play-v0"
env, train_dataset, val_dataset = og.make_env_and_datasets(dataset_name, render_mode = 'rgb_array')
episode_obs = []
episode_acts = []
last_start = 0

for i in range(len(train_dataset['observations'])):
    if( train_dataset['terminals'][i] == 1):
          episode_obs.append(train_dataset['observations'][last_start:i])
          episode_acts.append(train_dataset['actions'][last_start:i])
          last_start = i+1
          
episode_obs = np.array(episode_obs)
episode_acts = np.array(episode_acts)
"""


dataset_name = "cube-single-play-v0"
env, train_dataset, val_dataset = og.make_env_and_datasets(dataset_name, render_mode='rgb_array')




# 1. Get the 5 official target positions
target_positions = []
for task_id in range(1, 6):
    obs, info = env.reset(options={"task_id": task_id})
    goal_cube_pos = info['goal'][15:18]   # cube position only
    target_positions.append(goal_cube_pos)
target_positions = np.array(target_positions)    # shape: (5, 3)

success_threshold = 0.05   # official-style tolerance (5 cm)

print(target_positions)
# 2. Split the flat dataset into long play episodes (your code, fixed)
episode_obs = []
last_start = 0

for i in range(len(train_dataset['observations'])):
    if train_dataset['terminals'][i] == 1 or i == len(train_dataset['observations']) - 1:
        obs_slice = train_dataset['observations'][last_start:i+1]   # include terminal
        act_slice = train_dataset['actions'][last_start:i]
        episode_obs.append(np.array(obs_slice))
        last_start = i + 1

print(f"Found {len(episode_obs)} long play episodes (~1000 steps each)")

# 3. Extract goal-reaching sub-trajectories
goal_reaching_trajs = []        # list of (obs, act, goal) tuples
min_dist_stats = []

for ep_idx, ep_obs in enumerate(episode_obs):          # ep_obs: (T, 28)
    cube_pos = ep_obs[:, 15:18]                        # all cube positions in episode
    
    for t in range(1, len(ep_obs)):                    # start from t=1
        current_pos = cube_pos[t]
        distances = np.linalg.norm(target_positions - current_pos, axis=1)
        min_dist = distances.min()
        closest_goal_id = int(np.argmin(distances))
        
        if min_dist < success_threshold:
            # Found a successful placement at step t
            # Take the trajectory segment that led to this placement
            # (you can choose how long: e.g. last 100 steps, or from last pick)
            segment_obs = ep_obs[max(0, t-100):t+1]     # last 100 steps → success (adjust as needed)
            segment_act = train_dataset['actions'][max(0, t-100):t]
            
            goal_vector = target_positions[closest_goal_id]   # or full 28-dim goal from env
            
            goal_reaching_trajs.append({
                'observations': segment_obs,
                'actions': segment_act,
                'goal': goal_vector,
                'goal_id': closest_goal_id,
                'achieved_dist': min_dist
            })
            
            min_dist_stats.append(min_dist)
            break   # optional: take only the first success per episode

print(f"Extracted {len(goal_reaching_trajs)} goal-reaching trajectories")
print(f"Average distance to goal at success: {np.mean(min_dist_stats):.4f} m")

print(goal_reaching_trajs[0]['goal'])






"""

frames = []
obs, info = env.reset(seed=123)  # may not exactly match logged init state

for a in episode_acts[0]:
    obs, reward, terminated, truncated, info = env.step(a)
    print(reward)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    if terminated or truncated:
        break

media.write_video("demo.mp4", frames, fps=50)
"""


