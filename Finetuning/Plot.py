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


"""

# 1. Get the 5 official target positions
target_positions = []
for task_id in range(1, 6):
    obs, info = env.reset(options={"task_id": task_id})
    goal_cube_pos = info['goal'][15:18]   # cube position only
    goal_xyzs = env.unwrapped.task_infos[task_id - 1]["goal_xyzs"]
    target_positions.append(goal_cube_pos)
target_positions = np.array(target_positions)    # shape: (5, 3)

success_threshold = 0.05   # official-style tolerance (5 cm)


# 2. Split the flat dataset into long play episodes (your code, fixed)
episode_obs = []
last_start = 0

for i in range(len(train_dataset['observations'])):
    if train_dataset['terminals'][i] == 1 or i == len(train_dataset['observations']) - 1:
        obs_slice = train_dataset['observations'][last_start:i+1]   # include terminal
        act_slice = train_dataset['actions'][last_start:i+1]
        episode_obs.append(np.array(obs_slice))
        last_start = i + 1
        if(i != len(train_dataset['observations']) - 1 and ( np.array_equal(train_dataset['next_observations'][i], train_dataset['observations'][i+1]))):
            print('wrong')
            exit()

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




import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (11, 6.5)

# Generate x values (same density as your plot)
x = np.linspace(1, 30, 30)


normalized_scores = [
    14.88,  # Round 1
    21.06,  # Round 2
    24.54,  # Round 3
    24.69,  # Round 4
    31.67,  # Round 5
    31.69,  # Round 6
    31.52,  # Round 7
    32.32,  # Round 8
    33.20,  # Round 9
    32.97,  # Round 10
    32.40,  # Round 11
    32.28,  # Round 12
    33.94,  # Round 13
    33.68,  # Round 14
    34.04,  # Round 15
    34.14,  # Round 16
    33.14,  # Round 17
    33.02,  # Round 18
    40.94,  # Round 19
    44.28,  # Round 20
    45.37,  # Round 21
    46.57,  # Round 22
    47.47,  # Round 23
    47.92,  # Round 24
    48.17,  # Round 25
    48.21,  # Round 26
    48.25,  # Round 27
    48.30,  # Round 28
    48.30,  # Round 29
    48.32   # Round 30
]
import torch
print(torch.backends.mps.is_available())   # Should be True
print(torch.backends.mps.is_built())       # Should be True
exit()
fig, ax = plt.subplots()
ax.plot(x, normalized_scores, color='#C44E9B', linewidth=3.2, marker='o', markersize=8, markeredgecolor='white', markeredgewidth=0.8, label='Our Method', zorder=5)
#ax.fill_between(x, y1 - error1, y1 + error1, color='#C44E9B', alpha=0.22)
#ax.plot(x, y2, color='#4A90E2', linewidth=3.2, marker='s', markersize=8, markeredgecolor='white', markeredgewidth=0.8, label='Series 2', zorder=5)
#ax.fill_between(x, y2 - error2, y2 + error2, color='#4A90E2', alpha=0.25)

indices = np.linspace(18, len(x)-6, 10, dtype=int)
#ax.scatter(x[indices], y2[indices], color='#4A90E2', s=85, marker='s', edgecolor='white', linewidth=0.6, zorder=6)        # squares
#ax.scatter(x[indices+2], y2[indices+2], color='#4A90E2', s=95, marker='^', edgecolor='white', linewidth=0.6, zorder=6)        # triangles  (fixed!)
#ax.scatter(x[indices+5], y2[indices+5], color='#4A90E2', s=100, marker='*', edgecolor='white', linewidth=0.5, zorder=6)        # stars

# Vertical gray shaded region
#ax.axvspan(0.82, 1.18, color='gray', alpha=0.12, zorder=1)

# Labels and limits
ax.set_xlim(1, 30)
ax.set_ylim(0, 50)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# Legend
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()

