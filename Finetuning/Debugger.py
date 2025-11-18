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

