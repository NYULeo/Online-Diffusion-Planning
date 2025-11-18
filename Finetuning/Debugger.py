import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

"""
    @torch.no_grad()
    def sample_Trajs(self, s0: torch.Tensor) -> Tuple[List[List[torch.Tensor]], float]:
         s0 = s0.to(self.device)
         L = len(s0)
         Trajs = []
         with self.accelerator.split_between_processes(s0) as local_s0:
            local_results = []
            for s0_single in local_s0:
                  s0_single = s0_single.to(self.device)
                  traj = self.sample_Traj(s0_single)  # one trajectory per state
                  local_results.append(traj)
            Trajs.extend(self.accelerator.gather_for_metrics(local_results))
         
         Total_C = 0.0
         for i in range(len(Trajs)):
            x = Trajs[i][len(Trajs[i])-1].squeeze(0)
            x = x.to(self.device)
            c = self.get_C(x)
            Total_C += c
         avg_C = Total_C / L
         self.Lam.update(avg_C)
         return Trajs, avg_C


    def step(self, s0: torch.Tensor) -> float:
        Trajs, avg_C = self.sample_Trajs(s0)
        self.optimizer.zero_grad()
        Loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_reward = 0.0
        for i in range(len(s0)):
            adjoints, reward = self.make_a(Trajs[i])
            loss = self.adjoint_matching_loss(Trajs[i], adjoints)
            Loss += loss
            total_reward += reward
        avg_reward = total_reward / len(s0)
        Loss = Loss / len(s0)
        self.accelerator.backward(Loss)
        self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return Loss.detach().cpu().item(), avg_reward, avg_C
       
    def step(self, s0_batch: Tensor) -> Tuple[float, float, float]:
        # Split batch across processes
        with self.accelerator.split_between_processes(s0_batch) as local_s0:
            local_losses = []
            local_rewards = []
            local_avgCs = []

            for s0 in local_s0:
                traj = self.sample_Traj(s0)
                adjoints, reward = self.make_adjoint(traj)
                loss = self.adjoint_matching_loss(traj, adjoints)
                local_losses.append(loss)
                local_rewards.append(reward)
                # compute C for final state
                final_x = traj[-1].squeeze(0).to(self.device)
                C_val = self.reward_model.get_c(final_x)
                local_avgCs.append(C_val)

        # gather lists from all processes
        all_losses = self.accelerator.gather_for_metrics(local_losses, use_gather_object=True)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards, use_gather_object=True)
        all_avgCs = self.accelerator.gather_for_metrics(local_avgCs, use_gather_object=True)

        if self.accelerator.is_main_process:
            total_loss = float(sum(all_losses) / len(all_losses))
            total_reward = float(sum(all_rewards) / len(all_rewards))
            total_avgC = float(sum(all_avgCs) / len(all_avgCs))
            self.Lam.update(total_avgC)
            return total_loss, total_reward, total_avgC

    


    
    def make_a(self, X):
        X = [x.to(self.device) for x in X]
        steps_T = len(X)
        X_reversed = X[::-1] 
        a = []
        self.reward_model.eval()
        T = X_reversed[0].to(self.device)
        T_squeezed = T.squeeze(0) 
        reward, gradient = self.reward_model(T_squeezed, self.Lam.get_lam())
        gradient_flat = -1 * gradient.view(-1)  # [H*dim]
        a.append(gradient_flat)
        for i in range(steps_T - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now)
            T = X_reversed[i].to(self.device)
            T.requires_grad_(True)
            
            try:
                Jov = self.compute_jacobian_vectorized(T, i)
            except Exception as e:
                print(f"Vectorized Jacobian failed for step {i}, using fallback: {e}")
                Jov = self._compute_jacobian_elementwise(T, i)
            
            current_a = a[i].to(self.device)  # [H*dim]

        
            # Compute: a + dt * (k[i] * a + 2 * k[i] * Jov @ a)
            new_a = current_a + dt * (self.k[i] * current_a + 2 * self.k[i] * (Jov @ current_a))
            a.append(new_a)
            
        a.reverse()
        return a, reward.item()
    """




"""
        all_loss_tensors = self.accelerator.gather(local_loss_tensors)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards, use_gather_object=True)
        #print(f"All loss tensors: {all_loss_tensors}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
             # Compute average reward for logging
             avg_reward = float(sum(all_rewards) / len(all_rewards))
             all_loss_tensors.to(self.device)
             print(f"All loss tensors: {all_loss_tensors}")
             loss_for_backprop = all_loss_tensors.mean()

    
             #all_loss_tensors = [loss_tensor.to(self.device) for loss_tensor in all_loss_tensors]
             #print(f"All loss tensors: {all_loss_tensors}")
             #loss_for_backprop = torch.stack(all_loss_tensors).mean().to(self.device)
             
        

             self.optimizer.zero_grad()
             print(f"Loss before backward: {loss_for_backprop}")
             self.accelerator.backward(loss_for_backprop)
             self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), max_norm=1.0)
             self.optimizer.step()
             self.scheduler.step()
             print(f"Loss after backward: {loss_for_backprop}")

             # For logging compute float of 
             # loss
             avg_loss = loss_for_backprop.detach().item()
             return avg_loss, avg_reward, total_avgC
    
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from accelerate import Accelerator
from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch


def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before everything else
seed = 42
set_seed(seed)





accelerator = Accelerator()
device = accelerator.device
rank = accelerator.process_index

torch.manual_seed(seed + rank)
torch.cuda.manual_seed_all(seed + rank)
np.random.seed(seed + rank)
random.seed(seed + rank)


reward_state_dict, obs_dim, act_dim, reward_name = get_pretrained_reward('pointmaze', 44000, 'medium')
reward_net = Reward(obs_dim, act_dim)
reward_net.load_state_dict(reward_state_dict)
reward_net.eval()
reward_stats = get_pretrained_reward_stats(reward_name)



s = torch.tensor([ [1, 2, 3, 4], [4,5,6, 7], [7,8,9,10], [10, 11, 12, 13]], dtype = torch.float32)
a = torch.tensor([[1,2], [3,4], [5,6], [7,8]], dtype = torch.float32)

class SimpleDataset(Dataset):
    def __init__(self, s, a):
        self.s = s
        self.a = a
    
    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, idx):
        return self.s[idx], self.a[idx]

dataset = SimpleDataset(s, a)
generator = torch.Generator()
generator.manual_seed(seed)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(len(dataloader))
reward_net, dataloader = accelerator.prepare(reward_net, dataloader)
reward_net.to(device)
for batch in dataloader:
     s, a = batch
     s = s.to(device)
     a = a.to(device)
     batch_data = list(zip(s, a))

     base_reward_net = accelerator.unwrap_model(reward_net)
     with accelerator.split_between_processes(batch_data) as local_batch:
        local_rewards = []
        local_s_list = []
        local_a_list = []
        print(f"Local_batch: {len(local_batch)}")
        for s_item, a_item in local_batch:
            local_s_list.append(s_item)
            local_a_list.append(a_item)
        
      
            # Stack into tensors
            local_s = torch.stack(local_s_list)
            local_a = torch.stack(local_a_list)
            
            # Run inference
            with torch.no_grad():
                local_reward = base_reward_net(local_s, local_a)
                print(f'Local_reward: {local_reward}')
            local_rewards.append(local_reward)
            print(local_rewards)
    
     accelerator.wait_for_everyone()
     all_rewards = accelerator.gather_for_metrics(local_rewards, use_gather_object=False)
     
     if accelerator.is_main_process:
        # Concatenate all rewards
        
        #all_rewards = [r.to(device) if isinstance(r, torch.Tensor) else r for r in all_rewards]
        print(f"All_reward: {all_rewards}")
        #print(f"Collected rewards: mean={all_rewards.mean().item():.4f}, shape={all_rewards.shape}")
     exit()




import minari
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats

# ================== Load dataset & env ==================
dataset = minari.load_dataset('D4RL/pointmaze/medium-v2', download=True)
env = dataset.recover_environment()
env = env.unwrapped  # PointMaze2DEnv

# ================== Load your pretrained reward model ==================
reward_model_state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', 44000, 'medium')
reward_model = Reward(obs_dim, act_dim)
reward_model.load_state_dict(reward_model_state_dict)
reward_model.eval()

stats = get_pretrained_reward_stats(name)  # contains .norm_obs() method

# ================== Grid setup ==================
resolution = 200
x_min, x_max = -1, 11
y_min, y_max = -1, 11

x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

fixed_goal = np.array([9.0, 9.0])   # standard goal for medium-v2

obs_grid = np.column_stack([
    X.ravel(),
    Y.ravel(),
    np.full_like(X.ravel(), fixed_goal[0]),
    np.full_like(X.ravel(), fixed_goal[1])
]).astype(np.float32)   # (N, 4)

obs_grid = torch.from_numpy(obs_grid)

# ================== Max-action search ==================
n_act = 21  # slightly denser = smoother heatmap
acts = np.linspace(-1.0, 1.0, n_act)
AX, AY = np.meshgrid(acts, acts)
candidate_acts = np.column_stack([AX.ravel(), AY.ravel()]).astype(np.float32)
candidate_acts = torch.from_numpy(candidate_acts).float()

N, A = len(obs_grid), len(candidate_acts)

with torch.no_grad():
    # Repeat observations and actions
    obs_rep = obs_grid.unsqueeze(1).repeat(1, A, 1).reshape(N * A, 4)   # (N*A, 4)
    act_rep = candidate_acts.unsqueeze(0).repeat(N, 1, 1).reshape(N * A, 2)  # (N*A, 2)

    obs_rep = torch.from_numpy(obs_rep) if not torch.is_tensor(obs_rep) else obs_rep
    act_rep = torch.from_numpy(act_rep) if not torch.is_tensor(act_rep) else act_rep

    # === CRITICAL: apply your exact normalization ===
    obs_rep = stats.norm_obs(obs_rep)   # <-- this is what your model expects!
    act_rep = act_rep.float()
    obs_rep = obs_rep.float()

    rewards_flat = reward_model(obs_rep, act_rep)           # (N*A,)
    rewards_grid = rewards_flat.reshape(N, A)
    best_rewards = rewards_grid.max(dim=1)[0].cpu().numpy()

reward_map = best_rewards.reshape(resolution, resolution)

# ================== Plotting ==================
plt.figure(figsize=(12, 11))
cmap = LinearSegmentedColormap.from_list('reward', ['navy', 'royalblue', 'white', 'orange', 'red'], N=256)
im = plt.imshow(reward_map, extent=[x_min, x_max, y_min, y_max],
                cmap=cmap, origin='lower', interpolation='bilinear')

# === FIXED: correct attribute for walls in current d4rl-pointmaze2d ===
for wall in env.maze.walls:        # <-- this is the correct one (not layout_walls)
    (x0, y0), (x1, y1) = wall
    plt.plot([x0, x1], [y0, y1], color='black', linewidth=4, solid_capstyle='butt')

# Start & goal
plt.scatter(1.0, 1.0, c='lime', s=400, marker='o', edgecolor='black', linewidth=2, zorder=5, label='Start')
plt.scatter(fixed_goal[0], fixed_goal[1], c='yellow', s=500, marker='*', edgecolor='black', linewidth=3, zorder=5, label='Goal')

plt.colorbar(im, label='Reward (max over actions)', shrink=0.82)
plt.title(f'Reward Model Heatmap — pointmaze/medium-v2\nIteration 44000 | Goal @ (9,9)', fontsize=16)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')
plt.axis('equal')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.show()

"""

# save_reward_heatmap.py
import minari
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')                # ← crucial: no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats

# ================== Config ==================
DATASET_NAME = 'D4RL/pointmaze/medium-v2'
CHECKPOINT_STEP = 44000
GOAL = np.array([9.0, 9.0])                     # change if you want another goal
RESOLUTION = 400                                # 400×400 = publication quality
OUTPUT_PATH = f"reward_heatmap_medium_step{CHECKPOINT_STEP}_goal{GOAL[0]}_{GOAL[1]}.png"
BATCH_SIZE = 8192                               # ← prevents OOM on low-memory machines

print(f"Loading dataset {DATASET_NAME}...")
dataset = minari.load_dataset(DATASET_NAME, download=True)
env = dataset.recover_environment().unwrapped

print("Loading reward model...")
reward_model_state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', CHECKPOINT_STEP, 'medium')
reward_model = Reward(obs_dim, act_dim)
reward_model.load_state_dict(reward_model_state_dict)
reward_model.eval()
stats = get_pretrained_reward_stats(name)

# ================== Grid ==================
x_min, x_max = -1, 11
y_min, y_max = -1, 11
x = np.linspace(x_min, x_max, RESOLUTION)
y = np.linspace(y_min, y_max, RESOLUTION)
X, Y = np.meshgrid(x, y)

obs_base = np.column_stack([
    X.ravel(),
    Y.ravel(),
    np.full(RESOLUTION*RESOLUTION, GOAL[0]),
    np.full(RESOLUTION*RESOLUTION, GOAL[1])
]).astype(np.float32)

# ================== Dense action grid (for max-action reward) ==================
n_act = 25
acts = np.linspace(-1.0, 1.0, n_act)
AX, AY = np.meshgrid(acts, acts)
candidate_acts = np.column_stack([AX.ravel(), AY.ravel()]).astype(np.float32)
acts_tensor = torch.from_numpy(candidate_acts)

print("Evaluating reward model (this may take 10-30 seconds)...")
best_rewards = np.full(RESOLUTION*RESOLUTION, -1e10)

with torch.no_grad():
    for i in range(0, len(obs_base), BATCH_SIZE):
        batch_obs = torch.from_numpy(obs_base[i:i+BATCH_SIZE])
        batch_size = len(batch_obs)

        # Repeat actions for this batch
        obs_rep = batch_obs.unsqueeze(1).repeat(1, len(acts_tensor), 1).reshape(-1, 4)
        act_rep = acts_tensor.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, 2)

        obs_rep = stats.norm_obs(obs_rep)
        obs_rep = obs_rep.float()
        act_rep = act_rep.float()
        rewards = reward_model(obs_rep, act_rep).cpu().numpy()
        rewards = rewards.reshape(batch_size, len(acts_tensor))

        best_in_batch = rewards.max(axis=1)
        best_rewards[i:i+batch_size] = np.maximum(best_rewards[i:i+batch_size], best_in_batch)

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   → processed {i}/{len(obs_base)} positions")

reward_map = best_rewards.reshape(RESOLUTION, RESOLUTION)
print("Evaluation done. Plotting...")

# ================== Plot & Save ==================
plt.figure(figsize=(12, 11), dpi=300)
cmap = LinearSegmentedColormap.from_list('reward', ['navy', 'blue', 'white', 'orange', 'red'], N=256)
im = plt.imshow(reward_map, extent=[x_min, x_max, y_min, y_max],
                cmap=cmap, origin='lower', interpolation='bilinear')

# Walls
for wall in env.maze.walls:
    (x0, y0), (x1, y1) = wall
    plt.plot([x0, x1], [y0, y1], color='black', linewidth=3, solid_capstyle='butt')

# Markers
plt.scatter(1.0, 1.0, c='lime', s=500, marker='o', edgecolor='black', linewidth=3, zorder=5)
plt.scatter(GOAL[0], GOAL[1], c='yellow', s=700, marker='*', edgecolor='black', linewidth=3, zorder=5)

plt.colorbar(im, shrink=0.8, label='Max-action reward')
plt.title(f'Reward Model Heatmap\npointmaze/medium-v2 | step {CHECKPOINT_STEP} | goal @ ({GOAL[0]}, {GOAL[1]})', fontsize=16)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()

print(f"Saving to {OUTPUT_PATH} ...")
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()
print("Done! Heatmap saved.")
