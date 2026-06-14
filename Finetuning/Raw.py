import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import ogbench as og
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import minari
import sys

from sympy import Max
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from collections import deque
import gymnasium as gym
import gymnasium_robotics  # registers the envs
import numpy as np
import torch
import pickle
from scipy.ndimage import gaussian_filter1d
from Pretrain.Dataset import get_dataset
import ogbench
from Finetuning.Rollout import load_success_trajs
from typing import Optional, List
from torch.utils.data import Dataset
import torch.nn as nn
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras


goals = {'task_1': np.array( [ 0.0,       -1.0,        0.199599]), 
         'task_2': np.array([7.50000000e-01, 8.02418254e-18, 1.99598996e-01]),
         'task_3': np.array([-7.50000000e-01,  1.21832368e-19,  1.99598996e-01]),
         'task_4': np.array([0.75,     2.0,       0.199599]),
         'task_5': np.array([ 0.75,     -2.0,        0.199599])}
    
def check_cube_single_goal_reach(trajs, task_id):   
    goals = {'task_1': np.array( [ 0.0,       -1.0,        0.199599]), 
         'task_2': np.array([7.50000000e-01, 8.02418254e-18, 1.99598996e-01]),
         'task_3': np.array([-7.50000000e-01,  1.21832368e-19,  1.99598996e-01]),
         'task_4': np.array([0.75,     2.0,       0.199599]),
         'task_5': np.array([ 0.75,     -2.0,        0.199599])}
    
    total_dist = 0.0
    for traj in trajs:
           position = traj['observations'][-1][19:22]
           total_dist += np.linalg.norm(position - goals[f"task_{task_id}"])
    average_dist = total_dist/len(trajs)
    print(f"Task {task_id} average distance: {average_dist}")

def check_cube_double_goal_reach(trajs, task_id):   
    goals = {   'task_1': [np.array([0.00000000e+00, 4.40762988e-19, 1.99598996e-01]),  np.array([0.0,   1.0,   0.199599])], 
                'task_2': [np.array([-0.75,      1.0,        0.199599]),  np.array([0.75,     1.0,       0.199599])],
                'task_3': [np.array([0.0,       -2.0,        0.199599]),  np.array([0.0,      2.0,       0.199599])],
                'task_4': [np.array([0.0,        1.0,        0.199599]),  np.array([0.0,       -1.0,        0.199599])],
                'task_5': [np.array([0.00000000e+00,  -3.99397428e-18,   1.99213779e-01]),  np.array([0.00000000e+00,   9.37726514e-18,   5.99039293e-01])]     }
    total_dist = 0.0
    for traj in trajs:
           position_1 = traj['observations'][-1][19:22]
           position_2 = traj['observations'][-1][28:31]
           dist_1 = np.linalg.norm(position_1 - goals[f"task_{task_id}"][0])
           dist_2 = np.linalg.norm(position_2 - goals[f"task_{task_id}"][1])
           total_dist += dist_1 + dist_2
    average_dist = total_dist/len(trajs)
    print(f"Task {task_id} average distance: {average_dist}")





def train_critic_with_planner(
    trajs: List[TrajectoryDict],
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 20000,
    horizon: int = 32,
    gamma: float = 0.99,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    log_every: int = 1000,
):
    @torch.no_grad()
    def _generate_plans_batch(
           s0_planner_norm: np.ndarray,   # (B, d_s) in planner-normalized space
           planner: nn.Module,
           d_s: int, d_a: int, horizon: int,
           steps_T: int, num_karras: int, eta: float,
           device: torch.device,
    ) -> torch.Tensor:
   
        plans = []
        for s0 in s0_planner_norm:
           x = sample_euler_karras(
               s0, planner, d_s, d_a, horizon,
               num_steps=steps_T, num_karras=num_karras, eta=eta, device=device,
           )
           plans.append(x)
        return torch.from_numpy(np.stack(plans, axis=0)).float().to(device)

    
    device = check_device()
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    # ------------------------------------------------------------------ critic
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
    )
    critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------- planner
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=2, timestep_emb_type="fourier",
    ).to(device)
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(planner_proc.stats.obs_mean, device=device, dtype=torch.float32)
    planner_std  = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32,
    )

    # ----------------------------------------------------------- reward model
    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    ).to(device)
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)

    reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std  = torch.as_tensor(np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32)

    # ----------------------------------- critic stats: load once, never save
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset,
        task_id=task_id, step=old_critic_checkpoint,
    )
    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std  = torch.as_tensor(np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32)

    # ---------------------------------------------------- starting-state pool
    s0_pool = np.concatenate([t['observations'] for t in trajs], axis=0).astype(np.float32)

    # ----------------------------------------------------------------- optim
    optimizer = optim.Adam(critic.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32,
    )                                                                       # (n,)
    gamma_n     = gamma ** n

    critic.train()
    running = 0.0

    for k in range(1, num_steps + 1):
        # 1) sample raw start states
        idx    = np.random.randint(0, len(s0_pool), size=batch_size)
        s0_raw = s0_pool[idx]                                                # (B, d_s)

        with torch.no_grad():
            # 2) plan with the diffusion planner
            s0_p  = np.stack([planner_proc.preprocess(o) for o in s0_raw])
            plans = _generate_plans_batch(
                s0_p, planner, obs_dim, act_dim, horizon,
                steps_T, num_karras, eta, device,
            )                                                                # (B, H, d_s+d_a)

            # 3) recover RAW states from planner-norm; actions are already raw
            s_planner = plans[..., :obs_dim]                                 # (B, H, d_s)
            actions   = plans[..., obs_dim:]                                 # (B, H, d_a)
            s_raw     = s_planner * planner_std + planner_mean               # (B, H, d_s)

            # 4) reward model: r̂(s_t, a_t) for t = 0..n-1
            B, H, _ = s_raw.shape
            s_for_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat   = reward_net(
                s_for_r.reshape(B * n, -1),
                actions[:, :n].reshape(B * n, -1),
            ).reshape(B, n)                                                  # (B, n)

            # 5) discounted return + bootstrapped target value
            disc_return  = (gamma_pow_t.unsqueeze(0) * r_hat).sum(dim=1)     # (B,)
            s_n_critic   = (s_raw[:, n] - c_mean) / c_std                    # (B, d_s)
            v_bootstrap  = target_critic(s_n_critic)                         # (B,)
            target_value = disc_return + gamma_n * v_bootstrap               # (B,)

            # 6) input for V_β(s_0)
            s0_critic = (s_raw[:, 0] - c_mean) / c_std                       # (B, d_s)

        # 7) gradient step on V_β
        v_pred = critic(s0_critic)                                           # (B,)
        loss   = F.mse_loss(v_pred, target_value)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 8) Polyak target update
        with torch.no_grad():
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()
        if k % log_every == 0:
            print(f"  step {k:>6}/{num_steps}   loss = {running / log_every:.4f}")
            running = 0.0

    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print("critic saved.")

class CriticDataset_Reward(Dataset):
    def __init__(self, dataset_name: str, 
                       specific_dataset: str, 
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs: List[TrajectoryDict], 
                       target_reward: Optional[float] = None,
                       horizon: int = 32,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005,
                       task_id: Optional[int] = None):
        # ----- gather raw obs/actions to fit stats -----

        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)
        
        #get stats
        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0)+ 1e-8
        if(old_step is not None):
             self.stats = update_critic_stats(dataset_name, specific_dataset, stats, task_id, old_step, momentum)
        else:
             self.stats = stats
        
        device = check_device()
        _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
        reward_state, _, _ = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )
        reward_net = SimpleReward(
            obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
        ).to(device)
        reward_net.load_state_dict(reward_state)
        reward_net.eval()
        for p in reward_net.parameters():
            p.requires_grad_(False)
        reward_stat = get_reward_stats(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )

        transitions = []
        
        for traj in trajs:
            obs = traj['observations'] 
            acts = traj['actions']   
            T_traj = min(len(obs), len(acts))
            
            if T_traj < horizon:
                continue
            
            with torch.no_grad():
                obs_for_r = reward_stat.norm_obs(obs[:T_traj]).astype(np.float32)
                s_t = torch.as_tensor(obs_for_r, dtype=torch.float32, device=device)
                a_t = torch.as_tensor(acts[:T_traj], dtype=torch.float32, device=device)
                rews = reward_net(s_t, a_t).cpu().numpy().astype(np.float32)   # (T_traj,)  
            
            for t in range(len(obs) - horizon):
                 obs_chunk = self.stats.norm_obs(obs[t : t + horizon]).astype(np.float32)
                 rews_chunk = rews[t: min(t+horizon, len(rews))]
                 transitions.append((obs_chunk, rews_chunk))

        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id, new_step)
    
    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(critic_name) + f'_Critic_stats_{str(step)}.pkl'
        stats_dir = f'./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/'
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, 'wb') as f:
              pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __getitem__(self, idx):
        obs_chunk, rews_chunk = self.transitions[idx]
        return (
            torch.tensor(obs_chunk, dtype = torch.float32),
            torch.tensor(rews_chunk, dtype = torch.float32)
        )
    def __len__(self):
        return len(self.transitions)

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews[rews == 1.0] = target_reward
        return rews

class Critic_Buffer_Reward():
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs:  List[TrajectoryDict],
                       horizon: int = 32,
                       gamma: float = 0.99,
                       lam: float = 0.95,
                       task_id: Optional[int] = None,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       momentum: float = 0.005):
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.data = CriticDataset_Reward(dataset_name, 
                       specific_dataset, 
                       reward_hidden_layers,
                       reward_hidden_dim,
                       reward_checkpoint,
                       trajs, 
                       horizon,
                       old_step,  
                       new_step, 
                       momentum,
                       task_id)
       
     
    def obtain_training_data(self, target_critic: nn.Module, batch_size: int, device: str):
        loader = cycle(DataLoader(
            self.data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ))
        obs_chunks, rews_chunks = next(loader)      # (B, T, dim), (B, T)
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]

        with torch.no_grad():
            values = target_critic(obs_chunks)            # (B, T)

            deltas = (
                  rews_chunks[:, :-1]
                  + self.gamma * values[:, 1:]
                   - values[:, :-1]
              )                                             # (B, T-1)

            advantages = torch.zeros(B, T - 1, device=device)
            last_adv = torch.zeros(B, device=device)
            for t in reversed(range(T - 1)):
                last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                advantages[:, t] = last_adv

            value_targets = values[:, 0] + advantages[:, 0]   # (B,)

        return obs_chunks[:, 0], value_targets

def train_critic_with_reward(trajs: List[TrajectoryDict], 
                 dataset_name: str, 
                 specific_dataset: str, 
                 reward_hidden_layers: int,
                 reward_hidden_dim: int,
                 reward_checkpoint: int,
                 critic_hidden_layers: int, 
                 critic_hidden_dim: int, 
                 batch_size, 
                 num_steps, 
                 gamma, lam, horizon, 
                 lr, 
                 min_lr, 
                 tau, 
                 old_step: Optional[int] = None, 
                 new_step: int = 0, 
                 momentum: float = 0.005, 
                 task_id: Optional[int] = None):
    device = check_device()
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers).to(device)
    if(old_step is not None):
        critic_state_dict, _ = get_critic_model(dataset_name, specific_dataset, task_id = task_id, step = old_step)
        critic.load_state_dict(critic_state_dict)
    target_critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    optimizer = optim.Adam(critic.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = num_steps,   # one scheduler step per training step
            eta_min = min_lr
        )
    critic.train()
    buffer = Critic_Buffer_Reward(
                       dataset_name,
                       specific_dataset,
                       reward_hidden_layers,
                       reward_hidden_dim,
                       reward_checkpoint,
                       trajs,
                       horizon,
                       gamma,
                       lam,
                       task_id,
                       old_step,  
                       new_step, 
                       momentum)
    print(f"Training critic for {dataset_name}-{specific_dataset}")
    total_loss = 0.0
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, target_value = buffer.obtain_training_data(target_critic, batch_size, device)
           s = s.to(device)
           target_value = target_value.to(device)

           # Predicted Q-values
           q_pred = critic(s)
           loss = F.smooth_l1_loss(q_pred, target_value, beta = 1.0)
           total_loss += loss.item()

           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
           
           if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/200}")
                total_loss = 0.0
            
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")



"""
env_steps = [0, 1592, 1590, 1600, 1411, 1416, 1600, 1555, 1422, 1600, 1599, 1554]
total_steps = [0]
for i in range(1, len(env_steps)):
    total_steps.append(total_steps[i-1] + env_steps[i])

print(len(total_steps))
print(len(env_steps))
"""



"""
data = get_dataset('cube', 'single-play', task_id = 5)
trajs = data.get_trajectories()
"""



"""
for traj in trajs:
    rews = traj['rewards']
    rews[-1] = 50.0
    print(len(rews))
    rews = rews[len(rews)-50:]
    print(len(rews))
    rews = gaussian_filter1d(rews, sigma = 8.0, mode = 'nearest')
    print(rews)
    exit()
"""
"""
import ogbench
env, dataset, eval_dataset = ogbench.make_env_and_datasets(
                'cube-single-play-singletask-v0', render_mode="rgb_array"
            )

#for i in range(len(dataset['rewards'])):
print((dataset['rewards'].shape))
"""




"""
from Finetuning.Rollout import rollout
from Finetuning.utils import check_device
from Pretrain.utils import set_seed
device = check_device()
env_name = 'cube'
specific_train_dataset = 'single-play'
horizon = 32
checkpoint = 0
set_seed(8)
reward  =  rollout(
               env_name, 
               specific_train_dataset, 
               horizon, 
               steps_T = 200, 
               num_karras = 10, 
               eta = 0.8, 
               episode_length = 3000, 
               checkpoint_steps = checkpoint, 
               render = True,  
               base_seed = 1, 
               task_id = 4,
               continual_rollout = True,
               chunk_size = 32,
               device = device)

print(reward)
"""


from Finetuning.Rollout import load_success_trajs
from Finetuning.utils import TrajectoryDict, train_critic, test_critic
from Pretrain.utils import set_seed

env_name = 'cube'
specific_env = 'single-play'
task_id = 4
step = 0
traj_length = 200

set_seed(1)
"""
trajs = load_success_trajs(env_name, specific_env, task_id, step)
test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 80.0, 
            trajs = trajs,
            task_id = task_id)
"""
trajs = load_success_trajs(env_name, specific_env, task_id, step)
train_critic(trajs, 
             dataset_name = env_name, 
             specific_dataset = specific_env, 
             hidden_layers = 4, 
             hidden_dim = 512, 
             sigma = 3.0,
             batch_size = 256, 
             num_steps = 20000, 
             gamma = 0.99, 
             lam = 0.95, 
             horizon = 32, 
             lr = 5e-05, 
             min_lr = 1e-06, 
             tau = 0.005, 
             old_step = None, 
             new_step = 0, 
             momentum = 0.005, 
             target_reward = 80.0,
             task_id = task_id)

trajs = load_success_trajs(env_name, specific_env, task_id, step)
test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 4, 
            hidden_dim = 512, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 80.0, 
            trajs = trajs,
            task_id = task_id)





"""

from Finetuning.Rollout import get_success_trajs
from Finetuning.utils import train_critic
from Pretrain.utils import set_seed
from Finetuning.utils import test_critic
env_name = 'pointmaze'
specific_env = 'medium'
save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_0.pkl'
with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
trajs = get_success_trajs(trajs)


set_seed(1)
test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 3, 
            hidden_dim = 256, 
            checkpoint_step = 0, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 20.0, 
            trajs = trajs)

with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
trajs = get_success_trajs(trajs)


train_critic(trajs, 
             dataset_name = env_name, 
             specific_dataset = specific_env, 
             hidden_layers = 3, 
             hidden_dim = 256, 
             sigma = 7.0,
             batch_size = 256, 
             num_steps = 5000, 
             gamma = 0.99, 
             lam = 0.95, 
             horizon = 32, 
             lr = 1e-05, 
             min_lr = 1e-06, 
             tau = 0.005, 
             old_step = 0, 
             new_step = 10, 
             momentum = 0.005, 
             target_reward = 20.0)


with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
trajs = get_success_trajs(trajs)


test_critic(dataset_name = env_name, 
            specific_dataset = specific_env, 
            hidden_layers = 3, 
            hidden_dim = 256, 
            checkpoint_step = 10, 
            gamma = 0.99, 
            horizon = 32,  
            sigma = 3.0, 
            target_reward = 20.0, 
            trajs = trajs)

"""

