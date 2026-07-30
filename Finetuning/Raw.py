from pkgutil import get_data
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
from Pretrain.utils import ema_smooth
from Pretrain.Dataset import get_dataset
import ogbench
from Finetuning.Rollout import load_success_trajs
from Finetuning.utils import reward_processor
from typing import Optional, List
from torch.utils.data import Dataset
import torch.nn as nn
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras

def check_increase(rewards):
    for i in range(1, len(rewards)):
        if( rewards[i] < rewards[i-1]):
            return False
    
    return True


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

data = get_dataset('cube', 'double-play', task_id = 4, traj_length = 500)
trajs = data.get_trajectories()
print(gaussian_filter1d(trajs[16]['rewards']*50, sigma = 2.0, mode="nearest"))
exit()
rews = [-50, -50, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
rews = gaussian_filter1d(rews, sigma = 2.0, mode="nearest")
print (rews)
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





"""
class TotalReward_Critic(nn.Module):
    def __init__(
        self,
        device,
        config: RewardConfig,
        dataset_name: str,
        specific_dataset: str,
        reward_checkpoint: int,
        kernel_checkpoint: int,
        critic_checkpoint: int,
        task_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------ reward
        reward_state_dict, obs_dim, act_dim = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id
        )
        self.config.device = device
        self.reward_net = SimpleReward(
            obs_dim,
            act_dim,
            self.config.hidden_dim_reward,
            self.config.num_hidden_layers_reward,
        ).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()

        # ------------------------------------------------------------------ kernels
        self.kernels = []
        self.config.delta = F.softplus(
            torch.tensor(0.0, requires_grad=False), beta=self.config.beta
        ).to(self.config.device)

        kernel_state_dicts, obs_dim, act_dim = get_kernel(
            dataset_name, specific_dataset, kernel_checkpoint
        )
        if self.config.type_kernel == "robust":
            for sd in kernel_state_dicts:
                kernel_net = RobustTransitionKernel(
                    obs_dim,
                    act_dim,
                    self.config.num_hidden_layers_kernel,
                    self.config.hidden_dim_kernel,
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)
        else:
            for sd in kernel_state_dicts:
                kernel_net = MoGTransitionKernel(
                    obs_dim,
                    act_dim,
                    self.config.kernel_num_modes,
                    self.config.num_hidden_layers_kernel,
                    self.config.hidden_dim_kernel,
                    noise_floor=self.config.kernel_noise_floor,
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)

        # ------------------------------------------------------------------ critic
        critic_state_dict, critic_obs_dim = get_critic_model(
            dataset_name, specific_dataset, task_id, critic_checkpoint
        )
        self.critic = Critic(
            critic_obs_dim,
            self.config.hidden_dim_critic,
            self.config.num_hidden_layers_critic,
        ).to(self.config.device)
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()

        # ------------------------------------------------------------------ stats
        self.reward_stat = get_reward_stats(
            dataset_name, specific_dataset, reward_checkpoint, task_id
        )
        self.kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_checkpoint
        )
        self.critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id, 0
        )
        self.q_stats = get_Q_stats(
            dataset_name, specific_dataset, task_id, critic_checkpoint
        )

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim

        if not self.config.explore:
            self.config.gamma = 0.0

    # ---------------------------------------------------------------------- helpers
    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        if self.config.type_kernel == "robust":
            total = torch.tensor([0.0], device=self.config.device, requires_grad=True)
            for i in range(len(self.kernels)):
                mu, log_std = self.kernels[i](s, a)
                lp = self.kernels[i].log_prob(s_next, mu, log_std)
                total = total + lp
            avg = total / len(self.kernels)
        else:
            avg = compute_log_density_mog(self.kernels, s, a, s_next)
        x = self.config.min_log_prob - avg
        c = F.softplus(x, beta=self.config.beta)
        return c

    def reward_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.reward_stat.norm_obs(s_n)
        return torch.tensor(
            s_n, dtype=torch.float32, device=self.config.device, requires_grad=True
        )

    def kernel_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.kernel_stat.norm_obs(s_n)
        return torch.tensor(
            s_n, dtype=torch.float32, device=self.config.device, requires_grad=True
        )

    def critic_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.critic_stat.norm_obs(s_n)
        return torch.tensor(
            s_n, dtype=torch.float32, device=self.config.device, requires_grad=True
        )

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, self.config.d_s + self.config.d_a, device=self.config.device)
        A = torch.zeros(H, self.config.d_s + self.config.d_a, device=self.config.device)
        S[i, : self.config.d_s] = s_grad
        A[i, self.config.d_s :] = a_grad
        if s_next_grad is not None:
            S_next = torch.zeros(
                H, self.config.d_s + self.config.d_a, device=self.config.device
            )
            S_next[i + 1, : self.config.d_s] = s_next_grad
            return S, A, S_next
        return S, A

    def makeGrad_Critic(self, H, s_grad, i):
        S = torch.zeros(H, self.config.d_s + self.config.d_a, device=self.config.device)
        S[i, : self.config.critic_d_s] = s_grad
        return S

    def get_c(self, x):
        H, _ = x.shape
        C = torch.tensor(0.0, device=self.config.device)
        for i in range(H - 1):
            s = x[i, : self.config.d_s]
            a = x[i, self.config.d_s :].unsqueeze(0)
            s_next = x[i + 1, : self.config.d_s]
            s_k = self.kernel_processor(s).unsqueeze(0)
            s_next_k = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_k, a, s_next_k)
            C = C + c.squeeze(0)
        C = C / (H - 1) - self.config.delta
        return C

    # ---------------------------------------------------------------------- core
    def _compute_gae_style_return(
        self, x: torch.Tensor, lam: float, with_grad: bool = False
    ):
        H, D = x.shape
        device = self.config.device
        gamma = self.config.critic_gamma
        n = H

        # -------------------- pre-compute rewards and critic values --------------------
        rs = []          # r[0] ... r[n-2]
        r_grads = []     # (s_grad, a_grad) or None
        vs = []          # denormalised V[0] ... V[n-1]
        v_grads = []     # s_grad or None

        for i in range(n):
            s = x[i, : self.config.d_s]

            # reward (only needed for i = 0 ... n-2)
            if i < n - 1:
                a = x[i, self.config.d_s :]
                s_r = self.reward_processor(s).unsqueeze(0)
                a_t = a.unsqueeze(0)
                if with_grad:
                    s_r = s_r.requires_grad_(True)
                    a_t = a_t.requires_grad_(True)
                r = self.reward_net(s_r, a_t).squeeze(0)
                rs.append(r)
                if with_grad:
                    grads = torch.autograd.grad(
                        outputs=r,
                        inputs=(s_r, a_t),
                        grad_outputs=torch.ones_like(r),
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,
                    )
                    inv_std = torch.tensor(
                        1.0
                        / np.maximum(
                            self.reward_stat.obs_std, self.reward_stat.std_floor
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    r_s = grads[0].squeeze(0) * inv_std
                    r_a = grads[1].squeeze(0)
                    r_grads.append((r_s, r_a))
                else:
                    r_grads.append(None)

            # critic value (all states)
            s_c = self.critic_processor(s[: self.config.critic_d_s]).unsqueeze(0)
            if with_grad:
                s_c = s_c.requires_grad_(True)
            v = self.critic(s_c).squeeze(0)
            v_denorm = self.q_stats.Q_std * v + self.q_stats.Q_mean
            vs.append(v_denorm)
            if with_grad:
                grads = torch.autograd.grad(
                    outputs=v,
                    inputs=(s_c,),
                    grad_outputs=torch.ones_like(v),
                    create_graph=False,
                    retain_graph=False,
                )
                inv_std = torch.tensor(
                    1.0
                    / np.maximum(
                        self.critic_stat.obs_std, self.critic_stat.std_floor
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                v_s = grads[0].squeeze(0) * inv_std * self.q_stats.Q_std
                v_grads.append(v_s)
            else:
                v_grads.append(None)

        # -------------------- λ-weighted multi-step returns (each G averaged) --------------------
        plan_return = torch.tensor(0.0, device=device)
        coeff_r = [torch.tensor(0.0, device=device) for _ in range(n - 1)]
        coeff_v = [torch.tensor(0.0, device=device) for _ in range(n)]

        gae_lam = self.config.gae_lam

        if abs(gae_lam - 1.0) < 1e-8 or n == 1:
            # pure (n-1)-step averaged return
            k = max(n - 1, 1)
            partial = torch.tensor(0.0, device=device)
            for t in range(n - 1):
                partial = partial + (gamma ** t) * rs[t]
                coeff_r[t] = (gamma ** t) / k
            if n >= 1:
                partial = partial + (gamma ** (n - 1)) * vs[n - 1]
                coeff_v[n - 1] = (gamma ** (n - 1)) / k
            plan_return = partial / k
        else:
            w = 1.0 - gae_lam
            weight_sum = 0.0
            for h in range(2, n + 1):          # h=2 → k=1, h=3 → k=2, ...
                k = h - 1
                partial = torch.tensor(0.0, device=device)
                for t in range(k):
                    partial = partial + (gamma ** t) * rs[t]
                    coeff_r[t] = coeff_r[t] + w * (gamma ** t) / k
                partial = partial + (gamma ** k) * vs[k]
                coeff_v[k] = coeff_v[k] + w * (gamma ** k) / k

                plan_return = plan_return + w * (partial / k)
                weight_sum += w
                w *= gae_lam

            Z = max(weight_sum, 1e-8)
            plan_return = plan_return / Z
            for t in range(n - 1):
                coeff_r[t] = coeff_r[t] / Z
            for j in range(n):
                coeff_v[j] = coeff_v[j] / Z

        # -------------------- constraint terms (original meaning of lam) --------------------
        total_c = torch.tensor(0.0, device=device)
        c_grads = []
        for i in range(n - 1):
            s = x[i, : self.config.d_s]
            a = x[i, self.config.d_s :].unsqueeze(0)
            s_next = x[i + 1, : self.config.d_s]
            s_k = self.kernel_processor(s).unsqueeze(0)
            s_next_k = self.kernel_processor(s_next).unsqueeze(0)
            if with_grad:
                s_k = s_k.requires_grad_(True)
                a = a.requires_grad_(True)
                s_next_k = s_next_k.requires_grad_(True)
            c = self.sigmoid(s_k, a, s_next_k).squeeze(0)
            total_c = total_c + c
            if with_grad:
                grads = torch.autograd.grad(
                    outputs=c,
                    inputs=(s_k, a, s_next_k),
                    grad_outputs=torch.ones_like(c),
                    create_graph=True,
                    retain_graph=True,
                )
                inv_std = torch.tensor(
                    1.0
                    / np.maximum(
                        self.kernel_stat.obs_std, self.kernel_stat.std_floor
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                c_s = grads[0].squeeze(0) * inv_std
                c_a = grads[1].squeeze(0)
                c_s_next = grads[2].squeeze(0) * inv_std
                c_grads.append((c_s, c_a, c_s_next))
            else:
                c_grads.append(None)

        mean_c = total_c / (n - 1) if n > 1 else torch.tensor(0.0, device=device)
        total_reward = plan_return - lam * (mean_c - self.config.delta)

        if not with_grad:
            return total_reward, None

        # -------------------- assemble full trajectory gradient --------------------
        gradient = torch.zeros(H, D, device=device)

        # from the r_t terms
        for t in range(n - 1):
            if coeff_r[t] != 0:
                r_s, r_a = r_grads[t]
                g_s, g_a = self.makeGrad(H, r_s, r_a, t)
                gradient = gradient + coeff_r[t] * (g_s + g_a)

        # from the V(s_j) terms
        for j in range(n):
            if coeff_v[j] != 0:
                v_s = v_grads[j]
                g_v = self.makeGrad_Critic(H, v_s, j)
                gradient = gradient + coeff_v[j] * g_v

        # from the constraint terms
        for i in range(n - 1):
            c_s, c_a, c_s_next = c_grads[i]
            g_s, g_a, g_s_next = self.makeGrad(H, c_s, c_a, i, c_s_next)
            gradient = gradient - (lam / (n - 1)) * (g_s + g_a + g_s_next)

        return total_reward, gradient

    # ---------------------------------------------------------------------- public API
    def predict(self, x: torch.Tensor, lam: float):
        total_reward, _ = self._compute_gae_style_return(
            x, lam, with_grad=False
        )
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        total_reward, gradient = self._compute_gae_style_return(
            x, lam, with_grad=True
        )
        return total_reward, gradient
"""
