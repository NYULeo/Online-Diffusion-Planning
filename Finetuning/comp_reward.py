import sys
import os

from torch.optim.optimizer import required
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import torch
import torch.nn as nn
from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_total_mahalanobis_score
from Pretrain.Critic.nets import Critic
from Finetuning.utils import get_reward_model, get_kernel, get_reward_stats, get_kernel_stats, get_critic_model, get_critic_stats
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np



@dataclass
class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    max_mahalanobis_score: float
    #min_log_prob: float
    explore: bool = True
    gamma: float = 0.8
    critic_gamma: float = 0.99
    device = None
    d_s: int = 0 
    d_a: int = 0
    num_hidden_layers_kernel: int = 2
    hidden_dim_kernel: int = 256
    num_hidden_layers_reward: int = 1
    hidden_dim_reward: int = 128
    num_hidden_layers_critic: int = 1
    hidden_dim_critic: int = 128
    critic_d_s: int = 0
    delta: Optional[float] = None 
    

class TotalReward(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int):
        super().__init__()
        self.config = config
        self.config.device = device

        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.reward_net = SimpleReward(
            obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward
        ).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        self.kernels = []
        for sd in kernel_state_dicts:
            kernel_net = RobustTransitionKernel(
                obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
            ).to(self.config.device)
            kernel_net.load_state_dict(sd)
            kernel_net.eval()
            self.kernels.append(kernel_net)

        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        if not self.config.explore:
            self.config.gamma = 0.0

        self.config.delta = F.softplus(torch.tensor(0.0, device=self.config.device), beta=self.config.beta)

        # Cache normalization tensors once
        reward_obs_mean = np.asarray(self.reward_stat.obs_mean, dtype=np.float32)
        reward_obs_inv_std = 1.0 / np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)
        kernel_obs_mean = np.asarray(self.kernel_stat.obs_mean, dtype=np.float32)
        kernel_obs_inv_std = 1.0 / np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor)

        self.reward_obs_mean_t = torch.as_tensor(reward_obs_mean, dtype=torch.float32, device=self.config.device)
        self.reward_obs_inv_std_t = torch.as_tensor(reward_obs_inv_std, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_mean_t = torch.as_tensor(kernel_obs_mean, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_inv_std_t = torch.as_tensor(kernel_obs_inv_std, dtype=torch.float32, device=self.config.device)

    def get_beta(self):
        return self.config.beta

    def reward_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.reward_obs_mean_t) * self.reward_obs_inv_std_t

    def kernel_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.kernel_obs_mean_t) * self.kernel_obs_inv_std_t

    def sigmoid(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        lps = []
        for k in self.kernels:
            mu, log_std = k(s, a)
            lps.append(k.log_prob(s_next, mu, log_std))
        avg = torch.stack(lps, dim=0).mean(dim=0)
        x = self.config.min_log_prob - avg
        return F.softplus(x, beta=self.config.beta)

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        S[i, :self.config.d_s] = s_grad
        A[i, self.config.d_s:] = a_grad
        if s_next_grad is not None:
            S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
            S_next[i + 1, :self.config.d_s] = s_next_grad
            return S, A, S_next
        return S, A

    def get_c(self, x):
        H, _ = x.shape
        C = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C = C + c.squeeze(0)
        C = C / (H - 1)
        C = C - self.config.delta
        return C

    def predict(self, x: torch.Tensor, lam: float):
        H, _ = x.shape
        total_reward = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward = total_reward + (1.0 / H) * r.squeeze(0) - lam * ((1.0 / (H - 1)) * c.squeeze(0))

        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        s_norm_reward = self.reward_processor(s).unsqueeze(0)
        r = self.reward_net(s_norm_reward, a)
        total_reward = total_reward + (1.0 / H) * r.squeeze(0)
        total_reward = total_reward + lam * self.config.delta
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = x.new_zeros(())
        gradient = x.new_zeros((H, D))

        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
            a = a.requires_grad_(True)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads_r = torch.autograd.grad(
                outputs=r,
                inputs=(s_norm_reward, a),
                grad_outputs=torch.ones_like(r),
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )
            r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
            r_a = grads_r[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)

            grads_c = torch.autograd.grad(
                outputs=c,
                inputs=(s_norm_kernel, a, s_next_norm_kernel),
                grad_outputs=torch.ones_like(c),
                create_graph=True,
                retain_graph=True
            )
            c_s = grads_c[0].squeeze(0) * self.kernel_obs_inv_std_t
            c_a = grads_c[1].squeeze(0)
            c_s_next = grads_c[2].squeeze(0) * self.kernel_obs_inv_std_t
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient = gradient + (1.0 / H) * (r_s_grad + r_a_grad) - lam * (1.0 / (H - 1)) * (c_s_grad + c_a_grad + c_s_next_grad)
            total_reward = total_reward + (1.0 / H) * r.squeeze(0) - lam * ((1.0 / (H - 1)) * c.squeeze(0))

        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = a.requires_grad_(True)
        r = self.reward_net(s_norm_reward, a)

        grads_r = torch.autograd.grad(
            outputs=r,
            inputs=(s_norm_reward, a),
            grad_outputs=torch.ones_like(r),
            create_graph=False,
            retain_graph=False
        )
        r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
        r_a = grads_r[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H - 1)

        gradient = gradient + (1.0 / H) * (r_s_grad + r_a_grad)
        total_reward = total_reward + (1.0 / H) * r.squeeze(0)
        total_reward = total_reward + lam * self.config.delta
        return total_reward, gradient


class TotalReward_Critic(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int):
        super().__init__()
        self.config = config
        self.config.device = device

        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.reward_net = SimpleReward(
            obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward
        ).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()

        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, critic_checkpoint)
        self.critic = Critic(
            critic_obs_dim, self.config.hidden_dim_critic, self.config.num_hidden_layers_critic
        ).to(self.config.device)
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        self.kernels = []
        for sd in kernel_state_dicts:
            kernel_net = RobustTransitionKernel(
                obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
            ).to(self.config.device)
            kernel_net.load_state_dict(sd)
            kernel_net.eval()
            self.kernels.append(kernel_net)

        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, critic_checkpoint)

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim
        if not self.config.explore:
            self.config.gamma = 0.0

        self.config.delta = F.softplus(torch.tensor(0.0, device=self.config.device), beta=self.config.beta)

        # Cache normalization tensors once
        reward_obs_mean = np.asarray(self.reward_stat.obs_mean, dtype=np.float32)
        reward_obs_inv_std = 1.0 / np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)
        kernel_obs_mean = np.asarray(self.kernel_stat.obs_mean, dtype=np.float32)
        kernel_obs_inv_std = 1.0 / np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor)
        critic_obs_mean = np.asarray(self.critic_stat.obs_mean, dtype=np.float32)
        critic_obs_inv_std = 1.0 / np.maximum(self.critic_stat.obs_std, self.critic_stat.std_floor)

        self.reward_obs_mean_t = torch.as_tensor(reward_obs_mean, dtype=torch.float32, device=self.config.device)
        self.reward_obs_inv_std_t = torch.as_tensor(reward_obs_inv_std, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_mean_t = torch.as_tensor(kernel_obs_mean, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_inv_std_t = torch.as_tensor(kernel_obs_inv_std, dtype=torch.float32, device=self.config.device)
        self.critic_obs_mean_t = torch.as_tensor(critic_obs_mean, dtype=torch.float32, device=self.config.device)
        self.critic_obs_inv_std_t = torch.as_tensor(critic_obs_inv_std, dtype=torch.float32, device=self.config.device)

    def get_beta(self):
        return self.config.beta

    def reward_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.reward_obs_mean_t) * self.reward_obs_inv_std_t

    def kernel_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.kernel_obs_mean_t) * self.kernel_obs_inv_std_t

    def critic_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.critic_obs_mean_t) * self.critic_obs_inv_std_t

    def sigmoid(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        lps = []
        for k in self.kernels:
            mu, log_std = k(s, a)
            lps.append(k.log_prob(s_next, mu, log_std))
        avg = torch.stack(lps, dim=0).mean(dim=0)
        x = self.config.min_log_prob - avg
        return F.softplus(x, beta=self.config.beta)

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        S[i, :self.config.d_s] = s_grad
        A[i, self.config.d_s:] = a_grad
        if s_next_grad is not None:
            S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
            S_next[i + 1, :self.config.d_s] = s_next_grad
            return S, A, S_next
        return S, A

    def makeGrad_Critic(self, H, s_grad, i):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        S[i, :self.config.critic_d_s] = s_grad
        return S

    def get_c(self, x):
        H, _ = x.shape
        C = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C = C + c.squeeze(0)
        C = C / (H - 1)
        C = C - self.config.delta
        return C

    def predict(self, x: torch.Tensor, lam: float):
        # Keep your existing semantics (un-normalized running sum + terminal critic)
        H, _ = x.shape
        total_reward = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward = total_reward + r.squeeze(0) - lam * c.squeeze(0)

        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        r = self.reward_net(self.reward_processor(s).unsqueeze(0), a)

        final_s_critic = x[H - 1, :self.config.critic_d_s]
        v = self.critic(self.critic_processor(final_s_critic).unsqueeze(0))

        total_reward = total_reward + r.squeeze(0) + (self.config.critic_gamma ** (H - 1)) * v.squeeze(0)
        total_reward = total_reward + lam * self.config.delta
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        # Keep existing semantics from your class:
        # - per-step reward/constraint gradients
        # - terminal critic gradient contribution
        H, D = x.shape
        total_reward = x.new_zeros(())
        gradient = x.new_zeros((H, D))

        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
            a = a.requires_grad_(True)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads_r = torch.autograd.grad(
                outputs=r,
                inputs=(s_norm_reward, a),
                grad_outputs=torch.ones_like(r),
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )
            r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
            r_a = grads_r[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)

            grads_c = torch.autograd.grad(
                outputs=c,
                inputs=(s_norm_kernel, a, s_next_norm_kernel),
                grad_outputs=torch.ones_like(c),
                create_graph=True,
                retain_graph=True
            )
            c_s = grads_c[0].squeeze(0) * self.kernel_obs_inv_std_t
            c_a = grads_c[1].squeeze(0)
            c_s_next = grads_c[2].squeeze(0) * self.kernel_obs_inv_std_t
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient = gradient + (r_s_grad + r_a_grad) - lam * (c_s_grad + c_a_grad + c_s_next_grad)
            total_reward = total_reward + r.squeeze(0) - lam * c.squeeze(0)

        # last-step reward grad (kept)
        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = a.requires_grad_(True)
        r = self.reward_net(s_norm_reward, a)

        grads_r = torch.autograd.grad(
            outputs=r,
            inputs=(s_norm_reward, a),
            grad_outputs=torch.ones_like(r),
            create_graph=False,
            retain_graph=False
        )
        r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
        r_a = grads_r[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H - 1)

        # terminal critic grad
        final_s_critic = x[H - 1, :self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic).unsqueeze(0).requires_grad_(True)
        v = self.critic(final_s_norm_critic)
        grad_v = torch.autograd.grad(
            outputs=v,
            inputs=final_s_norm_critic,
            grad_outputs=torch.ones_like(v),
            create_graph=False,
            retain_graph=False
        )[0]
        v_s = grad_v.squeeze(0) * self.critic_obs_inv_std_t
        grad_critic = self.makeGrad_Critic(H, v_s, H - 1)

        # Preserve your current behavior (critic-only terminal contribution)
        gradient = gradient + (self.config.critic_gamma ** (H - 1)) * grad_critic
        total_reward = total_reward + (self.config.critic_gamma ** (H - 1)) * v.squeeze(0)

        total_reward = total_reward + lam * self.config.delta
        return total_reward, gradient



class TotalReward_Mahalanobis(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int):
        super().__init__()
        self.config = config
        self.config.device = device

        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.reward_net = SimpleReward(
            obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward
        ).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        self.kernels = []
        for sd in kernel_state_dicts:
            kernel_net = RobustTransitionKernel(
                obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
            ).to(self.config.device)
            kernel_net.load_state_dict(sd)
            kernel_net.eval()
            self.kernels.append(kernel_net)

        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        if not self.config.explore:
            self.config.gamma = 0.0

        self.config.delta = F.softplus(torch.tensor(0.0, device=self.config.device), beta=self.config.beta)

        # Cache normalization tensors once
        reward_obs_mean = np.asarray(self.reward_stat.obs_mean, dtype=np.float32)
        reward_obs_inv_std = 1.0 / np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)
        kernel_obs_mean = np.asarray(self.kernel_stat.obs_mean, dtype=np.float32)
        kernel_obs_inv_std = 1.0 / np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor)

        self.reward_obs_mean_t = torch.as_tensor(reward_obs_mean, dtype=torch.float32, device=self.config.device)
        self.reward_obs_inv_std_t = torch.as_tensor(reward_obs_inv_std, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_mean_t = torch.as_tensor(kernel_obs_mean, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_inv_std_t = torch.as_tensor(kernel_obs_inv_std, dtype=torch.float32, device=self.config.device)

    def get_beta(self):
        return self.config.beta

    def reward_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.reward_obs_mean_t) * self.reward_obs_inv_std_t

    def kernel_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.kernel_obs_mean_t) * self.kernel_obs_inv_std_t
    
    def sigmoid(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        D2 = compute_total_mahalanobis_score(self.kernels, s, a, s_next)
        tau = self.config.max_mahalanobis_score   # should be your calibrated τ (e.g. 95th percentile)
        # Normalized deviation
        normalized = (D2 - tau) / tau          # <0 → good, >0 → bad
         # Soft ReLU style
        return F.softplus(normalized, beta=self.config.beta)

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        S[i, :self.config.d_s] = s_grad
        A[i, self.config.d_s:] = a_grad
        if s_next_grad is not None:
            S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
            S_next[i + 1, :self.config.d_s] = s_next_grad
            return S, A, S_next
        return S, A

    def get_c(self, x):
        H, _ = x.shape
        C = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C = C + c.squeeze(0)
        C = C / (H - 1)
        C = C - self.config.delta
        return C

    def predict(self, x: torch.Tensor, lam: float):
        H, _ = x.shape
        total_reward = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward = total_reward + (1.0 / H) * r.squeeze(0) - lam * ((1.0 / (H - 1)) * c.squeeze(0))

        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        s_norm_reward = self.reward_processor(s).unsqueeze(0)
        r = self.reward_net(s_norm_reward, a)
        total_reward = total_reward + (1.0 / H) * r.squeeze(0)
        total_reward = total_reward + lam * self.config.delta
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = x.new_zeros(())
        gradient = x.new_zeros((H, D))

        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
            a = a.requires_grad_(True)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads_r = torch.autograd.grad(
                outputs=r,
                inputs=(s_norm_reward, a),
                grad_outputs=torch.ones_like(r),
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )
            r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
            r_a = grads_r[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)

            grads_c = torch.autograd.grad(
                outputs=c,
                inputs=(s_norm_kernel, a, s_next_norm_kernel),
                grad_outputs=torch.ones_like(c),
                create_graph=True,
                retain_graph=True
            )
            c_s = grads_c[0].squeeze(0) * self.kernel_obs_inv_std_t
            c_a = grads_c[1].squeeze(0)
            c_s_next = grads_c[2].squeeze(0) * self.kernel_obs_inv_std_t
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient = gradient + (1.0 / H) * (r_s_grad + r_a_grad) - lam * (1.0 / (H - 1)) * (c_s_grad + c_a_grad + c_s_next_grad)
            total_reward = total_reward + (1.0 / H) * r.squeeze(0) - lam * ((1.0 / (H - 1)) * c.squeeze(0))

        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = a.requires_grad_(True)
        r = self.reward_net(s_norm_reward, a)

        grads_r = torch.autograd.grad(
            outputs=r,
            inputs=(s_norm_reward, a),
            grad_outputs=torch.ones_like(r),
            create_graph=False,
            retain_graph=False
        )
        r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
        r_a = grads_r[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H - 1)

        gradient = gradient + (1.0 / H) * (r_s_grad + r_a_grad)
        total_reward = total_reward + (1.0 / H) * r.squeeze(0)
        total_reward = total_reward + lam * self.config.delta
        return total_reward, gradient

class TotalReward_Critic_Mahalanobis(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int):
        super().__init__()
        self.config = config
        self.config.device = device

        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.reward_net = SimpleReward(
            obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward
        ).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()

        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, critic_checkpoint)
        self.critic = Critic(
            critic_obs_dim, self.config.hidden_dim_critic, self.config.num_hidden_layers_critic
        ).to(self.config.device)
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        self.kernels = []
        for sd in kernel_state_dicts:
            kernel_net = RobustTransitionKernel(
                obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
            ).to(self.config.device)
            kernel_net.load_state_dict(sd)
            kernel_net.eval()
            self.kernels.append(kernel_net)

        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, critic_checkpoint)

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim
        if not self.config.explore:
            self.config.gamma = 0.0

        self.config.delta = F.softplus(torch.tensor(0.0, device=self.config.device), beta=self.config.beta)

        # Cache normalization tensors once
        reward_obs_mean = np.asarray(self.reward_stat.obs_mean, dtype=np.float32)
        reward_obs_inv_std = 1.0 / np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)
        kernel_obs_mean = np.asarray(self.kernel_stat.obs_mean, dtype=np.float32)
        kernel_obs_inv_std = 1.0 / np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor)
        critic_obs_mean = np.asarray(self.critic_stat.obs_mean, dtype=np.float32)
        critic_obs_inv_std = 1.0 / np.maximum(self.critic_stat.obs_std, self.critic_stat.std_floor)

        self.reward_obs_mean_t = torch.as_tensor(reward_obs_mean, dtype=torch.float32, device=self.config.device)
        self.reward_obs_inv_std_t = torch.as_tensor(reward_obs_inv_std, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_mean_t = torch.as_tensor(kernel_obs_mean, dtype=torch.float32, device=self.config.device)
        self.kernel_obs_inv_std_t = torch.as_tensor(kernel_obs_inv_std, dtype=torch.float32, device=self.config.device)
        self.critic_obs_mean_t = torch.as_tensor(critic_obs_mean, dtype=torch.float32, device=self.config.device)
        self.critic_obs_inv_std_t = torch.as_tensor(critic_obs_inv_std, dtype=torch.float32, device=self.config.device)

    def get_beta(self):
        return self.config.beta

    def reward_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.reward_obs_mean_t) * self.reward_obs_inv_std_t

    def kernel_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.kernel_obs_mean_t) * self.kernel_obs_inv_std_t

    def critic_processor(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.critic_obs_mean_t) * self.critic_obs_inv_std_t

    def sigmoid(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        D2 = compute_total_mahalanobis_score(self.kernels, s, a, s_next)
        tau = self.config.max_mahalanobis_score   # should be your calibrated τ (e.g. 95th percentile)
        # Normalized deviation
        normalized = (D2 - tau) / tau          # <0 → good, >0 → bad
         # Soft ReLU style
        return F.softplus(normalized, beta=self.config.beta)

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        S[i, :self.config.d_s] = s_grad
        A[i, self.config.d_s:] = a_grad
        if s_next_grad is not None:
            S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
            S_next[i + 1, :self.config.d_s] = s_next_grad
            return S, A, S_next
        return S, A

    def makeGrad_Critic(self, H, s_grad, i):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device=self.config.device)
        S[i, :self.config.critic_d_s] = s_grad
        return S

    def get_c(self, x):
        H, _ = x.shape
        C = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C = C + c.squeeze(0)
        C = C / (H - 1)
        C = C - self.config.delta
        return C

    def predict(self, x: torch.Tensor, lam: float):
        # Keep your existing semantics (un-normalized running sum + terminal critic)
        H, _ = x.shape
        total_reward = x.new_zeros(())
        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward = total_reward + r.squeeze(0) - lam * c.squeeze(0)

        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        r = self.reward_net(self.reward_processor(s).unsqueeze(0), a)

        final_s_critic = x[H - 1, :self.config.critic_d_s]
        v = self.critic(self.critic_processor(final_s_critic).unsqueeze(0))

        total_reward = total_reward + r.squeeze(0) + (self.config.critic_gamma ** (H - 1)) * v.squeeze(0)
        total_reward = total_reward + lam * self.config.delta
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        # Keep existing semantics from your class:
        # - per-step reward/constraint gradients
        # - terminal critic gradient contribution
        H, D = x.shape
        total_reward = x.new_zeros(())
        gradient = x.new_zeros((H, D))

        for i in range(H - 1):
            s = x[i, :self.config.d_s]
            a = x[i, self.config.d_s:].unsqueeze(0)
            s_next = x[i + 1, :self.config.d_s]

            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
            a = a.requires_grad_(True)
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True)

            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads_r = torch.autograd.grad(
                outputs=r,
                inputs=(s_norm_reward, a),
                grad_outputs=torch.ones_like(r),
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )
            r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
            r_a = grads_r[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)

            grads_c = torch.autograd.grad(
                outputs=c,
                inputs=(s_norm_kernel, a, s_next_norm_kernel),
                grad_outputs=torch.ones_like(c),
                create_graph=True,
                retain_graph=True
            )
            c_s = grads_c[0].squeeze(0) * self.kernel_obs_inv_std_t
            c_a = grads_c[1].squeeze(0)
            c_s_next = grads_c[2].squeeze(0) * self.kernel_obs_inv_std_t
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient = gradient + (r_s_grad + r_a_grad) - lam * (c_s_grad + c_a_grad + c_s_next_grad)
            total_reward = total_reward + r.squeeze(0) - lam * c.squeeze(0)

        # last-step reward grad (kept)
        s = x[H - 1, :self.config.d_s]
        a = x[H - 1, self.config.d_s:].unsqueeze(0)
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = a.requires_grad_(True)
        r = self.reward_net(s_norm_reward, a)

        grads_r = torch.autograd.grad(
            outputs=r,
            inputs=(s_norm_reward, a),
            grad_outputs=torch.ones_like(r),
            create_graph=False,
            retain_graph=False
        )
        r_s = grads_r[0].squeeze(0) * self.reward_obs_inv_std_t
        r_a = grads_r[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H - 1)

        # terminal critic grad
        final_s_critic = x[H - 1, :self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic).unsqueeze(0).requires_grad_(True)
        v = self.critic(final_s_norm_critic)
        grad_v = torch.autograd.grad(
            outputs=v,
            inputs=final_s_norm_critic,
            grad_outputs=torch.ones_like(v),
            create_graph=False,
            retain_graph=False
        )[0]
        v_s = grad_v.squeeze(0) * self.critic_obs_inv_std_t
        grad_critic = self.makeGrad_Critic(H, v_s, H - 1)

        # Preserve your current behavior (critic-only terminal contribution)
        gradient = gradient + (self.config.critic_gamma ** (H - 1)) * grad_critic
        total_reward = total_reward + (self.config.critic_gamma ** (H - 1)) * v.squeeze(0)

        total_reward = total_reward + lam * self.config.delta
        return total_reward, gradient









