from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
from Pretrain.Planners.Backbone.Dit import DiT1d
import torch
import torch.nn as nn
import torch.nn.functional as F

from Pretrain.Planners.Backbone.utils import cosine_alpha_sigma, cosine_beta, compute_dot_alpha_beta, get_pretrained_planner
import numpy as np
from typing import Optional
from torch import Tensor
from utils import Lambda, function
from traj_reward import RewardConfig, TotalReward
from torch.utils.data import DataLoader
from Pretrain.Planners.Backbone.UNet import TemporalUnet
from Pretrain.Dataset import get_env


@dataclass
class Parallel_AdjointMatchingConfig:
    """Configuration for the adjoint matching fine‑tuner."""

    horizon: int
    lr: float = 2e-4
    d_s: int = 0
    d_a: int = 0
    backbone_name: str = 'transformer'
    eta: float = 0.8
    num_steps: int = 500
    s: float = 0.008  # cosine schedule offset used in base drift
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lam: float = 1



class Parallel_AdjointMatchingFineTuner:
    def __init__(self,
                 env_name: str,
                 specific_env: str,
                 planner_checkpoint: int,
                 reward_model_checkpoint: int,
                 kernel_model_checkpoint: int,
                 AMConfig: Parallel_AdjointMatchingConfig,
                 RewardConfig: RewardConfig):
        self.config = AMConfig
        self.env, d_s, d_a = get_env(env_name, specific_env)
        self.config.d_s = d_s
        self.config.d_a = d_a

        # Load old_score_net (frozen)
        state_dict = get_pretrained_planner(env_name, specific_env, planner_checkpoint)
        if( env_name == 'kitchen'):
            self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(self.config.device)
        elif (env_name == 'pointmaze'):
            self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(self.config.device)
        else:
          raise ValueError(f"Invalid Environment: {env_name}")
        self.old_score_net.load_state_dict(state_dict)
        self.old_score_net.eval()

        # Load reward model
        self.reward_model = TotalReward(RewardConfig, env_name, specific_env, reward_model_checkpoint, kernel_model_checkpoint).to(self.config.device)
        self.reward_model.eval()

        # New trainable score net
        self.backbone_selection()
        # Wrap with DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            self.new_score_net = torch.nn.DataParallel(self.new_score_net)
        self.reset_parameters()
        self.new_score_net.train()
        self.optimizer = torch.optim.Adam(self.new_score_net.parameters(), lr=self.config.lr)
        self.t_asc = torch.linspace(1.0, 0.0, self.config.num_steps + 1, device=self.config.device)
        self.k = self.kt(self.t_asc)
        self.Lam = Lambda(lam = self.config.lam, beta = self.reward_model.config.beta, eta_lam = self.config.lr)
    
    def reset_parameters(self):
        self.new_score_net.load_state_dict(self.old_score_net.state_dict())

    def backbone_selection(self):
        if self.config.backbone_name == 'transformer':
            self.new_score_net = DiT1d(in_dim=(self.config.d_s + self.config.d_a), emb_dim=128, 
                         d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(self.config.device)
        elif self.config.backbone_name == 'unet':
            self.new_score_net = TemporalUnet(self.config.horizon, self.config.d_s + self.config.d_a).to(self.config.device)
        else:
            raise ValueError(f"Invalid backbone name: {self.config.backbone_name}")

    def kt(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = t.clamp(0.0, 1.0 - 1e-3)
        a = (math.pi / 2.0) * ((t_clamped + self.config.s) / (1.0 + self.config.s))
        return (-0.5) * (math.pi / (1.0 + self.config.s)) * torch.tan(a)

    def sigma_t(self, k: torch.Tensor) -> torch.Tensor:
        if(k < 0):
           return torch.sqrt(-2 * k)
        else:
           raise ValueError(f'K should be negative, but got {k}')

    def vector_field(self, x: torch.Tensor, t: torch.Tensor, score_model: torch.nn.Module) -> torch.Tensor:
        # x: (K, horizon, dim), t: either scalar or (K,)
        k = self.kt(t)
        return k.unsqueeze(-1).unsqueeze(-1) * x + k.unsqueeze(-1).unsqueeze(-1) * score_model(x, t)

    def sample_Traj_batch(self, s0_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        s0_batch: shape (K, d_s)
        returns: list of S tensors each of shape (K, horizon, dim)
        """
        self.new_score_net.eval()
        device = self.config.device
        K = s0_batch.shape[0]
        d_s, d_a = self.config.d_s, self.config.d_a
        horizon = self.config.horizon
        dim = d_s + d_a

        # Initialize x_T ~ N(0, I)
        x = torch.randn(K, horizon, dim, dtype=torch.float32, device=device)

        # Conditioning
        conditions = s0_batch.unsqueeze(1).expand(-1, horizon, d_s)  # (K, horizon, d_s)
        mask = torch.zeros(K, horizon, dim, dtype=torch.float32, device=device)
        mask[:, 0, :d_s] = 1.0
        y = torch.zeros(K, horizon, dim, dtype=torch.float32, device=device)
        y[:, 0, :d_s] = conditions[:, 0, :]

        x = mask * y + (1.0 - mask) * x

        trajs: List[torch.Tensor] = []
        trajs.append(x.detach().clone())

        for i in range(len(self.t_asc) - 1):
            t_now = self.t_asc[i]
            dt = (self.t_asc[i+1] - t_now).item()
            # Expand t_now for batch if needed
            t_batch = t_now.expand(K).to(device)

            # Score model forward
            score = self.new_score_net(x, t_batch)

            drift = self.k[i] * x

            if self.config.eta > 0:
                noise = torch.randn_like(x)
                noise_scale = self.config.eta * math.sqrt((-2 * self.k[i]) * (-dt))
                x = x + (drift + 2 * self.k[i] * score) * dt + noise_scale * noise
            else:
                x = x + (drift + 2 * self.k[i] * score) * dt

            x = mask * y + (1.0 - mask) * x

            trajs.append(x.detach().clone())
        
        self.new_score_net.train()
        return trajs

    def make_a_batch(self, trajs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        trajs: list length S, each tensor (K,n horizon, dim)
        returns:
          adjoints: list length S, each tensor (K, horizon*dim)
          reward_batch: tensor (K,)
        """
        device = self.config.device
        K = trajs[0].shape[0]
        S = len(trajs)
        horizon = self.config.horizon
        dim = self.config.d_s + self.config.d_a

        # Final state
        final_x = trajs[-1][:, 0, :].to(device)  # shape (K, dim)
        reward_batch, gradient_batch = self.reward_model(final_x, self.Lam.get_lam())  # assume returns (K,), (K, dim)
        # Flatten gradient
        adj0 = -gradient_batch.view(K, -1)  # (K, horizon*dim) if horizon>1 assumed

        adj_list = [adj0]

        for i in range(S - 1):
            T = trajs[S-1 - i].to(device)  # reversed
            dt = (self.t_asc[i+1] - self.t_asc[i]).to(device)
            T_flat = T.requires_grad_(True).view(K, -1)  # (K, horizon*dim)
            current_a = adj_list[-1]   

            # Compute Jacobian-vector product for each batch entry if feasible
            # Here we do simple per-sample vjp; for simplicity we loop batch
            # but you might vectorize further
            Jov_mv = []
            """
            for k_idx in range(K):
                grad_out = adj_list[-1][k_idx]  # (horizon*dim)
                grad_in = torch.autograd.grad(
                    outputs=self.old_score_net(T[k_idx:k_idx+1], self.t_asc[S-1-i].unsqueeze(0)),
                    inputs=T_flat[k_idx:k_idx+1],
                    grad_outputs=grad_out.unsqueeze(0),
                    retain_graph=False,
                    create_graph=False
                )[0].view(-1)
                Jov_mv.append(grad_in)
            """
            for k_idx in range(K):
                x_k = T_flat[k_idx:k_idx+1]       # shape (1, N)
                a_k = current_a[k_idx:k_idx+1]   # shape (1, N)

                    # define score_fn_k that returns score(x_k) flattened
                def score_fn_x(x_in):
                        return self.old_score_net(x_in.view(1, horizon, dim),
                                                  self.t_asc[S-1-i].unsqueeze(0)).view(1, -1)

                    # Compute Jacobian-vector product J @ a_k
                _, jvp_k = torch.autograd.functional.jvp(score_fn_x, (x_k,), (a_k, ),
                                                      create_graph=False, strict=False)
                    # shape of jvp_k: (1, N) matching current_a shape
                Jov_mv.append(jvp_k.squeeze(0))

            Jov_mv = torch.stack(Jov_mv, dim=0)  # (K, horizon*dim)
            adj_new = adj_list[-1] + dt * (self.k[i] * adj_list[-1] + 2 * self.k[i] * Jov_mv)
            adj_list.append(adj_new)

        adj_list = adj_list[::-1]  # reverse
        return adj_list, reward_batch

    def adjoint_matching_loss_batch(self,
                                    trajs: List[torch.Tensor],
                                    adjoints: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for batch.
        trajs: list length S, each tensor (K, horizon, dim)
        adjoints: list length S, each tensor (K, horizon*dim)
        returns: tensor of shape (K,)
        """
        device = self.config.device
        K = trajs[0].shape[0]
        S = len(trajs)
        horizon = self.config.horizon
        dim = self.config.d_s + self.config.d_a

        losses = []
        for i in range(S):
            x_i = trajs[i].to(device)
            t_batch = self.t_asc[i].expand(K).to(device)
            v_new = self.vector_field(x_i, t_batch, self.new_score_net).view(K, -1)
            v_old = self.vector_field(x_i, t_batch, self.old_score_net).view(K, -1)
            sigma = self.sigma_t(self.k[i]).to(device)
            sigma_batch = sigma.expand(K)
            term = ((v_new - v_old) * (2.0/sigma_batch).unsqueeze(-1) + sigma_batch.unsqueeze(-1) * adjoints[i])
            loss_i = term.pow(2).sum(dim=1)  # (K,)
            losses.append(loss_i)

        loss_tensor = torch.stack(losses, dim=0).sum(dim=0)  # (K,)
        return loss_tensor

    def step_batch(self, s0_batch: torch.Tensor) -> Tuple[float, float, float]:
        """
        s0_batch: torch.Tensor of shape (K, d_s)
        Returns (avg_loss, avg_reward, avg_C)
        """
        device = self.config.device
        s0_batch = s0_batch.to(device)
        assert s0_batch.ndim == 2 and s0_batch.shape[1] == self.config.d_s

        self.optimizer.zero_grad()

        # 1) sample trajectories for the batch
        trajs = self.sample_Traj_batch(s0_batch)

        # 2) compute adjoints + rewards
        adjoints_batch, reward_batch = self.make_a_batch(trajs)

        # 3) compute loss vector of shape (K,)
        loss_batch = self.adjoint_matching_loss_batch(trajs, adjoints_batch)

        # 4) compute mean loss, update lam
        avg_loss = loss_batch.mean()
        avg_reward = reward_batch.mean().item()

        # compute C using final states
        final_states = trajs[-1][:, 0, :].to(device)  # (K, dim)
        C_batch = self.reward_model.get_c(final_states)
        avg_C = C_batch.mean().item()
        self.config.lam = self.config.lam  # you might update lam differently
        self.Lam.update(avg_C)

        # 5) backprop and update
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.new_score_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return avg_loss.item(), avg_reward, avg_C



