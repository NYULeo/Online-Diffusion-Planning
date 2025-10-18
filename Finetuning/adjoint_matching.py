from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
from Pretrain.Planners.Backbone.Dit import DiT1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from Pretrain.Planners.Backbone.utils import cosine_alpha_sigma, cosine_beta, compute_dot_alpha_beta
import numpy as np
from typing import Optional
from torch import Tensor
from .utils import TotalReward, Lambda, function
from torch.utils.data import DataLoader




@dataclass
class FineTuningConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    horizon: int
    state_dim: int
    d_s: int
    d_a: int
    eta: float = 0.8
    num_steps: int = 500
    epoch: int = 100
    lr: float = 1e-4
    s: float = 0.008  # cosine schedule offset used in base drift
    device: str = "cuda"
    lam: float = 1e-3



class AdjointMatchingFineTuner:
    """
    Implements fine‑tuning via the adjoint matching algorithm for
    trajectory diffusion models with fixed initial state.

    Given a pretrained score network (frozen), a differentiable reward
    model and a trainable control network, this class simulates
    trajectories with a memoryless noise schedule, solves the lean
    adjoint backwards and computes the adjoint matching loss on the
    unclamped dimensions of the state.
    """

    def __init__(
        self,
        pretrained_score_net: DiT1d,
        new_score_net: DiT1d, 
        reward_model: TotalReward,
        config: FineTuningConfig,
    ) -> None:
        self.old_score_net = pretrained_score_net.to(config.device)
        self.old_score_net.eval()
        self.reward_model = reward_model.to(config.device)
        self.reward_model.eval()
        self.new_score_net = new_score_net.to(config.device)
        self.new_score_net.train()
        self.config = config
        self.optimizer = torch.optim.Adam(self.new_score_net.parameters(), lr=config.lr)
        self.t_asc = torch.linspace(1.0, 0.0, self.config.num_steps + 1, device = self.config.device)
        self.k = self.kt(self.t_asc) 
        self.Lam = Lambda(lam = config.lam)

    def vector_field(self, x: torch.Tensor, t: torch.Tensor, score_model: DiT1d) -> torch.Tensor:
        # Compute beta(t) from cosine schedule
        k = self.kt(t)
        v = k * x + k * score_model(x, t.unsqueeze(0))
        return v
    
    def sigma_t(self, k: torch.Tensor) -> torch.Tensor:
        if(k < 0):
           return torch.sqrt(-2 * k)
        else:
           raise ValueError(f'K should be negative, but got {k}')
    
    def kt(self, t: torch.Tensor) -> torch.Tensor:
       t = t.clamp(0.0, 1.0 - 1e-3)
       a = (math.pi / 2.0) * ((t + self.config.s) / (1.0 + self.config.s))
       return (-0.5) * (math.pi / (1.0 + self.config.s)) * torch.tan(a)    
    
    def compute_jacobian_vectorized(self, T, t_index):
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       def score_fn(x_flat):
           x_reshaped = x_flat.view_as(T)  # Reshape to original tensor shape
           score = self.old_score_net(x_reshaped, self.t_asc[t_index].unsqueeze(0), condition=None)
           return score.flatten()  # Return flattened score
    
       T_flat = T.flatten().detach().requires_grad_(True)
    
       # Use torch.autograd.functional.jacobian for efficient computation
       try:
           jacobian = torch.autograd.functional.jacobian(score_fn, T_flat, create_graph=True)
           return jacobian
       except Exception as e:
           print(f"Warning: Vectorized Jacobian failed, falling back to element-wise: {e}")
           # Fallback to original method if vectorized fails
           return self._compute_jacobian_elementwise(T, t_index)

    def _compute_jacobian_elementwise(self, T, t_index):
       score = self.old_score_net(T, self.t_asc[t_index].unsqueeze(0))
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       Jov = torch.zeros(H_dim, H_dim, device=self.config.device)
    
       for j in range(H_dim):
           # Create one-hot for j-th output element
           grad_outputs = torch.zeros_like(score)
           grad_outputs.view(-1)[j] = 1.0
        
           # Compute gradient of j-th output w.r.t input
           grad_j = torch.autograd.grad(
                outputs=score,
                inputs=T,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
           )[0]
            # Store j-th row of Jacobian
           Jov[j, :] = grad_j.view(-1)  # [H*dim]
       return Jov

    @torch.no_grad()
    def sample_Traj(self,
        s0: np.ndarray,
        ) -> np.ndarray:
        self.new_score_net.eval()
        s0_t = torch.tensor(s0, device=self.config.device, dtype=torch.float32)
        if ( (s0_t.shape[0] != self.config.d_s)   ):
             raise ValueError(f"s0 should have shape ({self.config.d_s},), but got {s0_t.shape}")
        dim = self.config.d_s + self.config.d_a
        
    
        # Initialize x_T ~ N(0, I) with shape (horizon, dim)
        x = torch.randn(self.config.horizon, dim, dtype=torch.float32, device=self.config.device).unsqueeze(0)
        conditions = s0_t.unsqueeze(0)
        mask = torch.zeros((1, self.config.horizon, dim), dtype = torch.float32, device = self.config.device)
        mask[:, 0, :self.config.d_s] = 1
        y = torch.zeros((1, self.config.horizon, dim), dtype = torch.float32, device = self.config.device)
        y[:, 0, :self.config.d_s] = conditions.clone()
        #x = apply_conditioning(x, conditions, d_s)
        x = mask * y + (1 - mask) * x
    
    
        X = []
        X.append(x.detach().clone())
        for i in range(len(self.t_asc) - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now).item()
            score = self.new_score_net(x, t_now.unsqueeze(0))
            drift = self.k[i] * x
        
            if self.config.eta > 0:
               noise = torch.randn_like(x)
               noise_scale = self.config.eta * math.sqrt((-2*self.k[i]) * (-dt))
               x = x + (drift +  2*self.k[i] * score) * dt + noise_scale * noise
            else:
               x = x + (drift +  2*self.k[i] * score) * dt
        
            x = mask * y + (1 - mask) * x
            X.append(x.detach().clone())
        #x = apply_conditioning(x, conditions, d_s)
        self.new_score_net.train()
        return  X

    def make_a(self, X):
        X = [x.to(self.config.device) for x in X]
        steps_T = len(X)
        X_reversed = X[::-1] 
        a = []
        self.reward_model.to(self.config.device).eval()
        T = X_reversed[0].to(self.config.device)
        T.requires_grad_(True)
        T_squeezed = T.squeeze(0) 
        output, C = self.reward_model(T_squeezed)
        gradient = torch.autograd.grad(
             outputs = output,
             inputs = T_squeezed,
             grad_outputs = torch.ones_like(output),
             create_graph = True,  # Allows higher-order gradients
             retain_graph = True
            )[0]
        gradient_flat = -1 * gradient.view(-1)  # [H*dim]
        a.append(gradient_flat)
        for i in range(steps_T - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now).item()
            T = X_reversed[i].to(self.config.device)
            T.requires_grad_(True)
            try:
                Jov = self.compute_jacobian_vectorized(T, i)
            except Exception as e:
                print(f"Vectorized Jacobian failed for step {i}, using fallback: {e}")
                Jov = self._compute_jacobian_elementwise(T, i)
        
            current_a = a[i].to(self.config.device)  # [H*dim]
        
            # Compute: a + dt * (k[i] * a + 2 * k[i] * Jov @ a)
            new_a = current_a + dt * (self.k[i] * current_a + 2 * self.k[i] * (Jov @ current_a))
            a.append(new_a)
            
        a.reverse()
        return a, C

    def adjoint_matching_loss(
        self,
        traj_x: List[torch.Tensor],
        adjoints: List[torch.Tensor]
    ) -> torch.Tensor:
        Loss = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        for i in range(len(traj_x)):
            traj_x_i = traj_x[i].to(self.config.device)
            adjoint_i = adjoints[i].to(self.config.device)
            v_new = self.vector_field(traj_x_i, self.t_asc[i], self.new_score_net).squeeze(0).flatten()
            v_old = self.vector_field(traj_x_i, self.t_asc[i], self.old_score_net).squeeze(0).flatten()
            sigma = self.sigma_t(self.k[i])
            Loss += ((v_new - v_old) * (2/sigma) + sigma * adjoint_i).pow(2).sum()
        return Loss

    def step(self, s0: np.ndarray) -> float:
        self.optimizer.zero_grad()
        Loss = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        Total_C = 0.0
        for i in range(len(s0)):
           traj_x = self.sample_Traj(s0[i])
           adjoints, C = self.make_a(traj_x)
           loss = self.adjoint_matching_loss(traj_x, adjoints)
           Loss += loss
        Loss = Loss / len(s0)
        Loss.backward()
        self.optimizer.step()
        return Loss.detach().cpu().item()
    
   # def finetune(self, )

        
