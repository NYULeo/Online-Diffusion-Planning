import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from Pretrain.Rewards.nets import ScalarReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
from Pretrain.Transition_Kernel.Kernel_Backbone import get_pretrained_kernel, get_pretrained_kernel_stats
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np



@dataclass
class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    explore: bool = True
    gamma: float = 0.8
    device = None
    d_s: int = 0 
    d_a: int = 0
    

class TotalReward(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int):
        super().__init__()
        self.config = config
        reward_state_dict, obs_dim, act_dim, reward_name = get_pretrained_reward(dataset_name, reward_checkpoint, specific_dataset)
        self.reward_net = ScalarReward(obs_dim, act_dim).to(device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()
        self.kernels = []
        self.config.device = device



        
        kernel_state_dicts, obs_dim, act_dim, kernel_name = get_pretrained_kernel(dataset_name, kernel_checkpoint, specific_dataset)
        for i in range(len(kernel_state_dicts)):
                kernel_net = RobustTransitionKernel(obs_dim, act_dim).to(self.config.device)
                kernel_net.load_state_dict(kernel_state_dicts[i])
                kernel_net.eval()
                self.kernels.append(kernel_net)
        

        self.reward_stat = get_pretrained_reward_stats(reward_name)
        self.kernel_stat = get_pretrained_kernel_stats(kernel_name)

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        if(not self.config.explore):
              self.config.gamma = 0.0
        
    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        total = torch.tensor([0.0], device = self.config.device, requires_grad = True)
        for i in range(len(self.kernels)):
            mu, log_std = self.kernels[i](s, a)
            lp = self.kernels[i].log_prob(s_next, mu, log_std)
            total = total + lp 
        avg = total / len(self.kernels)
        x =  self.config.min_log_prob - avg
        c = F.softplus(x, beta = self.config.beta)
        return c
    

    def reward_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.reward_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s
    
    def kernel_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.kernel_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        S[i][:self.config.d_s] = s_grad
        A[i][self.config.d_s:] = a_grad
        if s_next_grad is not None:
           S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
           S_next[i+1][:self.config.d_s] = s_next_grad
           return S, A, S_next
        return S, A
        
    def get_c(self, x):
        H, D = x.shape
        C = 0.0
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm = self.reward_processor(s).unsqueeze(0)
            a = x[i][self.config.d_s:].unsqueeze(0)
            s_next = x[i+1][:self.config.d_s]
            s_next_norm = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm, a, s_next_norm)
            C += c.item()
        C = C / (H-1)
        return C
   
    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device=self.config.device, requires_grad = False)
        gradient = torch.zeros(H, D, device = self.config.device, requires_grad = False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
            #s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
            a = x[i][self.config.d_s:].unsqueeze(0).requires_grad_(True)
            s_next = x[i+1][:self.config.d_s]
            #s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True)
            s_next_norm = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True)

            r = self.reward_net.predict(s_norm, a)
            var = self.reward_net.variance(s_norm, a)
            c = self.sigmoid(s_norm, a, s_next_norm)
           
            grads = torch.autograd.grad(
                        outputs = r,
                        inputs = (s_norm, a),
                        grad_outputs = torch.ones_like(r),
                        create_graph = False,
                        retain_graph = False
                    )
            r_s = grads[0].squeeze(0) * ((np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)))
            r_a = grads[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)
            

            grads = torch.autograd.grad(
                        outputs = var,
                        inputs = (s_norm, a),
                        grad_outputs = torch.ones_like(var),
                        create_graph = False,
                        retain_graph = False
                    )
            var_s = grads[0].squeeze(0) * (np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor))
            var_a = grads[1].squeeze(0)
            var_s_grad, var_a_grad = self.makeGrad(H, var_s, var_a, i)
            

            grads = torch.autograd.grad(
                        outputs = c,
                        inputs = (s_norm, a, s_next_norm),
                        grad_outputs = torch.ones_like(c),
                        create_graph = True,
                        retain_graph = True
                        
                    )
            c_s = grads[0].squeeze(0) 
            c_a = grads[1].squeeze(0)   
            c_s_next = grads[2].squeeze(0) * (np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor))
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)
           
            gradient +=  (1/H)*((r_s_grad + r_a_grad) + self.config.gamma * (var_s_grad + var_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)
            total_reward += (1/H)*(r.squeeze(0) + self.config.gamma * var.squeeze(0)) - lam * ((1/(H-1)) * c.squeeze(0) - (1/(H-1)))
            
        

        s = x[H-1][:self.config.d_s]
        s_norm = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(True)
        r = self.reward_net.predict(s_norm, a)
        var = self.reward_net.variance(s_norm, a)

        grads = torch.autograd.grad(
                        outputs = r,
                        inputs = (s_norm, a),
                        grad_outputs = torch.ones_like(r),
                        create_graph = False,
                        retain_graph = False
                )
        r_s = grads[0].squeeze(0) * (np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor))
        r_a = grads[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)


        grads = torch.autograd.grad(
                        outputs = var,
                        inputs = (s_norm, a),
                        grad_outputs = torch.ones_like(var),
                        create_graph = False,
                        retain_graph = False
                    )
        var_s = grads[0].squeeze(0) * (np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor))
        var_a = grads[1].squeeze(0)
        var_s_grad, var_a_grad = self.makeGrad(H, var_s, var_a, H-1)

        gradient += (1/H) * ((r_s_grad + r_a_grad) + self.config.gamma * (var_s_grad + var_a_grad)) 
        total_reward +=  (1/H) * (r.squeeze(0) + self.config.gamma * var.squeeze(0))
        return total_reward, gradient