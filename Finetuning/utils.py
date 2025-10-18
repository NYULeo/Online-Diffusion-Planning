import torch
import numpy as np
from numpy import shape
from Pretrain.Rewards.nets import ScalarReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from torch import nn
import torch




class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    gamma: float
    device: str
    d_s: int
    d_a: int

class TotalReward(nn.Module):
    def __init__(self, config: RewardConfig, reward_net: ScalarReward, kernel_net: RobustTransitionKernel):
        super().__init__()
        self.config = config
        self.reward_net = reward_net.to(self.config.device)
        self.kernel_net = kernel_net.to(self.config.device)
    
    def sigmoid(self, s, a, s_next):
        mu, log_std = self.kernel_net.forward(self.kernel_net, s, a, s_next)
        log_prob = self.kernel_net.log_prob(s_next, mu, log_std)
        x = self.config.min_log_prob - log_prob
        c = torch.softplus(x, beta = self.config.beta)
        return c

    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        C = 0.0
        for i in range(H-1):
            r = self.reward_net.predict(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0))
            var = self.reward_net.variance(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0))
            c = self.sigmoid(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0), x[i+1][:self.config.d_s].unsqueeze(0))
            C += c
            total_reward += (r.squeeze(0) + self.config.gamma * var.squeeze(0)) - (lam * c)
        
        r = self.reward_net.predict(x[H-1][:self.config.d_s].unsqueeze(0), x[H-1][self.config.d_s:].unsqueeze(0))
        var = self.reward_net.variance(x[H-1][:self.config.d_s].unsqueeze(0), x[H-1][self.config.d_s:].unsqueeze(0))
        total_reward += (r.squeeze(0) + self.config.gamma * var.squeeze(0))
        total_reward = total_reward * (1/H)
        C = C * (1/(H-1))
        return total_reward, C



class Lambda:
    def __init__(self, lam: float):
        self.lam = lam
    def update(self, C, eta_lam: float):
        self.lam = np.max(0, self.lam + eta_lam * C)
    def get_lam(self):
        return self.lam


def function(x, beta: float):
    return (1/beta)* np.log(1 + np.exp(x*beta))