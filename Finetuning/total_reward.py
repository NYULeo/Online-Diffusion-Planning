
from numpy import shape
from Pretrain.Rewards.nets import ScalarReward
from Pretrain.Transition_Kernel.Kernel_Net import TransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_log_prob
from torch import nn
import torch

class Lambdas:
    def __init__(self, horizon: int, lam: float):
         self.Lambdas = [lam  for _ in range(horizon-1)]
    


     


class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    gamma: float
    device: str
    d_s: int
    d_a: int

class TotalReward(nn.Module):
    def __init__(self, config: RewardConfig, reward_net: ScalarReward, kernel_net: TransitionKernel, Lambdas: Lambdas):
        super().__init__()
        self.config = config
        self.reward_net = reward_net.to(self.config.device)
        self.kernel_net = kernel_net.to(self.config.device)
        self.Lambdas 
    
    def sigmoid(self, s, a, s_next):
        log_prob = compute_log_prob(self.kernel_net, s, a, s_next)
        x = self.config.min_log_prob - log_prob
        c = torch.softplus(x, beta = self.config.beta)
        return c

    def forward(self, x: torch.Tensor):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        C = []
        for i in range(H-1):
            r = self.reward_net.predict(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0))
            var = self.reward_net.variance(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0))
            c = self.sigmoid(x[i][:self.config.d_s].unsqueeze(0), x[i][self.config.d_s:].unsqueeze(0), x[i+1][:self.config.d_s].unsqueeze(0))
            C.append(c)
            total_reward += (r.squeeze(0) + self.config.gamma * var.squeeze(0)) - (self.config.lam * c)
        return total_reward, C




    



        




