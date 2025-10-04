import torch
import torch.nn as nn


# Define the Gaussian forward dynamics model: inputs (s, a), outputs mean and log_std of s'
class TransitionKernel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, obs_dim)
        self.log_std_head = nn.Linear(hidden_dim, obs_dim)
   
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mu = self.mean_head(h)
        log_std = self.log_std_head(h)
        # Clamp log_std to reasonable range for numerical stability
        log_std = torch.clamp(log_std, min=-10.0, max=2.0)
        return mu, log_std

    def gaussian_nll(self, x, mu, log_std):
        var = torch.exp(2 * log_std)
        nll = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((x - mu) ** 2) / var
        return nll.sum(dim=-1).mean()