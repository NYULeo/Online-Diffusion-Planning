import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Define the Gaussian forward dynamics model: inputs (s, a), outputs mean and log_std of s'
"""
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
        log_std = torch.clamp(log_std, min=-3.0, max=2.0)
        return mu, log_std

    def gaussian_nll(self, x, mu, log_std):
        var = torch.exp(2 * log_std)
        var = torch.clamp(var, min=1e-3)
        nll = 0.5 * torch.log(2 * math.pi * var) + 0.5 * ((x - mu) ** 2) / var
        return nll.sum(dim=-1).mean()
"""
    
"""
class RobustTransitionKernel(nn.Module):
    def __init__(self, obs_dim, act_dim, min_log_std = -6.0, max_log_std = 4.0, noise_floor = 1e-2):
        super().__init__()
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, obs_dim)
        self.log_std_head = nn.Linear(hidden_dim, obs_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.noise_floor = noise_floor

    def forward(self, s, a):
        # s: (..., obs_dim), a: (..., act_dim)
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mu = self.mean_head(h)
        raw_log_std = self.log_std_head(h)
        # soft floor + clamp on upper side
        log_std = self.min_log_std + F.softplus(raw_log_std - self.min_log_std)
        log_std = torch.clamp(log_std, max=self.max_log_std)
        return mu, log_std

    def gaussian_nll(self, s_next, mu, log_std):
        # x, mu: (..., obs_dim); log_std: (..., obs_dim)
        var_pred = torch.exp(2 * log_std)
        var = var_pred + self.noise_floor  # additive floor
        # optional: clamp or clip residuals
        res = s_next - mu
        max_res = 10.0
        res = torch.clamp(res, -max_res, +max_res)
        nll = 0.5 * (torch.log(2 * math.pi * var) + (res ** 2) / var)
        # sum over state dims, but keep batch dims
        return nll.sum(dim=-1).mean()
    
    
    def log_prob(self, s_next, mu, log_std):
        var_pred = torch.exp(2 * log_std)
        var = var_pred + self.noise_floor
        var = torch.clamp(var, min=1e-8)
        res = s_next - mu
        res = torch.clamp(res, -10.0, 10.0)  
        mahal = 0.5 * (res ** 2 / var).sum(dim=-1)
        log_det = torch.log(var).sum(dim=-1)
        const = res.size(-1) * 0.5 * math.log(2 * math.pi)
        nll = const + 0.5 * log_det + mahal
        return -nll  
    
    def prob(self, s_next, mu, log_std):
        log_prob = self.log_prob(s_next, mu, log_std)
        var_pred = torch.exp(log_prob)
        return var_pred
    

    def log_prob(self, s_next, mu, log_std):
        # Compute log prob (not negative) — useful for testing / diagnostics
        var = torch.exp(2 * log_std) + self.noise_floor
        var = torch.clamp(var, min=1e-8)  # Prevent log(0)
        D = s_next.size(-1)
        # log prob per dimension
        lp = -0.5 * (((s_next - mu) ** 2) / var).sum(dim=-1)
        lp = lp - 0.5 * (D * math.log(2 * math.pi) + 2 * log_std.sum(dim=-1))
        return lp  # tensor of shape batch


"""

class RobustTransitionKernel(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        num_hidden_layers=2,
        hidden_dim=256,
        min_log_std=-6.0,
        max_log_std=4.0,
        noise_floor=1e-2,
    ):
        super().__init__()
        assert num_hidden_layers >= 1, "num_hidden_layers must be >= 1"

        layers = []
        # first layer
        layers.append(nn.Linear(obs_dim + act_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        # additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, obs_dim)
        self.log_std_head = nn.Linear(hidden_dim, obs_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.noise_floor = noise_floor

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mu = self.mean_head(h)
        raw_log_std = self.log_std_head(h)
        log_std = self.min_log_std + F.softplus(raw_log_std - self.min_log_std)
        log_std = torch.clamp(log_std, max=self.max_log_std)
        return mu, log_std
    
    def gaussian_nll(self, s_next, mu, log_std):
        # x, mu: (..., obs_dim); log_std: (..., obs_dim)
        var_pred = torch.exp(2 * log_std)
        var = var_pred + self.noise_floor  # additive floor
        # optional: clamp or clip residuals
        res = s_next - mu
        max_res = 10.0
        res = torch.clamp(res, -max_res, +max_res)
        nll = 0.5 * (torch.log(2 * math.pi * var) + (res ** 2) / var)
        # sum over state dims, but keep batch dims
        return nll.sum(dim=-1).mean()
    
    """
    def log_prob(self, s_next, mu, log_std):
        var_pred = torch.exp(2 * log_std)
        var = var_pred + self.noise_floor
        var = torch.clamp(var, min=1e-8)
        res = s_next - mu
        res = torch.clamp(res, -10.0, 10.0)  
        mahal = 0.5 * (res ** 2 / var).sum(dim=-1)
        log_det = torch.log(var).sum(dim=-1)
        const = res.size(-1) * 0.5 * math.log(2 * math.pi)
        nll = const + 0.5 * log_det + mahal
        return -nll  
    """
   
    def log_prob(self, s_next, mu, log_std):
        # Compute log prob (not negative) — useful for testing / diagnostics
        var = torch.exp(2 * log_std) + self.noise_floor
        var = torch.clamp(var, min=1e-8)  # Prevent log(0)
        D = s_next.size(-1)
        # log prob per dimension
        lp = -0.5 * (((s_next - mu) ** 2) / var).sum(dim=-1)
        lp = lp - 0.5 * (D * math.log(2 * math.pi) + 2 * log_std.sum(dim=-1))
        return lp  # tensor of shape batch

    def mahalanobis_distance_squared(self, s_next, s, a):
        """
        Compute squared Mahalanobis distance D² for batch of transitions.
        Returns tensor of shape (batch_size,)
        """
        mu, log_std = self.forward(s, a)          # (batch, obs_dim), (batch, obs_dim)
        var_pred = torch.exp(2 * log_std)         # predicted variance
        var = var_pred + self.noise_floor         # same as in your log_prob
        var = torch.clamp(var, min=1e-8)
        residual = s_next - mu
        # Optional: mild clipping for stability (you already do this in NLL)
        residual = torch.clamp(residual, -10.0, 10.0)
        # Squared Mahalanobis (diagonal covariance)
        D2 = ((residual ** 2) / var).sum(dim=-1)   # sum over state dimensions
        return D2
    
    def computeD(self, s, a, s_next):
        mu, log_std = self.forward(s, a) 
        var = torch.exp(2 * log_std) + self.noise_floor
        var = torch.clamp(var, min=1e-8)  # Prevent log(0)
        D = s_next.size(-1)
        Temp = 0.5 * (D * math.log(2 * math.pi) + 2 * log_std.sum(dim=-1))
        return Temp

    