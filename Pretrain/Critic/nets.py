
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn


"""
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden  = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()                              
        )
        #self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, obs):
        #return self.net([obs, act]).squeeze(-1) * self.scale
        return self.net(obs).squeeze(-1)
"""

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, hidden_layers=2):
        super().__init__()
        layers = []
        # Input layer
        layers.extend([
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ])
        # Hidden layers (repeat num_layers - 1 times; last "hidden" block is output)
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        # Output layer
        layers.extend([
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)



"""
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden  = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()                              
        )
        #self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, obs):
        #return self.net([obs, act]).squeeze(-1) * self.scale
        return self.net(obs).squeeze(-1)

"""






"""
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()                              
        )
        #self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, obs):
        #return self.net([obs, act]).squeeze(-1) * self.scale
        return self.net(obs).squeeze(-1)
"""

"""
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()                              
        )
        #self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, obs):
        #return self.net([obs, act]).squeeze(-1) * self.scale
        return self.net(obs).squeeze(-1)
"""


class CriticEnsemble(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, hidden_layers=2, num_heads=5):
        super().__init__()
        self.hidden = hidden_dim
        self.num_heads = num_heads
        self.critics = nn.ModuleList([
            Critic(obs_dim, hidden_dim, hidden_layers) for _ in range(num_heads)
        ])

    def forward(self, obs, aggregate="mean"):
        preds = torch.stack([c(obs) for c in self.critics], dim=-1)
        if aggregate == "mean":
            return preds.mean(dim=-1)
        elif aggregate == "min":
            return preds.min(dim=-1).values
        else:
            return preds