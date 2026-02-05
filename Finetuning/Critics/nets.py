
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn



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
            nn.Linear(hidden, 1),
            nn.ReLU()                              
        )
        #self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, obs):
        #return self.net([obs, act]).squeeze(-1) * self.scale
        return self.net(obs).squeeze(-1)
"""