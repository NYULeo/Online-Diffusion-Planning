
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
            #nn.ReLU(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
"""




"""
class Critic(nn.Module):
      def __init__(
          self,
          obs_dim,
          hidden_dim=128,
          hidden_layers=2,
          positive_output=False,
      ):
          super().__init__()
          self.positive_output = positive_output
          layers = [
              nn.Linear(obs_dim, hidden_dim),
              nn.LayerNorm(hidden_dim),
              nn.SiLU(),
          ]
          for _ in range(hidden_layers):
              layers.extend([
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.LayerNorm(hidden_dim),
                  nn.SiLU(),
              ])
          layers.append(nn.Linear(hidden_dim, 1))
          self.net = nn.Sequential(*layers)

      def forward(self, obs):
          value = self.net(obs).squeeze(-1)
          if self.positive_output:
              value = F.softplus(value)
          return value

"""



class Critic(nn.Module):
      def __init__(
          self,
          obs_dim,
          hidden_dim=128,
          hidden_layers=2,
          positive_output=False,
      ):
          super().__init__()
          self.positive_output = positive_output
          layers = [
              nn.Linear(obs_dim, hidden_dim),
              nn.LayerNorm(hidden_dim),
              nn.GELU(approximate='tanh'),
          ]
          for _ in range(hidden_layers):
              layers.extend([
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.LayerNorm(hidden_dim),
                  nn.GELU(approximate='tanh'),
              ])
          layers.append(nn.Linear(hidden_dim, 1))
          self.net = nn.Sequential(*layers)

      def forward(self, obs):
          value = self.net(obs).squeeze(-1)
          if self.positive_output:
              value = F.softplus(value)
          return value


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