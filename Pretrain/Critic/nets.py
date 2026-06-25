'''Critic networks for ODP (JAX/Flax port of the PyTorch originals).'''
import random

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

# Shared port plumbing (mirrors fql). Used for the CriticEnsemble vmapped ensemble.
from flax_utils import ensemblize

# NOTE (data pipeline): the torch original imported `from torch.utils.data import Dataset, DataLoader`
# at module scope. Per CONVERSION_GUIDE §13 datasets become numpy/fql-style `sample()` objects; this
# nets.py defines only the networks (no Dataset subclass), so the unused torch dataloader import is
# dropped. `random` / `numpy` kept (they were imported here and are numpy-side, not framework).


class Critic(nn.Module):
    obs_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2

    # init: fql-style (not torch-identical). This critic is trained from scratch in JAX.
    @nn.compact
    def __call__(self, obs):
        x = obs
        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        # Hidden layers (repeat num_layers - 1 times; last "hidden" block is output)
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        # Output layer
        x = nn.Dense(1)(x)
        # nn.relu(x)  # (commented out in the torch original)
        return x.squeeze(-1)


class CriticEnsemble(nn.Module):
    obs_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2
    num_heads: int = 5

    @nn.compact
    def __call__(self, obs, aggregate='mean'):
        # torch used nn.ModuleList of `num_heads` Critic copies; here we vmap Critic over a leading
        # ensemble axis (CONVERSION_GUIDE §11). Each member gets independent params.
        critic_module = ensemblize(Critic, self.num_heads)(
            obs_dim=self.obs_dim, hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers
        )
        # `preds` stacks the ensemble on the leading axis 0 (torch stacked on dim=-1); reduce on axis=0
        # to keep the result shape identical to the torch version.
        preds = critic_module(obs)
        if aggregate == 'mean':
            return preds.mean(axis=0)
        elif aggregate == 'min':
            return preds.min(axis=0)
        else:
            return preds
