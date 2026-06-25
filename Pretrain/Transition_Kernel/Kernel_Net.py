'''Gaussian / Mixture-of-Gaussians forward dynamics models (JAX/Flax port).

Inputs (s, a); outputs mean and log_std of s' (and mixture weights for the MoG variant).
'''
import math

import jax
import jax.numpy as jnp
import flax.linen as nn

from JAX_PORT.jax_utils import default_init

# Define the Gaussian forward dynamics model: inputs (s, a), outputs mean and log_std of s'


class RobustTransitionKernel(nn.Module):
    obs_dim: int
    act_dim: int
    num_hidden_layers: int = 2
    hidden_dim: int = 256
    min_log_std: float = -6.0
    max_log_std: float = 4.0
    noise_floor: float = 1e-2

    def setup(self):
        assert self.num_hidden_layers >= 1, 'num_hidden_layers must be >= 1'

        # nn.Sequential -> inline list of submodules (§2). Linen infers in_features.
        layers = []
        # first layer
        layers.append(nn.Dense(self.hidden_dim, kernel_init=default_init()))  # init: fql-style (not torch-identical)
        layers.append(nn.LayerNorm())
        layers.append(nn.relu)

        # additional hidden layers
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Dense(self.hidden_dim, kernel_init=default_init()))  # init: fql-style
            layers.append(nn.relu)

        self.net = layers
        self.mean_head = nn.Dense(self.obs_dim, kernel_init=default_init())  # init: fql-style
        self.log_std_head = nn.Dense(self.obs_dim, kernel_init=default_init())  # init: fql-style

    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        h = x
        for layer in self.net:
            h = layer(h)
        mu = self.mean_head(h)
        raw_log_std = self.log_std_head(h)
        log_std = self.min_log_std + jax.nn.softplus(raw_log_std - self.min_log_std)
        log_std = jnp.clip(log_std, a_max=self.max_log_std)
        return mu, log_std

    def gaussian_nll(self, s_next, mu, log_std):
        # x, mu: (..., obs_dim); log_std: (..., obs_dim)
        var_pred = jnp.exp(2 * log_std)
        var = var_pred + self.noise_floor  # additive floor
        # optional: clamp or clip residuals
        res = s_next - mu
        max_res = 10.0
        res = jnp.clip(res, -max_res, +max_res)
        nll = 0.5 * (jnp.log(2 * math.pi * var) + (res ** 2) / var)
        # sum over state dims, but keep batch dims
        return nll.sum(axis=-1).mean()

    def log_prob(self, s_next, mu, log_std):
        # Compute log prob (not negative) — useful for testing / diagnostics
        var = jnp.exp(2 * log_std) + self.noise_floor
        var = jnp.clip(var, a_min=1e-8)  # Prevent log(0)
        D = s_next.shape[-1]
        # log prob per dimension
        lp = -0.5 * (((s_next - mu) ** 2) / var).sum(axis=-1)
        lp = lp - 0.5 * (D * math.log(2 * math.pi) + 2 * log_std.sum(axis=-1))
        return lp  # tensor of shape batch

    def mahalanobis_distance_squared(self, s_next, s, a):
        '''
        Compute squared Mahalanobis distance D² for batch of transitions.
        Returns tensor of shape (batch_size,)
        '''
        mu, log_std = self(s, a)                  # (batch, obs_dim), (batch, obs_dim)
        var_pred = jnp.exp(2 * log_std)           # predicted variance
        var = var_pred + self.noise_floor         # same as in your log_prob
        var = jnp.clip(var, a_min=1e-8)
        residual = s_next - mu
        # Optional: mild clipping for stability (you already do this in NLL)
        residual = jnp.clip(residual, -10.0, 10.0)
        # Squared Mahalanobis (diagonal covariance)
        D2 = ((residual ** 2) / var).sum(axis=-1)  # sum over state dimensions
        return D2

    def computeD(self, s, a, s_next):
        mu, log_std = self(s, a)
        var = jnp.exp(2 * log_std) + self.noise_floor
        var = jnp.clip(var, a_min=1e-8)  # Prevent log(0)
        D = s_next.shape[-1]
        Temp = 0.5 * (D * math.log(2 * math.pi) + 2 * log_std.sum(axis=-1))
        return Temp


class MoGTransitionKernel(nn.Module):
    obs_dim: int
    act_dim: int
    num_modes: int = 8
    num_hidden_layers: int = 3
    hidden_dim: int = 512
    min_log_std: float = -6.0
    max_log_std: float = 4.0
    noise_floor: float = 1e-4

    def setup(self):
        # Shared Backbone (nn.Sequential -> inline list of submodules, §2).
        layers = [nn.Dense(self.hidden_dim, kernel_init=default_init()),  # init: fql-style
                  nn.LayerNorm(), nn.relu]
        for _ in range(self.num_hidden_layers - 1):
            layers += [nn.Dense(self.hidden_dim, kernel_init=default_init()),  # init: fql-style
                       nn.LayerNorm(), nn.relu]
        self.backbone = layers

        self.head = nn.Dense(self.num_modes * (self.obs_dim * 2 + 1), kernel_init=default_init())  # init: fql-style

    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        h = x
        for layer in self.backbone:
            h = layer(h)
        out = self.head(h).reshape(-1, self.num_modes, 2 * self.obs_dim + 1)

        mu = out[..., :self.obs_dim]
        log_std = out[..., self.obs_dim:2 * self.obs_dim]
        logits = out[..., -1]

        log_std = self.min_log_std + jax.nn.softplus(log_std - self.min_log_std)
        log_std = jnp.clip(log_std, a_max=self.max_log_std)

        weights = jax.nn.softmax(logits, axis=-1)

        return mu, log_std, weights

    def log_prob(self, s_next, mu, log_std, weights):

        var = jnp.exp(2 * log_std) + self.noise_floor
        var = jnp.clip(var, a_min=1e-6)

        residual = jnp.expand_dims(s_next, 1) - mu
        residual = jnp.clip(residual, -10.0, 10.0)

        mahal = ((residual ** 2) / var).sum(axis=-1)
        log_prob_per_mode = -0.5 * (
            mahal
            + self.obs_dim * math.log(2 * math.pi)
            + jnp.log(var).sum(axis=-1)
        )

        # Mixture log probability
        log_prob = jax.scipy.special.logsumexp(
            log_prob_per_mode + jnp.log(weights + 1e-8),
            axis=-1
        )
        return log_prob                     # ← Positive log probability

    def mog_nll(self, s_next, mu, log_std, weights):

        return -self.log_prob(s_next, mu, log_std, weights).mean()
