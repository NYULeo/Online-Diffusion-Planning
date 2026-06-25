'''Reward networks for ODP (JAX/Flax port).

Faithful torch->JAX(FQL-style) conversion of the reward nets: categorical / scalar (Beta) / scalar
ensemble reward models, plus the small MLP utilities. Public API (class/function names, call signatures,
hyperparameters, magic numbers) is preserved exactly; only the framework internals change.
'''
from typing import Any, Optional, List, Tuple, Dict, Sequence, Union

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
import distrax
from scipy.ndimage import gaussian_filter1d, convolve
import warnings

from flax_utils import MLP, TrainState


class CategoricalReward(nn.Module):

    deter_dim: int
    stoch_dim: int
    hidden_units: int = 1024
    num_layers: int = 5
    num_bins: int = 255
    symlog_range: float = 20.0

    @property
    def input_dim(self):
        return self.deter_dim + self.stoch_dim

    def _create_reward_bins(self):
        bins = jnp.linspace(-self.symlog_range, self.symlog_range, self.num_bins)
        return bins

    def symlog(self, x):
        return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

    def symexp(self, x):
        return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

    @nn.compact
    def __call__(self, deter, stoch):
        # reward_bins buffer (non-trainable in torch via register_buffer); recomputed deterministically.
        reward_bins = self._create_reward_bins()

        # Concatenate inputs
        x = jnp.concatenate([deter, stoch], axis=-1)

        # Forward through MLP layers (num_layers Linear + LayerNorm pairs; last is the output layer).
        # init: fql-style (not torch-identical)
        for i in range(self.num_layers):
            if i < self.num_layers - 1:  # Hidden layers
                x = nn.Dense(self.hidden_units)(x)
                x = nn.LayerNorm()(x)
                x = nn.silu(x)  # SiLU activation
            else:  # Output layer
                x = nn.Dense(self.num_bins)(x)

        # Get logits for reward bins
        reward_logits = x

        # Convert to probabilities
        reward_probs = jax.nn.softmax(reward_logits, axis=-1)

        # Compute expected reward value
        reward_mean = jnp.sum(reward_probs * reward_bins[None], axis=-1)

        return reward_logits, reward_probs, reward_mean

    def compute_loss(self, reward_logits, target_rewards):
        # Convert target rewards to symlog space
        target_symlog = self.symlog(target_rewards)

        # Create target bin indices using two-hot encoding
        target_bins = self._rewards_to_bins(target_symlog)

        # Compute cross-entropy loss
        log_probs = jax.nn.log_softmax(reward_logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, target_bins[..., None], axis=-1).squeeze(-1)
        loss = jnp.mean(nll)

        return loss

    def _rewards_to_bins(self, rewards):
        reward_bins = self._create_reward_bins()
        # Clamp rewards to bin range
        rewards = jnp.clip(rewards, -self.symlog_range, self.symlog_range)

        # Find the closest bin for each reward
        distances = jnp.abs(rewards[..., None] - reward_bins[None])
        bin_indices = jnp.argmin(distances, axis=-1)

        return bin_indices

    def predict_reward(self, deter, stoch):
        # torch used `with torch.no_grad()`; in JAX nothing is traced unless inside jax.grad.
        _, _, reward_mean = self.__call__(deter, stoch)
        # Convert from symlog space back to original reward space
        predicted_rewards = self.symexp(reward_mean)
        return predicted_rewards


class ScalarReward(nn.Module):

    obs_dim: int
    act_dim: int
    hidden_units: int = 1024
    num_layers: int = 5
    eps: float = 1e-4

    @property
    def input_dim(self):
        return self.obs_dim + self.act_dim

    @nn.compact
    def __call__(self, obs, act):
        # Concatenate inputs
        x = jnp.concatenate([obs, act], axis=-1)

        # Forward through MLP layers (num_layers Linear + LayerNorm pairs; last is the output layer).
        # init: fql-style (not torch-identical)
        for i in range(self.num_layers):
            if i < self.num_layers - 1:  # Hidden layers
                x = nn.Dense(self.hidden_units)(x)
                x = nn.LayerNorm()(x)
                x = nn.silu(x)  # SiLU activation
            else:  # Output layer - 2 outputs (alpha, beta params)
                x = nn.Dense(2)(x)

        raw_alpha = x[:, 0]  # shape (B,)
        raw_beta = x[:, 1]   # shape (B,)
        # Transform to positive
        alpha = jax.nn.softplus(raw_alpha) + 1e-4
        beta = jax.nn.softplus(raw_beta) + 1e-4

        return alpha, beta

    def predict(self, obs, act, agg: str = 'mean', ci: Optional[float] = None):
        alpha, beta = self.__call__(obs, act)
        dist = distrax.Beta(alpha, beta)  # Beta distribution

        if agg == 'mean':
            pred = alpha / (alpha + beta)               # E[R]
        elif agg == 'mode':
            # Only valid if alpha>1 and beta>1; fall back to mean otherwise
            mask = (alpha > 1) & (beta > 1)
            mode = (alpha - 1) / (alpha + beta - 2)
            mean = alpha / (alpha + beta)
            pred = jnp.where(mask, mode, mean)
        elif agg == 'median_approx':
            # Kerman (2011) approx: (a-1/3)/(a+b-2/3) for a,b>=1
            pred = (alpha - 1 / 3) / (alpha + beta - 2 / 3)
            pred = jnp.clip(pred, 0.0, 1.0)             # guard numerics
        else:
            raise ValueError("agg must be 'mean', 'mode', or 'median_approx'")

        if ci is None:
            return pred
        qlo = (1 - ci) / 2
        qhi = 1 - qlo
        # TODO(checkpoint-bridge): torch used Beta.icdf (inverse CDF / quantile). distrax.Beta exposes no
        # quantile; use the underlying TFP distribution's quantile for an exact match.
        lo = dist.distribution.quantile(jnp.full_like(alpha, qlo))
        hi = dist.distribution.quantile(jnp.full_like(alpha, qhi))
        return pred, (lo, hi)

    def loss(self, obs, act, r):
        alpha, beta = self.__call__(obs, act)
        dist = distrax.Beta(alpha, beta)
        r = jnp.clip(r, self.eps, 1 - self.eps)         # keep inside (0,1)
        nll = -dist.log_prob(r)                         # [B]
        return nll.mean()

    def variance(self, obs, act):
        alpha, beta = self.__call__(obs, act)
        var = (alpha * beta) / (((alpha + beta) ** 2) + (alpha + beta + 1))
        return var

    def compute_reward_gradients(self, obs, act, agg: str = 'mean', return_pred: bool = True):
        # JAX port: gradient of pred-sum w.r.t. the concatenated [obs, act] input (torch autograd.grad).
        x = jnp.concatenate([obs, act], axis=-1)

        def pred_fn(x_in):
            obs_split = x_in[..., :self.obs_dim]
            act_split = x_in[..., self.obs_dim:]
            alpha, beta = self.__call__(obs_split, act_split)
            if agg == 'mean':
                pred = alpha / (alpha + beta)
            elif agg == 'mode':
                mask = (alpha > 1) & (beta > 1)
                mode = (alpha - 1) / (alpha + beta - 2)
                mean = alpha / (alpha + beta)
                pred = jnp.where(mask, mode, mean)
            elif agg == 'median_approx':
                pred = (alpha - 1 / 3) / (alpha + beta - 2 / 3)
                pred = jnp.clip(pred, 0.0, 1.0)
            else:
                raise ValueError("agg must be 'mean', 'mode', or 'median_approx'")
            return pred

        # grad of sum-of-pred w.r.t. x (vectorized, matches torch grad of pred.sum()).
        pred = pred_fn(x)
        grad_input = jax.grad(lambda x_in: pred_fn(x_in).sum())(x)

        if return_pred:
            return grad_input, pred
        else:
            return grad_input


def compute_reward_gradients_per_sample(reward_net, obs, act, agg: str = 'mean'):
    # JAX port (§7): per-sample gradient of pred[i] w.r.t. the i-th concatenated [obs, act] input.
    # `reward_net` is a (model_def, params) pair or a TrainState-style callable exposing obs_dim and a
    # per-sample reward via __call__/apply. We differentiate the scalar reward of a single sample and
    # vmap over the batch so grad_input[i] = d pred[i] / d x[i] (matches torch retain_graph loop).
    x = jnp.concatenate([obs, act], axis=-1)

    def single_pred(x_row):
        # x_row: (D,) single concatenated sample -> add batch dim for the network, take scalar back.
        obs_split = x_row[None, :reward_net.obs_dim]
        act_split = x_row[None, reward_net.obs_dim:]
        alpha, beta = reward_net(obs_split, act_split)
        if agg == 'mean':
            pred = alpha / (alpha + beta)
        elif agg == 'mode':
            mask = (alpha > 1) & (beta > 1)
            mode = (alpha - 1) / (alpha + beta - 2)
            mean = alpha / (alpha + beta)
            pred = jnp.where(mask, mode, mean)
        elif agg == 'median_approx':
            pred = (alpha - 1 / 3) / (alpha + beta - 2 / 3)
            pred = jnp.clip(pred, 0.0, 1.0)
        else:
            raise ValueError("agg must be 'mean', 'mode', or 'median_approx'")
        return pred[0]

    pred = jax.vmap(single_pred)(x)                    # (B,)
    grad_input = jax.vmap(jax.grad(single_pred))(x)    # (B, D)

    return grad_input, pred


class Reward(nn.Module):

    obs_dim: int
    act_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, obs, act, train: bool = False):
        # init: fql-style (not torch-identical). BatchNorm1d -> nn.BatchNorm (needs train flag).
        x = jnp.concatenate([obs, act], axis=-1)
        x = nn.Dense(self.hidden)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


def gaussian_rewards(episode, sigma):
    if sigma > 0:
        reward_raw = episode['rewards']
        reward_smooth = gaussian_filter1d(reward_raw, sigma, mode='nearest')
        episode.update({'rewards_raw': reward_raw, 'rewards': reward_smooth})
        return episode


# from common import util


def get_network(param_shape, deconv=False):
    '''
    Parameters
    ----------
    param_shape: tuple, length:[(4, ), (2, )], optional

    deconv: boolean
        Only work when len(param_shape) == 4.

    Returns a flax.linen module faithful to the torch get_network (§3 conv axis: flax is channels-last).
    '''

    if len(param_shape) == 4:
        if deconv:
            in_channel, kernel_size, stride, out_channel = param_shape
            # torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride). flax infers in_channel.
            return nn.ConvTranspose(features=out_channel, kernel_size=(kernel_size, kernel_size),
                                    strides=(stride, stride))
        else:
            in_channel, kernel_size, stride, out_channel = param_shape
            # torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride). flax infers in_channel.
            return nn.Conv(features=out_channel, kernel_size=(kernel_size, kernel_size),
                           strides=(stride, stride))
    elif len(param_shape) == 2:
        in_dim, out_dim = param_shape
        return nn.Dense(out_dim)
    else:
        raise ValueError(f'Network shape {param_shape} illegal.')


class Swish(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = x * jax.nn.sigmoid(x)
        return x


def get_act_cls(act_fn_name):
    act_fn_name = act_fn_name.lower()
    if act_fn_name == 'tanh':
        act_cls = lambda: jnp.tanh
    elif act_fn_name == 'sigmoid':
        act_cls = lambda: jax.nn.sigmoid
    elif act_fn_name == 'relu':
        act_cls = lambda: jax.nn.relu
    elif act_fn_name == 'identity':
        act_cls = lambda: (lambda x: x)
    elif act_fn_name == 'swish':
        act_cls = lambda: nn.silu
    else:
        raise NotImplementedError(f"Activation functtion {act_fn_name} is not implemented. \
            Possible choice: ['tanh', 'sigmoid', 'relu', 'identity'].")
    return act_cls


class MLPNetwork(nn.Module):

    input_dim: int
    out_dim: int
    hidden_dims: Union[int, list]
    act_fn: Any = 'relu'
    out_act_fn: Any = 'identity'

    def setup(self):
        hidden_dims = self.hidden_dims
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [self.input_dim] + hidden_dims
        networks = []
        act_cls = get_act_cls(self.act_fn)
        out_act_cls = get_act_cls(self.out_act_fn)

        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = get_network([curr_shape, next_shape])
            networks.extend([curr_network, act_cls()])
        final_network = get_network([hidden_dims[-1], self.out_dim])
        networks.extend([final_network, out_act_cls()])
        self.networks = networks

    def __call__(self, input):
        x = input
        for net in self.networks:
            x = net(x)
        return x

    @property
    def weights(self):
        # Dense kernels of the contained linear layers (flax Dense layers in self.networks).
        return [net for net in self.networks if isinstance(net, nn.Dense)]


class SimpleReward(nn.Module):

    obs_dim: int
    act_dim: int
    hidden_dim: int
    hidden_layers: int

    @nn.compact
    def __call__(self, obs, act):
        # init: fql-style (not torch-identical).
        # torch built: [Linear, LayerNorm, SiLU] x (1 + hidden_layers) then a final Linear(hidden_dim, 1).
        x = jnp.concatenate([obs, act], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class EnsembleReward(nn.Module):

    obs_dim: int
    act_dim: int
    hidden_dim: int
    hidden_layers: int
    ensemble_size: int = 5

    def setup(self):
        self.members = [
            SimpleReward(self.obs_dim, self.act_dim, self.hidden_dim, self.hidden_layers)
            for _ in range(self.ensemble_size)
        ]

    def __call__(self, obs, act):
        if obs.ndim == 3:
            # Per-member inputs: (E, B, D)
            assert obs.shape[0] == self.ensemble_size, \
                f'expected leading dim {self.ensemble_size}, got {obs.shape[0]}'
            preds = [m(obs[i], act[i]) for i, m in enumerate(self.members)]
        else:
            # Shared batch: (B, D)
            preds = [m(obs, act) for m in self.members]
        return jnp.stack(preds, axis=0)   # (E, B)

    def predict(self, obs, act, return_std: bool = False):
        # torch used `@torch.no_grad()`; in JAX nothing is traced unless inside jax.grad.
        preds = self.__call__(obs, act)      # (E, B)
        mean = preds.mean(axis=0)
        if return_std:
            std = preds.std(axis=0)
            return mean, std
        return mean
