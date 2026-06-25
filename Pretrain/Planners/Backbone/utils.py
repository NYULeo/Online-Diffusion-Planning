'''Backbone utilities for ODP planners (schedules, timestep embeddings, conv blocks, attention,
loss/EMA helpers, checkpoint selection) — JAX/Flax (FQL-style) port of the original PyTorch module.'''
import math
import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops
from typing import Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import pickle
import os
from Dataset import get_PlannerName

from flax_utils import default_init, target_update


#-----------------------------------------------------------------------------#
#---------------------------------- Trainer ----------------------------------#
#-----------------------------------------------------------------------------#

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_params, current_params):
        # JAX params are immutable pytrees; the torch in-place EMA over module parameters
        # `ma = beta*ma + (1-beta)*current` maps to `target_update(current, ma, tau=1-beta)`
        # which computes `(1-beta)*current + beta*ma` (see CONVERSION_GUIDE §5).
        # API-CHANGE: torch took (ma_model, current_model) and mutated in place; here we take
        # and RETURN param pytrees (ma_params, current_params) since JAX params are functional.
        return target_update(current_params, ma_params, tau=1 - self.beta)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



#-----------------------------------------------------------------------------#
#---------------------------------- Schedule ----------------------------------#
#-----------------------------------------------------------------------------#

def cosine_beta(t, s: float = 0.008):
    """
    Continuous-time VP drift g(t)^2 = beta(t) for the cosine schedule.
    Using beta(t) = -2 d/dt log alpha(t) = (pi/(1+s)) * tan(a).
    """
    t = jnp.clip(t, 0.0, 1.0 - 1e-3)
    a = (math.pi / 2.0) * ((t + s) / (1.0 + s))
    return (math.pi / (1.0 + s)) * jnp.tan(a)

def cosine_alpha_sigma(t, s: float = 0.008):
    """Continuous cosine schedule for α(t) and σ(t).

    Reuse of the same function from ``diffusion_transformer.py``.  See
    that module for details.
    """
    t = jnp.clip(t, 0.0, 1.0 - 1e-3)
    factor = (t + s) / (1.0 + s)
    f_t = jnp.cos(    factor * (math.pi / 2)     )** 2
    f0 = jnp.cos( jnp.asarray((s / (1.0 + s)) * (math.pi / 2))  ) ** 2
    alpha_bar = jnp.clip(f_t / f0, 0.0, 1.0 - 1e-3)
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1.0 - alpha_bar)
    return alpha, sigma


def compute_dot_alpha_beta(t, s: float = 0.008):

    eps = 1e-3
    t2 = jnp.clip(t, 0.0, 1.0 - eps)

    # --- compute beta and dot_beta ---
    a = (math.pi / 2.0) * ((t2 + s) / (1.0 + s))
    da_dt = (math.pi / 2.0) * (1.0 / (1.0 + s))
    beta = (math.pi / (1.0 + s)) * jnp.tan(a)
    # derivative: β' = (π/(1+s)) * sec^2(a) * da/dt
    dot_beta = (math.pi / (1.0 + s)) * (1.0 / jnp.cos(a))**2 * da_dt

    # --- compute alpha and dot_alpha ---
    # Match cosine_alpha_sigma exactly
    factor = (t2 + s) / (1.0 + s)
    f_t = jnp.cos(factor * (math.pi / 2))** 2
    # Match cosine_alpha_sigma: use same tensor creation
    f0 = jnp.cos(jnp.asarray((s / (1.0 + s)) * (math.pi / 2)))** 2
    alpha_bar = jnp.clip(f_t / f0, 0.0, 1.0 - 1e-3)
    alpha = jnp.sqrt(alpha_bar)

    # derivative of f_t: d[cos^2(u)]/dt = - sin(2u) * du/dt
    # where u = factor * (math.pi / 2)
    u = factor * (math.pi / 2.0)
    du_dt = da_dt  # same as a's derivative
    dot_f_t = - jnp.sin(2.0 * u) * du_dt
    dot_alpha_bar = dot_f_t / f0
    # α = sqrt(α_bar) => dot α = dot α_bar / (2 * sqrt(α_bar))
    # => dot_alpha = dot_alpha_bar / (2 α)
    dot_alpha = dot_alpha_bar / (2.0 * alpha)

    return alpha, dot_alpha, beta, dot_beta




# 2. Autograd (derivative) version
def compute_dot_autograd(t, s: float = 0.008):

    # Clamp t to the valid range (matches compute_dot_alpha_beta / cosine_*).
    eps = 1e-3
    t2 = jnp.clip(t, 0.0, 1.0 - eps)

    # torch took grad of summed-output w.r.t. t with grad_outputs=ones (i.e. per-element
    # derivative). In JAX this is the elementwise derivative; since cosine_alpha_sigma/cosine_beta
    # act elementwise on t, jax.grad of the summed output gives exactly d/dt at each element
    # (off-diagonal Jacobian terms are zero). See CONVERSION_GUIDE §7.
    alpha_req, _ = cosine_alpha_sigma(t2, s=s)
    beta_req = cosine_beta(t2, s=s)

    dot_alpha = jax.grad(lambda tt: jnp.sum(cosine_alpha_sigma(tt, s=s)[0]))(t2)
    dot_beta = jax.grad(lambda tt: jnp.sum(cosine_beta(tt, s=s)))(t2)

    # detach and return
    return alpha_req, dot_alpha, beta_req, dot_beta




#import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

def _mish(x):
    '''Mish activation: x * tanh(softplus(x)) (torch nn.Mish has no direct flax equivalent).'''
    return x * jnp.tanh(jax.nn.softplus(x))


class PositionalEmbedding(nn.Module):
    dim: int
    max_positions: int = 10000
    endpoint: bool = False

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(start=0, stop=self.dim // 2, dtype=jnp.float32)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        # torch `x.ger(freqs)` is the outer product of 1-D x and 1-D freqs.
        x = jnp.outer(x, freqs.astype(x.dtype))
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x


class UntrainablePositionalEmbedding(nn.Module):
    dim: int
    max_positions: int = 10000
    endpoint: bool = False

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(start=0, stop=self.dim // 2, dtype=jnp.float32)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = jnp.einsum('...i,j->...ij', x, freqs.astype(x.dtype))
        # x = x.ger(freqs.to(x.dtype))
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.einsum('...i,j->...ij', x, emb.astype(x.dtype))
        # emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    dim: int
    scale: Any = 16

    @nn.compact
    def __call__(self, x, *, rng=None):
        # `freqs` is a frozen (requires_grad=False) buffer initialized from randn * scale.
        # In linen it is a non-trainable variable in the 'consts' collection. The init draws a
        # fresh sample at .init time; numerics are not torch-identical (init: fql-style).
        # API-CHANGE: added keyword-only `rng=` to allow seeding the const buffer init (otherwise
        # uses the module's 'params' rng stream).
        def _freqs_init(key, shape, dtype=jnp.float32):
            return jax.random.normal(key, shape, dtype) * self.scale
        freqs = self.variable(
            'consts', 'freqs',
            lambda: _freqs_init(rng if rng is not None else self.make_rng('params'), (self.dim // 8,)),
        ).value
        emb = jnp.einsum('...i,j->...ij', x, (2 * np.pi * freqs).astype(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], -1)
        # torch: nn.Sequential(nn.Linear(dim//4, dim), nn.Mish(), nn.Linear(dim, dim))
        emb = nn.Dense(self.dim, kernel_init=default_init())(emb)  # init: fql-style (not torch-identical)
        emb = _mish(emb)
        emb = nn.Dense(self.dim, kernel_init=default_init())(emb)  # init: fql-style (not torch-identical)
        return emb


class UntrainableFourierEmbedding(nn.Module):
    dim: int
    scale: Any = 16

    @nn.compact
    def __call__(self, x, *, rng=None):
        # Frozen (requires_grad=False) buffer; non-trainable 'consts' variable. init: fql-style.
        # API-CHANGE: added keyword-only `rng=` to allow seeding the const buffer init.
        def _freqs_init(key, shape, dtype=jnp.float32):
            return jax.random.normal(key, shape, dtype) * self.scale
        freqs = self.variable(
            'consts', 'freqs',
            lambda: _freqs_init(rng if rng is not None else self.make_rng('params'), (self.dim // 2,)),
        ).value
        emb = jnp.einsum('...i,j->...ij', x, (2 * np.pi * freqs).astype(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], -1)
        return emb


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
}



class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        # External shape is NCL (B, C, L) to match the torch Conv1d API. flax Conv is channels-last,
        # so transpose to NLC, conv, transpose back (CONVERSION_GUIDE §3).
        h = jnp.transpose(x, (0, 2, 1))
        h = nn.Conv(features=self.dim, kernel_size=(3,), strides=(2,), padding=[(1, 1)],
                    kernel_init=default_init())(h)  # init: fql-style (not torch-identical)
        return jnp.transpose(h, (0, 2, 1))

class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        h = jnp.transpose(x, (0, 2, 1))
        h = nn.ConvTranspose(features=self.dim, kernel_size=(4,), strides=(2,), padding=[(1, 1)],
                             kernel_init=default_init())(h)  # init: fql-style (not torch-identical)
        return jnp.transpose(h, (0, 2, 1))

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    inp_channels: int
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        # x is NCL (B, C, L). Conv on NLC (CONVERSION_GUIDE §3), then GroupNorm (flax GroupNorm
        # normalizes over the last/channel axis, so apply on NLC), then Mish, then back to NCL.
        h = jnp.transpose(x, (0, 2, 1))
        h = nn.Conv(features=self.out_channels, kernel_size=(self.kernel_size,), strides=(1,),
                    padding=[(self.kernel_size // 2, self.kernel_size // 2)],
                    kernel_init=default_init())(h)  # init: fql-style (not torch-identical)
        # torch reshaped `b c h -> b c 1 h` then GroupNorm(n_groups, out_channels); here channels are
        # last (NLC) so flax GroupNorm groups them directly.
        h = nn.GroupNorm(num_groups=self.n_groups)(h)
        h = _mish(h)
        return jnp.transpose(h, (0, 2, 1))

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    fn: nn.Module

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        # x is NCL (B, C, L); torch normalized over the channel axis (dim=1) with learned per-channel
        # scale `g` and shift `b` of shape (1, dim, 1).
        g = self.param('g', lambda key: jnp.ones((1, self.dim, 1)))
        b = self.param('b', lambda key: jnp.zeros((1, self.dim, 1)))
        var = jnp.var(x, axis=1, keepdims=True)
        mean = jnp.mean(x, axis=1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * g + b

class PreNorm(nn.Module):
    dim: int
    fn: nn.Module

    @nn.compact
    def __call__(self, x):
        x = LayerNorm(self.dim)(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        scale = self.dim_head ** -0.5
        hidden_dim = self.dim_head * self.heads
        # x is NCL (B, C, L). The torch 1x1 Conv1d acts as a pointwise channel projection; with
        # kernel_size=1 it is equivalent to a Dense over the channel axis. Transpose to NLC, project,
        # transpose back so the einsum logic below stays identical (CONVERSION_GUIDE §3).
        x_nlc = jnp.transpose(x, (0, 2, 1))
        qkv = nn.Conv(features=hidden_dim * 3, kernel_size=(1,), use_bias=False,
                      kernel_init=default_init())(x_nlc)  # init: fql-style (not torch-identical)
        qkv = jnp.transpose(qkv, (0, 2, 1))
        qkv = jnp.split(qkv, 3, axis=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * scale

        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum('b h d n, b h e n -> b h d e', k, v)

        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        out_nlc = jnp.transpose(out, (0, 2, 1))
        out_nlc = nn.Conv(features=self.dim, kernel_size=(1,),
                          kernel_init=default_init())(out_nlc)  # init: fql-style (not torch-identical)
        return jnp.transpose(out_nlc, (0, 2, 1))


def apply_conditioning(x, conditions, state_dim):
    # torch in-place: x[:, 0, :state_dim] = conditions.clone() (CONVERSION_GUIDE §9).
    x = x.at[:, 0, :state_dim].set(conditions)
    return x



#-----------------------------------------------------------------------------#
#---------------------------------- Selection --------------------------------#
#-----------------------------------------------------------------------------#
def get_pretrained_planner(dataset_name, specific_dataset, checkpoint_steps,
                           task_id=None):
    planner_name = get_PlannerName(dataset_name, specific_dataset, task_id)
    checkpoint_path = (
        f"./Pretrain/Planners/{dataset_name}/{specific_dataset}/Models/"
        f"{planner_name}_{checkpoint_steps}.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # TODO(checkpoint-bridge): the existing pretrained planner checkpoints are torch `.pt` files
    # holding {'ema': state_dict}. Originally: `torch.load(checkpoint_path, map_location='cpu')['ema']`.
    # Porting the ingest requires a torch state_dict -> flax param-tree remapper (Dense weight.T ->
    # kernel, bias -> bias, LayerNorm weight -> scale; see CONVERSION_GUIDE §10). Until that bridge
    # exists, this loader cannot return a JAX-native param pytree without torch installed.
    raise NotImplementedError(
        'get_pretrained_planner: torch .pt checkpoint ingest not yet ported to JAX. '
        'See TODO(checkpoint-bridge) and CONVERSION_GUIDE §10 for the required state_dict remap.'
    )




# Add this class before SDETrainer
class LossTracker:
    """Class to track and plot training losses"""

    def __init__(self, save_dir: str = "./logs/"):
        self.save_dir = save_dir
        self.losses = []
        self.steps = []
        self.learning_rates = []
        os.makedirs(save_dir, exist_ok=True)

    def log_loss(self, step: int, loss: float, lr: Optional[float] = None):
        """Log a loss value at a specific step"""
        self.steps.append(step)
        self.losses.append(loss)
        if lr is not None:
            self.learning_rates.append(lr)

    def save_logs(self, filename: str = "training_logs.pkl"):
        """Save logs to pickle file"""
        log_data = {
            'steps': self.steps,
            'losses': self.losses,
            'learning_rates': self.learning_rates
        }
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(log_data, f)
        print(f"Logs saved to {save_path}")

    def plot_loss_curve(self,
                       save_path: Optional[str] = None,
                       title: str = "Training Loss Curve",
                       show_lr: bool = False,
                       smooth_window: int = 50):
        """Plot the loss curve with optional smoothing and learning rate"""

        if not self.losses:
            print("No loss data to plot!")
            return

        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Convert to numpy arrays
        steps = np.array(self.steps)
        losses = np.array(self.losses)

        # Plot raw loss
        ax1.plot(steps, losses, alpha=0.3, color='blue', label='Raw Loss')

        # Plot smoothed loss
        if len(losses) > smooth_window:
            smoothed_losses = self._smooth_curve(losses, smooth_window)
            ax1.plot(steps, smoothed_losses, color='red', linewidth=2, label=f'Smoothed Loss (window={smooth_window})')

        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot learning rate on secondary axis if available
        if show_lr and self.learning_rates:
            ax2 = ax1.twinx()
            ax2.plot(steps, self.learning_rates, color='green', alpha=0.7, label='Learning Rate')
            ax2.set_ylabel('Learning Rate', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(self.save_dir, "loss_curve.png")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to {save_path}")

        # Show plot
        plt.show()

        return fig

    def _smooth_curve(self, data: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing to the data"""
        if window <= 1:
            return data

        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        # Pad the beginning to maintain the same length
        padded = np.full_like(data, np.nan)
        padded[window-1:] = smoothed
        return padded




def getName(env_name, specific_env):
     if(env_name == 'kitchen'):
          return 'Kitchen'
     elif(env_name == 'pointmaze'):
          if specific_env == 'open_dense':
               return 'PointMaze_OpenDense'
          elif specific_env == 'umaze':
               return 'PointMaze_Umaze'
          elif specific_env == 'large_dense':
               return 'PointMaze_LargeDense'
          elif specific_env== 'medium':
               return 'PointMaze_Medium'
          elif specific_env == 'umaze_dense':
               return 'PointMaze_UmazeDense'
          elif specific_env == 'large':
               return 'PointMaze_Large'
          elif specific_env == 'open':
               return 'PointMaze_Open'
          else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'antmaze'):
          if specific_env == 'medium_play':
               return 'AntMaze_MediumPlay'
          elif specific_env == 'umaze_diverse':
               return 'AntMaze_UmazeDiverse'
          elif specific_env == 'large_diverse':
               return 'AntMaze_LargeDiverse'
          elif specific_env == 'large_play':
               return 'AntMaze_LargePlay'
          elif specific_env == 'medium_diverse':
               return 'AntMaze_MediumDiverse'
          elif specific_env == 'umaze':
               return 'AntMaze_Umaze'
          else:
              raise ValueError(f"Invalid Dataset name: {specific_env}")
     elif(env_name == 'cube'):
         if specific_env == 'single-play':
             return 'Cube_SinglePlay'
         elif specific_env == 'single-noisy':
            return 'Cube_SingleNoisy'
         elif specific_env == 'double-play':
            return 'Cube_DoublePlay'
         elif specific_env == 'double-noisy':
            return 'Cube_DoubleNoisy'
         elif specific_env == 'triple-play':
            return 'Cube_TriplePlay'
         elif specific_env == 'triple-noisy':
            return 'Cube_TripleNoisy'
         elif specific_env == 'quadruple-play':
            return 'Cube_QuadruplePlay'
         elif specific_env == 'quadruple-noisy':
            return 'Cube_QuadrupleNoisy'
         else:
            raise ValueError(f"Invalid cube dataset name: {specific_env}")

     elif(env_name == 'ogpointmaze'):
         if specific_env == 'medium':
             return 'OG2DMaze_Medium'
         elif specific_env == 'large':
            return 'OG2DMaze_Large'
         elif specific_env == 'giant':
            return 'OG2DMaze_Giant'
         else:
            raise ValueError(f"Invalid ogpointmaze dataset name: {specific_env}")
     else:
           raise ValueError(f"Invalid environment name: {env_name}")
