'''Temporal U-Net planner backbone (ResidualTemporalBlock / TemporalUnet / ValueFunction) —
JAX/Flax (FQL-style) port of the original PyTorch module.'''
from typing import Any, Optional, List, Tuple, Dict, Sequence

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
import einops

# Shared port plumbing (mirrors fql).
from JAX_PORT.jax_utils import (
    MLP, ModuleDict, TrainState, nonpytree_field, default_init, ensemblize,
    target_update, save_agent, restore_agent, supply_rng,
)

from .utils import (
    SinusoidalEmbedding,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


def _mish(x):
    '''Mish activation: x * tanh(softplus(x)) (torch nn.Mish has no direct flax equivalent).'''
    return x * jnp.tanh(jax.nn.softplus(x))


class ResidualTemporalBlock(nn.Module):
    inp_channels: int
    out_channels: int
    embed_dim: int
    horizon: int
    kernel_size: int = 5

    def setup(self):
        # torch: nn.ModuleList([Conv1dBlock(inp, out, k), Conv1dBlock(out, out, k)])
        self.blocks = [
            Conv1dBlock(self.inp_channels, self.out_channels, self.kernel_size),
            Conv1dBlock(self.out_channels, self.out_channels, self.kernel_size),
        ]
        # torch: nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out_channels), Rearrange('batch t -> batch t 1'))
        # The Linear is the only parameterized layer; Mish + Rearrange are applied functionally in __call__.
        # init: fql-style (not torch-identical)
        self.time_mlp_dense = nn.Dense(self.out_channels, kernel_init=default_init())
        # torch: nn.Conv1d(inp_channels, out_channels, 1) if inp != out else nn.Identity()
        if self.inp_channels != self.out_channels:
            self.residual_conv = nn.Conv(features=self.out_channels, kernel_size=(1,),
                                         kernel_init=default_init())  # init: fql-style (not torch-identical)
        else:
            self.residual_conv = None

    def _time_mlp(self, t):
        # nn.Mish() -> nn.Linear(embed_dim, out_channels) -> Rearrange('batch t -> batch t 1')
        out = _mish(t)
        out = self.time_mlp_dense(out)
        return einops.rearrange(out, 'batch t -> batch t 1')

    def __call__(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self._time_mlp(t)
        out = self.blocks[1](out)
        if self.residual_conv is not None:
            # residual 1x1 Conv1d acts on NCL externally; flax Conv is channels-last, so transpose
            # to NLC, conv, transpose back (CONVERSION_GUIDE §3).
            res = jnp.transpose(x, (0, 2, 1))
            res = self.residual_conv(res)
            res = jnp.transpose(res, (0, 2, 1))
        else:
            res = x
        return out + res


class TemporalUnet(nn.Module):
    horizon: int
    transition_dim: int
    # cond_dim
    dim: int = 32
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    attention: bool = False

    def setup(self):
        horizon = self.horizon
        dim = self.dim
        dim_mults = self.dim_mults
        attention = self.attention
        transition_dim = self.transition_dim

        if (horizon % len(dim_mults)) != 0:
            raise ValueError(
                f"Horizon {horizon} must be divisible by the number of dimensions in dim_mults, "
                f"which is {len(dim_mults)}"
            )
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        # torch: nn.Sequential(SinusoidalEmbedding(dim), nn.Linear(dim, dim*4), nn.Mish(), nn.Linear(dim*4, dim))
        # SinusoidalEmbedding + the two Linears are submodules; Mish is applied functionally in __call__.
        self.time_embed = SinusoidalEmbedding(dim)
        self.time_dense1 = nn.Dense(dim * 4, kernel_init=default_init())  # init: fql-style (not torch-identical)
        self.time_dense2 = nn.Dense(dim, kernel_init=default_init())  # init: fql-style (not torch-identical)

        downs = []
        ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            downs.append([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else None,
                Downsample1d(dim_out) if not is_last else None,
            ])

            if not is_last:
                horizon = horizon // 2

        self.downs = downs

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else None
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            ups.append([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else None,
                Upsample1d(dim_in) if not is_last else None,
            ])

            if not is_last:
                horizon = horizon * 2

        self.ups = ups

        # torch: nn.Sequential(Conv1dBlock(dim, dim, kernel_size=5), nn.Conv1d(dim, transition_dim, 1))
        self.final_conv_block = Conv1dBlock(dim, dim, kernel_size=5)
        self.final_conv = nn.Conv(features=transition_dim, kernel_size=(1,),
                                  kernel_init=default_init())  # init: fql-style (not torch-identical)

    def _time_mlp(self, time):
        t = self.time_embed(time)
        t = self.time_dense1(t)
        t = _mish(t)
        t = self.time_dense2(t)
        return t

    def __call__(self, x, conditions, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self._time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            if attn is not None:
                x = attn(x)
            h.append(x)
            if downsample is not None:
                x = downsample(x)

        x = self.mid_block1(x, t)
        if self.mid_attn is not None:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            # print(x.shape)
            x = jnp.concatenate((x, h.pop()), axis=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            if attn is not None:
                x = attn(x)
            if upsample is not None:
                x = upsample(x)

        x = self.final_conv_block(x)
        # final 1x1 Conv1d on NCL externally; transpose to NLC, conv, transpose back (CONVERSION_GUIDE §3).
        x = jnp.transpose(x, (0, 2, 1))
        x = self.final_conv(x)
        x = jnp.transpose(x, (0, 2, 1))

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
