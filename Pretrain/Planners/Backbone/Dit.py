'''DiT (Diffusion Transformer) 1-D backbone for ODP planners — JAX/Flax (FQL-style) port of the
original PyTorch module.'''
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from .utils import SinusoidalEmbedding
from .BaseDiffusion import BaseNNDiffusion


def modulate(x, shift, scale):
    # torch: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale[:, None]) + shift[:, None]


def _approx_gelu(x):
    '''tanh-approximate GELU (torch `nn.GELU(approximate="tanh")`).'''
    return nn.gelu(x, approximate=True)


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    hidden_size: int
    n_heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, t, *, train: bool = False):
        # torch norm1/norm2: LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6).
        norm1 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        # adaLN_modulation: SiLU -> Linear(hidden_size, hidden_size * 6); zero-init the Linear so the
        # block starts as identity (adaLN-Zero), matching the torch `initialize_weights` zero-out.
        ada = nn.silu(t)
        ada = nn.Dense(self.hidden_size * 6, kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros)(ada)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(ada, 6, axis=1)

        # torch: nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True) called as
        # self.attn(x, x, x)[0] (self-attention). linen MultiHeadDotProductAttention with dropout that
        # needs a 'dropout' rng when not deterministic (CONVERSION_GUIDE §2, §6/§8).
        h = modulate(norm1(x), shift_msa, scale_msa)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, dropout_rate=self.dropout,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(h, h, deterministic=not train)
        x = x + gate_msa[:, None] * attn_out

        # mlp: Linear(hidden_size, hidden_size*4) -> approx_gelu -> Dropout -> Linear(hidden_size*4, hidden_size)
        m = modulate(norm2(x), shift_mlp, scale_mlp)
        m = nn.Dense(self.hidden_size * 4, kernel_init=nn.initializers.xavier_uniform(),
                     bias_init=nn.initializers.zeros)(m)
        m = _approx_gelu(m)
        m = nn.Dropout(rate=self.dropout)(m, deterministic=not train)
        m = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform(),
                     bias_init=nn.initializers.zeros)(m)
        x = x + gate_mlp[:, None] * m
        return x


class FinalLayer1d(nn.Module):
    hidden_size: int
    out_dim: int

    @nn.compact
    def __call__(self, x, t):
        # norm_final: LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6).
        norm_final = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        # adaLN_modulation: SiLU -> Linear(hidden_size, 2 * hidden_size); zero-init (adaLN-Zero).
        ada = nn.silu(t)
        ada = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros)(ada)
        shift, scale = jnp.split(ada, 2, axis=1)
        x = modulate(norm_final(x), shift, scale)
        # output linear: zero-init weight and bias (torch `initialize_weights` zeros the final linear).
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros)(x)


class DiT1d(BaseNNDiffusion):
    in_dim: int
    emb_dim: int
    d_model: int = 384
    n_heads: int = 6
    depth: int = 12
    dropout: float = 0.0
    timestep_emb_type: str = 'positional'
    timestep_emb_params: Optional[dict] = None

    def setup(self):
        # BaseNNDiffusion.setup() builds self.map_noise from (emb_dim, timestep_emb_type,
        # timestep_emb_params). Linen does not chain setup() automatically, so call it explicitly.
        super().setup()
        # torch: x_proj = Linear(in_dim, d_model). init: xavier_uniform weight, zero bias (basic_init).
        self.x_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
        # torch: map_emb = Sequential(Linear(emb_dim, d_model), Mish, Linear(d_model, d_model), Mish).
        # initialize_weights overrides the two Linear weights with normal(std=0.02); biases follow
        # basic_init (xavier weight then zero bias, but the weight is then overwritten by normal).
        self.map_emb_0 = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02),
                                  bias_init=nn.initializers.zeros)
        self.map_emb_2 = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02),
                                  bias_init=nn.initializers.zeros)
        self.pos_emb = SinusoidalEmbedding(self.d_model)
        # torch kept a python-side `pos_emb_cache`; linen modules are frozen, so the positional
        # embedding (a deterministic function of the horizon) is recomputed each call instead.
        self.blocks = [DiTBlock(self.d_model, self.n_heads, self.dropout) for _ in range(self.depth)]
        self.final_layer = FinalLayer1d(self.d_model, self.in_dim)

    def _map_emb(self, emb):
        '''map_emb forward: Linear -> Mish -> Linear -> Mish (torch nn.Mish = x*tanh(softplus(x))).'''
        emb = self.map_emb_0(emb)
        emb = emb * jnp.tanh(jax.nn.softplus(emb))
        emb = self.map_emb_2(emb)
        emb = emb * jnp.tanh(jax.nn.softplus(emb))
        return emb

    def __call__(self,
                 x: jnp.ndarray, noise: jnp.ndarray,
                 condition: Optional[jnp.ndarray] = None, *, train: bool = False):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        # torch cached pos_emb for a given horizon; recompute it here (deterministic in horizon).
        pos_emb = self.pos_emb(jnp.arange(x.shape[1]))

        x = self.x_proj(x) + pos_emb[None,]
        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + jnp.zeros_like(emb)
        emb = self._map_emb(emb)

        for block in self.blocks:
            x = block(x, emb, train=train)
        x = self.final_layer(x, emb)
        return x


class DiT1Ref(DiT1d):
    in_dim: int
    emb_dim: int
    d_model: int = 384
    n_heads: int = 6
    depth: int = 12
    dropout: float = 0.0
    timestep_emb_type: str = 'positional'
    timestep_emb_params: Optional[dict] = None

    def setup(self):
        super().setup()
        # torch: cross_attns = ModuleList([MultiheadAttention(d_model, n_heads, batch_first=True)
        # for _ in range(depth)]); cross-attention has no dropout in the torch construction.
        self.cross_attns = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, dropout_rate=0.0,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )
            for _ in range(self.depth)
        ]

    def __call__(self,
                 x: jnp.ndarray, noise: jnp.ndarray,
                 condition: Optional[jnp.ndarray] = None, *, train: bool = False):
        """
        Input:
            x:          (b, horizon, in_dim * 2), where the first half is the reference signal
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        pos_emb = self.pos_emb(jnp.arange(x.shape[1]))

        x_ref, x = jnp.split(x, 2, axis=-1)
        x_ref_bkp = x_ref  # jax arrays are immutable; the torch `.clone()` is unnecessary.

        x_ref = self.x_proj(x_ref) + pos_emb[None,]
        x = self.x_proj(x) + pos_emb[None,]
        emb = self.map_noise(noise)

        if condition is not None:
            emb = emb + condition
        emb = self._map_emb(emb)

        for cross_attn, block in zip(self.cross_attns, self.blocks):
            # torch: x, _ = cross_attn(x, x_ref, x_ref) -> query=x, key=value=x_ref.
            x = cross_attn(x, x_ref, deterministic=not train)
            x = block(x, emb, train=train)
        x = self.final_layer(x, emb)
        return jnp.concatenate([x_ref_bkp, x], axis=-1)
