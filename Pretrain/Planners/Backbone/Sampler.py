'''Diffusion samplers for the ODP planner backbone (reverse-SDE / Euler-Karras / hybrid).

JAX/Flax (FQL-style) port of the original torch module. The score model is called as a frozen,
no-grad network (a `jax_utils.TrainState` whose `__call__` omits `params=`, mirroring torch's
`@torch.no_grad()` samplers). RNG is threaded explicitly: every sampler takes a keyword-only
`rng=None` and splits a fresh subkey for the initial noise and for each per-step noise injection,
so two runs with the same key reproduce.
'''
from __future__ import annotations
import math
from typing import List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

#from .UNet import  TemporalUnet
from .Dit import DiT1d
from .utils import cosine_alpha_sigma, cosine_beta


# SPEED (logic-identical): cache a jitted frozen forward per planner. The per-step `score_model(x, t)` inside
# sample_euler_karras was running EAGERLY (op-by-op GPU dispatch), which is the dominant cost in the sequential
# plan loops of rollout + train_critic_with_planner2. Jitting `model_def.apply` (compile once per planner,
# then one fused call/step) gives identical numbers — mirrors the proven acc_adjoint_matching pattern
# (`_score_apply = jax.jit(model_def.apply)`).
_SCORE_APPLY_JIT = {}
def _jitted_score_apply(model_def):
    k = id(model_def)
    fn = _SCORE_APPLY_JIT.get(k)
    if fn is None:
        fn = jax.jit(model_def.apply)
        _SCORE_APPLY_JIT[k] = fn
    return fn


def clip_actions(x: jnp.ndarray, d_s: int) -> jnp.ndarray:
    actions = jnp.clip(x[..., d_s:], -1.0, 1.0)
    x = x.at[..., d_s:].set(actions)
    return x


def sample_reverse_sde(
    s0: np.ndarray,
    score_model: DiT1d,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int,
    eta: float,
    device: Optional[str] = None,
    *,
    rng=None,
) -> np.ndarray:
    s0_t = jnp.asarray(s0, dtype=jnp.float32)
    if ( (s0_t.shape[0] != d_s)   ):
        raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")
    dim = d_s + d_a
    t_asc = jnp.linspace(1.0, 0.0, steps_T + 1)
    #t_asc = jnp.linspace(0.0, 1.0, steps_T + 1)
    beta = cosine_beta(t_asc, s=0.008)
    alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)

    # Initialize x_T ~ N(0, I) with shape (horizon, dim)
    rng, k = jax.random.split(rng)
    x = jax.random.normal(k, (horizon, dim), dtype=jnp.float32)[None]
    conditions = s0_t[None]
    mask = jnp.zeros((1, horizon, dim), dtype=jnp.float32)
    mask = mask.at[:, 0, :d_s].set(1)
    y = jnp.zeros((1, horizon, dim), dtype=jnp.float32)
    y = y.at[:, 0, :d_s].set(conditions)
    #x = apply_conditioning(x, conditions, d_s)
    x = mask * y + (1 - mask) * x



    for i in range(len(t_asc) - 1):
        t_now, t_next = t_asc[i], t_asc[i + 1]
        dt = float(t_next - t_now)
        g2_val = float(beta[i])
        drift = -0.5 * g2_val * x
        #t_tensor = t_now.repeat(batch)
        score = score_model(x, t_now[None])


        if eta > 0:
            rng, k = jax.random.split(rng)
            noise = jax.random.normal(k, x.shape, dtype=x.dtype)
            noise_scale = eta * math.sqrt(g2_val * (-dt))
            x = x + ((drift - g2_val * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - g2_val * score) * dt

        x = mask * y + (1 - mask) * x

        x = clip_actions(x, d_s)


    #x = clip_actions(x, d_s)
    return np.asarray(x.squeeze(0))



# ================================
# 4. EULER + KARRAS with β(t) (50 steps)
# ================================
def sample_euler_karras(
    s0: np.ndarray,
    score_model,
    d_s: int,
    d_a: int,
    horizon: int,
    num_steps: int = 50,
    num_karras: int = 5,
    eta: float = 1.0,
    device: Optional[str] = None,
    *,
    rng=None,
) -> np.ndarray:
    s0_t = jnp.asarray(s0, dtype=jnp.float32)
    if s0_t.shape[0] != d_s:
        raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")

    dim = d_s + d_a

    # Karras β(t) + σ(t)
    t_grid, beta_1, sigma_grid = karras_beta_schedule(num_steps, device=device)
    #t_grid, beta_1, _, sigma_grid =  karras_cosine_interpolated_beta(num_steps, device=device)

    beta_2 = cosine_beta(t_grid, s=0.008)

    # Initialize x_T
    rng, k = jax.random.split(rng)
    x = jax.random.normal(k, (1, horizon, dim)) * sigma_grid[0]
    #x2 = jax.random.normal(k2, (1, horizon, dim))

    # Conditioning
    mask = jnp.zeros((1, horizon, dim))
    mask = mask.at[:, 0, :d_s].set(1.0)
    y = jnp.zeros_like(x)
    y = y.at[:, 0, :d_s].set(s0_t[None])
    x = mask * y + (1 - mask) * x



    for i in range(num_steps):
        t_now = t_grid[i]
        t_next = t_grid[i + 1] if i < num_steps - 1 else 0.0
        dt = float(t_next - t_now)
        if( i < num_karras ):
            beta_now = float(beta_1[i])
        else:
            beta_now = float(beta_2[i])

        # Drift
        drift = -0.5 * beta_now * x

        # Score — jitted frozen forward (compile-once, fused) instead of eager per-op dispatch. Identical
        # numbers to `score_model(x, t_now[None])` (== model_def.apply({'params': params}, x, t)). Guarded so
        # any non-TrainState score_model still works.
        if hasattr(score_model, 'model_def') and hasattr(score_model, 'params'):
            score = _jitted_score_apply(score_model.model_def)({'params': score_model.params}, x, t_now[None])
        else:
            score = score_model(x, t_now[None])

        # Euler step
        if eta > 0:
            rng, k = jax.random.split(rng)
            noise = jax.random.normal(k, x.shape, dtype=x.dtype)
            noise_scale = eta * math.sqrt(beta_now * (-dt))
            x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - beta_now * score) * dt

        # Conditioning
        x = mask * y + (1 - mask) * x
        x = clip_actions(x, d_s)

    return np.asarray(x.squeeze(0))


# ------------------------------------------------------------------ #
# 1. Hybrid schedule (Karras speed + cosine smoothness)
# ------------------------------------------------------------------ #
def hybrid_karras_cosine_schedule(
    num_steps: int = 50,
    sigma_min: float = 0.01,
    sigma_max: float = 30.0,
    cosine_s: float = 0.008,
    device: str = "cpu",
):
    """
    Returns
    -------
    t_grid      : Tensor[N+1]   in [1,0]
    beta_grid   : Tensor[N+1]   β(t)  (physics-consistent)
    sigma_grid  : Tensor[N+1]   σ(t) = √β(t)
    eta_grid    : Tensor[N+1]   η(t) = β(t)/2   ← memory-less
    """
    t = jnp.linspace(1.0, 0.0, num_steps + 1)

    # ---- Karras log-linear σ_k(t) (fast decay) ----
    rho = jnp.log(jnp.asarray(sigma_max / sigma_min))
    sigma_k = sigma_min * jnp.exp(rho * t)                   # σ_k(t)

    # ---- Target cosine α_bar(t) (smooth) ----
    f = jnp.cos((t + cosine_s) / (1.0 + cosine_s) * jnp.pi / 2) ** 2
    alpha_bar_target = f / f[0]                               # [0,1]

    # ---- VP marginals forced to the target ----
    sigma_sq = 1.0 - alpha_bar_target
    sigma = jnp.sqrt(sigma_sq)

    # ---- Physics-consistent β(t) from variance ODE ----
    d_sigma_sq = jnp.gradient(sigma_sq, t)                    # central diff
    beta = d_sigma_sq / alpha_bar_target
    beta = jnp.clip(beta, 1e-6, None)                         # numerical safety

    eta = beta / 2.0                                          # memory-less

    return t, beta, sigma, eta


# ------------------------------------------------------------------ #
# 2. Pure Karras schedule (for the first `num_karras` steps)
# ------------------------------------------------------------------ #
def karras_beta_schedule(
    num_steps: int = 50,
    sigma_min: float = 0.01,
    sigma_max: float = 30.0,
    device: str = "cpu",
):
    t = jnp.linspace(1.0, 0.0, num_steps + 1)
    rho = jnp.log(jnp.asarray(sigma_max / sigma_min))
    sigma_k = sigma_min * jnp.exp(rho * t)

    alpha = 1.0 / jnp.sqrt(1.0 + sigma_k ** 2)
    sigma = sigma_k * alpha

    # β(t) from VP-SDE marginals via numerical diff (dσ²/dt = β(t)·(1−σ²)). This MATCHES torch
    # Pretrain/Planners/Backbone/Sampler.py and the faithful Finetuning/utils.py copy. The previous
    # closed-form `2ρ·σ_k/(1+σ_k²)` DIVERGED from torch -> changed the diffusion sampler (drift/score/noise)
    # in the first num_karras steps of sample_euler_karras, affecting rollout + planner2 plan generation.
    sigma_sq = sigma ** 2
    d_sigma_sq = jnp.diff(sigma_sq, axis=0)
    dt = jnp.diff(t, axis=0)
    beta = d_sigma_sq / (1 - sigma_sq[:-1]) / dt
    beta = jnp.concatenate([beta, beta[-1][None]])  # pad last

    return t, beta, sigma


# ------------------------------------------------------------------ #
# 3. Adapted sampler (exact copy of your API)
# ------------------------------------------------------------------ #
def sample_euler_karras2(
    s0: np.ndarray,
    score_model,
    d_s: int,
    d_a: int,
    horizon: int,
    num_steps: int = 50,
    num_karras: int = 5,
    eta: float = 1.0,
    device: Optional[str] = None,
    *,
    rng=None,
) -> np.ndarray:
    s0_t = jnp.asarray(s0, dtype=jnp.float32)
    if s0_t.shape[0] != d_s:
        raise ValueError(f"s0 should have shape ({d_s},), got {s0_t.shape}")

    dim = d_s + d_a

    # ---------- 1. Schedules ----------
    # pure Karras for the first `num_karras` steps
    t_grid_k, beta_k, _ = karras_beta_schedule(num_steps, device=device)

    # hybrid (Karras speed + cosine smoothness) for the rest
    t_grid_h, beta_h, sigma_h, eta_h = hybrid_karras_cosine_schedule(
        num_steps, device=device
    )

    # align grids (both have the same t_grid)
    t_grid = t_grid_k

    # ---------- 2. Initial noise ----------
    rng, k = jax.random.split(rng)
    x = jax.random.normal(k, (1, horizon, dim)) * sigma_h[0]

    # ---------- 3. Conditioning ----------
    mask = jnp.zeros((1, horizon, dim))
    mask = mask.at[:, 0, :d_s].set(1.0)
    y = jnp.zeros_like(x)
    y = y.at[:, 0, :d_s].set(s0_t[None])
    x = mask * y + (1.0 - mask) * x

    # ---------- 4. Euler integration ----------
    for i in range(num_steps):
        t_now = t_grid[i]
        t_next = t_grid[i + 1] if i < num_steps - 1 else 0.0
        dt = float(t_next - t_now)

        # ----- β(t) switch -----
        if i < num_karras:
            beta_now = float(beta_k[i])
        else:
            beta_now = float(beta_h[i])

        # ----- drift & score -----
        drift = -0.5 * beta_now * x
        score = score_model(x, t_now[None])

        # ----- Euler step (with optional stochasticity) -----
        if eta > 0.0:
            rng, k = jax.random.split(rng)
            noise = jax.random.normal(k, x.shape, dtype=x.dtype)
            noise_scale = eta * math.sqrt(beta_now * (-dt))
            x = x + (drift - beta_now * score) * dt + noise_scale * noise
        else:
            x = x + (drift - beta_now * score) * dt

        # ----- conditioning & action clipping -----
        x = mask * y + (1.0 - mask) * x
        x = clip_actions(x, d_s)

    return np.asarray(x.squeeze(0))
