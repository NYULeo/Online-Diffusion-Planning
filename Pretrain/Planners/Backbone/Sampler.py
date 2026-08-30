from __future__ import annotations
import math
from re import A
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .UNet import  TemporalUnet
from .Dit import DiT1d
from .utils import cosine_alpha_sigma, cosine_beta
from torch import Tensor

def clip_actions(x: torch.Tensor, d_s: int) -> torch.Tensor:
    actions = torch.clamp(x[..., d_s:], -1.0, 1.0)
    x[..., d_s:] = actions
    return x

@torch.no_grad()
def sample_reverse_sde(
    s0: np.ndarray,
    score_model: DiT1d,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int,
    eta: float,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0_t = torch.tensor(s0, device=device, dtype=torch.float32)
    if ( (s0_t.shape[0] != d_s)   ):
        raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")
    dim = d_s + d_a
    t_asc = torch.linspace(1.0, 0.0, steps_T + 1, device=device)
    #t_asc = torch.linspace(0.0, 1.0, steps_T + 1, device=device)
    beta = cosine_beta(t_asc, s=0.008)
    alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)
    
    # Initialize x_T ~ N(0, I) with shape (horizon, dim)
    x = torch.randn(horizon, dim, dtype=torch.float32, device=device).unsqueeze(0)
    conditions = s0_t.unsqueeze(0)
    mask = torch.zeros((1, horizon, dim), dtype = torch.float32, device = device)
    mask[:, 0, :d_s] = 1
    y = torch.zeros((1, horizon, dim), dtype = torch.float32, device = device)
    y[:, 0, :d_s] = conditions.clone()
    #x = apply_conditioning(x, conditions, d_s)
    x = mask * y + (1 - mask) * x
    
    

    for i in range(len(t_asc) - 1):
        t_now, t_next = t_asc[i], t_asc[i + 1]
        dt = (t_next - t_now).item()
        g2_val = beta[i].item()
        drift = -0.5 * g2_val * x
        #t_tensor = t_now.repeat(batch)
        score = score_model(x, t_now.unsqueeze(0))
        
       
        if eta > 0:
            noise = torch.randn_like(x)
            noise_scale = eta * math.sqrt(g2_val * (-dt))
            x = x + ((drift - g2_val * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - g2_val * score) * dt
        
        x = mask * y + (1 - mask) * x

        x = clip_actions(x, d_s)
        
        
    #x = clip_actions(x, d_s)
    return x.squeeze(0).detach().cpu().numpy()



# ================================
# 2. KARRAS TIMESTEPS + β(t) from VP-SDE
# ================================
def karras_beta_schedule(
    num_steps: int = 50,
    sigma_min: float = 0.01,
    sigma_max: float = 30.0,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: t_grid, beta_grid, sigma_grid
    beta(t) computed from VP-SDE marginals using Karras timesteps.
    """
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    sigma_k = sigma_min * (sigma_max / sigma_min) ** t
    alpha = 1.0 / torch.sqrt(1.0 + sigma_k**2)
    sigma = sigma_k * alpha

    # Compute β(t) from dσ²/dt = β(t) * σ²(t)
    # From VP-SDE: dσ²/dt = β(t) * (1 - σ²(t))
    # But we use numerical diff for stability
    
    sigma_sq = sigma**2
    d_sigma_sq = torch.diff(sigma_sq, dim=0)
    dt = torch.diff(t, dim=0)
    beta = d_sigma_sq / (1 - sigma_sq[:-1]) / dt
    beta = torch.cat([beta, beta[-1].unsqueeze(0)])  # pad last

    return t, beta, sigma



# ================================
# 4. EULER + KARRAS with β(t) (50 steps)
# ================================
@torch.no_grad()
def sample_euler_karras_batch(
    s0: np.ndarray | torch.Tensor,
    score_model: torch.nn.Module,
    d_s: int,
    d_a: int,
    horizon: int,
    num_steps: int = 50,
    num_karras: int = 5,
    eta: float = 1.0,
    device: Optional[str] = None,
    num_samples: int = 1,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0_t = torch.as_tensor(s0, device=device, dtype=torch.float32)
    if s0_t.ndim == 1:
        if s0_t.shape[0] != d_s:
            raise ValueError(f"s0 should have shape ({d_s},), got {tuple(s0_t.shape)}")
        s0_t = s0_t.unsqueeze(0).repeat(num_samples, 1)
    elif s0_t.ndim == 2:
        if s0_t.shape[1] != d_s:
            raise ValueError(f"s0 should have shape (B, {d_s}), got {tuple(s0_t.shape)}")
        if num_samples > 1:
            s0_t = s0_t.repeat_interleave(num_samples, dim=0)
    else:
        raise ValueError(f"s0 must be 1D or 2D, got {tuple(s0_t.shape)}")
    batch_size = s0_t.shape[0]

    dim = d_s + d_a

    # Karras β(t) + σ(t)
    t_grid, beta_1, sigma_grid = karras_beta_schedule(num_steps, device=device)
    #t_grid, beta_1, _, sigma_grid =  karras_cosine_interpolated_beta(num_steps, device=device)

    beta_2 = cosine_beta(t_grid, s=0.008)

    # Initialize x_T
    x = torch.randn(batch_size, horizon, dim, device=device) * sigma_grid[0]
    #x2 = torch.randn(1, horizon, dim, device=device)

    # Conditioning
    mask = torch.zeros(batch_size, horizon, dim, device=device)
    mask[:, 0, :d_s] = 1.0
    y = torch.zeros_like(x)
    y[:, 0, :d_s] = s0_t
    x = mask * y + (1 - mask) * x
    
    

    for i in range(num_steps):
        t_now = t_grid[i]
        t_next = t_grid[i + 1] if i < num_steps - 1 else 0.0
        dt = (t_next - t_now).item()
        if( i < num_karras ):
            beta_now = beta_1[i].item()
        else:
            beta_now = beta_2[i].item()

        # Drift
        drift = -0.5 * beta_now * x

        # Score
        score = score_model(x, t_now.expand(batch_size))

        # Euler step
        if eta > 0:
            noise = torch.randn_like(x)
            noise_scale = eta * math.sqrt(beta_now * (-dt))
            x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - beta_now * score) * dt

        # Conditioning
        x = mask * y + (1 - mask) * x
        x = clip_actions(x, d_s)

    return x.cpu().numpy()


@torch.no_grad()
def sample_euler_karras(
    s0: np.ndarray,
    score_model: torch.nn.Module,
    d_s: int,
    d_a: int,
    horizon: int,
    num_steps: int = 50,
    num_karras: int = 5,
    eta: float = 1.0,
    device: Optional[str] = None,
) -> np.ndarray:
    return sample_euler_karras_batch(
        s0, score_model, d_s, d_a, horizon,
        num_steps=num_steps, num_karras=num_karras, eta=eta, device=device,
    )[0]


import math
import torch
import numpy as np
from typing import Optional

# ------------------------------------------------------------------ #
# 1. Hybrid schedule (Karras speed + cosine smoothness)
# ------------------------------------------------------------------ #
@torch.no_grad()
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
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    # ---- Karras log-linear σ_k(t) (fast decay) ----
    rho = torch.log(torch.tensor(sigma_max / sigma_min, device=device))
    sigma_k = sigma_min * torch.exp(rho * t)                 # σ_k(t)

    # ---- Target cosine α_bar(t) (smooth) ----
    f = torch.cos((t + cosine_s) / (1.0 + cosine_s) * torch.pi / 2) ** 2
    alpha_bar_target = f / f[0]                               # [0,1]

    # ---- VP marginals forced to the target ----
    sigma_sq = 1.0 - alpha_bar_target
    sigma = torch.sqrt(sigma_sq)

    # ---- Physics-consistent β(t) from variance ODE ----
    d_sigma_sq = torch.gradient(sigma_sq, spacing=t)[0]       # central diff
    beta = d_sigma_sq / alpha_bar_target
    beta = torch.clamp(beta, min=1e-6)                        # numerical safety

    eta = beta / 2.0                                          # memory-less

    return t, beta, sigma, eta


# ------------------------------------------------------------------ #
# 2. Pure Karras schedule (for the first `num_karras` steps)
# ------------------------------------------------------------------ #
@torch.no_grad()
def karras_beta_schedule(
    num_steps: int = 50,
    sigma_min: float = 0.01,
    sigma_max: float = 30.0,
    device: str = "cpu",
):
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    rho = torch.log(torch.tensor(sigma_max / sigma_min, device=device))
    sigma_k = sigma_min * torch.exp(rho * t)

    alpha = 1.0 / torch.sqrt(1.0 + sigma_k ** 2)
    sigma = sigma_k * alpha

    # analytic β(t) = 2ρ σ_k/(1+σ_k²)
    beta = 2.0 * rho * sigma_k / (1.0 + sigma_k ** 2)
    beta = torch.clamp(beta, min=1e-6)

    return t, beta, sigma


# ------------------------------------------------------------------ #
# 3. Adapted sampler (exact copy of your API)
# ------------------------------------------------------------------ #
@torch.no_grad()
def sample_euler_karras2(
    s0: np.ndarray,
    score_model: torch.nn.Module,
    d_s: int,
    d_a: int,
    horizon: int,
    num_steps: int = 50,
    num_karras: int = 5,
    eta: float = 1.0,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0_t = torch.tensor(s0, device=device, dtype=torch.float32)
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
    x = torch.randn(1, horizon, dim, device=device) * sigma_h[0]

    # ---------- 3. Conditioning ----------
    mask = torch.zeros(1, horizon, dim, device=device)
    mask[:, 0, :d_s] = 1.0
    y = torch.zeros_like(x)
    y[:, 0, :d_s] = s0_t.unsqueeze(0)
    x = mask * y + (1.0 - mask) * x

    # ---------- 4. Euler integration ----------
    for i in range(num_steps):
        t_now = t_grid[i]
        t_next = t_grid[i + 1] if i < num_steps - 1 else 0.0
        dt = (t_next - t_now).item()

        # ----- β(t) switch -----
        if i < num_karras:
            beta_now = beta_k[i].item()
        else:
            beta_now = beta_h[i].item()

        # ----- drift & score -----
        drift = -0.5 * beta_now * x
        score = score_model(x, t_now.unsqueeze(0))

        # ----- Euler step (with optional stochasticity) -----
        if eta > 0.0:
            noise = torch.randn_like(x)
            noise_scale = eta * math.sqrt(beta_now * (-dt))
            x = x + (drift - beta_now * score) * dt + noise_scale * noise
        else:
            x = x + (drift - beta_now * score) * dt

        # ----- conditioning & action clipping -----
        x = mask * y + (1.0 - mask) * x
        x = clip_actions(x, d_s)

    return x.squeeze(0).cpu().numpy()








"""

@torch.no_grad()
def sample_reverse_sde3(
    s0: np.ndarray,
    score_model: DiT1d,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int,
    eta: float,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0_t = torch.tensor(s0, device=device, dtype=torch.float32)
    if ( (s0_t.shape[0] != d_s)   ):
        raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")
    dim = d_s + d_a
    
    t_full = torch.linspace(0.0, 1.0, 8000+1, device = device)
    t_asc = torch.linspace(8000, 0.0, steps_T + 1, device=device).long()[:-1]
    beta = cosine_beta(t_full, s=0.008)
    #alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)
    
    # Initialize x_T ~ N(0, I) with shape (horizon, dim)
    x = torch.randn(horizon, dim, dtype=torch.float32, device=device).unsqueeze(0)
    conditions = s0_t.unsqueeze(0)
    mask = torch.zeros((1, horizon, dim), dtype = torch.float32, device = device)
    mask[:, 0, :d_s] = 1
    y = torch.zeros((1, horizon, dim), dtype = torch.float32, device = device)
    y[:, 0, :d_s] = conditions.clone()
    #x = apply_conditioning(x, conditions, d_s)
    x = mask * y + (1 - mask) * x
    

    for i in range(len(t_asc) - 1):
        k, k_next = t_asc[i].item(), t_asc[i+1].item()
        t = t_full[k]
        #t_now, t_next = t_asc[i], t_asc[i + 1]
        dt = (t_full[k_next] - t_full[k]).item()
        g2_val = beta[k].item()
        drift = -0.5 * g2_val * x
        #t_tensor = t_now.repeat(batch)
        score = score_model(x, t.unsqueeze(0))
        
        if eta > 0:
            noise = torch.randn_like(x)
            noise_scale = eta * math.sqrt(g2_val * (-dt))
            x = x + ((drift - g2_val * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - g2_val * score) * dt
        
        x = mask * y + (1 - mask) * x

        x = clip_actions(x, d_s)    
       
    #x = clip_actions(x, d_s)
    return x.squeeze(0).detach().cpu().numpy()





def cosine_alpha_sigma(t: torch.Tensor, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
   
    t = t.clamp(0.0, 1.0 - 1e-6)
    a = (math.pi / 2.0) * ((t + s) / (1.0 + s))
    a0 = (math.pi / 2.0) * (s / (1.0 + s))
    a0 = torch.tensor(a0, dtype = torch.float32)

    # ∫ β(s) ds = log(cos(a0) / cos(a))
    log_ratio = torch.log(torch.cos(a0) / (torch.cos(a) + 1e-12) + 1e-12)
    alpha = torch.exp(-0.5 * log_ratio)                    # α(t)
    sigma = torch.sqrt(1.0 - alpha**2)                     # σ(t)
    return alpha, sigma


# ------------------------------------------------------------------
# 3. DPM-SOLVER++ 2M — NO SIGN-LIKE VARIABLES, ALL SQRT SAFE
# ------------------------------------------------------------------
@torch.no_grad()
def sample_dpm_cosine(
    s0: np.ndarray,
    score_model: torch.nn.Module,   # returns s_θ(x,t)
    state_dim: int,
    action_dim: int,
    horizon: int,
    num_steps: int = 30,
    eta: float = 0.0,               # 0.0 → ODE, >0 → SDE
    device: Optional[str] = None,
) -> np.ndarray:
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0_t = torch.from_numpy(s0).to(device).float()
    if s0_t.shape[0] != state_dim:
        raise ValueError(f"s0 must have length {state_dim}")

    total_dim = state_dim + action_dim
    batch = 1

    # ---- Time grid: t = 1.0 → 0.0 ----
    t_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    alpha_grid, sigma_grid = cosine_alpha_sigma(t_grid)

    # ---- Initialize x_T ~ N(0,I) ----
    x = torch.randn(batch, horizon, total_dim, device=device)

    # ---- Conditioning (s0 at first timestep) ----
    mask = torch.zeros(batch, horizon, total_dim, device=device)
    mask[:, 0, :state_dim] = 1.0
    y = torch.zeros_like(x)
    y[:, 0, :state_dim] = s0_t.unsqueeze(0)
    x = mask * y + (1 - mask) * x

    # ---- DPM-Solver++ 2M loop ----
    for i in range(num_steps):
        t_cur = t_grid[i]
        t_nxt = t_grid[i + 1] if i < num_steps - 1 else 0.0

        alpha_cur = alpha_grid[i]
        sigma_cur = sigma_grid[i]
        alpha_nxt = alpha_grid[i + 1] if i < num_steps - 1 else torch.tensor(1.0, device=device)
        sigma_nxt = sigma_grid[i + 1] if i < num_steps - 1 else torch.tensor(0.0, device=device)

        # ---- 1. Current denoised estimate ----
        score_cur = score_model(x, t_cur.unsqueeze(0))                # s_θ
        eps_cur   = -score_cur * sigma_cur                            # ε_θ
        x0_cur    = (x - sigma_cur * eps_cur) / alpha_cur            # x̂₀(t_cur)

        # ---- 2. Midpoint (2nd-order) ----
        alpha_ratio = alpha_nxt / alpha_cur
        sigma_mid_sq = 0.5 * (sigma_cur**2 + sigma_nxt**2)
        mid_inner = sigma_mid_sq - (alpha_ratio * sigma_cur)**2
        mid_inner = torch.clamp(mid_inner, min=0.0)                   # ← SAFE
        x_mid = alpha_ratio * x + torch.sqrt(mid_inner) * x0_cur
        t_mid = 0.5 * (t_cur + t_nxt)

        score_mid = score_model(x_mid, t_mid.unsqueeze(0))
        eps_mid   = -score_mid * torch.sqrt(sigma_mid_sq)
        x0_mid    = (x_mid - torch.sqrt(sigma_mid_sq) * eps_mid) / alpha_ratio

        # ---- 3. Final DPM update ----
        h = t_nxt - t_cur                                            # negative
        r = sigma_nxt / sigma_cur

        term1 = alpha_nxt * x
        term2_inner = sigma_nxt**2 - (alpha_nxt * sigma_cur)**2
        term2_inner = torch.clamp(term2_inner, min=0.0)               # ← SAFE
        term2 = torch.sqrt(term2_inner) * x0_cur
        term3 = (h / 2.0) * r * (x0_cur - x0_mid)

        x = term1 + term2 + term3

        # ---- 4. SDE noise (eta > 0) — variance always ≥ 0 ----
        if eta > 0.0 and i < num_steps - 1:
           
            variance_inc = -h * (sigma_nxt**2 - sigma_cur**2)
            variance_inc = torch.clamp(variance_inc, min=1e-12)        # ← NEVER NEGATIVE
            noise_scale = eta * torch.sqrt(variance_inc)
            x = x + noise_scale * torch.randn_like(x)

        # ---- 5. Conditioning & clipping ----
        x = mask * y + (1 - mask) * x
        x = clip_actions(x, state_dim)

    return x.squeeze(0).cpu().numpy()




# ===================================================================
# 1. KARRAS NOISE SCHEDULE (non-uniform timesteps)
# ===================================================================
def karras_timesteps(
    num_steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cpu"
) -> torch.Tensor:
    
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    sigmas = sigma_min * (sigma_max / sigma_min) ** t
    return sigmas  # shape: (num_steps + 1,)


# ===================================================================
# 2. MAP KARRAS σ → COSINE VP-SDE MARGINALS (memoryless)
# ===================================================================
def vp_marginals_from_sigma(
    sigma_karras: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    alpha = 1.0 / torch.sqrt(1.0 + sigma_karras**2)
    sigma = sigma_karras * alpha  # renormalized
    return alpha, sigma


# ===================================================================
# 3. DPM-SOLVER++ 2M SAMPLING (15 steps, high quality)
# ===================================================================
@torch.no_grad()
def sample_dpm_karras_cosine(
    initial_state: np.ndarray,           # s0
    score_model: torch.nn.Module,        # DiT1d → takes (x, sigma)
    state_dim: int,                      # d_s
    action_dim: int,                      # d_a
    horizon: int,                        # trajectory length
    num_steps: int = 15,                 # ← ONLY 15 STEPS
    eta: float = 0.0,                    # 0.0 = ODE, >0 = SDE
    device: Optional[str] = None,
    ) -> np.ndarray:
    
    # --------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0 = torch.from_numpy(initial_state).to(device).float()
    if s0.shape[0] != state_dim:
        raise ValueError(f"initial_state must have length {state_dim}")

    total_dim = state_dim + action_dim
    batch_size = 1

    # --------------------------------------------------------------
    # 1. Get Karras noise levels
    # --------------------------------------------------------------
    sigma_levels = karras_timesteps(
        num_steps=num_steps,
        sigma_min=0.01,
        sigma_max=80.0,
        device=device
    )  # [σ_max, ..., σ_min]

    # --------------------------------------------------------------
    # 2. Map to VP-SDE marginals (α, σ)
    # --------------------------------------------------------------
    alpha_grid, sigma_grid = vp_marginals_from_sigma(sigma_levels)
    # alpha_grid[i]: signal scale at step i
    # sigma_grid[i]: noise scale at step i

    # --------------------------------------------------------------
    # 3. Initialize x_T ~ N(0, σ_max² I)
    # --------------------------------------------------------------
    x = torch.randn(batch_size, horizon, total_dim, device=device)
    x = x * sigma_grid[0]  # scale by initial noise

    # --------------------------------------------------------------
    # 4. Conditioning: fix s0 at t=0
    # --------------------------------------------------------------
    cond_mask = torch.zeros(batch_size, horizon, total_dim, device=device)
    cond_mask[:, 0, :state_dim] = 1.0
    cond_y = torch.zeros_like(x)
    cond_y[:, 0, :state_dim] = s0.unsqueeze(0)

    # --------------------------------------------------------------
    # 5. DPM-Solver++ 2M Loop
    # --------------------------------------------------------------
    for step in range(num_steps):
        sigma_cur = sigma_grid[step]
        alpha_cur = alpha_grid[step]
        sigma_next = sigma_grid[step + 1] if step < num_steps - 1 else 0.0
        alpha_next = alpha_grid[step + 1] if step < num_steps - 1 else 1.0

        # --- Model prediction: pass sigma, not t ---
        score = score_model(x, sigma_cur.unsqueeze(0))           # ← CLEAR: uses noise level
        eps = -score * sigma_cur                     # ε_θ = -σ s_θ
        x0_pred = (x - sigma_cur * eps) / alpha_cur  # denoised estimate

        # --- Midpoint for 2nd-order correction ---
        alpha_ratio = alpha_next / alpha_cur
        sigma_mid_sq = 0.5 * (sigma_cur**2 + sigma_next**2)
        mid_diff = sigma_mid_sq - (alpha_ratio * sigma_cur)**2
        mid_diff = torch.clamp(mid_diff, min=0.0)
        x_mid = alpha_ratio * x + torch.sqrt(mid_diff) * x0_pred

        # Model at midpoint
        score_mid = score_model(x_mid, torch.sqrt(sigma_mid_sq).unsqueeze(0))
        eps_mid = -score_mid * torch.sqrt(sigma_mid_sq)
        x0_mid = (x_mid - torch.sqrt(sigma_mid_sq) * eps_mid) / alpha_ratio

        # --- Final DPM update ---
        h = math.log(sigma_next / sigma_cur) if sigma_next > 0 else -12.0
        r = sigma_next / sigma_cur

        term1 = alpha_next * x
        term2_diff = sigma_next**2 - (alpha_next * sigma_cur)**2
        term2_diff = torch.clamp(term2_diff, min=0.0)
        term2 = torch.sqrt(term2_diff) * x0_pred
        term3 = (h / 2.0) * r * (x0_pred - x0_mid)

        x = term1 + term2 + term3

        # --- SDE noise (if eta > 0) ---
        if eta > 0.0 and step < num_steps - 1:
            noise_var = sigma_next**2 - sigma_cur**2
            noise_var = torch.clamp(noise_var, min=0.0)
            noise_scale = eta * torch.sqrt(noise_var)
            x = x + noise_scale * torch.randn_like(x)

        # --- Re-apply conditioning ---
        x = cond_mask * cond_y + (1 - cond_mask) * x
        x = clip_actions(x, state_dim)

    return x.squeeze(0).cpu().numpy()






# ===================================================================
# 1. COSINE SCHEDULE (VP-SDE MARGINALS) — MEMORYLESS
# ===================================================================
def cosine_alpha_sigma(t: torch.Tensor, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    
    t = t.clamp(0.0, 1.0 - 1e-6)
    a = (math.pi / 2.0) * ((t + s) / (1.0 + s))
    a0 = (math.pi / 2.0) * (s / (1.0 + s))
    a0 = torch.tensor(a0, dtype = torch.float32)
    log_ratio = torch.log(torch.cos(a0) / (torch.cos(a) + 1e-12) + 1e-12)
    alpha = torch.exp(-0.5 * log_ratio)
    sigma = torch.sqrt(1.0 - alpha**2)
    return alpha, sigma


# ===================================================================
# 2. DDIM SAMPLING (15 steps, VP-SDE compatible)
# ===================================================================
@torch.no_grad()
def sample_ddim(
    initial_state: np.ndarray,           # s0
    score_model: torch.nn.Module,        # DiT1d → outputs ε_θ(x,t) or s_θ
    state_dim: int,                      # d_s
    action_dim: int,                     # d_a
    horizon: int,                        # trajectory length
    num_steps: int = 15,                 # ← 15 steps
    eta: float = 0.0,                    # 0.0 = deterministic, >0 = stochastic
    guidance_scale: float = 0.0,         # CFG: 1.0 = no guidance
    device: Optional[str] = None,
) -> np.ndarray:
    
    # --------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s0 = torch.from_numpy(initial_state).to(device).float()
   

    total_dim = state_dim + action_dim
    batch_size = 1

    # --------------------------------------------------------------
    # 1. Time grid + marginals
    # --------------------------------------------------------------
    t_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # [1.0, ..., 0.0]
    alpha_grid, sigma_grid = cosine_alpha_sigma(t_grid, s=0.008)     # (N+1,)

    # --------------------------------------------------------------
    # 2. Initialize x_T ~ N(0, σ_max² I)
    # --------------------------------------------------------------
    x = torch.randn(batch_size, horizon, total_dim, device=device)
    x = x * sigma_grid[0]  # scale by initial noise

    # --------------------------------------------------------------
    # 3. Conditioning: fix s0 at t=0
    # --------------------------------------------------------------
    cond_mask = torch.zeros(batch_size, horizon, total_dim, device=device)
    cond_mask[:, 0, :state_dim] = 1.0
    cond_y = torch.zeros_like(x)
    cond_y[:, 0, :state_dim] = s0.unsqueeze(0)

    # --------------------------------------------------------------
    # 4. DDIM Loop
    # --------------------------------------------------------------
    for step in range(num_steps):
        t_cur = t_grid[step]
        t_next = t_grid[step + 1] if step < num_steps - 1 else 0.0

        alpha_cur = alpha_grid[step]
        alpha_next = alpha_grid[step + 1] if step < num_steps - 1 else 1.0
        sigma_cur = sigma_grid[step]

        # --- Model prediction (ε_θ) ---
        if guidance_scale > 1.0:
            # CFG: unconditional + conditional
            eps_uncond = score_model(x, t_cur.unsqueeze(0))
            eps_cond = score_model(x, t_cur.unsqueeze(0))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = score_model(x, t_cur.unsqueeze(0))  # or sigma_cur if model takes sigma

        # --- Denoised estimate ---
        x0_pred = (x - torch.sqrt(1.0 - alpha_cur) * eps) / torch.sqrt(alpha_cur)

        # --- DDIM update ---
        # Direction to predicted x0
        dir_xt = torch.sqrt(torch.tensor(1.0 - alpha_next, dtype = torch.float32)) * eps

        # DDIM step
        x = torch.sqrt(torch.tensor(alpha_next, dtype = torch.float32)) * x0_pred + dir_xt

        # --- Optional stochasticity (eta > 0) ---
        if eta > 0.0 and step < num_steps - 1:
            noise_scale = eta * torch.sqrt(alpha_grid[step] - alpha_grid[step + 1])
            x = x + noise_scale * torch.randn_like(x)

        # --- Re-apply conditioning ---
        x = cond_mask * cond_y + (1.0 - cond_mask) * x

        # --- Clip actions ---
        x = clip_actions(x, state_dim)

    return x.squeeze(0).cpu().numpy()
"""



