from __future__ import annotations
import math
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import  TemporalUnet
from .utils import cosine_alpha_sigma, cosine_beta

@torch.no_grad()
def sample_reverse_sde(
    s0: np.ndarray,
    score_model: TemporalUnet,
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
    beta = cosine_beta(t_asc, s=0.008)
    alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)
    
    # Initialize x_T ~ N(0, I) with shape (horizon, dim)
    x = torch.randn(horizon, dim, dtype=torch.float32, device=device)

     # Create mask for conditioning
    M = torch.zeros(horizon, dim, device=device)
    M[0, :d_s] = 1.0  # Fix first timestep's state dimensions

     # Fixed values (s0 for first timestep)
    Xfix = torch.zeros_like(x)
    Xfix[0, :d_s] = s0_t[:d_s]  # Use first timestep of s0

    for i in range(len(t_asc) - 1):
        t_now, t_next = t_asc[i], t_asc[i + 1]
        dt = (t_next - t_now).item()
        g2_val = beta[i].item()
        drift = -0.5 * g2_val * x
        #t_tensor = t_now.repeat(batch)
        score = score_model(x.unsqueeze(0), t_now.unsqueeze(0)).squeeze(0)

        if eta > 0:
            noise = torch.randn_like(x)
            noise_scale = eta * math.sqrt(g2_val * (-dt))
            x = x + ((drift - g2_val * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - g2_val * score) * dt
        
        noise_known = torch.randn(d_s, device=device, dtype=torch.float32)
        new_known = alpha[i+1] * s0_t + sigma[i+1] * noise_known
        Xfix[0, :d_s] = new_known[:d_s]
        x = M * Xfix + (1.0 - M) * x

    return x.detach().cpu().numpy()




"""

@torch.no_grad()
def sample_pf_ode(
    s0: np.ndarray,        
    score_model: TemporalUnet,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
   
    device = device or("cuda" if torch.cuda.is_available() else "cpu")
    D_tot = (d_s + d_a) * horizon


    t_asc = torch.linspace(1.0, 0.0, steps_T + 1, device=device)
    alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)
    beta = cosine_beta(t_asc, s = 0.008)                                  # g^2

    x = torch.randn(D_tot,  dtype = torch.float32, device=device)

    # mask for first D1 entries; fixed vector holding s0 in those slots
    s0_t = torch.tensor(s0, device=device, dtype=x.dtype)
    M = torch.zeros(D_tot, device=device); M[:d_s] = 1.0
    Xfix = torch.zeros(D_tot, device=device, dtype=x.dtype)
    Xfix[:d_s] = s0_t.copy()

    

    for i in range(len(t_asc)-1):
        t_now, t_next = t_asc[i], t_asc[i+1]
        
        dt = (t_next - t_now).item()                               # negative
        g2 = beta[i].item()
        drift = -0.5 * g2 * x
        
        score = score_model(x.unsqueeze(0), t_now.unsqueeze(0)).squeeze(0)

        # Explicit Euler for PF-ODE
        x = x + (drift - 0.5 * g2 * score) * dt

        x = M * Xfix + (1.0 - M) * x

    return x.detach().cpu().numpy()



@torch.no_grad()
def sample_reverse_sde(
    s0: np.ndarray,             # (d_s,)
    score_model: TemporalUnet,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int,
    eta: float,           # 1.0 = reverse SDE (stochastic), 0.0 = PF-ODE (deterministic)
    device: Optional[str] = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    D_tot = (d_s + d_a) * horizon

    # Time grid & VP schedule (Nichol–Dhariwal cosine)
    t_asc = torch.linspace(1.0, 0.0, steps_T + 1, device=device)
    alpha, sigma = cosine_alpha_sigma(t_asc, s=0.008)  # (T+1,)
    beta = cosine_beta(t_asc, s=0.008)                 # (T+1,) => g^2 = beta
   

    # Init x_T ~ N(0, I)
    x = torch.randn(D_tot, dtype = torch.float32, device=device)

    # Prefix & mask (first d_s dims fixed)
    s0_t = torch.as_tensor(s0, device=device, dtype=x.dtype)
    M = torch.zeros(D_tot, device=device, dtype=x.dtype); M[:d_s] = 1.0
    Xfix = torch.zeros(D_tot, device=device, dtype=x.dtype)
    Xfix[:d_s] = s0_t

    for i in range(len(t_asc) - 1):
        t_now, t_next = t_asc[i], t_asc[i+1]
        
        dt = (t_next - t_now).item()          # negative
        #print(dt)
        g2 = beta[i].item()               # g^2(t) = beta(t)
        drift = -0.5 * g2 * x                 # f(x,t) = -0.5 beta x
  
        # Score s_theta(x,t)
        score = score_model(x.unsqueeze(0), t_now.unsqueeze(0)).squeeze(0)
        
        # Unified predictor step
        noise = torch.randn_like(x, dtype=x.dtype) if eta > 0 else torch.zeros_like(x)
        x = x + ( (drift - g2 * score) * dt +   eta * (g2**0.5) * ((-1*dt)**0.5) * noise  )
        
       
        x = M * Xfix + (1.0 - M) * x

    return x.detach().cpu().numpy()


"""