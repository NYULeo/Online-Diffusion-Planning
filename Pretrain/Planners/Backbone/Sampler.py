from __future__ import annotations
import math
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import  TemporalUnet
from .Dit import DiT1d
from .utils import cosine_alpha_sigma, cosine_beta
from torch import Tensor






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
        
        
        #x = apply_conditioning(x, conditions, d_s)

    return x.squeeze(0).detach().cpu().numpy()


