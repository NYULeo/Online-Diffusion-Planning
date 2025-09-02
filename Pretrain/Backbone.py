"""
diffusion_unet.py
-----------------

This module implements offline training of a diffusion model on vector
data using a 1‑D UNet backbone.  Unlike the transformer‑based planner
in ``diffusion_transformer.py``, this network treats the entire
concatenated state‑action vector as a single one‑dimensional signal.

Training follows the same score‑matching objective derived from the
forward SDE described in the provided document【594946067701072†screenshot】:

  J(θ) = E_{t∼U(0,T)} E_{x0}[ ω(t)‖sθ(x_t,t) − ∇_{x_t}log p_t(x_t|x0)‖² ],

where x_t is produced by linearly interpolating between the clean
vector x0 and Gaussian noise using α(t) and σ(t), and the true
conditional score is −(x_t−α(t)x0)/σ(t)²【594946067701072†screenshot】.  A cosine schedule for α(t)
and σ(t) is used, and the loss is weighted by σ(t)².  The UNet
architecture consists of several convolutional layers with skip
connections; it injects a learned time embedding at multiple
resolutions to condition the network on the current diffusion time.

Example
-------
>>> from diffusion_unet import StateActionVectorDataset, UNet1D, SDETrainer
>>> trajectories = kitchen_dataset.get_trajectories()
>>> dataset = StateActionVectorDataset(trajectories)
>>> model = UNet1D(input_dim=dataset.feature_dim)
>>> trainer = SDETrainer(model)
>>> # one training step on a batch of vectors
>>> batch = torch.stack([dataset[i] for i in range(32)])
>>> loss = trainer.train_step(batch)
>>> loss.backward()

"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





def positional_encoding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings of scalar time steps.

    This function is reused from ``diffusion_transformer.py``.  It
    produces a tensor of shape ``(batch, embed_dim)`` containing
    sinusoids with exponentially spaced frequencies.  The embedding
    dimension must be even.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    half_dim = embed_dim // 2
    device = t.device
    exp_term = torch.arange(half_dim, dtype=torch.float32, device=device) / (half_dim - 1)
    frequencies = torch.exp(-math.log(10000.0) * exp_term)
    args = t.unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb


def cosine_beta(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Continuous-time VP drift g(t)^2 = beta(t) for the cosine schedule.
    Using beta(t) = -2 d/dt log alpha(t) = (pi/(1+s)) * tan(a).
    """
    t = t.clamp(0.0, 1.0 - 1e-6)
    a = (math.pi / 2.0) * (t + s) / (1.0 + s)
    return (math.pi / (1.0 + s)) * torch.tan(a)


def cosine_alpha_sigma(t: torch.Tensor, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    """Continuous cosine schedule for α(t) and σ(t).

    Reuse of the same function from ``diffusion_transformer.py``.  See
    that module for details.
    """
    t = t.clamp(0.0, 1.0 - 1e-6)
    factor = (t + s) / (1.0 + s)
    f_t = torch.cos(    factor * (math.pi / 2)     )** 2
    f0 = torch.cos( torch.tensor((s / (1.0 + s)) * (math.pi / 2))  ) ** 2
    alpha_bar = (f_t / f0).clamp(0.0, 1.0 - 1e-6)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt(1.0 - alpha_bar)
    return alpha, sigma


class UNet1D(nn.Module):
    """A minimal 1‑D UNet suitable for flat vectors.

    The network treats each input vector of length ``input_dim`` as a
    1‑D signal with one channel.  It performs two rounds of downsampling
    via strided convolutions and two corresponding upsampling stages via
    transposed convolutions.  Skip connections are used to combine
    features from the down and up paths.  A learnable time embedding is
    added at multiple resolutions to condition the network on the
    diffusion time.  This is a simplified UNet design tailored for
    low‑dimensional data; more complex architectures can be adopted for
    higher‑resolution signals.
    """
     
    def __init__(
        self,
        input_dim: int,
        base_channels: int = 64,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        # Time embedding MLP: takes sinusoidal features and outputs a vector
        # that will be linearly projected into each block's channel space.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        # Separate projections for each resolution
        self.to_time_h1 = nn.Linear(base_channels * 4, base_channels)
        self.to_time_h2 = nn.Linear(base_channels * 4, base_channels * 2)
        self.to_time_h3 = nn.Linear(base_channels * 4, base_channels * 4)
        self.to_time_u2 = nn.Linear(base_channels * 4, base_channels * 2)
        self.to_time_u1 = nn.Linear(base_channels * 4, base_channels)
        # Downsampling path
        self.conv_in = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        # Upsampling path
        self.up1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        # Final convolution
        self.conv_out = nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)

        # Normalisation layers
        self.norm_h1 = nn.GroupNorm(8, base_channels)
        self.norm_h2 = nn.GroupNorm(8, base_channels * 2)
        self.norm_h3 = nn.GroupNorm(8, base_channels * 4)
        self.norm_u2 = nn.GroupNorm(8, base_channels * 2)
        self.norm_u1 = nn.GroupNorm(8, base_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the predicted score for a batch of vectors.

        Args:
            x: Tensor of shape ``(batch, input_dim)`` representing the noisy
               vector ``x_t``.
            t: Tensor of shape ``(batch,)`` containing continuous time steps
               in ``[0,1]``.

        Returns:
            Tensor of shape ``(batch, input_dim)`` containing the predicted
            score for each vector element.
        """
        #B, N = x.shape
        B, N = x.shape
        assert N == self.input_dim, f"expected input dimension {self.input_dim}, got {N}"
        # Time embedding
        # We use the same positional encoding as in the transformer
        temb = self.time_mlp(positional_encoding(t, self.time_mlp[0].in_features))
        # Project time embedding into channel dimensions for each block
        t_h1 = self.to_time_h1(temb).unsqueeze(-1)  # (B, C1, 1)
        t_h2 = self.to_time_h2(temb).unsqueeze(-1)  # (B, C2, 1)
        t_h3 = self.to_time_h3(temb).unsqueeze(-1)  # (B, C3, 1)
        t_u2 = self.to_time_u2(temb).unsqueeze(-1)  # (B, C2, 1)
        t_u1 = self.to_time_u1(temb).unsqueeze(-1)  # (B, C1, 1)
        # Reshape x to (B,1,N)
        h = x.unsqueeze(1)
        # Down 1
        h1 = self.conv_in(h)
        h1 = self.norm_h1(h1)
        h1 = F.gelu(h1 + t_h1)
        # Down 2
        h2 = self.down1(h1)
        h2 = self.norm_h2(h2)
        h2 = F.gelu(h2 + t_h2)
        # Down 3
        h3 = self.down2(h2)
        h3 = self.norm_h3(h3)
        h3 = F.gelu(h3 + t_h3)
        # Up 1: combine with h2
        u2 = self.up1(h3)
        # Ensure shapes match before adding skip connection (may differ by one element)
        if u2.shape[-1] > h2.shape[-1]:
            u2 = u2[..., :h2.shape[-1]]
        elif u2.shape[-1] < h2.shape[-1]:
            h2 = h2[..., :u2.shape[-1]]
        u2 = u2 + h2
        u2 = self.norm_u2(u2)
        u2 = F.gelu(u2 + t_u2)
        # Up 2: combine with h1
        u1 = self.up2(u2)
        if u1.shape[-1] > h1.shape[-1]:
            u1 = u1[..., :h1.shape[-1]]
        elif u1.shape[-1] < h1.shape[-1]:
            h1 = h1[..., :u1.shape[-1]]
        u1 = u1 + h1
        u1 = self.norm_u1(u1)
        u1 = F.gelu(u1 + t_u1)
        # Output
        out = self.conv_out(u1)
        out = out.squeeze(1)
        return out


class SDETrainer:
    """Offline trainer for score‑based diffusion models with a UNet backbone.

    This trainer implements the score matching objective for SDEs
    described in the provided algorithm.  Given a batch of clean
    vectors ``x0`` it samples a random time ``t∈[0,1]``, constructs
    ``x_t = α(t)x0 + σ(t)ε``, computes the target score
    ``-(x_t−α(t)x0)/σ(t)²``, evaluates the UNet prediction and returns
    the weighted MSE.  The loss uses a ``σ(t)²`` weighting term by
    default【594946067701072†screenshot】.
    """

    def __init__(self, model: UNet1D, s: float = 0.008, weight_type: str = "sigma2", device: Optional[torch.device] = None):
        self.model = model
        self.s = s
        self.weight_type = weight_type
        if device is None:
            device = next(model.parameters()).device
        self.device = device

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        x0 = x0.to(self.device)
        B, D = x0.shape
        # Sample time uniformly
        t = torch.rand(B, device=self.device)
        
        # Compute α and σ
        alpha, sigma = cosine_alpha_sigma(t, self.s)
        
        alpha_b = alpha.view(B, 1)
        sigma_b = sigma.view(B, 1)
        # Sample noise
        eps = torch.randn_like(x0, dtype = x0.dtype)
        # Construct x_t
        x_t = alpha_b * x0 + sigma_b * eps
        # Compute target score
        target = - (x_t - alpha_b * x0) / (sigma_b ** 2 + 1e-8)
        # Model prediction
        pred = self.model(x_t, t)
        # Weight
        if self.weight_type == "sigma2":
            weight = sigma.view(B, 1) ** 2
        
        elif self.weight_type == "beta":
            beta = cosine_beta(t, self.s)                     # (B,)
            weight = beta.view(B,1)

        else:
            raise ValueError(f"Unsupported weight_type {self.weight_type}")
        mse = F.mse_loss(pred, target, reduction='none')
        loss = (weight * mse).mean()
        return loss


@torch.no_grad()
def sample_pf_ode(
    s0: np.ndarray,        
    score_model: UNet1D,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Explicit-Euler integration of the PF-ODE:
      dx/dt = f(x,t) - 0.5 g(t)^2 s_theta(x,t)
    with the same VP schedule; hard-prefix via alpha(t)*v projection.
    """
    device = device or("cuda" if torch.cuda.is_available() else "cpu")
    D_tot = (d_s + d_a) * horizon


    t_asc = torch.linspace(0.0, 1.0, steps_T + 1, device=device)
    alpha, sigma = cosine_alpha_sigma(t_asc, s = 0.008)
    beta = cosine_beta(t_asc, s = 0.008)                                  # g^2
    idx_desc = torch.arange(steps_T, -1, -1, device=device)

    x = torch.randn(D_tot, device=device)

    # mask for first D1 entries; fixed vector holding s0 in those slots
    s0_t = torch.tensor(s0, device=device, dtype=x.dtype)
    M = torch.zeros(D_tot, device=device); M[:d_s] = 1.0
    Xfix = torch.zeros(D_tot, device=device, dtype=x.dtype)

    

    for i in range(len(idx_desc) - 1):
        k_now  = idx_desc[i]
        k_next = idx_desc[i + 1]
        t_now, t_next = t_asc[k_now], t_asc[k_next]
        
        dt = (t_next - t_now).item()                               # negative
        g2 = beta[k_now].item()
        drift = -0.5 * g2 * x
        

        score = score_model(x.unsqueeze(0), t_now.unsqueeze(0)).squeeze(0)

        # Explicit Euler for PF-ODE
        x = x + (drift - 0.5 * g2 * score) * dt

        # Hard-prefix: deterministic forward path y = alpha(t_next) * v
        known_next = alpha[k_next].item() * s0_t
        Xfix[:d_s] = known_next
        x = M * Xfix + (1.0 - M) * x

    return x.detach().cpu().numpy()


@torch.no_grad()
def sample_reverse_sde(
    s0: np.ndarray,             # (d_s,)
    score_model: UNet1D,
    d_s: int,
    d_a: int,
    horizon: int,
    steps_T: int = 100,
    eta: float = 1.0,           # 1.0 = reverse SDE (stochastic), 0.0 = PF-ODE (deterministic)
    device: Optional[str] = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    D_tot = (d_s + d_a) * horizon

    # Time grid & VP schedule (Nichol–Dhariwal cosine)
    t_asc = torch.linspace(1.0, 0.0, steps_T + 1, device=device)
    alpha, sigma = cosine_alpha_sigma(t_asc, s=0.008)  # (T+1,)
    beta = cosine_beta(t_asc, s=0.008)                 # (T+1,) => g^2 = beta
    #idx_desc = torch.arange(steps_T, -1, -1, device=device)  # T..0
    c_eta = 0.5 * (1.0 + eta**2)    # score coefficient in drift (SDE/ODE unification)

    # Init x_T ~ N(0, I)
    x = torch.randn(D_tot, dtype = torch.float32, device=device)

    # Prefix & mask (first d_s dims fixed)
    s0_t = torch.as_tensor(s0, device=device, dtype=x.dtype)
    M = torch.zeros(D_tot, device=device, dtype=x.dtype); M[:d_s] = 1.0
    Xfix = torch.zeros(D_tot, device=device, dtype=x.dtype)

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
        
        x = x + ( (drift - c_eta * g2 * score) * dt + eta * (g2**0.5) * ((-1*dt)**0.5) * noise )
        
        # Masked projection with forward-noised prefix at t_next
        if eta > 0:
            z = torch.randn(d_s, device=device, dtype=x.dtype)
            known_next = alpha[i].item() * s0_t + sigma[i].item() * z
        else:
            known_next = alpha[i].item() * s0_t

        Xfix[:d_s] = known_next
        x = M * Xfix + (1.0 - M) * x

    return x.detach().cpu().numpy()