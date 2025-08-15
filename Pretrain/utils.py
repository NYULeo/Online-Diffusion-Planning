import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


#SDE schedules & helpers 
def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    factor = (t + s) / (1.0 + s) * math.pi / 2.0
    ft = torch.cos(factor) ** 2
    f0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2.0, device=t.device)) ** 2
    return (ft / f0).clamp(0.0, 1.0)

def alpha_sigma_from_alpha_bar(alpha_bar: torch.Tensor):
    alpha = alpha_bar.sqrt()
    sigma = (1.0 - alpha_bar).clamp(min=0.0).sqrt()
    return alpha, sigma

def beta_from_alpha_bar(t: torch.Tensor, s: float = 0.008, eps: float = 1e-5) -> torch.Tensor:
    # Only compute gradients if t requires gradients
    if t.requires_grad:
        t = t.detach().requires_grad_(True)
        ab = cosine_alpha_bar(t, s)
        log_ab = torch.log(ab.clamp(min=eps))
        grad = torch.autograd.grad(log_ab.sum(), t, create_graph=False)[0]
        beta = -grad
        return beta.detach()
    else:
        # For non-gradient computation, use a simpler approach
        # This is an approximation that avoids the gradient computation
        t_clamped = t.clamp(0.0, 1.0)
        factor = (t_clamped + s) / (1.0 + s) * math.pi / 2.0
        ft = torch.cos(factor) ** 2
        f0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2.0, device=t.device)) ** 2
        alpha_bar = (ft / f0).clamp(0.0, 1.0)
        # Approximate beta as the derivative of alpha_bar
        beta = torch.sin(2 * factor) * math.pi / (2 * (1.0 + s)) * alpha_bar
        return beta

#Embeddings
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, min_freq: float = 1.0, max_freq: float = 1000.0):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        half = max(1, dim // 2)
        freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] in [0,1]
        angles = x[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb

class PositionalEmbedding1D(nn.Module):
    def __init__(self, length: int, pos_dim: int):
        super().__init__()
        if length <= 0:
            raise ValueError("length must be positive")
        if pos_dim <= 0:
            raise ValueError("pos_dim must be positive")
        self.pe = nn.Embedding(length, pos_dim)

    def forward(self, H: int) -> torch.Tensor:
        idx = torch.arange(H, device=self.pe.weight.device)
        return self.pe(idx)  # [H, dim]


#Training utils
def get_loader(npy_path: str, batch_size: int, workers: int):
    # Import here to avoid circular import
    from Dataset import TrajectoryDataset
    ds = TrajectoryDataset(npy_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)

def build_model(backbone: str, feat_dim: int, hidden: int, time_dim: int, pos_dim: int, cond_dim: int, task_specific: bool = False):
    # Import here to avoid circular import
    from Backbone import TrajectoryUNet1D, TemporalTransformer, TrajectoryUNet1D_TaskSpecific, TemporalTransformer_TaskSpecific
    
    if task_specific:
        # Task-specific models (no conditioning)
        if backbone == "unet":
            return TrajectoryUNet1D_TaskSpecific(feat_dim=feat_dim, hidden=hidden, time_dim=time_dim, pos_dim=pos_dim)
        elif backbone == "transformer":
            return TemporalTransformer_TaskSpecific(feat_dim=feat_dim, d_model=hidden, num_layers=6, time_dim=time_dim, pos_dim=pos_dim)
        else:
            raise ValueError("backbone must be 'unet' or 'transformer'")
    else:
        # Conditional models (with task conditioning)
        if backbone == "unet":
            return TrajectoryUNet1D(feat_dim=feat_dim, hidden=hidden, time_dim=time_dim, pos_dim=pos_dim, cond_dim=cond_dim)
        elif backbone == "transformer":
            return TemporalTransformer(feat_dim=feat_dim, d_model=hidden, num_layers=6, time_dim=time_dim, pos_dim=pos_dim, cond_dim=cond_dim)
        else:
            raise ValueError("backbone must be 'unet' or 'transformer'")

