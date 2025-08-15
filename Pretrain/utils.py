import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


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
    """Sinusoidal time embedding for diffusion models."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PositionalEmbedding1D(nn.Module):
    """1D positional embedding for trajectory sequences."""
    
    def __init__(self, max_len: int, pos_dim: int):
        super().__init__()
        self.max_len = max_len
        self.pos_dim = pos_dim
        
        # For very long sequences, create embedding on-demand
        if max_len > 10000:
            print(f"⚠️  Large trajectory length ({max_len:,}), using on-demand positional embedding")
            self.register_buffer('pos_emb', None)
        else:
            self.register_buffer('pos_emb', self._get_pos_emb())

    def _get_pos_emb(self):
        pos_emb = torch.zeros(self.max_len, self.pos_dim)
        for pos in range(self.max_len):
            for i in range(self.pos_dim):
                if i % 2 == 0:
                    pos_emb[pos, i] = math.sin(pos / (10000 ** (i / self.pos_dim)))
                else:
                    pos_emb[pos, i] = math.cos(pos / (10000 ** ((i-1) / self.pos_dim)))
        return pos_emb

    def forward(self, seq_len: int) -> torch.Tensor:
        if self.pos_emb is None:
            # Create embedding on-demand for large sequences
            # Use CPU device for large sequences to avoid memory issues
            pos_emb = torch.zeros(seq_len, self.pos_dim)
            for pos in range(seq_len):
                for i in range(self.pos_dim):
                    if i % 2 == 0:
                        pos_emb[pos, i] = math.sin(pos / (10000 ** (i / self.pos_dim)))
                    else:
                        pos_emb[pos, i] = math.cos(pos / (10000 ** ((i-1) / self.pos_dim)))
            return pos_emb
        else:
            return self.pos_emb[:seq_len]


#Training utils
def get_loader(traj_file: str, batch_size: int, workers: int = 4):
    """Get data loader for preprocessed numpy data."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Load preprocessed data
    arr = np.load(traj_file)
    N, H, D = arr.shape
    print(f"📊 Loaded data: {N} trajectories, {H} horizon, {D} features")
    
    # Convert to tensor
    data = torch.from_numpy(arr).float()
    dataset = TensorDataset(data)
    
    # Create loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    
    return loader


class FullTrajectoryDataset:
    """Dataset for full trajectories loaded from numpy file."""
    
    def __init__(self, trajectories, dtype="float32"):
        self.trajectories = trajectories
        self.dtype = dtype
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        if self.dtype == "float32":
            return torch.from_numpy(self.trajectories[idx]).float()
        elif self.dtype == "float16":
            return torch.from_numpy(self.trajectories[idx]).half()
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")


def get_full_trajectory_loader(traj_file: str, batch_size: int, workers: int = 4, dtype: str = "float32"):
    """Get data loader for full trajectory data."""
    from torch.utils.data import DataLoader
    
    # Load preprocessed data
    trajectories = np.load(traj_file, allow_pickle=True)
    N = len(trajectories)
    
    # Get dimensions from first trajectory
    first_traj = trajectories[0]
    H = first_traj.shape[0]
    D = first_traj.shape[1]
    
    print(f"📊 Loaded full trajectories: {N} trajectories, {H:,} steps each, {D} features")
    
    # Create dataset and loader
    dataset = FullTrajectoryDataset(trajectories, dtype=dtype)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    
    return loader

def build_model(backbone: str, feat_dim: int, hidden: int, time_dim: int, pos_dim: int):
    """
    Build task-specific diffusion model.
    
    Args:
        backbone: Model architecture ('unet' or 'transformer')
        feat_dim: Input/output feature dimension (task-specific)
        hidden: Hidden dimension size
        time_dim: Time embedding dimension
        pos_dim: Positional embedding dimension
        
    Returns:
        Task-specific diffusion model
    """
    # Import here to avoid circular import
    from Backbone import TrajectoryUNet1D, TemporalTransformer
    
    if backbone == "unet":
        return TrajectoryUNet1D(feat_dim=feat_dim, hidden=hidden, time_dim=time_dim, pos_dim=pos_dim)
    elif backbone == "transformer":
        return TemporalTransformer(feat_dim=feat_dim, d_model=hidden, num_layers=6, time_dim=time_dim, pos_dim=pos_dim)
    else:
        raise ValueError("backbone must be 'unet' or 'transformer'")

