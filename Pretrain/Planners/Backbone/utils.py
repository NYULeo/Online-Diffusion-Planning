import math
import numpy as np
from sympy.logic.boolalg import true
import torch
from torch import Tensor
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from typing import List, Tuple, Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import os
from Dataset import get_PlannerName



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
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



#-----------------------------------------------------------------------------#
#---------------------------------- Schedule ----------------------------------#
#-----------------------------------------------------------------------------#

def cosine_beta(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Continuous-time VP drift g(t)^2 = beta(t) for the cosine schedule.
    Using beta(t) = -2 d/dt log alpha(t) = (pi/(1+s)) * tan(a).
    """
    t = t.clamp(0.0, 1.0 - 1e-3)
    a = (math.pi / 2.0) * ((t + s) / (1.0 + s))
    return (math.pi / (1.0 + s)) * torch.tan(a)


def cosine_alpha_sigma(t: torch.Tensor, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    """Continuous cosine schedule for α(t) and σ(t).

    Reuse of the same function from ``diffusion_transformer.py``.  See
    that module for details.
    """
    t = t.clamp(0.0, 1.0 - 1e-3)
    factor = (t + s) / (1.0 + s)
    f_t = torch.cos(    factor * (math.pi / 2)     )** 2
    f0 = torch.cos( torch.tensor((s / (1.0 + s)) * (math.pi / 2))  ) ** 2
    alpha_bar = (f_t / f0).clamp(0.0, 1.0 - 1e-3)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt(1.0 - alpha_bar)
    return alpha, sigma

def compute_dot_alpha_beta(t: Tensor, s: float = 0.008
                        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
   
    eps = 1e-3
    t2 = t.clamp(0.0, 1.0 - eps)
    
    # --- compute beta and dot_beta ---
    a = (math.pi / 2.0) * ((t2 + s) / (1.0 + s))
    da_dt = (math.pi / 2.0) * (1.0 / (1.0 + s))
    beta = (math.pi / (1.0 + s)) * torch.tan(a)
    # derivative: β' = (π/(1+s)) * sec^2(a) * da/dt
    dot_beta = (math.pi / (1.0 + s)) * (1.0 / torch.cos(a))**2 * da_dt
    
    # --- compute alpha and dot_alpha ---
    # Match cosine_alpha_sigma exactly
    factor = (t2 + s) / (1.0 + s)
    f_t = torch.cos(factor * (math.pi / 2))** 2
    # Match cosine_alpha_sigma: use same tensor creation
    f0 = torch.cos(torch.tensor((s / (1.0 + s)) * (math.pi / 2)))** 2
    alpha_bar = (f_t / f0).clamp(0.0, 1.0 - 1e-3)
    alpha = torch.sqrt(alpha_bar)
    
    # derivative of f_t: d[cos^2(u)]/dt = - sin(2u) * du/dt
    # where u = factor * (math.pi / 2)
    u = factor * (math.pi / 2.0)
    du_dt = da_dt  # same as a's derivative
    dot_f_t = - torch.sin(2.0 * u) * du_dt
    dot_alpha_bar = dot_f_t / f0
    # α = sqrt(α_bar) => dot α = dot α_bar / (2 * sqrt(α_bar))
    # => dot_alpha = dot_alpha_bar / (2 α)
    dot_alpha = dot_alpha_bar / (2.0 * alpha)
    
    return alpha, dot_alpha, beta, dot_beta


    


# 2. Autograd (derivative) version
def compute_dot_autograd(t: Tensor, s: float = 0.008
                         ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
 
    # Make t require gradient
    eps = 1e-3
    t2 = t.clamp(0.0, 1.0 - eps)
    
    
    t_req = t2.clone().detach().requires_grad_(True)
    
    alpha_req, _ = cosine_alpha_sigma(t_req, s=s)
    beta_req = cosine_beta(t_req, s=s)
    
    dot_alpha = torch.autograd.grad(
        outputs=alpha_req,
        inputs=t_req,
        grad_outputs=torch.ones_like(alpha_req),
        create_graph=True,
        retain_graph=False
    )[0]
    
    dot_beta = torch.autograd.grad(
        outputs=beta_req,
        inputs=t_req,
        grad_outputs=torch.ones_like(beta_req),
        create_graph=True,
        retain_graph=False
    )[0]
    
    # detach and return
    return alpha_req, dot_alpha, beta_req, dot_beta
    



#import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class UntrainablePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.einsum('...i,j->...ij', x, freqs.to(x.dtype))
        # x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.einsum('...i,j->...ij', x, emb.to(x.dtype))
        # emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return emb


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
}



class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)



def apply_conditioning(x, conditions, state_dim):
    x[:, 0, :state_dim] = conditions.clone()
    return x



#-----------------------------------------------------------------------------#
#---------------------------------- Selection --------------------------------#
#-----------------------------------------------------------------------------#


def get_pretrained_planner(dataset_name, specific_dataset, checkpoint_steps):
      planner_name = get_PlannerName(dataset_name, specific_dataset)
      checkpoint_path = f"./Pretrain/Planners/{dataset_name}/{specific_dataset}/Models/{planner_name}_{checkpoint_steps}.pt"
      if not os.path.exists(checkpoint_path):
          raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      return checkpoint['ema']







"""




#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}


"""



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

