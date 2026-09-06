import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
import random
import os
import pickle


def init_wandb_run(
    name: str,
    config: dict,
    *,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
):
    """Initialize one stage run inside an optional shared pipeline group."""
    import wandb

    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "kaiwen_hu-uc-berkeley"),
        project=os.environ.get("WANDB_PROJECT", "ODP"),
        name=name,
        group=os.environ.get("WANDB_RUN_GROUP", group),
        job_type=job_type,
        config=config,
    )


def wandb_log(metrics: dict, step: Optional[int] = None) -> None:
    """Log only when the current process owns an initialized W&B run."""
    import wandb

    if wandb.run is not None:
        wandb.log(metrics, step=step)


@torch.no_grad()
def regression_diagnostics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Scale-aware regression diagnostics for training and evaluation logs."""
    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if prediction.numel() != target.numel() or prediction.numel() == 0:
        raise ValueError("prediction and target must be non-empty tensors of equal size")

    error = prediction - target
    target_std = target.std(unbiased=False)
    pred_std = prediction.std(unbiased=False)
    scale = target_std.clamp_min(1e-8)
    centered_pred = prediction - prediction.mean()
    centered_target = target - target.mean()
    correlation = (
        (centered_pred * centered_target).mean()
        / (pred_std * target_std).clamp_min(1e-8)
    )
    positive = target > 0
    background = ~positive

    def masked_mae(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(error[mask].abs().mean().item())

    return {
        "mae": float(error.abs().mean().item()),
        "normalized_mae": float((error.abs().mean() / scale).item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "bias": float(error.mean().item()),
        "normalized_bias": float((error.mean().abs() / scale).item()),
        "correlation": float(correlation.clamp(-1.0, 1.0).item()),
        "pred_mean": float(prediction.mean().item()),
        "pred_std": float(pred_std.item()),
        "target_mean": float(target.mean().item()),
        "target_std": float(target_std.item()),
        "std_ratio": float((pred_std / scale).item()),
        "pred_negative_fraction": float((prediction < 0).float().mean().item()),
        "target_negative_fraction": float((target < 0).float().mean().item()),
        "positive_fraction": float(positive.float().mean().item()),
        "positive_mae": masked_mae(positive),
        "background_mae": masked_mae(background),
    }


def cycle(dl):
    while True:
        for data in dl:
            yield data


class SAStats:
    obs_mean: np.ndarray
    obs_std:  np.ndarray
    #act_min =  np.array([-1.0] * 9)
    #act_max =  np.array([ 1.0] * 9)
    eps: float = 1e-3
    std_floor: float = 1e-3   

    # ---- observation ----
    def norm_obs(self, s: np.ndarray) -> np.ndarray:
        std = np.maximum(self.obs_std, self.std_floor)
        return (s - self.obs_mean) / (std)

    def denorm_obs(self, s: np.ndarray) -> np.ndarray:
        std = np.maximum(self.obs_std, self.std_floor)
        return s * (std) + self.obs_mean

"""
    def norm_act(self, a):
    # map [low, high] -> [-1, 1]
         return -1.0 + 2.0 * (a - self.act_min) / np.maximum(self.act_max - self.act_min, self.eps)

    def denorm_act(self, a_norm):
    # map [-1, 1] -> [low, high]
         return ((a_norm + 1.0) / 2.0) * (self.act_max - self.act_min) + self.act_min
"""






def set_seed(seed=0):
    # Python random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    # PyTorch deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def compare_models_state_dict(model1, model2, tolerance=1e-6):
    """
    Compare two models by their state dictionaries.
    Returns True if models are identical within tolerance.
    """
    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Check if they have the same keys
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        print("Models have different parameter names")
        return False
    
    # Compare each parameter
    for key in state_dict1.keys():
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # Check shapes
        if param1.shape != param2.shape:
            print(f"Parameter {key} has different shapes: {param1.shape} vs {param2.shape}")
            return False
        
        # Check values
        if not torch.allclose(param1, param2, atol=tolerance):
            print(f"Parameter {key} has different values (max diff: {torch.max(torch.abs(param1 - param2))})")
            return False
    
    print("Models are identical!")
    return True




def ema_smooth(rewards, alpha = 0.99):
   
    rewards = np.asarray(rewards)
    assert rewards.ndim == 1, "rewards must be 1D (length T)"
    assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"

    beta = 1.0 - alpha
    rewards_smooth = np.zeros_like(rewards)

    # Initialize EMA with the first reward
    rewards_smooth[0] = rewards[0]
    for t in range(1, len(rewards)):
        rewards_smooth[t] = alpha * rewards_smooth[t - 1] + beta * rewards[t]

    return rewards_smooth



def check_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using M3 GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Falling back to CPU (no GPU acceleration)")
    return device
