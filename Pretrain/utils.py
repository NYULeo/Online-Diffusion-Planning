import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
import random
import os
import pickle


def wandb_log(metrics: dict, step: Optional[int] = None) -> None:
    """Log only when the current process owns an initialized W&B run."""
    import wandb

    if wandb.run is not None:
        wandb.log(metrics, step=step)


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