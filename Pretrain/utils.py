import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
import random
import os

def compute_log_prob(model, s, a, s_next, device="cpu"):
   
    s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    s_next = torch.tensor(s_next, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        mu, log_std = model(s, a)
        sigma = torch.exp(log_std)
        D = mu.size(-1)
        # Compute log prob per dimension and sum
        log_prob = -0.5 * (((s_next - mu) / sigma) ** 2).sum(dim=-1)
        log_prob += -0.5 * (D * torch.log(torch.tensor(2 * torch.pi)) + 2 * log_std.sum(dim=-1))
    model.train()
    return log_prob.item()


def total_pro(traj, model):
    prob = 0
    count = 0
    for i in range(len(traj)):
        for j in range(len(traj[i]['actions'])):
            s, a, s_next = traj[i]['observations'][j], traj[i]['actions'][j], traj[i]['observations'][j+1]
            prob += compute_log_prob(model, s, a, s_next)
            count += 1
    return (prob / count)

def set_seed(seed=42):
    """
    Set all random seeds for reproducible results.
    
    Args:
        seed (int): Random seed value
    """
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
