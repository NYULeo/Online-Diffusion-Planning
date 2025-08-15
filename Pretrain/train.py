
#!/usr/bin/env python3
"""
Diffusion trajectory planner for HumanoidBench offline dataset. Supports two backbones:
 - "unet": Temporal 1D U-Net (convolutional down/up sampling over time)
 - "transformer": Temporal Transformer with cross-step attention (causal removed; full-attention over horizon)

Features:
 - cosine alpha_bar schedule, DSM training objective (sigma^2 or beta weighting)
 - forward closed-form sampling x_t = alpha x0 + sigma eps
 - reverse-time SDE / probability-flow ODE sampler (eta controls stochasticity)
 - local HumanoidBench dataset processing from hbench.pickle
 - clamp initial state dims to preserve current environment state
 - configurable conditioning vector (goal / task embedding)
 - checkpointing & sample saving
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Backbone import TrajectoryUNet1D, TemporalTransformer

from utils import (
    cosine_alpha_bar, 
    alpha_sigma_from_alpha_bar, 
    beta_from_alpha_bar, 
    SinusoidalEmbedding, 
    PositionalEmbedding1D, 
    get_loader, 
    build_model,
    get_full_trajectory_loader
)
from Dataset import TrajectoryDataset, HumanoidBenchDataset


#Sampling and Loss
@dataclass
class LossConfig:
    weight_type: str = "sigma2"
    s: float = 0.008
    clamp_first_step: bool = True
    clamp_mask_dim: int = 0

def sample_xt(traj0: torch.Tensor, t: torch.Tensor, s: float = 0.008):
    alpha_bar = cosine_alpha_bar(t, s=s)
    alpha, sigma = alpha_sigma_from_alpha_bar(alpha_bar)
    eps = torch.randn_like(traj0)
    alpha_r = alpha[:, None, None]
    sigma_r = sigma[:, None, None]
    x_t = alpha_r * traj0 + sigma_r * eps
    return x_t, eps, alpha, sigma

def dsm_loss(model: nn.Module, traj0: torch.Tensor, pos_emb: torch.Tensor, cfg: LossConfig):
    B = traj0.shape[0]
    device = traj0.device
    t = torch.rand(B, device=device)
    x_t, _, alpha, sigma = sample_xt(traj0, t, s=cfg.s)
    if cfg.clamp_first_step and cfg.clamp_mask_dim > 0:
        x_t[:,0,:cfg.clamp_mask_dim] = traj0[:,0,:cfg.clamp_mask_dim]
    alpha_r = alpha[:, None, None]
    sigma_r = sigma[:, None, None]
    # Add more robust numerical stability
    sigma_sq = sigma_r ** 2
    sigma_sq = torch.clamp(sigma_sq, min=1e-8)
    target = -(x_t - alpha_r * traj0) / sigma_sq
    
    # Task-specific models don't use conditioning
    pred = model(x_t, t, pos_emb)
    
    if cfg.weight_type == "sigma2":
        w = (sigma ** 2).detach()[:, None, None]
    elif cfg.weight_type == "beta":
        # Compute beta without gradients to avoid backward issues
        beta = beta_from_alpha_bar(t.detach(), s=cfg.s)
        w = beta.detach()[:, None, None]
    else:
        raise ValueError("weight_type must be 'sigma2' or 'beta'")
    loss = (w * (pred - target) ** 2).mean()
    return loss

#Reverse-Time Sampling
@torch.no_grad()
def plan_trajectories(model: nn.Module, horizon: int, feat_dim: int, steps: int = 1000, eta: float = 1.0, s: float = 0.008, device: str = "cuda", batch: int = 1, first_step: Optional[torch.Tensor] = None, clamp_mask_dim: int = 0, pos_dim: int = 128):
    model.eval()
    B = batch
    device_tensor = torch.device(device)
    x = torch.randn(B, horizon, feat_dim, device=device_tensor)
    # Create positional embedding once and reuse
    pos_emb_module = PositionalEmbedding1D(horizon, pos_dim).to(device_tensor)
    pos_emb = pos_emb_module(horizon).to(device_tensor)
    del pos_emb_module  # Clean up to prevent memory leak
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device_tensor)
    for i in range(steps):
        t_k = ts[i].expand(B)
        t_k1 = ts[i+1].expand(B)
        dt = (t_k1 - t_k)[0]
        
        # Task-specific models don't use conditioning
        pred = model(x, t_k, pos_emb)
        
        # Clamp output for numerical stability
        pred = torch.clamp(pred, -1, 1)
        
        # Reverse-time SDE step
        alpha_bar_k = cosine_alpha_bar(t_k, s)
        alpha_bar_k1 = cosine_alpha_bar(t_k1, s)
        alpha_k, sigma_k = alpha_sigma_from_alpha_bar(alpha_bar_k)
        alpha_k1, sigma_k1 = alpha_sigma_from_alpha_bar(alpha_bar_k1)
        
        # Compute drift and diffusion terms
        # Fix broadcasting by adding proper dimensions
        alpha_k = alpha_k[:, None, None]  # [B, 1, 1]
        alpha_k1 = alpha_k1[:, None, None]  # [B, 1, 1]
        sigma_k = sigma_k[:, None, None]  # [B, 1, 1]
        sigma_k1 = sigma_k1[:, None, None]  # [B, 1, 1]
        
        drift = (alpha_k1 - alpha_k) / dt * x + (sigma_k1 - sigma_k) / dt * pred
        diffusion = eta * sigma_k * torch.randn_like(x)
        
        x = x + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        
        # Clamp first step if specified
        if first_step is not None and clamp_mask_dim > 0:
            x[:, 0, :clamp_mask_dim] = first_step[:, :clamp_mask_dim]
    
    return x

#Training 
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Task-specific training only - task_name is required
    if not hasattr(args, 'task_name') or args.task_name is None:
        raise ValueError("task_name is required for task-specific training")
    
    print(f"🎯 Training task-specific model for: {args.task_name}")
    
    # Load data and determine dimensions
    if args.traj_file.endswith('.npy'):
        # Load preprocessed numpy data
        trajectories = np.load(args.traj_file, allow_pickle=True)  # Load list of trajectories
        N = len(trajectories)  # Number of trajectories
        
        # Get dimensions from first trajectory
        first_traj = trajectories[0]
        H = first_traj.shape[0]  # Trajectory length (number of steps)
        D = first_traj.shape[1]  # Feature dimension
        
        print(f"✅ Loaded preprocessed data: {N} trajectories, {H:,} steps each, {D} features")
        
        # Use task-specific dimensions if available
        if hasattr(args, 'feat_dim'):
            D = args.feat_dim
            print(f"🎯 Using task-specific feature dimension: {D}")
    else:
        # Load raw data using HumanoidBenchDataset
        dataset = HumanoidBenchDataset(args.traj_file, selected_tasks=[args.task_name])
        N = len(dataset)
        H = dataset.trajectory_length  # Full trajectory length
        D = dataset.trajectories[0].shape[1] if dataset.trajectories else 70
        print(f"✅ Loaded dataset: {N} trajectories, {H:,} steps each, {D} features")
    
    # Build model with task-specific dimensions
    model = build_model(args.backbone, feat_dim=D, hidden=args.hidden, time_dim=args.time_dim, 
                       pos_dim=args.pos_dim).to(device)
    
    # Use configurable optimizer
    if args.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    cfg = LossConfig(weight_type=args.weight, s=args.s, clamp_first_step=args.clamp_first_step, clamp_mask_dim=args.clamp_mask_dim)
    
    # Create data loader
    if args.traj_file.endswith('.npy'):
        # Use the new full trajectory loader
        loader = get_full_trajectory_loader(args.traj_file, args.batch_size, args.workers, args.dtype)
    else:
        # Use HumanoidBenchDataset directly
        dataset = HumanoidBenchDataset(args.traj_file, selected_tasks=[args.task_name])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    
    pos_emb_module = PositionalEmbedding1D(H, pos_dim=args.pos_dim).to(device)
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            # Task-specific training: full trajectory
            if isinstance(batch, (list, tuple)):
                traj = batch[0].to(device)  # Handle TensorDataset format
            else:
                traj = batch.to(device)
            
            # Create fresh positional embeddings for this batch
            pos_emb = pos_emb_module(H).detach()
            
            loss = dsm_loss(model, traj, pos_emb, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            
            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.6f}")
            
            if global_step % args.sample_every == 0 and global_step > 0:
                with torch.no_grad():
                    first_step = None
                    if args.clamp_first_step and args.clamp_mask_dim > 0:
                        # use zeros placeholder or user-provided current state
                        first_step = torch.zeros(args.sample_bs, D, device=device)
                    
                    # Task-specific models: generate samples without conditioning
                    samples = plan_trajectories(model, horizon=H, feat_dim=D, steps=args.steps, 
                                              eta=args.eta, s=args.s, device=str(device), batch=args.sample_bs, 
                                              first_step=first_step, 
                                              clamp_mask_dim=args.clamp_mask_dim, pos_dim=args.pos_dim)
                    
                    np.save(os.path.join(args.outdir, f"sample_{global_step:07d}.npy"), samples.cpu().numpy())
                    print("Saved sample", global_step)
            
            global_step += 1
        
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.outdir, f"ckpt_epoch{epoch}.pt"))
        print("Saved checkpoint for epoch", epoch)
