
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
    build_model
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

def dsm_loss(model: nn.Module, traj0: torch.Tensor, pos_emb: torch.Tensor, cfg: LossConfig, cond: Optional[torch.Tensor] = None, task_specific: bool = False):
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
    
    if task_specific:
        # Task-specific models don't use conditioning
        pred = model(x_t, t, pos_emb)
    else:
        # Conditional models use task conditioning
        pred = model(x_t, t, pos_emb, cond) if cond is not None else model(x_t, t, pos_emb, None)
    
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
def plan_trajectories(model: nn.Module, horizon: int, feat_dim: int, steps: int = 1000, eta: float = 1.0, s: float = 0.008, device: str = "cuda", batch: int = 1, cond: Optional[torch.Tensor] = None, first_step: Optional[torch.Tensor] = None, clamp_mask_dim: int = 0, pos_dim: int = 128, task_specific: bool = False):
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
        
        if task_specific:
            # Task-specific models don't use conditioning
            pred = model(x, t_k, pos_emb)
        else:
            # Conditional models use task conditioning
            pred = model(x, t_k, pos_emb, cond)
        
        # Clamp output for numerical stability
        pred = torch.clamp(pred, -1, 1)
        
        # Reverse-time SDE step
        alpha_bar_k = cosine_alpha_bar(t_k, s)
        alpha_bar_k1 = cosine_alpha_bar(t_k1, s)
        alpha_k, sigma_k = alpha_sigma_from_alpha_bar(alpha_bar_k)
        alpha_k1, sigma_k1 = alpha_sigma_from_alpha_bar(alpha_bar_k1)
        
        # Compute drift and diffusion terms
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
    
    # Determine if we're using task-specific training
    task_specific = getattr(args, 'task_specific', False)
    
    # Load data and determine dimensions
    if args.traj_file.endswith('.npy'):
        # Load preprocessed numpy data
        arr = np.load(args.traj_file)
        N, H, D = arr.shape
        cond_dim = 0
        ctx = None
        
        # Check if task conditions exist (only for non-task-specific training)
        if not task_specific:
            task_cond_path = args.traj_file.replace('_trajectories.npy', '_task_conditions.npy')
            if os.path.exists(task_cond_path):
                task_conditions = np.load(task_cond_path)
                cond_dim = task_conditions.shape[1]
                print(f"✅ Loaded task conditions with {cond_dim} task dimensions")
                ctx = task_conditions
            else:
                print("⚠️  No task conditions found, training without task conditioning")
        else:
            print("🎯 Task-specific training: no task conditioning needed")
    else:
        # Load raw data using HumanoidBenchDataset
        dataset = HumanoidBenchDataset(args.traj_file, horizon=64, stride=8, 
                                      selected_tasks=args.tasks, task_conditioning=not task_specific)
        N = len(dataset)
        H = dataset.horizon
        D = dataset.trajectories.shape[2] if hasattr(dataset, 'trajectories') else 70
        cond_dim = len(dataset.selected_tasks) if hasattr(dataset, 'selected_tasks') and not task_specific else 0
        ctx = dataset.task_conditions if hasattr(dataset, 'task_conditions') and not task_specific else None
        print(f"✅ Loaded dataset with {N} trajectories, {H} horizon, {D} features")
        if task_specific:
            print("🎯 Task-specific training: no task conditioning")
        else:
            print(f"📊 Task conditioning: {cond_dim} task dimensions")
    
    # Load optional context file (only for non-task-specific training)
    if not task_specific and args.ctx_file is not None and args.ctx_file != "":
        ctx = np.load(args.ctx_file)
        cond_dim = ctx.shape[1]
    
    # Build model
    model = build_model(args.backbone, feat_dim=D, hidden=args.hidden, time_dim=args.time_dim, 
                       pos_dim=args.pos_dim, cond_dim=cond_dim, task_specific=task_specific).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfg = LossConfig(weight_type=args.weight, s=args.s, clamp_first_step=args.clamp_first_step, clamp_mask_dim=args.clamp_mask_dim)
    
    # Create data loader
    if args.traj_file.endswith('.npy'):
        loader = get_loader(args.traj_file, args.batch_size, args.workers)
    else:
        # Use HumanoidBenchDataset directly
        dataset = HumanoidBenchDataset(args.traj_file, horizon=64, stride=8, 
                                      selected_tasks=args.tasks, task_conditioning=not task_specific)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    
    pos_emb_module = PositionalEmbedding1D(H, pos_dim=args.pos_dim).to(device)
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2 and not task_specific:
                # New format: (trajectory, task_condition) - only for conditional training
                traj, task_cond = batch
                traj = traj.to(device)
                task_cond = task_cond.to(device)
                cond = task_cond
            else:
                # Old format: just trajectory - for task-specific training
                traj = batch.to(device)
                cond = None
                
                # Use context if available (only for non-task-specific training)
                if not task_specific and ctx is not None:
                    # For now, use a simple approach: load context based on global step
                    # This assumes the DataLoader shuffles consistently
                    batch_start = (global_step * args.batch_size) % len(ctx)
                    batch_end = min(batch_start + args.batch_size, len(ctx))
                    if batch_start < len(ctx):
                        ctx_batch = ctx[batch_start:batch_end]
                        # Pad if batch is smaller than expected
                        if len(ctx_batch) < args.batch_size:
                            padding = np.zeros((args.batch_size - len(ctx_batch), ctx_batch.shape[1]), dtype=ctx_batch.dtype)
                            ctx_batch = np.concatenate([ctx_batch, padding], axis=0)
                        cond = torch.from_numpy(ctx_batch).to(device)
            
            # Create fresh positional embeddings for this batch
            pos_emb = pos_emb_module(H).detach()
            
            loss = dsm_loss(model, traj, pos_emb, cfg, cond=cond, task_specific=task_specific)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.6f}")
            
            if global_step % args.sample_every == 0 and global_step > 0:
                with torch.no_grad():
                    first_step = None
                    if args.clamp_first_step and args.clamp_mask_dim > 0:
                        # use zeros placeholder or user-provided current state
                        first_step = torch.zeros(args.sample_bs, D, device=device)
                    
                    if task_specific:
                        # Task-specific models: generate samples without conditioning
                        samples = plan_trajectories(model, horizon=H, feat_dim=D, steps=args.steps, 
                                                  eta=args.eta, s=args.s, device=str(device), batch=args.sample_bs, 
                                                  cond=None, first_step=first_step, 
                                                  clamp_mask_dim=args.clamp_mask_dim, pos_dim=args.pos_dim,
                                                  task_specific=True)
                    else:
                        # Conditional models: generate samples for different tasks if task conditioning is available
                        if cond_dim > 0:
                            # Generate one sample per task
                            all_samples = []
                            for task_idx in range(min(cond_dim, args.sample_bs)):
                                task_cond = torch.zeros(1, cond_dim, device=device)
                                task_cond[0, task_idx] = 1.0
                                sample = plan_trajectories(model, horizon=H, feat_dim=D, steps=args.steps, 
                                                         eta=args.eta, s=args.s, device=str(device), batch=1, 
                                                         cond=task_cond, first_step=first_step, 
                                                         clamp_mask_dim=args.clamp_mask_dim, pos_dim=args.pos_dim,
                                                         task_specific=False)
                                all_samples.append(sample)
                            
                            # Concatenate samples
                            samples = torch.cat(all_samples, dim=0)
                        else:
                            # Generate samples without task conditioning
                            samples = plan_trajectories(model, horizon=H, feat_dim=D, steps=args.steps, 
                                                      eta=args.eta, s=args.s, device=str(device), batch=args.sample_bs, 
                                                      cond=None, first_step=first_step, 
                                                      clamp_mask_dim=args.clamp_mask_dim, pos_dim=args.pos_dim,
                                                      task_specific=False)
                    
                    np.save(os.path.join(args.outdir, f"sample_{global_step:07d}.npy"), samples.cpu().numpy())
                    print("Saved sample", global_step)
            
            global_step += 1
        
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.outdir, f"ckpt_epoch{epoch}.pt"))
        print("Saved checkpoint for epoch", epoch)
    
    return model
