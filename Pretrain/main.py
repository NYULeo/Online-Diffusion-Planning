#!/usr/bin/env python3
"""
Task-Specific Diffusion Planner Training for HumanoidBench
Uses local hbench.pickle dataset with 14 tasks.
"""

import argparse
import os
import torch
from train import train
from Dataset import get_task_specific_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Task-Specific Diffusion Planner for HumanoidBench - Full Trajectory Training")
    p.add_argument("--traj_file", type=str, default="", help="training .npy [N,H,D] or .pkl file (if not provided, will use local hbench.pickle)")
    p.add_argument("--outdir", type=str, default="./hbench_runs")
    p.add_argument("--backbone", type=str, default="unet", choices=["unet","transformer"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--pos_dim", type=int, default=128)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--sample_bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--s", type=float, default=0.008)
    p.add_argument("--weight", type=str, default="sigma2", choices=["sigma2","beta"])
    p.add_argument("--clamp_first_step", action="store_true")
    p.add_argument("--clamp_mask_dim", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    
    # Advanced parameters
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"], help="Optimizer type (default: adamw)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], help="Data type (default: float32)")
    
    # Task-specific parameters
    p.add_argument("--task_name", type=str, required=True, help="Specific task name for training (e.g., h1-run-v0, h1-walk-v0, etc.)")
    p.add_argument("--max_trajectory_length", type=int, default=1000, help="Maximum trajectory length to use for training (default: 1000, use -1 for full trajectory)")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    if args.hidden <= 0:
        raise ValueError("hidden dimension must be positive")
    if args.time_dim <= 0:
        raise ValueError("time_dim must be positive")
    if args.pos_dim <= 0:
        raise ValueError("pos_dim must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.eta < 0:
        raise ValueError("eta must be non-negative")
    if args.s <= 0:
        raise ValueError("s must be positive")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Prepare dataset and get task-specific dimensions
    if not args.traj_file:
        print("🤖 Preparing HumanoidBench dataset from local hbench.pickle...")
        print(f"🎯 Task: {args.task_name}")
        print(f"📏 Using full trajectory length from dataset")
        
        # Get task-specific dataset with automatic dimension detection
        traj_path, task_dims = get_task_specific_dataset(
            task_name=args.task_name,
            max_trajectory_length=args.max_trajectory_length
        )
        
        # Update args with task-specific dimensions
        args.traj_file = traj_path
        args.feat_dim = task_dims['joint_dims']  # Use joint dimensions (obs + action)
        
        print(f"✅ Using task-specific dataset: {traj_path}")
        print(f"📊 Task dimensions: {task_dims}")
        print(f"🎯 Model input/output dimension: {args.feat_dim}")
    else:
        # Use provided trajectory file
        print(f"📁 Using provided trajectory file: {args.traj_file}")
        # Try to determine dimensions from file
        import numpy as np
        try:
            arr = np.load(args.traj_file)
            args.feat_dim = arr.shape[2]
            print(f"📊 Detected feature dimension: {args.feat_dim}")
        except:
            print("⚠️  Could not determine feature dimension from file, using default")
            args.feat_dim = 70  # Default fallback
    
    # Start training
    train(args)
