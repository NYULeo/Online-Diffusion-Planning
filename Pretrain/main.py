#!/usr/bin/env python3
"""
Main training script for HumanoidBench diffusion trajectory planning.
Uses local hbench.pickle dataset with 14 tasks.
Supports both task-conditioned and task-specific training.
"""
import argparse
import os
from train import train

def parse_args():
    p = argparse.ArgumentParser(description="Diffusion planner for HumanoidBench (UNet + Transformer backbones)")
    p.add_argument("--traj_file", type=str, default="", help="training .npy [N,H,D] or .pkl file (if not provided, will use local hbench.pickle)")
    p.add_argument("--ctx_file", type=str, default="", help="optional context .npy [N,C]")
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
    
    # Task selection options
    p.add_argument("--tasks", type=str, nargs='+', default=None, 
                   help="Specific tasks to train on (e.g., h1-run-v0 h1-walk-v0). If not specified, uses all 14 tasks.")
    p.add_argument("--no_task_conditioning", action="store_true", 
                   help="Disable task conditioning (train on all tasks without task-specific conditioning)")
    p.add_argument("--task_specific", action="store_true",
                   help="Train separate models for each task (no conditioning, each task gets its own model)")
    p.add_argument("--task_name", type=str, default=None,
                   help="Specific task name for task-specific training (e.g., h1-run-v0)")
    
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
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Prepare HumanoidBench dataset if no traj_file provided
    if not args.traj_file:
        print("🤖 Preparing HumanoidBench dataset from local hbench.pickle...")
        
        if args.task_specific and args.task_name:
            # Task-specific training: train separate model for one task
            print(f"🎯 Training task-specific model for: {args.task_name}")
            from Dataset import get_task_specific_dataset
            traj_path = get_task_specific_dataset(args.task_name)
            args.traj_file = traj_path
            print(f"✅ Using task-specific dataset: {traj_path}")
            
        elif args.task_specific and args.tasks:
            # Task-specific training: train separate models for multiple tasks
            print(f"🎯 Training task-specific models for: {args.tasks}")
            from Dataset import get_task_specific_dataset
            # For now, use the first task (you can extend this to train multiple models)
            traj_path = get_task_specific_dataset(args.tasks[0])
            args.traj_file = traj_path
            print(f"✅ Using task-specific dataset for {args.tasks[0]}: {traj_path}")
            
        else:
            # Standard training with or without task conditioning
            task_conditioning = not args.no_task_conditioning
            
            if args.tasks:
                print(f"🎯 Using selected tasks: {args.tasks}")
                print(f"📊 Task conditioning: {'enabled' if task_conditioning else 'disabled'}")
            else:
                print("🎯 Using all 14 HumanoidBench tasks")
                print(f"📊 Task conditioning: {'enabled' if task_conditioning else 'disabled'}")
            
            # Prepare dataset with task selection
            from Dataset import prepare_humanoidbench_for_training
            traj_path = prepare_humanoidbench_for_training(
                data_path="../hbench.pickle",  # Use local file
                selected_tasks=args.tasks,
                task_conditioning=task_conditioning
            )
            
            if traj_path:
                args.traj_file = traj_path
                print(f"✅ Using HumanoidBench dataset: {traj_path}")
                
                # Check if task conditions were created
                task_cond_path = traj_path.replace('_trajectories.npy', '_task_conditions.npy')
                if os.path.exists(task_cond_path) and task_conditioning:
                    print(f"✅ Task conditions available: {task_cond_path}")
                elif not task_conditioning:
                    print("ℹ️  Training without task conditioning")
            else:
                print("❌ Failed to prepare HumanoidBench dataset")
                print("Please ensure hbench.pickle is in the parent directory")
                raise SystemExit(1)
    
    train(args)
