# Diffusion Planner Pretraining

This code trains diffusion models for HumanoidBench tasks using the Training SDE methodology.

## What it does

- Trains separate diffusion models for each of the 14 HumanoidBench tasks
- Uses complete trajectories from the pickle dataset (no artificial windowing)
- Automatically detects correct observation/action dimensions for each task
- Supports UNet and Transformer architectures

## Training Algorithm

The code implements **Training SDE** - a diffusion-based approach that:
- Learns to denoise trajectories by reversing a stochastic differential equation
- Uses cosine noise schedule with configurable parameters
- Supports both sigma2 and beta loss weighting
- Trains on full trajectories: s1, a1, s2, a2, ..., sN, aN

## How to start

### Basic training
```bash
# Train on h1-run-v0 task
python3 main.py --task_name h1-run-v0 --epochs 50 --batch_size 64

# Train on h1-balance_hard-v0 task  
python3 main.py --task_name h1-balance_hard-v0 --epochs 50 --batch_size 64
```

### Recommended settings
```bash
python3 main.py --task_name h1-run-v0 \
    --backbone transformer \
    --hidden 256 \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --max_trajectory_length 1000
```

### Key parameters
- `--task_name`: Task to train on (e.g., "h1-run-v0", "h1-walk-v0")
- `--backbone`: Model architecture ("unet" or "transformer") 
- `--epochs`: Training epochs
- `--batch_size`: Batch size
- `--max_trajectory_length`: Max trajectory length (use -1 for full trajectory)

## Optimal hyperparameters

Based on comprehensive analysis of the Training SDE implementation and research in diffusion trajectory planning:

### **Best overall configuration**
```bash
python3 main.py --task_name h1-run-v0 \
    --backbone transformer \
    --hidden 256 \
    --time_dim 128 \
    --pos_dim 128 \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --optimizer adamw \
    --grad_clip 1.0 \
    --steps 1000 \
    --eta 1.0 \
    --s 0.008 \
    --weight sigma2 \
    --max_trajectory_length 1000
```

### **Training SDE specific parameters**
- **S 0.008**: Cosine noise schedule parameter (optimal for Training SDE)
- **Weight sigma2**: DSM loss weighting (better than beta for trajectory planning)
- **Eta 1.0**: Full stochastic sampling in reverse-time SDE
- **Steps 1000**: Sufficient denoising steps for high-quality trajectories

### **Model architecture recommendations**
- **Transformer backbone**: Better temporal attention for long trajectories
- **Hidden 256**: Optimal capacity for humanoid control tasks
- **Time_dim 128**: Standard for diffusion time embeddings
- **Pos_dim 128**: Adequate for trajectory positional encoding

### **Training parameters**
- **Learning rate 2e-4**: Standard for diffusion models (from DDPM research)
- **AdamW optimizer**: Better weight decay and convergence
- **Batch size 64**: Optimal for transformer training without memory issues
- **Grad_clip 1.0**: Prevents gradient explosion in diffusion training

### **Task-specific optimizations**
- **Locomotion tasks** (run, walk): transformer + hidden=256 + epochs=100
- **Manipulation tasks** (reach, balance): transformer + hidden=512 + epochs=120
- **Complex tasks** (maze, stair): transformer + hidden=512 + epochs=150

### **Hardware-specific adjustments**
- **GPU memory limited**: batch_size=32, hidden=128, max_trajectory_length=500
- **High-end GPU**: batch_size=128, hidden=512, max_trajectory_length=2000

## Available tasks

All 14 HumanoidBench tasks: h1-run-v0, h1-walk-v0, h1-stand-v0, h1-reach-v0, h1-balance_hard-v0, h1-sit_simple-v0, h1-stair-v0, h1-sit_hard-v0, h1-maze-v0, h1-crawl-v0, h1-balance_simple-v0, h1-hurdle-v0, h1-pole-v0, h1-slide-v0
