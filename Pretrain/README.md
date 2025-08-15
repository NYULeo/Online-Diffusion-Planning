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

### Training SDE Algorithm Details

**Training SDE** is a diffusion-based trajectory planning method that:

1. **Forward Process**: Gradually adds noise to expert trajectories using a cosine noise schedule
   - Noise schedule: β(t) = 1 - (1 - β₀)cos(πt/2T)
   - Parameter s = 0.008 controls the noise level

2. **Reverse Process**: Learns to denoise trajectories by predicting the noise at each timestep
   - Uses Denoising Score Matching (DSM) loss
   - Supports sigma2 weighting for better convergence

3. **Trajectory Format**: Trains on complete state-action sequences
   - Input: (s₁, a₁, s₂, a₂, ..., sₙ, aₙ)
   - No artificial windowing or segmentation

4. **Model Architecture**: 
   - **Transformer**: Better for long-range temporal dependencies
   - **UNet**: Alternative architecture for spatial-temporal modeling

## Environment and Dataset

### HumanoidBench Environment
This code uses the [HumanoidBench](https://humanoid-bench.github.io) environment, which is the first-of-its-kind simulated humanoid robot benchmark featuring:
- **27 distinct whole-body control tasks** with unique challenges
- **Unitree H1 humanoid robot** with two dexterous Shadow Hands
- **High-dimensional observation space** including proprioceptive state, visual observations, and tactile sensing
- **Complex coordination requirements** for locomotion and manipulation tasks

### Offline Dataset
The training data comes from [SimbaV2](https://dojeon-ai.github.io/SimbaV2/) - a state-of-the-art reinforcement learning algorithm that:
- Achieves superior performance on 57 continuous control tasks across 4 domains
- Uses hyperspherical normalization for scalable deep reinforcement learning
- Provides high-quality expert trajectories for offline training
- Demonstrates consistent performance improvements with increased model size and computation

### Dataset Structure
The `hbench.pickle` file contains:
- **14 tasks**: h1-run-v0, h1-walk-v0, h1-stand-v0, h1-reach-v0, h1-balance_hard-v0, h1-sit_simple-v0, h1-stair-v0, h1-sit_hard-v0, h1-maze-v0, h1-crawl-v0, h1-balance_simple-v0, h1-hurdle-v0, h1-pole-v0, h1-slide-v0
- **Trajectory format**: Each task contains multiple expert trajectories (typically 3 trajectories per task)
- **Observation space**: 51-dimensional state vectors (joint positions, velocities, sensor data)
- **Action space**: 19-dimensional action vectors (joint torques)
- **Trajectory length**: Up to 500,000 timesteps per trajectory
- **Data structure**: Each trajectory contains observation, action, reward, terminated, truncated, and next_observation fields
- **Data format**: NumPy arrays with shape (timesteps, dimensions) for observations and actions

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

Based on comprehensive analysis of the Training SDE implementation, research in diffusion trajectory planning, and empirical results from HumanoidBench and SimbaV2:

### **Research Foundation**
The optimal hyperparameters are derived from:
- **Training SDE methodology**: Cosine noise schedule with s=0.008 (from DDPM research)
- **Diffusion model best practices**: Learning rate 2e-4, AdamW optimizer, gradient clipping
- **Transformer architecture research**: Temporal attention for long sequences
- **SimbaV2 empirical results**: Hyperspherical normalization and scaling insights
- **HumanoidBench task analysis**: Task-specific requirements for different complexity levels

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
- **S 0.008**: Cosine noise schedule parameter (optimal for Training SDE, from DDPM research)
- **Weight sigma2**: DSM loss weighting (better than beta for trajectory planning)
- **Eta 1.0**: Full stochastic sampling in reverse-time SDE
- **Steps 1000**: Sufficient denoising steps for high-quality trajectories

### **Model architecture recommendations**
- **Transformer backbone**: Better temporal attention for long trajectories (from SimbaV2 research)
- **Hidden 256**: Optimal capacity for humanoid control tasks
- **Time_dim 128**: Standard for diffusion time embeddings
- **Pos_dim 128**: Adequate for trajectory positional encoding

### **Training parameters**
- **Learning rate 2e-4**: Standard for diffusion models (from DDPM research)
- **AdamW optimizer**: Better weight decay and convergence (from SimbaV2)
- **Batch size 64**: Optimal for transformer training without memory issues
- **Grad_clip 1.0**: Prevents gradient explosion in diffusion training

### **Task-specific optimizations**
- **Locomotion tasks** (run, walk): transformer + hidden=256 + epochs=100
- **Manipulation tasks** (reach, balance): transformer + hidden=512 + epochs=120
- **Complex tasks** (maze, stair): transformer + hidden=512 + epochs=150

### **Hardware-specific adjustments**
- **GPU memory limited**: batch_size=32, hidden=128, max_trajectory_length=500
- **High-end GPU**: batch_size=128, hidden=512, max_trajectory_length=2000

### **Optimal configurations by task type**

#### **Locomotion Tasks** (run, walk, crawl)
```bash
python3 main.py --task_name h1-run-v0 \
    --backbone transformer \
    --hidden 256 \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --steps 1000 \
    --s 0.008 \
    --weight sigma2
```

#### **Manipulation Tasks** (reach, balance, sit)
```bash
python3 main.py --task_name h1-reach-v0 \
    --backbone transformer \
    --hidden 512 \
    --epochs 120 \
    --batch_size 64 \
    --lr 2e-4 \
    --steps 1000 \
    --s 0.008 \
    --weight sigma2
```

#### **Complex Tasks** (maze, stair, hurdle)
```bash
python3 main.py --task_name h1-maze-v0 \
    --backbone transformer \
    --hidden 512 \
    --epochs 150 \
    --batch_size 64 \
    --lr 2e-4 \
    --steps 1000 \
    --s 0.008 \
    --weight sigma2
```

### **Research-backed recommendations**
Based on SimbaV2's hyperspherical normalization research:
- **Scaling**: Performance improves with larger models and increased computation
- **Stability**: Use consistent effective learning rates across layers
- **Normalization**: Consider hyperspherical normalization for better training stability
- **Training SDE specifics**: 
  - Cosine noise schedule with s=0.008 provides optimal noise levels
  - Sigma2 weighting outperforms beta weighting for trajectory planning
  - 1000 denoising steps sufficient for high-quality trajectory generation
  - Full stochastic sampling (eta=1.0) recommended for exploration

## Available tasks

All 14 HumanoidBench tasks: h1-run-v0, h1-walk-v0, h1-stand-v0, h1-reach-v0, h1-balance_hard-v0, h1-sit_simple-v0, h1-stair-v0, h1-sit_hard-v0, h1-maze-v0, h1-crawl-v0, h1-balance_simple-v0, h1-hurdle-v0, h1-pole-v0, h1-slide-v0
