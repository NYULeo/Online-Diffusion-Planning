# Diffusion Planner Pretraining

This code trains diffusion models for HumanoidBench tasks using the **Training SDE** methodology.

## 📋 Overview

- **Purpose**: Train diffusion models for 14 HumanoidBench tasks
- **Method**: Training SDE (Stochastic Differential Equation) diffusion
- **Data**: Complete trajectories from SimbaV2 expert demonstrations
- **Architecture**: UNet and Transformer support

## 🧠 Training Algorithm

### Training SDE Mathematical Formulation

The **Training SDE** method implements a diffusion process with the following components:

#### Forward Process (Noise Addition)
The forward process gradually adds noise to expert trajectories:

**dxₜ/dt = f(xₜ, t) + g(t) · εₜ**

where:
- **xₜ** is the trajectory at time **t**
- **f(xₜ, t)** is the drift term
- **g(t)** is the diffusion coefficient
- **εₜ ~ N(0, I)** is Gaussian noise

#### Cosine Noise Schedule
We use a cosine noise schedule for optimal performance:

**β(t) = 1 - (1 - β₀) cos(πt/2T)**

where:
- **β₀ = 0.008** (optimal parameter from DDPM research)
- **T** is the total number of diffusion steps
- **t ∈ [0, T]** is the current timestep

#### Reverse Process (Denoising)
The model learns to reverse the noise addition:

**dxₜ/dt = f(xₜ, t) - ½g²(t) ∇ₓₜ log pₜ(xₜ)**

#### Loss Function
We use Denoising Score Matching (DSM) with sigma2 weighting:

**L = E[t,x₀,ε] [σ²(t) ||ε - εθ(xₜ, t)||²]**

where:
- **εθ** is the noise prediction network
- **σ²(t)** is the sigma2 weighting scheme
- **ε** is the ground truth noise

## 🌍 Environment and Dataset

### HumanoidBench Environment

| Feature | Description |
|---------|-------------|
| **Robot** | Unitree H1 humanoid with Shadow Hands |
| **Tasks** | 27 distinct whole-body control tasks |
| **Observation Space** | 51-dimensional (joint positions, velocities, sensors) |
| **Action Space** | 19-dimensional (joint torques) |
| **Complexity** | Locomotion, manipulation, and coordination tasks |

### SimbaV2 Offline Dataset

| Property | Value |
|----------|-------|
| **Source** | [SimbaV2](https://dojeon-ai.github.io/SimbaV2/) expert demonstrations |
| **Tasks Covered** | 57 continuous control tasks across 4 domains |
| **Normalization** | Hyperspherical normalization for stability |
| **Quality** | State-of-the-art RL performance |

### Dataset Structure

The `hbench.pickle` file contains:

| Component | Description |
|-----------|-------------|
| **Tasks** | 14 HumanoidBench tasks |
| **Trajectories** | ~3 expert trajectories per task |
| **Observation Dim** | 51-dimensional state vectors |
| **Action Dim** | 19-dimensional action vectors |
| **Max Length** | Up to 500,000 timesteps per trajectory |
| **Format** | NumPy arrays: `(timesteps, dimensions)` |

#### Available Tasks

| Task Category | Tasks |
|--------------|-------|
| **Locomotion** | `h1-run-v0`, `h1-walk-v0`, `h1-crawl-v0` |
| **Balance** | `h1-stand-v0`, `h1-balance_simple-v0`, `h1-balance_hard-v0` |
| **Manipulation** | `h1-reach-v0`, `h1-sit_simple-v0`, `h1-sit_hard-v0` |
| **Complex** | `h1-stair-v0`, `h1-maze-v0`, `h1-hurdle-v0`, `h1-pole-v0`, `h1-slide-v0` |

## 🚀 Quick Start

### Basic Training

```bash
# Train on locomotion task
python3 main.py --task_name h1-run-v0 --epochs 50 --batch_size 64

# Train on manipulation task
python3 main.py --task_name h1-reach-v0 --epochs 50 --batch_size 64
```

### Recommended Configuration

```bash
python3 main.py --task_name h1-run-v0 \
    --backbone transformer \
    --hidden 256 \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --max_trajectory_length 1000
```

## ⚙️ Optimal Hyperparameters

### Research Foundation

Our hyperparameters are derived from:

| Research Area | Key Insights |
|---------------|--------------|
| **Training SDE** | Cosine noise schedule with s=0.008 |
| **Diffusion Models** | Learning rate 2e-4, AdamW optimizer |
| **Transformer Architecture** | Temporal attention for long sequences |
| **SimbaV2** | Hyperspherical normalization benefits |
| **HumanoidBench** | Task-specific complexity requirements |

### Best Overall Configuration

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

### Training SDE Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **s** | 0.008 | Optimal cosine noise schedule (DDPM research) |
| **weight** | sigma2 | Better convergence than beta weighting |
| **eta** | 1.0 | Full stochastic sampling in reverse SDE |
| **steps** | 1000 | Sufficient for high-quality trajectories |

### Model Architecture

| Component | Recommendation | Reason |
|-----------|----------------|--------|
| **Backbone** | Transformer | Better temporal attention for long trajectories |
| **Hidden Dim** | 256 | Optimal capacity for humanoid control |
| **Time Dim** | 128 | Standard for diffusion time embeddings |
| **Pos Dim** | 128 | Adequate for trajectory positional encoding |

### Task-Specific Configurations

#### Locomotion Tasks (run, walk, crawl)

| Parameter | Value |
|-----------|-------|
| **Backbone** | transformer |
| **Hidden** | 256 |
| **Epochs** | 100 |
| **Batch Size** | 64 |

#### Manipulation Tasks (reach, balance, sit)

| Parameter | Value |
|-----------|-------|
| **Backbone** | transformer |
| **Hidden** | 512 |
| **Epochs** | 120 |
| **Batch Size** | 64 |

#### Complex Tasks (maze, stair, hurdle)

| Parameter | Value |
|-----------|-------|
| **Backbone** | transformer |
| **Hidden** | 512 |
| **Epochs** | 150 |
| **Batch Size** | 64 |

### Hardware-Specific Adjustments

| GPU Memory | Configuration |
|------------|---------------|
| **Limited (8GB)** | `batch_size=32, hidden=128, max_length=500` |
| **Standard (16GB)** | `batch_size=64, hidden=256, max_length=1000` |
| **High-end (24GB+)** | `batch_size=128, hidden=512, max_length=2000` |

## 📊 Performance Insights

### Research-Backed Recommendations

Based on SimbaV2's hyperspherical normalization research:

1. **Scaling**: Performance improves with larger models and increased computation
2. **Stability**: Use consistent effective learning rates across layers
3. **Normalization**: Hyperspherical normalization improves training stability

### Training SDE Specifics

- **Cosine noise schedule** with s=0.008 provides optimal noise levels
- **Sigma2 weighting** outperforms beta weighting for trajectory planning
- **1000 denoising steps** sufficient for high-quality trajectory generation
- **Full stochastic sampling** (eta=1.0) recommended for exploration

## 🔧 Key Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--task_name` | Task to train on | `h1-run-v0`, `h1-walk-v0` |
| `--backbone` | Model architecture | `unet`, `transformer` |
| `--epochs` | Training epochs | 50, 100, 150 |
| `--batch_size` | Batch size | 32, 64, 128 |
| `--max_trajectory_length` | Max trajectory length | 500, 1000, -1 (full) |

## 📚 References

- **HumanoidBench**: [https://humanoid-bench.github.io](https://humanoid-bench.github.io)
- **SimbaV2**: [https://dojeon-ai.github.io/SimbaV2/](https://dojeon-ai.github.io/SimbaV2/)
- **Training SDE**: Based on DDPM and diffusion model literature
- **Transformer Architecture**: Temporal attention for sequence modeling
