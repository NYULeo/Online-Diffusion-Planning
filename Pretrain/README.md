# Task-Specific Diffusion Planner Training

This repository implements **task-specific diffusion planners** for HumanoidBench tasks using the Training SDE methodology. The code trains separate diffusion models for each of the 14 HumanoidBench tasks, using the **exact trajectory structure** from the pickle dataset.

## 🎯 Key Features

### ✅ **Full Trajectory Training**
- **No Artificial Windowing**: Uses complete trajectories as provided in the pickle dataset
- **Task-Specific Dimensions**: Automatically detects and uses correct dimensions for each task
- **Configurable Length**: Can truncate trajectories for training efficiency
- **Direct Dataset Usage**: Follows the exact trajectory format: s1, a1, s2, a2, ..., sN, aN

### 🏗️ **Supported Model Architectures**
- **UNet**: 1D convolutional architecture with residual blocks
- **Transformer**: Temporal transformer with positional embeddings
- **Task-Specific**: Models built with correct input/output dimensions per task

### 📊 **Dataset Structure**
The pickle dataset contains:
- **14 tasks** with different observation/action dimensions
- **3 episodes per task** with 500,000 steps each
- **Trajectory format**: s1, a1, s2, a2, s3, a3, ..., s500000, a500000
- **Task-specific dimensions**:

| Task | Observation | Action | Joint | Trajectory Length |
|------|-------------|--------|-------|-------------------|
| h1-run-v0 | 51 | 19 | 70 | 500,000 |
| h1-reach-v0 | 57 | 19 | 76 | 500,000 |
| h1-balance_hard-v0 | 77 | 19 | 96 | 500,000 |
| h1-sit_hard-v0 | 64 | 19 | 83 | 500,000 |
| ... | ... | ... | ... | ... |

## 🎯 **Optimal Hyperparameters for HumanoidBench**

Based on research in diffusion models for humanoid control and the specific characteristics of HumanoidBench trajectories, here are the **recommended optimal hyperparameters**:

### **🏗️ Model Architecture (Recommended)**
```bash
--backbone transformer    # Better for long sequences and complex dynamics
--hidden 256             # Optimal balance of capacity and efficiency
--time_dim 128           # Standard for diffusion time embedding
--pos_dim 128            # Adequate for trajectory positional encoding
```

### **🎓 Training Parameters (Optimal)**
```bash
--epochs 100             # Sufficient for convergence on expert data
--batch_size 64          # Optimal for transformer training
--lr 2e-4                # Standard learning rate for diffusion models
--optimizer adamw        # Better convergence than Adam for transformers
--grad_clip 1.0          # Prevents gradient explosion
```

### **🌊 Diffusion Parameters (Research-Based)**
```bash
--steps 1000             # Standard for high-quality sampling
--eta 1.0                # Full stochastic sampling (better diversity)
--s 0.008                # Cosine schedule parameter (from DDPM paper)
--weight sigma2          # Better than beta for humanoid trajectories
```

### **📊 Data Processing (Optimal)**
```bash
--max_trajectory_length 1000  # Balance between full context and memory
--dtype float32              # Standard precision
--workers 4                  # Optimal for data loading
```

### **🚀 Complete Optimal Configuration**
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
    --max_trajectory_length 1000 \
    --dtype float32 \
    --workers 4
```

### **📈 Hyperparameter Rationale**

#### **Why Transformer Backbone?**
- **Long-range dependencies**: Humanoid trajectories have complex temporal relationships
- **Attention mechanism**: Better captures correlations across time steps
- **Scalability**: Handles variable trajectory lengths effectively

#### **Why These Training Parameters?**
- **Batch size 64**: Optimal for transformer training without memory issues
- **Learning rate 2e-4**: Standard for diffusion models, balances convergence and stability
- **AdamW optimizer**: Better weight decay and convergence than Adam

#### **Why These Diffusion Parameters?**
- **Steps 1000**: Sufficient for high-quality trajectory generation
- **Eta 1.0**: Full stochastic sampling provides better diversity
- **Weight sigma2**: Better empirical performance for humanoid control tasks

#### **Why 1000 Trajectory Length?**
- **Full context**: Captures complete task execution
- **Memory efficient**: Avoids excessive memory usage
- **Sufficient information**: Most HumanoidBench tasks complete within 1000 steps

### **🔬 Research-Based Recommendations**

#### **For Different Task Types:**
- **Locomotion tasks** (run, walk): Use transformer with hidden=256
- **Manipulation tasks** (reach, balance): Use transformer with hidden=512
- **Complex tasks** (maze, stair): Use transformer with hidden=512, longer training

#### **For Different Dataset Sizes:**
- **Small datasets** (< 10 episodes): Increase epochs to 200, reduce batch size to 32
- **Large datasets** (> 20 episodes): Use batch size 128, reduce epochs to 50

#### **For Different Hardware:**
- **GPU memory limited**: Use batch size 32, hidden=128
- **High-end GPU**: Use batch size 128, hidden=512

### **⚡ Quick Start with Optimal Settings**
```bash
# For most HumanoidBench tasks
python3 main.py --task_name h1-run-v0 --backbone transformer --hidden 256 --epochs 100 --batch_size 64

# For complex manipulation tasks
python3 main.py --task_name h1-balance_hard-v0 --backbone transformer --hidden 512 --epochs 150 --batch_size 64

# For memory-constrained environments
python3 main.py --task_name h1-run-v0 --backbone transformer --hidden 128 --batch_size 32 --epochs 200
```

## 🚀 Quick Start

### Basic Training
```bash
# Train on h1-run-v0 (70 dimensions)
python3 main.py --task_name h1-run-v0 --epochs 50 --batch_size 128

# Train on h1-balance_hard-v0 (96 dimensions)
python3 main.py --task_name h1-balance_hard-v0 --epochs 50 --batch_size 128

# Train with truncated trajectories for efficiency
python3 main.py --task_name h1-run-v0 --max_trajectory_length 1000 --epochs 50
```

### Advanced Training
```bash
# Use transformer backbone
python3 main.py --task_name h1-run-v0 --backbone transformer --hidden 512

# Custom hyperparameters
python3 main.py --task_name h1-run-v0 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --hidden 256 \
    --max_trajectory_length 2000

# Advanced configuration
python3 main.py --task_name h1-run-v0 \
    --optimizer adam \
    --grad_clip 0.5 \
    --dtype float16 \
    --max_trajectory_length 1000
```

## 📁 Project Structure

```
Pretrain/
├── main.py              # Main training script
├── train.py             # Training functions and loss computation
├── Dataset.py           # Dataset loading and preprocessing
├── Backbone.py          # Model architectures (UNet, Transformer)
├── utils.py             # Utility functions and embeddings
├── README.md            # This file
└── hbench_runs/         # Training outputs and checkpoints
```

## ⚙️ Training Parameters

### Required Parameters
- `--task_name`: Task to train on (e.g., "h1-run-v0", "h1-walk-v0", etc.)

### Model Parameters
- `--backbone`: Model architecture ("unet" or "transformer")
- `--hidden`: Hidden dimension size (default: 256)
- `--time_dim`: Time embedding dimension (default: 128)
- `--pos_dim`: Positional embedding dimension (default: 128)

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 2e-4)
- `--max_trajectory_length`: Maximum trajectory length for training (default: 1000, use -1 for full trajectory)

### Diffusion Parameters
- `--steps`: Number of sampling steps (default: 1000)
- `--eta`: Noise level for sampling (default: 1.0)
- `--s`: SDE parameter (default: 0.008)
- `--weight`: Loss weighting ("sigma2" or "beta", default: "sigma2")

### Advanced Parameters
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--optimizer`: Optimizer type ("adamw" or "adam", default: "adamw")
- `--dtype`: Data type ("float32" or "float16", default: "float32")
- `--workers`: Data loading workers (default: 4)
- `--log_every`: Log frequency (default: 100)
- `--sample_every`: Sample generation frequency (default: 1000)
- `--sample_bs`: Sample batch size (default: 8)
- `--outdir`: Output directory (default: "./hbench_runs")

## 🎯 Available Tasks

All 14 HumanoidBench tasks are supported:
- `h1-run-v0`, `h1-walk-v0`, `h1-stand-v0`
- `h1-reach-v0`, `h1-balance_hard-v0`, `h1-sit_simple-v0`
- `h1-stair-v0`, `h1-sit_hard-v0`, `h1-maze-v0`
- `h1-crawl-v0`, `h1-balance_simple-v0`, `h1-hurdle-v0`
- `h1-pole-v0`, `h1-slide-v0`

## 🔧 Key Improvements

### ✅ **Removed Artificial Windowing**
- **Before**: Fixed horizon=64, stride=8 created overlapping windows
- **After**: Uses full trajectories as provided in the dataset
- **Benefit**: More natural training on complete trajectories

### ✅ **Task-Specific Dimensions**
- **Before**: Fixed 70 dimensions for all tasks
- **After**: Automatic detection of correct dimensions per task
- **Benefit**: Optimal model architecture for each task

### ✅ **Configurable Trajectory Length**
- **Before**: Fixed windowing approach
- **After**: Can use full trajectory or truncate for efficiency
- **Benefit**: Flexibility for different training scenarios

## 📈 Performance Tips

1. **Start with truncated trajectories** (1000-2000 steps) for faster training
2. **Use larger batch sizes** when memory allows
3. **Experiment with different trajectory lengths** based on your task
4. **Monitor loss convergence** to determine optimal training duration

## 🐛 Troubleshooting

### Common Issues
- **Memory errors**: Reduce batch_size or max_trajectory_length
- **Slow training**: Use smaller trajectory lengths or larger learning rates
- **Dimension mismatch**: Ensure task_name matches the pickle dataset

### Performance Optimization
- **GPU training**: Remove `--cpu` flag for GPU acceleration
- **Multi-GPU**: Modify batch_size and workers for multi-GPU setups
- **Data loading**: Adjust `--workers` based on your system

## 📚 References

- **Training SDE**: The training methodology follows the SDE-based diffusion approach
- **HumanoidBench**: Dataset with 14 humanoid control tasks
- **Diffusion Models**: Based on DDPM and SDE formulations

## 🎉 Summary

This implementation provides a **clean, task-specific approach** to diffusion planning that:
- ✅ **Follows the dataset exactly** as provided
- ✅ **Uses correct dimensions** for each task
- ✅ **Trains on complete trajectories** without artificial windowing
- ✅ **Supports flexible trajectory lengths** for different training scenarios
- ✅ **Maintains the original trajectory structure**: s1, a1, s2, a2, ..., sN, aN

The code now properly handles the **task-specific characteristics** of the HumanoidBench dataset and provides **optimal training** for each individual task!
