# HumanoidBench Diffusion Trajectory Planning

This repository implements diffusion models for trajectory planning on the HumanoidBench dataset, supporting both task-conditioned and task-specific training approaches.

## Dataset Structure

The code expects a `hbench.pickle` file in the parent directory containing 14 HumanoidBench tasks:

```
hbench.pickle
├── h1-run-v0 (70 features: 51 obs + 19 actions)
├── h1-walk-v0 (70 features: 51 obs + 19 actions)
├── h1-stand-v0 (70 features: 51 obs + 19 actions)
├── h1-reach-v0 (76 features: 57 obs + 19 actions)
├── h1-balance_hard-v0 (96 features: 77 obs + 19 actions)
├── h1-sit_simple-v0 (70 features: 51 obs + 19 actions)
├── h1-stair-v0 (70 features: 51 obs + 19 actions)
├── h1-sit_hard-v0 (83 features: 64 obs + 19 actions)
├── h1-maze-v0 (70 features: 51 obs + 19 actions)
├── h1-crawl-v0 (70 features: 51 obs + 19 actions)
├── h1-balance_simple-v0 (83 features: 64 obs + 19 actions)
├── h1-hurdle-v0 (70 features: 51 obs + 19 actions)
├── h1-pole-v0 (70 features: 51 obs + 19 actions)
└── h1-slide-v0 (70 features: 51 obs + 19 actions)
```

**Note**: Different tasks have different observation dimensions, so task-specific training is recommended.

## Installation

```bash
pip install torch torchvision torchaudio numpy
```

## Usage

### Task-Specific Training (Recommended)

Train separate models for each task:

```bash
# Train model for running task
python3 main.py --task_specific --task_name h1-run-v0 --epochs 50 --batch_size 128

# Train model for walking task
python3 main.py --task_specific --task_name h1-walk-v0 --epochs 50 --batch_size 128

# Train model for reach task (different feature dimension)
python3 main.py --task_specific --task_name h1-reach-v0 --epochs 50 --batch_size 128
```

### Task-Conditioned Training

Train one model for multiple tasks (only works for tasks with same feature dimensions):

```bash
# Train on tasks with same feature dimensions (70 features)
python3 main.py --tasks h1-run-v0 h1-walk-v0 h1-stand-v0 --epochs 50 --batch_size 128

# Train without task conditioning
python3 main.py --tasks h1-run-v0 h1-walk-v0 h1-stand-v0 --no_task_conditioning --epochs 50 --batch_size 128
```

### Command Line Arguments

```bash
python3 main.py [OPTIONS]

Options:
  --task_specific              Train separate models for each task
  --task_name TASK_NAME        Specific task name for task-specific training
  --tasks TASKS [TASKS ...]    List of tasks for conditional training
  --no_task_conditioning       Disable task conditioning
  --backbone {unet,transformer} Model backbone
  --epochs EPOCHS              Number of training epochs
  --batch_size BATCH_SIZE      Batch size
  --lr LR                      Learning rate
  --hidden HIDDEN              Hidden dimension
  --time_dim TIME_DIM          Time embedding dimension
  --pos_dim POS_DIM            Positional embedding dimension
  --steps STEPS                Number of sampling steps
  --eta ETA                    Sampling noise level
  --s S                        SDE schedule parameter
  --weight {sigma2,beta}       Loss weighting
  --outdir OUTDIR              Output directory
  --cpu                        Use CPU for training
  --log_every LOG_EVERY        Log frequency
  --sample_every SAMPLE_EVERY  Sample generation frequency
```

## Model Architectures

### UNet Backbone
- 1D convolutional architecture for temporal data
- Downsampling and upsampling with residual connections
- Supports both conditional and task-specific variants

### Transformer Backbone
- Self-attention based architecture
- Full attention over temporal horizon
- Supports both conditional and task-specific variants

## Mathematical Formulation

### Task-Specific Models
Each task gets its own model learning:
$$p_i(\tau) = \text{Model}_i(\tau, t)$$

### Task-Conditioned Models
One model learns conditional distribution:
$$p(\tau | c) = \text{Model}(\tau, t, c)$$

### Training Objective
Denoising Score Matching (DSM):
$$\mathcal{L} = \mathbb{E}_{t,\tau_0} \left[ w(t) \| \epsilon_\theta(\tau_t, t) - \epsilon \|^2 \right]$$

### Sampling
Reverse-time SDE:
$$dx_t = [f(t)x_t - g²(t)s_θ(x_t, t)]dt + ηg(t)dw_t$$

## File Structure

```
Pretrain/
├── Backbone.py              # Model architectures
├── Dataset.py               # Dataset loading and preprocessing
├── main.py                  # Main training script
├── train.py                 # Training functions
├── utils.py                 # Utility functions
└── README.md               # This file
```

## Notes

- The `hbench.pickle` file should be placed in the parent directory
- Different tasks have different feature dimensions, so task-specific training is recommended
- Training outputs are saved to the specified output directory
- Models automatically adapt to the feature dimensions of each task
