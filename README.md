# Online Diffusion Planning

A reinforcement learning framework that uses diffusion models for online planning and decision-making in continuous control tasks.

## Overview

This repository implements diffusion-based planning algorithms for reinforcement learning, supporting multiple environments including Franka Kitchen, PointMaze, HumanoidBench, and IsaacLab. The framework consists of two main stages: **pretraining** diffusion planners on offline datasets and **finetuning** them for specific tasks.

## Features

- 🎯 **Diffusion-based Planning**: Uses stochastic differential equations (SDEs) for trajectory generation
- 🏗️ **Modular Architecture**: Separate modules for planners, transition kernels, and reward models
- 🌍 **Multiple Environments**: Support for various continuous control environments
- 🔄 **Pretraining & Finetuning**: Two-stage training pipeline for optimal performance
- 🚀 **Automated Workflow**: Scripts for seamless development and training cycles

## Repository Structure

```
Online-Diffusion-Planning/
├── Environment/          # Environment wrappers
│   ├── franka_kitchen_env.py
│   ├── pointmaze_env.py
│   ├── humanoid_bench_env.py
│   └── isaaclab_env.py
├── Pretrain/            # Pretraining scripts and models
│   ├── Planners/        # Diffusion planner models
│   ├── Transition_Kernel/  # Transition kernel models
│   ├── Rewards/         # Reward models
│   └── pretrain_script.py
├── Finetuning/          # Finetuning scripts
│   ├── Kernels/         # Finetuned kernels
│   ├── Planners/        # Finetuned planners
│   └── Rewards/         # Finetuned rewards
└── requirements/        # Dependency files
```

## Installation

### Prerequisites

- Python 3.10 (recommended for D4RL compatibility)
- CUDA-capable GPU (for training)
- Git

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/NYULeo/Online-Diffusion-Planning.git
cd Online-Diffusion-Planning
```

2. **Create a virtual environment**:
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements/requirements.txt
```

For macOS-specific dependencies:
```bash
pip install -r requirements/requirements_macos_fixed.txt
```

### Environment-Specific Setup

#### D4RL (for Kitchen and Maze environments)

D4RL requires Python < 3.11. If using Python 3.10:

```bash
pip install d4rl
```

For Python 3.11+, see `INSTALLATION_SUMMARY.md` for workarounds.

#### Isaac Lab (for IsaacLab environments)

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/).

## Quick Start

### Pretraining a Diffusion Planner

```bash
cd Pretrain
python pretrain_script.py
```

The script trains a diffusion planner on offline data. Modify `pretrain_script.py` to configure:
- Dataset name (`pointmaze`, `kitchen`, etc.)
- Specific dataset variant (`medium`, `partial`, etc.)
- Training hyperparameters (learning rate, batch size, etc.)

### Running Rollouts

```bash
python Planner_Rollout.py
```

This generates trajectories using the trained planner.

### Finetuning

```bash
cd Finetuning
python finetune_script.py
```

Finetunes pretrained models for specific tasks.

## Supported Environments

### PointMaze
- Variants: `umaze`, `medium`, `large`
- 2D navigation tasks with obstacles

### Franka Kitchen
- Variants: `partial`, `complete`, `mixed`
- Multi-task manipulation with a 9-DoF robot arm
- Tasks: microwave, kettle, light switch, slide cabinet, etc.

### HumanoidBench
- Humanoid locomotion and manipulation tasks

### IsaacLab
- Physics simulation environments using NVIDIA Isaac Sim

## Workflow Automation

The repository includes automated workflow scripts for seamless development. See `WORKFLOW_README.md` for detailed documentation.

### Quick Workflow

```bash
# Make script executable
chmod +x workflow.sh

# Run automated workflow
./workflow.sh "Your commit message"
```

This will:
1. Commit and push changes to GitHub
2. Update the remote Berkeley server
3. Start training
4. Download results when complete

## Documentation

- **`WORKFLOW_README.md`**: Detailed workflow automation guide
- **`INSTALLATION_SUMMARY.md`**: Installation notes and troubleshooting
- **`Pretrain/Pretraining and Sampling SDE.pdf`**: Theoretical background on SDEs
- **`Pretrain/Training SDE.pdf`**: Training methodology

## Citation

If you use this code in your research, please cite:

```bibtex
@software{online_diffusion_planning,
  title={Online Diffusion Planning},
  author={NYULeo},
  year={2025},
  url={https://github.com/NYULeo/Online-Diffusion-Planning}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This repository builds upon several open-source projects:
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [D4RL](https://github.com/Farama-Foundation/d4rl)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

## Contact

For questions or issues, please open an issue on GitHub.

