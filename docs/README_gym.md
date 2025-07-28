# Gymnasium Environments

Unified wrapper for [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) MuJoCo and [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) environments.

## Available Environments

### MuJoCo Environments
- `HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`
- `Ant-v4`, `Humanoid-v4`, `Swimmer-v4`
- `InvertedPendulum-v4`, `InvertedDoublePendulum-v4`
- `Reacher-v4`, `Pusher-v4`, `Thrower-v4`, `Striker-v4`

### Robotics Environments

**Maze Environments:**
- `maze2d-umaze-v1`, `maze2d-medium-v1`, `maze2d-large-v1`
- `antmaze-umaze-v0`, `antmaze-medium-play-v0`, `antmaze-large-play-v0`

**Adroit Arm Environments:**
- `pen-human-v0`, `hammer-human-v0`, `door-human-v0`, `relocate-human-v0`
- `pen-cloned-v0`, `hammer-cloned-v0`, `door-cloned-v0`, `relocate-cloned-v0`
- `pen-expert-v0`, `hammer-expert-v0`, `door-expert-v0`, `relocate-expert-v0`

**Franka Kitchen Environments:**
- `kitchen-complete-v0`, `kitchen-partial-v0`, `kitchen-mixed-v0`

## Installation

### Using uv (Recommended)

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the gymnasium environment dependencies:

```bash
# Install dependencies using uv
uv pip install -r requirements/requirements_gym.txt

# For CUDA support, install PyTorch separately (with increased timeout):
export UV_HTTP_TIMEOUT=300  # 5 minutes timeout
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# If CUDA installation fails, use CPU-only PyTorch:
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

Or create a virtual environment and install:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements/requirements_gym.txt

# For CUDA support, install PyTorch separately (with increased timeout):
export UV_HTTP_TIMEOUT=300  # 5 minutes timeout
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# If CUDA installation fails, use CPU-only PyTorch:
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

### Alternative: Using pip

```bash
pip install -r requirements/requirements_gym.txt
```

## Usage

### MuJoCo Environments

```python
from Environment.gym_mujoco_env import GymMuJoCoEnv

# Create MuJoCo environment
env = GymMuJoCoEnv("HalfCheetah-v5", num_envs=4)

# Reset and step
observations = env.reset()
actions = torch.randn(4, env.num_actions)
observations, rewards, dones, infos = env.step(actions)

env.close()
```

### Robotics Environments

```python
from Environment.gym_robotics_env import GymRoboticsEnv

# Create Robotics environment
env = GymRoboticsEnv("maze2d-umaze-v1", num_envs=4)

# Reset and step
observations = env.reset()
actions = torch.randn(4, env.num_actions)
observations, rewards, dones, infos = env.step(actions)

env.close()
```

## Multi-Goal API (Robotics)

Robotics environments use multi-goal API with observations containing:
- `observation`: Environment state
- `desired_goal`: Target goal
- `achieved_goal`: Current achievement

The wrapper automatically flattens these into a single tensor.

## Testing

```bash
# Run unified tests
pytest tests/test_gym_envs.py

# Run individual tests
pytest tests/test_gym_mujoco_env.py
pytest tests/test_gym_robotics_env.py

# Run examples
python tests/example_gym_usage.py
``` 