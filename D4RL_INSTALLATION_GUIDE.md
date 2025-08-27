# D4RL Installation Guide

## Issue Summary

The kitchen environment integration requires D4RL (Datasets for Deep Data-Driven Reinforcement Learning), but D4RL has a Python version constraint that requires Python < 3.11. However, your system is running Python 3.13.

## Current Status

✅ **Successfully Installed:**
- `loguru` - Logging utility
- `tyro` - Command line arguments  
- `h5py>=3.0.0` - Required for D4RL dataset loading

❌ **Installation Failed:**
- `d4rl` - Kitchen datasets (Python version constraint)

## Solutions

### Option 1: Use Python 3.10 or 3.9 (Recommended)

The most straightforward solution is to use a compatible Python version:

```bash
# Install Python 3.10 (if available)
brew install python@3.10

# Create a virtual environment with Python 3.10
python3.10 -m venv kitchen_env
source kitchen_env/bin/activate

# Install dependencies
pip install -r requirements/requirements.txt
```

### Option 2: Use Conda Environment

```bash
# Create conda environment with Python 3.10
conda create -n kitchen_env python=3.10
conda activate kitchen_env

# Install dependencies
pip install -r requirements/requirements.txt
```

### Option 3: Use Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
```

### Option 4: Manual D4RL Installation (Advanced)

If you want to force install D4RL on Python 3.13:

```bash
# Clone D4RL repository
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl

# Edit pyproject.toml to remove Python version constraint
# Change: python_requires=">=3.7,<3.11"
# To: python_requires=">=3.7"

# Install manually
pip install -e .
```

## Current Workaround

The kitchen environment has been updated to handle missing D4RL gracefully:

1. **Environment**: Will use a fallback environment (CartPole) if D4RL is not available
2. **Dataset**: Will raise a clear error message if D4RL is not available

## Testing Without D4RL

You can still test the kitchen environment structure without D4RL:

```python
from Environment.kitchen_env import KitchenEnv

# This will use a fallback environment
env = KitchenEnv("kitchen-partial-v0", num_envs=1)
print(f"Environment created: {env.env_name}")
print(f"Observation space: {env.num_obs} dimensions")
print(f"Action space: {env.num_actions} dimensions")
```

## Full Functionality

To get full kitchen environment functionality with real kitchen datasets:

1. **Use Python 3.10 or 3.9**
2. **Install all dependencies**: `pip install -r requirements/requirements.txt`
3. **Test kitchen environment**: 
   ```python
   from Environment.kitchen_env import KitchenEnv
   env = KitchenEnv("kitchen-partial-v0", num_envs=1)
   ```

## Alternative Datasets

If you cannot install D4RL, you can still use other datasets in the repository:

- **HumanoidBench**: `hbench.pickle` (already available)
- **Custom datasets**: Create your own datasets following the same format

## Next Steps

1. **Choose a solution** from the options above
2. **Install D4RL** with a compatible Python version
3. **Test the kitchen environment** with real datasets
4. **Use for diffusion planning experiments**

## Support

If you encounter issues:

1. Check Python version: `python3 --version`
2. Verify D4RL installation: `python3 -c "import d4rl; print('D4RL works!')"`
3. Test kitchen environment: `python3 -c "from Environment.kitchen_env import KitchenEnv; print('Kitchen env works!')"`
