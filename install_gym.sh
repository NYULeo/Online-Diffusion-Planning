#!/bin/bash

# Unified installation script for Gymnasium environments (MuJoCo + Robotics)
# Uses uv for fast and reliable package management

echo "Installing Gymnasium environment dependencies (MuJoCo + Robotics) using uv..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed successfully!"
else
    echo "uv found: $(uv --version)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install unified Gymnasium dependencies using uv
echo "Installing unified Gymnasium requirements with uv..."
uv pip install -r requirements/requirements_gym.txt

# Install PyTorch with CUDA support separately with increased timeout
echo "Installing PyTorch with CUDA support..."
export UV_HTTP_TIMEOUT=300  # Increase timeout to 5 minutes
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# If the above fails, try CPU-only PyTorch as fallback
if [ $? -ne 0 ]; then
    echo "CUDA PyTorch installation failed. Installing CPU-only PyTorch as fallback..."
    uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
fi

# Verify installation
echo "Verifying installation..."
python3 -c "
import gymnasium as gym
import gymnasium_robotics
import mujoco
import torch
print('✓ Gymnasium with MuJoCo installed successfully')
print('✓ Gymnasium Robotics installed successfully')
print('✓ MuJoCo physics engine installed successfully')
print('✓ PyTorch installed successfully')
"

echo "Installation completed successfully!"
echo ""
echo "Virtual environment is activated. You can now use both MuJoCo and Robotics environments:"
echo "  python tests/example_mujoco_usage.py"
echo "  python tests/example_robotics_usage.py"
echo ""
echo "Or run the tests:"
echo "  pytest tests/test_gym_mujoco_env.py"
echo "  pytest tests/test_gym_robotics_env.py"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate" 