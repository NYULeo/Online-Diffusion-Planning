#!/bin/bash
# Script to activate the Python 3.10 environment with D4RL

echo "Activating Python 3.10 environment with D4RL..."

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Set Python 3.10 as local version
pyenv local 3.10.12

# Activate the virtual environment
source .venv310/bin/activate

echo "Environment activated!"
echo "Python version: $(python --version)"
echo "D4RL version: $(python -c 'import pkg_resources; print(pkg_resources.get_distribution("D4RL").version)')"
echo ""
echo "✅ Installed packages:"
echo "   - PyTorch, JAX, NumPy, Pandas"
echo "   - Gymnasium, Gymnasium-Robotics"
echo "   - Matplotlib, MoviePy, Pygame"
echo "   - Wandb, Tyro, Loguru"
echo "   - HuggingFace Hub, Minari"
echo "   - TensorDict, TorchRL"
echo "   - MediaPy, SymPy"
echo ""
echo "📝 Note: Some D4RL environments require MuJoCo to be installed separately."
echo "   If you need MuJoCo environments, follow the instructions at:"
echo "   https://github.com/openai/mujoco-py#install-mujoco"
