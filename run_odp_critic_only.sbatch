#!/bin/bash
#SBATCH --account=bhpx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --job-name=odp_critic
#SBATCH --output=/projects/bhpx/khu5/ODP/logs/%x-%j.log

# ============================================================================
# Critic training ONLY (teammate's runbook step 3). For A/B-testing the critic
# against a swapped-in reward checkpoint. Reward + kernel are NOT retrained.
# NOTE: the active path is train_critic_with_reward -> it uses the offline data
#       and the REWARD model; it does NOT use the planner.
# Launch:  mkdir -p /projects/bhpx/khu5/ODP/logs && sbatch run_odp_critic_only.sbatch
# ============================================================================
set -e

REPO="/projects/bhpx/khu5/ODP"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate odp

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

#train critic
cd "$REPO/Finetuning"
CUDA_VISIBLE_DEVICES=0 python train_critic_script.py
