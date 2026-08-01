#!/bin/bash
#SBATCH --account=bhpx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --job-name=odp_rkc
#SBATCH --output=/projects/bhpx/khu5/ODP/logs/%x-%j.log

# ============================================================================
# Teammate's runbook, steps 1-3 (reward -> kernel -> critic), verbatim.
# Launch:  mkdir -p /projects/bhpx/khu5/ODP/logs && sbatch run_odp_stage2to4.sbatch
# ============================================================================
set -e

REPO="/projects/bhpx/khu5/ODP"

# --- conda: a batch shell does not source ~/.bashrc, so the hook must be sourced ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate odp

export MUJOCO_GL=egl        # headless node: MuJoCo has no display
export PYTHONUNBUFFERED=1   # so `tail -f` shows output live

#train reward
cd "$REPO/Pretrain"
CUDA_VISIBLE_DEVICES=0 python train_reward_script.py

#train kernel
cd "$REPO/Pretrain"
CUDA_VISIBLE_DEVICES=0 python train_kernel_script.py

#train critic
cd "$REPO/Finetuning"
CUDA_VISIBLE_DEVICES=0 python train_critic_script.py
