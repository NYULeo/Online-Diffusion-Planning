#!/bin/bash
#SBATCH --account=bhpx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=09:00:00
#SBATCH --job-name=odp_ft
#SBATCH --output=/projects/bhpx/khu5/ODP/logs/%x-%j.log

# ============================================================================
# Teammate's runbook, steps 4-5 (critic warmup -> finetune), verbatim.
# Requires run_odp_stage2to4.sbatch to have finished first.
# Launch:  mkdir -p /projects/bhpx/khu5/ODP/logs && sbatch run_odp_stage5to6.sbatch
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

cd "$REPO/Finetuning"

#warm up the critic
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --multi_gpu --num_processes=4  train_critic_script2.py

#finetune
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1000000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch --multi_gpu --num_processes=4   finetune_script2.py | tee output.txt

#rollout / eval -- not in the teammate's list; read-only, runs after training,
#                  cannot affect any trained weight. Prints "Checkpoint: N Success Rate: X".
CUDA_VISIBLE_DEVICES=0 python Rollout.py
