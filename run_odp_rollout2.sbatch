#!/bin/bash
#SBATCH --account=bhpx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=odp_ro2
#SBATCH --output=/projects/bhpx/khu5/ODP/logs/%x-%j.log

# ============================================================================
# Rollout2.py only -- no reward/kernel/critic/finetune. Evaluates an EXISTING
# planner checkpoint (Rollout2's __main__ has `checkpoint = 90`).
#
# AS SHIPPED Rollout2.py runs the ONE-SHOT RENDER path and then exit()s at
# line 634, so it produces demo.mp4 + a single return value -- NOT a success
# rate. To get the 100-seed success rate instead, comment out the first
# rollout(...) call (lines 617-633) AND the `exit()` on line 634.
#
# Launch:  mkdir -p /projects/bhpx/khu5/ODP/logs && sbatch run_odp_rollout2.sbatch
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

export MUJOCO_GL=egl        # headless node: offscreen GL for env.render()
export PYTHONUNBUFFERED=1

cd "$REPO/Finetuning"

# --- the checkpoint Rollout2.py asks for must exist, else it dies deep in the run ---
CKPT_STEP=$(grep -oE '^[[:space:]]*checkpoint = [0-9]+' Rollout2.py | head -1 | grep -oE '[0-9]+')
PL="$REPO/Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_${CKPT_STEP}.pt"
echo "Rollout2 wants checkpoint=${CKPT_STEP} -> $PL"
[ -f "$PL" ] || { echo "FATAL: missing $PL"; echo "Available:"; ls -1 "$REPO"/Finetuning/Planners/cube/single-play/ | sort -V; exit 1; }

# --- render=True writes demo.mp4 via mediapy, which needs ffmpeg on PATH ---
if grep -qE '^[[:space:]]*render = True' Rollout2.py; then
  command -v ffmpeg >/dev/null \
    || echo "WARNING: render=True is active but ffmpeg is NOT on PATH -> the video write will fail at the very end (the rollout itself will already have run). Fix: conda install -c conda-forge ffmpeg"
fi

python Rollout2.py
