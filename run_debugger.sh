#!/usr/bin/env bash
# =============================================================================
# run_debugger.sh -- guarantees the pipeline runs THIS repo's code, verified to be
# byte-identical to origin/Debugger. Aborts before touching a GPU otherwise.
#
# Fixes the two ways bash2.sh can silently run the wrong thing:
#   1. bash2.sh uses `cd Online-Diffusion-Planning/Pretrain` (relative + wrong repo
#      name) -> can cd into a DIFFERENT clone. Here every path is absolute.
#   2. bash2.sh has no `set -e` -> a failed cd cascades silently. Here it aborts.
#
# The python commands themselves are byte-identical to bash2.sh.
#
# Usage (from anywhere):   bash /projects/bhpx/khu5/ODP/run_debugger.sh
# =============================================================================
set -euo pipefail

# Absolute, resolved -- no ambiguity about which copy of the code runs.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO"

echo "================================================================"
echo " REPO (resolved) : $REPO"
echo "================================================================"

# ---- HARD GATE 1: .py must be byte-identical to origin/Debugger --------------
git fetch origin --quiet
DIFF="$(git diff --name-only origin/Debugger -- '*.py' || true)"
if [ -n "$DIFF" ]; then
  echo "ABORT: these .py differ from origin/Debugger:"
  echo "$DIFF" | sed 's/^/    /'
  echo "Fix with:  git fetch origin && git reset --hard origin/Debugger"
  exit 1
fi
echo "OK  .py identical to origin/Debugger @ $(git rev-parse --short origin/Debugger)"
echo "OK  branch=$(git branch --show-current)  HEAD=$(git rev-parse --short HEAD)"

# ---- HARD GATE 2: show the config that will actually run ---------------------
echo "--- live config (AST-resolved, comment blocks skipped) ---"
python3 - <<'PY'
import ast, glob, os
for f in ['Pretrain/pretrain_script4.py','Pretrain/train_reward_script.py',
          'Pretrain/train_kernel_script.py','Finetuning/train_critic_script.py',
          'Finetuning/train_critic_script2.py','Finetuning/finetune_script2.py',
          'Finetuning/Rollout2.py']:
    if not os.path.exists(f): print(f"    {f:34s} <missing>"); continue
    t = ast.parse(open(f).read()); got = []
    for n in ast.walk(t):
        if isinstance(n, ast.Assign):
            for x in n.targets:
                if isinstance(x, ast.Name) and x.id in (
                    'dataset','dataset_name','specific_dataset','env_name',
                    'specific_env','specific_train_dataset','task_id','checkpoint'):
                    got.append(f"{x.id}={ast.unparse(n.value)[:16]}")
    print(f"    {f:34s} {', '.join(dict.fromkeys(got))[:84]}")
PY
echo "----------------------------------------------------------------"

# ---- env ---------------------------------------------------------------------
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
LOGDIR="$REPO/logs"; mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

stage () { echo; echo "======== $1  ($(date '+%F %T')) ========"; echo "  cwd=$(pwd -P)"; }

# ---- stages: identical python commands to bash2.sh, absolute cwd -------------
cd "$REPO/Pretrain"

#pretrain planner
stage "1 PRETRAIN"
CUDA_VISIBLE_DEVICES=0 python pretrain_script4.py 2>&1 | tee "$LOGDIR/1_pretrain.log"

#train reward
stage "2 REWARD"
CUDA_VISIBLE_DEVICES=0 python train_reward_script.py 2>&1 | tee "$LOGDIR/2_reward.log"

#train kernel
stage "3 KERNEL"
CUDA_VISIBLE_DEVICES=0 python train_kernel_script.py 2>&1 | tee "$LOGDIR/3_kernel.log"

cd "$REPO/Finetuning"

#train critic
stage "4 CRITIC"
CUDA_VISIBLE_DEVICES=0 python train_critic_script.py 2>&1 | tee "$LOGDIR/4_critic.log"

#warm up the critic
stage "5 CRITIC WARMUP"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --multi_gpu --num_processes=4  train_critic_script2.py 2>&1 | tee "$LOGDIR/5_critic_warmup.log"

#finetune
stage "6 FINETUNE"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1000000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch --multi_gpu --num_processes=4   finetune_script2.py 2>&1 | tee output.txt "$LOGDIR/6_finetune.log"

#rollout
stage "7 ROLLOUT2"
CUDA_VISIBLE_DEVICES=0 python Rollout2.py 2>&1 | tee "$LOGDIR/7_rollout2.log"

echo; echo "======== ALL STAGES DONE  ($(date '+%F %T')) ========"
