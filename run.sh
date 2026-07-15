#!/usr/bin/env bash
# =============================================================================
# run_cube_single_full.sh — FULL from-scratch pipeline, cube/single-play/task4.
# Runs the repo's scripts EXACTLY AS-IS (teammate's canonical code, post git reset).
# No checkpoints reused, no config patching, no verification of param values.
# Stages:  pretrain -> reward -> kernel -> critic -> finetune -> rollout.
#
#   nohup bash run_cube_single_full.sh > full.log 2>&1 &   ;   tail -f full.log
# =============================================================================
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
[ -d Pretrain ] && [ -d Finetuning ] || { echo "ERROR: run from the repo root (needs Pretrain/ and Finetuning/)."; exit 1; }

# torch finds CUDA-12 cuDNN/cublas (change 'odp' if your env name differs):
NV="$HOME/miniconda3/envs/odp/lib/python3.10/site-packages/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"

python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda unavailable'; print('torch', torch.__version__, 'CUDA OK')"

echo "==== [1/6] PRETRAIN (pretrain_script4.py) — from scratch, LONG ===="
cd "$ROOT/Pretrain" && python -u pretrain_script4.py; cd "$ROOT"

echo "==== [2/6] REWARD   (train_reward_script.py) ===="
cd "$ROOT/Pretrain" && python -u train_reward_script.py; cd "$ROOT"

echo "==== [3/6] KERNEL   (train_kernel_script.py) ===="
cd "$ROOT/Pretrain" && python -u train_kernel_script.py; cd "$ROOT"

echo "==== [4/6] CRITIC   (Finetuning/train_critic_script.py) ===="
cd "$ROOT" && python -u Finetuning/train_critic_script.py
CRITIC="Finetuning/Critics/cube/single-play/Models/Cube_SinglePlay_task4_Critic_0.pkl"
[ -f "$CRITIC" ] || { echo "ERROR: critic stage produced no checkpoint ($CRITIC). Check the active training call in Finetuning/train_critic_script.py."; exit 1; }

echo "==== [5/6] FINETUNE (finetune_script2.py) ===="
cd "$ROOT" && python -u Finetuning/finetune_script2.py

echo "==== [6/6] ROLLOUT  (Rollout.py) — prints 'Checkpoint: N  Success Rate: X' ===="
cd "$ROOT" && python -u Finetuning/Rollout.py
echo "==== DONE ===="
