#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_NAME="${CONFIG_NAME:-cube_single}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${CONFIG_NAME}-$(date '+%Y%m%d-%H%M%S')}"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"
echo "W&B pipeline group: $WANDB_RUN_GROUP"

require_artifact() {
  local artifact="$1"
  if [[ ! -s "$artifact" ]]; then
    echo "ABORT: required artifact is missing or empty: $artifact" >&2
    exit 1
  fi
  echo "Artifact OK: $artifact"
}

CUDA_VISIBLE_DEVICES=0 python Pretrain/pretrain_script2.py \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_DIR/1_pretrain.log"
require_artifact "$REPO/Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_0.pt"

CUDA_VISIBLE_DEVICES=0 python Pretrain/train_reward_script2.py \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_DIR/2_reward.log"
require_artifact "$REPO/Finetuning/Rewards/cube/single/Models/Cube_Single_Task4_Reward_0.pkl"
require_artifact "$REPO/Finetuning/Rewards/cube/single/Stats/Cube_Single_Task4_Reward_stats_0.pkl"

CUDA_VISIBLE_DEVICES=1 python Pretrain/train_kernel_script2.py \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_DIR/3_kernel.log"
require_artifact "$REPO/Finetuning/Kernels/cube/single/Models/0/Cube_Single_Kernel_0.pkl"
require_artifact "$REPO/Finetuning/Kernels/cube/single/Stats/Cube_Single_Kernel_stats_0.pkl"

CUDA_VISIBLE_DEVICES=0 python Finetuning/train_critic_script.py \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_DIR/4_critic.log"
require_artifact "$REPO/Finetuning/Critics/cube/single-play/Models/Cube_SinglePlay_task4_Critic_0.pkl"
require_artifact "$REPO/Finetuning/Critics/cube/single-play/Stats/Cube_SinglePlay_task4_Q_stats_0.pkl"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 \
  Finetuning/train_critic_script2.py --config-name "$CONFIG_NAME" \
  2>&1 | tee "$LOG_DIR/5_critic_warmup.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1000000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch --multi_gpu --num_processes=4 \
  Finetuning/finetune_script.py --config-name "$CONFIG_NAME" \
  2>&1 | tee output.txt "$LOG_DIR/6_finetune.log"
require_artifact "$REPO/Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_60.pt"

CUDA_VISIBLE_DEVICES=0 python Finetuning/Rollout.py \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_DIR/7_rollout.log"
