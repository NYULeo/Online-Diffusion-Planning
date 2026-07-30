#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
NV="$HOME/miniconda3/envs/${CONDA_ENV:-odp}/lib/python3.10/site-packages/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"

# ===== 每个模型: 1=重新训练  0=用 checkpoints2 里的现成 ckpt =====
TRAIN_PLANNER=0
TRAIN_REWARD=1
TRAIN_KERNEL=1
TRAIN_CRITIC=1
CKPT="${CKPT:-$ROOT/checkpoints}"       # 现成 ckpt 文件夹 (可用 CKPT=/路径 覆盖)

# 1. PLANNER
if [ "$TRAIN_PLANNER" = 1 ]; then
  cd "$ROOT/Pretrain" && python -u pretrain_script4.py; cd "$ROOT"
else
  mkdir -p Finetuning/Planners/cube/single-play Pretrain/Planners/cube/single-play/Stats
  cp "$CKPT/Planner/Model/Cube_SinglePlay_task4_Planner_0.pt"      Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_0.pt
  cp "$CKPT/Planner/stats/Cube_SinglePlay_task4_Planner_stats.pkl" Pretrain/Planners/cube/single-play/Stats/Cube_SinglePlay_task4_Planner_stats.pkl
fi

# 2. REWARD
if [ "$TRAIN_REWARD" = 1 ]; then
  cd "$ROOT/Pretrain" && python -u train_reward_script.py; cd "$ROOT"
else
  mkdir -p Finetuning/Rewards/cube/single/Models Finetuning/Rewards/cube/single/Stats
  cp "$CKPT/Reward/Model/Cube_Single_Task4_Reward_0.pkl"       Finetuning/Rewards/cube/single/Models/Cube_Single_Task4_Reward_0.pkl
  cp "$CKPT/Reward/stats/Cube_Single_Task4_Reward_stats_0.pkl" Finetuning/Rewards/cube/single/Stats/Cube_Single_Task4_Reward_stats_0.pkl
fi

# 3. KERNEL
if [ "$TRAIN_KERNEL" = 1 ]; then
  cd "$ROOT/Pretrain" && python -u train_kernel_script.py; cd "$ROOT"
else
  mkdir -p Finetuning/Kernels/cube/single/Models/0 Finetuning/Kernels/cube/single/Stats
  cp "$CKPT"/Kernel/Model/0/Cube_Single_Kernel_*.pkl          Finetuning/Kernels/cube/single/Models/0/
  cp "$CKPT/Kernel/stats/Cube_Single_Kernel_stats_0.pkl"      Finetuning/Kernels/cube/single/Stats/Cube_Single_Kernel_stats_0.pkl
fi

# 4. CRITIC
if [ "$TRAIN_CRITIC" = 1 ]; then
  cd "$ROOT" && python -u Finetuning/train_critic_script.py
else
  mkdir -p Finetuning/Critics/cube/single-play/Models Finetuning/Critics/cube/single-play/Stats
  cp "$CKPT/Critic/Model/Cube_SinglePlay_task4_Critic_0.pkl"       Finetuning/Critics/cube/single-play/Models/Cube_SinglePlay_task4_Critic_0.pkl
  cp "$CKPT/Critic/stats/Cube_SinglePlay_task4_Critic_stats_0.pkl" Finetuning/Critics/cube/single-play/Stats/Cube_SinglePlay_task4_Critic_stats_0.pkl
fi

# 5. FINETUNE + 6. ROLLOUT (总是跑)
cd "$ROOT" && python -u Finetuning/finetune_script2.py
cd "$ROOT" && python -u Finetuning/Rollout.py
