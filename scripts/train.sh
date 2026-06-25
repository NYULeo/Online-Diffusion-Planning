#!/usr/bin/env bash
# Train one full cube pipeline run (pretrain -> kernel -> reward -> critic -> finetune).
#
# Override any setting via environment variables; extra flags pass straight through to the runner.
#   VARIANT=single TASK=4 SEED=1 bash scripts/train.sh
#   VARIANT=double bash scripts/train.sh --no-wandb
#   STAGES=pretrain,kernel bash scripts/train.sh      # run a subset of stages
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

VARIANT="${VARIANT:-single}"   # single | double | triple | quadruple
TASK="${TASK:-4}"              # cube singletask id (1-5)
SEED="${SEED:-1}"
STAGES="${STAGES:-pretrain,kernel,reward,critic,finetune}"

echo "[train] variant=$VARIANT task=$TASK seed=$SEED stages=$STAGES extra=$*"
python run_cube_pipeline.py \
  --variant "$VARIANT" --task "$TASK" --seed "$SEED" --stages "$STAGES" "$@"
