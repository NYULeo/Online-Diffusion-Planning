#!/usr/bin/env bash
# Multi-round experiment sweep: run the full cube pipeline once per seed (optionally per variant/task).
# Each (variant, task, seed) is a separate sequential run; results land in its own wandb group/run.
#
#   SEEDS="0 1 2" bash scripts/sweep.sh
#   VARIANT=single TASK=4 SEEDS="0 1 2 3 4" bash scripts/sweep.sh
#   VARIANTS="single double" SEEDS="0 1" bash scripts/sweep.sh --no-wandb
#   STAGES=pretrain SEEDS="0 1 2" bash scripts/sweep.sh        # sweep just one stage
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

VARIANTS="${VARIANTS:-${VARIANT:-single}}"   # space-separated list
TASKS="${TASKS:-${TASK:-4}}"                 # space-separated list
SEEDS="${SEEDS:-0 1 2}"                       # space-separated list
STAGES="${STAGES:-pretrain,kernel,reward,critic,finetune}"

n=0
for variant in $VARIANTS; do
  for task in $TASKS; do
    for seed in $SEEDS; do
      n=$((n+1))
      echo ""
      echo "########## run $n : variant=$variant task=$task seed=$seed ##########"
      python run_cube_pipeline.py \
        --variant "$variant" --task "$task" --seed "$seed" --stages "$STAGES" "$@"
    done
  done
done
echo ""
echo "[sweep] finished $n run(s)."
