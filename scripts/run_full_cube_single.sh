#!/usr/bin/env bash
# Launch the FULL cube-single (task 4) pipeline in the background and log to a file, so it keeps running
# after you disconnect. Uses the VERIFIED config (the one that ran end-to-end in smoke), with the
# original per-stage hyperparameters (see docs/FULL_RUN_cube_single.md).
#
#   bash scripts/run_full_cube_single.sh
#   SEED=0 bash scripts/run_full_cube_single.sh
#   STAGES=pretrain bash scripts/run_full_cube_single.sh      # just one stage
#
# Watch progress:   tail -f logs/full_cube_single_*.log
# Stop:             kill the PID printed at the end (or: pkill -f run_cube_pipeline.py)
#
# NOTE: this is the verified critic=False finetune path. Your teammate's exact cube-single result uses a
# DIFFERENT finetune config (critic=True, offline=True, eta=0.0, ...; see docs/cube_single_combination.md).
# Confirm the reward-pretrain source + that config with your collaborator before switching to it.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SEED="${SEED:-1}"
STAGES="${STAGES:-pretrain,kernel,reward,critic,finetune}"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)
LOG="logs/full_cube_single_seed${SEED}_${TS}.log"

echo "[run_full] launching cube single-play task4, seed=$SEED, stages=$STAGES"
echo "[run_full] logging to $LOG"
nohup python -u run_cube_pipeline.py \
    --variant single --task 4 --seed "$SEED" --stages "$STAGES" "$@" \
    > "$LOG" 2>&1 &
PID=$!
echo "[run_full] started PID $PID"
echo "[run_full] watch:  tail -f $LOG"
echo "[run_full] stop :  kill $PID"
