#!/usr/bin/env bash
# Fast wiring check: run all five stages with tiny step counts (no real training).
# Use this first on a new machine to confirm deps / env / data download all work.
#   bash scripts/smoke.sh
#   VARIANT=single TASK=4 bash scripts/smoke.sh --no-wandb
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

VARIANT="${VARIANT:-single}"
TASK="${TASK:-4}"

echo "[smoke] variant=$VARIANT task=$TASK extra=$*"
python run_cube_pipeline.py --variant "$VARIANT" --task "$TASK" --smoke "$@"
