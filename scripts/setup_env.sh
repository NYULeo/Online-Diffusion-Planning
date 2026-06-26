#!/usr/bin/env bash
# Create a clean conda env and install the JAX/cube deps in one shot.
# Avoids the "already-installed package" conflicts you get when reusing a polluted env
# (e.g. one that has torch / gymnasium-robotics / pettingzoo from the original torch project).
#
#   bash scripts/setup_env.sh                 # env name: odp
#   ENV=odp_gpu bash scripts/setup_env.sh     # custom env name
#   JAX=cpu bash scripts/setup_env.sh         # CPU build instead of CUDA 12
#
# After it finishes:  conda activate <env>  &&  bash scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

ENV="${ENV:-odp}"
PYVER="${PYVER:-3.10}"
JAX="${JAX:-cuda12}"      # cuda12 | cpu | cuda11_pip

echo "[setup] creating fresh conda env '$ENV' (python $PYVER), jax build: $JAX"
conda create -n "$ENV" "python=$PYVER" -y

# Run pip inside the new env without needing 'conda activate' in a non-interactive shell.
RUN=(conda run -n "$ENV" --no-capture-output)

"${RUN[@]}" python -m pip install -U pip wheel

if [ "$JAX" = "cuda12" ]; then
  "${RUN[@]}" pip install -r requirements.txt
else
  # Swap the jax build line on the fly (don't edit the file).
  tmp="$(mktemp)"
  sed "s/^jax\[cuda12\].*/jax[$JAX]>=0.4.26/" requirements.txt > "$tmp"
  "${RUN[@]}" pip install -r "$tmp"
  rm -f "$tmp"
fi

echo ""
echo "[setup] verifying import + devices:"
"${RUN[@]}" python -c "import jax, flax, optax, distrax, ogbench, gymnasium as g; print('jax', jax.__version__, '| gymnasium', g.__version__, '| devices', jax.devices())"
echo ""
echo "[setup] done. Next:"
echo "    conda activate $ENV"
echo "    bash scripts/smoke.sh"
