#!/bin/bash
#SBATCH --account=bhpx-dtai-gh         # [VERIFY ON SERVER] sacctmgr show assoc user=$USER
#SBATCH --partition=ghx4               # [VERIFY ON SERVER] sinfo -s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1              # stages 1-4 are SINGLE-GPU code -> asking 1 GPU queues far faster
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03:00:00                # stages 2-4 only (reward 30k + kernel 5k + critic); pretrain SKIPPED
#SBATCH --job-name=odp_rkc
#SBATCH --output=/projects/bhpx/khu5/ODP/logs/%x-%j.log   # absolute -> always under ODP/logs regardless of submit dir (Slurm does NOT expand $VARS here)
                                       # NOTE: this dir MUST exist at submit time -- launch with:  mkdir -p /projects/bhpx/khu5/ODP/logs && sbatch run_odp.sbatch

# ============================================================================
# ODP 6-stage cube/single-play/task4 pipeline on NCSA DeltaAI (4x GH200, aarch64)
# Stages 1-4 & 6 = single-GPU; Stage 5 (finetune) = 4-GPU via accelerate.
# ============================================================================
set -euo pipefail

# ---- user-editable paths ---------------------------------------------------
REPO="/projects/bhpx/khu5/ODP"         # [VERIFY ON SERVER] confirm your repo is here on the shared project FS (tail -f reads this same shared inode)
CONDA_ENV="odp"

# ---- conda activation (non-interactive shell does NOT source ~/.bashrc) -----
# Source the conda hook by absolute path so `conda activate` is defined.
# [VERIFY ON SERVER] confirm conda base; repo evidence points at a user miniconda under $HOME.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  # last resort: a `conda` module (name unverified) OR conda already on PATH
  # module load miniforge3   # [VERIFY ON SERVER] uncomment + fix name if the two paths above are absent
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

# module load cuda           # [VERIFY ON SERVER] only if torch needs a system CUDA; the aarch64 wheels bundle their own
# module load cudnn          # [VERIFY ON SERVER] only if you hit "libcudnn* not found"; else rely on $CONDA_PREFIX libs

# ---- runtime env -----------------------------------------------------------
export MUJOCO_GL=egl               # MUST be set before Stage 1: every dataset __init__ builds a render_mode='rgb_array' env
                                   # [VERIFY ON SERVER] if EGL unavailable on the GH200 image:  export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1          # live tailing of the Python layer
ulimit -n 65535 || true           # raise FD limit (fork/DataLoader/AsyncVectorEnv)

# ---- thread/arena caps: cure for exit 134 SIGABRT (SYS_futex EINTR) on the 288-core Grace node ----
# Unbounded glibc malloc arenas + OpenMP/BLAS threads (default = node core count ~288) cause
# futex contention -> abort mid-run. Cap them to a sane value (independent of --cpus-per-task).
export MALLOC_ARENA_MAX=4          # the direct fix for glibc-arena futex storms on many-core NUMA nodes
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# cuDNN/cuBLAS shim: prefer the env's own bundled nvidia libs (aarch64-safe), guarded so a missing dir is a no-op.
NV="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
mkdir -p "$REPO/logs"   # belt-and-suspenders for the banner echoes; the Slurm --output dir must already exist (see launch cmd)

banner() { echo; echo "==== $* ===="; echo "     $(date '+%F %T')"; echo; }

# ---- run_stage: run one stage, capture its real exit code, fail loudly -----
run_stage() {
  local name="$1"; shift
  banner "STAGE START: $name"
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    banner "STAGE FAILED: $name (exit=$rc) -- aborting pipeline"
    exit "$rc"
  fi
  banner "STAGE OK: $name"
}

# ===========================================================================
# FAIL-FAST PRECHECKS (seconds, not hours) -- catch the two hard blockers early
# ===========================================================================
banner "PRECHECK: imports + GPUs + dataset cache"

# Prove the thread/arena caps are active in THIS run's log (so a recurrence isn't ambiguous).
echo "env caps: MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS | node cores=$(nproc) | affinity=$(python -c 'import os;print(len(os.sched_getaffinity(0)))')"

# 1) aarch64 torch + native deps import, and all 4 GPUs visible.
python -c "import torch, accelerate, gymnasium, gymnasium_robotics, minari, ogbench, mujoco, einops, scipy, sympy; \
assert torch.cuda.is_available(), 'CUDA not available -- aarch64 torch/module issue'; \
n=torch.cuda.device_count(); assert n>=1, f'expected >=1 GPU, saw {n}'; \
print('imports OK | torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', n, '|', torch.cuda.get_device_capability(0))" \
  || { echo "FATAL: import/GPU precheck failed"; exit 1; }

# 2) ogbench dataset cache present (compute nodes are air-gapped -> auto-download would HANG, not error).
#    ogbench stores datasets as .npz under ~/.ogbench/data; the singletask envs REUSE the base
#    'cube-single-play/noisy-v0' files, so match loosely (recursive) rather than by exact env id.
OGB_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench}"
for base in cube-single-play cube-single-noisy; do
  if ! find "$OGB_DIR" -type f -name "${base}*.npz" 2>/dev/null | grep -q .; then
    echo "FATAL: ogbench '${base}*' data not found under $OGB_DIR."
    echo "       Pre-download on the LOGIN node first (compute nodes have no network):"
    echo "       conda activate $CONDA_ENV && MUJOCO_GL=egl python -c \"import ogbench; [ogbench.make_env_and_datasets(d, render_mode='rgb_array') for d in ['cube-single-play-singletask-task4-v0','cube-single-noisy-singletask-task4-v0']]\""
    exit 1
  fi
done
echo "precheck OK (ogbench cube-single data present under $OGB_DIR)"

# ===========================================================================
# STAGES 1-6 (strictly sequential; set -e + run_stage abort on any failure)
# ===========================================================================

# ---- REUSE the existing planner: pretrain code is unchanged, so the planner from a
#      previous run is still valid. Stage 1 is intentionally SKIPPED here.
PLANNER="$REPO/Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_0.pt"
[ -f "$PLANNER" ] || { echo "FATAL: no existing planner at $PLANNER -- run run_odp_stage1to4.sbatch instead"; exit 1; }
echo "reusing planner: $PLANNER ($(date -r "$PLANNER" '+%F %T' 2>/dev/null || stat -c %y "$PLANNER"))"

# Stage 2 -- Reward MLP (single GPU, num_steps=30000). Run from Pretrain/.
run_stage "2 REWARD" \
  env CUDA_VISIBLE_DEVICES=0 bash -c 'cd "$0"/Pretrain && python -u train_reward_script.py' "$REPO"

# Stage 3 -- Kernel MoG ensemble x10 (single GPU, num_steps=5000). Run from Pretrain/.
run_stage "3 KERNEL" \
  env CUDA_VISIBLE_DEVICES=0 bash -c 'cd "$0"/Pretrain && python -u train_kernel_script.py' "$REPO"

# Stage 4 -- Critic (single GPU). Reads Stage-2 reward .pkl; self-chdirs to repo root at import.
run_stage "4 CRITIC" \
  env CUDA_VISIBLE_DEVICES=0 python -u Finetuning/train_critic_script.py

banner "DONE (stages 2-4 OK, planner reused) -- now submit run_odp_stage5to6.sbatch"
