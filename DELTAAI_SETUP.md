# Running the ODP Cube Pipeline on NCSA DeltaAI (4× GH200, aarch64)

Copy-pasteable end-to-end guide for the 6-stage ODP pipeline (`cube / single-play / task4`) on DeltaAI GH200.
Synthesized from a static audit of `~/ODP` (20 readers). Items needing server-side confirmation are marked **[VERIFY ON SERVER]**.

Authoritative driver = `~/ODP/run_pipeline.sh`. The repo's `README.md` / `WORKFLOW_README.md` / `INSTALLATION_SUMMARY.md` are **stale** — trust `run_pipeline.sh` + the script source, not the markdown docs.

---

## 0. Key gotchas — read first

1. **aarch64 torch wheels = the #1 install blocker.** `requirements.txt` pins `torch==2.6.0 --index-url .../cu124`, which is **x86_64-only**. A verbatim `pip install -r requirements.txt` on the Grace (ARM) CPU fails. Install aarch64 CUDA torch separately (NGC container or a cu126/cu128 aarch64 index) — §2c. **[VERIFY ON SERVER]**
2. **Pretrain/reward/kernel/critic + rollout are single-GPU by design; only finetune is multi-GPU.** Stages 1–4 and 6 have zero distributed code (`torch.device('cuda')` = GPU 0). Only **Stage 5 (finetune)** uses `accelerate`, and only if launched with `accelerate launch --num_processes 4`. Plain `python` → uses **1 of 4 GPUs**.
3. **`MUJOCO_GL` must be set.** The cube env is created with `render_mode="rgb_array"` at dataset load, so a missing `MUJOCO_GL` crashes at Stage 1 even with `render=False`. `run_pipeline.sh` sets `MUJOCO_GL=egl`; export it yourself if running stages by hand. Fallback `osmesa`. **[VERIFY ON SERVER]**
4. **ogbench datasets must be pre-downloaded** on a login node (compute nodes are usually air-gapped) — §3.
5. **`accelerate launch` is NOT in the repo scripts.** `run_pipeline.sh` runs finetune as plain `python` (1 GPU). To use 4 GPUs, override it — §5 Stage 5.
6. **Don't copy `bash.txt`'s env vars blindly.** It has `CUDA_VISIBLE_DEVICES=0,2,3,4` (invalid), gloo backend, `NCCL_*_DISABLE`, x86_64 `PYGLFW_LIBRARY` — all workarounds for a different broken box. On GH200/NVLink use NCCL + `CUDA_VISIBLE_DEVICES=0,1,2,3`.
7. **One real code fix, ALREADY APPLIED for you:** `Finetuning/Rollout.py` — the active `rollout(...)` call (line ~873) was missing the required `num_layers` arg (added to the signature in the latest commit `e197322`) → would `TypeError`. Fixed to `num_layers=2` (matches `backbone_layers=2` in pretrain/finetune).
   - NOTE: an earlier draft also flagged a `train_reward_script.py` chdir bug — that was a **false alarm**. `Reward_Backbone.py` does `os.chdir(project_root)` at module top (lines 4–5), so reward outputs land correctly in `./Finetuning/Rewards/...`. Do NOT change it.
8. **Use root `requirements.txt`** — NOT `requirements/requirements.txt` / `requirements_playground.txt` / `requirements_macos*.txt` (different, broader repo; conflicting pins; none of their extra packages are imported by Pretrain/Finetuning).
9. **Python must be 3.10** (`.python-version` = `3.10.12`; `jax-jumpy` requires `<3.11`; `run_pipeline.sh` hardcodes a `python3.10` path substring).
10. **`num_workers` explosion.** `Finetune_Backbone2.py` sets `num_workers = os.cpu_count()//2` per process. On a ~72-core Grace × 4 ranks that's ~144 DataLoader workers + up to 8 MuJoCo subprocs/rank. Bound `--cpus-per-task`, `ulimit -n 65535`, and lower `num_workers`/`rollout_num_envs` if it thrashes.

---

## 1. Get the node (Slurm) — single node, 4× GH200

> **[VERIFY ON SERVER]** account/partition/flags below come from prior GH200 usage on the *same* cluster (a different project). Confirm `--account`, partition names, and GPU-request syntax via `module avail` / `sinfo` / the DeltaAI user guide. Prior facts: account form `<code>-dtai-gh`; partitions `ghx4` (batch) & `ghx4-interactive` (`srun --pty`); project space `/projects/<code>/<user>`; login `dtai-login.delta.ncsa.illinois.edu`.

### Interactive (setup / smoke / short runs)
```bash
srun --account=<CODE>-dtai-gh --partition=ghx4-interactive \
     --nodes=1 --gpus-per-node=4 --cpus-per-task=64 --time=04:00:00 --pty bash
```
> May need `--gres=gpu:4` instead of/with `--gpus-per-node=4`. **[VERIFY]**

### Batch (full pipeline) — `sbatch run_odp.sbatch`
```bash
#!/bin/bash
#SBATCH --account=<CODE>-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=odp
#SBATCH --output=odp-%j.out

module purge
module load cuda            # [VERIFY] exact name
module load miniforge3      # [VERIFY] anaconda3 / miniforge3 / miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate odp
export MUJOCO_GL=egl
ulimit -n 65535
cd /projects/<CODE>/<user>/ODP
bash run_pipeline.sh        # NOTE: Stage 5 runs single-GPU as written — see §5
```
Store repo + checkpoints + datasets under `/projects/<CODE>/<user>/`, not `$HOME` (quota). Monitor: `squeue --me`, `tail -f odp-<jobid>.out`.

---

## 2. Modules + conda env (Python 3.10, aarch64-safe torch)

Create the env **natively on DeltaAI** — never copy an x86_64 env/wheels.

```bash
module purge
module avail cuda ; module avail conda anaconda miniforge   # find real names
module load cuda           # [VERIFY]
module load miniforge3     # [VERIFY]
conda create -n odp python=3.10.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate odp
```

### 2c. aarch64 CUDA PyTorch — the riskiest step (do NOT `pip install -r requirements.txt`)
- **Option A (recommended):** NVIDIA NGC PyTorch container (`nvcr.io/nvidia/pytorch:YY.MM-py3`, an early-2025 tag shipping torch ~2.6 + CUDA 12.x for aarch64/Hopper sm_90) via DeltaAI's container runtime. **[VERIFY tag→torch 2.6 + GH200]**
- **Option B:** aarch64 CUDA index:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
```
> **[VERIFY]** that `torch==2.6.0` cp310 `linux_aarch64` wheels exist at cu126; else use the container or a newer torch (the code only uses long-stable APIs: `torch.autograd.functional.jvp`, `torch.autograd.grad(create_graph=True)`, bf16 autocast, explicit `weights_only`).

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_capability(0))"
# expect: 2.6.0  12.x  True  4   and  (9, 0)  for Hopper
```

### 2d. Install the rest (torch lines omitted)
```bash
pip install "numpy<2.0" accelerate einops \
    "gymnasium>=1.2.0" "gymnasium-robotics>=1.4.0" minari ogbench \
    "h5py>=3.0.0" "jax-jumpy==1.0.0" \
    matplotlib mediapy scipy scikit-learn sympy seaborn imageio tqdm loguru
```
- `accelerate>=0.21` is required (code uses `gather_for_metrics(use_gather_object=...)`). `mixed_precision='bf16'` is hardcoded in `Accelerator(...)`, overriding any `accelerate config` precision.
- **Riskiest aarch64 packages** (may need source build; **[VERIFY]**): `ogbench`, `minari`, `gymnasium-robotics`, `jax-jumpy`, + native MuJoCo pulled transitively. Sci stack (numpy/scipy/matplotlib/sklearn/sympy/seaborn) has good conda-forge aarch64 builds — `conda install -c conda-forge <pkg>` if a pip build fails.
- `torchvision/torchaudio/h5py/jax-jumpy` are pinned but never directly imported (companion/transitive) — keep them.

### 2e. Smoke-import (on a GPU allocation)
```bash
MUJOCO_GL=egl python -c "import torch, accelerate, ogbench, gymnasium, minari, einops, scipy, sympy, seaborn, mediapy, imageio, loguru, tqdm; print('imports OK', torch.cuda.is_available())"
```

### 2f. cuDNN/cuBLAS `LD_LIBRARY_PATH` shim
`run_pipeline.sh` line 5 prepends `$HOME/miniconda3/envs/odp/.../nvidia/{cudnn,cublas}/lib`, guarded by `[ -d "$NV" ]`. On DeltaAI that hardcoded path won't exist → guard no-ops → possible "shared library not found". Fix: edit the `NV=` path to use `$CONDA_PREFIX`, or `module load` cudnn. Check: `find "$CONDA_PREFIX" -iname 'libcudnn*'`.

---

## 3. Pre-download ogbench datasets (login node, network up)

Datasets for `cube/single-play/task4`:
- `cube-single-play-singletask-task4-v0` (main)
- `cube-single-noisy-singletask-task4-v0` (reward/kernel stages)

On the **login node** (env active):
```bash
python -c "import ogbench; [ogbench.make_env_and_datasets(d, render_mode='rgb_array') for d in ['cube-single-play-singletask-task4-v0','cube-single-noisy-singletask-task4-v0']]"
du -sh ~/.ogbench   # confirm cache location/size  [VERIFY]
```

---

## 4. Place teammate checkpoints

Repo ships no checkpoints. `run_pipeline.sh` per-stage toggle `1`=retrain, `0`=copy from `$CKPT` (default `$ROOT/checkpoints`). Layout → consumed path:

| Source (`$CKPT/…`) | Copied to |
|---|---|
| `Planner/Model/Cube_SinglePlay_task4_Planner_0.pt` | `Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_0.pt` |
| `Planner/stats/Cube_SinglePlay_task4_Planner_stats.pkl` | `Pretrain/Planners/cube/single-play/Stats/…` |
| `Reward/Model/Cube_Single_Task4_Reward_0.pkl` | `Finetuning/Rewards/cube/single/Models/…` |
| `Reward/stats/Cube_Single_Task4_Reward_stats_0.pkl` | `Finetuning/Rewards/cube/single/Stats/…` |
| `Kernel/Model/0/Cube_Single_Kernel_*.pkl` (**all 10**) | `Finetuning/Kernels/cube/single/Models/0/` |
| `Kernel/stats/Cube_Single_Kernel_stats_0.pkl` | `Finetuning/Kernels/cube/single/Stats/…` |
| `Critic/Model/Cube_SinglePlay_task4_Critic_0.pkl` | `Finetuning/Critics/cube/single-play/Models/…` |
| `Critic/stats/Cube_SinglePlay_task4_Critic_stats_0.pkl` | `Finetuning/Critics/cube/single-play/Stats/…` |

**Directory asymmetry:** Planner & Critic → `cube/single-play/…`; Reward & Kernel → `cube/single/…` (no `-play`). Kernel needs **all 10** ensemble members.

---

## 5. Run the 6 stages

Assume: `conda activate odp`, `cd .../ODP`, `export MUJOCO_GL=egl`. (The Rollout `num_layers` fix is already applied.)

**Stage 1 — Pretrain planner (single GPU):** DiT planner, `num_steps=1_000_000`. First run downloads the dataset unless pre-cached.
```bash
CUDA_VISIBLE_DEVICES=0 python -u Pretrain/pretrain_script4.py
```
→ `Finetuning/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_0.pt`

**Stage 2 — Reward (single GPU):** reward MLP, `num_steps=30000`.
```bash
CUDA_VISIBLE_DEVICES=0 python -u Pretrain/train_reward_script.py
```
→ `Finetuning/Rewards/cube/single/Models/Cube_Single_Task4_Reward_0.pkl`

**Stage 3 — Kernel (single GPU):** MoG ensemble ×10, `num_steps=5000`.
```bash
cd Pretrain && CUDA_VISIBLE_DEVICES=0 python -u train_kernel_script.py; cd ..
```

**Stage 4 — Critic (single GPU):** `Finetuning/train_critic_script.py` (reads Stage-2 reward).
```bash
CUDA_VISIBLE_DEVICES=0 python -u Finetuning/train_critic_script.py
```

**Stage 5 — Finetune (adjoint matching) — 4 GPUs via accelerate.** The ONLY multi-GPU stage; override the plain-`python` in `run_pipeline.sh`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
    Finetuning/finetune_script2.py
```
(equivalent: `torchrun --standalone --nnodes 1 --nproc_per_node 4 Finetuning/finetune_script2.py` — pick ONE launcher, don't also `srun`-spawn)
- Prefer default NCCL over NVLink; don't set gloo/`NCCL_*_DISABLE` unless NCCL actually hangs. **[VERIFY]**
- GPU memory is a non-issue (small DiT + MLPs, `diffusion_steps=10`) on 96GB.
- CPU pressure: each rank spawns up to 8 MuJoCo `AsyncVectorEnv` subprocs + `num_workers=cpu//2` → lower `rollout_num_envs`/`num_rollout_processes`/`num_workers` if it thrashes.

**Stage 6 — Rollout/eval (single GPU):**
```bash
CUDA_VISIBLE_DEVICES=0 python -u Finetuning/Rollout.py
```
Prints `Checkpoint: N Success Rate: X`. `render=False` (no ffmpeg needed).

**Or the driver:** after staging ckpts/data, `bash run_pipeline.sh` runs 1–6 — but Stage 5 single-GPU. For 4-GPU finetune, edit its line 52 to the `accelerate launch` form above.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| pip can't find torch / wrong-arch | cu124 index is x86_64-only, node is aarch64 | Install aarch64 torch (NGC / cu126), then rest without torch lines (§2c–d) |
| OpenGL/EGL crash at Stage 1 | `MUJOCO_GL` unset; env made with `rgb_array` | `export MUJOCO_GL=egl` (try `osmesa` if EGL absent) |
| Job hangs downloading data | ogbench auto-download on air-gapped node | Pre-download on login node (§3) |
| Finetune uses only 1/4 GPUs | plain `python` → `num_processes=1` | `accelerate launch --multi_gpu --num_processes 4` (§5) |
| `TypeError: rollout() missing … 'num_layers'` | latest commit added required arg; call omitted it | **Already fixed** (`num_layers=2`) |
| cuDNN/cuBLAS "shared library not found" | `LD_LIBRARY_PATH` shim points at nonexistent path | Edit `NV=` to `$CONDA_PREFIX/...` or `module load cudnn` |
| "Too many open files" / fork storm | `num_workers=cpu//2` ×4 + MuJoCo subprocs | `ulimit -n 65535`; lower workers/envs; bound `--cpus-per-task` |
| `FileNotFoundError` reward/kernel at finetune | placed under `single-play` instead of `single` | Reward/Kernel → `cube/single/`; Planner/Critic → `cube/single-play/` (§4) |
| `torch.cuda.is_available()` False on GPU node | aarch64 torch w/o CUDA, or cuda module not loaded | `module load cuda`; reinstall aarch64 CUDA torch; check `get_device_capability(0)==(9,0)` |
| jax-jumpy / gymnasium conflict | wrong requirements file or Python≠3.10 | Root `requirements.txt` only; `python=3.10.12` |

**Bottom line:** two hard blockers = aarch64 torch install (§2c) + pre-staging ogbench data (§3). One code fix already applied (Rollout `num_layers=2`). Stages 1–4 & 6 are single-GPU (`CUDA_VISIBLE_DEVICES=0 python …`); only Stage 5 uses `accelerate launch --num_processes 4`.
