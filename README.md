# Online Diffusion Planning — JAX/Flax port

A JAX/[Flax](https://github.com/google/flax) port of Online Diffusion Planning, written in the style of
[FQL](https://github.com/seohongpark/fql). Diffusion planner (DiT/UNet) + transition kernel + reward +
critic are pretrained, then the planner is fine-tuned with adjoint matching. **No PyTorch** — every stage
trains from scratch in JAX and checkpoints in Flax format.

## Install

Python 3.10, via conda. `requirements.txt` is one-shot (it pins the CUDA 12 GPU build of JAX), so on a
CUDA server this is all you need:

```bash
conda create -n odp python=3.10 -y
conda activate odp
pip install -r requirements.txt

wandb login                    # optional, for online logging
```

That's it — `pip install -r requirements.txt` installs JAX (GPU) + Flax/optax/distrax + the envs.
> Tip: if `pip install` warns about conflicts with `torch`/`gymnasium-robotics`/`pettingzoo`, your env
> isn't fresh (those are leftovers, not used by cube). Use a clean env — `bash scripts/setup_env.sh` does
> the `conda create` + install for you.


- **CPU only / macOS**, or a non-CUDA-12 server: edit the first line of `requirements.txt`
  (`jax[cuda12]>=0.4.26`) to `jax[cpu]>=0.4.26` (or `jax[cuda11_pip]>=0.4.26`) before installing.
- Verify the GPU is visible: `python -c "import jax; print(jax.devices())"`.

## Run

Shell entry points live in `scripts/` (they wrap `run_cube_pipeline.py`). Override settings with env vars;
extra flags pass through to the runner.

```bash
# 0) verify the whole pipeline wiring with tiny step counts (do this first on a new machine):
bash scripts/smoke.sh

# 1) one full run: pretrain -> kernel -> reward -> critic -> finetune
VARIANT=single TASK=4 SEED=1 bash scripts/train.sh

# 2) multi-round experiments (one run per seed / variant / task):
SEEDS="0 1 2" VARIANT=single bash scripts/sweep.sh

# subsets / no wandb:
STAGES=pretrain,kernel bash scripts/train.sh
bash scripts/train.sh --no-wandb
```

`VARIANT` ∈ {single, double, triple, quadruple}; `TASK` ∈ 1–5. cube data is fetched by `ogbench` on first
run. Metrics are logged to wandb (project `odp-cube`), namespaced per stage
(`pretrain/loss`, `kernel/avg_loss`, `reward/loss`, `critic/loss`, `finetune/reward`, …).

You can also call the runner directly: `python run_cube_pipeline.py --variant single --task 4 --seed 1`.

## Layout

```
flax_utils.py            # shared plumbing (TrainState, ModuleDict, MLP, target_update) — mirrors fql/utils/flax_utils.py
run_cube_pipeline.py     # the 5-stage orchestration engine (parameterized by --variant/--task/--seed)
scripts/                 # shell entry points: smoke.sh, train.sh, sweep.sh
requirements.txt         # all Python deps (JAX stack + envs)
Pretrain/                # planner (Planners/Backbone), reward (Rewards), kernel (Transition_Kernel),
                         # critic (Critic), Dataset.py, utils.py + per-stage *_script.py entry points
Finetuning/              # adjoint-matching finetuning: utils.py, traj_reward.py, adjoint_matching.py,
                         # acc_adjoint_matching.py, Finetune_Backbone.py, Rollout.py, finetune_script.py
docs/                    # CONVERSION_GUIDE.md, PORT_REPORT.md, JAX_PORT_README.md (the torch->JAX port record)
```

## Notes

- Stages chain via checkpoints (each saves at a step the finetuner then loads); the step constants are at
  the top of `run_cube_pipeline.py`.
- Loading the original authors' **PyTorch** checkpoints is not supported (a torch→flax key remap is still a
  TODO; see `docs/JAX_PORT_README.md`). Everything here trains from scratch instead.
