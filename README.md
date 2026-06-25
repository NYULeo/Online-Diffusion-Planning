# Online Diffusion Planning — JAX/Flax port

A JAX/[Flax](https://github.com/google/flax) port of Online Diffusion Planning, written in the style of
[FQL](https://github.com/seohongpark/fql). Diffusion planner (DiT/UNet) + transition kernel + reward +
critic are pretrained, then the planner is fine-tuned with adjoint matching. **No PyTorch** — every stage
trains from scratch in JAX and checkpoints in Flax format.

## Install

Python 3.10. Install the `jax` build for your hardware first, then the rest:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

pip install "jax[cuda12]"      # GPU  (or  pip install "jax[cpu]"  for CPU)
pip install -r requirements.txt

wandb login                    # optional, for online logging
```

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
