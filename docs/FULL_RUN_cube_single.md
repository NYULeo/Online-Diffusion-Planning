# Full run (cube-single) with the ORIGINAL hyperparameters — for JAX-vs-torch comparison

## The command

Run all five stages on **cube / single-play, task 4** with the original hyperparameters:

```bash
conda activate odp                 # the env from README §Install
cd ~/ODP-jax                       # your repo checkout
VARIANT=single TASK=4 SEED=1 bash scripts/train.sh
```

This is the full (non-`--smoke`) path. It runs, in order:
`pretrain → kernel → reward → critic → finetune`, logging to wandb project `odp-cube`,
group `cube-single-seed1`.

> One run = one process. To go stage-by-stage (e.g. inspect each checkpoint, or parallelize across
> machines), use `--stages`:
> ```bash
> VARIANT=single TASK=4 bash scripts/train.sh --stages pretrain
> VARIANT=single TASK=4 bash scripts/train.sh --stages kernel,reward,critic
> VARIANT=single TASK=4 bash scripts/train.sh --stages finetune
> ```

## Hyperparameters used (full run) and where they come from

These are taken verbatim from the **original torch entry scripts on `main`** and applied to cube-single.

| Stage | Hyperparameters | Original source (`main`) |
|---|---|---|
| **pretrain** | num_steps=1,000,000, batch_size=128, lr=2e-4, horizon=32, stride=1, backbone=transformer | `Pretrain/pretrain_script.py` |
| **kernel** (MoG) | num_steps=5,000, save_freq=1,000, batch_size=512, lr=1e-4, ensemble_size=10, num_modes=10, num_hidden_layers=4, hidden_dim=514, λ_reg=1e-3, noise_floor=5e-4 | `Pretrain/train_kernel_script.py` |
| **reward** | num_steps=100,000, hidden_layers=4, hidden_dim=512, batch_size=256, lr=1e-4, min_lr=5e-6, sigma=4.0, target_reward=300.0 | `Pretrain/train_reward_script.py` (cube block) |
| **critic** | num_steps=50,000, hidden_layers=5, hidden_dim=512, batch_size=256, lr=1e-5, min_lr=1e-6, tau=0.005, gamma=0.99, horizon=32, sigma=8.0, target_reward=50.0 | `Pretrain/train_critic_script.py` |
| **finetune** | finetune_steps=1,000,000, finetune_rounds=10, finetune_batch_size=12, finetune_batch_per_sample=3, finetune_lr=2e-4, diffusion_steps=30, eta=0.8, num_karras=2, λ=0.01, rollout_length=1000, rollout_num_envs=1, beta=1.0, min_log_prob=150.0 | `Finetuning/finetune_script.py` + `FinetuningConfig`/`Acc_AdjointMatchingConfig` defaults |

## ⚠️ Read this before comparing — the original repo has NO unified cube-single config

A clean apples-to-apples comparison is **not** something the original torch repo directly provides, for
three reasons. You should decide how to handle each:

1. **Each torch entry script targets a DIFFERENT cube variant / task.** On `main`:
   - `pretrain_script.py` → cube **triple-play, task 5**
   - `train_kernel_script.py` → cube **double**
   - `train_reward_script.py` → cube **double, task 4**
   - `train_critic_script.py` → cube **single-play, task 4**
   - `finetune_script.py` → **kitchen/partial** (not cube at all)

   So there is no torch run of "all 5 stages on cube-single". This runner applies each stage's
   **hyperparameter values** to a single consistent env (cube-single, task 4). To compare against torch,
   run the **same** torch stages on cube-single too (you'll have to edit the torch scripts' env to match).

2. **The finetune `critic` is OFF by default** (`critic=False`) and the per-iteration kernel re-train is
   commented out in the original — so the finetune updates **planner + kernel + reward** (not critic).
   See `docs/TRAINING_PIPELINE.md`. Both this port and torch share this default, so it's consistent.

3. **A couple of things were NOT verifiable statically and matter for numeric parity:**
   - **Parameter init is fql-style, not torch-identical** (`default_init` = variance_scaling vs torch
     kaiming). With the same seed, JAX and torch will NOT have bit-identical initial weights, so per-step
     losses won't match exactly — compare **learning curves / final metrics (success rate, normalized
     score)**, not step-0 loss.
   - **Pretrained torch checkpoints can't be loaded** (the torch→flax checkpoint bridge is a TODO). This
     run trains everything from scratch in JAX, which is the right setup for a from-scratch JAX-vs-torch
     comparison anyway.

### Recommended comparison protocol

- Compare **end-of-stage metrics**, not raw losses: planner validation loss curve; reward/critic eval
  loss; finetune **rollout score + success rate** per round (logged to wandb as `evaluation/*` /
  `finetune/*`).
- Use the **same env (cube-single, task 4), same seeds**, same step counts on both sides.
- Expect the JAX run to be the more meaningful reference for "does the ported algorithm learn the same
  thing", since exact numeric equivalence is precluded by the init difference above.

## Runtime note

The full finetune is heavy: 10 rounds, each = 100k AM steps (eager) + a 1000-step env rollout + kernel
& reward re-training. On the autotuning-impaired GPU this is long. `scripts/sweep.sh` can run multiple
seeds; `XLA_FLAGS=--xla_gpu_autotune_level=0` is already set by default in `run_cube_pipeline.py`.
