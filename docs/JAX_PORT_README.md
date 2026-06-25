# ODP — JAX/Flax Port

## 1. What this is

This is a **PyTorch → JAX/Flax port of the ODP codebase, done FQL-style** (Flow-Q-Learning
conventions). Networks become `flax.linen` modules (`forward` → `__call__`, `nn.Module` →
`flax.linen.Module`, `setup()`/`@nn.compact` idioms); training moves to **`optax`** (`adamw` +
`cosine_decay_schedule` + `clip_by_global_norm`) driven by the small `flax_utils.py` plumbing
(`TrainState`, `ModuleDict`, `ensemblize`, `target_update`, `default_init`, `save_agent`/`restore_agent`).
In-place tensor writes become `x.at[...].set(...)`; autograd / input-gradients become
`jax.grad`/`jax.vjp`/`jax.jvp`/`jax.jacrev`; RNG becomes explicit (`set_seed` returns a
`jax.random.PRNGKey`, threaded through callers).

**Frozen-API principle.** Every public class name, function name, positional argument, and
hyperparameter/constant is preserved byte-for-byte. The **only** sanctioned signature changes are
trailing keyword-only `rng=`/`seed=` additions (JAX has no global RNG) plus a small set of explicitly
`# API-CHANGE`-flagged deltas where torch's stateful/in-place semantics are impossible in JAX. Even
pre-existing latent torch bugs were preserved verbatim, not "fixed."

> **Status:** all 31 core files are `ast.parse`-clean and free of executable torch. The 3 previously-open
> cross-file HIGH issues are now **fixed** (see §5). **Nothing has been runtime-executed** — see "Known
> limitations" below. Read `docs/CONVERSION_GUIDE.md` for the full rule set and `docs/PORT_REPORT.md` for the
> per-file conversion record. Weights & Biases logging and a sequential cube-double training pipeline have
> been added (§4b, `run_cube_pipeline.py`, `plain `import wandb` (FQL-style)`).

## 2. Module map (by dependency tier)

**Tier 0 — shared plumbing**
- `flax_utils.py` — `TrainState`/`ModuleDict`, `ensemblize`, `target_update`, `default_init`,
  `save_agent`/`restore_agent`. Imported by everything that trains or holds params.

**Tier 1 — Pretrain spine** (`Pretrain/`)
- `Dataset.py` — dataset loaders & trajectory windowing (D4RL/OGBench/minari); numpy fields + fql-style `sample()`.
- `utils.py` — `set_seed` (→PRNGKey), `check_device` (→backend string), `compare_models_state_dict`, EMA helpers.
- `Planners/Backbone/BaseDiffusion.py` — abstract diffusion base (`map_noise` built in `setup()`).
- `Planners/Backbone/utils.py` — backbone blocks (Conv1d/attention/embeddings), `EMA`→`target_update`, `get_pretrained_planner` (checkpoint-bridge TODO).
- `Planners/Backbone/Dit.py` — DiT1d / DiT1Ref transformer denoisers (adaLN-Zero, MHA, `train` flag for dropout).
- `Planners/Backbone/UNet.py` — `TemporalUnet` denoiser (NCL↔NLC transpose at conv boundaries).
- `Planners/Backbone/Sampler.py` — diffusion samplers (`sample_euler_karras`, `sample_reverse_sde`, karras schedules); all take `*, rng=`.
- `Planners/Backbone/Trainer.py` — `SDETrainer` (TrainState + optax; EMA via pytree copy; flax-serialized `save`).
- `Critic/nets.py` — `Critic` / `CriticEnsemble` (linen; ensemble agg on axis=0).
- `Critic/train_critic.py` — critic training (TrainState, optax cosine, `optax.huber_loss`).
- `Rewards/nets.py` — reward nets (`SimpleReward`/`EnsembleReward`/`ScalarReward`; distrax.Beta; vmap(grad) per-sample grads).
- `Rewards/Reward_Backbone.py` — reward training/eval orchestration (`train_reward*`/`test_Model*`, save/load).
- `Transition_Kernel/Kernel_Net.py` — `RobustTransitionKernel` / `MoGTransitionKernel` (linen; clamps/return tuples byte-identical).
- `Transition_Kernel/Kernel_Backbone.py` — kernel training + `compute_log_density`/`compute_total_mahalanobis_score` (consume python lists of `(model_def, params)` tuples).

**Tier 2 — Finetuning** (`Finetuning/`)
- `utils.py` — shared finetuning utilities: datasets (`Critic*`), rollouts (`rollout_parallel{,2,3}`), critic trainers (`train_critic*`), kernel trainers, thresholds, `karras_beta_schedule`/`clip_actions`. Imported by every other finetuning file.
- `Rollout.py` — rollout / replan loop + kernel loading for evaluation.
- `traj_reward.py` / `traj_reward2.py` / `traj_reward3.py` — `TotalReward`/`TotalReward_Critic` holders that compose reward+kernel(+critic) subnets and expose the (reward, gradient) entrypoint (vjp input-grads). Three nominally-interchangeable variants used by different drivers.
- `adjoint_matching.py` — `AdjointMatchingFineTuner` + config (score nets → TrainState; jacrev/vjp; `params=`/`rng=`).
- `acc_adjoint_matching.py` — accelerate-style variant on a single device (jvp/jacrev/vjp; `_SingleDeviceAccelerator` shim).
- `Finetune_Backbone.py` — `OnlineFinetuner` orchestrator + `FinetuningConfig`/`Train_*_Config`/`AlphaSchedulerConfig`.

> Sibling files such as `Rollout2..5.py`, `Finetune_Backbone2.py`, `pretrain_script2..5.py`, etc. are
> experiment variants outside the 31-file core set; they share the core treatment but were not separately
> verified.

## 3. Entry points & how to run

All scripts assume the repo root is importable (they `sys.path.append(project_root)` + `chdir`, matching
the original torch layout). Run from the repo root, e.g. `python -m Pretrain.pretrain_script` or
`python Pretrain/pretrain_script.py`.

| Script | Purpose |
|---|---|
| `Pretrain/pretrain_script.py` | Train the diffusion planner (`SDETrainer`). |
| `Pretrain/train_critic_script.py` | Train the critic. |
| `Pretrain/train_kernel_script.py` | Train the transition kernel (MoG). |
| `Pretrain/train_reward_script.py` | Train the reward model / ensemble. |
| `Pretrain/test_kernel_script.py` / `test_reward_script.py` | Eval the kernel / reward model. |
| `Pretrain/Planner_Rollout.py` | Roll out a pretrained planner (live `__main__` calls are commented; needs the checkpoint bridge). |
| `Finetuning/finetune_script.py` | Adjoint-matching finetune of the planner (`OnlineFinetuner.finetune_planner`). |
| `Finetuning/train_critic_script.py` | Finetuning-side critic train/eval. |

Reproducibility note: several entry scripts compute `rng = set_seed(1)` but do not yet thread the key into
the trainer/callee (which then self-seeds with `PRNGKey(0)`). If `seed=1` matters, pass `seed=1` /
`rng=set_seed(1)` explicitly (LOW issues in `docs/PORT_REPORT.md` §8d).

## 4. Installation

The port targets **Python 3.10** (recommended for OGBench/D4RL compatibility) and the JAX stack — **no
PyTorch is needed to run the converted code**. Nothing is installed in this checkout, so set up a fresh
environment first.

### 4.1 Create an environment

```bash
# from the repo root (/Users/kaiwenhu/ODP)
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip wheel setuptools
```

(or with conda: `conda create -n odp python=3.10 -y && conda activate odp`)

### 4.2 Install the JAX runtime

Pick the `jax`/`jaxlib` build that matches your hardware — this is the one install that is
accelerator-specific:

```bash
# CPU (works everywhere; slow for full training):
pip install -U "jax[cpu]"

# NVIDIA GPU (CUDA 12):
pip install -U "jax[cuda12]"

# Apple Silicon (experimental Metal backend) — CPU is the safer default on macOS:
#   pip install -U jax-metal      # optional; otherwise use the CPU build above
```

Then the framework layer used by the converted code:

```bash
pip install -U flax optax distrax einops chex ml_collections
```

### 4.3 Install logging, data, and environment deps

```bash
# logging + utilities
pip install -U wandb tqdm numpy "numpy<2.0" matplotlib loguru

# environments / datasets (cube-double comes from OGBench)
pip install -U "gymnasium<1.0.0" ogbench minari h5py
```

`ogbench` provides the **cube** tasks (`cube-double-play-*`) used by the default pipeline. `gymnasium`'s
`AsyncVectorEnv` backs the parallel rollouts. `minari`/`h5py` back the other (D4RL/maze) loaders in
`Pretrain/Dataset.py` — only needed if you train those envs.

### 4.4 One-line install (CPU)

```bash
pip install -U "jax[cpu]" flax optax distrax einops chex ml_collections \
               wandb tqdm "numpy<2.0" matplotlib loguru "gymnasium<1.0.0" ogbench minari h5py
```

> Reference dependency lists from the original project live in `requirements/` (these still pin **torch**,
> which the JAX port does not require — install it only if you also want to run the original torch code or
> ingest legacy torch checkpoints; see §5 checkpoint bridge).

### 4.5 Weights & Biases

Metric logging goes through `plain `import wandb` (FQL-style)` (repo root), which is a **safe no-op unless a run is
active** — training behaves identically whether or not wandb is installed/enabled.

```bash
wandb login                      # once, for online logging
# or run without an account:
export WANDB_MODE=offline        # logs locally; sync later with `wandb sync`
export ODP_WANDB=0               # fully disable wandb (pure no-op)
```

### 4.6 Verify the install (no training)

```bash
python -c "import jax, flax, optax, distrax, ogbench; print('jax', jax.__version__, jax.devices())"
# exercise the whole pipeline wiring with tiny step counts:
python run_cube_pipeline.py --smoke
# cube single instead of double:
python run_cube_pipeline.py --variant single --task 4
```

## 4b. Quick start — cube-double training pipeline

`run_cube_pipeline.py` (repo root) runs all five training stages **sequentially** on the cube
*double* environment, each as its own wandb run:

`pretrain → kernel → reward → critic → finetune`

```bash
# full pipeline (online wandb):
python run_cube_pipeline.py

# a subset / resume mid-pipeline:
python run_cube_pipeline.py --stages kernel,reward,critic

# fast end-to-end wiring check (tiny steps):
python run_cube_pipeline.py --smoke
# cube single instead of double:
python run_cube_pipeline.py --variant single --task 4

# without wandb:
python run_cube_pipeline.py --no-wandb
```

The stages chain via checkpoints: pretrain/kernel/reward/critic each save at a known step, and the
finetune stage loads exactly those (`PRETRAIN_STEPS`/`KERNEL_STEPS`/`REWARD_STEPS`/`CRITIC_STEPS` at the
top of the script — edit them together if you change step counts). wandb metrics are namespaced per stage
(`pretrain/loss`, `kernel/avg_loss`, `reward/loss`, `critic/loss`, `finetune/reward`, …) and all five
runs share one wandb **group** (`cube-double-seed<seed>`) so you can compare them on one dashboard.

> The cube naming has two spellings on purpose: kernel/reward use `specific='double'` while
> planner/critic/finetune use `'double-play'` — both resolve to the same checkpoint stem. This is the
> original ODP convention; the runner handles it for you.

## 4c. Dependencies (summary)

The port has **not** been run in this checkout — install per §4 before any execution.

- **Runtime:** `jax`/`jaxlib` (accelerator-matched), `flax`, `optax`, `distrax`, `einops`, `chex`,
  `ml_collections`.
- **Logging/util:** `wandb`, `tqdm`, `numpy<2.0`, `matplotlib`, `loguru`.
- **Envs/data:** `gymnasium<1.0.0`, `ogbench` (cube), `minari`, `h5py`.
- `distrax.Beta(...).distribution.quantile` backs the reward-CI path (no active caller exercises it today
  — confirm it resolves at runtime if you enable that path).

## 5. Known limitations — NOT YET RUNTIME-VERIFIED

JAX/Flax/optax/distrax are not installed here, so the **only** gate applied was `ast.parse` plus
AST-level torch-residue scans. Linen `.init`/`.apply` tracing, optax wiring, attention/conv shape
round-trips, gradient numerics, and the checkpoint bridge are all **unverified**.

**Previously-open HIGH issues — now FIXED** (the finetuning subsystem is now internally consistent;
still requires a JAX env to runtime-verify):
1. ✅ `Finetuning/utils.py` `rollout_parallel2` — now builds the planner via `DiT1d.init` →
   `flax.serialization.from_state_dict` → `TrainState.create` (mirrors `rollout_parallel`), and threads
   `rng=` into every `sample_euler_karras` call (no more `.to()/.load_state_dict()/.eval()` on a linen module).
2. ✅ `Finetuning/acc_adjoint_matching.py` — the `reward_model.eval()` call (a no-op on the JAX container)
   was removed.
3. ✅ Kernel-list type consistency — every list fed to `Kernel_Backbone.compute_*` is now a python list of
   `(model_def, params)` tuples (`Rollout.load_kernel`, `traj_reward2/3` mog branches), matching the §11
   contract used by `utils.py`.

### Checkpoint-bridge TODO (torch state_dict → flax pytree remap)

Loading legacy torch `.pt`/`.pkl` checkpoints requires a key/layout remap (none implemented yet):

- **Dense:** `weight (out, in)` → `kernel (in, out)` = `weight.T`; `bias` → `bias`.
- **LayerNorm:** `weight` → `scale`; `bias` → `bias`.

This is needed wherever a pretrained net is ingested: `Planners/Backbone/utils.py :: get_pretrained_planner`
(currently raises `NotImplementedError`), `Planners/Backbone/Trainer.py :: save/selector`,
`Critic/train_critic.py` critic loaders, `Finetuning/traj_reward{,2,3}.py` subnet ingest,
`adjoint_matching.py`/`acc_adjoint_matching.py` planner ingest, `Planner_Rollout.py` + `Rollout.py` loaders,
and the new `Finetuning/utils.py` critic/planner/reward/kernel restores (`get_critic_model`,
`load_kernel_ensemble`, etc.). JAX-to-JAX (flax-serialized) save/restore is already self-consistent; only
legacy torch ingest needs the remap.

## 6. Why the directory layout was preserved

The physical directory structure was **intentionally kept identical** to the torch original (same paths,
same module names, same intra-repo imports). This keeps every `from Pretrain... import` / `from
Finetuning... import` and every public symbol working unchanged, so the port is a drop-in replacement
rather than a reorganization. The cost is that some pre-existing import quirks were preserved (now
normalized only where they were outright broken, e.g. bare `from utils import` → `from Finetuning.utils
import`); the benefit is that the frozen-API contract holds end-to-end and the diff stays reviewable.
