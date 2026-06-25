# ODP — JAX/Flax Port

## 1. What this is

This is a **PyTorch → JAX/Flax port of the ODP codebase, done FQL-style** (Flow-Q-Learning
conventions). Networks become `flax.linen` modules (`forward` → `__call__`, `nn.Module` →
`flax.linen.Module`, `setup()`/`@nn.compact` idioms); training moves to **`optax`** (`adamw` +
`cosine_decay_schedule` + `clip_by_global_norm`) driven by the small `JAX_PORT/jax_utils.py` plumbing
(`TrainState`, `ModuleDict`, `ensemblize`, `target_update`, `default_init`, `save_agent`/`restore_agent`).
In-place tensor writes become `x.at[...].set(...)`; autograd / input-gradients become
`jax.grad`/`jax.vjp`/`jax.jvp`/`jax.jacrev`; RNG becomes explicit (`set_seed` returns a
`jax.random.PRNGKey`, threaded through callers).

**Frozen-API principle.** Every public class name, function name, positional argument, and
hyperparameter/constant is preserved byte-for-byte. The **only** sanctioned signature changes are
trailing keyword-only `rng=`/`seed=` additions (JAX has no global RNG) plus a small set of explicitly
`# API-CHANGE`-flagged deltas where torch's stateful/in-place semantics are impossible in JAX. Even
pre-existing latent torch bugs were preserved verbatim, not "fixed."

> **Status:** all 31 core files are `ast.parse`-clean; 30/31 are free of executable torch. **Nothing has
> been runtime-executed** — see `PORT_REPORT.md` §8e and the "Known limitations" section below. Read
> `CONVERSION_GUIDE.md` for the full rule set and `PORT_REPORT.md` for the per-file conversion record.

## 2. Module map (by dependency tier)

**Tier 0 — shared plumbing**
- `JAX_PORT/jax_utils.py` — `TrainState`/`ModuleDict`, `ensemblize`, `target_update`, `default_init`,
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
`rng=set_seed(1)` explicitly (LOW issues in `PORT_REPORT.md` §8d).

## 4. Dependencies (none installed in this environment)

The port has **not** been run here — install before any execution:

```
pip install jax jaxlib flax optax distrax einops
pip install gym gymnasium minari ogbench
```

- `jaxlib` must match your accelerator (CPU / CUDA / TPU build).
- `distrax` is needed for `Rewards/nets.py` (Beta reward). Confirm `distrax.Beta(...).distribution.quantile`
  resolves at runtime — it backs the reward-CI path (no active caller exercises it today).
- `gym`/`gymnasium` + `minari`/`ogbench` back `Pretrain/Dataset.py`'s loaders (D4RL/OGBench).

## 5. Known limitations — NOT YET RUNTIME-VERIFIED

JAX/Flax/optax/distrax are not installed here, so the **only** gate applied was `ast.parse` plus
AST-level torch-residue scans. Linen `.init`/`.apply` tracing, optax wiring, attention/conv shape
round-trips, gradient numerics, and the checkpoint bridge are all **unverified**.

**Open HIGH issues to fix before the finetuning subsystem can run** (see `PORT_REPORT.md` §8d/§8e):
1. `Finetuning/utils.py` `rollout_parallel2` still calls `.to()/.load_state_dict()/.eval()` on a linen
   `DiT1d` — fix by mirroring `rollout_parallel3` (init → `from_state_dict` → `TrainState.create`).
2. `Finetuning/acc_adjoint_matching.py:623` `reward_model.eval()` on a plain container — delete the line.
3. Kernel-list mismatch: `Kernel_Backbone.compute_*` require python lists of `(model_def, params)` tuples,
   but `utils.train_kernel*` / `Rollout.load_kernel` pass lists of `TrainState` — append tuples instead.

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
