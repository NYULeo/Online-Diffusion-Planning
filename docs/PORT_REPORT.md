# ODP torch → JAX/Flax Port Report

## 1. Overview

This port converts the ODP codebase from PyTorch to JAX/Flax in an FQL (Flow-Q-Learning) style.
The framework target is **Flax `linen`** for all networks (`forward` → `__call__`, `nn.Module` →
`flax.linen.Module` with dataclass attributes, `@nn.compact`/`setup()` idioms) and the
`flax_utils.py` plumbing (`TrainState` / `ModuleDict`, `ensemblize`, `target_update`,
`default_init`, `save_agent`/`restore_agent`) for training and parameter management. Optimizers move
to `optax` (`adamw` + `cosine_decay_schedule` + `clip_by_global_norm`), in-place tensor writes become
`.at[...].set(...)`, autograd/input-gradients become `jax.grad`/`jax.vjp`/`jax.jvp`/`jax.jacrev`, and
RNG becomes explicit (`set_seed` returns a `jax.random.PRNGKey`, threaded through callers). The guiding
rule throughout is the **frozen-API principle**: every public class name, function name, positional
argument, and hyperparameter/constant is preserved byte-for-byte. The **only** sanctioned signature
changes are trailing keyword-only `rng=`/`seed=` additions (mandated by §8 because JAX has no global RNG)
and a small set of explicitly-flagged `# API-CHANGE` deltas where torch's stateful/in-place semantics
are impossible in JAX. 31 per-file agents converted the files below; 7 subsystem agents verified
cross-file consistency. **No code was runtime-executed** — JAX/Flax/optax/distrax are not installed in
this environment, so the only gate applied was `ast.parse` (syntax) plus AST-level torch-residue scans.

## 2. Per-file conversion table (31 files)

| File | ast_ok | torch_removed | api_preserved | LOC (before→after) | Note |
|---|---|---|---|---|---|
| Pretrain/utils.py | ✅ | ✅ | ✅ | 129→130 | set_seed→PRNGKey; check_device→backend str; compare via flax serialization |
| Pretrain/Dataset.py | ✅ | ✅ | ❌ | 1422→1429 | Dropped `torch.utils.data.Dataset` base; windows/conditions are np.float32 |
| Pretrain/Planners/Backbone/BaseDiffusion.py | ✅ | ✅ | ✅ | 44→46 | Abstract base → linen; map_noise built in setup() |
| Pretrain/Planners/Backbone/utils.py | ✅ | ✅ | ❌ | 627→665 | Backbone helpers→linen; EMA→target_update; get_pretrained_planner = bridge TODO |
| Pretrain/Planners/Backbone/Dit.py | ✅ | ✅ | ❌ | 181→213 | DiT→linen; MultiHeadDotProductAttention; `train` flag for dropout |
| Pretrain/Planners/Backbone/UNet.py | ✅ | ✅ | ✅ | 248→302 | TemporalUnet→linen; NCL↔NLC transpose at conv boundary |
| Pretrain/Critic/nets.py | ✅ | ✅ | ✅ | 173→182 | Critic/CriticEnsemble→linen; ensemblize; agg axis dim=-1→axis=0 |
| Pretrain/Rewards/nets.py | ✅ | ✅ | ✅ | 677→616 | Reward nets→linen; distrax.Beta; vmap(grad) per-sample grads |
| Pretrain/Transition_Kernel/Kernel_Net.py | ✅ | ✅ | ✅ | 185→175 | Robust/MoG kernels→linen; clamps/return tuples byte-identical |
| Pretrain/Planners/Backbone/Sampler.py | ✅ | ✅ | ✅ | 752→762 | Live samplers→jnp + rng=; archived samplers kept as inert string |
| Pretrain/Planners/Backbone/Trainer.py | ✅ | ✅ | ❌ | 402→516 | SDETrainer→TrainState+optax; seed=, Loss/selector rng=; save→flax |
| Pretrain/Critic/train_critic.py | ✅ | ✅ | ✅ | 639→694 | Critic train→TrainState; optax cosine; Huber; train_critic rng= |
| Finetuning/traj_reward2.py | ✅ | ✅ | ✅ | 508→497 | TotalReward holders→(model_def,params); vjp input-grads |
| Finetuning/traj_reward3.py | ✅ | ✅ | ❌ | 511→499 | Same as traj_reward2; dropped nn.Module base; forward→__call__ |
| Finetuning/adjoint_matching.py | ✅ | ✅ | ❌ | 288→300 | Score nets→TrainState; jacrev/vjp; params=/rng= added |
| Finetuning/acc_adjoint_matching.py | ✅ | ✅ | ❌ | 788→838 | Accelerate→single-device; jvp/jacrev/vjp; rng=/seed=; device='cpu' |
| Finetuning/Finetune_Backbone.py | ✅ | ✅ | ✅ | 917→760 | Orchestrator; _SingleDeviceAccelerator shim; seed=/rng= |
| Pretrain/pretrain_script.py | ✅ | ✅ | ✅ | 33→23 | Entry; dropped torch device; rng=set_seed(1) |
| Pretrain/train_critic_script.py | ✅ | ✅ | ✅ | 267→268 | Entry; rng threaded into live train_critic |
| Pretrain/train_kernel_script.py | ✅ | ✅ | ✅ | 51→53 | Entry; rng threaded into train_mog_kernel/test_kernel_mog |
| Pretrain/train_reward_script.py | ✅ | ✅ | ✅ | 239→263 | Entry; rng captured (not forwarded — callee unported) |
| Pretrain/test_kernel_script.py | ✅ | ✅ | ✅ | 51→54 | Entry; rng threaded into test_kernel |
| Pretrain/test_reward_script.py | ✅ | ✅ | ✅ | 30→25 | Entry; rng captured (not forwarded — callee unported) |
| Finetuning/finetune_script.py | ✅ | ✅ | ❌ | 59→53 | Entry; dropped mp.spawn (was eager no-op); finetune_planner(seed=rng) |
| Finetuning/train_critic_script.py | ✅ | ✅ | ✅ | 167→165 | Entry; removed stray import torch; rng=set_seed(1) |
| Pretrain/Planner_Rollout.py | ✅ | ✅ | ❌ | 327→347 | Rollout; rollout/rollout_parallel seed=; checkpoint-bridge TODOs |
| *(see note below — 31 agents, 26 distinct files listed)* | | | | | |

> Note: the JSON contained 26 distinct conversion records (several "31 per-file agents" cover the
> shared entry-script set and duplicate-named files such as `pretrain_script{2..5}.py` which mirror
> `pretrain_script.py`). Every record above is reproduced faithfully; the duplicated-pattern scripts
> (`pretrain_script2/3/4/5.py`) share the `pretrain_script.py` treatment and the same open RNG caveat.

**Totals: 26/26 reported files `ast_ok = true`. `torch_fully_removed = true` for every file in this
set.** `public_api_preserved = false` for 8 files — in all 8 cases the break is a *sanctioned* one
(rng=/seed=/params= additions or dropping an impossible torch base class / stateful mutation), fully
documented in §3 below.

## 3. All public-API changes across the port

### 3a. RNG/seed threading (mandated by CONVERSION_GUIDE §8 — JAX has no global RNG)

These are all **trailing keyword-only** additions; existing positional/keyword calls are unaffected.

- `Pretrain/utils.py :: set_seed` — now **returns** `jax.random.PRNGKey(seed)` (was `None`); still seeds
  `random`/`np.random`/`PYTHONHASHSEED`; dropped all torch seeding/cudnn flags.
- `Pretrain/Planners/Backbone/utils.py :: FourierEmbedding.__call__ / UntrainableFourierEmbedding.__call__`
  — added keyword-only `rng=None` (to draw the frozen `freqs` buffer sample).
- `Pretrain/Planners/Backbone/Dit.py :: DiTBlock.__call__ / DiT1d.__call__ / DiT1Ref.__call__` — added
  trailing `train: bool = False` (gates dropout `deterministic`; dropout key supplied via `rngs={'dropout': key}`).
- `Pretrain/Planners/Backbone/Sampler.py` — `sample_reverse_sde`, `sample_euler_karras`,
  `sample_euler_karras2` gained trailing keyword-only `rng=None` (after the positional `device=`).
- `Pretrain/Planners/Backbone/Trainer.py :: SDETrainer.__init__` — added keyword-only `seed: int = 0`;
  `SDETrainer.Loss` / `SDETrainer.selector` gained trailing `rng=None`.
- `Pretrain/Critic/train_critic.py :: train_critic` — added trailing `rng=None`.
- `Finetuning/adjoint_matching.py` — `sample_Traj`, `step` gained `rng=None`.
- `Finetuning/acc_adjoint_matching.py` — `sample_Traj`, `sample_Traj_karras`, `step` gained `rng=None`;
  `finetune_planner` gained `seed=None`; new private helper `_next_rng`.
- `Finetuning/Finetune_Backbone.py :: OnlineFinetuner.finetune_planner` gained `seed=None`;
  `get_generated_plans` gained `rng=None`.
- `Finetuning/finetune_script.py :: set_seed` — same return-key change as Pretrain/utils.py.
- `Pretrain/Planner_Rollout.py` — `rollout`, `rollout_parallel` gained trailing keyword-only `seed=None`.

### 3b. `params=`/`new_score_net=` additions (functional gradient flow — NOT the rng exception)

- `Finetuning/adjoint_matching.py :: vector_field` — added `*, params=None`;
  `adjoint_matching_loss` — added trailing `params=None`. Needed so `jax.grad` flows only through the
  trainable new_score_net (torch relied on live `requires_grad` state).
- `Finetuning/acc_adjoint_matching.py :: adjoint_matching_loss` — added trailing `new_score_net=None`.

### 3c. Stateful/in-place semantics that could not survive (flagged `# API-CHANGE`)

- `Pretrain/utils.py :: check_device` — returns a backend **string** (`jax.default_backend()`),
  not a `torch.device`.
- `Pretrain/utils.py :: compare_models_state_dict` — operands are now flax param trees (read via
  `flax.serialization.to_state_dict`), compared with `jnp.allclose`; signature/return semantics unchanged.
- `Pretrain/Planners/Backbone/utils.py :: EMA.update_model_average` — now takes `(ma_params, current_params)`
  pytrees and **returns** the updated EMA tree (was in-place `.data` mutation of torch modules).
- `Pretrain/Planners/Backbone/utils.py :: get_pretrained_planner` — body **raises NotImplementedError**
  (checkpoint-bridge TODO); signature unchanged.
- `Pretrain/Planners/Backbone/Trainer.py :: reset_parameters / step_ema` — copy/return EMA pytree;
  `save` — pickles `flax.serialization.to_state_dict(ema_params)` (format changed, path layout preserved).
- `Finetuning/traj_reward3.py :: TotalReward / TotalReward_Critic` — dropped `torch.nn.Module` base;
  `forward` → `__call__` (callers `reward_model(x, lam)` still work). Same for `traj_reward2.py` (base dropped).
- `Finetuning/adjoint_matching.py :: AdjointMatchingConfig.device` — removed `torch.device(...)` class attr.
- `Finetuning/acc_adjoint_matching.py :: Acc_AdjointMatchingConfig.device` → `'cpu'`;
  `set_optimizer_and_scheduler` (round>1) — rebuilds cosine schedule from base lr (optax has no
  `param_groups`, so LR restarts at base each round — flagged for multi-round-schedule review).
- `Finetuning/finetune_script.py :: __main__` — removed `torch.multiprocessing mp.spawn(...)` wrapper
  (the original passed the eagerly-evaluated `finetune_planner()` result into spawn, so it never actually
  parallelized); now calls `finetune_planner(seed=rng)` directly. **If true multi-device parallelism was
  intended, it must be reintroduced via jax pmap/sharding.**
- `Finetuning/Finetune_Backbone.py` — `accelerate.Accelerator` replaced by an in-file
  `_SingleDeviceAccelerator` shim; device-info prints use `jax.devices()`.

## 4. Cross-file inconsistencies (from the 7-agent verify pass)

### HIGH severity (open — must fix before the subsystem can run)

1. **Rewards trainer un-converted.** `Pretrain/Rewards/Reward_Backbone.py` still has live
   `import torch`/`optim`/`F`/`DataLoader` and torch throughout `train_reward*`/`test_Model*`/save/load.
   It constructs the now-linen `SimpleReward`/`EnsembleReward` with `.to(device)`, uses
   `optimizer.zero_grad/backward/step`, `reward_net.parameters()`, `load_state_dict`/`state_dict`/`eval`,
   `torch.randint` (un-keyed bootstrap), and `F.smooth_l1_loss`/`torch.where`/`torch.relu`. The whole
   rewards call graph is broken until ported.
   *Fix:* port per §6/§13: optax `clip_by_global_norm(1.0)` + `adamw(cosine_decay_schedule, wd=1e-4)`,
   `TrainState.apply_loss_fn`, numpy `sample()` datasets, `optax.huber_loss(delta=1.0)`, `jnp.where`,
   `jax.nn.relu`, `_bootstrap_per_member(..., *, rng)` with `jax.random.randint`, flax save/restore.

2. **Kernel trainer un-converted.** `Pretrain/Transition_Kernel/Kernel_Backbone.py` still does live
   `import torch`/`optim`/`DataLoader` (49 executable torch lines) and drives the now-linen
   `RobustTransitionKernel`/`MoGTransitionKernel` via the torch object protocol (`.to()`, `.eval()`,
   `.train()`, `.state_dict()`, `.load_state_dict()`, direct `m(s,a)`). No `.init`/`.apply`/TrainState
   anywhere. The caller crashes at first construction.
   *Fix:* port to TrainState + `model_def.init(...)`/`apply`; convert datasets to numpy `sample()`;
   optax + flax serialization; kernels stay python lists per §11.

3. **Kernel entry scripts thread `rng=` into un-ported callees (TypeError).**
   `Pretrain/train_kernel_script.py:41,54` pass `rng=` into `train_mog_kernel`/`test_kernel_mog`, and
   `Pretrain/test_kernel_script.py:21` passes `rng=` into `test_kernel` — none of those live torch
   signatures accept `rng`/`**kwargs`. Hard TypeError at call time.
   *Fix:* part of HIGH #2 — add keyword-only `*, rng=None` to those three entry points when porting
   `Kernel_Backbone.py`.

4. **`adjoint_matching.py` TotalReward construction is arg-shifted.**
   `Finetuning/adjoint_matching.py:97-98` calls `TotalReward(RewardConfig, env_name, specific_env,
   reward_model_checkpoint, kernel_model_checkpoint)` (5 args, no `device`), but the converted
   `TotalReward.__init__` is `(self, device, config, dataset_name, specific_dataset, reward_checkpoint,
   kernel_checkpoint)` — `device` is first. Every arg is shifted by one and `kernel_checkpoint` is unbound
   → TypeError.
   *Fix:* `TotalReward(<device-or-None>, RewardConfig, env_name, specific_env, reward_model_checkpoint,
   kernel_model_checkpoint)`. (May be a latent bug from the torch original; either way the converted
   callee requires `device` first.)

5. **Finetuning shared dependency un-converted: `Finetuning/utils.py`.** Full torch imports + executable
   torch throughout (`torch.randint`, `torch.load`/`save`, `torch.no_grad` train loops, `torch.device`,
   the `*Dataset` classes, `karras_beta_schedule`/`clip_actions`, `rollout_parallel2`, `train_*`). Every
   other finetuning file imports from it, so the converted siblings mix jnp ops with torch tensors / feed
   flax TrainStates into torch `.state_dict()`/`.eval()` code.
   *Fix:* port per the guide; make `karras_beta_schedule`/`clip_actions` byte-match the jnp twins already
   in `Pretrain/Planners/Backbone/Sampler.py`; `clip_actions` must become
   `x = x.at[..., d_s:].set(jnp.clip(...))`.

6. **`Finetuning/Rollout.py` un-converted** and calls the converted `Sampler.sample_euler_karras`
   **without `rng=`** (would hit `jax.random.split(None)`). Imported by `Finetune_Backbone.py`.
   *Fix:* port (drop `@torch.no_grad`, `.at[].set()`, thread `*, rng=`, `set_seed`→PRNGKey).

7. **Converted callers feed JAX objects into still-torch utils.** `acc_adjoint_matching.py:836` passes a
   flax `TrainState` into `Finetuning/utils.py :: save_planner` which calls `.eval()/.state_dict()/torch.save`;
   `acc_adjoint_matching.py:125,392,414,444` mixes torch-tensor `karras_beta_schedule`/`clip_actions`
   results into jnp expressions. Both resolve once HIGH #5 (utils.py) is ported.

### MEDIUM severity (open)

- **`finetune_script.py` builds `FinetuningConfig` missing 5 required fields** (`AlphaConfig`,
  `critic_model_checkpoint`, `train_reward_config`, `train_kernel_config`, `train_critic_config`) →
  `TypeError: missing 5 required positional arguments`. Likely pre-existing drift in the torch example
  script. *Fix:* pass them explicitly (or add defaults only if the torch original had them).
- **`Kernel_Backbone.py:936` positional MoG construction binds `noise_floor` into `min_log_std`.**
  `MoGTransitionKernel(obs_dim, act_dim, num_modes, num_hidden_layers, hidden_dim, noise_floor)` passes 6
  positional args but the linen dataclass field order is
  `(..., hidden_dim, min_log_std, max_log_std, noise_floor)`, so the 6th binds to `min_log_std`. Likely
  pre-existing. *Fix:* make the call keyword-based; do **not** reorder the Kernel_Net.py fields.
- **`save_planner`/`get_planner` (utils.py) torch save/load against TrainState** — partly covered by
  existing `TODO(checkpoint-bridge)`; needs the real torch→flax key remap.

### LOW severity (open / informational)

- **Unused-seed threading** in `pretrain_script.py`, `train_reward_script.py`, `test_reward_script.py`,
  `Finetuning/train_critic_script.py`: `rng = set_seed(1)` is computed but never threaded; `SDETrainer`
  self-seeds via `seed=0` (so `set_seed(1)` is ignored), and `train_reward`/`test_Model` self-seed.
  Reproducibility intent is silently lost. *Fix:* pass `seed=1` to `SDETrainer` (or add `rng=` to the
  reward callees), or drop the unused assignment.
- **`pretrain_script.py` (and 2/3/4/5)** construct `SDETrainer(...)` without `seed=`, so the computed key
  from `set_seed(1)` is discarded (training runs off `seed=0`). Same root cause as above; suggested fix is
  to have `SDETrainer` accept the PRNGKey directly (`rng=None`) and pass `SDETrainer(..., rng=set_seed(1))`.
- **FourierEmbedding `rng=` knob is dead through DiT1d.** `DiT1d.__call__` calls `self.map_noise(noise)`
  without forwarding `rng=`, so the const `freqs` always inits via `make_rng('params')`. Not a correctness
  bug (deterministic-per-init); the advertised knob is just unreachable. *Fix:* drop the knob or thread rng.
- **Cosine clamp constant `1.0 - 1e-3`** in live `cosine_beta`/`cosine_alpha_sigma`/`compute_dot_alpha_beta`
  (utils.py) vs `1.0 - 1e-6` in the archived dead-string sampler. The three live helpers are mutually
  consistent; could not diff against the true torch original (not a git repo / no backup). *Fix:* confirm
  against the torch source; the dead `1e-6` string is never imported.
- **`Planner_Rollout.py` dead-path bugs** (only reached when `critic=True` and the commented `__main__`
  rollout is enabled): `Critic_Processor` is referenced but never imported (NameError), and
  `self.critic.apply({'params':...}, obs, act)` passes 2 inputs to the 1-input linen `Critic.__call__`.
  Both pre-exist in the torch original. *Fix:* reconcile when wiring the checkpoint bridge.
- **`Pretrain/Critic/train_critic.py:559 test_critic`** passes `(dataset_name, specific_dataset, task_id,
  checkpoint_step)` into `get_critic_model(dataset_name, specific_dataset, step, task_id=None)` — task_id
  and checkpoint_step are swapped. **Confirmed pre-existing** in the torch original; faithfully preserved.
- **`get_env`/`get_dataset` param-name mismatch** (`episode_length` into `traj_length` slot) — benign,
  all call sites pass `None`. Pre-existing.
- **Bare `set_seed(1)` at `Finetuning/train_critic_script.py:154`** discards the key (the matching
  branches at 58/197 capture it). Low impact in that branch.
- **Import-path fragility:** `adjoint_matching.py:24` and `Finetune_Backbone.py:32` use `from utils import`
  (no top-level `/Users/kaiwenhu/ODP/utils.py`); the defs live in `Finetuning/utils.py`. Likely
  pre-existing; normalize to `from Finetuning.utils import`.
- **Style:** several `Pretrain/Dataset.py` lines and `train_reward_script.py` lines exceed 120 cols
  (pre-existing registry/import lines and dead triple-quoted blocks). Cosmetic.

## 5. Checkpoint-bridge TODOs (torch .pt/.pkl ingest → flax param-tree remap still needed)

The canonical remap rule across all of these: **Dense weight (out,in) → kernel (in,out) transposed;
bias → bias; LayerNorm weight → scale**. None of these are implemented yet — they require torch + the
real checkpoints + a JAX runtime to validate numerics.

- `Pretrain/Planners/Backbone/utils.py :: get_pretrained_planner` — currently raises `NotImplementedError`.
  This blocks `Trainer.selector`, `Trainer.save` consumers, and all planner rollouts.
- `Pretrain/Planners/Backbone/Trainer.py :: save / selector` — save emits flax-serialized pytree (path
  layout preserved); selector loads via `flax.serialization.from_state_dict` once the bridge yields a
  flax-compatible tree.
- `Pretrain/Critic/train_critic.py :: save_critic / save_to_finetuning / get_critic_model / test_critic` —
  JAX-to-JAX is self-consistent; **legacy torch `.pkl` ingest needs the key remap**.
- `Finetuning/traj_reward2.py` / `traj_reward3.py` — reward/kernel/critic subnet params ingested from
  torch state_dicts in `__init__` need the remap.
- `Finetuning/adjoint_matching.py` / `acc_adjoint_matching.py` — `get_pretrained_planner`/`get_planner`
  EMA planner state_dict → flax DiT1d param tree; reward/kernel/critic ingest via the traj_reward bridge.
- `Pretrain/Planner_Rollout.py :: ActionSelector.__init__ / rollout / rollout_parallel` — torch critic and
  EMA planner `load_state_dict`/`torch.load` ingests; `critic_params` is currently `None` with a TODO.
- `Finetuning/utils.py` (un-converted): `get_reward_model`/`get_kernel`/`get_planner`/`save_planner` etc.
- `Finetuning/Rewards`/`Kernel` backbones (un-converted): `save_model`/`load_model`/`get_pretrained_*`.
- `Pretrain/Rewards/nets.py :: ScalarReward.predict (ci= path)` — torch `Beta.icdf` mapped to
  `dist.distribution.quantile` on the wrapped TFP Beta; confirm `distrax.Beta.distribution.quantile`
  exists at runtime (else use `scipy.special.betaincinv` / Newton). No active caller exercises this path.

## 6. What still needs the user

**Nothing was runtime-executed.** JAX, Flax, optax, and distrax are not installed in this environment, so
the only verification applied was `ast.parse` (syntax) plus AST-level scans confirming no *executable*
torch remains in the converted files (every residual `torch` token is inside a triple-quoted
archived-code string, a docstring, or a `# TODO/# API-CHANGE` comment). Linen `.init`/`.apply` tracing,
optax wiring, distrax availability, attention/conv shape round-trips, and gradient numerics are all
**unverified**.

### Recommended next steps to runtime-verify

1. **Install deps** (in the user's JAX env): `jax`, `jaxlib` (matching accelerator), `flax`, `optax`,
   `distrax`, plus the existing `gym/gymnasium`, `minari`, `ogbench`, `einops`. Confirm
   `distrax.Beta(...).distribution.quantile` resolves (needed only for the reward CI path).
2. **Finish the two un-converted backbones FIRST** — `Pretrain/Rewards/Reward_Backbone.py` and
   `Pretrain/Transition_Kernel/Kernel_Backbone.py` — and the finetuning shared deps
   `Finetuning/utils.py` and `Finetuning/Rollout.py`. These four files block their entire subsystems
   (HIGH #1, #2, #3, #5, #6, #7) and are the largest remaining risk.
3. **Fix the two concrete caller breaks:** `adjoint_matching.py` TotalReward arg-shift (HIGH #4) and
   `finetune_script.py` missing FinetuningConfig fields (MEDIUM).
4. **Smoke-test the cleanly-converted Pretrain spine first** (it has no un-converted dependency except
   the checkpoint bridge): `import` `Pretrain/Critic/nets.py`, `Pretrain/Rewards/nets.py`,
   `Pretrain/Transition_Kernel/Kernel_Net.py`, `Pretrain/Planners/Backbone/{BaseDiffusion,utils,Dit,UNet,
   Sampler}.py`, then `model_def.init(rng, *example_inputs)` each network and confirm output shapes.
   *Expected failure modes:* flax "non-default field follows default" on DiT1d/DiT1Ref dataclass
   inheritance; `MultiHeadDotProductAttention` feature-dim inference; the const-`freqs` `'consts'`
   variable-collection wiring on FourierEmbedding; `jnp.clip` `a_min`/`a_max` vs `min`/`max` kwarg on
   older JAX.
5. **Implement the checkpoint bridge** (torch state_dict → flax pytree key remap, §5) — until then
   `get_pretrained_planner` raises `NotImplementedError` and every rollout/finetune path that loads a
   pretrained planner will stop there. *Expected failure mode:* `jax.random.split(None)` in samplers if a
   caller forgets to thread `rng=` (notably the still-un-converted Rollout/utils callers).
6. **Thread the seeds** in the entry scripts (LOW findings) if run reproducibility tied to `set_seed(1)`
   matters: pass `seed=1`/`rng=set_seed(1)` into `SDETrainer` and the reward callees.

### Files/areas to review first
`Finetuning/utils.py`, `Pretrain/Rewards/Reward_Backbone.py`,
`Pretrain/Transition_Kernel/Kernel_Backbone.py`, `Finetuning/Rollout.py` (the four un-converted
load-bearing files), then `Finetuning/adjoint_matching.py:97-98` and `Finetuning/finetune_script.py:37-47`
(the two concrete caller breaks), then the checkpoint bridge in
`Pretrain/Planners/Backbone/utils.py :: get_pretrained_planner`.

---

## Final confirmation

- **ast.parse:** **26/26** reported converted files passed (every file in this report parsed cleanly; all
  31 per-file agents reported `ast_ok = true`).
- **Open HIGH-severity cross-file inconsistencies:** **7** (rewards trainer un-converted; kernel trainer
  un-converted; kernel entry scripts thread rng= into un-ported callees; adjoint_matching TotalReward
  arg-shift; Finetuning/utils.py un-converted; Finetuning/Rollout.py un-converted; converted callers feed
  JAX objects into still-torch utils). All trace to four un-converted load-bearing files plus one concrete
  caller arg-shift.

## 8. Round 3 — utils.py finished, fixes applied, dead code removed, final verify

This round closed the last large gap from §4 (`Finetuning/utils.py` was the dominant un-converted
load-bearing file), applied the three concrete cross-file caller fixes, swept dead torch-era code out of
the whole spine, and ran a final 4-subsystem verification pass. The frozen-API principle (§1) held
throughout: the only sanctioned signature deltas remain trailing keyword-only `rng=`/`seed=` additions.

### 8a. `Finetuning/utils.py` tail completion

All **11** still-torch tail symbols in `/Users/kaiwenhu/ODP/Finetuning/utils.py` were converted to
JAX/Flax, each mirroring its already-converted twin (`train_critic`, `rollout_parallel`, `CriticDataset`,
`Critic_Buffer`):

`rollout_parallel3`, `check_device`, `compute_threshold_mahalanobis`, `compute_threshold_mahalanobis_mog`,
`compute_threshold_log_prob`, `compute_threshold_log_prob_mog`, `train_critic_with_planner`,
`CriticDataset_Reward`, `Critic_Buffer_Reward`, `train_critic_with_reward`, `train_critic_with_planner2`.

Mechanics: trainable critics use `jax_utils.TrainState` + `optax` (`cosine_decay_schedule` +
`clip_by_global_norm` + `adam/adamw`); per-step gradient via `@jax.jit _update` →
`train_state.apply_loss_fn` with `optax.huber_loss` (== torch `smooth_l1_loss` beta=1.0). Target nets are
frozen `TrainState`s updated with `target_update(online, tgt, tau)`. Frozen planner/reward nets are
`TrainState`s called without `params=` (== torch `no_grad`); kernels are a python list of
`(model_def, params)` tuples (per §11). `torch.device/.cuda/.to(device)` dropped; `.item()`→`float()`;
`.detach().cpu().numpy()`→`np.asarray`; `torch.cat`→`np.concatenate`; `torch.quantile`→`np.quantile`;
in-place `advantages[:,t]=`→`.at[:,t].set()`; `torch.clamp`→`jnp.clip`; `F.smooth_l1_loss`→
`optax.huber_loss`. `check_device` now returns `jax.default_backend()` (matches `Pretrain/utils.py`). All
`sample_euler_karras` calls thread `rng=`. Datasets keep numpy fields with fql-style `sample()`;
`DataLoader`/`cycle` replaced by numpy sampling. **Result: `ast.parse` OK, executable torch residue 0.**

**API changes (all the sanctioned rng= class — §3a):** `rng=None` (trailing keyword-only) added to
`rollout_parallel3`, `train_critic_with_planner`, `CriticDataset_Reward`, `train_critic_with_reward`,
`train_critic_with_planner2`. `rollout_parallel3` also dropped its `device: torch.device` annotation →
`device: str`. `CriticDataset_Reward` dropped the torch `Dataset` base class (now a plain class, matching
the converted `CriticDataset`; `Dataset`/`DataLoader` no longer imported). `EMA.update_model_average` was
noted for completeness only — not in scope, not modified.

**New checkpoint-bridge TODOs (added to the §5 list):** `train_critic_with_planner` /
`train_critic_with_reward` / `train_critic_with_planner2` — `get_critic_model` returns either a saved flax
param tree (new) or a torch state_dict (legacy) needing the per-Dense remap before `from_state_dict`;
planner (DiT1d) and reward (SimpleReward) params restored via `flax.serialization.from_state_dict` into a
frozen `TrainState`; `load_kernel_ensemble` rebuilds each kernel as a `(model_def, params)` pair (python
list, not a vmapped ensemble per §11); `CriticDataset_Reward` reward params and `rollout_parallel3` planner
params likewise restored into frozen `TrainState`s. All legacy torch state_dicts need the standard remap.

Out-of-scope code was left byte-identical: the commented-out `Critic_Test_Dataset`/`test_critic`
triple-quoted block and `rollout_parallel2` (which still contained `.to(device)`/`load_state_dict`/`F.` at
the time of this conversion — see 8d for `rollout_parallel2`'s status in the final verify).

### 8b. The 3 cross-file fixups — ALL RESOLVED

All three surgical fixes were applied in `/Users/kaiwenhu/ODP/Finetuning`; all three edited files pass
`ast.parse`. This closes §4 HIGH #4 and the two named MEDIUM/LOW caller items.

1. **RESOLVED — `adjoint_matching.py` (~L97-98) TotalReward arg-shift (was §4 HIGH #4).**
   `traj_reward.py :: TotalReward.__init__` is `(self, device, config, dataset_name, specific_dataset,
   reward_checkpoint, kernel_checkpoint)` — `device` is FIRST. The call passed 5 positionals starting with
   `RewardConfig` (no `device`), so every arg was off by one. Fix: inserted `None` as the leading `device`
   slot → `TotalReward(None, RewardConfig, env_name, specific_env, reward_model_checkpoint,
   kernel_model_checkpoint)`. Passed values otherwise identical.

2. **RESOLVED — `finetune_script.py` (~L37-47) missing 5 required `FinetuningConfig` fields (was §4
   MEDIUM).** Added `AlphaConfig`, `critic_model_checkpoint`, `train_reward_config`, `train_kernel_config`,
   `train_critic_config`. The `train_*_config` use the sub-config dataclasses' own defaults
   (`Train_Reward_Config()`, `Train_Kernel_Config()`, `Train_Critic_Config()`); `critic_model_checkpoint=0`;
   `AlphaConfig=AlphaSchedulerConfig(alpha_start=1.0, alpha_end=1.0, total_steps=1000000)` (3 required
   fields, no defaults, so a neutral constant-alpha schedule was supplied since the script has no opinion).
   Imports extended to pull `Train_Reward_Config`/`Train_Kernel_Config`/`Train_Critic_Config` from
   `Finetune_Backbone` and `AlphaSchedulerConfig` from `Finetuning.utils`.

3. **RESOLVED — bare `from utils import` import-path normalization (was §4 LOW).**
   `adjoint_matching.py` L24 `from utils import Lambda, function` → `from Finetuning.utils import ...`;
   `Finetune_Backbone.py` L32 `from utils import TrajectoryDict, ...` → `from Finetuning.utils import ...`.
   Imported names unchanged; all symbols verified present in `Finetuning/utils.py`; no bare
   `from utils import` remains (no top-level `utils.py` exists).

### 8c. Dead-code tidy (whole spine)

A conservative dead-code sweep removed only (rule 1) bare triple-quoted **archived-code string statements**
that are runtime no-ops and provably NOT docstrings (never the first statement of a module/class/function
body), and (rule 2) earlier shadowed **duplicate top-level defs**. Every genuine docstring, every live
symbol, all imports, signatures, defaults, constants, and comments were preserved. Each file was AST-gated
(parse OK) and its live top-level public-symbol set confirmed identical before/after.

**Totals: 22 files touched (1 left untouched), ~2,840 dead lines removed, 1 duplicate top-level def
removed** (`karras_beta_schedule`, the earlier numerical-diff variant in `Sampler.py`; Python last-wins
already used the later analytic def, so the removal is behavior-preserving). Notable per-file deletions:
`Finetuning/traj_reward.py` −270 (archived torch `TotalReward` + driver script);
`Pretrain/Transition_Kernel/Kernel_Backbone.py` −189 (9 archived torch blocks);
`Pretrain/Rewards/Reward_Backbone.py` −167 (4 blocks); `Pretrain/Rewards/nets.py` −186 (5 SimpleReward/
EnsembleModel torch impls); `Pretrain/Planners/Backbone/Sampler.py` −460 (dup def + a 413-line archived
torch-sampler string); `Finetuning/utils.py` −152 (3 archived torch blocks); `Pretrain/Dataset.py` −379
(6 blocks); plus smaller removals in `Pretrain/utils.py`, `Critic/nets.py`, `Critic/train_critic.py`,
`Kernel_Net.py`, `UNet.py`, `Trainer.py`, `Planner_Rollout.py`, `acc_adjoint_matching.py`,
`Planners/Backbone/utils.py`. `Finetuning/traj_reward2.py` and `traj_reward3.py` were left **untouched** on
purpose: their lone torch-bearing triple-quoted blocks are the class **docstring** of `TotalReward` (first
statement of the class body), which the docstring-protection rule forbids removing. No live symbol was
renamed, removed, or reordered in any file; no executable torch was removed (it was already 0 in these
files — only dead-string and comment `torch` mentions changed).

### 8d. Final per-subsystem verify

| Subsystem | Files | all_ast_ok | all_torch_removed | Verdict |
|---|---|---|---|---|
| pretrain-spine | 14 | ✅ | ✅ | **PASS** (with minor low-severity notes) |
| finetuning | 8 | ✅ | ❌ | **FAIL** (2 files with executable torch method-call residue + 1 HIGH API mismatch) |
| entry-scripts | 9 | ✅ | ✅ | **PASS** (with caveats — RNG-threading slips only) |

**REMAINING inconsistencies by severity (post-fixup):**

**HIGH (3 — all in the finetuning subsystem):**
1. `Finetuning/utils.py:2066-2078` `rollout_parallel2` — executable torch residue on a linen module:
   `DiT1d(...).to(device)`, `model.load_state_dict(state_dict)`, `model.eval()`. DiT1d is now linen — these
   methods don't exist and no `TrainState` is built. **Called by `Finetune_Backbone.py:698` in the main
   finetune loop → will crash.** (This function was explicitly out of scope in 8a; it now surfaces as the
   sole remaining torch hole in utils.py.) *Fix:* mirror `rollout_parallel3` — `DiT1d(...)` (drop
   `.to`), `model.init(...)`, `from_state_dict`, `TrainState.create`; thread `rng=` into the
   `sample_euler_karras` calls.
2. `Finetuning/acc_adjoint_matching.py:623` `finetune_planner` — `reward_model.eval()` on a
   `traj_reward3.TotalReward`/`TotalReward_Critic` plain container (no `.eval()`) → `AttributeError` in the
   hot path. *Fix:* delete the line; frozen-net semantics are already correct via no-grad `TrainState`
   calls without `params=`.
3. **Kernel-list representation mismatch.** `Kernel_Backbone.compute_log_density` /
   `compute_total_mahalanobis_score` (+ `_mog` variants) iterate `for model_def, params in kernels` and read
   `kernels[0][0].noise_floor` — they REQUIRE a python list of `(model_def, params)` tuples (§11). But
   `Finetuning/utils.py train_kernel` (L992-1015) & `train_kernel_mog` (L1148-1170) build lists of
   `TrainState` and pass them straight in, and `Finetuning/Rollout.py load_kernel` (L603-612) does the same
   → unpacking a 6-field `TrainState` PyTreeNode raises at runtime. *Fix:* append `(model_def, params)`
   tuples (the correct pattern already lives in `utils.train_critic_with_planner2.load_kernel_ensemble`).

**MEDIUM (3):**
- `traj_reward2.py:211,421` `TotalReward.forward`/`TotalReward_Critic.forward` — divergent entrypoint name
  vs `traj_reward.py`/`traj_reward3.py` which use `__call__`. Today's callers import the `__call__` variants
  so nothing breaks, but the naming inconsistency violates the frozen-API rule across the three
  nominally-interchangeable variants. *Fix:* rename `forward`→`__call__`.
- `Finetuning/Rollout.py:468,655` — `TrainState.create(model, state_dict)` passes the raw pickled state_dict
  as params with no `init` + `from_state_dict`, inconsistent with the established ingest pattern. *Fix:* use
  init→from_state_dict→create.
- `Kernel_Backbone.py:854 test_kernel_mog` — deliberate documented MoG kwarg change vs the frozen torch
  positional binding (torch's 6th positional landed on `min_log_std`, a latent torch bug; the port passes
  `noise_floor=` by keyword). This alters Mahalanobis/log-density diagnostic numerics. *Decision needed:*
  either reproduce the torch positional binding to be byte-faithful, or mark it `# API-CHANGE (intentional
  bug-fix vs torch)`.

**LOW (several, all latent/cosmetic):** samplers in `Sampler.py` (and `Rollout.sample_euler_karras_replan`,
which has no callers) declare `*, rng=None` but call `jax.random.split(rng)` without a None-guard — a
footgun, not an in-scope break, since live callers always pass a key; `CriticEnsemble`/`EnsembleReward`
raw fall-through (non-{mean,min} aggregate) returns leading-axis layout vs torch's trailing-axis (no active
caller hits it); RNG-threading slips in entry scripts (`set_seed(1)` key dropped, so callees silently use
`PRNGKey(0)` — affects `pretrain_script`, `train_reward_script`, `test_reward_script`,
`Finetuning/train_critic_script`); `adjoint_matching.step` return annotation `-> float` vs actual 3-tuple;
plus the pre-existing torch latent bugs (`test_critic` get_critic_model arg-swap; `Planner_Rollout`
`Critic_Processor` NameError) that were correctly **preserved**, not "fixed," per the frozen-API golden rule.

### 8e. Final status

- **`ast.parse` clean: 31/31 core files.** Every core file in the port parses cleanly.
- **torch-free (zero EXECUTABLE torch): 30/31 core files.** The lone exception is `Finetuning/utils.py`,
  whose `rollout_parallel2` (HIGH #1 above) still carries `.to()`/`.load_state_dict()`/`.eval()` on a linen
  DiT1d. (`acc_adjoint_matching.py:623`'s `reward_model.eval()` is the second executable-torch-residue site
  flagged by verify; it is a torch-style call on a plain container rather than a torch import, but it must
  still be deleted — so depending on how strictly one counts the call site, treat **utils.py + 1 line in
  acc_adjoint_matching.py** as the two remaining executable-torch-residue locations.) All other `torch`
  tokens repo-wide are now confined to comments, docstrings, and checkpoint-bridge/API-CHANGE notes.
- **Open HIGH issues: 3** (all finetuning; all from the final verify): `rollout_parallel2` torch residue;
  `acc_adjoint_matching.py:623` `reward_model.eval()`; kernel-list `(model_def, params)`-vs-`TrainState`
  representation mismatch. The two previously-open HIGH backbones from §4 (`Reward_Backbone.py`,
  `Kernel_Backbone.py`) and the un-converted `Finetuning/utils.py`/`Rollout.py` are now converted; §4 HIGH
  #4 (TotalReward arg-shift) is RESOLVED.
- **Open MEDIUM: 3** (traj_reward2 `forward`→`__call__`; Rollout.py raw state_dict ingest; test_kernel_mog
  MoG kwarg decision). **Open LOW: several** (sampler None-guards, ensemble raw-axis layout, entry-script
  seed threading, cosmetic annotations, preserved pre-existing torch latent bugs).
- **Dead lines removed this round: ~2,840** across 22 files; **1 duplicate top-level def removed**
  (`karras_beta_schedule`).
- **Still unverified at runtime:** JAX/Flax/optax/distrax remain uninstalled here, so all linen
  `.init`/`.apply` tracing, optax wiring, shape round-trips, gradient numerics, and the checkpoint bridge
  are unexercised. The single most important next step: **fix the 3 open HIGH finetuning issues** (two are
  a few-line `TrainState`-pattern fix + a line deletion; the third is the kernel-list tuple fix), then
  install deps and smoke-test the clean pretrain-spine via `model_def.init(rng, *example_inputs)` before
  attempting any finetune path.
