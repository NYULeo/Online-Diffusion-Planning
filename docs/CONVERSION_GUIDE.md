# ODP torch → JAX (FQL-style) CONVERSION GUIDE

Authoritative mapping rules for porting `/Users/kaiwenhu/ODP` (PyTorch) to JAX/Flax in the style of
`/Users/kaiwenhu/fql` (Flow Q-Learning). Every conversion agent MUST follow this document so that
~30 files converted in parallel stay mutually consistent and import-compatible.

Shared plumbing lives in `flax_utils.py` (mirrors fql's `utils/flax_utils.py` +
`utils/networks.py`). Import framework primitives from there — do **not** re-implement them.

---

## 0. THE GOLDEN RULE — PUBLIC API IS FROZEN

> **Every top-level class name, top-level function name, and call signature stays IDENTICAL to the
> torch version. Every hyperparameter, constant, and magic number stays IDENTICAL. Only the
> framework internals change.**

This is the single invariant that lets files convert in parallel without coordination. Concretely:

- Do not rename `DiT1d`, `RobustTransitionKernel`, `SimpleReward`, `train_critic`, `sample_euler_karras`,
  `OnlineFinetuner`, `AdjointMatchingFineTuner`, `rollout`, etc. Keep the exact names.
- Do not reorder, rename, add, or drop positional/keyword arguments. `def train_reward(dataset_name,
  hidden_layers, hidden_dim, batch_size, num_steps, save_freq, lr, min_lr, sigma=None, ...)` keeps that
  exact signature.
- Do not change default values, schedule constants (`s=0.008`), clamp bounds (`min_log_std=-6.0`,
  `max_log_std=4.0`), `noise_floor`, `eta`, `tau`, discount `gamma`, etc.
- Keep module-level constants and dataclass field names/defaults identical (e.g. `AdjointMatchingConfig`,
  `RewardConfig`, `FinetuningConfig`, `KernelConfig`).
- Preserve return *shapes* and *semantics*. A function returning `(mu, log_std)` still returns
  `(mu, log_std)`; a critic returning a squeezed scalar per batch element still does.

### The ONE allowed signature change: threading an RNG key

JAX has no global RNG. Any function/method that was *implicitly stochastic* in torch (calls
`torch.randn`, `torch.rand`, `randint`, `randperm`, dropout, `.normal_()`, `torch.multinomial`, samples a
distribution) MUST take an explicit key. Use this **uniform convention everywhere**:

- Add a **keyword-only** parameter named `rng` (or `seed` for the outermost policy/sample entry points,
  matching fql which uses `seed=`), with default `None`.
- Add it at the **end** of the signature so existing positional calls are unaffected.
- Inside, split as needed: `rng, subkey = jax.random.split(rng)`.

```python
# torch
def sample_euler_karras(score_model, d_s, d_a, horizon, num_steps=50, num_karras=5, eta=1.0, device=None):
    x = torch.randn(1, horizon, d_s + d_a, device=device)
    ...

# jax (key threaded, everything else frozen)
def sample_euler_karras(score_model, d_s, d_a, horizon, num_steps=50, num_karras=5, eta=1.0, *, rng=None):
    rng, k = jax.random.split(rng)
    x = jax.random.normal(k, (1, horizon, d_s + d_a))
    ...
```

If a *class* was stochastic, store an `rng` attribute (PyTreeNode field) or accept `rng=` on the
stochastic methods. Samplers that the planner/rollout call must accept `rng=` and split internally per
step. **Document any deviation inline with a `# API-CHANGE:` comment** so the verify pass can find it.

---

## 1. Standard import header

Every converted `.py` file starts from this header (delete unused lines, keep ordering):

```python
'''<one-line module purpose — copied/adapted from the torch docstring>'''
from typing import Any, Optional, List, Tuple, Dict, Sequence

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax

# einops works on jax arrays unchanged — keep torch-side einops calls as-is.
from einops import rearrange, repeat  # only if used

# Shared port plumbing (mirrors fql). Adjust the relative path to JAX_PORT as needed.
from JAX_PORT.jax_utils import (
    MLP, ModuleDict, TrainState, nonpytree_field, default_init, ensemblize,
    target_update, save_agent, restore_agent, supply_rng,
)
```

Keep the original ODP intra-repo imports (`from Pretrain.Dataset import get_dataset`, etc.) — the call
graph and module layout are unchanged (see MANIFEST for path-resolution rules). Only the *framework*
imports (`torch`, `torch.nn`, `torch.optim`, `torch.nn.functional as F`) get replaced.

Single quotes. 120-column lines. `distrax` for distributions if/when a torch `torch.distributions`
object is used.

---

## 2. `torch.nn.Module` → `flax.linen.Module`

Two valid styles (match fql, which uses both):
- **`setup()` style**: declare submodules as attributes in `setup(self)`, use them in `__call__`. Best
  when the torch `__init__` built named submodules (`self.net = ...`, `self.mean_head = ...`).
- **`@nn.compact` style**: define layers inline inside `__call__`. Best for short forward passes.

Linen modules are **frozen dataclasses**: constructor args become class-level annotated attributes;
there is no `__init__`/`super().__init__()`. Forward pass is `__call__` (rename torch `forward`).

> **Frozen-API note:** linen requires hyperparameters to be dataclass *attributes*, so the torch
> `__init__(self, obs_dim, hidden=32)` becomes attributes `obs_dim: int` / `hidden: int = 32`. The
> *construction call* `RobustTransitionKernel(obs_dim, act_dim, num_hidden_layers=2, ...)` is unchanged
> because dataclass fields accept the same positional/keyword args in the same order. Preserve field
> order to keep positional construction working.

### Example: `Critic` (setup style, real ODP code from `Pretrain/Critic/nets.py`)

```python
# torch
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            ... (8 blocks) ...,
            nn.Linear(hidden, 1), nn.ReLU())
    def forward(self, obs):
        return self.net(obs).squeeze(-1)
```

```python
# jax (setup style; identical name, identical construction signature)
class Critic(nn.Module):
    obs_dim: int
    hidden: int = 32

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(8):                       # keep the exact block count from torch
            x = nn.Dense(self.hidden)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        x = nn.Dense(1)(x)
        x = nn.relu(x)
        return x.squeeze(-1)
```

Note: in linen, `nn.Dense` infers `in_features` from the input, so torch's explicit `obs_dim` first-layer
size is *not* passed to `nn.Dense` — it is only needed for the example-input shape used at `init`. Keep
`obs_dim` as an attribute anyway (frozen API) even if unused internally.

### Layer-by-layer module map

| torch | flax.linen | notes |
|---|---|---|
| `nn.Linear(in, out)` | `nn.Dense(out)` | linen infers `in`; bias on by default in both. |
| `nn.Conv1d/Conv2d(in,out,k,stride,padding)` | `nn.Conv(features=out, kernel_size=(k,), strides=(s,), padding=...)` | **data-format differs — see §3**. |
| `nn.ConvTranspose*` | `nn.ConvTranspose` | same axis caveat as Conv. |
| `nn.LayerNorm(dim)` | `nn.LayerNorm()` | linen normalizes last axis; drop the explicit `dim`. |
| `nn.GroupNorm(g, ch)` | `nn.GroupNorm(num_groups=g)` | operates on last (channel) axis in flax. |
| `nn.BatchNorm*` | `nn.BatchNorm(use_running_average=not train)` | needs `train` flag + `batch_stats` collection; rare here. |
| `nn.Embedding(n, d)` | `nn.Embed(num_embeddings=n, features=d)` | |
| `nn.Sequential([...])` | inline the layers in `__call__` (compact) or a python loop | no direct `Sequential`; unroll faithfully. |
| `nn.ModuleList([...])` | a python `list` of submodules built in `setup`, **or** `ensemblize` (see §11) | for ensembles prefer `ensemblize`. |
| `nn.Parameter(t)` | `self.param('name', init_fn, shape)` | see §4. |
| `nn.Dropout(p)` | `nn.Dropout(rate=p)` then `(x, deterministic=not train)` | needs a `'dropout'` rng — see §6. |
| `nn.Identity()` | `jax_utils.Identity()` or just pass through | |

### Activations / functional ops

| torch | flax / jax |
|---|---|
| `F.relu`,`nn.ReLU` | `nn.relu` / `jax.nn.relu` |
| `F.silu`,`nn.SiLU`,`Swish` | `nn.silu` |
| `F.gelu`,`nn.GELU` | `nn.gelu` |
| `F.softplus` | `jax.nn.softplus` |
| `F.softmax(x,dim=d)` | `jax.nn.softmax(x, axis=d)` |
| `F.log_softmax` | `jax.nn.log_softmax` |
| `torch.tanh`,`torch.sigmoid` | `jnp.tanh`,`jax.nn.sigmoid` |
| `F.mse_loss(a,b)` | `jnp.mean((a - b) ** 2)` |
| `F.l1_loss` | `jnp.mean(jnp.abs(a - b))` |
| `F.softplus(x).logsumexp / torch.logsumexp` | `jax.scipy.special.logsumexp` |

`Swish`/`SiLU` modules in `Rewards/nets.py` → replace with `nn.silu` directly (drop the module).

---

## 3. Conv data-format & axis differences (CRITICAL — affects `Dit.py`, `UNet.py`, `Planners/Backbone/utils.py`)

- **torch** convs use **channels-first**: `Conv1d` expects `(N, C, L)`, `Conv2d` expects `(N, C, H, W)`.
- **flax** convs use **channels-last**: `nn.Conv` expects `(N, L, C)` / `(N, H, W, C)`.

The ODP planners are 1-D temporal models (`Conv1dBlock`, `ResidualTemporalBlock`, `TemporalUnet`) that
operate on `(batch, channels, horizon)`. Two faithful options — **pick per-module and document it**:

1. **Transpose at the conv boundary** (least invasive, keeps the rest of the tensor logic identical):
   ```python
   # torch: x is (B, C, L); conv1d
   h = self.conv(x)
   # jax: x is (B, C, L) coming in -> NLC for the conv -> back to NCL
   h = jnp.transpose(x, (0, 2, 1))
   h = nn.Conv(features=out_ch, kernel_size=(k,), strides=(s,), padding=pad)(h)
   h = jnp.transpose(h, (0, 2, 1))
   ```
2. **Carry NLC throughout the module** and transpose once at the module's input/output. Cleaner for
   `TemporalUnet` where many convs chain. If you do this, the module's *external* input/output shape MUST
   stay `(B, C, L)` to keep the public API frozen.

`einops.Rearrange('b c l -> b l c')` (already used in `UNet.py`/`utils.py` via
`einops.layers.torch.Rearrange`) → use functional `einops.rearrange(x, 'b c l -> b l c')` (the
`einops.layers.torch` layer class does not exist for jax; replace the layer with an inline `rearrange`
call in `__call__`).

Padding: torch integer `padding=p` (symmetric) → flax `padding=[(p, p)]` for 1-D (per-dim tuple list), or
`'SAME'`/`'VALID'` strings where the torch code intended that. Match the torch arithmetic exactly.

---

## 4. Parameter init faithfulness

`default_init()` in `jax_utils` (= fql's `variance_scaling(scale,'fan_avg','uniform')`) is the **house
style** but is **not** numerically identical to torch defaults. Rules:

- **If the model will be trained from scratch in JAX** (planner pretrain, reward/kernel/critic training):
  use `default_init()` / linen defaults. Document with `# init: fql-style (not torch-identical)`.
- **If a torch checkpoint is loaded** (`get_pretrained_planner`, `get_pretrained_reward`, `get_kernel`,
  `get_critic_model`, `load_state_dict`): init values are overwritten by the checkpoint, so init choice is
  irrelevant to numerics — but the **param tree structure/names must line up** with the converter used to
  ingest the checkpoint (see §10). Flag these modules `checkpoint-loaded`.
- Torch default reference (document the deviation if you must match it without a checkpoint):
  - `nn.Linear`: weight `kaiming_uniform_(a=sqrt(5))`, bias `uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`.
    Closest flax: `nn.initializers.he_uniform()` for the kernel; custom bias init for exact match.
  - `nn.Conv`: same kaiming_uniform scheme.
  - `nn.LayerNorm`: scale=1, bias=0 (flax default matches).
  - `nn.Embedding`: `normal(0,1)` (flax `nn.Embed` default is `variance_scaling`/normal — set
    `embedding_init=nn.initializers.normal(stddev=1.0)` to match exactly).

`nn.Parameter`:
```python
# torch:  self.scale = nn.Parameter(torch.tensor(5.0))
# jax (compact): scale = self.param('scale', lambda key: jnp.array(5.0))
```
Use `self.param('name', init_fn, shape)` for trainable params; `self.variable('batch_stats', ...)` for
non-trainable state.

---

## 5. Optimizers, schedules, grad clipping, EMA → optax

| torch | optax |
|---|---|
| `torch.optim.Adam(params, lr)` | `optax.adam(lr)` |
| `torch.optim.AdamW(params, lr, weight_decay=w)` | `optax.adamw(lr, weight_decay=w)` |
| `torch.optim.SGD(params, lr, momentum=m)` | `optax.sgd(lr, momentum=m)` |
| `optimizer.zero_grad(); loss.backward(); optimizer.step()` | `TrainState.apply_gradients` / `apply_loss_fn` (see §6) |
| `clip_grad_norm_(params, max_norm)` | `optax.clip_by_global_norm(max_norm)` chained **before** the optimizer in `optax.chain(...)` |
| LR schedule (cosine/linear `min_lr`→`lr`) | `optax.cosine_decay_schedule` / `optax.linear_schedule`; pass the schedule fn as `learning_rate=`. ODP uses `min_lr` + `lr` → `optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr/lr)`. Keep `lr`,`min_lr`,`num_steps` identical. |
| grad clip + Adam together | `optax.chain(optax.clip_by_global_norm(c), optax.adam(lr))` |

EMA (ODP's `EMA` class / `update_model_average` and planner target nets):
```python
# torch: ema_param = decay * ema_param + (1 - decay) * param   (in-place)
# jax:   ema_params = target_update(params, ema_params, tau=1 - decay)
```
`jax_utils.target_update(params, target_params, tau)` computes `tau*params + (1-tau)*target`. Map torch
`decay` → `tau = 1 - decay`. Keep the exact decay constant.

LR scheduling that torch did with `scheduler.step()` each iteration: fold the schedule into the optax
`learning_rate=schedule_fn` (it reads `opt_state.count`); delete the manual `.step()`.

---

## 6. Autograd: `backward()/step()` → `jax.grad` + `TrainState`

The fql pattern (mirror it):
```python
def loss_fn(params):
    ...
    return loss, info_dict          # has_aux

new_train_state, info = train_state.apply_loss_fn(loss_fn)   # grads + step + grad stats
```
or directly:
```python
grads, info = jax.grad(loss_fn, has_aux=True)(params)
new_train_state = train_state.apply_gradients(grads=grads)
```
`jax.value_and_grad(loss_fn, has_aux=True)` when you also want the loss value.

- `optimizer.zero_grad()` → delete (grads are functional, not accumulated).
- `loss.backward()` → delete; gradient comes from `jax.grad`.
- `optimizer.step()` → `apply_gradients` (returns a *new* state — assign it back).
- The torch training loop mutates the model in place; the JAX loop **rebinds** the trainer/agent each
  iteration: `agent, info = agent.update(batch)` (agent is a `flax.struct.PyTreeNode`). Trainers
  (`SDETrainer`, `OnlineFinetuner`, the various `train_*` functions) should hold a `TrainState` (or be a
  PyTreeNode) and JIT the per-step update.

### Stop-gradient / `no_grad` / `.detach()`
- `x.detach()` → `jax.lax.stop_gradient(x)`.
- `with torch.no_grad():` (eval / target computation) → call the network **without** passing
  `params=` (fql convention: omitting `params` uses stored params and does **not** flow gradients), or
  wrap the result in `jax.lax.stop_gradient`. For pure inference (samplers, rollout) just call the apply
  fn; nothing is traced unless inside a `jax.grad`.
- `@torch.no_grad()` decorator on samplers/rollout → drop it; ensure those fns are not differentiated.
- Frozen pretrained nets (`old_score_net.eval()`, `requires_grad_(False)`): never pass their params as
  `grad_params`; call via `train_state(...)` (no `params=`) or `stop_gradient`.

### RNG inside loss/update
Thread a key: `rng, sub = jax.random.split(rng)` and pass `sub` to stochastic sub-calls (matches fql's
`critic_loss(self, batch, grad_params, rng)`).

---

## 7. Custom autograd / backprop-through-ODE (adjoint matching) — `adjoint_matching.py`, `acc_adjoint_matching.py`, `AM.py`, `traj_reward*.py`

These files use:
- `torch.autograd.grad(outputs, inputs, grad_outputs=v, create_graph=, retain_graph=)`
- `torch.autograd.functional.jacobian(fn, x)`
- `torch.autograd.functional.jvp(fn, x, v)`
- `x.requires_grad_(True)` + `reward_model.get_c(x)` then gradient of `C` w.r.t. `x`

Mapping:

| torch | jax |
|---|---|
| `g = autograd.grad(out, inp, grad_outputs=v)[0]` (vector-Jacobian product) | `_, vjp_fn = jax.vjp(fn, inp); g = vjp_fn(v)[0]` |
| `J = autograd.functional.jacobian(fn, x)` | `J = jax.jacrev(fn)(x)` (or `jax.jacfwd` if wider-than-tall) |
| `out, jvp_out = autograd.functional.jvp(fn, x, v)` | `out, jvp_out = jax.jvp(fn, (x,), (v,))` |
| `grad of scalar C wrt x` (`x.requires_grad_(True); C.backward()`) | `g = jax.grad(lambda x: C_fn(x))(x)` |
| `create_graph=True` (need 2nd-order) | nest: `jax.grad(jax.grad(...))` / `jax.jvp` of a `jax.grad` — JAX composes naturally |
| `.detach()` on the carried adjoint state | `jax.lax.stop_gradient` |

The **lean-adjoint backward solve** is a hand-rolled reverse-time loop (python `for` over fixed
`num_steps`). **Keep it as a python loop** (faithfulness > cleverness) — do NOT convert to `lax.scan`
unless trivially safe. Inside, the per-step Jacobian/vjp of the frozen `old_score_net` is computed with
`jax.vjp`/`jax.jacrev` as above. The frozen score net is called without `params=` (no grad through it
except where the algorithm explicitly differentiates the *network output w.r.t. its input* — that is an
*input* gradient via `jax.vjp`/`jax.jacrev`, independent of param-gradients).

`compute_jacobian_vectorized(T, t_index)` (real ODP code) → 
```python
def score_fn(x_flat):
    x_reshaped = x_flat.reshape(T.shape)
    score = old_score_net(x_reshaped, t_asc[t_index][None], condition=None)   # via TrainState, no params=
    return score.reshape(-1)
jacobian = jax.jacrev(score_fn)(T.reshape(-1))
```
Keep the exact `eta`, `s`, `lam`, `num_steps`, schedule (`kt`, `sigma_t`, `cosine_alpha_sigma`) math
byte-for-byte; only the differentiation backend changes.

The reward gradient `compute_reward_gradients_per_sample(reward_net, obs, act, agg='mean')`
(`Rewards/nets.py`) → `jax.grad`/`jax.vmap(jax.grad(...))` over the per-sample reward.

---

## 8. RNG threading convention (recap — enforce uniformly)

- No global seed. `set_seed(seed)` (`Pretrain/utils.py`, `Finetuning/...`) becomes a function that
  **returns** a `jax.random.PRNGKey(seed)`; keep the name and the `random.seed`/`np.random.seed` calls
  (for numpy-side dataset shuffling) but add `return jax.random.PRNGKey(seed)`. Callers thread the key.
- Every stochastic op gets a key (see table in §9). Split once per consumer: `rng, k1, k2 = jax.random.split(rng, 3)`.
- Samplers (`sample_reverse_sde`, `sample_euler_karras`, `sample_euler_karras2`, `sample_dpm_*`,
  `sample_ddim`, `sample_euler_karras_replan`): accept `*, rng=None`; split a fresh subkey for the initial
  noise and inside each step that injects noise. The number of `split` calls should match the number of
  stochastic draws so two runs with the same key reproduce.
- Dropout: pass `rngs={'dropout': dropout_key}` to `.apply(...)`; modules call
  `nn.Dropout(rate)(x, deterministic=not train)`.
- Linen `.init(init_rng, *example_inputs)` consumes a key for parameter init.

---

## 9. Tensor-ops cheat-sheet

| torch | jax |
|---|---|
| `torch.tensor(x)` / `torch.as_tensor` | `jnp.asarray(x)` (or `np.asarray` for host data) |
| `.to(device)`, `.cuda()`, `.cpu()` | **drop** (JAX places automatically); `jax.device_put(x)` only if explicitly needed |
| `.item()` | `float(x)` / `x.item()` (works on jax scalars; forces host sync) |
| `.numpy()`, `.detach().cpu().numpy()` | `np.asarray(x)` |
| `.view(...)`, `.reshape(...)` | `x.reshape(...)` |
| `.permute(dims)` | `jnp.transpose(x, dims)` |
| `.transpose(d0, d1)` | `jnp.swapaxes(x, d0, d1)` |
| `.squeeze(d)` / `.unsqueeze(d)` | `jnp.squeeze(x, d)` / `jnp.expand_dims(x, d)` (or `x[..., None]`) |
| `torch.cat([...], dim=d)` | `jnp.concatenate([...], axis=d)` |
| `torch.stack([...], dim=d)` | `jnp.stack([...], axis=d)` |
| `torch.where(c, a, b)` | `jnp.where(c, a, b)` |
| `torch.clamp(x, lo, hi)` / `.clamp(min=, max=)` | `jnp.clip(x, lo, hi)` |
| `torch.einsum('...', a, b)` | `jnp.einsum('...', a, b)` |
| `torch.exp/log/sqrt/abs/sum/mean/min/max` | `jnp.*`; `dim=` → `axis=`; `.values`/`.indices` on min/max → use `jnp.min`/`jnp.argmin` separately |
| `x.sum(dim=-1)`, `x.mean(dim=d)` | `x.sum(axis=-1)`, `x.mean(axis=d)` |
| `torch.linspace`, `torch.arange` | `jnp.linspace`, `jnp.arange` |
| `torch.zeros/ones/full/zeros_like` | `jnp.zeros/ones/full/zeros_like` |
| `torch.randn(shape, device=)` | `jax.random.normal(key, shape)` |
| `torch.rand(shape)` | `jax.random.uniform(key, shape)` |
| `torch.randint(lo, hi, shape)` | `jax.random.randint(key, shape, lo, hi)` |
| `torch.randperm(n)` | `jax.random.permutation(key, n)` |
| `torch.normal(mu, std)` | `mu + std * jax.random.normal(key, mu.shape)` |
| `torch.multinomial(p, k)` | `jax.random.categorical(key, jnp.log(p), shape=...)` |
| `F.softplus`, `F.softmax`, `logsumexp` | `jax.nn.softplus`, `jax.nn.softmax`, `jax.scipy.special.logsumexp` |
| `torch.distributions.Normal/Beta/...` | `distrax.Normal` / `distrax.Beta` / etc. (`.log_prob`, `.sample(seed=key)`, `.mode()`) |

### In-place ops & indexing (JAX arrays are immutable)
| torch (in-place) | jax (functional) |
|---|---|
| `x[i] = v` | `x = x.at[i].set(v)` |
| `x[i] += v` | `x = x.at[i].add(v)` |
| `x[mask] = v` | `x = jnp.where(mask, v, x)` |
| `x.clamp_(lo, hi)` | `x = jnp.clip(x, lo, hi)` |
| `x.normal_()` | `x = jax.random.normal(key, x.shape)` |
| advanced index read `x[idxs]` | unchanged: `x[idxs]` works |

`initial[:, 0, :d_s] = current_state` (real ODP `create_initial`) →
`initial = initial.at[:, 0, :d_s].set(current_state)`.

### dtype / broadcasting
- Default float is `float32` in both — fine. Keep explicit casts: `x.float()` → `x.astype(jnp.float32)`,
  `.long()` → `.astype(jnp.int32)`.
- Broadcasting rules are NumPy/JAX-identical to torch for the patterns used here. `t.unsqueeze(0)` →
  `t[None]`.

---

## 10. Save / load: `state_dict()`/`load_state_dict()` → `flax.serialization`

- Training-time save/restore of converted-in-JAX models: use `jax_utils.save_agent` / `restore_agent`
  (pickle of `flax.serialization.to_state_dict`) — mirrors fql. Keep the ODP filename/path conventions
  (`getName`, `get_CriticName`, `get_reward_name`, `save_to_finetuning`, the `Models/..._{step}.pkl`
  layout) byte-for-byte so the rest of the pipeline finds checkpoints.
- `torch.save(checkpoint, path)` / `torch.load(path, map_location=)` of converted models → replace with
  `flax.serialization.to_state_dict` + pickle (or `save_agent`).
- **Ingesting existing torch `.pt`/`.pkl` checkpoints**: this is a real concern (`get_pretrained_planner`,
  `get_pretrained_reward`, `get_kernel`, `get_critic_model` load pre-trained torch weights). The loader
  must map torch `state_dict` keys → the flax param tree. Mark these functions `checkpoint-loaded` and,
  unless instructed otherwise, implement a small key-remap helper (torch `layer.weight`→flax `kernel`
  **transposed** for Dense: torch Linear weight is `(out, in)`, flax Dense kernel is `(in, out)`, so
  `kernel = torch_weight.T`; `layer.bias`→`bias`; LayerNorm `weight`→`scale`, `bias`→`bias`). Do NOT
  silently change checkpoint formats. If unsure, leave a `# TODO(checkpoint-bridge):` and keep the public
  signature.

---

## 11. Ensembles: `nn.ModuleList` → `ensemblize` or python list

- `CriticEnsemble` (`nn.ModuleList([Critic(...) for _ in range(num_heads)])`), `EnsembleReward`,
  `EnsembleModel`, kernel ensembles: prefer `jax_utils.ensemblize(Critic, num_heads)` which vmaps params
  over a leading axis (fql's `Value` uses this for `num_ensembles`). Aggregate with `preds.mean(axis=0)` /
  `preds.min(axis=0)` (note: leading axis 0, where torch stacked on `dim=-1`).
- Keep the public `aggregate="mean"|"min"` argument and the constructor signature identical. The
  aggregation axis changes from torch `dim=-1` to jax `axis=0` because `ensemblize` stacks on the leading
  axis — adjust the reduction axis accordingly while keeping the *result* shape identical.
- Lists of *separately-checkpointed* kernels (`List[RobustTransitionKernel]` passed to
  `compute_log_density`, `compute_total_mahalanobis_score`) stay python lists of `(model_def, params)` —
  these are independently-loaded models, not a vmapped ensemble. Keep them as lists.

---

## 12. Control flow: jit-friendly vs python-side

- **Keep python `for` loops** for: diffusion sampler step loops (fixed `num_steps`), adjoint backward
  solves, rollout horizons, training epoch loops. This matches fql (its Euler flow sampler is a plain
  python `for i in range(flow_steps)`). Faithfulness first.
- Use `jax.lax.scan` / `jax.lax.fori_loop` ONLY where the body is trivially pure and it's an obvious win,
  and only if it doesn't change numerics. When in doubt, python loop.
- `jax.jit` the hot per-step update/loss (like fql's `@jax.jit def update`). Do NOT jit functions that
  call `env.step`, do file I/O, or build python data structures.
- Avoid data-dependent python `if` on traced arrays inside jitted code → use `jnp.where` / `jax.lax.cond`.
  Outside jit (rollout/orchestration), keep python control flow.

---

## 13. Data: datasets / DataLoader → numpy batching (fql `Dataset` style)

- ODP `torch.utils.data.Dataset` subclasses (`CriticDataset`, `RewardDataset`, `KernelDataset`,
  `PlannerDataset`, `test_dataset`, `CriticDataset_Reward`, etc.) and `DataLoader`/`cycle(dl)`:
  - Keep the dataset class name and `__init__` signature. Store fields as **numpy arrays** (not tensors).
  - Replace `__getitem__`/`DataLoader` iteration with fql-style `sample(batch_size)` returning a `dict`
    of numpy arrays (see `fql/utils/datasets.py::Dataset.sample`). `cycle(dl)` → a generator yielding
    `dataset.sample(batch_size)`.
  - Random index sampling uses `np.random.randint` (host-side, numpy) — fine to keep numpy RNG for data
    shuffling; only the *model* stochasticity needs jax keys.
- `Pretrain/Dataset.py` (`get_dataset`, `get_env`, `Planner_Processor`, the various `*Dataset` builders,
  normalization `SAStats`): this is **mostly numpy/gym glue with no torch autograd** — convert tensor
  fields to numpy, drop `.to(device)`, keep everything else. Flag low-risk.
- Convert dataset tensors to `np.ndarray`; convert to `jnp` only at the model boundary (inside jitted
  fns). Mirrors fql, which keeps datasets in numpy and lets jit handle the host→device transfer.

---

## 14. Per-file conversion checklist (run before declaring a file done)

1. Public API frozen? (class/fn names, arg order, defaults, constants identical; only `rng=` added at end
   where stochastic, marked `# API-CHANGE:` if anywhere else).
2. `torch`/`torch.nn`/`torch.optim`/`F` imports fully removed; standard header in place.
3. All `nn.Module` → linen; `forward`→`__call__`; conv axis handled (§3); inits chosen + documented (§4).
4. Every `randn/rand/randint/randperm/normal/dropout/multinomial` takes a threaded key (§8).
5. `.backward()/.step()/zero_grad()` gone; gradients via `jax.grad`/`apply_loss_fn`; `.detach()`→
   `stop_gradient`; `no_grad` handled (§6); custom autograd → `jax.vjp/jacrev/jvp/grad` (§7).
6. In-place writes → `.at[].set()/.add()`; `dim=`→`axis=`; `.cat`→`concatenate` (§9).
7. Save/load via `flax.serialization`/`save_agent`; checkpoint-bridge TODO left if torch ckpt ingested (§10).
8. Intra-repo ODP imports unchanged; file still imports the same names it did before.
9. `python3 -c "import ast; ast.parse(open(FILE).read())"` passes; single quotes; ≤120 cols.

---

## 15. Quick reference: which files hit which hard cases (see MANIFEST for full detail)

- **Conv axis (§3):** `Pretrain/Planners/Backbone/Dit.py`, `UNet.py`, `Planners/Backbone/utils.py`,
  `Rewards/nets.py` (any conv path).
- **Custom autograd / backprop-through-ODE (§7):** `Finetuning/adjoint_matching.py`,
  `acc_adjoint_matching.py`, `AM.py`, `traj_reward.py`, `traj_reward2.py`, `traj_reward3.py`,
  `Rewards/nets.py::compute_reward_gradients_per_sample`.
- **Samplers / RNG step loops (§8, §12):** `Pretrain/Planners/Backbone/Sampler.py`,
  `Finetuning/Rollout.py`.
- **Training loops → TrainState (§6):** `Planners/Backbone/Trainer.py`, `Critic/train_critic.py`,
  `Rewards/Reward_Backbone.py`, `Transition_Kernel/Kernel_Backbone.py`, `Finetuning/utils.py` (train_*),
  `Finetune_Backbone.py`.
- **Ensembles (§11):** `Critic/nets.py::CriticEnsemble`, `Rewards/nets.py::EnsembleReward/EnsembleModel`,
  kernel ensembles.
- **Checkpoint ingest (§10):** all `get_pretrained_*` / `get_planner` / `get_kernel` / `get_critic_model`.
- **Mostly numpy/no-torch (low risk):** `Pretrain/utils.py`, `Pretrain/Dataset.py` (tensor→np only).
