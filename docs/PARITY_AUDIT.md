# Parity Audit — JAX port (`~/ODP`) vs Torch original (`~/Online-Diffusion-Planning`)

Function-by-function diff of the active **cube / single-play / task4** path (offline=True, critic=True, MoG kernel, eta=0). Done via 8 parallel comparison agents. Each item is classified DIVERGENCE (changes numerical results / training behavior) vs BENIGN (jnp↔torch idiom: jit, vmap, RNG threading, `.at[].set`, device, eager — same numbers).

## TL;DR — real divergences ranked

| # | Severity | File | What | Fix |
|---|---|---|---|---|
| 1 | **HIGH** | `Pretrain/Planners/Backbone/Sampler.py` | `karras_beta_schedule` β(t) is a **closed-form** `2ρσ_k/(1+σ_k²)` in JAX vs torch's **finite-difference** `Δσ²/((1−σ²)Δt)`. Changes the diffusion sampler used in **rollout + planner2 plan-gen**. Also inconsistent with JAX's *own* `utils.py` copy (which is finite-diff). | Restore finite-diff β in Sampler.py |
| 2 | **HIGH** | `Finetuning/Finetune_Backbone.py` | JAX never passes `old_planner_checkpoint` to `AMFineTuner.finetune_planner`, so the adjoint **reference net** (`old_score_net`) stays at the pretrained planner all 30 rounds. Torch passes `step*per_round_steps` → reloads it to the previous round's planner each round. | Pass `old_planner_checkpoint=step*per_round_steps` |
| 3 | **MED-HIGH** | `Finetuning/acc_adjoint_matching.py` | Per-round LR: JAX rebuilds cosine from `config.finetune_lr` (2e-5) **every round** → stays ~2e-5; torch continues from the **live decayed** lr → decays toward ~0 by round 30. | Read live lr from opt state (like torch) |
| 4 | **MED** | `Finetuning/Finetune_Backbone.py` | JAX runs `gather_and_sync_trajs_and_buffer` **every round** (rebuilds `PlannerDataset` from online rollouts); torch gates it behind `if not offline` → planner-conditioning dataset stays frozen in offline mode. | Gate behind `not self.config.offline` |
| 5 | LOW | `Finetuning/Rollout.py` | `__main__` eval default checkpoint = **90** (JAX) vs **39** (torch). Env-overridable (`ODP_EVAL_*`). | Set to the checkpoint you actually compare |
| 6 | LOW/benign | all nets | Weight init: JAX `default_init` (variance_scaling fan_avg, zero bias) vs torch Kaiming-uniform (fan_in, nonzero bias). From-scratch only → different seed, washes out. | none needed (note for checkpoint reuse) |
| 7 | LOW | `Pretrain/Transition_Kernel/Kernel_Backbone.py` | Kernel optimizer `optax.adamw` (decoupled wd) vs torch `Adam(weight_decay=1e-5)` (L2). Tiny magnitude. | optional: match |

---

## HIGH — details + fixes

### 1. `karras_beta_schedule` β formula (Sampler.py)
- JAX `Pretrain/Planners/Backbone/Sampler.py:207-224`: `beta = 2*rho*sigma_k/(1+sigma_k**2)` (analytic).
- Torch `Pretrain/Planners/Backbone/Sampler.py:81-106`: `beta = diff(sigma²)/(1-sigma²[:-1])/diff(t)`, last element padded (finite-difference of the VP variance).
- `t` and `sigma` are algebraically identical on both; **only β differs**. β feeds drift `-0.5·β·x`, score step `−β·score·dt`, and noise `η·√(β·−dt)` for the first `num_karras` steps of `sample_euler_karras`.
- Blast radius: **rollout** (`Finetuning/Rollout.py`) + **planner2 plan generation** (`train_critic_with_planner2`). NOT the AM step (that imports the *other*, faithful, `karras_beta_schedule` from `utils.py:1791`). So JAX is even internally inconsistent (AM samples plans with finite-diff β, rollout/planner2 with closed-form β).
- Fix: replace the closed-form β in `Sampler.py` with the finite-difference formula (copy from `utils.py:1791` / torch `Sampler.py:81`).

### 2. Stale adjoint reference net — `old_planner_checkpoint` not passed (Finetune_Backbone.py)
- Torch `Finetune_Backbone2.py:633`: `self.AMFineTuner.finetune_planner(dataloader, reward_model, step+1, old_planner_checkpoint = step*per_round_steps)`.
- JAX `Finetune_Backbone.py:723`: `self.AMFineTuner.finetune_planner(dataloader, self.reward_model, step+1)` — **arg omitted**.
- `AMFineTuner.finetune_planner` (both, ~line 752/682): `if old_planner_checkpoint is not None: self.reset_old_score_net(old_planner_checkpoint)` → reloads `old_score_net` (the frozen reference used in the lean-adjoint JVP) from that planner checkpoint.
- Effect: torch's adjoint is computed relative to the **previous round's** planner (incremental refinement); JAX's relative to the **pretrained** planner for every round. Different AM gradient direction in rounds 2–30.
- Fix: in `Finetune_Backbone.py:723`, pass `old_planner_checkpoint = step * self.config.AMConfig.per_round_steps` (note: `step` not `step+1`, matching torch — torch passes `step*per_round_steps` while `round=step+1`).

### 3. Per-round LR restart (acc_adjoint_matching.py)
- JAX `acc_adjoint_matching.py:767`: `set_optimizer_and_scheduler(new_lr=self.config.finetune_lr, new_steps=finetune_total_steps-(round-1)*per_round_steps)`.
- Torch `acc_adjoint_matching.py:691`: `set_optimizer_and_scheduler(new_lr=self.optimizer.param_groups[0]['lr'], new_steps=...)`.
- With 90 total steps / 3 per round / 30 rounds: torch follows one cosine 2e-5→~0 across the whole run; JAX resets to 2e-5 each round (3 steps ≈ no decay) → trains at ~2e-5 in late rounds. The port's own comment acknowledges this.
- Fix: read the live decayed lr from the optax opt-state (or track it) and pass that as `new_lr`, matching torch.

### 4. Offline planner-dataset growth (Finetune_Backbone.py)
- JAX `Finetune_Backbone.py:763`: `update_reward = self.gather_and_sync_trajs_and_buffer(trajs)` — called **unconditionally** each round; it extends `Finetune_Buffer`/`Train_Buffer`/`Train_Kernel_Buffer` with the round's rollouts and **rebuilds `self.PlannerDataset`**.
- Torch `Finetune_Backbone2.py:671-672`: `if(not self.config.offline): update_reward = self.gather_and_sync_trajs_and_buffer(trajs)` — skipped in offline mode → `PlannerDataset` stays the initial offline dataset.
- Effect: the AM step samples planner-conditioning `s0` from a dataset that **grows with online rollouts** in JAX vs a **frozen offline** dataset in torch. Shifts the conditioning distribution across rounds.
- Fix: gate the call behind `if not self.config.offline:` (matching torch).

---

## Verified FAITHFUL (no divergence) — safe to skip in the review

- **`TotalReward_Critic.predict` / `__call__`/`forward`** (`traj_reward.py` vs `traj_reward3.py`): every term identical — `Σ γ^i·r_i − λ·Σc_i`, terminal `r_{H-1}` computed-but-commented-out on both, terminal `γ^(H-1)·v`, `+λ·δ`, the `1/max(std,floor)` input-grad rescales, `get_c`, `sigmoid`, processors. (Only conditional `__init__` net-dim *inference from checkpoint* — resolves to the same nets for our config.)
- **Critic target** (`train_critic_with_planner2` + offline `train_critic`): clamp(±10)/5, γ exponents, running mean/std normalization (α=0.99), Huber δ=1, Polyak τ=0.005, Bellman `r+γ^H·q_next·(1−done)`, n-step return — all identical. **0 divergences.**
- **Reward net + training**: `SimpleReward` active class = `[Linear,LN,SiLU]×(1+hidden_layers)` + bare `Linear(_,1)` (final ReLU commented out on both); `boost_signal` (×target_reward), `gaussian_filter1d(mode='nearest', truncate=200/sigma)`, SAStats norm, Huber loss, AdamW — all identical. **0 divergences.**
- **MoG kernel math**: architecture, mu/log_std/logits split, softplus log_std floor + max clamp, softmax weights, noise_floor added to **variance**, per-mode Gaussian `−0.5(mahal + d·log2π + Σlog var)`, mixture `logsumexp(+log(w+1e-8))`, ensemble `logsumexp−log(K)`, disagreement penalty, all hyperparams — identical.
- **DiT1d architecture**: in_dim=d_s+d_a, emb=128, d_model=256, heads=4, depth=2, adaLN-Zero, FinalLayer zero-init — identical. **SDE training loss** + EMA(0.9999) + AdamW + all pretrain hyperparams — identical.
- **AM step math**: a0 = −reward_scaling/α/reward_std·grad, lean-adjoint JVP recursion, loss `((Δv·2/σ+σ·adjoint)²).mean()` with `min(.,rsf²·1.6)` clip, λ update, AlphaScheduler, EMA, grad-clip 1.0 — identical.
- **Rollout/eval logic**: success criterion (`sum(rewards)==1.0` eval; `rewards[-1]==1.0` in-loop), chunk_size best-of-28 retry list (identical order), eval sampling `steps_T=10/num_karras=1/eta=0`, 50 seeds, EMA weights, `reward_processor`/`get_normalized_score`/`check_success_rate` — identical (except the closed-form β from #1 flowing into the sampler).

## Benign idioms (everywhere, not divergences)
Single-device accelerator shim (vs 8-GPU `accelerate`), `jnp`↔`torch`, `optax.chain`↔explicit optimizer+scheduler+clip, `@jax.jit`/`vmap`, threaded `rng=` vs implicit RNG, `TrainState`/`target_update` vs `nn.Module`/manual EMA, host-side `sample()` vs `DataLoader`, `SyncVectorEnv` vs `AsyncVectorEnv`, pickle vs torch.save.

---

## Single-device performance adaptations (NOT result-changing — for the perf discussion)

The torch original splits work across 8 GPUs (rollout 1 env/GPU; AM 32 trajs/GPU all-reduced to 256; critic plan-gen pipelined eagerly). The single-GPU JAX port serialized all of it → ~30 min/round vs torch's 5–10. Three changes restore throughput **without changing what the model learns**:

| # | Change | File | Fidelity |
|---|---|---|---|
| A | **Batched diffusion sampling** `sample_euler_karras_batch` used in critic plan-gen (`_generate_feasible_plans`) + rollout (`rollout_parallel2`). ~5k–10k sequential batch-1 diffusions/round → tens of batched calls. | `Sampler.py`, `utils.py` | **Distribution-identical** sampling (same karras schedule / frozen DiT forward — per-sample, no batch mixing / Euler / clip; only the init-noise RNG is drawn once as `(B,..)` → i.i.d. plans from the *same* distribution, not bit-identical RNG). The critic is a value estimator over a plan *distribution*, so its learned value function is unchanged. |
| A′ | Critic **feasibility** batched by flattening `n·(H-1)` transitions into one `compute_log_density_mog`. | `utils.py` | **Bit-identical** — transitions are independent in the MoG density, so flatten == per-plan `is_plan_feasible`. |
| B | **AM gradient micro-batching** (`ODP_AM_MICRO`, default 32): accumulate `jax.grad` over chunks of m trajs, weighted by `chunk_size/N`. Restores the faithful **256-traj** gradient on one GPU (which OOMs in a single backward — the reason the smoke ran `--bs 4` = 1/8 gradient, leaving success=0). | `acc_adjoint_matching.py` | **Bit-identical** to torch's 8-GPU all-reduce mean over the full batch (mean-of-size-weighted-chunk-means == overall mean); trajs/adjoints already materialized → no re-sampling. |

Run config: smoke (speed) `--bs 4` (32 trajs, micro inactive); faithful run `--bs 32 --bps 8 --rounds 30` (256 trajs, auto micro-batched). Do **not** set `ODP_VMAP=1` at 256 trajs (the whole-batch vmapped path OOMs; micro-batching is the memory-safe route).
