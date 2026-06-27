# ODP — Training Pipeline (how each part is trained, and how finetune works)

This document describes the **training logic of the original ODP algorithm**, grounded in the actual
code (`Pretrain/`, `Finetuning/`). It explains the order in which the four components are pretrained, and
exactly what the finetune stage does (and which models it updates).

> TL;DR: four components are **pretrained independently and offline** (planner, transition kernel, reward,
> critic), then the **planner** is fine-tuned online with **adjoint matching**, while the kernel and reward
> are periodically re-trained to track the changing policy. The **critic is OFF by default** during finetune.

---

## 0. The cast (what each model is)

| Component | Class / file | What it models |
|---|---|---|
| **Planner** | `DiT1d` (`Pretrain/Planners/Backbone/Dit.py`) | A **diffusion model** over trajectories τ = (state‖action per step), length = `horizon`. Generates plans. |
| **Transition kernel** | `MoGTransitionKernel` / `RobustTransitionKernel` (`Pretrain/Transition_Kernel/Kernel_Net.py`) | Environment dynamics **p(s′ \| s, a)** (ensemble). Used as an in-distribution **constraint**. |
| **Reward** | `SimpleReward` (`Pretrain/Rewards/nets.py`) | A differentiable reward **r(s, a)**. |
| **Critic** | `Critic` (`Pretrain/Critic/nets.py`) | A value function V/Q. Optional; off by default in finetune. |

---

## 1. Pretraining (stages 1–4) — independent & offline

Each is trained on its own from the offline dataset and saved to a checkpoint. They do **not** depend on
each other, so order among them doesn't matter (the pipeline runs planner → kernel → reward → critic).

### 1a. Planner — `Pretrain/pretrain_script.py` → `SDETrainer.train()`
- Builds `PlannerDataset`: sliding windows of length `horizon` over the offline trajectories, each window
  a `(state‖action)` sequence; also computes/saves the **normalization stats** (`Planner_Processor`).
- Trains `DiT1d` by **score matching** for a VP diffusion: predict the analytic Gaussian score
  `target = -(x_t - α·x_0) / (σ² )` (`SDETrainer._loss`), weighted by `weight_type` (default `sigma2`),
  optimizer AdamW + cosine LR, plus an **EMA** copy of the weights.
- Saves the planner (the EMA params) — this is the object that finetune later loads and updates.

### 1b. Transition kernel — `Pretrain/train_kernel_script.py` → `train_mog_kernel(...)`
- Builds `(s, a, s′)` transitions from the dataset, trains an **ensemble of MoG kernels** by minimizing
  mixture NLL (`mog_nll`) + a disagreement regularizer.
- Saves the ensemble + normalization stats. Later gives the finetuner a **log-density** of each generated
  transition (how "in-distribution / physically plausible" a generated plan is).

### 1c. Reward — `Pretrain/train_reward_script.py` → `train_reward(...)`
- Trains `SimpleReward` r(s,a) to regress the (processed) reward (Huber loss, optax AdamW + cosine).
- Saves the reward net + stats. Used as the **differentiable reward** during finetune.

### 1d. Critic — `Pretrain/train_critic_script.py` → `train_critic(...)`
- Trains a value/critic via n-step returns + a target network (Polyak `tau`). Saves the critic.
- **Only needed if finetune runs with `critic=True` (not the default).**

Each stage also writes a **finetuning copy** of its checkpoint (`save_to_finetuning`, saved at *step 0*)
under `Finetuning/{Planners,Kernels,Rewards,Critics}/...`, which is what the finetuner loads.

---

## 2. Finetune (stage 5) — `Finetuning/finetune_script.py` → `OnlineFinetuner.finetune_planner()`

**Goal:** fine-tune the pretrained **planner** so its generated trajectories get **higher reward** while
**staying in-distribution** (kernel constraint). The method is **adjoint matching**
(`Finetuning/acc_adjoint_matching.py` :: `Acc_AdjointMatchingFineTuner`).

### 2a. Setup (`OnlineFinetuner.__init__`)
- Loads the four pretrained checkpoints (planner / kernel / reward / [critic]).
- Builds `reward_model = TotalReward(...)` (`Finetuning/traj_reward.py`): wraps the frozen reward + kernel
  (+ critic) and exposes the **(reward, input-gradient)** entry points used by adjoint matching.
- Builds the `AMFineTuner` with a **frozen** `old_score_net` (the pretrained planner) and a **trainable**
  `new_score_net` (initialized from it).

### 2b. The objective
For a generated trajectory, per step:

```
total_reward = Σ_i  (1/H) · γ^i · r(s_i, a_i)   −   λ · (1/(H-1)) · constraint_i
constraint_i = softplus( min_log_prob − mean_kernel_log_density(s_i, a_i, s_{i+1}) )
```

i.e. **maximize reward** minus **λ × a penalty for leaving the data manifold** (the kernel's
log-density). `λ` is a Lagrange multiplier, updated online.

### 2c. The per-round loop (`for step in range(finetune_rounds)`)

Each round does, in order:

1. **Adjoint-matching update of the planner** — `AMFineTuner.finetune_planner(...)` runs
   `per_round_steps` AM steps. Each step (`step()` → `sample_Traj_karras()`):
   - samples a trajectory with the current planner (Euler–Karras diffusion sampling);
   - computes the objective above (reward via `reward_model`, constraint via the kernel);
   - solves the **lean-adjoint backward** along the diffusion chain (per-step `jax.jvp` of the frozen
     `old_score_net`) to get the gradient of the objective w.r.t. the planner;
   - applies the **adjoint-matching loss** to update `new_score_net` (AdamW + grad clip);
   - periodically updates `λ`; maintains an **EMA** planner and saves it as the new planner checkpoint.
2. **Rollout** — `rollout_parallel2(...)` runs the updated planner in the real environment to collect new
   trajectories into the buffers (and reports score / success rate).
3. **Re-train reward** — `if update_reward:` → `train_reward[_ensemble](...)` on the new buffer (the policy
   moved, so the reward model is refreshed on the new data distribution).
4. **Re-train kernel** — `if self.config.kernel and self.config.update_kernel:` → `train_kernel[_mog](...)`
   + recompute the constraint threshold (refresh the dynamics model / in-distribution check).
5. **Re-train critic** — `if self.config.critic and self.config.update_critic:` → `train_critic(...)`.
   **Skipped by default** (`critic=False`).

> **Code note:** in the original loop there is *also* a kernel/reward re-train block **inside** the
> `is_main_process` section that is **commented out** (`Finetune_Backbone.py`, the `"""..."""` around the
> per-iteration `train_kernel`); the active kernel re-train is the `update_kernel` block that runs once per
> round (step 4 above). The port preserved this commented-out state.

---

## 3. So — which models does finetune actually update?

Per the **original defaults** (`FinetuningConfig`):

| Model | Updated during finetune? | Flag (default) |
|---|---|---|
| **Planner** | ✅ yes — the whole point (adjoint matching) | always |
| **Kernel** | ✅ yes — re-trained each round | `kernel=True`, `update_kernel=True` |
| **Reward** | ✅ yes — re-trained each round (when exploring / `update_reward`) | (gated by `offline` / `update_reward`) |
| **Critic** | ❌ no — **off by default** | `critic=False` |

So it is **planner + kernel + reward** by default, **not** critic. (This is the opposite of "only critic
and planner".) The reasoning is standard online finetuning: once the planner changes, the dynamics model
(kernel) and reward model are refreshed on the new policy's data so they keep giving valid constraint /
reward signals.

### Relevant default knobs (`FinetuningConfig` / `Acc_AdjointMatchingConfig`)

```
offline = False          critic = False           kernel = True / update_kernel = True
update_critic = True      finetune_rounds = 10     finetune_steps = 1_000_000
diffusion_steps = 30      eta = 0.8                num_karras = 2
per_round_steps = finetune_steps // finetune_rounds
lam = 0.01 (initial)     eta_lam = 0.001          reward_scaling_factor = 100000
```

---

## 4. How this maps to the runner (`run_cube_pipeline.py`)

The pipeline runs the five stages in order on cube (default `--variant double`, or `--variant single`):

```
pretrain → kernel → reward → critic → finetune
```

with the same component architectures threaded into the finetune `RewardConfig` so the checkpoints load.
The finetune loads each component's *step-0* finetuning checkpoint; `task_id` flows via
`train_reward_config.task_id`. To validate wiring quickly, `scripts/smoke.sh` uses tiny step counts
(and a 1-round, short-rollout finetune).

> If you want to validate **only** the planner's adjoint-matching path (skip the per-round kernel/reward
> re-training, which is the heavy part), set `kernel=False` (and keep `critic=False`) in the finetune
> `FinetuningConfig`. Ask and this can be exposed as a runner flag.
