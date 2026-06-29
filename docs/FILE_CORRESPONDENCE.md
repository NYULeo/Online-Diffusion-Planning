# JAX ⇄ Torch file correspondence (cube single-play / task4)

Two repos, **same directory layout**:
- **JAX port:** `~/ODP` (on the GPU box: `~/ODP-jax`)
- **Torch original:** `~/Online-Diffusion-Planning`

> Naming gotcha: torch's *active* finetune modules are the **numbered** variants
> (`Finetune_Backbone2.py`, `traj_reward3.py`) — there are older unnumbered/`2` copies left around.
> The JAX port uses the **unnumbered** names. The active set is what `finetune_script2.py` imports.

---

## Entry points / orchestration

| Role | JAX | Torch |
|---|---|---|
| Whole pipeline (5 stages, one command) | `run_cube_pipeline.py` | *(none — separate scripts per stage)* |
| Finetune entry + cube/single config | `run_cube_pipeline.py :: stage_finetune` | `Finetuning/finetune_script2.py` (cube/single-play block, the only un-commented `if __name__` config) |
| Per-stage launch (torch) | the `--stages` flag | `Pretrain/pretrain_script.py`, `Pretrain/train_kernel_script.py`, `Pretrain/train_reward_script.py`, `Finetuning/train_critic_script.py` |

---

## 1. Planner — diffusion pretrain
| Component | JAX | Torch |
|---|---|---|
| Trainer (SDETrainer) | `Pretrain/Planners/Backbone/Trainer.py` | same path |
| Network **DiT1d** | `Pretrain/Planners/Backbone/Dit.py` | same path |
| Sampler `sample_euler_karras` | `Pretrain/Planners/Backbone/Sampler.py` | same path |
| Backbone utils (schedules/EMA) | `Pretrain/Planners/Backbone/utils.py` | same path |

## 2. Transition kernel — MoG
| Component | JAX | Torch |
|---|---|---|
| Train + `compute_log_density_mog` | `Pretrain/Transition_Kernel/Kernel_Backbone.py` | same path |
| Network **MoGTransitionKernel** | `Pretrain/Transition_Kernel/Kernel_Net.py` | same path |

## 3. Reward
| Component | JAX | Torch |
|---|---|---|
| Train (`train_reward`) | `Pretrain/Rewards/Reward_Backbone.py` | same path |
| Network **SimpleReward** (active = 4-arg, **no** final ReLU) | `Pretrain/Rewards/nets.py` | same path |

## 4. Critic
| Component | JAX | Torch |
|---|---|---|
| Network **Critic** (active = 3-arg, ReLU commented out) | `Pretrain/Critic/nets.py` | `Pretrain/Critic/nets.py` |
| Offline-dataset train (`train_critic`, `CriticDataset`) | `Pretrain/Critic/train_critic.py` | **`Finetuning/utils.py`** (torch keeps it in utils) |
| Planner-rollout train (`train_critic_with_planner2`, normalized target) | `Finetuning/utils.py` | `Finetuning/utils.py` |

## 5. Finetune — adjoint matching (the core)
| Component | JAX | Torch |
|---|---|---|
| Orchestrator: round loop (AM → rollout → per-round critic retrain) — `OnlineFinetuner.finetune_planner` | `Finetuning/Finetune_Backbone.py` | **`Finetuning/Finetune_Backbone2.py`** |
| AM step (adjoint-matching diffusion finetune) — `Acc_AdjointMatchingFineTuner` | `Finetuning/acc_adjoint_matching.py` | `Finetuning/acc_adjoint_matching.py` |
| Reward+constraint+critic aggregation — `TotalReward_Critic.predict` / gradient | `Finetuning/traj_reward.py` | **`Finetuning/traj_reward3.py`** |
| Configs (`FinetuningConfig`, `Train_*_Config`, `RewardConfig`, `AlphaSchedulerConfig`) | `Finetuning/Finetune_Backbone.py` + `traj_reward.py` + `utils.py` | `Finetuning/Finetune_Backbone2.py` + `traj_reward3.py` + `utils.py` |

## 6. Rollout / evaluation + shared utils
| Component | JAX | Torch |
|---|---|---|
| Rollout (in-loop `rollout_parallel2` + eval `rollout`) | `Finetuning/Rollout.py` | `Finetuning/Rollout.py` |
| Shared utils: `rollout_parallel2`, `get_planner/reward/kernel/critic`, `save_*`, `KernelConfig`, `train_critic_with_planner2` | `Finetuning/utils.py` | `Finetuning/utils.py` |

---

## The 3 places that matter most for logic parity (where to focus the review)
1. **`TotalReward_Critic.predict`** — JAX `traj_reward.py` ⇄ torch `traj_reward3.py`.
   The per-step reward+constraint sum and the terminal critic term `Σ γ^i·r_i − λ·Σc_i + γ^(H-1)·v + λ·δ`.
2. **`OnlineFinetuner.finetune_planner` round loop** — JAX `Finetune_Backbone.py` ⇄ torch `Finetune_Backbone2.py`.
   Offline branch: rollout for metric, then per-round `train_critic_with_planner2` (critic retrain), then continue.
3. **`train_critic_with_planner2`** — both `Finetuning/utils.py`.
   Normalized target = running-mean/std of `clamp(±10)/5`-scaled reward over kernel-feasible planner plans.

## JAX-only implementation notes (NOT logic differences)
- Single-device shim replaces torch's 8-GPU `accelerate` (so rollout prints 1 `success rate:` line, not 8; the metric is the same — compare `Average Success Rate`).
- `jit` added to hot paths (AM step, `compute_log_density_mog` feasibility) — compiled-once, identical numbers.
- `ODP_PREDICT_DEBUG=1` prints the `predict()` decomposition (reward_net_sum / critic_v / critic_term / constraint).
