# The cube single-play (task 4) training combination — reconstructed from the original repo

The original ODP repo stores configs as **numbered variant scripts** (a scrapbook; most blocks are inside
`"""..."""`). The **cube single-play, task 4** result was produced by this specific combination of them
(all paths/values transcribed verbatim from `main`):

| Stage | Script (on `main`) | Env / task |
|---|---|---|
| pretrain (planner) | `Pretrain/pretrain_script4.py` | cube **single-play, task 4** (scripts 2/3/4/5 = tasks 2/3/4/5, identical hyperparams) |
| kernel (MoG) | `Pretrain/train_kernel_script2.py` (cube block) | cube **single**, task 4 |
| critic | `Pretrain/train_critic_script2.py` | cube **single-play, task 4** |
| reward | ⚠️ **no single-play reward pretrain script exists** (see note) | — |
| finetune | `Finetuning/finetune_script2.py` (the `env='cube'/'single-play'` block, ~L443) | cube **single-play, task 4** |

> ⚠️ **Reward gap:** `train_reward_script.py`'s only cube block is `specific='double'`. There is **no**
> cube-`single` reward pretrain script in the repo. So either (a) your teammate edited the reward script's
> `specific_dataset` to `'single'` (same hyperparams), or (b) the single-play reward checkpoint came from a
> separate run. You'll need to confirm with them. (The finetune below loads `reward_model_checkpoint=0`,
> i.e. it expects a pretrained single reward to already exist.)

---

## Per-stage hyperparameters (cube single-play, task 4)

### pretrain — `pretrain_script4.py`
```
SDETrainer(dataset='cube', specific='single-play', task_id=4, horizon=32,
           backbone='transformer', num_steps=1_000_000, batch_size=128, lr=2e-4, stride=1)
```

### kernel (MoG) — `train_kernel_script2.py`
```
train_mog_kernel(dataset='cube', specific='single', task_id=4, trajs=success_trajs,
                 batch_size=512, lr=1e-4, num_steps=5000, save_freq=1000,
                 ensemble_size=10, num_modes=10, num_hidden_layers=4, hidden_dim=514,
                 λ_reg=1e-3, noise_floor=5e-4)
```
(note: passes `trajs = load_success_trajs('cube','single-play', task_id=4, step=0)` — success rollouts.)

### critic — `train_critic_script2.py`  (differs from train_critic_script.py!)
```
train_critic(dataset='cube', specific='single-play', task_id=4,
             hidden_layers=4, hidden_dim=512, batch_size=256, num_steps=70000,
             gamma=0.99, horizon=32, lr=5e-5, min_lr=1e-6, tau=0.005,
             sigma=3.0, target_reward=80.0)
```
(uses dataset trajectories + `trajs_task4_success_0.pkl`.)

### reward — (single-play; hyperparams from the cube block of `train_reward_script.py`, run on `single`)
```
train_reward(dataset='cube', specific='single', task_id=4,
             hidden_layers=4, hidden_dim=512, batch_size=256, num_steps=100000, save_freq=100000,
             lr=1e-4, min_lr=5e-6, sigma=4.0, alpha=None, target_reward=300.0, traj_length=None)
```

### finetune — `finetune_script2.py` (cube single-play block, ~L443)
This is **substantially different** from the kitchen `finetune_script.py` default — note `critic=True`,
`offline=True`, `update_kernel=False`, `eta=0.0`, `diffusion_steps=10`, `continual_rollout=True`:

```
env='cube', specific='single-play', task_id=4
finetune_buffer_cutoff_length=100,  train_buffer_cutoff_length=200

AlphaConfig = AlphaSchedulerConfig(alpha_start=1.0, alpha_end=0.1, total_steps=300, decay=True)
AMConfig    = Acc_AdjointMatchingConfig(horizon=32, eta=0.0)
RWConfig    = RewardConfig(beta=1.0, min_log_prob=-110.0, quantile=0.999,
                           number_of_generated_plans=50, critic_gamma=0.99, explore=False)

TrainRewardConfig = Train_Reward_Config(hidden_layers=4, hidden_dim=512, batch_size=256,
                                        num_steps=30000, lr=5e-3, min_lr=5e-4, sigma=4.0,
                                        target_reward=500.0, task_id=4)
TrainKernelConfig = Train_Kernel_Config(batch_size=512, num_steps=5000, lr=1e-4, ensemble_size=10,
                                        num_hidden_layers=4, hidden_dim=514, type_kernel='mog',
                                        kernel_num_modes=10, kernel_noise_floor=5e-4, λ_reg=1e-3)
TrainCriticConfig = Train_Critic_Config(hidden_layers=4, hidden_dim=512, batch_size=256, num_steps=20,
                                        lr=1e-5, min_lr=1e-6, horizon=128, tau=0.005, gamma=0.99,
                                        data_conservation=True, momentum=0.1)

FinetuningConfig(
    planner_checkpoint=0, reward_model_checkpoint=0, kernel_model_checkpoint=0, critic_model_checkpoint=0,
    offline=True, critic=True, update_critic=True, kernel=True, update_kernel=False,
    buffer_size=200000, finetune_buffer_cutoff_length=100, train_buffer_cutoff_length=200,
    finetune_steps=90, finetune_rounds=30, diffusion_steps=10, karras_percent=0.1, Loss_Clip_percent=0.0,
    finetune_batch_size=32, finetune_batch_per_sample=8, finetune_lr=2e-5,
    initial_lam=0.05, eta_lam=0.5, gradient_accumulate_every=1, update_lambda_every=1,
    reward_scaling_factor=150, MaxEnt=False, Entropy_Scaling_Factor=0.5,
    rollout_length=4000, rollout_num_envs=8, continual_rollout=True, chunk_size=31, num_rollout_processes=8)
```

---

## What this means vs. what I had wired before

My runner's finetune was mirroring the **kitchen** default (`critic=False`, `update_kernel=True`,
`eta=0.8`, `diffusion_steps=30`, etc.). The real **cube-single** finetune is different on several axes:

- **`critic=True`** → uses `TotalReward_Critic` (loads + uses the critic; the critic-conditioned reward
  gradient path). My earlier statement "critic is off by default" was about the *default*; this specific
  config turns it **on**.
- **`offline=True`** → after each round's rollout the loop `continue`s, changing which models retrain.
- **`update_kernel=False`** → kernel is **not** retrained per round (only used as a fixed constraint).
- **`eta=0.0`** → deterministic Euler sampling (no injected noise).
- Smaller/different: `diffusion_steps=10`, `finetune_steps=90`, `finetune_rounds=30` (→ per_round=3),
  `finetune_batch_size=32`, `finetune_batch_per_sample=8`, `finetune_lr=2e-5`, `continual_rollout=True`,
  `rollout_length=4000`, `rollout_num_envs=8`.

So to reproduce the teammate's cube-single result (for the JAX-vs-torch comparison), the runner's
`--variant single` finetune should use **exactly this config**, not the kitchen default.

**Next step:** wire this cube-single config into `run_cube_pipeline.py` (and resolve the reward-pretrain
gap with your teammate). This will exercise the `critic=True` / `offline=True` code path, which the smoke
run (critic=False) did not — so expect to shake out a couple more runtime issues on that path.
