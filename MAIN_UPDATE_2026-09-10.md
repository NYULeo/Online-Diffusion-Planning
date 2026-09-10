# Main update: first-hit critic consistency and Hydra configuration

This document describes the changes applied to the Cube Single pipeline on 2026-09-10.

## Invariant kept

The reward model is still `SimpleReward(s, a)`. Its architecture, state/action inputs, Huber training loss, and Cube Single smoothing setting (`sigma: 4.0`) were not replaced by a state-only or next-state reward.

OGBench Cube rewards keep their native convention:

- `-1`: task not yet completed;
- `0`: successful transition.

No `-1/0 -> 0/1` relabeling or reward multiplication is performed. The configured reward target is `1.0`, so the native scale is retained.

## Training behavior changes

### Dataset

- Cube Single trajectories are split at natural episode boundaries and truncated at the first `reward == 0` inside each episode.
- Data after the first success is discarded; it is not turned into a new trajectory.
- The OGBench training and validation datasets are now selectable with `split: train|val`.
- Planner and reward training use `train`; validation uses the independent OGBench `val` split.

This makes success an absorbing endpoint and prevents a state visited after success from receiving an unrelated negative value target.

### Reward model

- Training is unchanged: `r_hat = reward_net(s, a)` with Smooth-L1 loss.
- Reward validation now uses the OGBench validation split and the normalization statistics fitted on training data.
- Validation no longer overwrites the saved training statistics.
- W&B records train loss, learning rate, validation loss, and prediction distribution.

### Critic-1: offline initialization

Critic-1 is now a supervised Monte-Carlo value fit; it no longer bootstraps from a randomly initialized target critic.

For learned rewards clipped to `[-1, 0]`, define the soft continuation probability

```text
c_t = mask_t * clip(-r_hat_t, 0, 1)
G_t = r_hat_t + gamma * c_t * G_(t+1)
G_T = r_hat_T
```

The network is trained with

```text
L_critic1 = SmoothL1(V_phi(s_t), symlog(G_t / value_scale)).
```

Every state before the first hit receives a target, including states from trajectories shorter than the old `horizon=200`. This removes the old train/test mismatch and the case where an exact-horizon trajectory produced zero samples.

The saved checkpoint is the trained online critic itself. `lam`, `horizon`, and `tau` were removed from the Critic-1 YAML section because they are bootstrap parameters and no longer affect this stage.

Critic-1 validation reconstructs exactly the same MC targets on held-out trajectories and reports encoded Smooth-L1 loss, decoded raw MAE, raw bias, prediction mean, and target mean.

### Critic-2: planner-conditioned update

- Critic-2 remains the planner-generated, bootstrapped stage (`train_critic_with_planner7`).
- Kernel-infeasible plans are dropped. The former fallback that silently accepted every infeasible plan was removed.
- The learned `[-1,0]` reward is treated as a soft first-hit signal. Rewards after the first hit and bootstrap values after the first hit are suppressed.
- Reward and critic targets use the same `Q_scale` convention before `symlog` encoding.
- The existing compact CPU-row gather is retained, avoiding the previous multi-GPU `all_gather_object` 25+ GiB allocation.

### Finetuning objective

`TotalReward_Critic.predict()` and `forward()` now use the same first-hit return:

```text
J = sum_t gamma^t * survival_before_t * r_hat_t
    + gamma^H * survival_after_H * Q_scale * symexp(V(s_H))
    - lambda * constraint.
```

The analytical gradient returned by `forward()` is computed from this same objective, including the survival factor. Reward evaluation remains `reward_net(s, a)`.

### Bellman diagnostics

The multi-horizon probe now matches Critic-2 training:

- same reward clipping/first-hit semantics;
- same `Q_scale` decoding;
- same kernel feasibility threshold;
- same suppression of bootstrap after success;
- validation trajectories instead of training trajectories.

W&B keys are under `bellman_prob/*`.

## Configuration and launch changes

- The hkw Hydra structure was ported to main.
- All six pipeline entrypoints read their own section from `Finetuning/conf/cube_single.yaml`.
- Python files no longer contain competing active parameter blocks.
- Any value can be overridden explicitly, for example:

```bash
python Finetuning/train_critic_script.py \
  --config-name cube_single \
  scripts.train_critic_script.lr=0.0001
```

- `finetune_script.py` is an alias of the Hydra `finetune_script2.py` entrypoint.
- `hydra-core>=1.3,<1.4` was added to both requirements files.
- The pipeline scripts check required artifacts and now correctly expect the final finetuned planner checkpoint at step `90`, not `60`.

Available launchers:

```bash
bash run_hydra_pipeline.sh
bash run_hydra_from_critic.sh
bash run_hydra_from_finetune.sh
```

## W&B layout

Each stage creates a separate run in the same W&B group:

- planner: `planner/*`;
- reward: `reward/*`;
- kernel: `kernel/*`;
- Critic-1: `critic_1/*`;
- Critic-2 warmup: `critic_warmup/*`;
- finetune critic: `finetune/critic/*`;
- Bellman probe: `bellman_prob/*`.

The entity, project, and group come from Hydra/environment settings rather than being fixed only inside the finetuner.

## Checkpoint compatibility and rerun scope

Because the trajectory boundary and Critic-1 target definitions changed, old planner/reward/critic checkpoints are not a controlled comparison with this pipeline.

Recommended clean rerun: planner -> reward -> kernel -> Critic-1 -> Critic-2 -> finetune.

The kernel learning algorithm and dataset are unchanged, so an existing kernel checkpoint can be reused if its configuration and files exactly match the YAML profile. Planner, reward, both critic stages, and finetune should otherwise be rerun.

A rollback reference was created before this update:

```text
main-before-critic-consistency-20260910
```

The local `report.pdf` and `report2.pdf` files are not part of the commit.
