# Hydra experiment profiles

`cube_single.yaml` contains the effective Cube Single parameters from main commit `f8740fd`. Parameters are grouped by entrypoint so initial critic, warmup critic, and per-round critic settings have distinct scopes.

Select a profile with:

```bash
python <entrypoint>.py --config-name cube_single
```

Override a value without editing Python:

```bash
python <entrypoint>.py --config-name cube_single scripts.finetune_script2.settings.finetune_lr=0.00002
```

Run the complete configured pipeline from the repository root:

```bash
bash run_hydra_pipeline.sh
```

Hydra changes configuration delivery only. Training, logging, model construction, losses, datasets, and rollout behavior come from main.

The Cube Single critic uses the main-branch symlog representation: initial critic checkpoint `-1`, planner7 warmup checkpoint `0`, and `Q_scale × symexp(V)` decoding. Debugger-era `Q_mean/Q_std` critics are not compatible; rerun critic, warmup, and finetune after switching representations.

Planner7 candidate generation and kernel filtering follow main. The sampling implementation can be selected with `scripts.train_critic_script2.vectorized_sampling` or `scripts.finetune_script2.critic_update.vectorized_sampling`.

The planner profile reproduces main's Cube Single planner configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python Pretrain/pretrain_script4.py --config-name cube_single
```
