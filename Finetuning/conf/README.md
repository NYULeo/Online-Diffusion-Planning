# Hydra experiment profiles

The algorithm and migration notes for this update are in
[`MAIN_UPDATE_2026-09-10.md`](../../MAIN_UPDATE_2026-09-10.md).

`cube_single.yaml` contains the Cube Single pipeline parameters. Parameters are grouped by entrypoint so initial critic, warmup critic, and per-round critic settings have distinct scopes.

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

Hydra is the only input configuration mechanism for these entrypoints. Every stage logs to its own W&B run, while `WANDB_RUN_GROUP` groups a complete pipeline.

The Cube Single critic uses the symlog representation: initial critic checkpoint `-1`, planner7 warmup checkpoint `0`, and `Q_scale × symexp(V)` decoding. Older `Q_mean/Q_std` critics are not compatible; rerun critic, warmup, and finetune after switching representations.

Planner7 candidate generation and kernel filtering follow main. The sampling implementation can be selected with `scripts.train_critic_script2.vectorized_sampling` or `scripts.finetune_script2.critic_update.vectorized_sampling`.

The planner profile reproduces main's Cube Single planner configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python Pretrain/pretrain_script4.py --config-name cube_single
```
