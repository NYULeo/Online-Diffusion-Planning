# Hydra experiment profiles

Each environment/task profile contains every stage of the pipeline in one YAML file. `cube_single.yaml` covers planner, reward, kernel, critic, critic warmup, finetune, and rollout for Cube Single task 4.

Select a profile with:

```bash
python <entrypoint>.py --config-name cube_single
```

Override a value without editing Python:

```bash
python <entrypoint>.py --config-name cube_single finetuning.finetune_lr=0.00002
```

Run the complete configured pipeline from the repository root:

```bash
bash run_hydra_pipeline.sh
```

The pipeline creates one W&B group per launch and one run per stage. Metrics are namespaced as `planner/*`, `reward/*`, `kernel/*`, `critic/*`, `critic_warmup/*`, `finetune/*`, and `rollout/*`.

The Cube Single critic uses the main-branch symlog representation: initial critic checkpoint `-1`, planner7 warmup checkpoint `0`, and `Q_scale × symexp(V)` decoding. Debugger-era `Q_mean/Q_std` critics are not compatible; rerun critic, warmup, and finetune after switching representations.

Planner7 candidate generation and kernel filtering are vectorized in GPU chunks. Restore the scalar main-branch path for debugging with `critic_warmup.vectorized_sampling=false` (warmup) or `critic_training.vectorized_sampling=false` (per-round critic updates).

The planner profile uses a four-GPU effective batch of 256 (`batch_size=256`, `gradient_accumulate_every=1`). Restore the original single-GPU execution without changing code:

```bash
CUDA_VISIBLE_DEVICES=0 python Pretrain/pretrain_script4.py --config-name cube_single \
  planner_pretrain.data_parallel=false \
  planner_pretrain.batch_size=128 \
  planner_pretrain.gradient_accumulate_every=2
```
