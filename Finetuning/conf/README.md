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
