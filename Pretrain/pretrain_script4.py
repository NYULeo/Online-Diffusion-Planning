from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Pretrain.Planners.Backbone.Trainer import SDETrainer
from Pretrain.utils import init_wandb_run, set_seed


@hydra.main(version_base="1.3", config_path="../Finetuning/conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    print(OmegaConf.to_yaml(config, resolve=True))
    if config.run.validate_only:
        return

    env = config.environment
    planner = config.planner_pretrain
    set_seed(int(config.run.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb_run(
        f"{env.dataset_name}-{env.specific_dataset}-task{env.task_id}-planner",
        {
            "stage": "planner",
            "resolved_hydra_config": OmegaConf.to_container(config, resolve=True),
        },
        group=config.wandb.group,
        job_type="planner",
    )
    try:
        trainer = SDETrainer(
            env.dataset_name,
            env.specific_dataset,
            env.task_id,
            planner.horizon,
            backbone_name=planner.backbone_name,
            backbone_layers=planner.backbone_layers,
            num_steps=planner.num_steps,
            batch_size=planner.batch_size,
            lr=planner.lr,
            device=device,
            stride=planner.stride,
        )
        trainer.train()
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
