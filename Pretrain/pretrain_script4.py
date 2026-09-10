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
from Pretrain.utils import set_seed


@hydra.main(version_base="1.3", config_path="../Finetuning/conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    stage = config.scripts.pretrain_script4
    if config.run.validate_only:
        print(OmegaConf.to_yaml(stage, resolve=True))
        return

    import wandb

    set_seed(int(config.run.seed))
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        group=os.environ.get("WANDB_RUN_GROUP", config.wandb.group),
        name=f"{stage.dataset_name}-{stage.specific_dataset}-task{stage.task_id}-planner",
        config=OmegaConf.to_container(stage, resolve=True),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SDETrainer(
        stage.dataset_name,
        stage.specific_dataset,
        stage.task_id,
        stage.horizon,
        backbone_name=stage.backbone_name,
        backbone_layers=stage.backbone_layers,
        num_steps=stage.num_steps,
        batch_size=stage.batch_size,
        lr=stage.lr,
        device=device,
        stride=stage.stride,
    )
    print(
        f"dataset_name: {stage.dataset_name}, "
        f"specific_dataset: {stage.specific_dataset}, task_id: {stage.task_id}, "
        f"backbone_layers: {stage.backbone_layers}"
    )
    try:
        trainer.train()
    finally:
        run.finish()


if __name__ == "__main__":
    main()
