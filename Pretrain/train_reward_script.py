from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Pretrain.Rewards.Reward_Backbone import test_Model, train_reward
from Pretrain.utils import set_seed


@hydra.main(version_base="1.3", config_path="../Finetuning/conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    stage = config.scripts.train_reward_script
    if config.run.validate_only:
        print(OmegaConf.to_yaml(stage, resolve=True))
        return

    import wandb

    set_seed(int(config.run.seed))
    hp = OmegaConf.to_container(stage, resolve=True)
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        group=os.environ.get("WANDB_RUN_GROUP", config.wandb.group),
        name=f"{stage.dataset_name}-{stage.specific_dataset}-task{stage.task_id}-reward",
        config=hp,
    )
    try:
        train_reward(
            dataset_name=stage.dataset_name,
            hidden_layers=stage.hidden_layers,
            hidden_dim=stage.hidden_dim,
            batch_size=stage.batch_size,
            num_steps=stage.num_steps,
            save_freq=stage.save_freq,
            lr=stage.lr,
            min_lr=stage.min_lr,
            sigma=stage.sigma,
            alpha=stage.alpha,
            target_reward=stage.target_reward,
            specific_dataset=stage.specific_dataset,
            task_id=stage.task_id,
            traj_length=stage.traj_length,
        )
        test_Model(
            stage.dataset_name,
            hidden_layers=stage.hidden_layers,
            hidden_dim=stage.hidden_dim,
            specific_dataset=stage.specific_dataset,
            trajs=None,
            sigma=stage.sigma,
            alpha=stage.alpha,
            target_reward=stage.target_reward,
            task_id=stage.task_id,
            traj_length=stage.traj_length,
            save_freq=stage.save_freq,
            num_steps=stage.num_steps,
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
