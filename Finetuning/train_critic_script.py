from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Finetuning.utils import test_critic_with_reward, train_critic_with_reward
from Pretrain.Dataset import get_dataset
from Pretrain.utils import set_seed


@hydra.main(version_base="1.3", config_path="conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    stage = config.scripts.train_critic_script
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
        name=f"{stage.dataset_name}-{stage.specific_dataset}-task{stage.task_id}-critic_1",
        config=hp,
    )
    data = get_dataset(
        stage.dataset_name,
        stage.specific_dataset,
        task_id=stage.task_id,
        traj_length=stage.traj_length,
    )
    try:
        train_critic_with_reward(
            trajs=data.get_trajectories(),
            dataset_name=stage.dataset_name,
            specific_dataset=stage.specific_dataset,
            reward_hidden_layers=stage.reward_hidden_layers,
            reward_hidden_dim=stage.reward_hidden_dim,
            reward_checkpoint=stage.reward_checkpoint,
            critic_hidden_layers=stage.critic_hidden_layers,
            critic_hidden_dim=stage.critic_hidden_dim,
            batch_size=stage.batch_size,
            num_steps=stage.num_steps,
            gamma=stage.gamma,
            lr=stage.lr,
            min_lr=stage.min_lr,
            old_step=stage.old_step,
            new_step=stage.new_step,
            momentum=stage.momentum,
            value_scale=stage.value_scale,
            task_id=stage.task_id,
        )
        test_critic_with_reward(
            trajs=data.get_trajectories(split="val"),
            dataset_name=stage.dataset_name,
            specific_dataset=stage.specific_dataset,
            reward_hidden_layers=stage.reward_hidden_layers,
            reward_hidden_dim=stage.reward_hidden_dim,
            reward_checkpoint=stage.reward_checkpoint,
            critic_hidden_layers=stage.critic_hidden_layers,
            critic_hidden_dim=stage.critic_hidden_dim,
            critic_checkpoint=stage.new_step,
            batch_size=stage.batch_size,
            gamma=stage.gamma,
            value_scale=stage.value_scale,
            task_id=stage.task_id,
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
