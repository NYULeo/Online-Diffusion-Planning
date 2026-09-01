from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Pretrain.Rewards.Reward_Backbone import test_Model, train_reward
from Pretrain.utils import init_wandb_run, set_seed


def optional_array(value):
    return None if value is None else np.asarray(value, dtype=np.float32)


@hydra.main(version_base="1.3", config_path="../Finetuning/conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    print(OmegaConf.to_yaml(config, resolve=True))
    if config.run.validate_only:
        return

    env = config.environment
    reward = config.reward_pretrain
    set_seed(int(config.run.seed))
    wandb_run = init_wandb_run(
        f"{env.dataset_name}-{env.specific_dataset}-task{env.task_id}-reward",
        {
            "stage": "reward",
            "resolved_hydra_config": OmegaConf.to_container(config, resolve=True),
        },
        group=config.wandb.group,
        job_type="reward",
    )
    try:
        train_reward(
            dataset_name=env.dataset_name,
            hidden_layers=reward.hidden_layers,
            hidden_dim=reward.hidden_dim,
            batch_size=reward.batch_size,
            num_steps=reward.num_steps,
            save_freq=reward.save_freq,
            lr=reward.lr,
            min_lr=reward.min_lr,
            sigma=reward.sigma,
            alpha=reward.alpha,
            target_reward=reward.target_reward,
            specific_dataset=reward.specific_dataset,
            task_id=env.task_id,
            goal=optional_array(reward.train_goal),
            traj_length=reward.traj_length,
        )
        test_Model(
            env.dataset_name,
            hidden_layers=reward.hidden_layers,
            hidden_dim=reward.hidden_dim,
            specific_dataset=reward.specific_dataset,
            trajs=None,
            sigma=reward.sigma,
            alpha=reward.alpha,
            target_reward=reward.target_reward,
            task_id=env.task_id,
            traj_length=reward.traj_length,
            save_freq=reward.save_freq,
            num_steps=reward.num_steps,
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
