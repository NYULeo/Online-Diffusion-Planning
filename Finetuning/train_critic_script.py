from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Finetuning.utils import test_critic, train_critic_with_reward
from Pretrain.Dataset import get_dataset
from Pretrain.utils import init_wandb_run, set_seed


@hydra.main(version_base="1.3", config_path="conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    print(OmegaConf.to_yaml(config, resolve=True))
    if config.run.validate_only:
        return

    env = config.environment
    critic = config.scripts.train_critic_script
    set_seed(int(config.run.seed))
    data = get_dataset(
        env.dataset_name,
        env.specific_dataset,
        task_id=env.task_id,
        traj_length=critic.traj_length,
    )
    trajectories = data.get_trajectories()
    wandb_run = init_wandb_run(
        f"{env.dataset_name}-{env.specific_dataset}-task{env.task_id}-critic",
        {
            "stage": "critic",
            "resolved_hydra_config": OmegaConf.to_container(config, resolve=True),
        },
        group=config.wandb.group,
        job_type="critic",
    )
    try:
        train_critic_with_reward(
            trajectories,
            dataset_name=env.dataset_name,
            specific_dataset=env.specific_dataset,
            reward_hidden_layers=critic.reward_hidden_layers,
            reward_hidden_dim=critic.reward_hidden_dim,
            reward_checkpoint=critic.reward_checkpoint,
            critic_hidden_layers=critic.critic_hidden_layers,
            critic_hidden_dim=critic.critic_hidden_dim,
            batch_size=critic.batch_size,
            num_steps=critic.num_steps,
            gamma=critic.gamma,
            lam=critic.lam,
            horizon=critic.horizon,
            lr=critic.lr,
            min_lr=critic.min_lr,
            tau=critic.tau,
            old_step=critic.old_step,
            new_step=critic.new_step,
            momentum=critic.momentum,
            value_scale=critic.value_scale,
            task_id=env.task_id,
        )
        test_critic(
            dataset_name=env.dataset_name,
            specific_dataset=env.specific_dataset,
            hidden_layers=critic.critic_hidden_layers,
            hidden_dim=critic.critic_hidden_dim,
            checkpoint_step=critic.reward_checkpoint,
            critic_checkpoint=critic.new_step,
            gamma=critic.gamma,
            horizon=critic.horizon,
            value_scale=critic.value_scale,
            sigma=critic.test_sigma,
            target_reward=critic.test_target_reward,
            trajs=data.get_trajectories(),
            task_id=env.task_id,
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
