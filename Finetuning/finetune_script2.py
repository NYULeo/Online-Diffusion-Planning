from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Finetuning"))
os.chdir(REPO_ROOT)

from acc_adjoint_matching import Acc_AdjointMatchingConfig
from Finetune_Backbone3 import (
    FinetuningConfig,
    OnlineFinetuner,
    Train_Critic_Config,
    Train_Kernel_Config,
    Train_Reward_Config,
)
from traj_reward4 import RewardConfig
from utils import AlphaSchedulerConfig


def optional_array(value: Any):
    return None if value is None else np.asarray(value, dtype=np.float32)


def build_finetuning_config(config: DictConfig) -> FinetuningConfig:
    stage = config.scripts.finetune_script2
    settings = stage.settings

    alpha_config = AlphaSchedulerConfig(**OmegaConf.to_container(
        stage.alpha_scheduler, resolve=True
    ))

    am_values = OmegaConf.to_container(stage.adjoint_matching, resolve=True)
    runtime_names = (
        "step_start_ema",
        "ema_decay",
        "save_freq",
        "save_model_freq",
        "log_freq",
    )
    runtime_values = {name: am_values.pop(name) for name in runtime_names}
    am_config = Acc_AdjointMatchingConfig(**am_values)
    for name, value in runtime_values.items():
        setattr(am_config, name, value)

    reward_config = RewardConfig(**OmegaConf.to_container(
        stage.reward_objective, resolve=True
    ))

    reward_values = OmegaConf.to_container(stage.reward_model, resolve=True)
    for name in ("train_goal", "rollout_goal", "rollout_start_cells"):
        reward_values[name] = optional_array(reward_values[name])
    reward_values["task_id"] = stage.task_id
    reward_training = Train_Reward_Config(**reward_values)

    kernel_values = OmegaConf.to_container(stage.kernel_model, resolve=True)
    kernel_values["λ_reg"] = kernel_values.pop("lambda_reg")
    kernel_training = Train_Kernel_Config(**kernel_values)
    critic_training = Train_Critic_Config(**OmegaConf.to_container(
        stage.critic_update, resolve=True
    ))

    return FinetuningConfig(
        AMConfig=am_config,
        RewardConfig=reward_config,
        AlphaConfig=alpha_config,
        dataset_name=stage.dataset_name,
        specific_dataset=stage.specific_dataset,
        planner_checkpoint=settings.planner_checkpoint,
        reward_model_checkpoint=settings.reward_model_checkpoint,
        kernel_model_checkpoint=settings.kernel_model_checkpoint,
        critic_model_checkpoint=settings.critic_model_checkpoint,
        train_reward_config=reward_training,
        train_kernel_config=kernel_training,
        train_critic_config=critic_training,
        offline=settings.offline,
        critic=settings.critic,
        update_critic=settings.update_critic,
        kernel=settings.kernel,
        update_kernel=settings.update_kernel,
        buffer_size=settings.buffer_size,
        finetune_buffer_cutoff_length=settings.finetune_buffer_cutoff_length,
        train_buffer_cutoff_length=settings.train_buffer_cutoff_length,
        finetune_suffix_cut_length=settings.finetune_suffix_cut_length,
        finetune_steps=settings.finetune_steps,
        finetune_rounds=settings.finetune_rounds,
        diffusion_steps=settings.diffusion_steps,
        karras_percent=settings.karras_percent,
        Loss_Clip_percent=settings.loss_clip_percent,
        finetune_batch_size=settings.finetune_batch_size,
        finetune_batch_per_sample=settings.finetune_batch_per_sample,
        finetune_lr=settings.finetune_lr,
        initial_lam=settings.initial_lam,
        eta_lam=settings.eta_lam,
        gradient_accumulate_every=settings.gradient_accumulate_every,
        update_lambda_every=settings.update_lambda_every,
        reward_scaling_factor=settings.reward_scaling_factor,
        MaxEnt=settings.max_ent,
        Entropy_Scaling_Factor=settings.entropy_scaling_factor,
        rollout_length=settings.rollout_length,
        rollout_num_envs=settings.rollout_num_envs,
        num_rollout_processes=settings.num_rollout_processes,
        continual_rollout=settings.continual_rollout,
        chunk_size=settings.chunk_size,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base="1.3", config_path="conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    finetuning_config = build_finetuning_config(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(OmegaConf.to_yaml(config.scripts.finetune_script2, resolve=True))
    if config.run.validate_only:
        return

    os.environ["WANDB_ENTITY"] = str(config.wandb.entity)
    os.environ["WANDB_PROJECT"] = str(config.wandb.project)
    os.environ.setdefault("WANDB_RUN_GROUP", str(config.wandb.group))
    set_seed(int(config.run.seed))
    OnlineFinetuner(finetuning_config).finetune_planner()


if __name__ == "__main__":
    main()
