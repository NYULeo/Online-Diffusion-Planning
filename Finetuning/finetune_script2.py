from __future__ import annotations

import os
import random
import shlex
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingConfig
from Finetuning.Finetune_Backbone3 import (
    FinetuningConfig,
    OnlineFinetuner,
    Train_Critic_Config,
    Train_Kernel_Config,
    Train_Reward_Config,
)
from Finetuning.traj_reward4 import RewardConfig
from Finetuning.utils import AlphaSchedulerConfig


def optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=np.float32)


def build_finetuning_config(config: dict[str, Any]) -> FinetuningConfig:
    """Convert the resolved Hydra tree into the existing training dataclasses."""
    environment = config["environment"]
    finetuning = config["finetuning"]
    if finetuning["finetune_steps"] % finetuning["finetune_rounds"] != 0:
        raise ValueError("finetuning.finetune_steps must be divisible by finetuning.finetune_rounds")

    alpha_config = AlphaSchedulerConfig(**config["alpha"])

    am_values = dict(config["adjoint_matching"])
    am_runtime_fields = {
        name: am_values.pop(name)
        for name in ("save_freq", "save_model_freq", "log_freq")
    }
    am_config = Acc_AdjointMatchingConfig(**am_values)
    for name, value in am_runtime_fields.items():
        setattr(am_config, name, value)

    reward_config = RewardConfig(**config["reward"])

    reward_values = dict(config["reward_training"])
    for name in ("train_goal", "rollout_goal", "rollout_start_cells"):
        reward_values[name] = optional_array(reward_values[name])
    reward_values["task_id"] = environment["task_id"]
    reward_training = Train_Reward_Config(**reward_values)

    kernel_values = dict(config["kernel_training"])
    kernel_values["λ_reg"] = kernel_values.pop("lambda_reg")
    kernel_training = Train_Kernel_Config(**kernel_values)
    critic_training = Train_Critic_Config(**config["critic_training"])

    return FinetuningConfig(
        AMConfig=am_config,
        RewardConfig=reward_config,
        AlphaConfig=alpha_config,
        dataset_name=environment["dataset_name"],
        specific_dataset=environment["specific_dataset"],
        planner_checkpoint=finetuning["planner_checkpoint"],
        reward_model_checkpoint=finetuning["reward_model_checkpoint"],
        kernel_model_checkpoint=finetuning["kernel_model_checkpoint"],
        critic_model_checkpoint=finetuning["critic_model_checkpoint"],
        train_reward_config=reward_training,
        train_kernel_config=kernel_training,
        train_critic_config=critic_training,
        offline=finetuning["offline"],
        critic=finetuning["critic"],
        update_critic=finetuning["update_critic"],
        kernel=finetuning["kernel"],
        update_kernel=finetuning["update_kernel"],
        buffer_size=finetuning["buffer_size"],
        finetune_buffer_cutoff_length=finetuning["finetune_buffer_cutoff_length"],
        train_buffer_cutoff_length=finetuning["train_buffer_cutoff_length"],
        finetune_steps=finetuning["finetune_steps"],
        finetune_rounds=finetuning["finetune_rounds"],
        diffusion_steps=finetuning["diffusion_steps"],
        karras_percent=finetuning["karras_percent"],
        Loss_Clip_percent=finetuning["loss_clip_percent"],
        finetune_batch_size=finetuning["finetune_batch_size"],
        finetune_batch_per_sample=finetuning["finetune_batch_per_sample"],
        finetune_lr=finetuning["finetune_lr"],
        initial_lam=finetuning["initial_lam"],
        eta_lam=finetuning["eta_lam"],
        gradient_accumulate_every=finetuning["gradient_accumulate_every"],
        update_lambda_every=finetuning["update_lambda_every"],
        reward_scaling_factor=finetuning["reward_scaling_factor"],
        MaxEnt=finetuning["max_ent"],
        Entropy_Scaling_Factor=finetuning["entropy_scaling_factor"],
        rollout_length=finetuning["rollout_length"],
        rollout_num_envs=finetuning["rollout_num_envs"],
        num_rollout_processes=finetuning["num_rollout_processes"],
        continual_rollout=finetuning["continual_rollout"],
        chunk_size=finetuning["chunk_size"],
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
    os.environ["PYTHONHASHSEED"] = str(seed)


@hydra.main(version_base="1.3", config_path="conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved Hydra configuration must be a mapping")
    finetuning_config = build_finetuning_config(resolved)

    if int(os.environ.get("RANK", "0")) == 0:
        print("Resolved Hydra configuration:")
        print(OmegaConf.to_yaml(config, resolve=True))
        print(f"Hydra overrides: {HydraConfig.get().overrides.task}")
        print("Launch command:")
        print(" ".join(shlex.quote(value) for value in sys.argv))

    if bool(resolved["run"]["validate_only"]):
        return

    os.environ.setdefault("WANDB_RUN_GROUP", str(resolved["wandb"]["group"]))
    set_seed(int(resolved["run"]["seed"]))
    finetuner = OnlineFinetuner(finetuning_config)
    if finetuner.wandb_run is not None:
        finetuner.wandb_run.config.update(
            {
                "resolved_hydra_config": resolved,
                "hydra_overrides": HydraConfig.get().overrides.task,
            },
            allow_val_change=False,
        )
    finetuner.finetune_planner()


if __name__ == "__main__":
    main()
