from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Pretrain.Transition_Kernel.Kernel_Backbone import test_kernel_mog, train_mog_kernel
from Pretrain.utils import init_wandb_run, set_seed


@hydra.main(version_base="1.3", config_path="../Finetuning/conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    print(OmegaConf.to_yaml(config, resolve=True))
    if config.run.validate_only:
        return

    env = config.environment
    kernel = config.kernel_pretrain
    set_seed(int(config.run.seed))
    wandb_run = init_wandb_run(
        f"{env.dataset_name}-{env.specific_dataset}-task{env.task_id}-kernel",
        {
            "stage": "kernel",
            "resolved_hydra_config": OmegaConf.to_container(config, resolve=True),
        },
        group=config.wandb.group,
        job_type="kernel",
    )
    try:
        train_mog_kernel(
            dataset_name=env.dataset_name,
            specific_dataset=kernel.specific_dataset,
            task_id=kernel.task_id,
            trajs=None,
            batch_size=kernel.batch_size,
            lr=kernel.lr,
            num_steps=kernel.num_steps,
            save_freq=kernel.save_freq,
            ensemble_size=kernel.ensemble_size,
            num_modes=kernel.num_modes,
            num_hidden_layers=kernel.num_hidden_layers,
            hidden_dim=kernel.hidden_dim,
            λ_reg=kernel.lambda_reg,
            noise_floor=kernel.noise_floor,
        )
        test_kernel_mog(
            dataset_name=env.dataset_name,
            specific_dataset=kernel.specific_dataset,
            task_id=kernel.task_id,
            trajs=None,
            save_freq=kernel.num_steps,
            num_steps=kernel.num_steps,
            num_hidden_layers=kernel.num_hidden_layers,
            hidden_dim=kernel.hidden_dim,
            ensemble_size=kernel.ensemble_size,
            num_modes=kernel.num_modes,
            quantile=kernel.test_quantile,
            noise_floor=kernel.noise_floor,
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
