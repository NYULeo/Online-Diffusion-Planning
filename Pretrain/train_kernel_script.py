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
from Pretrain.utils import set_seed


@hydra.main(version_base="1.3", config_path="../Finetuning/conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    stage = config.scripts.train_kernel_script
    if config.run.validate_only:
        print(OmegaConf.to_yaml(stage, resolve=True))
        return

    import wandb

    set_seed(int(config.run.seed))
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        group=os.environ.get("WANDB_RUN_GROUP", config.wandb.group),
        name=f"{stage.dataset_name}-{stage.specific_dataset}-kernel",
        config=OmegaConf.to_container(stage, resolve=True),
    )
    try:
        train_mog_kernel(
            dataset_name=stage.dataset_name,
            specific_dataset=stage.specific_dataset,
            task_id=stage.task_id,
            batch_size=stage.batch_size,
            lr=stage.lr,
            num_steps=stage.num_steps,
            save_freq=stage.save_freq,
            ensemble_size=stage.ensemble_size,
            num_modes=stage.num_modes,
            num_hidden_layers=stage.num_hidden_layers,
            hidden_dim=stage.hidden_dim,
            λ_reg=stage.lambda_reg,
            noise_floor=stage.noise_floor,
        )
        test_kernel_mog(
            dataset_name=stage.dataset_name,
            specific_dataset=stage.specific_dataset,
            task_id=stage.task_id,
            trajs=None,
            save_freq=stage.save_freq,
            num_steps=stage.num_steps,
            num_hidden_layers=stage.num_hidden_layers,
            hidden_dim=stage.hidden_dim,
            ensemble_size=stage.ensemble_size,
            num_modes=stage.num_modes,
            quantile=stage.test_quantile,
            noise_floor=stage.noise_floor,
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
