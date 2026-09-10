from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from Finetuning.utils import KernelConfig, test_critic, train_critic_with_planner7
from Pretrain.Dataset import get_dataset
from Pretrain.utils import set_seed


@hydra.main(version_base="1.3", config_path="conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    stage = config.scripts.train_critic_script2
    if config.run.validate_only:
        if int(os.environ.get("RANK", "0")) == 0:
            print(OmegaConf.to_yaml(stage, resolve=True))
        return

    import wandb

    set_seed(int(config.run.seed))
    accelerator = Accelerator(mixed_precision=stage.mixed_precision)
    run = None
    if accelerator.is_main_process:
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            group=os.environ.get("WANDB_RUN_GROUP", config.wandb.group),
            name=f"{stage.dataset_name}-{stage.specific_dataset}-task{stage.task_id}-critic_2",
            config=OmegaConf.to_container(stage, resolve=True),
        )

    data = get_dataset(
        stage.dataset_name,
        stage.specific_dataset,
        task_id=stage.task_id,
        traj_length=stage.traj_length,
    )
    trajectories = data.get_trajectories(suffix_length=stage.train_horizon)
    kernel = stage.kernel
    kernel_config = KernelConfig(
        checkpoint=kernel.checkpoint,
        type_kernel=kernel.type_kernel,
        num_hidden_layers=kernel.num_hidden_layers,
        hidden_dim=kernel.hidden_dim,
        num_modes=kernel.num_modes,
        noise_floor=kernel.noise_floor,
        min_log_prob=kernel.min_log_prob,
        oversample=kernel.oversample,
    )
    try:
        accelerator.wait_for_everyone()
        train_critic_with_planner7(
            trajs=trajectories,
            dataset_name=stage.dataset_name,
            specific_dataset=stage.specific_dataset,
            planner_checkpoint=stage.planner_checkpoint,
            reward_checkpoint=stage.reward_checkpoint,
            old_critic_checkpoint=stage.old_critic_checkpoint,
            backbone_layers=stage.backbone_layers,
            hidden_layers=stage.hidden_layers,
            hidden_dim=stage.hidden_dim,
            kernel_config=kernel_config,
            reward_hidden_layers=stage.reward_hidden_layers,
            reward_hidden_dim=stage.reward_hidden_dim,
            batch_size=stage.batch_size,
            num_steps=stage.num_steps,
            resample_every=stage.resample_every,
            vectorized_sampling=stage.vectorized_sampling,
            plan_chunk_size=stage.plan_chunk_size,
            horizon=stage.train_horizon,
            gamma=stage.gamma,
            lam=stage.lam,
            rho=stage.rho,
            lr=stage.lr,
            min_lr=stage.min_lr,
            tau=stage.tau,
            steps_T=stage.diffusion_steps,
            num_karras=stage.num_karras,
            eta=stage.eta,
            new_step=stage.new_step,
            task_id=stage.task_id,
            mix_reset=stage.mix_reset,
            n_reset=stage.n_reset,
            log_every=stage.log_every,
            accelerator=accelerator,
        )
        accelerator.wait_for_everyone()
        test_critic(
            dataset_name=stage.dataset_name,
            specific_dataset=stage.specific_dataset,
            hidden_layers=stage.hidden_layers,
            hidden_dim=stage.hidden_dim,
            checkpoint_step=stage.new_step,
            critic_checkpoint=stage.new_step,
            gamma=stage.gamma,
            horizon=stage.test_horizon,
            value_scale=stage.test_value_scale,
            sigma=stage.test_sigma,
            target_reward=stage.test_target_reward,
            trajs=data.get_trajectories(),
            task_id=stage.task_id,
        )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
