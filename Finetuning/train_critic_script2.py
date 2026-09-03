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
from Pretrain.utils import init_wandb_run, set_seed


@hydra.main(version_base="1.3", config_path="conf", config_name="cube_single")
def main(config: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    OmegaConf.set_struct(config, True)
    if int(os.environ.get("RANK", "0")) == 0:
        print(OmegaConf.to_yaml(config, resolve=True))
    if config.run.validate_only:
        return

    env = config.environment
    warmup = config.critic_warmup
    set_seed(int(config.run.seed))
    data = get_dataset(
        env.dataset_name,
        env.specific_dataset,
        task_id=env.task_id,
        traj_length=warmup.traj_length,
    )
    trajectories = data.get_trajectories()
    accelerator = Accelerator(mixed_precision=warmup.mixed_precision)
    wandb_run = None
    if accelerator.is_main_process:
        wandb_run = init_wandb_run(
            f"{env.dataset_name}-{env.specific_dataset}-task{env.task_id}-critic-warmup",
            {
                "stage": "critic_warmup",
                "resolved_hydra_config": OmegaConf.to_container(config, resolve=True),
                "num_processes": accelerator.num_processes,
            },
            group=config.wandb.group,
            job_type="critic_warmup",
        )
        wandb_run.define_metric("critic_warmup_step")
        wandb_run.define_metric("critic_warmup/*", step_metric="critic_warmup_step")

    kernel = warmup.kernel
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
    train_critic_with_planner7(
        trajs=trajectories,
        dataset_name=env.dataset_name,
        specific_dataset=env.specific_dataset,
        planner_checkpoint=warmup.planner_checkpoint,
        reward_checkpoint=warmup.reward_checkpoint,
        old_critic_checkpoint=warmup.old_critic_checkpoint,
        backbone_layers=warmup.backbone_layers,
        hidden_layers=warmup.hidden_layers,
        hidden_dim=warmup.hidden_dim,
        kernel_config=kernel_config,
        reward_hidden_layers=warmup.reward_hidden_layers,
        reward_hidden_dim=warmup.reward_hidden_dim,
        batch_size=warmup.batch_size,
        num_steps=warmup.num_steps,
        resample_every=warmup.resample_every,
        vectorized_sampling=warmup.vectorized_sampling,
        plan_chunk_size=warmup.plan_chunk_size,
        horizon=warmup.horizon,
        gamma=warmup.gamma,
        lam=warmup.lam,
        rho=warmup.rho,
        lr=warmup.lr,
        min_lr=warmup.min_lr,
        tau=warmup.tau,
        steps_T=warmup.diffusion_steps,
        num_karras=warmup.num_karras,
        eta=warmup.eta,
        new_step=warmup.new_step,
        task_id=env.task_id,
        log_every=warmup.log_every,
        accelerator=accelerator,
    )

    accelerator.wait_for_everyone()
    test_critic(
        dataset_name=env.dataset_name,
        specific_dataset=env.specific_dataset,
        hidden_layers=warmup.hidden_layers,
        hidden_dim=warmup.hidden_dim,
        checkpoint_step=warmup.new_step,
        critic_checkpoint=warmup.new_step,
        gamma=warmup.gamma,
        horizon=warmup.test_horizon,
        value_scale=config.critic_pretrain.value_scale,
        sigma=warmup.test_sigma,
        target_reward=warmup.test_target_reward,
        trajs=data.get_trajectories(),
        task_id=env.task_id,
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
