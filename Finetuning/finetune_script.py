'''Entry-point script: build the finetuning configs and run OnlineFinetuner (JAX/Flax port).'''
import sys
import os
# Change to project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Finetuning.Finetune_Backbone import (
    OnlineFinetuner, FinetuningConfig,
    Train_Reward_Config, Train_Kernel_Config, Train_Critic_Config,
)
from Finetuning.utils import AlphaSchedulerConfig
from Finetuning.adjoint_matching import AdjointMatchingConfig
from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingConfig
from Finetuning.traj_reward import RewardConfig
import random
import numpy as np
import jax


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


if __name__ == "__main__":
    # Example usage of the Adjoint Matching training without a dataset.
    # In practice,
    #
    # replace the reward and backbone initialisations with
    # loading of your pretrained models.
    # TODO(checkpoint-bridge): original loaded pretrained models via torch.load.
    env_name = 'kitchen'
    specific_env = 'partial'
    #AMConfig = AdjointMatchingConfig(horizon = 32)
    AMConfig = Acc_AdjointMatchingConfig(horizon = 32)

    RWConfig = RewardConfig(beta = 1.0, min_log_prob = 150.0, explore = False)

    AlphaConfig = AlphaSchedulerConfig(alpha_start = 1.0, alpha_end = 1.0, total_steps = 1000000)

    FTConfig = FinetuningConfig(
        AMConfig = AMConfig,
        RewardConfig = RWConfig,
        AlphaConfig = AlphaConfig,
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 990000,
        reward_model_checkpoint = 10000,
        kernel_model_checkpoint = 50000,
        critic_model_checkpoint = 0,
        train_reward_config = Train_Reward_Config(),
        train_kernel_config = Train_Kernel_Config(),
        train_critic_config = Train_Critic_Config(),
        finetune_steps = 1000000,
        finetune_batch_size  = 12,
        finetune_lr = 2e-4)
    rng = set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    # API-CHANGE: torch.multiprocessing.mp.spawn(...) dropped; the original
    # `mp.spawn(OnlineFinetuner.finetune_planner(), ...)` already called
    # finetune_planner() eagerly, so we call it directly and thread the jax rng key.
    OnlineFinetuner.finetune_planner(seed=rng)
