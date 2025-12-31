import sys
import os

from sympy import true
# Change to project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from Finetune_Backbone import OnlineFinetuner, FinetuningConfig, Train_Kernel_Config, Train_Reward_Config
from adjoint_matching import AdjointMatchingConfig
from acc_adjoint_matching import Acc_AdjointMatchingConfig
from traj_reward import RewardConfig
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
              
if __name__ == "__main__":
    # Example usage of the Adjoint Matching training without a dataset.
    # In practice, 
    # 
    # replace the reward and backbone initialisations with
    # loading of your pretrained models (e.g. via torch.load).
    env_name = 'pointmaze'
    specific_env = 'medium'
    AMConfig = Acc_AdjointMatchingConfig(horizon = 32)
    #RWConfig = RewardConfig(beta = 1.0, min_log_prob = 15.0, explore = False) 
    RWConfig = RewardConfig(
               beta = 1.0, 
               min_log_prob = 5.0, 
               explore = False) 
    
    TrainRewardConfig = Train_Reward_Config(
                          batch_size = 256, 
                          num_steps = 1000, 
                          lr = 3e-4, 
                          sigma = 7.0, 
                          target_reward = 1.0, 
                          goal = np.array([[6, 1]], dtype = int))
    
    TrainKernelConfig = Train_Kernel_Config(
                            batch_size = 256, 
                            num_steps = 1000,
                            lr = 3e-4,
                            ensemble_size = 10,
                            λ_reg = 1e-3)
    
    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 0,
        reward_model_checkpoint = 0,
        kernel_model_checkpoint = 0,
        finetune_steps = 3000,
        finetune_rounds = 300,
        diffusion_steps = 50,
        karras_percent = 0.05,
        Loss_Clip_percent = 0.75,
        finetune_batch_size = 8,
        finetune_lr = 2e-05,
        initial_lam = 0.05,
        eta_lam = 0.5,
        gradient_accumulate_every = 1,
        update_lambda_every = 1,
        reward_scaling_factor = 50,
        MaxEnt = False,
        Entropy_Scaling_Factor = 0.5,
        train_reward_config = TrainRewardConfig,
        train_kernel_config = TrainKernelConfig) 
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()

#finetune_lr = 1e-05,

