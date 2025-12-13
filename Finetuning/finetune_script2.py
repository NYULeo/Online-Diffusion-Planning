import sys
import os

from sympy import true
# Change to project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from Finetune_Backbone import OnlineFinetuner, FinetuningConfig
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
    #AMConfig = AdjointMatchingConfig(horizon = 32) 
    AMConfig = Acc_AdjointMatchingConfig(horizon = 32)
    
    #RWConfig = RewardConfig(beta = 1.0, min_log_prob = 15.0, explore = False) 
    RWConfig = RewardConfig(beta = 1.0, min_log_prob = 5.0, explore = False) 
    
    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 1000000,
        reward_model_checkpoint = 15000,
        kernel_model_checkpoint = 34000,
        finetune_steps = 5000,
        diffusion_steps = 50,
        karras_percent = 0.05,
        Loss_Clip_percent = 0.75,
        finetune_batch_size = 12,
        finetune_lr = 2e-05,
        #inital_lam = 2.5,
        inital_lam = 0.0,
        eta_lam = 0.5,
        gradient_accumulate_every = 1,
        update_lambda_every = 1,
        reward_scaling_factor = 50,
        MaxEnt = True,
        Entropy_Scaling_Factor = 0.0) 
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()

#finetune_lr = 1e-05,