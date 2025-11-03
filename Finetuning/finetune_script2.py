import sys
import os
# Change to project root directory
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#os.chdir(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    RWConfig = RewardConfig(beta = 1.0, min_log_prob = 15.0, explore = False) 
    
    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 1000000,
        reward_model_checkpoint = 44000,
        kernel_model_checkpoint = 34000,
        finetune_steps = 10000,
        finetune_batch_size = 64,
        finetune_lr = 1e-4)
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    #mp.spawn(OnlineFinetuner.finetune_planner(), args=(), nprocs=4)
    OnlineFinetuner.finetune_planner()


