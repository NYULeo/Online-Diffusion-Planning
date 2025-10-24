import sys
import os
# Change to project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Finetune_Backbone import OnlineFinetuner, FinetuningConfig
from adjoint_matching import AdjointMatchingConfig
from acc_adjoint_matching import Acc_AdjointMatchingConfig
from traj_reward import RewardConfig


              
if __name__ == "__main__":
    # Example usage of the Adjoint Matching training without a dataset.
    # In practice, 
    # 
    # replace the reward and backbone initialisations with
    # loading of your pretrained models (e.g. via torch.load).
    env_name = 'kitchen'
    specific_env = 'partial'
    #AMConfig = AdjointMatchingConfig(horizon = 32) 
    AMConfig = Acc_AdjointMatchingConfig(horizon = 32)
    
    RWConfig = RewardConfig(beta = 1.0, min_log_prob = 150.0, explore = False) 
    
    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 990000,
        reward_model_checkpoint = 10000,
        kernel_model_checkpoint = 50000,
        finetune_steps = 1000000,
        finetune_batch_size  = 12,
        finetune_lr = 2e-4)
    
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()

    
   
   
