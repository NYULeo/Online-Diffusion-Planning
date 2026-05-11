import sys
import os
# Change to project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from utils import AlphaSchedulerConfig
from Finetune_Backbone import OnlineFinetuner, FinetuningConfig, Train_Critic_Config, Train_Kernel_Config, Train_Reward_Config
from adjoint_matching import AdjointMatchingConfig
#from acc_adjoint_matching import Acc_AdjointMatchingConfig
from AM import Acc_AdjointMatchingConfig
#from traj_reward import RewardConfig
from comp_reward import RewardConfig
import random
import numpy as np
import torch
import json
import os
from pathlib import Path
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""
def load_finetuning_args(env_name: str, specific_env: str, base_path: str = None) -> FinetuningConfig:
   
    if base_path is None:
        # Get the project root directory (two levels up from this file)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    args_file = os.path.join(base_path, "Finetuning", "args", env_name, specific_env, "hyperparameters.json")
    
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Configuration file not found: {args_file}")
    
    with open(args_file, 'r') as f:
        args = json.load(f)
    
    # Extract environment details
    env_details = args.get('env_details', {})
    pretrained_models = args.get('pretrained_models', {})
    
    # Build Acc_AdjointMatchingConfig
    am_config_dict = args.get('adjoint_matching_config', {})
    AMConfig = Acc_AdjointMatchingConfig(**am_config_dict)
    
    # Build RewardConfig
    reward_config_dict = args.get('reward_config', {})
    # Handle delta if it's a string representation of a tensor
    if 'delta' in reward_config_dict and isinstance(reward_config_dict['delta'], str):
        # Extract float value from tensor string if needed
        delta_str = reward_config_dict['delta']
        if 'tensor' in delta_str.lower():
            import re
            match = re.search(r'([+-]?[0-9]*\.?[0-9]+)', delta_str)
            if match:
                reward_config_dict['delta'] = float(match.group(1))
            else:
                reward_config_dict['delta'] = None
    RWConfig = RewardConfig(**reward_config_dict)
    
    # Build Train_Reward_Config
    reward_training_dict = args.get('reward_training', {})
    # Convert train_goal and rollout_goal lists to numpy arrays
    if 'train_goal' in reward_training_dict and reward_training_dict['train_goal'] is not None:
        reward_training_dict['train_goal'] = np.array(reward_training_dict['train_goal'])
    if 'rollout_goal' in reward_training_dict and reward_training_dict['rollout_goal'] is not None:
        reward_training_dict['rollout_goal'] = np.array(reward_training_dict['rollout_goal'])
    TrainRewardConfig = Train_Reward_Config(**reward_training_dict)
    
    # Build Train_Kernel_Config
    kernel_training_dict = args.get('kernel_training', {})
    # Handle λ_reg (unicode lambda) - check both unicode and regular lambda
    if 'λ_reg' in kernel_training_dict:
        kernel_training_dict['λ_reg'] = kernel_training_dict['λ_reg']
    elif 'lambda_reg' in kernel_training_dict:
        kernel_training_dict['λ_reg'] = kernel_training_dict.pop('lambda_reg')
    TrainKernelConfig = Train_Kernel_Config(**kernel_training_dict)
    
    # Build Train_Critic_Config
    critic_training_dict = args.get('critic_training', {})
    TrainCriticConfig = Train_Critic_Config(**critic_training_dict)
    
    # Build FinetuningConfig
    finetuning_hyperparams = args.get('finetuning_hyperparameters', {})
    exploration_hyperparams = args.get('exploration_hyperparameters', {})
    
    # Determine if critic is used (check if critic_model_checkpoint is set or critic_training exists)
    use_critic = pretrained_models.get('critic_model_checkpoint', None) is not None or 'critic_training' in args
    
    FTConfig = FinetuningConfig(
        AMConfig=AMConfig,
        RewardConfig=RWConfig,
        dataset_name=env_details.get('dataset_name', env_name),
        specific_dataset=env_details.get('specific_dataset', specific_env),
        planner_checkpoint=pretrained_models.get('planner_checkpoint', 0),
        reward_model_checkpoint=pretrained_models.get('reward_model_checkpoint', 0),
        kernel_model_checkpoint=pretrained_models.get('kernel_model_checkpoint', 0),
        critic_model_checkpoint=pretrained_models.get('critic_model_checkpoint', 0),
        critic=use_critic,
        buffer_size=finetuning_hyperparams.get('buffer_size', 5500),
        finetune_steps=finetuning_hyperparams.get('finetune_total_steps', 3000),
        finetune_rounds=finetuning_hyperparams.get('finetune_rounds', 300),
        diffusion_steps=finetuning_hyperparams.get('diffusion_steps', 50),
        karras_percent=finetuning_hyperparams.get('karras_percent', 0.05),
        Loss_Clip_percent=finetuning_hyperparams.get('Loss_Clip_percent', 0.75),
        finetune_batch_size=finetuning_hyperparams.get('finetune_batch_size', 8),
        finetune_lr=finetuning_hyperparams.get('finetune_lr', 2e-5),
        initial_lam=finetuning_hyperparams.get('initial_lam', 0.05),
        eta_lam=finetuning_hyperparams.get('eta_lam', 0.5),
        gradient_accumulate_every=finetuning_hyperparams.get('gradient_accumulate_every', 1),
        update_lambda_every=finetuning_hyperparams.get('update_lambda_every', 1),
        reward_scaling_factor=finetuning_hyperparams.get('reward_scaling_factor', 50),
        MaxEnt=finetuning_hyperparams.get('MaxEnt', False),
        Entropy_Scaling_Factor=finetuning_hyperparams.get('Entropy_Scaling_Factor', 0.5),
        rollout_length=exploration_hyperparams.get('rollout_length', 2000),
        rollout_num_envs=exploration_hyperparams.get('rollout_num_envs', 1),
        train_reward_config=TrainRewardConfig,
        train_kernel_config=TrainKernelConfig,
        train_critic_config=TrainCriticConfig
    )
    return FTConfig
"""

def load_finetuning_args(env_name: str, specific_env: str, base_path: str = None) -> FinetuningConfig:
   
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    args_file = os.path.join(base_path, "Finetuning", "args", env_name, specific_env, "hyperparameters.json")
    
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Configuration file not found: {args_file}")
    
    with open(args_file, 'r') as f:
        args = json.load(f)
    
    env_details = args.get('env_details', {})
    pretrained_models = args.get('pretrained_models', {})
    model_updates = args.get('model_updates', {})
    finetuning_hyperparams = args.get('finetuning_hyperparameters', {})
    exploration_hyperparams = args.get('exploration_hyperparameters', {})
    
    # Build Acc_AdjointMatchingConfig
    am_config_dict = args.get('adjoint_matching_config', {})
    AMConfig = Acc_AdjointMatchingConfig(**am_config_dict)
    
    # Build RewardConfig
    reward_config_dict = args.get('reward_config', {})
    if 'delta' in reward_config_dict and isinstance(reward_config_dict['delta'], str):
        delta_str = reward_config_dict['delta']
        if 'tensor' in delta_str.lower():
            import re
            match = re.search(r'([+-]?[0-9]*\.?[0-9]+)', delta_str)
            if match:
                reward_config_dict['delta'] = float(match.group(1))
            else:
                reward_config_dict['delta'] = None
    RWConfig = RewardConfig(**reward_config_dict)
    alpha_config_dict =  args.get('alpha_config', {})
    AlphaConfig = AlphaSchedulerConfig(**alpha_config_dict)

    
    # Build Train_Reward_Config
    reward_training_dict = args.get('reward_training', {})
    if 'train_goal' in reward_training_dict and reward_training_dict['train_goal'] is not None:
        reward_training_dict['train_goal'] = np.array(reward_training_dict['train_goal'])
    if 'rollout_goal' in reward_training_dict and reward_training_dict['rollout_goal'] is not None:
        reward_training_dict['rollout_goal'] = np.array(reward_training_dict['rollout_goal'])
    if 'rollout_start_cells' in reward_training_dict and reward_training_dict['rollout_start_cells'] is not None:
        reward_training_dict['rollout_start_cells'] = np.array(reward_training_dict['rollout_start_cells'])
    TrainRewardConfig = Train_Reward_Config(**reward_training_dict)
    
    # Build Train_Kernel_Config
    kernel_training_dict = args.get('kernel_training', {})
    if 'λ_reg' in kernel_training_dict:
        pass
    elif 'lambda_reg' in kernel_training_dict:
        kernel_training_dict['λ_reg'] = kernel_training_dict.pop('lambda_reg')
    TrainKernelConfig = Train_Kernel_Config(**kernel_training_dict)
    
    # Build Train_Critic_Config (handle None when critic=False)
    critic_training_dict = args.get('critic_training') or {}
    TrainCriticConfig = Train_Critic_Config(**critic_training_dict)
    
    # Use saved critic/kernel flags when available, else infer from critic_training presence
    use_critic = model_updates.get('critic', args.get('critic_training') is not None)
    use_kernel = model_updates.get('kernel', False)
    
    FTConfig = FinetuningConfig(
        AMConfig=AMConfig,
        RewardConfig=RWConfig,
        AlphaConfig=AlphaConfig,
        dataset_name=env_details.get('dataset_name', env_name),
        specific_dataset=env_details.get('specific_dataset', specific_env),
        planner_checkpoint=pretrained_models.get('planner_checkpoint', 0),
        reward_model_checkpoint=pretrained_models.get('reward_model_checkpoint', 0),
        kernel_model_checkpoint=pretrained_models.get('kernel_model_checkpoint', 0),
        critic_model_checkpoint=pretrained_models.get('critic_model_checkpoint', 0),
        critic=use_critic,
        kernel=use_kernel,
        buffer_size=finetuning_hyperparams.get('buffer_size', 5500),
        finetune_steps=finetuning_hyperparams.get('finetune_total_steps', 3000),
        finetune_rounds=finetuning_hyperparams.get('finetune_rounds', 300),
        diffusion_steps=finetuning_hyperparams.get('diffusion_steps', 50),
        karras_percent=finetuning_hyperparams.get('karras_percent', 0.05),
        Loss_Clip_percent=finetuning_hyperparams.get('Loss_Clip_percent', 0.75),
        finetune_batch_size=finetuning_hyperparams.get('finetune_batch_size', 8),
        finetune_batch_per_sample=finetuning_hyperparams.get('finetune_batch_per_sample', 3),
        finetune_lr=finetuning_hyperparams.get('finetune_lr', 2e-5),
        initial_lam=finetuning_hyperparams.get('initial_lam', 0.05),
        eta_lam=finetuning_hyperparams.get('eta_lam', 0.5),
        gradient_accumulate_every=finetuning_hyperparams.get('gradient_accumulate_every', 1),
        update_lambda_every=finetuning_hyperparams.get('update_lambda_every', 1),
        reward_scaling_factor=finetuning_hyperparams.get('reward_scaling_factor', 50),
        MaxEnt=finetuning_hyperparams.get('MaxEnt', False),
        Entropy_Scaling_Factor=finetuning_hyperparams.get('Entropy_Scaling_Factor', 0.5),
        rollout_length=exploration_hyperparams.get('rollout_length', 2000),
        rollout_num_envs=exploration_hyperparams.get('rollout_num_envs', 1),
        num_rollout_processes=exploration_hyperparams.get('num_rollout_processes'),
        continual_rollout=exploration_hyperparams.get('continual_rollout', False),
        train_reward_config=TrainRewardConfig,
        train_kernel_config=TrainKernelConfig,
        train_critic_config=TrainCriticConfig
    )
    return FTConfig

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#finetune_lr = 1e-05,

if __name__ == "__main__":
    # Example usage of the Adjoint Matching training without a dataset.
    # In practice, 
    # 
    # replace the reward and backbone initialisations with
    # loading of your pretrained models (e.g. via torch.load).
    """
    FTConfig = load_finetuning_args('pointmaze', 'medium')
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()
    """
    

    """
    env_name = 'pointmaze'
    specific_env = 'medium'
    AlphaConfig = AlphaSchedulerConfig(alpha_start = 1.0, alpha_end = 0.01, total_steps = 300, decay = True)
    AMConfig = Acc_AdjointMatchingConfig(horizon = 32)

    #RWConfig = RewardConfig(beta = 1.0, min_log_prob = 15.0, explore = False) 
    RWConfig = RewardConfig(
               beta = 1.0, 
               #max_mahalanobis_score = 3.5,
               min_log_prob = 5.0,
               quantile = 0.95,
               critic_gamma = 1.0,
               explore = False,
               constraint_type = 'log_prob') 

    
    TrainRewardConfig = Train_Reward_Config(
                          hidden_layers = 1,
                          hidden_dim = 32,
                          batch_size = 256, 
                          num_steps = 400, 
                          lr = 1e-4, 
                          sigma = 7.0, 
                          target_reward = 20.0, 
                          train_goal = np.array([[-2.5, -2.5]], dtype = np.float32),
                          rollout_goal = np.array([[6, 1]]),
                          rollout_start_cells = np.array([[6,6], [5,4], [2,4], [2,1]]))
      
    TrainKernelConfig = Train_Kernel_Config(
                            batch_size = 256, 
                            num_steps = 1000,
                            lr = 3e-4,
                            ensemble_size = 10,
                            num_hidden_layers = 2,
                            hidden_dim = 256,
                            type_kernel = 'robust',
                            λ_reg = 1e-3)
    
    TrainCriticConfig = Train_Critic_Config(
                            hidden_layers = 1,
                            hidden_dim = 128,
                            batch_size = 256,
                            num_steps = 5000,
                            lr = 1e-05,
                            tau = 0.005,
                            gamma = 0.95,
                            data_conservation = True)
    
    

    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        AlphaConfig = AlphaConfig,
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 0,
        reward_model_checkpoint = 0,
        kernel_model_checkpoint = 0,
        critic_model_checkpoint = 0,
        critic = True,
        kernel = True,
        buffer_size = 5500,
        finetune_steps = 300,
        finetune_rounds = 30,
        diffusion_steps = 50,
        karras_percent = 0.05,
        Loss_Clip_percent = 0.75,
        finetune_batch_size = 12,
        finetune_batch_per_sample = 6,
        finetune_lr = 2e-05,
        initial_lam = 0.05,
        eta_lam = 0.5,
        gradient_accumulate_every = 1,
        update_lambda_every = 1,
        reward_scaling_factor = 50,
        MaxEnt = False,
        Entropy_Scaling_Factor = 0.5,
        rollout_length = 4000,  # or your desired value
        rollout_num_envs = 1, 
        continual_rollout = True,
        num_rollout_processes = 4,
        train_reward_config = TrainRewardConfig,
        train_kernel_config = TrainKernelConfig,
        train_critic_config = TrainCriticConfig) 
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()
    """
    
    


    """
    env_name = 'pointmaze'
    specific_env = 'large'
    AlphaConfig = AlphaSchedulerConfig(alpha_start = 1.0, alpha_end = 0.1, total_steps = 300, decay = False)
    AMConfig = Acc_AdjointMatchingConfig(horizon = 70)

    #RWConfig = RewardConfig(beta = 1.0, min_log_prob = 15.0, explore = False) 
    RWConfig = RewardConfig(
               beta = 1.0, 
               min_log_prob = 17.5, 
               critic_gamma = 1.0,
               explore = False) 

    
    TrainRewardConfig = Train_Reward_Config(
                          hidden_layers = 2,
                          hidden_dim = 128,
                          batch_size = 512, 
                          num_steps = 2000, 
                          lr = 1e-04, 
                          sigma = 50.0, 
                          target_reward = 100.0, 
                          train_goal = np.array([[4.0, -3.0]], dtype = np.float32),
                          rollout_goal = np.array([[7, 10]]),
                          rollout_start_cells =  np.array([[3, 6], [1, 10], [3, 10], [7, 6], [3, 4], [7, 1]]))
      
    TrainKernelConfig = Train_Kernel_Config(
                            batch_size = 256, 
                            num_steps = 1000,
                            lr = 3e-4,
                            ensemble_size = 10,
                            num_hidden_layers = 2,
                            hidden_dim = 256,
                            λ_reg = 1e-3)
    
    TrainCriticConfig = Train_Critic_Config(
                            hidden_layers = 2,
                            hidden_dim = 512,
                            batch_size = 512,
                            num_steps = 10000,
                            lr = 3e-05,
                            tau = 0.005,
                            gamma = 0.99,
                            data_conservation = True)
    


    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        AlphaConfig = AlphaConfig,
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 0,
        reward_model_checkpoint = 0,
        kernel_model_checkpoint = 0,
        critic_model_checkpoint = 0,
        critic = True,
        kernel = False,
        buffer_size = 5500,
        finetune_steps = 300,
        finetune_rounds = 30,
        diffusion_steps = 50,
        karras_percent = 0.05,
        Loss_Clip_percent = 0.75,
        finetune_batch_size = 32,
        finetune_batch_per_sample = 3,
        finetune_lr = 2e-05,
        initial_lam = 0.05,
        eta_lam = 0.5,
        gradient_accumulate_every = 5,
        update_lambda_every = 1,
        reward_scaling_factor = 10,
        MaxEnt = False,
        Entropy_Scaling_Factor = 0.5,
        rollout_length = 8000,  # or your desired value
        rollout_num_envs = 1, 
        continual_rollout = True,
        num_rollout_processes = 4,
        train_reward_config = TrainRewardConfig,
        train_kernel_config = TrainKernelConfig,
        train_critic_config = TrainCriticConfig) 
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()
   """

    

    
    env_name = 'cube'
    specific_env = 'single-play'
    AlphaConfig = AlphaSchedulerConfig(alpha_start = 1.0, alpha_end = 0.01, total_steps = 300, decay = True)
    AMConfig = Acc_AdjointMatchingConfig(horizon = 32)

    #RWConfig = RewardConfig(beta = 1.0, min_log_prob = 15.0, explore = False) 
    RWConfig = RewardConfig(
               beta = 1.0, 
               quantile = 0.999,
               min_log_prob = -110.0,
               constraint_adapt = True,
               number_of_generated_plans = 32,
               #max_mahalanobis_score = 100.0,
               critic_gamma = 1.0,
               explore = False,
               constraint_type = 'log_prob') 
  
    TrainRewardConfig = Train_Reward_Config(
                          hidden_layers = 4,
                          hidden_dim = 512,
                          batch_size = 256, 
                          num_steps = 5000, 
                          lr = 1e-04, 
                          sigma = 3.0, 
                          target_reward = 50.0, 
                          train_goal = None,
                          task_id = 4)
      
    TrainKernelConfig = Train_Kernel_Config(
                            batch_size = 512, 
                            num_steps = 5000,
                            lr = 1e-4,
                            ensemble_size = 10,
                            num_hidden_layers = 4,
                            hidden_dim = 514,
                            type_kernel = 'mog',
                            kernel_num_modes = 10,
                            kernel_noise_floor = 5e-4,
                            λ_reg = 1e-3)
    
    TrainCriticConfig = Train_Critic_Config(
                            hidden_layers = 4,
                            hidden_dim = 512,
                            batch_size = 256,
                            num_steps = 60000,
                            lr = 6e-05,
                            min_lr = 9e-06,
                            tau = 0.005,
                            gamma = 0.99,
                            data_conservation = True)
    
    FTConfig = FinetuningConfig(
        AMConfig = AMConfig, 
        RewardConfig = RWConfig, 
        AlphaConfig = AlphaConfig,
        dataset_name = env_name,
        specific_dataset = specific_env,
        planner_checkpoint = 0,
        reward_model_checkpoint = 0,
        kernel_model_checkpoint = 0,
        critic_model_checkpoint = 0,
        critic = True,
        update_critic = True,
        kernel = True,
        update_kernel = False,
        buffer_size = 20000,
        finetune_buffer_cutoff_length = 50,
        train_buffer_cutoff_length = 200,
        finetune_steps = 300,
        finetune_rounds = 30,
        diffusion_steps = 200,
        karras_percent = 0.05,
        Loss_Clip_percent = 0.75,
        finetune_batch_size = 16,
        finetune_batch_per_sample = 3,
        finetune_lr = 2e-05,
        initial_lam = 0.05,
        eta_lam = 0.5,
        gradient_accumulate_every = 1,
        update_lambda_every = 1,
        reward_scaling_factor = 50,
        MaxEnt = False,
        Entropy_Scaling_Factor = 0.5,
        rollout_length = 4000,  # or your desired value
        rollout_num_envs = 3, 
        continual_rollout = True,
        num_rollout_processes = 8,
        train_reward_config = TrainRewardConfig,
        train_kernel_config = TrainKernelConfig,
        train_critic_config = TrainCriticConfig) 
    set_seed(1)
    OnlineFinetuner = OnlineFinetuner(FTConfig)
    OnlineFinetuner.finetune_planner()
    
    