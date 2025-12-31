import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from dataclasses import dataclass
from gymnasium.vector import AsyncVectorEnv
from utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker
from traj_reward import RewardConfig, TotalReward
from adjoint_matching import AdjointMatchingFineTuner, AdjointMatchingConfig
from acc_adjoint_matching import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
from Finetuning.Rollout import rollout
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName, get_dataset, Planner_Processor
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from typing import List
from utils import TrajectoryDict, rollout_parallel, get_planner, save_planner, train_reward, train_kernel
from Pretrain.Dataset import get_env
from torch.utils.data import DataLoader, DistributedSampler
import torch
import copy
import os
from accelerate import Accelerator
import math
import numpy as np
from typing import Optional, Dict

@dataclass
class Train_Reward_Config: 
    batch_size: int = 32
    num_steps: int = 1000
    lr: float = 2e-4
    sigma: float = 7.0
    target_reward: Optional[float] = None
    goal: Optional[np.ndarray] = None

@dataclass
class Train_Kernel_Config:
    batch_size: int = 256
    num_steps: int = 1000
    lr: float = 1e-3
    ensemble_size: int = 10
    λ_reg: float = 1e-3

@dataclass
class FinetuningConfig():
    AMConfig: AdjointMatchingConfig | Acc_AdjointMatchingConfig
    RewardConfig: RewardConfig
    dataset_name: str
    specific_dataset: str
    planner_checkpoint: int
    reward_model_checkpoint: int
    kernel_model_checkpoint: int
    train_reward_config: Train_Reward_Config  # Moved before fields with defaults
    train_kernel_config: Train_Kernel_Config  # Moved before fields with defaults
    finetune_steps: int = 1000000
    finetune_rounds: int = 10
    diffusion_steps: int = 30
    karras_percent: float = 0.05
    Loss_Clip_percent: float = 0.75
    finetune_batch_size: int = 12
    finetune_lr: float = 2e-4
    initial_lam: float = 0.01
    eta_lam: float = 0.001
    gradient_accumulate_every: int = 1
    update_lambda_every: int = 5
    reward_scaling_factor: float = 100000
    MaxEnt: bool = False
    Entropy_Scaling_Factor: float = 0.5
    rollout_length: int = 4000
    rollout_num_envs: int = 4
   
    



class OnlineFinetuner():
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.config.AMConfig.finetune_total_steps = self.config.finetune_steps
        self.config.AMConfig.per_round_steps = self.config.finetune_steps // self.config.finetune_rounds
        self.config.AMConfig.diffusion_steps = self.config.diffusion_steps
        self.config.AMConfig.num_karras = math.ceil(self.config.diffusion_steps * self.config.karras_percent)
        self.config.AMConfig.num_Loss_Clip_steps = math.ceil(self.config.diffusion_steps * self.config.Loss_Clip_percent)
        self.config.AMConfig.dataset_name = self.config.dataset_name
        self.config.AMConfig.specific_dataset = self.config.specific_dataset
        self.config.AMConfig.finetune_lr = self.config.finetune_lr
        self.env, d_s, d_a = get_env(self.config.dataset_name, self.config.specific_dataset)
        self.config.AMConfig.d_s = d_s
        self.config.AMConfig.d_a = d_a
        self.config.AMConfig.lam = self.config.initial_lam
        self.config.AMConfig.eta_lam = self.config.eta_lam
        self.config.AMConfig.update_ema_every = self.config.update_lambda_every
        self.config.AMConfig.reward_scaling_factor = self.config.reward_scaling_factor
        self.config.AMConfig.update_lambda_every = self.config.update_lambda_every
        self.config.AMConfig.MaxEnt = self.config.MaxEnt
        self.config.AMConfig.Entropy_Scaling_Factor = self.config.Entropy_Scaling_Factor
       
        self.accelerator = Accelerator(mixed_precision='bf16')
        self.device = self.accelerator.device
        
        self.Initialize_BufferDataset()
        self.set_reward_model(self.device)
        self.AMFineTuner = Acc_AdjointMatchingFineTuner(
                   self.accelerator,
                   self.config.planner_checkpoint, 
                   self.config.AMConfig)
    
    def Initialize_BufferDataset(self):
        self.Buffer = []
        dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
        trajs = dataset.get_trajectories()
        self.Buffer.extend(trajs)
        self.PlannerDataset = PlannerDataset(
                   self.Buffer, 
                   self.config.AMConfig.horizon, 
                   self.config.dataset_name, 
                   self.config.specific_dataset)

    def sync_bufferDataset(self):
        #sync the buffer across all processes
        self.accelerator.wait_for_everyone() 
        if self.accelerator.is_main_process:
             buffer_list = [self.Buffer]  # Wrap in list for gather_object
        else:
             buffer_list = [None]
        gathered = self.accelerator.gather_for_metrics(buffer_list, use_gather_object=True)
        if gathered and gathered[0] is not None:
             self.Buffer = gathered[0]
        self.accelerator.wait_for_everyone() 
        # sync the planner dataset across all processes
        self.PlannerDataset = PlannerDataset(
                    self.Buffer, 
                    self.config.AMConfig.horizon, 
                    self.config.dataset_name, 
                    self.config.specific_dataset
             )
        self.accelerator.wait_for_everyone()
    
    def set_reward_model(self, device):
        self.reward_model = TotalReward(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint)
    
    def finetune_planner(self):
        if self.accelerator.is_main_process:
            print("Env Details: ------------------------------------------------------------------------------")
            print(f"env_name: {self.config.dataset_name}")
            print(f"specific_env: {self.config.specific_dataset}")
            print('Pretrained Model Details: -----------------------------------------------------------------')
            print(f"planner_checkpoint: {self.config.planner_checkpoint}")
            print(f"reward_model_checkpoint: {self.config.reward_model_checkpoint}")
            print(f"kernel_model_checkpoint: {self.config.kernel_model_checkpoint}")
            print('Finetuning Hyperparameters: ---------------------------------------------------------------')
            print(f"finetune_batch_size: {self.config.finetune_batch_size}")
            print(f"finetune_lr: {self.config.AMConfig.finetune_lr}")
            print(f"reward_scaling_factor: {self.config.AMConfig.reward_scaling_factor}")
            print(f"finetune_total_steps: {self.config.AMConfig.finetune_total_steps}")
            print(f"finetuning rounds: {self.config.finetune_rounds}")
            print(f"diffusion_steps: { self.config.AMConfig.diffusion_steps}")
            print(f"karras steps: {self.config.AMConfig.num_karras}")
            print(f"Loss clip steps: {self.config.AMConfig.num_Loss_Clip_steps}")
            print(f"gradient accumulate every: {self.config.gradient_accumulate_every}")
            print(f"Initial lambda: {self.config.AMConfig.lam}")
            print(f"eta_lam: {self.config.AMConfig.eta_lam}")
            print(f"update_lambda_every: {self.config.AMConfig.update_lambda_every}")
            print(f"Differential Entropy Maximization:: {self.config.AMConfig.MaxEnt}")
            print(f"Entropy Scaling Factor: {self.config.AMConfig.Entropy_Scaling_Factor}")
            print(f"Exploration Hyperarameters: --------------------------------------------------------------")
            print(f"Rollout Episode Length: {self.config.rollout_length}")
            print(f"Rollout Number of Environments: {self.config.rollout_num_envs}")
            print(f"Rollout Goal Cell: {self.config.train_reward_config.goal}")
            print('Device Details: ---------------------------------------------------------------------------')
            print(f"The device is: {self.device}")
            print(f"The number of GPUs is: {torch.cuda.device_count()}")
            print(f"The GPU name is: {torch.cuda.get_device_name(0)}")
            print('-------------------------------------------------------------------------------------------')
        


        if self.accelerator.is_main_process:
                print(f"Starting Rollout")
                trajs = rollout_parallel(self.config.dataset_name, 
                                         self.config.specific_dataset, 
                                         horizon = self.config.AMConfig.horizon, 
                                         steps_T = self.config.diffusion_steps, 
                                         num_karras = self.config.AMConfig.num_karras, 
                                         eta = self.config.AMConfig.eta, 
                                         episode_length = self.config.rollout_length, 
                                         checkpoint_step = 0, 
                                         num_envs = self.config.rollout_num_envs, 
                                         goal_cell = self.config.train_reward_config.goal)
        self.accelerator.wait_for_everyone()
        for step in range(self.config.finetune_rounds):
            if (torch.cuda.device_count() > 1):
                sampler = DistributedSampler(self.PlannerDataset, shuffle=True, drop_last=True)
                dataloader = DataLoader(self.PlannerDataset, self.config.finetune_batch_size, pin_memory = True, num_workers = 2,  sampler = sampler,  drop_last = True)
            else:
                dataloader = DataLoader(self.PlannerDataset, self.config.finetune_batch_size, pin_memory = True, num_workers = 2, shuffle = True, drop_last = True)
            if self.accelerator.is_main_process:
                 print(f"Finetuning round {step+1} started")
            self.accelerator.wait_for_everyone()
            self.AMFineTuner.finetune_planner(dataloader, self.reward_model, step+1)
            self.accelerator.wait_for_everyone()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()  
            self.accelerator.wait_for_everyone() 
            
            if self.accelerator.is_main_process:
                print(f"Starting Rollout")
                trajs = rollout_parallel(self.config.dataset_name, 
                                         self.config.specific_dataset, 
                                         horizon = self.config.AMConfig.horizon, 
                                         steps_T = self.config.diffusion_steps, 
                                         num_karras = self.config.AMConfig.num_karras, 
                                         eta = self.config.AMConfig.eta, 
                                         episode_length = self.config.rollout_length, 
                                         checkpoint_step = ((step+1) * self.config.AMConfig.per_round_steps), 
                                         num_envs = self.config.rollout_num_envs, 
                                         goal_cell = self.config.train_reward_config.goal,
                                         device = self.device)
                print(f"Rollout Completed")
                self.Buffer.extend(trajs)
                print(f"Starting Reward Training")
                train_reward(self.Buffer, 
                             dataset_name = self.config.dataset_name, 
                             batch_size = self.config.train_reward_config.batch_size, 
                             num_steps = self.config.train_reward_config.num_steps, 
                             lr = self.config.train_reward_config.lr, 
                             sigma = self.config.train_reward_config.sigma, 
                             step = ((step+1) * self.config.AMConfig.per_round_steps), 
                             target_reward = self.config.train_reward_config.target_reward, 
                             specific_dataset = self.config.specific_dataset, 
                             goal = self.config.train_reward_config.goal)
                print(f"Starting Kernel Training")
                train_kernel(self.Buffer, 
                             dataset_name = self.config.dataset_name, 
                             specific_dataset = self.config.specific_dataset,
                             batch_size = self.config.train_kernel_config.batch_size, 
                             lr = self.config.train_kernel_config.lr, 
                             num_steps = self.config.train_kernel_config.num_steps,
                             ensemble_size = self.config.train_kernel_config.ensemble_size, 
                             λ_reg = self.config.train_kernel_config.λ_reg, 
                             step = ((step+1) * self.config.AMConfig.per_round_steps))
            
            #sync the buffer across all processes
            self.accelerator.wait_for_everyone()
            self.accelerator.wait_for_everyone()
            self.sync_bufferDataset()
    
            #set the new total reward model
            self.config.reward_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
            self.config.kernel_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
            self.set_reward_model(self.device)
            if self.accelerator.is_main_process:
                 print(f"Finetuning round {step+1} completed")
        
     
        
            




     



    










