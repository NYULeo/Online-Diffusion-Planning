import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker
from traj_reward import RewardConfig, TotalReward
from adjoint_matching import AdjointMatchingFineTuner, AdjointMatchingConfig
from acc_adjoint_matching import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName
from Pretrain.Dataset import get_dataset
from typing import List
from utils import TrajectoryDict
from Pretrain.Dataset import get_env
from torch.utils.data import DataLoader, DistributedSampler
import torch
import copy
import os
from accelerate import Accelerator
import torch.multiprocessing as mp


@dataclass
class FinetuningConfig():
    AMConfig: AdjointMatchingConfig | Acc_AdjointMatchingConfig
    RewardConfig: RewardConfig
    dataset_name: str
    specific_dataset: str
    planner_checkpoint: int
    reward_model_checkpoint: int
    kernel_model_checkpoint: int
    finetune_steps: int = 1000000
    finetune_batch_size: int = 12
    finetune_lr: float = 2e-4
    epoch: int = 100
    #save_freq= 10000
    #log_freq = 10
    #step_start_ema = 1000




class OnlineFinetuner():
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.config.AMConfig.finetune_steps = self.config.finetune_steps
        self.config.AMConfig.dataset_name =self.config.dataset_name
        self.config.AMConfig.specific_dataset = self.config.specific_dataset
        self.config.AMConfig.finetune_lr = self.config.finetune_lr
        self.env, d_s, d_a = get_env(self.config.dataset_name, self.config.specific_dataset)
        self.config.AMConfig.d_s = d_s
        self.config.AMConfig.d_a = d_a
       

        self.accelerator = Accelerator(mixed_precision='no')
        self.device = self.accelerator.device

        
        self.Initialize_reward_model(self.device)
      
        self.AMFineTuner = Acc_AdjointMatchingFineTuner(
                   self.accelerator,
                   self.config.planner_checkpoint, 
                   self.config.AMConfig)

        self.Initialize_Buffer()

        self.PlannerDataset = PlannerDataset(self.Buffer, self.config.AMConfig.horizon, self.config.dataset_name, self.config.specific_dataset)
        #self.logdir =  f"./Results/{self.config.dataset_name}/{self.config.specific_dataset}/{'Models'}/"
        #self.reward_tracker = RewardTracker(save_dir="./logs/")
    
    def Initialize_Buffer(self):
        self.Buffer = []
        dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
        trajs = dataset.get_trajectories()
        self.Buffer.extend(trajs)
    
    def Initialize_reward_model(self, device):
        self.reward_model = TotalReward(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint)
 
    def update_dataset(self, trajs: List[TrajectoryDict]):
        self.Buffer.extend(trajs)
        self.PlannerDataset = PlannerDataset(self.Buffer, self.config.AMConfig.horizon, self.config.dataset_name, self.config.specific_dataset)

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
            print(f"finetune_lr: {self.config.finetune_lr}")
            print(f"reward_scaling_factor: {self.config.AMConfig.reward_scaling_factor}")
            print(f"finetune_steps: {self.config.finetune_steps}")
            print(f"sampling steps: {self.config.AMConfig.num_steps}")
            print('Device Details: ---------------------------------------------------------------------------')
            print(f"The device is: {self.device}")
            print(f"The number of GPUs is: {torch.cuda.device_count()}")
            print(f"The GPU name is: {torch.cuda.get_device_name(0)}")
            print('-------------------------------------------------------------------------------------------')
        if (torch.cuda.device_count() > 1):
             sampler = DistributedSampler(self.PlannerDataset, shuffle=True, drop_last=True)
             dataloader = DataLoader(self.PlannerDataset, self.config.finetune_batch_size, pin_memory = True, num_workers = 2,  sampler = sampler,  drop_last = True)
        else:
             dataloader = DataLoader(self.PlannerDataset, self.config.finetune_batch_size, pin_memory = True, num_workers = 2, shuffle = True, drop_last = True)
       
        mp.spawn(self.AMFineTuner.finetune_planner, args=(dataloader, self.reward_model), nprocs = 2)
        self.AMFineTuner.finetune_planner(dataloader, self.reward_model)
            




     



    










