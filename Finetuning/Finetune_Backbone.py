import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker
from traj_reward import RewardConfig
from adjoint_matching import AdjointMatchingFineTuner, AdjointMatchingConfig
from acc_adjoint_matching import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName
from Pretrain.Dataset import get_dataset
from typing import List
from utils import TrajectoryDict
from Pretrain.Dataset import get_env
from torch.utils.data import DataLoader
import torch
import copy
import os

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
        

        self.AMFineTuner = Acc_AdjointMatchingFineTuner(
            self.config.planner_checkpoint, 
            self.config.reward_model_checkpoint,
            self.config.kernel_model_checkpoint,
            self.config.AMConfig,
            self.config.RewardConfig)

        self.Buffer = []
        dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
        trajs = dataset.get_trajectories()
        self.Buffer.extend(trajs)
        self.PlannerDataset = PlannerDataset(self.Buffer, self.config.AMConfig.horizon, self.config.dataset_name, self.config.specific_dataset)
        #self.logdir =  f"./Results/{self.config.dataset_name}/{self.config.specific_dataset}/{'Models'}/"
        #self.reward_tracker = RewardTracker(save_dir="./logs/")
    
 
    def update_dataset(self, trajs: List[TrajectoryDict]):
        self.Buffer.extend(trajs)
        self.PlannerDataset = PlannerDataset(self.Buffer, self.config.AMConfig.horizon, self.config.dataset_name, self.config.specific_dataset)
    
    """
    def save(self, step):
        self.eval()
        data = {
            'dataset_name': self.config.dataset_name,
            'specific_dataset': self.config.specific_dataset,
            'step': step,
            'ema': self.ema_model.state_dict()
        }
        model_name = get_PlannerName(self.config.dataset_name, self.config.specific_dataset)
        file_name = model_name + '_' + str(step) + '.pt'
        os.makedirs(self.logdir, exist_ok=True)
        savepath = os.path.join(self.logdir, file_name)
        torch.save(data, savepath)
        print(f'Saved model to {savepath}', flush=True)
    """

    def finetune_planner(self):
        print(self.config.AMConfig.device)
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
        print(f"finetune_steps: {self.config.finetune_steps}")
        print(f"sampling steps: {self.config.AMConfig.num_steps}")
        print('-------------------------------------------------------------------------------------------')
        dataloader = cycle(DataLoader(self.PlannerDataset, self.config.finetune_batch_size, shuffle = True, pin_memory = True, num_workers = 8))
        self.AMFineTuner.finetune_planner(dataloader)
            




     



    










