import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker
from traj_reward import RewardConfig
from adjoint_matching import AdjointMatchingFineTuner, AdjointMatchingConfig
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName
from Pretrain.Dataset import get_dataset
from typing import List
from utils import TrajectoryDict
from torch.utils.data import DataLoader
import torch
import copy
import os

@dataclass
class FinetuningConfig():
    AMConfig: AdjointMatchingConfig
    RewardConfig: RewardConfig
    dataset_name: str
    specific_dataset: str
    planner_checkpoint: int
    reward_model_checkpoint: int
    kernel_model_checkpoint: int
    epoch: int = 100
    finetune_steps: int = 1000000
    finetune_batch_size: int = 64
    ema_decay = 0.999
    update_ema_every = 2
    save_freq= 10000
    log_freq = 1
    step_start_ema = 1000




class OnlineFinetuner():
    def __init__(self, config: FinetuningConfig):
        self.config = config
        
        self.AMFineTuner = AdjointMatchingFineTuner(
            self.config.dataset_name, 
            self.config.specific_dataset, 
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.AMFineTuner.optimizer, self.config.finetune_steps)
        self.ema = EMA(self.config.ema_decay)
        self.ema_model = copy.deepcopy(self.AMFineTuner.new_score_net).to(self.config.AMConfig.device)
        for p in self.ema_model.parameters():
              p.requires_grad_(False)
        self.logdir =  f"./Results/{self.config.dataset_name}/{self.config.specific_dataset}/{'Models'}/"
        self.reward_tracker = RewardTracker(save_dir="./logs/")
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.AMFineTuner.new_score_net.state_dict())

    def step_ema(self, step):
        if step < self.config.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.AMFineTuner.new_score_net)
    


    def update_dataset(self, trajs: List[TrajectoryDict]):
        self.Buffer.extend(trajs)
        self.PlannerDataset = PlannerDataset(self.Buffer, self.config.AMConfig.horizon, self.config.dataset_name, self.config.specific_dataset)
    
    def save(self, step):
        self.ema_model.eval()
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


    def finetune_planner(self):
        print(self.config.AMConfig.device)
        print(f"finetune_batch_size: {self.config.finetune_batch_size}")
        print(f"finetune_steps: {self.config.finetune_steps}")
        dataloader = cycle(DataLoader(self.PlannerDataset, self.config.finetune_batch_size, shuffle = True, pin_memory = True, num_workers = 8))
        total_loss = 0.0
        total_reward = 0.0
        total_C = 0.0
        step = 0
        while step < self.config.finetune_steps:
             conds = next(dataloader)
             loss, avg_reward, avg_C = self.AMFineTuner.step(conds)
             total_loss += loss
             total_reward += avg_reward
             total_C += avg_C
             self.scheduler.step()
             current_lr = self.AMFineTuner.optimizer.param_groups[0]['lr']
             self.reward_tracker.log_reward(step, avg_reward, current_lr)
             
             if ((step % self.config.update_ema_every) == 0):
                self.step_ema(step)

             if ((step % self.config.log_freq) == 0):
                 print('---------------------------------------------------------')
                 print(f"step: {step}, loss {total_loss / self.config.log_freq}")
                 print(f"step: {step}, reward {total_reward / self.config.log_freq}")
                 print(f"step: {step}, constraint {total_C / self.config.log_freq}")
                 total_loss = 0.0
                 total_reward = 0.0
                 total_C = 0.0
             
             if ((step % self.config.save_freq == 0) and (step!=0)):
                  model_name = get_PlannerName(self.config.dataset_name, self.config.specific_dataset)
                  self.reward_tracker.save_logs(f"{model_name}_finetune_reward_logs.pkl")
                  self.reward_tracker.plot_reward_curve(
                  save_path=f"./plots/{self.config.dataset_name}/{self.config.specific_dataset}/{'Plots'}/{model_name}_finetune_reward_curve.png",
                  title=f"{model_name} Finetuning Avg Reward",
                  show_lr=True,
                  smooth_window=50
                  ) 
              
             step = step+1
        #self.save(step)
            




     



    










