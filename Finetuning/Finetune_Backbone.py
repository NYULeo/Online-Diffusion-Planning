import sys
import os

from ogbench.locomaze import task_id
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from dataclasses import dataclass
from gymnasium.vector import AsyncVectorEnv
from Finetuning.utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker, get_trajs, get_success_trajs, check_Critic, get_kernel, get_new_critic_stats
#from Finetuning.traj_reward import RewardConfig, TotalReward, TotalReward_Critic
from Finetuning.comp_reward import RewardConfig, TotalReward, TotalReward_Critic, TotalReward_Critic_Mahalanobis, TotalReward_Mahalanobis
from adjoint_matching import AdjointMatchingFineTuner, AdjointMatchingConfig
#from acc_adjoint_matching import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
from AM import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
from Finetuning.Rollout import rollout
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName, get_dataset, Planner_Processor
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from typing import List
from utils import TrajectoryDict, rollout_parallel, get_planner, rollout_parallel2, save_planner, train_reward, train_kernel, train_kernel_mog, train_critic, save_trajs, AlphaSchedulerConfig, checktrajs, rollout_parallel3
from Pretrain.Dataset import get_env
from torch.utils.data import DataLoader, DistributedSampler
from accelerate.utils import broadcast
import torch
import copy
import os
from accelerate import Accelerator
import math
import numpy as np
from typing import Optional, Dict
import json
from dataclasses import asdict
from random import random
import random



@dataclass
class Train_Reward_Config: 
    hidden_layers: int = 1
    hidden_dim: int = 128
    batch_size: int = 32
    num_steps: int = 1000
    lr: float = 2e-4
    sigma: float = 7.0
    target_reward: Optional[float] = None
    train_goal: Optional[np.ndarray] = None
    task_id: Optional[int] = None
    rollout_goal: Optional[np.ndarray] = None
    rollout_start_cells: Optional[np.ndarray] = None

@dataclass
class Train_Kernel_Config:
    batch_size: int = 256
    num_steps: int = 1000
    lr: float = 1e-3
    ensemble_size: int = 10
    num_hidden_layers: int = 2
    hidden_dim: int = 256
    type_kernel: str = 'robust' or 'mog'
    kernel_num_modes: Optional[int] = 8
    kernel_noise_floor: Optional[float] = 1e-4
    λ_reg: float = 1e-3

@dataclass
class Train_Critic_Config:
    hidden_layers: int = 2
    hidden_dim: int = 128
    batch_size: int = 256
    num_steps: int = 3000
    lr: float = 5e-05
    min_lr: float = 1e-05
    tau: float = 0.005
    gamma: float = 1.0
    data_conservation: bool = False
    retrain_critic: bool = False
    momentum: float = 0.005

@dataclass
class FinetuningConfig():
    AMConfig: AdjointMatchingConfig | Acc_AdjointMatchingConfig
    RewardConfig: RewardConfig 
    AlphaConfig: AlphaSchedulerConfig
    dataset_name: str
    specific_dataset: str
    planner_checkpoint: int
    reward_model_checkpoint: int
    kernel_model_checkpoint: int
    critic_model_checkpoint: int
    train_reward_config: Train_Reward_Config 
    train_kernel_config: Train_Kernel_Config 
    train_critic_config: Train_Critic_Config
    offline: bool = False
    critic: bool = False
    update_critic: bool = True
    kernel: bool = False
    update_kernel: bool = True
    buffer_size: int = 100000
    finetune_buffer_cutoff_length: Optional[int] = None
    train_buffer_cutoff_length: Optional[int] = None
    finetune_steps: int = 1000000
    finetune_rounds: int = 10
    diffusion_steps: int = 30
    karras_percent: float = 0.05
    Loss_Clip_percent: float = 0.75
    finetune_batch_size: int = 12
    finetune_batch_per_sample: int = 3
    finetune_lr: float = 1e-6
    initial_lam: float = 0.01
    eta_lam: float = 0.001
    gradient_accumulate_every: int = 1
    update_lambda_every: int = 5
    reward_scaling_factor: float = 100000
    MaxEnt: bool = False
    Entropy_Scaling_Factor: float = 0.5
    rollout_length: int = 1000
    rollout_num_envs: int = 1
    num_rollout_processes: Optional[int] = None 
    continual_rollout: bool = False
    chunk_size: int = 10
   
def save_hyperparameters(config: FinetuningConfig, filepath: Optional[str] = None):
    if filepath is None:
        os.makedirs(f"./Finetuning/args/{config.dataset_name}/{config.specific_dataset}/", exist_ok=True)
        filepath = f"./Finetuning/args/{config.dataset_name}/{config.specific_dataset}/hyperparameters.json"
    
    def convert_to_json_serializable(obj):
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            # Handle other custom objects by converting to string
            return str(obj)
        return obj
    
    # Convert all config dataclasses to dictionaries
    hyperparams = {
        'env_details': {
            'dataset_name': config.dataset_name,
            'specific_dataset': config.specific_dataset,
        },
        'pretrained_models': {
            'planner_checkpoint': config.planner_checkpoint,
            'reward_model_checkpoint': config.reward_model_checkpoint,
            'kernel_model_checkpoint': config.kernel_model_checkpoint,
            'critic_model_checkpoint': getattr(config, 'critic_model_checkpoint', None),
        },
        'adjoint_matching_config': convert_to_json_serializable(asdict(config.AMConfig)),
        'reward_config': convert_to_json_serializable(asdict(config.RewardConfig)),
        'alpha_config': convert_to_json_serializable(asdict(config.AlphaConfig)),
        'finetuning_hyperparameters': {
            'finetune_batch_size': config.finetune_batch_size,
            'finetune_batch_per_sample': config.finetune_batch_per_sample,
            'buffer_size': config.buffer_size,
            'finetune_buffer_cutoff_length': config.finetune_buffer_cutoff_length,
            'train_buffer_cutoff_length': config.train_buffer_cutoff_length,
            'finetune_lr': config.finetune_lr,
            'reward_scaling_factor': config.reward_scaling_factor,
            'finetune_total_steps': config.finetune_steps,
            'finetune_rounds': config.finetune_rounds,
            'diffusion_steps': config.diffusion_steps,
            'karras_percent': config.karras_percent,
            'Loss_Clip_percent': config.Loss_Clip_percent,
            'gradient_accumulate_every': config.gradient_accumulate_every,
            'initial_lam': config.initial_lam,
            'eta_lam': config.eta_lam,
            'update_lambda_every': config.update_lambda_every,
            'MaxEnt': config.MaxEnt,
            'Entropy_Scaling_Factor': config.Entropy_Scaling_Factor
        },

        'exploration_hyperparameters': {
            'rollout_length': config.rollout_length,
            'rollout_num_envs': config.rollout_num_envs,
            'num_rollout_processes': config.num_rollout_processes,
            'continual_rollout': config.continual_rollout,
            'rollout_goal': config.train_reward_config.rollout_goal.tolist() if hasattr(config.train_reward_config.rollout_goal, 'tolist') else config.train_reward_config.rollout_goal,
        },

        'model_updates': {
             'critic': config.critic,
             'kernel': config.kernel,
        },

        'reward_training': convert_to_json_serializable(asdict(config.train_reward_config)),
        'kernel_training': convert_to_json_serializable(asdict(config.train_kernel_config)),
        'critic_training': convert_to_json_serializable(asdict(config.train_critic_config)) if config.critic else None,
        'device_info': {
            'device': str(config.AMConfig.device),
            'num_gpus': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    }
    
    # Handle numpy arrays, torch.device, and other non-JSON-serializable types
    hyperparams = convert_to_json_serializable(hyperparams)
    
    # Save with pretty printing (indent=4 makes it human-readable)
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=4, sort_keys=False)
    
    print(f"Hyperparameters saved to {filepath}")

class OnlineFinetuner():
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.config.RewardConfig.num_hidden_layers_kernel = self.config.train_kernel_config.num_hidden_layers
        self.config.RewardConfig.hidden_dim_kernel = self.config.train_kernel_config.hidden_dim
        self.config.RewardConfig.num_hidden_layers_reward = self.config.train_reward_config.hidden_layers
        self.config.RewardConfig.hidden_dim_reward = self.config.train_reward_config.hidden_dim
        self.config.RewardConfig.num_hidden_layers_critic = self.config.train_critic_config.hidden_layers
        self.config.RewardConfig.hidden_dim_critic = self.config.train_critic_config.hidden_dim
        self.config.RewardConfig.type_kernel = self.config.train_kernel_config.type_kernel
        self.config.RewardConfig.kernel_num_modes = self.config.train_kernel_config.kernel_num_modes
        self.config.RewardConfig.kernel_noise_floor = self.config.train_kernel_config.kernel_noise_floor
        self.config.AMConfig.alpha_scheduler_config = self.config.AlphaConfig
        self.config.AMConfig.finetune_total_steps = self.config.finetune_steps
        self.config.AMConfig.batch_per_sample = self.config.finetune_batch_per_sample
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
        self.config.AMConfig.update_kernel = self.config.kernel
        self.config.AMConfig.eta_lam = self.config.eta_lam
        if self.config.AMConfig.update_kernel:
             self.config.AMConfig.lam = self.config.initial_lam
        else:
             self.config.AMConfig.lam = 0.0
        self.config.AMConfig.update_ema_every = self.config.update_lambda_every
        self.config.AMConfig.reward_scaling_factor = self.config.reward_scaling_factor
        self.config.AMConfig.update_lambda_every = self.config.update_lambda_every
        self.config.AMConfig.MaxEnt = self.config.MaxEnt
        self.config.AMConfig.Entropy_Scaling_Factor = self.config.Entropy_Scaling_Factor
       
        self.accelerator = Accelerator(mixed_precision = 'bf16')
        self.device = self.accelerator.device
        
        self.Initialize_BufferDataset()
        self.set_reward_model(self.device)
        self.AMFineTuner = Acc_AdjointMatchingFineTuner(
                   self.accelerator,
                   self.config.planner_checkpoint, 
                   self.config.AMConfig)

    def Initialize_BufferDataset(self):
        self.Finetune_Buffer = []
        self.Train_Buffer = []
        self.Train_Kernel_Buffer = []
        
        if(self.config.train_reward_config.task_id is not None):
            #trajs = get_trajs(self.config.dataset_name, self.config.specific_dataset, 0, self.config.train_reward_config.task_id)
            dataset_reward = get_dataset(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id, traj_length = self.config.train_buffer_cutoff_length)
            trajs_reward = dataset_reward.get_trajectories()
            dataset_kernel = get_dataset(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id)
            trajs_kernel = dataset_kernel.get_trajectories()
            self.Finetune_Buffer.extend(trajs_reward)
            self.Train_Buffer.extend(trajs_reward)
            self.Train_Kernel_Buffer.extend(trajs_kernel)
        else:
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
            trajs = dataset.get_trajectories()
            self.Finetune_Buffer.extend(trajs)
            self.Train_Buffer.extend(trajs)
            self.Train_Kernel_Buffer.extend(trajs)

        self.PlannerDataset = PlannerDataset(
                   self.Finetune_Buffer, 
                   self.config.AMConfig.horizon, 
                   self.config.dataset_name, 
                   self.config.specific_dataset, 
                   self.config.finetune_buffer_cutoff_length)

    def set_reward_model(self, device):
        if self.config.critic:
            critic_exist = check_Critic(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id, step = self.config.critic_model_checkpoint)
            if(not critic_exist):
                if(self.config.RewardConfig.constraint_type == 'log_prob'):
                    self.reward_model = TotalReward(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, task_id = self.config.train_reward_config.task_id)
                else:
                    self.reward_model = TotalReward_Mahalanobis(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, task_id = self.config.train_reward_config.task_id)
            else:
                print(f"Critic exists at step {self.config.kernel_model_checkpoint}")
                if(self.config.RewardConfig.constraint_type == 'log_prob'):
                    self.reward_model = TotalReward_Critic(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, self.config.critic_model_checkpoint, task_id = self.config.train_reward_config.task_id)
                else:
                    self.reward_model = TotalReward_Critic_Mahalanobis(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, self.config.critic_model_checkpoint, task_id = self.config.train_reward_config.task_id)
        else:
            if(self.config.RewardConfig.constraint_type == 'log_prob'):
                 self.reward_model = TotalReward(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, task_id = self.config.train_reward_config.task_id)
            else:
                 self.reward_model = TotalReward_Mahalanobis(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, task_id = self.config.train_reward_config.task_id)
    
    def gather_and_sync_trajs_and_buffer(self, local_trajs):
        # Gather local trajectories from all processes
        gathered_trajs_list = self.accelerator.gather_for_metrics([local_trajs if local_trajs else []], use_gather_object=True)
        self.accelerator.wait_for_everyone()
        
        update_reward = False
        if self.accelerator.is_main_process:
            # Flatten and extend buffer only on main
            collected_trajs = []
            for process_trajs in gathered_trajs_list:
               if process_trajs:
                  collected_trajs.extend(process_trajs)
            
            num_rollout = (self.config.num_rollout_processes 
               if self.config.num_rollout_processes is not None 
               else self.accelerator.num_processes)
            print(f"Rollout Completed: Collected {len(collected_trajs)} trajectories across {num_rollout} rollout processes")
            #print(f"Rollout Completed: Collected {len(collected_trajs)} trajectories across {self.accelerator.num_processes} processes")
            
            self.Train_Kernel_Buffer.extend(collected_trajs)
            success_trajs = get_success_trajs(collected_trajs)
            if(len(success_trajs) > 0):
                 self.Train_Buffer.extend(success_trajs)
                 self.Finetune_Buffer.extend(success_trajs)
                 update_reward = True
            if len(self.Finetune_Buffer) > self.config.buffer_size:
                 num_to_remove = len(self.Finetune_Buffer) - self.config.buffer_size
                 self.Finetune_Buffer = self.Finetune_Buffer[num_to_remove:] 
                 #print(f"Buffer size limited to {self.config.buffer_size}, removed {num_to_remove} oldest trajectories")
    
            # Prepare updated buffer for sync
            buffer_for_sync = [self.Finetune_Buffer]
        else:
            buffer_for_sync = [None]
        
        # Broadcast full updated buffer to all processes
        synced_buffer = self.accelerator.gather_for_metrics(buffer_for_sync, use_gather_object=True)
        if synced_buffer[0] is not None:
             self.Finetune_Buffer = synced_buffer[0]
        flag = torch.tensor([1 if update_reward else 0], device=self.accelerator.device, dtype=torch.int64)
        flag = broadcast(flag, from_process=0)   # accelerate.utils.broadcast
        update_reward = bool(flag.item())

        self.PlannerDataset = PlannerDataset(
                 self.Finetune_Buffer,
                 self.config.AMConfig.horizon,
                 self.config.dataset_name,
                 self.config.specific_dataset,
                 self.config.finetune_buffer_cutoff_length
         )
        return update_reward
   
    def collect_critic_buffer(self, local_trajs):
          # ALL processes must participate in gather_for_metrics (collective operation)
          gathered_trajs_list = self.accelerator.gather_for_metrics([local_trajs if local_trajs else []], use_gather_object=True)
          self.accelerator.wait_for_everyone()
          #train_critic = False
          new_critic_stats = None
          # Only main process needs to process the gathered data
          if self.accelerator.is_main_process:
              critic_buffer = []
              for process_trajs in gathered_trajs_list:
                  if process_trajs:
                      critic_buffer.extend(process_trajs)
              critic_buffer = get_success_trajs(critic_buffer)
              if(len(critic_buffer) > 1):
                  #train_critic = True
                  new_critic_stats = get_new_critic_stats(critic_buffer)
              if self.config.train_critic_config.data_conservation:
                  critic_buffer = self.data_conservation_update(critic_buffer)
              #return critic_buffer, train_critic
          else:
              #return None, train_critic  # Other processes don't need the buffer
              critic_buffer = None
          """
          flag = torch.tensor([1 if train_critic else 0], device=self.accelerator.device, dtype=torch.int64)
          flag = broadcast(flag, from_process=0)  # accelerate.utils.broadcast
          train_critic = bool(flag.item())
          """
          
          return critic_buffer, new_critic_stats
   
    def data_conservation_update(self, critic_buffer):
        if(self.config.train_reward_config.task_id is not None):
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id, traj_length = self.config.train_buffer_cutoff_length)
            trajs = dataset.get_trajectories()
        else:
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
            trajs = dataset.get_trajectories()
        #trajs = self.Train_Buffer.copy()
        
        if(len(critic_buffer) < 2):
             critic_buffer.extend(trajs)
        else:
             half_size_1 = len(trajs) // 2
             half_pretrained_trajs = random.sample(trajs, half_size_1)
             half_size_2 = len(critic_buffer) // 2
             half_buffer_trajs = random.sample(critic_buffer, half_size_2)
             critic_buffer = half_pretrained_trajs + half_buffer_trajs
        return critic_buffer

    """
    def get_generated_plans(self, number_of_generated_plans: int):
        dataloader = cycle(DataLoader(self.PlannerDataset, batch_size = 12, shuffle = False))
        generated_plans = []
        for i in range(number_of_generated_plans):
            s0 = next(dataloader)
            s0 = s0.squeeze(0).cpu().numpy()
            x = sample_euler_karras(s0, 
                               self.AMFineTuner.new_score_net, 
                               self.config.AMConfig.d_s, 
                               self.config.AMConfig.d_a, 
                               self.config.AMConfig.horizon,  
                               self.config.AMConfig.diffusion_steps, 
                               self.config.AMConfig.num_karras, 
                               self.config.AMConfig.eta, 
                               self.device)
            
            generated_plans.append(x)
        return generated_plans
     """

    def get_generated_plans(self, number_of_generated_plans: int):
         # Build global s0 batch deterministically on all ranks
         #  (all processes must run same code before split/gather)
         loader = DataLoader(
                 self.PlannerDataset,
                 batch_size=number_of_generated_plans,
                 shuffle=False,
                 drop_last=False,
          )
         s0_batch = next(iter(loader))  # torch.Tensor, shape (N, d_s) if enough data
          # Optional safety: trim/pad logic if dataset is smaller than N
         s0_batch = s0_batch[:number_of_generated_plans]

          #    Split s0 list across processes
         with self.accelerator.split_between_processes(s0_batch) as local_s0_batch:
            local_generated = []
            for s0 in local_s0_batch:
                 s0_np = s0.detach().cpu().numpy()
                 x = sample_euler_karras(
                    s0_np,
                    self.AMFineTuner.new_score_net,
                    self.config.AMConfig.d_s,
                    self.config.AMConfig.d_a,
                    self.config.AMConfig.horizon,
                    self.config.AMConfig.diffusion_steps,
                    self.config.AMConfig.num_karras,
                    self.config.AMConfig.eta,
                    self.device,
                  )
                 local_generated.append(x)  # each x: np.ndarray (H, d_s+d_a)

         self.accelerator.wait_for_everyone()

         # Gather python lists from all ranks
         gathered = self.accelerator.gather_for_metrics(
                [local_generated], use_gather_object=True
         )

         generated_plans = []
         for per_rank in gathered:
                generated_plans.extend(per_rank)

         return generated_plans[:number_of_generated_plans]
    
    

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
            print(f"Offline: {self.config.offline}")
            print(f"finetune_batch_size: {self.config.finetune_batch_size}")
            print(f"finetune_batch_per_sample: {self.config.finetune_batch_per_sample}")
            print(f"Finetuning Buffer Size: {self.config.buffer_size}")
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
            print(f"Rollout Goal Cell: {self.config.train_reward_config.rollout_goal}")
            print(f"Rollout Start Cells: {self.config.train_reward_config.rollout_start_cells}")
            print(f"Critic: {self.config.critic}")
            print(f"Rollout Episode Length: {self.config.rollout_length}")
            print(f"Continual Rollout: {self.config.continual_rollout}")
            print(f"Number of Rollout Processes: {self.config.num_rollout_processes}")
            print('Reward Training Hyperparameters: ----------------------------------------------------------')
            print(f"Reward Hidden Layers: {self.config.train_reward_config.hidden_layers}")
            print(f'Reward Hidden Dim: {self.config.train_reward_config.hidden_dim}')
            print(f"Reward Batch Size: {self.config.train_reward_config.batch_size}")
            print(f"Reward Number of steps: {self.config.train_reward_config.num_steps}")
            print(f"Reward Learning Rate: {self.config.train_reward_config.lr}")
            print(f"Reward Sigma: {self.config.train_reward_config.sigma}")
            print(f"Reward Target Reward: {self.config.train_reward_config.target_reward}")
            print('Kernel Training Hyperparameters: ----------------------------------------------------------')
            print(f'Kernel Hidden Layers: {self.config.train_kernel_config.num_hidden_layers}')
            print(f'Kernel Hidden Dim: {self.config.train_kernel_config.hidden_dim}')
            print(f'Kernel Ensemble Size: {self.config.train_kernel_config.ensemble_size}')
            print(f'Kernel Batch Size: {self.config.train_kernel_config.batch_size}')
            print(f'Kernel Learning Rate: {self.config.train_kernel_config.lr}')
            print(f'Kernel Number of Steps: {self.config.train_kernel_config.num_steps}')
            print(f'Kernel Lambda Regularization: {self.config.train_kernel_config.λ_reg}')
            if(self.config.critic):
                print(f"Critic Training Hyperarameters: ----------------------------------------------------------")
                print(f"Critic Hidden Layers: {self.config.train_critic_config.hidden_layers}")
                print(f"Critic Hidden Dim: {self.config.train_critic_config.hidden_dim}")
                print(f"Critic Model Checkpoint: {self.config.critic_model_checkpoint}")
                print(f"Critic Batch Size: {self.config.train_critic_config.batch_size}")
                print(f"Critic Number of Steps: {self.config.train_critic_config.num_steps}")
                print(f"Critic Learning Rate: {self.config.train_critic_config.lr}")
                print(f"Critic Tau: {self.config.train_critic_config.tau}")
                print(f"Critic Gamma: {self.config.train_critic_config.gamma}")
                print(f"Data Conservation: {self.config.train_critic_config.data_conservation}")
            print('Device Details: ---------------------------------------------------------------------------')
            print(f"The device is: {self.device}")
            print(f"The number of GPUs is: {torch.cuda.device_count()}")
            print(f"The GPU name is: {torch.cuda.get_device_name(0)}")
            print('-------------------------------------------------------------------------------------------')
        
        if self.accelerator.is_main_process:
             save_hyperparameters(self.config)
        """
        if self.accelerator.is_main_process:
             print(f"Starting Rollout")
             trajs, score, _ = rollout_parallel(self.config.dataset_name, 
                                         self.config.specific_dataset, 
                                         horizon = self.config.AMConfig.horizon, 
                                         steps_T = self.config.diffusion_steps, 
                                         num_karras = self.config.AMConfig.num_karras, 
                                         eta = self.config.AMConfig.eta, 
                                         episode_length = self.config.rollout_length, 
                                         checkpoint_step = 0, 
                                         num_envs = 4, 
                                         goal_cell = self.config.train_reward_config.rollout_goal)
             print(f"Total Number of Environment Steps: {0}")
             print(f"Average Normalized Score: {score:.2f}")
        """
        self.accelerator.wait_for_everyone()
        
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        num_envs_per_process = self.config.rollout_num_envs  # Total envs = base * world_size
        last_critic_update_step = 0
        last_reward_update_step = 0
        for step in range(self.config.finetune_rounds):
            if (torch.cuda.device_count() > 1):
                world_size = self.accelerator.num_processes
                num_workers = min(8, max(1, os.cpu_count() // (2 * world_size)))  # 
                sampler = DistributedSampler(self.PlannerDataset, shuffle=True, drop_last=True)
                sampler.set_epoch(step)
                """
                dataloader = DataLoader(
                    self.PlannerDataset, 
                    self.config.finetune_batch_size, 
                    pin_memory = True, 
                    num_workers = (os.cpu_count() // 2),  
                    sampler = sampler,  
                    drop_last = True)
                """
                dataloader = DataLoader(
                    self.PlannerDataset,
                    self.config.finetune_batch_size,
                    sampler = sampler,
                    drop_last = True,
                    pin_memory = True,
                    num_workers = num_workers,
                    persistent_workers = True,
                    prefetch_factor = 4)

            else:
                dataloader = DataLoader(
                    self.PlannerDataset, 
                    self.config.finetune_batch_size, 
                    pin_memory = True, 
                    num_workers = (os.cpu_count() // 2), 
                    #num_workers = 0,
                    shuffle = True, 
                    drop_last = True)
            
            if self.accelerator.is_main_process:
                 print(f"Finetuning round {step+1} started")
                 print(f"Max Mahalanobis Score: {self.config.RewardConfig.max_mahalanobis_score}")
            
            self.AMFineTuner.finetune_planner(dataloader, self.reward_model, step+1)
            self.accelerator.wait_for_everyone()
            #if self.accelerator.is_main_process:
            if(self.config.RewardConfig.constraint_adapt == True):
                x_generated_plans = self.get_generated_plans(self.config.RewardConfig.number_of_generated_plans)
                if self.accelerator.is_main_process:
                    print(f"Generated {len(x_generated_plans)} plans for cosntraint adaptation")
            else:
                    x_generated_plans = None
            self.accelerator.wait_for_everyone()

            if torch.cuda.is_available():
                  torch.cuda.synchronize()  
            self.accelerator.wait_for_everyone() 

            if self.accelerator.is_main_process:
                  print(f"Starting Rollout")
                  
            
            num_rollout_procs = self.config.num_rollout_processes
            do_rollout = (num_rollout_procs is None) or (rank < num_rollout_procs)
            #seed_base = rank * num_envs_per_process
            """
            trajs, score, total_steps = rollout_parallel(self.config.dataset_name, 
                                         self.config.specific_dataset, 
                                         horizon = self.config.AMConfig.horizon, 
                                         steps_T = self.config.diffusion_steps, 
                                         num_karras = self.config.AMConfig.num_karras, 
                                         eta = self.config.AMConfig.eta, 
                                         episode_length = self.config.rollout_length, 
                                         checkpoint_step = ((step+1) * self.config.AMConfig.per_round_steps), 
                                         num_envs = self.config.rollout_num_envs, 
                                         goal_cell = self.config.train_reward_config.rollout_goal,
                                         device = self.device,
                                         start_cells = self.config.train_reward_config.rollout_start_cells,
                                         seed_base = seed_base) 
            """ 
            if do_rollout:
                seed_base = rank * num_envs_per_process
                """
                trajs, score, success_rate, total_steps = rollout_parallel3(self.config.dataset_name, 
                                             self.config.specific_dataset, 
                                             horizon = self.config.AMConfig.horizon, 
                                             steps_T = self.config.diffusion_steps, 
                                             num_karras = self.config.AMConfig.num_karras, 
                                             eta = self.config.AMConfig.eta, 
                                             episode_length = self.config.rollout_length, 
                                             checkpoint_step = ((step+1) * self.config.AMConfig.per_round_steps), 
                                             num_envs = self.config.rollout_num_envs, 
                                             goal_cell = self.config.train_reward_config.rollout_goal,
                                             device = self.device,
                                             start_cells = self.config.train_reward_config.rollout_start_cells,
                                             task_id = self.config.train_reward_config.task_id,
                                             seed_base = seed_base,
                                             continual_rollout = self.config.continual_rollout,
                                             chunk_size = self.config.chunk_size)
                """
                
                trajs, score,  total_steps = rollout_parallel2(self.config.dataset_name, 
                                             self.config.specific_dataset, 
                                             horizon = self.config.AMConfig.horizon, 
                                             steps_T = self.config.diffusion_steps, 
                                             num_karras = self.config.AMConfig.num_karras, 
                                             eta = self.config.AMConfig.eta, 
                                             episode_length = self.config.rollout_length, 
                                             checkpoint_step = ((step+1) * self.config.AMConfig.per_round_steps), 
                                             num_envs = self.config.rollout_num_envs, 
                                             goal_cell = self.config.train_reward_config.rollout_goal,
                                             device = self.device,
                                             start_cells = self.config.train_reward_config.rollout_start_cells,
                                             task_id = self.config.train_reward_config.task_id,
                                             seed_base = seed_base,
                                             continual_rollout = self.config.continual_rollout,
                                             chunk_size = self.config.chunk_size)
       
                #trajs = get_success_trajs(trajs)
                #print(checktrajs(trajs)) 
            else:
                trajs, score, total_steps = [], 0.0, 0.0, 0
            
            self.accelerator.wait_for_everyone()                    
            
            if self.accelerator.is_main_process:
                  print(f"Rollout Completed")
            
            update_reward = self.gather_and_sync_trajs_and_buffer(trajs)
            if self.config.critic:
                 critic_buffer, new_critic_stats = self.collect_critic_buffer(trajs)
                 if self.accelerator.is_main_process:
                     print(f"Number of trajectories for critic training: {len(critic_buffer)}")
                     if(new_critic_stats is not None):
                         print("Training Critic")
                     else:
                         print("Do not Train Critic")
                 self.accelerator.wait_for_everyone()
                 
            
            #collect the score and number of env stepsacross all processes
            gathered_scores = self.accelerator.gather_for_metrics(torch.tensor([score], device=self.device, dtype = torch.float32),  use_gather_object=False)
            #gathered_success_rates = self.accelerator.gather_for_metrics(torch.tensor([success_rate], device=self.device, dtype = torch.float32), use_gather_object=False)
            gathered_steps = self.accelerator.gather_for_metrics(torch.tensor([total_steps], device=self.device, dtype = torch.int64),  use_gather_object=False)
            if self.accelerator.is_main_process:
                 total_steps = gathered_steps.int().sum().item()
                 num_rollout = (num_rollout_procs if num_rollout_procs is not None 
                               else self.accelerator.num_processes)
                 rollout_scores = gathered_scores.float()[:num_rollout]
                 #rollout_success_rates = gathered_success_rates.float()[:num_rollout]
                 avg_score = rollout_scores.float().mean().item()
                 #avg_success_rate = rollout_success_rates.float().mean().item()
                 #avg_score = gathered_scores.float().mean().item()
                 print(f"Total Number of Environment Steps: {total_steps}")
                 #print(f"Average Success Rate: {avg_success_rate:.2f}")
                 print(f"Average Normalized Score: {avg_score:.2f}")
            self.accelerator.wait_for_everyone()  
            
            if(self.config.offline):
                if(self.accelerator.is_main_process):
                     print(f"Finetuning round {step+1} completed")
                     print()
                self.accelerator.wait_for_everyone()
                continue

            
            
            if self.accelerator.is_main_process:
                  #print(f"Starting Reward Training")
                  if update_reward:
                      print(f"Starting Reward Training")
                      train_reward(self.Train_Buffer, 
                             dataset_name = self.config.dataset_name, 
                             hidden_layers = self.config.train_reward_config.hidden_layers,
                             hidden_dim = self.config.train_reward_config.hidden_dim,
                             batch_size = self.config.train_reward_config.batch_size, 
                             num_steps = self.config.train_reward_config.num_steps, 
                             lr = self.config.train_reward_config.lr, 
                             sigma = self.config.train_reward_config.sigma, 
                             step = ((step+1) * self.config.AMConfig.per_round_steps), 
                             target_reward = self.config.train_reward_config.target_reward, 
                             specific_dataset = self.config.specific_dataset, 
                             goal = self.config.train_reward_config.train_goal,
                             task_id = self.config.train_reward_config.task_id)
                  """
                  if self.config.kernel:
                      print(f"Starting Kernel Training")
                      if(self.config.train_kernel_config.type_kernel == 'robust'):
                          threshold = train_kernel(self.Train_Kernel_Buffer, 
                             dataset_name = self.config.dataset_name, 
                             specific_dataset = self.config.specific_dataset,
                             batch_size = self.config.train_kernel_config.batch_size, 
                             lr = self.config.train_kernel_config.lr, 
                             num_steps = self.config.train_kernel_config.num_steps,
                             ensemble_size = self.config.train_kernel_config.ensemble_size, 
                             λ_reg = self.config.train_kernel_config.λ_reg, 
                             num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                             hidden_dim = self.config.train_kernel_config.hidden_dim,
                             step = ((step+1) * self.config.AMConfig.per_round_steps),
                             constraint_type = self.config.RewardConfig.constraint_type,
                             quantile = self.config.RewardConfig.quantile, 
                             x_generated_plans = x_generated_plans)
                      
                      elif(self.config.train_kernel_config.type_kernel == 'mog'):
                          threshold = train_kernel_mog(self.Train_Kernel_Buffer,
                                      dataset_name = self.config.dataset_name,
                                      specific_dataset = self.config.specific_dataset,
                                      batch_size = self.config.train_kernel_config.batch_size,
                                      lr = self.config.train_kernel_config.lr,
                                      num_steps = self.config.train_kernel_config.num_steps,
                                      ensemble_size = self.config.train_kernel_config.ensemble_size,
                                      λ_reg = self.config.train_kernel_config.λ_reg,
                                      num_modes = self.config.train_kernel_config.kernel_num_modes,
                                      num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                                      hidden_dim = self.config.train_kernel_config.hidden_dim,
                                      kernel_noise_floor = self.config.train_kernel_config.kernel_noise_floor,
                                      step = ((step+1) * self.config.AMConfig.per_round_steps),
                                      constraint_type = self.config.RewardConfig.constraint_type,
                                      quantile = self.config.RewardConfig.quantile,
                                      x_generated_plans = x_generated_plans)
                  """

                  if self.config.critic and self.config.update_critic and (new_critic_stats is not None):
                      print(f"Starting Critic Training")
                      #save_trajs(critic_buffer, self.config.dataset_name, self.config.specific_dataset, ((step+1) * self.config.AMConfig.per_round_steps))
                      print(f"Number of trajectories of Critic Training: {len(critic_buffer)}")
                      if(self.config.train_critic_config.retrain_critic):
                           new_critic_stats = None
                      train_critic(critic_buffer, 
                                   dataset_name = self.config.dataset_name, 
                                   specific_dataset = self.config.specific_dataset, 
                                   hidden_layers = self.config.train_critic_config.hidden_layers,
                                   hidden_dim = self.config.train_critic_config.hidden_dim,
                                   sigma = self.config.train_reward_config.sigma, 
                                   batch_size = self.config.train_critic_config.batch_size, 
                                   num_steps = self.config.train_critic_config.num_steps, 
                                   gamma = self.config.train_critic_config.gamma, 
                                   horizon = self.config.AMConfig.horizon, 
                                   lr = self.config.train_critic_config.lr, 
                                   min_lr = self.config.train_critic_config.min_lr,
                                   tau = self.config.train_critic_config.tau, 
                                   old_step = last_critic_update_step,
                                   new_step = ((step+1) * self.config.AMConfig.per_round_steps), 
                                   new_stats = new_critic_stats,
                                   momentum = self.config.train_critic_config.momentum,
                                   goal = self.config.train_reward_config.train_goal, 
                                   target_reward = self.config.train_reward_config.target_reward,
                                   task_id = self.config.train_reward_config.task_id)
                                    
            self.accelerator.wait_for_everyone()
            if self.config.kernel and self.config.update_kernel:
                      if self.accelerator.is_main_process:
                           print(f"Starting Kernel Training")
                      if(self.config.train_kernel_config.type_kernel == 'robust'):
                          threshold = train_kernel(self.Train_Kernel_Buffer, 
                             dataset_name = self.config.dataset_name, 
                             specific_dataset = self.config.specific_dataset,
                             batch_size = self.config.train_kernel_config.batch_size, 
                             lr = self.config.train_kernel_config.lr, 
                             num_steps = self.config.train_kernel_config.num_steps,
                             ensemble_size = self.config.train_kernel_config.ensemble_size, 
                             λ_reg = self.config.train_kernel_config.λ_reg, 
                             num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                             hidden_dim = self.config.train_kernel_config.hidden_dim,
                             step = ((step+1) * self.config.AMConfig.per_round_steps),
                             constraint_type = self.config.RewardConfig.constraint_type,
                             quantile = self.config.RewardConfig.quantile, 
                             x_generated_plans = x_generated_plans,
                             accelerator = self.accelerator)
                      
                      elif(self.config.train_kernel_config.type_kernel == 'mog'):
                          threshold = train_kernel_mog(self.Train_Kernel_Buffer,
                                      dataset_name = self.config.dataset_name,
                                      specific_dataset = self.config.specific_dataset,
                                      batch_size = self.config.train_kernel_config.batch_size,
                                      lr = self.config.train_kernel_config.lr,
                                      num_steps = self.config.train_kernel_config.num_steps,
                                      ensemble_size = self.config.train_kernel_config.ensemble_size,
                                      λ_reg = self.config.train_kernel_config.λ_reg,
                                      num_modes = self.config.train_kernel_config.kernel_num_modes,
                                      num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                                      hidden_dim = self.config.train_kernel_config.hidden_dim,
                                      kernel_noise_floor = self.config.train_kernel_config.kernel_noise_floor,
                                      step = ((step+1) * self.config.AMConfig.per_round_steps),
                                      constraint_type = self.config.RewardConfig.constraint_type,
                                      quantile = self.config.RewardConfig.quantile,
                                      x_generated_plans = x_generated_plans,
                                      accelerator = self.accelerator)
                      
                      if threshold is not None:
                          if(self.config.RewardConfig.constraint_type == 'mahalanobis' ):
                               self.config.RewardConfig.max_mahalanobis_score = threshold
                          elif(self.config.RewardConfig.constraint_type == 'log_prob'):
                               self.config.RewardConfig.min_log_prob = threshold
                                        
            """
            self.accelerator.wait_for_everyone()
            stats = torch.tensor([threshold], device = self.accelerator.device)
            stats = broadcast(stats, from_process=0)
            threshold = stats.tolist()[0]
            """
            
            #set the new total reward model
            if update_reward:
                  self.config.reward_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
                  last_reward_update_step = ((step+1) * self.config.AMConfig.per_round_steps)
            else:
                  self.config.reward_model_checkpoint = last_reward_update_step

            if(self.config.kernel and self.config.update_kernel):
                  self.config.kernel_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
            else:
                  self.config.kernel_model_checkpoint = 0
            
            if(self.config.critic and self.config.update_critic):
                if (new_critic_stats is not None):
                     self.config.critic_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
                     last_critic_update_step = ((step+1) * self.config.AMConfig.per_round_steps)
                else:
                     self.config.critic_model_checkpoint = last_critic_update_step
            else:
                 self.config.critic_model_checkpoint = 0
            #if self.config.RewardConfig.max_mahalanobis_score < threshold: 
            
            """
            if(self.config.RewardConfig.constraint_type == 'mahalanobis' ):
                 self.config.RewardConfig.max_mahalanobis_score = threshold
            elif(self.config.RewardConfig.constraint_type == 'log_prob'):
                 self.config.RewardConfig.min_log_prob = threshold
            """
            
            
            #self.config.critic_model_checkpoint = 0
            self.set_reward_model(self.device)
            if self.accelerator.is_main_process:
                   print(f"Finetuning round {step+1} completed")
                   print()
            self.accelerator.wait_for_everyone()
            

     
        
            




     



    










