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
from Finetuning.utils import karras_beta_schedule, karras_beta_schedule
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName, get_dataset, Planner_Processor
from typing import List
from utils import TrajectoryDict
from Pretrain.Dataset import get_env
from torch.utils.data import DataLoader, DistributedSampler
from Pretrain.Planners.Backbone.utils import get_pretrained_planner
import torch
import copy
import os
from accelerate import Accelerator
import torch.multiprocessing as mp
import math
import numpy as np



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
    diffusion_steps: int = 30
    karras_percent: float = 0.05
    Loss_Clip_percent: float = 0.75
    finetune_batch_size: int = 12
    finetune_lr: float = 2e-4
    inital_lam: float = 0.01
    eta_lam: float = 0.001
    gradient_accumulate_every: int = 1
    update_lambda_every: int = 5
    reward_scaling_factor: float = 100000
    MaxEnt: bool = False
    Entropy_Scaling_Factor: float = 0.5
    
"""
def rollout_parallel(env_name, specific_env, horizon = 32, steps_T = 50, num_karras = 10, eta = 0.8, episode_length = 4000, critic = False, checkpoint_steps = 1000000, num_envs=8):
     
     #print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     #print(f"Running {num_envs} environments in parallel")
     device = "cuda" if torch.cuda.is_available() else "cpu"
     #print(f"Using device {device}")
     
     # Create environment factory function
     env, d_s, d_a = get_env(env_name, specific_env)
     def make_env():
         env, _, _ = get_env(env_name, specific_env)
         return env
     
     # Create vectorized environment
     vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])
     maze = env.unwrapped.maze  # Access the internal Maze object
     maze_map = maze.maze_map
     rows, cols = len(maze_map), len(maze_map[0])
    
     # Find all free cells (not walls)
     free_cells = []
     for row in range(rows):
       for col in range(cols):
          if maze_map[row][col] != 1:  # 1 = wall; others are free/open
               free_cells.append(np.array([row, col]))
     free_cells = np.array(free_cells)
    
     # Get Planner
     state_dict = get_pretrained_planner(env_name, specific_env, checkpoint_steps)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     
     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env)
     goal_cell  = np.array([6, 1], dtype=int) 
     start_cells = []
     for i in range(len(free_cells)):
        if(np.array_equal(free_cells[i], goal_cell)):
            continue
        else:
            start_cells.append(free_cells[i].copy())
     start_cells = np.array(start_cells)
     normalized_scores = []
     for start_cell in start_cells:
       # Reset all environments
       seeds = list(range(num_envs)) 
       s0_vec = vec_env.reset(seed = seeds, options={"goal_cell": goal_cell, "reset_cell": start_cell})
       current_states = s0_vec[0]['observation']
     
       # Store trajectories for each environment
       all_rewards = [0.0 for _ in range(num_envs)]
       done_envs = [False for _ in range(num_envs)]
       observations = [[] for _ in range(num_envs)]
       acts = [[] for _ in range(num_envs)]
       rewards = [[] for _ in range(num_envs)]
       for env_idx in range(num_envs):
          observations[env_idx].append(current_states[env_idx].copy())
     
       for i in range(episode_length):
          actions = np.zeros((num_envs, d_a))
         
          # Generate actions for each environment
          for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             current_state = current_states[env_idx]
             current_state_norm = planner_processor.preprocess(current_state)
             x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
             action = x[0, d_s:(d_s+d_a)].copy()
             actions[env_idx] = action
         
          # Step all environments at once
          obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
         
          # Update trajectories
          for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             
             observations[env_idx].append(obs_vec['observation'][env_idx].copy())
             acts[env_idx].append(actions[env_idx].copy())
             rewards[env_idx].append(rewards_vec[env_idx])
             all_rewards[env_idx] += rewards_vec[env_idx]
             
             current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
             if terminated_vec[env_idx] or truncated_vec[env_idx]:
                 done_envs[env_idx] = True
                 #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
          # Check if all environments are done
          if all(done_envs):
             #print("All environments completed!")
             break
         
       #vec_env.close()
     
       # Find the trajectory with the maximum reward
       trajs = [[] for _ in range(num_envs)]
       for env_idx in range(num_envs):
          trajs[env_idx] = {
              'observations': np.asarray(observations[env_idx].copy()),
              'actions': np.asarray(acts[env_idx].copy()),
              'rewards': np.asarray(rewards[env_idx].copy())
          }
          #best_idx = np.argmax(all_rewards)
          #best_reward = all_rewards[best_idx]
          #best_trajectory = trajs[best_idx]
       normalized_scores.append(get_normalized_score(trajs))
       
         # Save the best trajectory in the same format as single rollout
       
     vec_env.close()
"""


class OnlineFinetuner():
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.config.AMConfig.finetune_steps = self.config.finetune_steps
        self.config.AMConfig.diffusion_steps = self.config.diffusion_steps
        self.config.AMConfig.num_karras = math.ceil(self.config.diffusion_steps * self.config.karras_percent)
        self.config.AMConfig.num_Loss_Clip_steps = math.ceil(self.config.diffusion_steps * self.config.Loss_Clip_percent)
        self.config.AMConfig.dataset_name = self.config.dataset_name
        self.config.AMConfig.specific_dataset = self.config.specific_dataset
        self.config.AMConfig.finetune_lr = self.config.finetune_lr
        self.env, d_s, d_a = get_env(self.config.dataset_name, self.config.specific_dataset)
        self.config.AMConfig.d_s = d_s
        self.config.AMConfig.d_a = d_a
        self.config.AMConfig.lam = self.config.inital_lam
        self.config.AMConfig.eta_lam = self.config.eta_lam
        self.config.AMConfig.update_ema_every = self.config.update_lambda_every
        self.config.AMConfig.reward_scaling_factor = self.config.reward_scaling_factor
        self.config.AMConfig.update_lambda_every = self.config.update_lambda_every
        self.config.AMConfig.MaxEnt = self.config.MaxEnt
        self.config.AMConfig.Entropy_Scaling_Factor = self.config.Entropy_Scaling_Factor
       

        self.accelerator = Accelerator(mixed_precision='bf16')
        self.device = self.accelerator.device

        
        self.Initialize_reward_model(self.device)
      
        self.AMFineTuner = Acc_AdjointMatchingFineTuner(
                   self.accelerator,
                   self.config.planner_checkpoint, 
                   self.config.AMConfig)

        self.Initialize_Buffer()

        self.PlannerDataset = PlannerDataset(self.Buffer, self.config.AMConfig.horizon, self.config.dataset_name, self.config.specific_dataset)
        #self.initialize_score_net()
        #self.logdir =  f"./Results/{self.config.dataset_name}/{self.config.specific_dataset}/{'Models'}/"
        #self.reward_tracker = RewardTracker(save_dir="./logs/")
    
    def Initialize_Buffer(self):
        self.Buffer = []
        dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
        trajs = dataset.get_trajectories()
        self.Buffer.extend(trajs)
    
    def initialize_score_net(self):
        state_dict = get_pretrained_planner(self.config.dataset_name, self.config.specific_dataset, self.config.planner_checkpoint)
        if( self.config.dataset_name == 'kitchen'):
              self.model = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'pointmaze'):
              self.model = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        else:
              raise ValueError(f"Invalid Environment: {self.config.dataset_name}")
        self.model.load_state_dict(state_dict)
        for p in self.model.parameters():
              p.requires_grad_(False)
        self.model.eval()


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
            print(f"finetune_lr: {self.config.AMConfig.finetune_lr}")
            print(f"reward_scaling_factor: {self.config.AMConfig.reward_scaling_factor}")
            print(f"finetune_steps: {self.config.AMConfig.finetune_steps}")
            print(f"diffusion_steps: { self.config.AMConfig.diffusion_steps}")
            print(f"karras steps: {self.config.AMConfig.num_karras}")
            print(f"Loss clip steps: {self.config.AMConfig.num_Loss_Clip_steps}")
            print(f"gradient accumulate every: {self.config.gradient_accumulate_every}")
            print(f"Initial lambda: {self.config.AMConfig.lam}")
            print(f"eta_lam: {self.config.AMConfig.eta_lam}")
            print(f"update_lambda_every: {self.config.AMConfig.update_lambda_every}")
            print(f"Differential Entropy Maximization:: {self.config.AMConfig.MaxEnt}")
            print(f"Entropy Scaling Factor: {self.config.AMConfig.Entropy_Scaling_Factor}")
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
        

       
        
        #mp.spawn(self.AMFineTuner.finetune_planner, args=(dataloader, self.reward_model), nprocs = 2)
        self.AMFineTuner.finetune_planner(dataloader, self.reward_model)
        """
        if self.accelerator.is_main_process:
             self.model.load_state_dict(new_score_net_state_dict)
             self.model.eval()
        """
            




     



    










