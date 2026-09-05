from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Pretrain.Planners.Backbone.Dit import DiT1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from Finetuning.utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker, karras_beta_schedule, clip_actions, save_planner, get_planner, getName, AlphaScheduler, AlphaSchedulerConfig, load_dit
from Pretrain.Planners.Backbone.utils import cosine_alpha_sigma, cosine_beta, compute_dot_alpha_beta, get_pretrained_planner
import numpy as np
from Pretrain.Dataset import get_PlannerName
from typing import Optional, Union
from torch import Tensor
#from Finetuning.traj_reward3 import RewardConfig, TotalReward, TotalReward_Critic
from Finetuning.traj_reward4 import RewardConfig, TotalReward, TotalReward_Critic
from torch.utils.data import DataLoader
from Pretrain.Planners.Backbone.UNet import TemporalUnet
from Pretrain.Dataset import get_env
from torch.autograd.functional import jvp
import copy
from torch.cuda.amp import GradScaler
try:
    from accelerate import Accelerator
except ImportError:
    raise ImportError("accelerate is required but not installed. Run: pip install accelerate")
from accelerate.utils import broadcast
from Pretrain.utils import wandb_log
import pickle


@dataclass
class Acc_AdjointMatchingConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    horizon: int
    d_s: Optional[int] = None
    d_a: Optional[int] = None
    dataset_name: Optional[str] = None
    specific_dataset: Optional[str] = None
    task_id: Optional[int] = None
    backbone_name: str = 'transformer'
    backbone_layers: int = 2
    eta: float = 0.8
    diffusion_steps: int = 30
    num_karras: int = 2
    num_Loss_Clip_steps: int = 35
    s: float = 0.008  # cosine schedule offset used in base drift
    sigma_min: float = 0.01
    sigma_max: float = 30.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step_start_ema = 50
    ema_decay = 0.99
    update_ema_every = 6
    finetune_lr: float = 1e-4
    finetune_total_steps: int = 500
    per_round_steps: int = 100
    lam: float = 0.01
    eta_lam: float = 0.001
    batch_per_sample: int = 3
    reward_scaling_factor: float = 100000
    alpha_scheduler_config: Optional[AlphaSchedulerConfig] = None
    update_lambda_every = 3
    update_kernel: bool = False
    MaxEnt: bool = False
    Entropy_Scaling_Factor: float = 0.5

    save_freq = 10
    save_model_freq = 50
    log_freq = 10



class Acc_AdjointMatchingFineTuner:
    """
    Implements fine‑tuning via the adjoint matching algorithm for
    trajectory diffusion models with fixed initial state.

    Given a pretrained score network (frozen), a differentiable reward
    model and a trainable control network, this class simulates
    trajectories with a memoryless noise schedule, solves the lean
    adjoint backwards and computes the adjoint matching loss on the
    unclamped dimensions of the state.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        planner_checkpoint: int,
        AMConfig: Acc_AdjointMatchingConfig,
        ) -> None:

        self.config = AMConfig
        self.accelerator = accelerator
        torch.set_grad_enabled(True)
        self.device = self.accelerator.device
        rank = self.accelerator.process_index
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)

        self.ema = EMA(self.config.ema_decay)
        self.t_asc = torch.linspace(1.0, 0.0, self.config.diffusion_steps + 1, device = self.device)
        self.k = self.kt(self.t_asc)
        self.t_asc_reversed = torch.flip(self.t_asc, dims=[0])
        self.k_reversed = torch.flip(self.k, dims=[0])
        self.t_grid, self.beta_1, self.sigma_grid = karras_beta_schedule(self.config.diffusion_steps, self.config.sigma_min, self.config.sigma_max, self.device)
        self.beta_2 = cosine_beta(self.t_grid, s = self.config.s)

        self.set_old_score_net(planner_checkpoint)
        self.set_new_score_net()
        self.set_ema_model()
        self.set_optimizer_and_scheduler()
        self.set_alpha_scheduler()
        self.set_lambda()
        self.set_reward_tracker()
        self.Initial_Conds = []


    def Accelerate_Prepare(self, dataloader: DataLoader, reward_model: Union[TotalReward, TotalReward_Critic], round: int):
         if round == 1:
              self.new_score_net, self.old_score_net, self.optimizer, self.scheduler, dataloader, reward_model = self.accelerator.prepare(self.new_score_net, self.old_score_net, self.optimizer, self.scheduler, dataloader, reward_model)
         else:
              dataloader, reward_model = self.accelerator.prepare(dataloader, reward_model)
         self.new_score_net.train()
         self.old_score_net.eval()
         return dataloader, reward_model

    """
    def set_ema_model(self):
          self.ema_model = copy.deepcopy(self.new_score_net)
          for p in self.ema_model.parameters():
              p.requires_grad_(False)
          self.ema_model.eval()
    """

    def set_ema_model(self):
        try:
            base_model = self.accelerator.unwrap_model(self.new_score_net)
        except (AttributeError, RuntimeError):
            # If unwrap fails, model is not wrapped yet (e.g., during __init__)
            base_model = self.new_score_net

        self.ema_model = copy.deepcopy(base_model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    def set_lambda(self, beta: Optional[float] = None):
        if beta is None:
           self.Lam = Lambda(lam = self.config.lam, beta = 1.0, eta_lam = self.config.eta_lam)
        else:
           self.Lam = Lambda(lam = self.config.lam, beta = beta, eta_lam = self.config.eta_lam)

    def sync_lambda(self):
        lam_val = self.Lam.get_lam() if self.accelerator.is_main_process else 0.0
        lam_tensor = torch.tensor(lam_val, dtype = torch.float32,device=self.device)
        lam_tensor = broadcast(lam_tensor, from_process=0)
        self.Lam.set_lam(lam_tensor.item())

    def set_optimizer_and_scheduler(self, new_lr=None, new_alpha=None, new_steps=None):
         # Use provided values or fall back to config defaults
         lr = new_lr if new_lr is not None else self.config.finetune_lr
         steps = new_steps if new_steps is not None else self.config.finetune_total_steps

         # Create new optimizer
         self.optimizer = torch.optim.Adam(
             self.new_score_net.parameters(), lr=lr, weight_decay = 1e-2)
         self.optimizer.zero_grad()
         # Create new scheduler
         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, steps)

    def set_alpha_scheduler(self):
        self.alpha_scheduler = AlphaScheduler(config=self.config.alpha_scheduler_config)

    def set_old_score_net(self, planner_checkpoint: int):
        state_dict = get_planner(self.config.dataset_name, self.config.specific_dataset, planner_checkpoint, self.config.task_id)
        #state_dict = get_pretrained_planner(self.config.dataset_name, self.config.specific_dataset, planner_checkpoint)
        """
        if( self.config.dataset_name == 'kitchen'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth = self.config.backbone_layers, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'pointmaze'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth = self.config.backbone_layers, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'cube'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth = self.config.backbone_layers, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'ogpointmaze'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth = self.config.backbone_layers, timestep_emb_type="fourier")
        else:
              raise ValueError(f"Invalid Environment: {self.config.dataset_name}")
        self.old_score_net.load_state_dict(state_dict)
        """
        self.old_score_net = load_dit(self.config.d_s, self.config.d_a, state_dict, self.config.backbone_layers, self.device, self.config.dataset_name, eval_mode = True)
        for p in self.old_score_net.parameters():
              p.requires_grad_(False)
        self.old_score_net.eval()

    """
    def reset_old_score_net(self, old_planner_checkpoint: int):
        state_dict = get_planner(self.config.dataset_name, self.config.specific_dataset, old_planner_checkpoint, self.config.task_id)
        #state_dict = get_pretrained_planner(self.config.dataset_name, self.config.specific_dataset, planner_checkpoint)
        if( self.config.dataset_name == 'kitchen'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'pointmaze'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'cube'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'ogpointmaze'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        else:
              raise ValueError(f"Invalid Environment: {self.config.dataset_name}")
        self.old_score_net.load_state_dict(state_dict)
        for p in self.old_score_net.parameters():
              p.requires_grad_(False)
        self.old_score_net.eval()
    """
    def set_new_score_net(self):
         if(self.config.backbone_name == 'transformer'):
              self.new_score_net = DiT1d(
                   in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128,
                   d_model = 256, n_heads = 256//64, depth = self.config.backbone_layers, timestep_emb_type="fourier")
              self.new_score_net.load_state_dict(self.old_score_net.state_dict())
              self.new_score_net.train()
         elif(self.config.backbone_name == 'unet'):
              self.new_score_net = TemporalUnet(self.config.horizon, self.config.d_s + self.config.d_a)
              self.new_score_net.load_state_dict(self.old_score_net.state_dict())
              self.new_score_net.train()

    def set_reward_tracker(self):
        self.reward_tracker = RewardTracker(save_dir=f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/logs/")

    def step_ema(self, step):
        self.ema_model.to(self.device)
        base_new_score_net = self.accelerator.unwrap_model(self.new_score_net)
        if step < self.config.step_start_ema:
            self.ema_model.load_state_dict(base_new_score_net.state_dict())
            return
        self.ema.update_model_average(self.ema_model, base_new_score_net)

    """
    def save(self, round: int):
        self.logdir = f"./Finetuning/Planners/{self.config.dataset_name}/{self.config.specific_dataset}/"
        self.ema_model.eval()

        data = {
            'dataset_name': self.config.dataset_name,
            'specific_dataset': self.config.specific_dataset,
            'step': round,
            'ema': self.ema_model.state_dict()
        }
        model_name = get_PlannerName(self.config.dataset_name, self.config.specific_dataset)
        file_name = model_name + '_Planner_' + str(round) + '.pt'
        os.makedirs(self.logdir, exist_ok=True)
        savepath = os.path.join(self.logdir, file_name)
        torch.save(data, savepath)
        print(f"saved model to {savepath}")
    """
    def reset_old_score_net(self, old_planner_checkpoint: int):
         state_dict = get_planner(self.config.dataset_name, self.config.specific_dataset,
                             old_planner_checkpoint, self.config.task_id)
         base = self.accelerator.unwrap_model(self.old_score_net)
         base.load_state_dict(state_dict)
         for p in self.old_score_net.parameters():
             p.requires_grad_(False)
         self.old_score_net.eval()

    def set_new_score_net2(self):
         base_old = self.accelerator.unwrap_model(self.old_score_net)
         base_new = self.accelerator.unwrap_model(self.new_score_net)
         base_new.load_state_dict(base_old.state_dict())
         self.new_score_net.train()

    def save_initial_conds(self, step: int):
        filename = 'Initial_Conds_' + str(step) + '.pkl'
        save_dir =  f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/"
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(self.Initial_Conds, f)
        print(f"Initial Conditions saved to {save_path}")

    def vector_field(self, x: torch.Tensor, t: torch.Tensor, score_model: DiT1d) -> torch.Tensor:
        # Compute beta(t) from cosine schedule
        k = self.kt(t).detach().to(self.device)
        v = k * x + k * score_model(x, t.unsqueeze(0))
        return v

    def sigma_t(self, k: torch.Tensor) -> torch.Tensor:
        if(float(k) < 0):
           return torch.sqrt(-2 * k)
        else:
           raise ValueError(f'K should be negative, but got {k.item()}')

    def kt(self, t: torch.Tensor) -> torch.Tensor:
       t = t.clamp(0.0, 1.0 - 1e-3)
       a = (math.pi / 2.0) * ((t + self.config.s) / (1.0 + self.config.s))
       return (-0.5) * (math.pi / (1.0 + self.config.s)) * torch.tan(a)

    def compute_jacobian_vectorized(self, T, t_index):
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       def score_fn(x_flat):
           x_reshaped = x_flat.view_as(T)  # Reshape to original tensor shape
           score = self.old_score_net(x_reshaped, self.t_asc[t_index].unsqueeze(0))
           return score.flatten()  # Return flattened score

       T_flat = T.flatten().detach().requires_grad_(True)

       # Use torch.autograd.functional.jacobian for efficient computation
       try:
           jacobian = torch.autograd.functional.jacobian(score_fn, T_flat, create_graph=False)
           return jacobian
       except Exception as e:
           print(f"Warning: Vectorized Jacobian failed, falling back to element-wise: {e}")
           # Fallback to original method if vectorized fails
           return self._compute_jacobian_elementwise(T, t_index)

    def _compute_jacobian_elementwise(self, T, t_index):
       score = self.old_score_net(T, self.t_asc[t_index].unsqueeze(0))
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       Jov = torch.zeros(H_dim, H_dim, device = self.device)

       for j in range(H_dim):
           # Create one-hot for j-th output element
           grad_outputs = torch.zeros_like(score)
           grad_outputs.view(-1)[j] = 1.0

           # Compute gradient of j-th output w.r.t input
           grad_j = torch.autograd.grad(
                outputs = score,
                inputs = T,
                grad_outputs = grad_outputs,
                create_graph=False,
                retain_graph=False
           )[0]
            # Store j-th row of Jacobian
           Jov[j, :] = grad_j.view(-1)  # [H*dim]
       return Jov

    @torch.no_grad()
    def sample_Traj(self,
        s0: torch.Tensor,
        reward_model: Union[TotalReward, TotalReward_Critic]
        ) ->  torch.Tensor:
        self.new_score_net.eval()

        s0_t = s0.to(self.device)
        if ( (s0_t.shape[0] != self.config.d_s)   ):
             raise ValueError(f"s0 should have shape ({self.config.d_s},), but got {s0_t.shape[0]}")
        dim = self.config.d_s + self.config.d_a

        # Initialize x_T ~ N(0, I) with shape (horizon, dim)
        x = torch.randn(self.config.horizon, dim, dtype=torch.float32, device=self.device).unsqueeze(0)
        conditions = s0_t.unsqueeze(0)
        mask = torch.zeros((1, self.config.horizon, dim), dtype = torch.float32, device = self.device)
        mask[:, 0, :self.config.d_s] = 1
        y = torch.zeros((1, self.config.horizon, dim), dtype = torch.float32, device = self.device)
        y[:, 0, :self.config.d_s] = conditions.clone()
        #x = apply_conditioning(x, conditions, d_s)
        x = mask * y + (1 - mask) * x

        X = []
        X.append(x.detach().clone())
        for i in range(len(self.t_asc) - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now).item()
            score = self.new_score_net(x, t_now.unsqueeze(0))
            #drift = self.k[i] * x

            if self.config.eta > 0:
               noise = torch.randn_like(x)
               noise_scale = self.config.eta * torch.sqrt((-2*self.k[i]) * (-dt))
               x = x + ((self.k[i] * x) +  (2*self.k[i] * score)) * dt + (noise_scale * noise)
            else:
               x = x + ((self.k[i] * x) +  (2*self.k[i] * score)) * dt

            x = mask * y + (1 - mask) * x
            X.append(x.detach().clone().to(self.device))
        #x = apply_conditioning(x, conditions, d_s)
        self.new_score_net.train()
        reward = reward_model.predict(X[-1].squeeze(0).to(self.device), self.Lam.get_lam())
        return  torch.stack(X).to(self.device), reward

    @torch.no_grad()
    def sample_Traj_karras(self,
        s0: torch.Tensor, reward_model: Union[TotalReward, TotalReward_Critic]
        ) ->  torch.Tensor:
        self.new_score_net.eval()

        s0_t = s0.to(self.device)
        dim = self.config.d_s + self.config.d_a

        # Initialize x_T
        x = torch.randn(1, self.config.horizon, dim, dtype=torch.float32, device=self.device) * self.sigma_grid[0]
        mask = torch.zeros(1, self.config.horizon, dim, dtype=torch.float32, device=self.device)
        mask[:, 0, :self.config.d_s] = 1.0
        y = torch.zeros((1, self.config.horizon, dim), dtype = torch.float32, device = self.device)
        y[:, 0, :self.config.d_s] = s0_t.unsqueeze(0)
        x = mask * y + (1 - mask) * x

        X = []
        X.append(x.detach().clone())
        for i in range(self.config.diffusion_steps):
             t_now = self.t_grid[i]
             t_next = self.t_grid[i + 1] if i < self.config.diffusion_steps - 1 else 0.0
             dt = (t_next - t_now).item()
             if( i < self.config.num_karras ):
                  beta_now = self.beta_1[i].item()
             else:
                  beta_now = self.beta_2[i].item()
             # Drift
             drift = -0.5 * beta_now * x
             # Score
             score = self.new_score_net(x, t_now.unsqueeze(0))
            # Euler step
             if self.config.eta > 0:
                 noise = torch.randn_like(x)
                 noise_scale = self.config.eta * math.sqrt(beta_now * (-dt))
                 x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
             else:
                 x = x + (drift - beta_now * score) * dt
             x = mask * y + (1 - mask) * x
             x = clip_actions(x, self.config.d_s)
             X.append(x.detach().clone().to(self.device))

        self.new_score_net.train()
        reward = reward_model.predict(X[-1].squeeze(0).to(self.device), self.Lam.get_lam())
        return torch.stack(X).to(self.device), reward

    @torch.no_grad()
    def sample_trajs_karras_batch(
        self,
        s0_batch: torch.Tensor,
        reward_model: Union[TotalReward, TotalReward_Critic],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch model forwards while preserving the scalar sampler RNG order."""
        self.new_score_net.eval()
        repeated_s0 = s0_batch.to(self.device).repeat_interleave(
            self.config.batch_per_sample, dim=0
        )
        batch_size = repeated_s0.shape[0]
        dim = self.config.d_s + self.config.d_a

        initial_noise = []
        step_noise = []
        for _ in range(batch_size):
            initial = torch.randn(
                1, self.config.horizon, dim,
                dtype=torch.float32, device=self.device,
            )
            initial_noise.append(initial)
            if self.config.eta > 0:
                step_noise.append(
                    [torch.randn_like(initial) for _ in range(self.config.diffusion_steps)]
                )

        x = torch.cat(initial_noise, dim=0) * self.sigma_grid[0]
        mask = torch.zeros_like(x)
        mask[:, 0, :self.config.d_s] = 1.0
        conditioned = torch.zeros_like(x)
        conditioned[:, 0, :self.config.d_s] = repeated_s0
        x = mask * conditioned + (1 - mask) * x

        states = [x.detach().clone()]
        for i in range(self.config.diffusion_steps):
            t_now = self.t_grid[i]
            t_next = self.t_grid[i + 1] if i < self.config.diffusion_steps - 1 else 0.0
            dt = (t_next - t_now).item()
            beta_now = (
                self.beta_1[i].item()
                if i < self.config.num_karras
                else self.beta_2[i].item()
            )
            drift = -0.5 * beta_now * x
            score = self.new_score_net(x, t_now.expand(batch_size))
            if self.config.eta > 0:
                noise = torch.cat([trajectory_noise[i] for trajectory_noise in step_noise], dim=0)
                noise_scale = self.config.eta * math.sqrt(beta_now * (-dt))
                x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
            else:
                x = x + (drift - beta_now * score) * dt
            x = mask * conditioned + (1 - mask) * x
            x = clip_actions(x, self.config.d_s)
            states.append(x.detach().clone())

        self.new_score_net.train()
        rewards = torch.stack(
            [reward_model.predict(plan, self.Lam.get_lam()) for plan in states[-1]]
        )
        trajectories = torch.stack(states, dim=1).unsqueeze(2)
        return trajectories, rewards

    def make_a(self, X, reward_model: Union[TotalReward, TotalReward_Critic], reward_std: float):
        X = [x.to(self.device) if x.device != self.device else x for x in X]
        steps_T = len(X)
        X_reversed = X[::-1]
        a = []
        T = X_reversed[0]
        T_squeezed = T.squeeze(0).to(self.device)
        reward, gradient = reward_model(T_squeezed, self.Lam.get_lam())
        #grad_norm = torch.norm(gradient, p=2).clamp(min=1e-8)
        #gradient = gradient * (1.0 / grad_norm)
        #print(f"Reward Gradeint Norm: {gradient.norm().item()}")
        if(self.config.MaxEnt):
            score = self.old_score_net(T, torch.tensor(0.0).unsqueeze(0).to(self.device))
            EntGrad = -1 * score
            EntGrad = EntGrad.detach().to(self.device)
        else:
            EntGrad = torch.zeros_like(gradient).detach().unsqueeze(0).to(self.device)


        #current_lr = self.optimizer.param_groups[0]['lr']
        alpha = self.alpha_scheduler.get_alpha()
        #a0 =  (-1 * ((self.config.reward_scaling_factor/alpha)/reward_std) * gradient).detach().unsqueeze(0).to(self.device) + (self.config.Entropy_Scaling_Factor * (-1) * EntGrad)
        a0 =  (-1 * ((self.config.reward_scaling_factor/alpha)) * gradient).detach().unsqueeze(0).to(self.device) + (self.config.Entropy_Scaling_Factor * (-1) * EntGrad)
        #print(f"gradient norm: {gradient.norm().item()}")
        #max_norm = 5.0
        #a0 =   a0 * torch.clamp(max_norm / torch.norm(a0), max=1.0)
        #print(f"a0: {a0.norm().item()}")
        if(a0.norm().item() == 0.0):
            print(f"a0 is 0")

        a.append(a0)

        #a.append(torch.zeros_like(gradient).unsqueeze(0).to(self.device))
        for i in range(steps_T - 1):
            #t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            t_now, t_next = self.t_asc_reversed[i], self.t_asc_reversed[i+1]
            dt = (t_now - t_next)
            #dt = (t_next - t_now)
            T = X_reversed[i].to(self.device)
            T.requires_grad_(True)
            current_a = a[i].to(self.device)

            _, jvp_out = jvp(
                self.old_score_net,
                (T, t_now.unsqueeze(0)),
                (current_a, torch.zeros_like(t_now.unsqueeze(0))),
                create_graph=False,
            )
            Jov_a = jvp_out.to(self.device)
            new_a = current_a + dt * (
                self.k_reversed[i] * current_a + 2 * self.k_reversed[i] * Jov_a
            )
            new_a = new_a.detach().clone().to(self.device)
            a.append(new_a)
        a.reverse()
        return a, reward

    def make_a_batch(
        self,
        trajectories: torch.Tensor,
        reward_model: Union[TotalReward, TotalReward_Critic],
        reward_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate adjoints for all local trajectories with one JVP per time point."""
        trajectory_count, step_count = trajectories.shape[:2]
        reversed_states = torch.flip(trajectories[:, :, 0], dims=[1]).to(self.device)
        terminal_adjoints = []
        rewards = []

        alpha = self.alpha_scheduler.get_alpha()

        for terminal_state in reversed_states[:, 0]:
            reward, gradient = reward_model(terminal_state, self.Lam.get_lam())
            if self.config.MaxEnt:
                score = self.old_score_net(
                    terminal_state.unsqueeze(0),
                    torch.zeros(1, device=self.device),
                ).squeeze(0)
                entropy_gradient = -score.detach()
            else:
                entropy_gradient = torch.zeros_like(gradient)
            terminal_adjoint = (
                -1 * (self.config.reward_scaling_factor / alpha) * gradient
                - self.config.Entropy_Scaling_Factor * entropy_gradient
            ).detach()
            terminal_adjoints.append(terminal_adjoint)
            rewards.append(reward)

        current_adjoint = torch.stack(terminal_adjoints, dim=0)
        reversed_adjoints = [current_adjoint]
        for i in range(step_count - 1):
            t_now = self.t_asc_reversed[i]
            t_next = self.t_asc_reversed[i + 1]
            dt = t_now - t_next
            state_batch = reversed_states[:, i]
            time_batch = t_now.expand(trajectory_count)
            _, jvp_out = jvp(
                self.old_score_net,
                (state_batch, time_batch),
                (current_adjoint, torch.zeros_like(time_batch)),
                create_graph=False,
            )
            current_adjoint = (
                current_adjoint
                + dt * (
                    self.k_reversed[i] * current_adjoint
                    + 2 * self.k_reversed[i] * jvp_out
                )
            ).detach()
            reversed_adjoints.append(current_adjoint)

        adjoints = torch.stack(reversed_adjoints[::-1], dim=1)
        return adjoints, torch.stack(rewards)

    """
    def make_a(self, X, reward_model: TotalReward, reward_std: float):
        base_old_score_net = self.accelerator.unwrap_model(self.old_score_net)
        for p in base_old_score_net.parameters():
            p.requires_grad_(True)

        X = [x.to(self.device) if x.device != self.device else x for x in X]
        steps_T = len(X)
        X_reversed = X[::-1]
        a = []
        T = X_reversed[0]
        T_squeezed = T.squeeze(0).to(self.device)

        reward, gradient = reward_model(T_squeezed, self.Lam.get_lam())

        if self.config.MaxEnt:
            score = self.old_score_net(T, torch.tensor(0.0, device=self.device).unsqueeze(0))
            EntGrad = -score.detach()
        else:
            EntGrad = torch.zeros_like(gradient).unsqueeze(0).to(self.device)

        alpha = self.alpha_scheduler.get_alpha()
        a0 = (-1 * ((self.config.reward_scaling_factor / alpha) / reward_std) * gradient).unsqueeze(0).to(self.device) \
             + (self.config.Entropy_Scaling_Factor * (-1) * EntGrad)

        a.append(a0)

        t_asc_reversed = torch.flip(self.t_asc, dims=[0]).to(self.device)
        k_reversed = torch.flip(self.k, dims=[0]).to(self.device)

        for i in range(steps_T - 1):
            dt = (t_asc_reversed[i] - t_asc_reversed[i + 1])
            T = X_reversed[i].to(self.device)
            T.requires_grad_(True)
            current_a = a[i].to(self.device)

            # Use jvp with minimal graph
            _, jvp_out = jvp(
                lambda x, t: self.old_score_net(x, t),
                (T, t_asc_reversed[i].unsqueeze(0)),
                (current_a, torch.zeros_like(t_asc_reversed[i].unsqueeze(0)).to(self.device)),
                create_graph=False
            )
            Jov_a = jvp_out.to(self.device)

            new_a = current_a + dt * ((k_reversed[i] * current_a) + (2 * k_reversed[i] * Jov_a))
            new_a = new_a.detach().clone().to(self.device)
            a.append(new_a)

            # Cleanup inside loop
            torch.cuda.empty_cache()
            gc.collect()

        a.reverse()
        for p in base_old_score_net.parameters():
            p.requires_grad_(False)

        torch.cuda.empty_cache()
        gc.collect()
        return a, reward
    """
    def adjoint_matching_loss(
        self,
        traj_x: List[torch.Tensor],
        adjoints: List[torch.Tensor]
    ) -> torch.Tensor:
        trajectory = torch.cat([state.detach() for state in traj_x], dim=0).to(self.device)
        adjoint = torch.cat([value.detach() for value in adjoints], dim=0).to(self.device)
        step_count = trajectory.shape[0]
        times = self.t_asc[:step_count]
        k_values = self.k[:step_count].view(-1, 1, 1)

        new_score = self.new_score_net(trajectory, times)
        with torch.no_grad():
            old_score = self.old_score_net(trajectory, times)
        v_new = k_values * trajectory + k_values * new_score
        v_old = k_values * trajectory + k_values * old_score
        sigma = torch.sqrt((-2 * self.k[:step_count]).clamp_min(1e-12)).view(-1, 1, 1)
        losses = ((v_new - v_old) * (2 / sigma) + sigma * adjoint).square().mean(dim=(1, 2))

        clip_count = min(self.config.num_Loss_Clip_steps + 1, step_count)
        return losses[clip_count:].sum() / step_count

    def adjoint_matching_loss_batch(
        self,
        trajectories: torch.Tensor,
        adjoints: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the unchanged per-time loss for every local trajectory at once."""
        trajectory_count, step_count, _, horizon, dimension = trajectories.shape
        flat_trajectories = trajectories[:, :, 0].detach().reshape(
            trajectory_count * step_count, horizon, dimension
        )
        flat_adjoints = adjoints.detach().reshape(
            trajectory_count * step_count, horizon, dimension
        )
        times = self.t_asc[:step_count].repeat(trajectory_count)
        k_values = self.k[:step_count].repeat(trajectory_count).view(-1, 1, 1)

        new_score = self.new_score_net(flat_trajectories, times)
        with torch.no_grad():
            old_score = self.old_score_net(flat_trajectories, times)
        v_new = k_values * flat_trajectories + k_values * new_score
        v_old = k_values * flat_trajectories + k_values * old_score
        sigma = torch.sqrt((-2 * k_values).clamp_min(1e-12))
        losses = (
            (v_new - v_old) * (2 / sigma) + sigma * flat_adjoints
        ).square().mean(dim=(1, 2)).reshape(trajectory_count, step_count)

        clip_count = min(self.config.num_Loss_Clip_steps + 1, step_count)
        return losses[:, clip_count:].sum() / (trajectory_count * step_count)

    def step(self, s0_batch: torch.Tensor, reward_model: Union[TotalReward, TotalReward_Critic]) -> Tuple[float, float, float]:
        # 1. Split batch across processes
        base_reward_model = self.accelerator.unwrap_model(reward_model)
        with self.accelerator.split_between_processes(s0_batch) as local_s0:
            if len(local_s0) == 0:
                raise RuntimeError("each rank must receive at least one initial state")
            with self.accelerator.autocast():
                local_trajs, local_rewards = self.sample_trajs_karras_batch(
                    local_s0, base_reward_model
                )
            with torch.no_grad():
                local_final_Cs = torch.stack(
                    [base_reward_model.get_c(plan) for plan in local_trajs[:, -1, 0]]
                )
        local_Cs_det = local_final_Cs.detach()
        # 2. Gather C values and update lambda on main process
        all_final_Cs = self.accelerator.gather_for_metrics(local_Cs_det, use_gather_object = False)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards, use_gather_object = False)
        if self.accelerator.is_main_process:
            total_avgC = float(all_final_Cs.mean().item())
            #reward_std = float(all_rewards.std().item())
            reward_std = float(torch.max(all_rewards).item() - torch.min(all_rewards).item())

        else:
            total_avgC = 0.0
            reward_std = 0.0

        stats = torch.tensor([total_avgC, reward_std], device=self.device)
        stats = broadcast(stats, from_process=0)
        total_avgC, reward_std = stats.tolist()

        # 3. Each trajectory stays on the rank that generated it.  The previous
        # all-gather + immediate re-split transferred the full diffusion history
        # without changing the global objective.
        with self.accelerator.autocast():
            local_adjoints, optimized_rewards = self.make_a_batch(
                local_trajs, reward_model, reward_std
            )
            local_loss = self.adjoint_matching_loss_batch(
                local_trajs, local_adjoints
            )
        local_rewards = optimized_rewards.mean()

        global_loss = self.accelerator.reduce(local_loss, reduction="mean")



        # 5. Backward and optimizer step only on main process or all processes
        """
        self.optimizer.zero_grad()
        self.new_score_net.zero_grad()
        self.accelerator.backward(global_loss)
        self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.alpha_scheduler.step_alpha()
        """
        # 5. Backward and (maybe) optimizer step under accelerate accumulation
        with self.accelerator.accumulate(self.new_score_net):
            self.accelerator.backward(global_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.alpha_scheduler.step_alpha()
                self.optimizer.zero_grad()


         # 6. Logging: gather detached metrics
        local_loss_det = local_loss.detach()
        local_rewards_det = local_rewards.detach()
        all_losses = self.accelerator.gather_for_metrics(local_loss_det, use_gather_object=False)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards_det, use_gather_object=False)

        #if self.accelerator.is_main_process:
        if self.accelerator.is_main_process:
             #if isinstance(all_losses, torch.Tensor):
            avg_loss = float(all_losses.mean().item())
            avg_reward = float(all_rewards.mean().item())
            return avg_loss, avg_reward, total_avgC
        return 0, 0, 0

    def finetune_planner(self, dataloader: DataLoader, reward_model: Union[TotalReward, TotalReward_Critic], round: int, old_planner_checkpoint: Optional[int] = None):
        if old_planner_checkpoint is not None:
            self.reset_old_score_net(old_planner_checkpoint)
            self.set_new_score_net2()
        reward_model.eval()

        if(round > 1):
            self.set_lambda(reward_model.get_beta())
            self.set_ema_model()
            self.set_optimizer_and_scheduler(new_lr = self.optimizer.param_groups[0]['lr'], new_steps = self.config.finetune_total_steps - ((round-1)*self.config.per_round_steps))


        if self.accelerator.is_main_process:
             print(f"Starting Preparing")
        dataloader, reward_model = self.Accelerate_Prepare(dataloader, reward_model, round)
        self.accelerator.wait_for_everyone()
        dataloader = cycle(dataloader)
        if self.accelerator.is_main_process:
             print(f"Starting Finetuning")

        step = 0
        total_loss = 0.0
        total_reward = 0.0
        pure_reward = 0.0
        total_C = 0.0
        Lambda_C = 0.0
        #total_var_reward = 0.0


        #conds = next(dataloader)
        while step < self.config.per_round_steps:
             conds = next(dataloader)
             loss, avg_reward, avg_C = self.step(conds, reward_model)

             if self.accelerator.is_main_process:
                total_loss += loss
                total_reward += avg_reward
                total_C += avg_C
                Lambda_C += avg_C


                Reward = avg_reward + (self.Lam.get_lam() * avg_C)
                self.reward_tracker.log_reward(((round-1)*self.config.per_round_steps+step), Reward, avg_C)
                pure_reward += Reward


                if (step % self.config.update_lambda_every == 0) and (self.config.update_kernel):
                     self.Lam.update(Lambda_C / self.config.update_lambda_every)
                     Lambda_C = 0.0
                     print(f"step: {step}, lambda: {self.Lam.get_lam()}")

                if ((((round-1)*self.config.per_round_steps + step) % self.config.update_ema_every) == 0):
                     self.step_ema(((round-1)*self.config.per_round_steps + step))

                global_step = (round - 1) * self.config.per_round_steps + step
                wandb_log(
                    {
                        "finetune/step": global_step,
                        "finetune/loss": loss,
                        "finetune/reward": avg_reward,
                        "finetune/objective": Reward,
                        "finetune/constraint": avg_C,
                        "finetune/alpha": self.alpha_scheduler.get_alpha(),
                        "finetune/lambda": self.Lam.get_lam(),
                    }
                )

                if ((step % self.config.log_freq) == 0):
                    print('---------------------------------------------------------')
                    if(step == 0):
                         print(f"round: {round}, step: {step}, loss {total_loss}")
                         print(f"round: {round}, step: {step}, total reward {total_reward}")
                         print(f"round: {round}, step: {step}, reward {pure_reward }")
                         print(f"round: {round}, step: {step}, constraint {total_C}")
                         print(f"round: {round}, step: {step}, alpha {self.alpha_scheduler.get_alpha()}")
                    else:
                         print(f"round: {round}, step: {step}, loss {total_loss / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, total reward {total_reward / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, reward {pure_reward / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, constraint {total_C / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, alpha {self.alpha_scheduler.get_alpha()}")
                    total_loss = 0.0
                    total_reward = 0.0
                    pure_reward = 0.0
                    total_C = 0.0


                if ((step % self.config.save_freq == 0) and (step!=0)):
                    model_name = getName(self.config.dataset_name, self.config.specific_dataset)
                    #model_name = get_PlannerName(self.config.dataset_name, self.config.specific_dataset)
                    self.reward_tracker.save_logs(f"{model_name}_step{((round-1)*self.config.per_round_steps+step)}_finetune_reward_logs.pkl")
                    self.reward_tracker.plot_reward_curve(
                    save_path=f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/logs/{model_name}_step{((round-1)*self.config.per_round_steps+step)}_finetune_reward_curve.png",
                    title=f"{model_name} of step {((round-1)*self.config.per_round_steps+step)} Finetuning Avg Reward",
                    show_constraint=True,
                    smooth_window=50,
                  )
                """
                if ( (step % self.config.save_model_freq == 0) and (step!=0)):
                    self.save(step)
                    #self.save_initial_conds(step)
                """
             if(step % self.config.update_lambda_every == 0):
                 self.sync_lambda()

             step = step+1

        if self.accelerator.is_main_process:
             save_planner(self.ema_model, self.config.dataset_name, self.config.specific_dataset, (round*self.config.per_round_steps), task_id = self.config.task_id)
        self.accelerator.wait_for_everyone()
        if torch.cuda.is_available():
             torch.cuda.synchronize()
        self.accelerator.wait_for_everyone()
