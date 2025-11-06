from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from pandas._libs.tslibs import dt64arr_to_periodarr
from Pretrain.Planners.Backbone.Dit import DiT1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from Finetuning.utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker
from Pretrain.Planners.Backbone.utils import cosine_alpha_sigma, cosine_beta, compute_dot_alpha_beta, get_pretrained_planner
import numpy as np
from Pretrain.Dataset import get_PlannerName
from typing import Optional
from torch import Tensor
from Finetuning.traj_reward import RewardConfig, TotalReward
from torch.utils.data import DataLoader
from Pretrain.Planners.Backbone.UNet import TemporalUnet
from Pretrain.Dataset import get_env
from torch.autograd.functional import jvp 
import copy
try:
    from accelerate import Accelerator
except ImportError:
    raise ImportError("accelerate is required but not installed. Run: pip install accelerate")
from accelerate.utils import broadcast



@dataclass
class Acc_AdjointMatchingConfig:
    """Configuration for the adjoint matching fine‑tuner."""

    horizon: int
    finetune_lr: float = 1e-4
    finetune_steps: int = 10000
    d_s: Optional[int] = None
    d_a: Optional[int] = None
    dataset_name: Optional[str] = None
    specific_dataset: Optional[str] = None
    backbone_name: str = 'transformer'
    eta: float = 0.8
    num_steps: int = 500
    s: float = 0.008  # cosine schedule offset used in base drift
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lam: float = 0.0
    reward_scaling_factor: float = 100000
    step_start_ema = 10
    ema_decay = 0.999
    update_ema_every = 2
    update_lambda_every = 5
    save_freq = 50
    log_freq = 1



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
        self.device = self.accelerator.device
        rank = self.accelerator.process_index
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)
        #torch.manual_seed(42)
        #torch.cuda.manual_seed_all(42)
        
        self.ema = EMA(self.config.ema_decay)
        self.t_asc = torch.linspace(1.0, 0.0, self.config.num_steps + 1, device = self.device)
        self.k = self.kt(self.t_asc) 
        
        self.set_old_score_net(planner_checkpoint)
        self.set_new_score_net()
        self.set_ema_model()
        self.set_optimizer_and_scheduler()
        self.set_lambda()
        self.set_reward_tracker()
    

    def Accelerate_Prepare(self, dataloader: DataLoader, reward_model: TotalReward):
         self.new_score_net, self.old_score_net, self.optimizer, self.scheduler, dataloader, reward_model = self.accelerator.prepare(self.new_score_net, self.old_score_net, self.optimizer, self.scheduler, dataloader, reward_model)
         self.new_score_net.train()
         self.old_score_net.eval()
         return dataloader, reward_model

    def set_ema_model(self):
          self.ema_model = copy.deepcopy(self.new_score_net)
          for p in self.ema_model.parameters():
              p.requires_grad_(False)
          self.ema_model.eval()
          
    def set_lambda(self, beta: Optional[float] = None):
        if beta is None:
           self.Lam = Lambda(lam = self.config.lam, beta = 1.0, eta_lam = self.config.finetune_lr)
        else:
           self.Lam = Lambda(lam = self.config.lam, beta = beta, eta_lam = self.config.finetune_lr)
    
    def sync_lambda(self):
        lam_val = self.Lam.get_lam() if self.accelerator.is_main_process else 0.0
        lam_tensor = torch.tensor(lam_val, dtype = torch.float32,device=self.device)
        lam_tensor = broadcast(lam_tensor, from_process=0)
        self.Lam.set_lam(lam_tensor.item())

    def set_optimizer_and_scheduler(self, new_lr=None, new_steps=None):
          # Use provided values or fall back to config defaults
         lr = new_lr if new_lr is not None else self.config.finetune_lr
         steps = new_steps if new_steps is not None else self.config.finetune_steps
    
          # Create new optimizer
         self.optimizer = torch.optim.AdamW(
             self.new_score_net.parameters(), lr=lr, weight_decay = 1e-2)
    
         # Create new scheduler
         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, steps)
    
    def set_old_score_net(self, planner_checkpoint: int):
        state_dict = get_pretrained_planner(self.config.dataset_name, self.config.specific_dataset, planner_checkpoint)
        if( self.config.dataset_name == 'kitchen'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'pointmaze'):
              self.old_score_net = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        else:
              raise ValueError(f"Invalid Environment: {self.config.dataset_name}")
        self.old_score_net.load_state_dict(state_dict)
        for p in self.old_score_net.parameters():
              p.requires_grad_(False)
        self.old_score_net.eval()

    def set_new_score_net(self):
         if(self.config.backbone_name == 'transformer'):
              self.new_score_net = DiT1d(
                   in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128,
                   d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
              self.new_score_net.load_state_dict(self.old_score_net.state_dict())
              self.new_score_net.train()
         elif(self.config.backbone_name == 'unet'):
              self.new_score_net = TemporalUnet(self.config.horizon, self.config.d_s + self.config.d_a)
              self.new_score_net.load_state_dict(self.old_score_net.state_dict())
              self.new_score_net.train()

    def set_reward_tracker(self):
        self.logdir =  f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/Models/"
        self.reward_tracker = RewardTracker(save_dir=f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/logs/")

    def step_ema(self, step):
        self.ema_model.to(self.device)
        base_new_score_net = self.accelerator.unwrap_model(self.new_score_net)
        if step < self.config.step_start_ema:
            self.ema_model.load_state_dict(base_new_score_net.state_dict())
            return
        self.ema.update_model_average(self.ema_model, base_new_score_net)
        
        
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
        print(f"saved model to {savepath}")
    
    def vector_field(self, x: torch.Tensor, t: torch.Tensor, score_model: DiT1d) -> torch.Tensor:
        # Compute beta(t) from cosine schedule
        k = self.kt(t).detach().to(self.device)
        v = k * x + k * score_model(x, t.unsqueeze(0))
        return v
    
    def sigma_t(self, k: torch.Tensor) -> torch.Tensor:
        if(float(k) < 0):
           return torch.sqrt(-2 * k)
        else:
           raise ValueError(f'K should be negative, but got {k}')

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
        ) ->  torch.Tensor:
        self.new_score_net.eval()

        s0_t = s0.to(self.device)
        if ( (s0_t.shape[0] != self.config.d_s)   ):
             raise ValueError(f"s0 should have shape ({self.config.d_s},), but got {s0_t.shape[-1]}")
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
        return  torch.stack(X).to(self.device)

    def make_a(self, X, reward_model: TotalReward):
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
        #print(f"gradient norm: {gradient.norm().item()}")
        
        #print(f"gradient norm: {gradient.norm()}")
        t_asc_reversed = torch.flip(self.t_asc, dims = [0]).to(self.device)
        k_reversed = torch.flip(self.k, dims = [0]).to(self.device)
        #a0 = (-1 * self.config.reward_scaling_factor * gradient).detach().unsqueeze(0).to(self.device)
        a0 = ( self.config.reward_scaling_factor * gradient).detach().unsqueeze(0).to(self.device)
        #print(f"a0 Norm: {a0.norm().item()}")
        a.append(a0)
        #a.append(torch.zeros_like(gradient).unsqueeze(0).to(self.device))
        for i in range(steps_T - 1):
            #t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            t_now, t_next = t_asc_reversed[i], t_asc_reversed[i+1]
            dt = (t_now - t_next)
            #dt = (t_next - t_now)
            T = X_reversed[i].to(self.device)
            T.requires_grad_(True)
            current_a = a[i].to(self.device) 
           
            y, jvp_out = jvp(self.old_score_net, (T, t_now.unsqueeze(0)), (current_a, torch.zeros_like(t_now.unsqueeze(0)).to(self.device))) 
            Jov_a = jvp_out.to(self.device)
            new_a = current_a  + dt * ( (k_reversed[i] * current_a) + (2 * k_reversed[i] * Jov_a) )
            new_a = new_a.detach().clone().to(self.device)
            a.append(new_a)
        a.reverse()
        for p in base_old_score_net.parameters():
              p.requires_grad_(False)
        return a, reward
           
    def adjoint_matching_loss(
        self,
        traj_x: List[torch.Tensor],
        adjoints: List[torch.Tensor]
    ) -> torch.Tensor:
        Loss = torch.tensor(0.0, device = self.device, requires_grad=True)
        for i in range(len(traj_x)):
            traj_x_i = traj_x[i].detach().to(self.device)
            adjoint_i = adjoints[i].unsqueeze(0).flatten().detach().to(self.device)
            v_new = self.vector_field(traj_x_i, self.t_asc[i].detach().to(self.device), self.new_score_net).squeeze(0).flatten().to(self.device)
            v_old = self.vector_field(traj_x_i, self.t_asc[i].detach().to(self.device), self.old_score_net).squeeze(0).flatten().detach().to(self.device)
            sigma = self.sigma_t(self.k[i]).detach().to(self.device)
            Loss = Loss + ((v_new -  v_old)*(2/sigma) + (sigma * adjoint_i)).pow(2).mean()
        Loss = Loss / len(traj_x)
        return Loss
    
    def step(self, s0_batch: torch.Tensor, reward_model: TotalReward) -> Tuple[float, float, float]:
        # 1. Split batch across processes
        base_reward_model = self.accelerator.unwrap_model(reward_model)
        with self.accelerator.split_between_processes(s0_batch) as local_s0:
            local_trajs = []
            local_final_Cs = []
            for s0 in local_s0:
                s0 = s0.to(self.device)
                traj = self.sample_Traj(s0)  
                local_trajs.append(traj)
                final_x = traj[-1].squeeze(0).to(self.device)
                C_val = base_reward_model.get_c(final_x)
                local_final_Cs.append(C_val)
            local_trajs = torch.stack(local_trajs).to(self.device)
            local_final_Cs = torch.stack(local_final_Cs).mean()
        
        self.accelerator.wait_for_everyone()
        
        local_Cs_det = local_final_Cs.detach()
        # 2. Gather C values and update lambda on main process
        all_final_Cs = self.accelerator.gather_for_metrics(local_Cs_det, use_gather_object=False)
        all_trajs = self.accelerator.gather_for_metrics(local_trajs, use_gather_object=False)
        if self.accelerator.is_main_process:
            total_avgC = float(all_final_Cs.mean().item())
        
       
        self.accelerator.wait_for_everyone()
        
        
        # 3. Compute adjoints, rewards & loss tensors for each trajectory
        with self.accelerator.split_between_processes(all_trajs) as local_trajs2:
            local_loss_tensors = []
            local_rewards = []
            for traj in local_trajs2:
                traj = [traj[i] for i in range(traj.shape[0])]
                adjoint, reward = self.make_a(traj, reward_model)
                loss_tensor = self.adjoint_matching_loss(traj, adjoint)  # tensor with grad
                local_loss_tensors.append(loss_tensor)
                local_rewards.append(reward)
            local_loss = torch.stack(local_loss_tensors).mean()
            local_rewards = torch.stack(local_rewards).mean()
            
            
           
        
        self.accelerator.wait_for_everyone()
          # 4. Gather loss tensors & reward floats across processes
        
        loss_global = self.accelerator.reduce(local_loss, reduction="mean")
         
        # Check whether the loss_global still has gradient
        if self.accelerator.is_main_process:
             print(f"loss_global.requires_grad = {loss_global.requires_grad}")

         # 5. Backward and optimizer step only on main process or all processes?
        self.optimizer.zero_grad()
        self.accelerator.backward(loss_global)
                
        total_grad_norm = 0.0
        for param in self.accelerator.unwrap_model(self.new_score_net).parameters():
             if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        if self.accelerator.is_main_process:
               print(f"Gradient norm before clipping: {total_grad_norm}")
        self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

         # 6. Logging: gather detached metrics
        local_loss_det = local_loss.detach()
        local_rewards_det = local_rewards.detach()
        all_losses = self.accelerator.gather_for_metrics(local_loss_det, use_gather_object=False)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards_det, use_gather_object=False)

        if self.accelerator.is_main_process:
            if isinstance(all_losses, torch.Tensor):
                 avg_loss = float(all_losses.mean().item())
                 avg_reward = float(all_rewards.mean().item())
                 return avg_loss, avg_reward, total_avgC
        
        return 0.0, 0.0, 0.0

    def finetune_planner(self, dataloader: DataLoader, reward_model: TotalReward):
        reward_model.eval()
        self.set_optimizer_and_scheduler()
        self.set_ema_model()
        self.set_lambda(reward_model.get_beta())
        self.set_reward_tracker()
        
        print(f"Starting Preparing")
        dataloader, reward_model = self.Accelerate_Prepare(dataloader, reward_model)
        self.accelerator.wait_for_everyone()
        dataloader = cycle(dataloader)
        print(f"Starting Finetuning")
        
        step = 0
        total_loss = 0.0
        total_reward = 0.0
        total_C = 0.0
        Lambda_C = 0.0
        #total_var_reward = 0.0

       
        conds = next(dataloader)
        while step < self.config.finetune_steps:
             #conds = next(dataloader)
             
             loss, avg_reward, avg_C = self.step(conds, reward_model)
             print(f"Lambda: {self.Lam.get_lam()}")

             self.accelerator.wait_for_everyone()
             
             if self.accelerator.is_main_process:
                total_loss += loss
                total_reward += avg_reward
                total_C += avg_C
                Lambda_C += avg_C
            
                current_lr = self.optimizer.param_groups[0]['lr']
                self.reward_tracker.log_reward(step, avg_reward, current_lr)
                
                self.Lam.set_lam(step*1.0)
                """
                if step % self.config.update_lambda_every == 0:
                     self.Lam.update(Lambda_C / self.config.update_lambda_every)  # compute update only on main process
                     Lambda_C = 0.0
                """
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
                    #self.save(step)
                    model_name = get_PlannerName(self.config.dataset_name, self.config.specific_dataset)
                    self.reward_tracker.save_logs(f"{model_name}_finetune_reward_logs.pkl")
                    self.reward_tracker.plot_reward_curve(
                    save_path=f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/logs/{model_name}_finetune_reward_curve.png",
                    title=f"{model_name} Finetuning Avg Reward",
                    show_lr=True,
                    smooth_window=5,
                  ) 
             
             self.sync_lambda()
             """
             if(step % self.config.update_lambda_every == 0):
                 self.sync_lambda()
             """
             step = step+1
             self.accelerator.wait_for_everyone()
        
