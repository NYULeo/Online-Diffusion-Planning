from itertools import accumulate
import torch
from typing import Optional
from .utils import cosine_alpha_sigma, cosine_beta, EMA, cycle
import torch.nn.functional as F
from typing import Dict
import copy
from Dataset import get_env
from torch.utils.data import DataLoader
import numpy as np
from Backbone.Dit import DiT1d
import os
from Dataset import get_PlannerName, PlannerDataset
from utils import set_seed




class SDETrainer:
    def __init__(
        self,
        dataset_name,
        specific_dataset,
        horizon,
        num_steps = 10000,
        batch_size = 64,
        lr=2e-4,
        device: Optional[torch.device] = None,
        update_ema_every = 2,
        step_start_ema = 1000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        save_freq= 2000,
        log_freq = 10,
        s: float = 0.008,                  # cosine offset
        weight_type: str = 'sigma2',         # {"one", "sigma2", "beta"}
        eps: float = 1e-5,                 # clamp for t, ᾱ stability
        

    ):
        self.device = device
        self.dataset_name = dataset_name
        self.specific_dataset = specific_dataset
        self.state_dim, self.action_dim = get_env(self.dataset_name, self.specific_dataset)
        self.model_name = get_PlannerName(self.dataset_name, self.specific_dataset)
        self.model = DiT1d(
            in_dim = (self.state_dim + self.action_dim), emb_dim = 128,
            d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.reset_parameters()
        self.ema = EMA(ema_decay)
        self.horizon = horizon
        self.s = s
        self.weight_type = weight_type
        self.eps = eps
        self.update_ema_every = update_ema_every
        self.gradient_accumulate_every = gradient_accumulate_every
        self.lr = lr
        self.step_start_ema = step_start_ema
        #self.optim = torch.optim.AdamW(self.model.parameters(), self.lr)
        self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
        self.num_steps = num_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.num_steps)
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.logdir = "./Checkpoints/"


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        self.model.eval()
        self.ema_model.eval()
        data = {
            'step': self.step,
            'ema': self.ema_model.state_dict()
        }
        file_mame = self.model_name + '_' + str(epoch) + '.pt'
        os.makedirs(self.logdir, exist_ok=True)
        savepath = os.path.join(self.logdir, file_mame)
        torch.save(data, savepath)
        print(f'Saved model to {savepath}', flush=True)



    def train(self):
        print(self.device)
        dataset = PlannerDataset(self.dataset_name, self.specific_dataset, self.horizon, self.state_dim, self.action_dim)
        dataloader = cycle(DataLoader(dataset, self.batch_size, shuffle = True, pin_memory = True, num_workers = 8))
        print(f"Training planner for {self.dataset_name}-{self.specific_dataset} Dataset")
        print(f"Horizon: {self.horizon}, Epochs: {self.num_steps}, Batch Size: {self.batch_size}, Learning Rate; {self.lr}")
        
        self.model.train()
        self.ema_model.eval()
        for p in self.ema_model.parameters():
              p.requires_grad_(False)
        self.step = 0
        total_loss = 0
        for step in range(self.num_steps):
            for i in range(self.gradient_accumulate_every):
                traj, cond = next(dataloader)
                loss = self.Loss(traj.to(self.device), cond.to(self.device))
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.optim.zero_grad()
            #self.scheduler.step()
            total_loss += loss.item()

            if ((self.step % self.update_ema_every) == 0):
                self.step_ema()
            
            if ((self.step % self.log_freq) == 0):
                print(f"step {self.step} loss {total_loss/self.log_freq}")
                total_loss = 0
            
            if (self.step % self.save_freq == 0):
                self.save(self.step)

            self.step += 1
            
        



    def Loss(self, x0: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:                       # (B,H,D)
        B, H, D = x0.shape
        mask = torch.zeros((B, H, D), dtype = torch.float32, device = self.device)
        y = torch.zeros((B, H, D), dtype = torch.float32, device = self.device)
        mask[:, 0, :self.state_dim] = 1
        y[:, 0, :self.state_dim] = conditions.clone()

        # 1) sample time t ~ U(eps, 1 - eps), per sample (shape: (B,))
        t = torch.rand(B, device=self.device) * (1.0 - 2*self.eps) + self.eps

        # 2) α(t), σ(t) from cosine schedule (return 1D tensors, then expand to (B,1,1))
        alpha, sigma = cosine_alpha_sigma(t, self.s)     # (B,), (B,)
        alpha_b = alpha.view(B, 1, 1)                    # -> (B,1,1) for broadcasting
        sigma_b = sigma.view(B, 1, 1)                    # -> (B,1,1)

        # 3) perturbation
        eps = torch.randn_like(x0, dtype = x0.dtype)                       # (B,H,D)
        x_t = alpha_b * x0 + sigma_b * eps               # (B,H,D)

       
        #x_t = apply_conditioning(x_t, conditions, self.state_dim)
        xt_clamped = mask * y + (1 - mask) * x_t
        # 4) analytic Gaussian score target for VP
        target = -(xt_clamped - alpha_b * x0) / ( sigma_b**2 + 1e-8)   # (B,H,D)  (Song et al.) :contentReference[oaicite:2]{index=2}

        # 5) model prediction (must match (B,H,D)); pass per-sample t
        pred = self.model(xt_clamped, t)                        # (B,H,D)


        # 6) loss weighting λ(t)
        if self.weight_type == "one":
            lam = torch.ones(B, device=self.device)      # classic VP choice
        elif self.weight_type == "sigma2":
            lam = sigma.pow(2)                           # common balancing heuristic (more VE-like)
        elif self.weight_type == "beta":
            beta = cosine_beta(t, self.s)                # g(t)^2 = β(t) for VP-SDE
            lam = beta
        else:
            raise ValueError(f"Unsupported weight_type {self.weight_type}")

        # 7) weighted MSE; λ(t) is per-sample => apply after summing over (H,D)
        diff = (pred - target) * (1 - mask)
        mse = diff.pow(2).sum(dim = (1,2)) 
        loss = (lam * mse).mean()
        loss = loss/((H*D) - self.state_dim)
        return loss



