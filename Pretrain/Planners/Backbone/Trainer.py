import os
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
import torch
from typing import Optional
from .utils import cosine_alpha_sigma, cosine_beta, EMA, cycle
import torch.nn.functional as F
from typing import Dict
import copy
from Dataset import get_env, determine_stride
from torch.utils.data import DataLoader
import numpy as np
from .Dit import DiT1d
from .UNet import TemporalUnet
import os
from Dataset import get_PlannerName, PlannerDataset, PlannerDataset_Rollout
from .utils import LossTracker, get_pretrained_planner, getName
try:
    from Pretrain.utils import wandb_log
except ModuleNotFoundError:
    from utils import wandb_log
import json


class SDETrainer:
    def __init__(
        self,
        dataset_name,
        specific_dataset,
        task_id,
        horizon,
        backbone_name,
        backbone_layers = 2,
        num_steps = 1000000,
        batch_size = 128,
        lr=2e-4,
        device: Optional[torch.device] = None,
        update_ema_every = 2,
        step_start_ema = 1000,
        gradient_accumulate_every=2,
        ema_decay=0.9999,
        save_freq= 200000,
        log_freq = 10,
        s: float = 0.008,                  # cosine offset
        weight_type: str = 'sigma2',         # {"one", "sigma2", "beta"}
        eps: float = 1e-5,               # clamp for t, ᾱ stability
        stride: Optional[int] = 1,
        data_parallel: bool = False,
    ):
        self.device = device
        self.dataset_name = dataset_name
        self.specific_dataset = specific_dataset
        self.task_id = task_id
        _, self.state_dim, self.action_dim = get_env(self.dataset_name, self.specific_dataset)
        if(determine_stride(self.dataset_name, self.specific_dataset)):
            self.Dimension = self.state_dim
            self.stride = stride
        else:
            self.Dimension = self.state_dim + self.action_dim
            self.stride = 1
        self.backbone_name = backbone_name
        self.backbone_selection(backbone_layers)
        self.model_name = get_PlannerName(self.dataset_name, self.specific_dataset, self.task_id)
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
        self.optim = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=1e-5)
        #self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
        self.num_steps = num_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.num_steps)
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.save_freq = save_freq
        #self.logdir = f"./{self.dataset_name}_{self.specific_dataset}_checkpoints/"
        self.logdir = os.path.join(
            REPO_ROOT,
            "Pretrain",
            f"{self.dataset_name}_{self.specific_dataset}"
             + (f"_task{self.task_id}" if self.task_id is not None else "")
             + "_checkpoints",
        )
        self.loss_tracker = LossTracker(save_dir="./logs/")
        self.data_parallel = bool(data_parallel and torch.cuda.device_count() > 1)
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            print(f"Planner DataParallel enabled on {torch.cuda.device_count()} GPUs")

    def base_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model
    
    def save_hyperparameters(self, filepath: Optional[str] = None):
        if filepath is None:
            os.makedirs(f"./Pretrain/Planners/args/{self.dataset_name}/{self.specific_dataset}/", exist_ok=True)
            filepath = f"./Pretrain/Planners/args/{self.dataset_name}/{self.specific_dataset}/hyperparameters.json"
    
        def convert_to_json_serializable(obj):
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
                return str(obj)
            return obj
    
        # Get optimizer and scheduler info
        optimizer_type = type(self.optim).__name__
        optimizer_params = {
             'type': optimizer_type,
             'lr': self.lr,
             'weight_decay': self.optim.param_groups[0].get('weight_decay', 0)
         }
    
        scheduler_type = type(self.scheduler).__name__
        scheduler_params = {
             'type': scheduler_type,
             'T_max': self.num_steps if hasattr(self.scheduler, 'T_max') else None
        }
    
        model_for_info = self.base_model()
       # Get model architecture info
        model_info = {
             'backbone_name': self.backbone_name,
             'state_dim': int(self.state_dim),
             'action_dim': int(self.action_dim),
             'horizon': self.horizon,
         }
    
        # Add backbone-specific parameters if available
        if hasattr(model_for_info, 'in_dim'):
              model_info['model_in_dim'] = int(model_for_info.in_dim)
        if hasattr(model_for_info, 'emb_dim'):
              model_info['model_emb_dim'] = int(model_for_info.emb_dim)
        if hasattr(model_for_info, 'd_model'):
              model_info['model_d_model'] = int(model_for_info.d_model)
        if hasattr(model_for_info, 'n_heads'):
              model_info['model_n_heads'] = int(model_for_info.n_heads)
        if hasattr(model_for_info, 'depth'):
              model_info['model_depth'] = int(model_for_info.depth)
    
        # Compile all hyperparameters
        hyperparams = {
           'env_details': {
                'dataset_name': self.dataset_name,
                'specific_dataset': self.specific_dataset,
                'state_dim': int(self.state_dim),
                'action_dim': int(self.action_dim),
            },
           'model_architecture': model_info,
           'training_hyperparameters': {
                'horizon': self.horizon,
                'num_steps': self.num_steps,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'gradient_accumulate_every': self.gradient_accumulate_every,
                'data_parallel': self.data_parallel,
                'optimizer': optimizer_params,
                'scheduler': scheduler_params,
            },
           'ema_hyperparameters': {
                'ema_decay': self.ema.beta,
                'update_ema_every': self.update_ema_every,
                'step_start_ema': self.step_start_ema,
            },
           'training_config': {
                 'save_freq': self.save_freq,
                 'log_freq': self.log_freq,
                 'logdir': self.logdir,
                 'model_name': self.model_name,
             },
           'sde_hyperparameters': {
                's': self.s,
                'weight_type':self.weight_type,
                'eps': self.eps,
            }
        }
    
        # Handle numpy arrays, torch.device, and other non-JSON-serializable types
        hyperparams = convert_to_json_serializable(hyperparams)
    
        # Save with pretty printing (indent=4 makes it human-readable)
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=4, sort_keys=False)
    
        print(f"Pretraining hyperparameters saved to {filepath}", flush=True)

    def backbone_selection(self, backbone_layers):
         if(self.backbone_name == 'transformer'):
              self.model = DiT1d(
                   in_dim = self.Dimension, emb_dim = 128,
                   d_model = 256, n_heads = 256//64, depth = backbone_layers, timestep_emb_type="fourier").to(self.device)
         elif(self.backbone_name == 'unet'):
              self.model = TemporalUnet(self.horizon, self.Dimension).to(self.device)
              
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.base_model().state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.base_model())
    

    """
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        self.model.eval()
        self.ema_model.eval()
        data = {
            'dataset_name': self.dataset_name,
            'specific_dataset': self.specific_dataset,
            'step': self.step,
            'ema': self.ema_model.state_dict()
        }
        if(epoch == self.num_steps):
            name = getName(self.dataset_name, self.specific_dataset)
            file_name = f"{name}_Planner_{str(0)}.pt"
            dir = f"./Finetuning/Planners/{self.dataset_name}/{self.specific_dataset}/"
            os.makedirs(dir, exist_ok=True)
            savepath = os.path.join(dir, file_name)
        else:
            file_name = self.model_name + '_' + str(epoch) + '.pt'
            os.makedirs(self.logdir, exist_ok=True)
            savepath = os.path.join(self.logdir, file_name)
        torch.save(data, savepath)
        print(f'Saved model to {savepath}', flush=True)
    """

    def save(self, epoch):
        self.model.eval()
        self.ema_model.eval()
        data = {
              'dataset_name': self.dataset_name,
              'specific_dataset': self.specific_dataset,
              'task_id': self.task_id,                     # NEW
              'step': self.step,
              'ema': self.ema_model.state_dict(),
        }
        if epoch == self.num_steps:
            file_name = f"{self.model_name}_0.pt"
            save_dir = os.path.join(
                REPO_ROOT, "Finetuning", "Planners",
                self.dataset_name, self.specific_dataset,
            )
        else:
            file_name = f"{self.model_name}_{epoch}.pt"
            save_dir = self.logdir
        os.makedirs(save_dir, exist_ok=True)
        savepath = os.path.join(save_dir, file_name)
        torch.save(data, savepath)
        print(f'Saved model to {savepath}', flush=True)

    def train(self):
        print(self.device)
        dataset = PlannerDataset(self.dataset_name, self.specific_dataset, self.task_id, self.horizon, self.state_dim, self.action_dim, self.stride)
        dataloader = cycle(DataLoader(dataset, self.batch_size, shuffle = True, pin_memory = True, num_workers = 8))
        print(f"Training planner for {self.dataset_name}-{self.specific_dataset} Dataset")
        print(f"Backbone:{self.backbone_name}, Horizon: {self.horizon}, Epochs: {self.num_steps}, Batch Size: {self.batch_size}, Learning Rate; {self.lr}")
        
        # Save hyperparameters at the start of training
        self.save_hyperparameters()

        self.model.train()
        self.ema_model.eval()
        for p in self.ema_model.parameters():
              p.requires_grad_(False)
        self.step = 0
        total_loss = 0
        while(self.step < self.num_steps):
            for i in range(self.gradient_accumulate_every):
                traj, cond = next(dataloader)
                loss = self.Loss(traj.to(self.device), cond.to(self.device))
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.optim.zero_grad()
            self.scheduler.step()
            total_loss += loss.item()
            self.loss_tracker.log_loss(self.step, loss.item(), self.optim.param_groups[0]['lr'])

            if ((self.step % self.update_ema_every) == 0):
                self.step_ema()
            
            if ((self.step % self.log_freq) == 0):
                logged_loss = total_loss / self.log_freq
                print(f"step {self.step} loss {logged_loss}")
                wandb_log(
                    {"planner/loss": logged_loss, "planner/lr": self.optim.param_groups[0]['lr']},
                    step=self.step,
                )
                total_loss = 0
            
            if ((self.step % self.save_freq == 0) and (self.step!=0)):
                self.save(self.step)
                self.loss_tracker.save_logs(f"{self.model_name}_logs.pkl")
                self.loss_tracker.plot_loss_curve(
                      save_path=f"./plots/{self.model_name}_loss_curve.png",
                      title=f"{self.model_name} Training Loss",
                      show_lr=True,
                      smooth_window=50)

            self.step += 1
        # Final save and plot
        self.save(self.step)
        self.loss_tracker.save_logs(f"{self.model_name}_final_logs.pkl")
        self.loss_tracker.plot_loss_curve(
             save_path=f"./plots/{self.model_name}_final_loss_curve.png",
             title=f"{self.model_name} Training Loss",
             show_lr=True,
             smooth_window=50)
    
    def selector(self, specific_dataset, times = 1000):
         dataset = PlannerDataset_Rollout(self.dataset_name, specific_dataset, self.specific_dataset, self.horizon, self.state_dim, self.action_dim)
         dataloader = DataLoader(dataset, 10, shuffle = True, pin_memory = True)
         N = len(dataloader)
         min_Loss = float('inf')
         checkpoint = self.save_freq
         best_checkpoint = 0
         validation_tracker = LossTracker(save_dir="./logs/")
         print(f"Loss of {self.model_name} on {specific_dataset} dataset. Running {times} times for each checkpoints")
         while(checkpoint <= self.num_steps):
            self.backbone_selection()
            state_dict = get_pretrained_planner(self.dataset_name, self.specific_dataset, checkpoint, self.task_id)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            avg_loss = 0
            for i in range(times):
                total_loss = 0
                for traj, cond in dataloader:
                    loss = self.Loss(traj.to(self.device), cond.to(self.device))
                    total_loss += loss.item()
                Loss = total_loss/N
                avg_loss += Loss
            final_loss = avg_loss/times
            if(final_loss < min_Loss):
                 min_Loss = final_loss
                 best_checkpoint = checkpoint
            print(f"Checkpoint: {checkpoint} Loss: {final_loss}")
            validation_tracker.log_loss(checkpoint, final_loss)
            checkpoint += self.save_freq  
         print(f"Best Checkpoint: {best_checkpoint}, Loss: {min_Loss}")  
         #self.loss_tracker.save_logs(f"{self.model_name}_{specific_dataset}_validation_loss_curve.pkl")
         
         validation_tracker.plot_loss_curve(
             save_path=f"./plots/{self.model_name}_{specific_dataset}_validation_loss_curve.png",
             title=f"{self.model_name} {specific_dataset} Validation Loss",
             show_lr=False,
             smooth_window=5)
         
         return best_checkpoint, min_Loss

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
