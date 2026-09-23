from accelerate import Accelerator
import math
from accelerate.utils.offload import offload_weight
import torch.distributed as dist
from typing import Optional, List
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Pretrain.utils import wandb_log
from Pretrain.Dataset import get_env, get_dataset, Planner_Processor
from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Critic.nets import Critic
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Planners.Backbone.Sampler import (
    karras_beta_schedule as planner_karras_beta_schedule,
    sample_euler_karras,
)
from Pretrain.Planners.Backbone.utils import cosine_beta as planner_cosine_beta
from Pretrain.Transition_Kernel.Kernel_Net import (
    MoGTransitionKernel,
    RobustTransitionKernel,
)
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_log_density_mog
from Finetuning.metrics import (
    evaluate_critic, 
    compute_j_by_state, 
    td_residual_stats, 
    value_grad_stats,
    evaluate_critic_hat_return,
)
import os
import pickle
import wandb
from torch.utils.data import Dataset, DataLoader
from Pretrain.utils import wandb_log, SAStats
from Finetuning.utils import (
    KernelConfig,
    TrajectoryDict,
    Q_Scale,
    get_kernel,
    get_kernel_stats,
    get_planner,
    get_reward_model,
    get_reward_stats,
    get_critic_model,
    get_critic_stats,
    obtain_and_save_critic_stats,
    save_critic,
    save_Q_scale,
    get_Q_scale,
    get_CriticName,
    update_critic_stats,
    check_device,
    cycle,
    _compact_tensor_rows_for_object_gather,
    symexp,
    symlog,
)




class CriticDataset_Reward(Dataset):
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs: List[TrajectoryDict],
                       horizon: int = 32,
                       old_step: Optional[int] = None,
                       new_step: int = 0,
                       momentum: float = 0.005,
                       value_scale: float = 5.0,
                       task_id: Optional[int] = None):
        obs_all = [traj["observations"] for traj in trajs]
        obs_all = np.concatenate(obs_all, axis=0)

        stats = SAStats()
        stats.obs_mean = obs_all.mean(axis=0)
        stats.obs_std = obs_all.std(axis=0) + 1e-8
        if old_step is not None:
            self.stats = update_critic_stats(
                dataset_name, specific_dataset, stats, task_id, old_step, momentum
            )
        else:
            self.stats = stats

        device = check_device()
        _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)
        reward_state, _, _ = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )
        reward_net = SimpleReward(
            obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
        ).to(device)
        reward_net.load_state_dict(reward_state)
        reward_net.eval()
        for p in reward_net.parameters():
            p.requires_grad_(False)
        reward_stat = get_reward_stats(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )

        transitions = []
        goal_states = []

        for traj in trajs:
            obs = np.asarray(traj["observations"])
            acts = np.asarray(traj["actions"])
            n_obs = len(obs)
            if n_obs == 0:
                continue

            raw_masks = traj.get("masks", None)
            if raw_masks is None:
                masks = np.ones(n_obs, dtype=np.float32)
            else:
                masks = np.asarray(raw_masks[:n_obs], dtype=np.float32)
                if len(masks) < n_obs:
                    pad = np.ones(n_obs - len(masks), dtype=np.float32)
                    masks = np.concatenate([masks, pad], axis=0)

            # Type A last frame is the only GT row
            if masks[-1] == 0.0:
                goal_states.append(self.stats.norm_obs(obs[-1]).astype(np.float32))

            n_act = min(len(acts), max(n_obs - 1, 0))
            rews = np.zeros(n_obs, dtype=np.float32)
            if n_act > 0:
                with torch.no_grad():
                    s_t = torch.as_tensor(
                        reward_stat.norm_obs(obs[:n_act]).astype(np.float32),
                        device=device,
                    )
                    a_t = torch.as_tensor(
                        acts[:n_act].astype(np.float32), device=device,
                    )
                    rews[:n_act] = (
                        reward_net(s_t, a_t).cpu().numpy().astype(np.float32)
                        / value_scale
                    )

            if n_obs < 2:
                continue

            # pad so every window has length `horizon` (pad is NOT a goal)
            if n_obs < horizon:
                pad_n = horizon - n_obs
                obs_w = np.concatenate([obs, np.repeat(obs[-1:], pad_n, axis=0)], axis=0)
                rew_w = np.concatenate([rews, np.zeros(pad_n, dtype=np.float32)], axis=0)
                m_w = np.concatenate([masks, np.ones(pad_n, dtype=np.float32)], axis=0)
                starts = [0]
            else:
                obs_w, rew_w, m_w = obs, rews, masks
                # include a window that ENDS on the last state (the goal on Type A)
                starts = range(n_obs - horizon + 1)

            for t in starts:
                transitions.append((
                    self.stats.norm_obs(obs_w[t : t + horizon]).astype(np.float32),
                    rew_w[t : t + horizon].astype(np.float32),
                    m_w[t : t + horizon].astype(np.float32),
                ))

        self.transitions = transitions
        self.goal_states = goal_states
        self.save_stats(dataset_name, specific_dataset, task_id, new_step)

    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None, step: int = 0):
        critic_name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name = str(critic_name) + f"_Critic_stats_{str(step)}.pkl"
        stats_dir = f"./Finetuning/Critics/{dataset_name}/{specific_dataset}/Stats/"
        os.makedirs(stats_dir, exist_ok=True)
        savepath = os.path.join(stats_dir, stats_name)
        with open(savepath, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __getitem__(self, idx):
        obs_chunk, rews_chunk, mask_chunk = self.transitions[idx]
        return (
            torch.tensor(obs_chunk, dtype=torch.float32),
            torch.tensor(rews_chunk, dtype=torch.float32),
            torch.tensor(mask_chunk, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.transitions)

class Critic_Buffer_Reward():
    def __init__(self, dataset_name: str,
                       specific_dataset: str,
                       reward_hidden_layers: int,
                       reward_hidden_dim: int,
                       reward_checkpoint: int,
                       trajs:  List[TrajectoryDict],
                       horizon: int = 32,
                       gamma: float = 0.99,
                       lam: float = 0.95,
                       task_id: Optional[int] = None,
                       old_step: Optional[int] = None,  
                       new_step: int = 0, 
                       value_scale: float = 5.0,
                       momentum: float = 0.005):
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.data = CriticDataset_Reward(
            dataset_name         = dataset_name,
            specific_dataset     = specific_dataset,
            reward_hidden_layers = reward_hidden_layers,
            reward_hidden_dim    = reward_hidden_dim,
            reward_checkpoint    = reward_checkpoint,
            trajs                = trajs,
            horizon              = horizon,
            old_step             = old_step,
            new_step             = new_step,
            momentum             = momentum,
            value_scale          = value_scale,
            task_id              = task_id,
        )
   
    """
    def obtain_training_data(self, target_critic: nn.Module, batch_size: int, tgt_mean: torch.Tensor, tgt_std: torch.Tensor, device: str):
        loader = cycle(DataLoader(
            self.data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ))
        obs_chunks, rews_chunks = next(loader)      # (B, T, dim), (B, T)
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]
        

        with torch.no_grad():
            values = target_critic(obs_chunks)            # (B, T)

            deltas = (
                  rews_chunks[:, :-1]
                  + self.gamma * values[:, 1:]
                   - values[:, :-1]
              )                                             # (B, T-1)

            advantages = torch.zeros(B, T - 1, device=device)
            last_adv = torch.zeros(B, device=device)
            for t in reversed(range(T - 1)):
                last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                advantages[:, t] = last_adv

            #value_targets = values[:, 0] + advantages[:, 0]   # (B,)
            with torch.no_grad():
                 values = target_critic(obs_chunks)                      # (B, T)
                 deltas = (
                       rews_chunks[:, :-1]
                       + self.gamma * values[:, 1:]
                       - values[:, :-1]
                 )                                                       # (B, T-1)

                  # GAE advantages
                 advantages = torch.zeros_like(deltas)
                 last_adv = torch.zeros(B, device=device)
                 for t in reversed(range(deltas.shape[1])):
                     last_adv = deltas[:, t] + self.gamma * self.lam * last_adv
                     advantages[:, t] = last_adv

                 # === ADD NORMALIZATION HERE ===
                 value_targets = values[:, 0] + advantages[:, 0]         # raw targets
                
                 
                 # Normalize advantages and targets (running stats or batch stats)
                 adv_mean = advantages.mean()
                 adv_std  = advantages.std() + 1e-8
                 advantages = (advantages - adv_mean) / adv_std
                 
                 alpha = 0.99
                 tgt_mean_new = value_targets.mean()
                 tgt_std_new  = value_targets.std() + 1e-8
                 tgt_mean_new = alpha * tgt_mean + ((1 - alpha) * tgt_mean_new)
                 tgt_std_new = alpha * tgt_std + ((1 - alpha) * tgt_std_new)
                 value_targets = (value_targets - tgt_mean_new) / tgt_std_new
                 # =================================
                 

        return obs_chunks[:, 0], value_targets, tgt_mean_new, tgt_std_new
        #return obs_chunks[:, 0], value_targets
    """
    
    """
    def obtain_training_data(self, target_critic: nn.Module, batch, tgt_mean: torch.Tensor, tgt_std: torch.Tensor, device: str):
        
        obs_chunks, rews_chunks, mask_chunks = batch
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        mask_chunks = mask_chunks.to(device)
        m = mask_chunks[:, :-1]   # (B, T-1), same time index as r_t
        B, T = obs_chunks.shape[0], obs_chunks.shape[1]
        
        with torch.no_grad():
                 values = target_critic(obs_chunks)                      # (B, T)
                 values = symexp(values)
                 #values = torch.clamp(values, float('-inf'), 0.0)
                 deltas = (
                       rews_chunks[:, :-1]
                       + self.gamma * m * values[:, 1:]
                       #+ self.gamma * values[:, 1:]
                       - values[:, :-1]
                 )                                                       # (B, T-1)

                  # GAE advantages
                 advantages = torch.zeros_like(deltas)
                 last_adv = torch.zeros(B, device=device)
                 for t in reversed(range(deltas.shape[1])):
                     last_adv = deltas[:, t] + self.gamma * self.lam * m[:, t] * last_adv
                     #last_adv = deltas[:, t] + self.gamma * self.lam  * last_adv
                     advantages[:, t] = last_adv

                 # === ADD NORMALIZATION HERE ===
                 value_targets = values[:, 0] + advantages[:, 0]         # raw targets
                
                 
                 
                 # Normalize advantages and targets (running stats or batch stats)
                 adv_mean = advantages.mean()
                 adv_std  = advantages.std() + 1e-8
                 advantages = (advantages - adv_mean) / adv_std
                 
                 alpha = 0.99
                 tgt_mean_new = value_targets.mean()
                 tgt_std_new  = value_targets.std() + 1e-8
                 tgt_mean_new = alpha * tgt_mean + ((1 - alpha) * tgt_mean_new)
                 tgt_std_new = alpha * tgt_std + ((1 - alpha) * tgt_std_new)
                 #value_targets = (value_targets - tgt_mean_new) / tgt_std_new
                 # =================================
                
                 

        return obs_chunks[:, 0], value_targets, tgt_mean_new, tgt_std_new
        #return obs_chunks[:, 0], value_targets
    """

    def obtain_training_data(self, target_critic: nn.Module, batch,
                             tgt_mean: torch.Tensor, tgt_std: torch.Tensor,
                             device: str):
        obs_chunks, rews_chunks, mask_chunks = batch
        obs_chunks = obs_chunks.to(device)
        rews_chunks = rews_chunks.to(device)
        mask_chunks = mask_chunks.to(device)
        B, T = obs_chunks.shape[:2]

        with torch.no_grad():
            V = symexp(target_critic(obs_chunks))          # return units
            done_next = (mask_chunks[:, 1:] == 0).float()  # next state is goal

            deltas = (
                rews_chunks[:, :-1]
                + self.gamma * (1.0 - done_next) * V[:, 1:]
                - V[:, :-1]
            )

            advantages = torch.zeros_like(deltas)
            last_adv = torch.zeros(B, device=device)
            for t in reversed(range(deltas.shape[1])):
                last_adv = (
                    deltas[:, t]
                    + self.gamma * self.lam * (1.0 - done_next[:, t]) * last_adv
                )
                advantages[:, t] = last_adv

            y_raw = V[:, 0] + advantages[:, 0]
            # only if this window *starts* on a goal (rare); real GT is the extra Huber
            is_goal = (mask_chunks[:, 0] == 0)
            y_raw = torch.where(is_goal, torch.zeros_like(y_raw), y_raw)
            y_head = symlog(y_raw)

        return obs_chunks[:, 0], y_head, tgt_mean, tgt_std

def train_critic_with_reward(trajs: List[TrajectoryDict], 
                 dataset_name: str, 
                 specific_dataset: str, 
                 reward_hidden_layers: int,
                 reward_hidden_dim: int,
                 reward_checkpoint: int,
                 critic_hidden_layers: int, 
                 critic_hidden_dim: int, 
                 batch_size, 
                 num_steps, 
                 gamma, lam, horizon, 
                 lr, 
                 min_lr, 
                 tau, 
                 old_step: Optional[int] = None, 
                 new_step: int = 0, 
                 momentum: float = 0.005, 
                 value_scale: float = 5.0,
                 task_id: Optional[int] = None):
    device = check_device()
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers).to(device)
    if(old_step is not None):
        critic_state_dict, _ = get_critic_model(dataset_name, specific_dataset, task_id = task_id, step = old_step)
        critic.load_state_dict(critic_state_dict)
    target_critic = Critic(obs_dim, critic_hidden_dim, critic_hidden_layers).to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    optimizer = optim.AdamW(critic.parameters(), lr = lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = num_steps,   # one scheduler step per training step
            eta_min = min_lr
        )
    critic.train()
    NS = 0 if new_step == -1 else new_step
    buffer = Critic_Buffer_Reward(
                       dataset_name,
                       specific_dataset,
                       reward_hidden_layers,
                       reward_hidden_dim,
                       reward_checkpoint,
                       trajs,
                       horizon,
                       gamma,
                       lam,
                       task_id,
                       old_step,  
                       NS, 
                       value_scale,
                       momentum)
    g = torch.Generator()
    g.manual_seed(1)
    loader = cycle(
        DataLoader(
            buffer.data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=g,
        )
    )
    print(f"Training critic for {dataset_name}-{specific_dataset}")
    total_loss = 0.0

    """
    tgt_mean = torch.zeros(1, device=device)
    tgt_std = torch.ones(1, device=device)
    for k in range(1, num_steps + 1):  # number of passes over dataset
           batch = next(loader)
           s, target_value, tgt_mean, tgt_std = buffer.obtain_training_data(target_critic, batch, tgt_mean, tgt_std, device)
           s = s.to(device)
           target_value = target_value.to(device)
           target_value = symlog(target_value)

           # Predicted Q-values
           q_pred = critic(s)
           loss = F.smooth_l1_loss(q_pred, target_value, beta = 1.0)
           #loss = F.mse_loss(q_pred, target_value)
           total_loss += loss.item()

           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
           
           if(k % 200 == 0):
                print(f"Critic Training step {k} loss: {total_loss/200}")
                wandb.log({"loss": total_loss/200, "step": k})     
                total_loss = 0.0
            
           # Soft update target network
           for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")
    q_scale = Q_Scale()
    q_scale.Q_scale = value_scale
    save_Q_scale(q_scale, dataset_name, specific_dataset, task_id)
    print(f"mean: {tgt_mean.item()}, std: {tgt_std.item()}")
    """
    


    tgt_mean = torch.zeros(1, device=device)
    tgt_std = torch.ones(1, device=device)
    goal_states = buffer.data.goal_states
    print(f"goal-state GT rows: {len(goal_states)}")

    for k in range(1, num_steps + 1):
        batch = next(loader)
        s, target_value, tgt_mean, tgt_std = buffer.obtain_training_data(
            target_critic, batch, tgt_mean, tgt_std, device,
        )
        s = s.to(device)
        target_value = target_value.to(device)
        # target_value is already symlog(y_raw) — do NOT symlog again

        q_pred = critic(s)
        loss = F.smooth_l1_loss(q_pred, target_value, beta=1.0)

        if len(goal_states) > 0:
            n_g = min(batch_size, len(goal_states))
            idx = np.random.randint(0, len(goal_states), size=n_g)
            s_g = torch.as_tensor(
                np.stack([goal_states[i] for i in idx], axis=0),
                dtype=torch.float32, device=device,
            )
            loss = loss + F.smooth_l1_loss(
                critic(s_g), torch.zeros(n_g, device=device),
            )

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if(k % 1000 == 0):
                print(f"Critic Training step {k} loss: {total_loss/1000}")
                wandb.log({"loss": total_loss/1000, "step": k})     
                total_loss = 0.0
            
           # Soft update target network
        for param, tgt_param in zip(critic.parameters(), target_critic.parameters()):
               tgt_param.data.mul_(1 - tau)
               tgt_param.data.add_(tau * param.data)
    target_critic.eval()
    save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
    print(f"critic model saved")
    q_scale = Q_Scale()
    q_scale.Q_scale = value_scale
    save_Q_scale(q_scale, dataset_name, specific_dataset, task_id)
    print(f"mean: {tgt_mean.item()}, std: {tgt_std.item()}")



def train_critic_with_planner7(
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    old_critic_checkpoint: Optional[int],
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    kernel_config: KernelConfig,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    batch_size: int = 64,
    num_steps: int = 100,
    resample_every: int = 10,
    vectorized_sampling: bool = True,
    traj_length: Optional[int] = None,
    plan_chunk_size: int = 256,
    horizon: int = 32,
    gamma: float = 0.99,
    lam: Optional[float] = None,
    rho: float = 0.0,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    tau: float = 0.005,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    new_step: int = 0,
    task_id: Optional[int] = None,
    mix_reset: bool = False,
    n_reset: int = 256,
    log_every: int = 20,
    w_all: float = 0.0,
    w_near: float = 0.0,
    w_goal: float = 0.0,
    accelerator=None,
    wandb_prefix: str = "critic_warmup",
    wandb_step_metric: str = "critic_warmup_step",
    wandb_step_offset: int = 0,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist
    if accelerator is None:
        accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    # ---------------------------------------------------------------- helpers
    def load_kernel_ensemble(
        dataset_name: str,
        specific_dataset: str,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        kernel_state_dicts, _, _ = get_kernel(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        kernels = []
        if kernel_config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_net = RobustTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)
        else:
            for sd in kernel_state_dicts:
                k_net = MoGTransitionKernel(
                    obs_dim, act_dim,
                    kernel_config.num_modes,
                    kernel_config.num_hidden_layers, kernel_config.hidden_dim,
                    noise_floor=kernel_config.noise_floor,
                ).to(device)
                k_net.load_state_dict(sd)
                k_net.eval()
                for p in k_net.parameters():
                    p.requires_grad_(False)
                kernels.append(k_net)

        kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_config.checkpoint,
        )
        k_mean = torch.as_tensor(kernel_stat.obs_mean, device=device, dtype=torch.float32)
        k_std = torch.as_tensor(
            np.maximum(kernel_stat.obs_std, 1e-3), device=device, dtype=torch.float32
        )
        return kernels, k_mean, k_std

    @torch.no_grad()
    def is_plan_feasible(
        s_raw_plan: torch.Tensor,
        a_raw_plan: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
    ) -> bool:
        s_k = (s_raw_plan - k_mean) / k_std
        s_t = s_k[:-1]
        a_t = a_raw_plan[:-1]
        s_tp1 = s_k[1:]

        if kernel_config.type_kernel == 'robust':
            total = torch.zeros(s_t.shape[0], device=device)
            for k_net in kernels:
                mu, log_std = k_net(s_t, a_t)
                lp = k_net.log_prob(s_tp1, mu, log_std)
                total = total + lp
            avg_lp = total / len(kernels)
        else:
            avg_lp = compute_log_density_mog(kernels, s_t, a_t, s_tp1)

        return bool((avg_lp > kernel_config.min_log_prob).all().item())

    @torch.no_grad()
    def sample_plans_batched(
        normalized_s0: torch.Tensor,
        planner: nn.Module,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        oversample: int,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        conditions = normalized_s0.repeat_interleave(oversample, dim=0)
        candidate_count = conditions.shape[0]
        dimension = obs_dim + act_dim
        t_grid, beta_1, sigma_grid = planner_karras_beta_schedule(
            steps_T, device=device
        )
        beta_2 = planner_cosine_beta(t_grid, s=0.008)

        # Preserve scalar sampler initial-noise call order. Warmup uses eta=0,
        # so there are no per-step stochastic draws to interleave by candidate.
        initial_noise = torch.cat(
            [
                torch.randn(1, horizon, dimension, device=device)
                for _ in range(candidate_count)
            ],
            dim=0,
        )

        generated = []
        for start in range(0, candidate_count, chunk_size):
            stop = min(start + chunk_size, candidate_count)
            cond = conditions[start:stop]
            x = initial_noise[start:stop] * sigma_grid[0]
            current_batch = x.shape[0]
            mask = torch.zeros_like(x)
            mask[:, 0, :obs_dim] = 1.0
            conditioned = torch.zeros_like(x)
            conditioned[:, 0, :obs_dim] = cond
            x = mask * conditioned + (1 - mask) * x

            for diffusion_step in range(steps_T):
                t_now = t_grid[diffusion_step]
                t_next = (
                    t_grid[diffusion_step + 1]
                    if diffusion_step < steps_T - 1
                    else 0.0
                )
                dt = (t_next - t_now).item()
                beta_now = (
                    beta_1[diffusion_step].item()
                    if diffusion_step < num_karras
                    else beta_2[diffusion_step].item()
                )
                drift = -0.5 * beta_now * x
                score = planner(x, t_now.expand(current_batch))
                if eta > 0:
                    noise = torch.randn_like(x)
                    noise_scale = eta * math.sqrt(beta_now * (-dt))
                    x = x + (drift - beta_now * score) * dt + noise_scale * noise
                else:
                    x = x + (drift - beta_now * score) * dt
                x = mask * conditioned + (1 - mask) * x
                x[..., obs_dim:] = torch.clamp(x[..., obs_dim:], -1.0, 1.0)
            generated.append(x)

        return torch.cat(generated, dim=0)

    @torch.no_grad()
    def batched_feasible_mask(
        plans: torch.Tensor,
        kernels: List[nn.Module],
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
    ) -> torch.Tensor:
       
        state_raw = plans[..., :obs_dim] * planner_std + planner_mean
        state_kernel = (state_raw - k_mean) / k_std
        state = state_kernel[:, :-1].reshape(-1, obs_dim)
        next_state = state_kernel[:, 1:].reshape(-1, obs_dim)
        action = torch.clamp(plans[:, :-1, obs_dim:], -1.0, 1.0).reshape(
            state.shape[0], -1
        )

        if kernel_config.type_kernel == 'robust':
            average_log_prob = torch.zeros(state.shape[0], device=plans.device)
            for kernel in kernels:
                mu, log_std = kernel(state, action)
                average_log_prob += kernel.log_prob(next_state, mu, log_std)
            average_log_prob /= len(kernels)
        else:
            average_log_prob = compute_log_density_mog(
                kernels, state, action, next_state
            )

        transition_count = plans.shape[1] - 1
        return (
            average_log_prob.view(plans.shape[0], transition_count)
            > kernel_config.min_log_prob
        ).all(dim=1)

    @torch.no_grad()
    def _generate_feasible_plans_parallel(
        play_pool: np.ndarray,          # === CHANGED === was s0_pool
        reset_pool: np.ndarray,
        planner: nn.Module,
        planner_proc: Planner_Processor,
        planner_mean: torch.Tensor,
        planner_std: torch.Tensor,
        kernels: List[nn.Module],
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        kernel_config: KernelConfig,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        batch_size: int,
        training_step: int,
        vectorized_sampling: bool,
        plan_chunk_size: int,
        device: torch.device,
        accelerator,
        use_mix_reset: bool = False,
    ):

        oversample = kernel_config.oversample

        if accelerator.is_main_process:
            rng = np.random.RandomState(training_step + 10007)
            #rng = np.random.RandomState(42)
            if use_mix_reset:
                n_r = batch_size // 2
                selected_s0 = np.concatenate(
                  [
                      play_pool[rng.randint(0, len(play_pool), size=batch_size - n_r)],
                      reset_pool[rng.randint(0, len(reset_pool), size=n_r)],
                  ],
                    axis=0,
             )
                rng.shuffle(selected_s0)
            else:
                selected_s0 = play_pool[rng.randint(0, len(play_pool), size=batch_size)]
        else:
            selected_s0 = np.empty((batch_size, play_pool.shape[1]), dtype=np.float32)

        selected_s0_tensor = torch.from_numpy(selected_s0).to(device)
        if accelerator.num_processes > 1:
            dist.broadcast(selected_s0_tensor, src=0)
        selected_s0 = selected_s0_tensor.cpu().numpy()

        # 2. Split the batch_size s0 across GPUs
        local_s0_indices = np.array_split(
            np.arange(batch_size), accelerator.num_processes
        )[accelerator.process_index]
        local_s0 = selected_s0[local_s0_indices]

        local_accepted = []
        if vectorized_sampling:
            normalized_s0 = torch.as_tensor(
                np.stack([planner_proc.preprocess(state) for state in local_s0]),
                dtype=torch.float32,
                device=device,
            )
            local_plans = sample_plans_batched(
                normalized_s0=normalized_s0,
                planner=planner,
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=horizon,
                steps_T=steps_T,
                num_karras=num_karras,
                eta=eta,
                oversample=oversample,
                chunk_size=plan_chunk_size,
                device=device,
            )
            feasible = batched_feasible_mask(
                plans=local_plans,
                kernels=kernels,
                planner_mean=planner_mean,
                planner_std=planner_std,
                k_mean=k_mean,
                k_std=k_std,
                kernel_config=kernel_config,
                obs_dim=obs_dim,
            )
            # Each unbound tensor is otherwise a view of the full accepted-plan
            # storage. Pickling those views for all_gather_object serializes the
            # full backing storage once per plan (several GiB instead of MiB).
            """
            local_accepted.extend(
                _compact_tensor_rows_for_object_gather(local_plans[feasible])
            )
            """
            kept = local_plans[feasible] if feasible.any() else local_plans  # === CHANGED ===
            local_accepted.extend(_compact_tensor_rows_for_object_gather(kept))
        else:
            for s0_raw in local_s0:
                s0_p = planner_proc.preprocess(s0_raw)
                accepted_for_this_s0 = []

                for _ in range(oversample):
                    x = sample_euler_karras(
                        s0_p, planner, obs_dim, act_dim, horizon,
                        num_steps=steps_T, num_karras=num_karras,
                        eta=eta, device=device,
                    )
                    x_t = torch.from_numpy(x).float().to(device)

                    s_planner = x_t[..., :obs_dim]
                    a_raw = torch.clamp(x_t[..., obs_dim:], -1.0, 1.0)
                    s_raw_pl = s_planner * planner_std + planner_mean

                    if is_plan_feasible(
                        s_raw_plan=s_raw_pl,
                        a_raw_plan=a_raw,
                        kernels=kernels,
                        k_mean=k_mean,
                        k_std=k_std,
                        kernel_config=kernel_config,
                        device=device,
                    ):
                        accepted_for_this_s0.append(x_t.cpu())

                local_accepted.extend(accepted_for_this_s0)

        # 4. Collect from all GPUs
        if accelerator.num_processes > 1:
            all_accepted_lists = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_accepted_lists, local_accepted)
        else:
            all_accepted_lists = [local_accepted]

        all_plans = [p for sublist in all_accepted_lists for p in sublist]
        if not all_plans:
            raise RuntimeError("planner7 found no kernel-feasible plans")
        plans = torch.stack(all_plans).to(device)
        return plans, None
    
    def _split(traj_list, suffix_only=False, type_a_only=False):
        play, goal = [], []
        for traj in traj_list:
            obs = np.asarray(traj["observations"], dtype=np.float32)
            n = len(obs)
            raw = traj.get("masks", None)
            if raw is None:
                masks = np.ones(n, dtype=np.float32)
            else:
                masks = np.asarray(raw[:n], dtype=np.float32)
                if len(masks) < n:
                    masks = np.concatenate(
                        [masks, np.ones(n - len(masks), dtype=np.float32)]
                    )
            if type_a_only and not (n > 0 and float(masks[-1]) == 0.0):
                continue
            play.append(obs[masks != 0.0])
            if not suffix_only:
                goal.append(obs[masks == 0.0])
        play_pool = (
            np.concatenate(play, 0)
            if any(len(x) for x in play)
            else np.zeros((0, obs_dim), dtype=np.float32)
        )
        if suffix_only:
            return play_pool
        goal_pool = (
            np.concatenate(goal, 0)
            if any(len(x) for x in goal)
            else np.zeros((0, obs_dim), dtype=np.float32)
        )
        return play_pool, goal_pool

    def _train_reset_pool(dataset_name, specific_dataset, task_id, n=256):
        env, _, _ = get_env(dataset_name, specific_dataset, task_id=task_id)
        rows = []
        for i in range(n):
            ob, _ = env.reset(
                seed=10_000 + i,
                options=dict(task_id=task_id),
            )
            rows.append(np.asarray(ob, dtype=np.float32))
        return np.stack(rows, axis=0)
    
    def plans_to_batch(plans):
        s_planner = plans[..., :obs_dim]
        actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
        s_raw = s_planner * planner_std + planner_mean
        N, H, _ = s_raw.shape
        n_loc = H - 1
        s_for_r = (s_raw[:, :n_loc] - r_mean) / r_std
        r_hat = reward_net(
            s_for_r.reshape(N * n_loc, -1),
            actions[:, :n_loc].reshape(N * n_loc, -1),
        ).reshape(N, n_loc)
        r_hat = r_hat / Scale.Q_scale

        if lam is not None:
            plan_targets = torch.zeros(N, device=device)
            w = 1.0 - lam
            weight_sum = 0.0
            for L in range(1, n_loc + 1):
                discounts = gamma_pow_t[:L]
                disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                s_L = (s_raw[:, L] - c_mean) / c_std
                v_boot = symexp(target_critic(s_L))
                plan_targets = plan_targets + w * (disc_return + (gamma ** L) * v_boot)
                weight_sum += w
                w *= lam
            plan_targets = plan_targets / max(weight_sum, 1e-8)
        else:
            r_list = []
            for L in range(1, n_loc + 1):
                discounts = gamma_pow_t[:L]
                disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                s_L = (s_raw[:, L] - c_mean) / c_std
                v_boot = symexp(target_critic(s_L))
                r_list.append(disc_return + (gamma ** L) * v_boot)
            R = torch.stack(r_list, dim=1)
            plan_targets = R.mean(dim=1) - rho * R.std(dim=1, unbiased=False).clamp(min=0.0)

        s0_raw = s_raw[:, 0]
        s0_key = torch.round(s0_raw * 1e5) / 1e5
        unique_s0, inverse_indices = torch.unique(s0_key, dim=0, return_inverse=True)
        U = unique_s0.shape[0]
        averaged_targets = torch.zeros(U, device=device)
        counts = torch.zeros(U, device=device)
        averaged_targets.index_add_(0, inverse_indices, plan_targets)
        counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
        averaged_targets = (averaged_targets / counts.clamp(min=1.0)).detach()
        averaged_targets = symlog(averaged_targets)
        s0_critic = ((unique_s0 - c_mean) / c_std).detach()
        return s0_critic, averaged_targets

    def _stat(t):
        if t is None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return (
            t.mean().item(),
            t.std(unbiased=False).item(),
            t.min().item(),
            t.max().item(),
        )

    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset, task_id=task_id)
    data = get_dataset(
        dataset_name, specific_dataset, task_id=task_id, traj_length=None,
    )
    all_trajs = data.get_trajectories()
    if traj_length is not None:
        data.traj_length = traj_length
        near_trajs = data.get_trajectories()
    else:
        near_trajs = []

    critic = Critic(obs_dim, hidden_dim, hidden_layers)
    if old_critic_checkpoint is not None:
        critic_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id=task_id, step=old_critic_checkpoint,
        )
        critic.load_state_dict(critic_state)

    target_critic = Critic(obs_dim, hidden_dim, hidden_layers)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    for p in target_critic.parameters():
        p.requires_grad_(False)
    target_critic = target_critic.to(device)

    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(device)

    planner_proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    planner_mean = torch.as_tensor(
        planner_proc.stats.obs_mean, device=device, dtype=torch.float32
    )
    planner_std = torch.as_tensor(
        np.maximum(planner_proc.stats.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    reward_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    )
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_net = reward_net.to(device)

    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    if old_critic_checkpoint is not None:
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )
    else:
        if is_main:
            critic_stat = obtain_and_save_critic_stats(
                all_trajs, dataset_name, specific_dataset, task_id, step=0
            )
        accelerator.wait_for_everyone()
        critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id=task_id, step=0,
        )

    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    all_pool, goal_pool = _split(all_trajs)
    near_pool = (
        _split(near_trajs, suffix_only=True, type_a_only=True)
        if len(near_trajs)
        else np.zeros((0, obs_dim), dtype=np.float32)
    )
    accelerator.wait_for_everyone()
    J_by_state = compute_j_by_state(
            dataset_name, specific_dataset, task_id,
            planner_checkpoint, reward_checkpoint, old_critic_checkpoint,
            backbone_layers, hidden_layers, hidden_dim,
            reward_hidden_layers, reward_hidden_dim, all_trajs,
            accelerator=accelerator,
    )
    accelerator.wait_for_everyone()
    if is_main:
        print(
            f"planner7 pools: all={len(all_pool)} near={len(near_pool)} "
            f"goal={len(goal_pool)} traj_length={traj_length}"
        )
        print("testing critic quality droping the failed episodes")
        """
        evaluate_critic(
                    dataset_name, specific_dataset, task_id, old_critic_checkpoint,
                    hidden_layers, hidden_dim, all_trajs, J_by_state, gamma, 
                    drop_timeouts=True, value_decode="symlog", reward_scale=500.0,
        )
        """
        evaluate_critic_hat_return(
                 dataset_name, specific_dataset, task_id,
                 old_critic_checkpoint, reward_checkpoint,  hidden_layers, hidden_dim, 
                 reward_hidden_layers, reward_hidden_dim,
                 all_trajs, gamma, drop_timeouts=True, value_decode="symlog",
        )
        print("testing critic quality keeping the failed episodes")
        """
        evaluate_critic(
                    dataset_name, specific_dataset, task_id, old_critic_checkpoint,
                    hidden_layers, hidden_dim, all_trajs, J_by_state, gamma,
                    drop_timeouts=False, value_decode="symlog", reward_scale=500.0,
         )
        """
        evaluate_critic_hat_return(
                 dataset_name, specific_dataset, task_id,
                 old_critic_checkpoint, reward_checkpoint,  hidden_layers, hidden_dim, 
                 reward_hidden_layers, reward_hidden_dim,
                 all_trajs, gamma, drop_timeouts=False, value_decode="symlog",
        )
        print()
        td_residual_stats(
                   dataset_name, specific_dataset, task_id, hidden_layers, hidden_dim, old_critic_checkpoint,
                   all_trajs, gamma,
        )
        print()
        value_grad_stats(
                   dataset_name, specific_dataset, task_id, hidden_layers, hidden_dim, old_critic_checkpoint,
                   all_trajs, batch_size = 256,
        )

    reset_pool = (
        _train_reset_pool(dataset_name, specific_dataset, task_id, n=n_reset)
        if mix_reset else None
    )

    Scale = get_Q_scale(dataset_name, specific_dataset, task_id)

    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )
    
    """
    def plans_to_batch(plans):
        s_planner = plans[..., :obs_dim]
        actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
        s_raw = s_planner * planner_std + planner_mean
        N, H, _ = s_raw.shape
        n_loc = H - 1
        s_for_r = (s_raw[:, :n_loc] - r_mean) / r_std
        r_hat = reward_net(
            s_for_r.reshape(N * n_loc, -1),
            actions[:, :n_loc].reshape(N * n_loc, -1),
        ).reshape(N, n_loc)
        r_hat = r_hat / Scale.Q_scale

        if lam is not None:
            plan_targets = torch.zeros(N, device=device)
            w = 1.0 - lam
            weight_sum = 0.0
            for L in range(1, n_loc + 1):
                discounts = gamma_pow_t[:L]
                disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                s_L = (s_raw[:, L] - c_mean) / c_std
                v_boot = symexp(target_critic(s_L))
                plan_targets = plan_targets + w * (disc_return + (gamma ** L) * v_boot)
                weight_sum += w
                w *= lam
            plan_targets = plan_targets / max(weight_sum, 1e-8)
        else:
            r_list = []
            for L in range(1, n_loc + 1):
                discounts = gamma_pow_t[:L]
                disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                s_L = (s_raw[:, L] - c_mean) / c_std
                v_boot = symexp(target_critic(s_L))
                r_list.append(disc_return + (gamma ** L) * v_boot)
            R = torch.stack(r_list, dim=1)
            plan_targets = R.mean(dim=1) - rho * R.std(dim=1, unbiased=False).clamp(min=0.0)

        s0_raw = s_raw[:, 0]
        s0_key = torch.round(s0_raw * 1e5) / 1e5
        unique_s0, inverse_indices = torch.unique(s0_key, dim=0, return_inverse=True)
        U = unique_s0.shape[0]
        averaged_targets = torch.zeros(U, device=device)
        counts = torch.zeros(U, device=device)
        averaged_targets.index_add_(0, inverse_indices, plan_targets)
        counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
        averaged_targets = (averaged_targets / counts.clamp(min=1.0)).detach()
        averaged_targets = symlog(averaged_targets)
        s0_critic = ((unique_s0 - c_mean) / c_std).detach()
        return s0_critic, averaged_targets

    def _stat(t):
        if t is None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return (
            t.mean().item(),
            t.std(unbiased=False).item(),
            t.min().item(),
            t.max().item(),
        )
    """

    critic.train()
    s_all = y_all = s_near = y_near = g_critic = None
    v_near = v_goal = None
    B_eff = 0
    sampling_seconds = 0.0

    for k in range(1, num_steps + 1):
        if (k - 1) % resample_every == 0:
            with torch.no_grad():
                sampling_started = time.perf_counter()
                n_all = max(8, batch_size // 3)
                n_near = max(8, batch_size // 3)
                n_goal_b = max(8, batch_size - n_all - n_near)
                if len(near_pool) == 0:
                    n_near = 0
                if len(goal_pool) == 0:
                    n_goal_b = 0
                if len(all_pool) == 0:
                    raise RuntimeError("planner7 all_pool is empty")

                plans_all, _ = _generate_feasible_plans_parallel(
                    play_pool=all_pool,
                    reset_pool=reset_pool,
                    planner=planner,
                    planner_proc=planner_proc,
                    planner_mean=planner_mean,
                    planner_std=planner_std,
                    kernels=kernels,
                    k_mean=k_mean,
                    k_std=k_std,
                    kernel_config=kernel_config,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    horizon=horizon,
                    steps_T=steps_T,
                    num_karras=num_karras,
                    eta=eta,
                    batch_size=n_all,
                    training_step=k,
                    vectorized_sampling=vectorized_sampling,
                    plan_chunk_size=plan_chunk_size,
                    device=device,
                    accelerator=accelerator,
                    use_mix_reset=mix_reset,
                )
                s_all, y_all = plans_to_batch(plans_all)
                B_eff = plans_all.shape[0]

                if n_near > 0 and len(near_pool) > 0:
                    plans_near, _ = _generate_feasible_plans_parallel(
                        play_pool=near_pool,
                        reset_pool=None,
                        planner=planner,
                        planner_proc=planner_proc,
                        planner_mean=planner_mean,
                        planner_std=planner_std,
                        kernels=kernels,
                        k_mean=k_mean,
                        k_std=k_std,
                        kernel_config=kernel_config,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        horizon=horizon,
                        steps_T=steps_T,
                        num_karras=num_karras,
                        eta=eta,
                        batch_size=n_near,
                        training_step=k + 10000,
                        vectorized_sampling=vectorized_sampling,
                        plan_chunk_size=plan_chunk_size,
                        device=device,
                        accelerator=accelerator,
                        use_mix_reset=False,
                    )
                    s_near, y_near = plans_to_batch(plans_near)
                else:
                    s_near, y_near = None, None

                if n_goal_b > 0 and len(goal_pool) > 0:
                    rng_g = np.random.RandomState(k + 20011)
                    n_g = min(len(goal_pool), n_goal_b)
                    g_idx = rng_g.randint(0, len(goal_pool), size=n_g)
                    g_raw = torch.as_tensor(
                        goal_pool[g_idx], device=device, dtype=torch.float32,
                    )
                    g_critic = ((g_raw - c_mean) / c_std).detach()
                else:
                    g_critic = None
                sampling_seconds = time.perf_counter() - sampling_started

        v_all = critic(s_all)
        loss_all = F.smooth_l1_loss(v_all, y_all, beta=1.0)

        if s_near is not None:
            v_near = critic(s_near)
            loss_near = F.smooth_l1_loss(v_near, y_near, beta=1.0)
        else:
            v_near = None
            loss_near = torch.zeros((), device=device, dtype=loss_all.dtype)

        if g_critic is not None:
            v_goal = critic(g_critic)
            loss_goal = F.smooth_l1_loss(v_goal, torch.zeros_like(v_goal), beta=1.0)
        else:
            v_goal = None
            loss_goal = torch.zeros((), device=device, dtype=loss_all.dtype)

        loss = w_all * loss_all + w_near * loss_near + w_goal * loss_goal

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        if log_every > 0 and k % log_every == 0 and is_main:
            yn = _stat(y_near)
            ya = _stat(y_all)
            pn = _stat(v_near)
            vg_mean = float("nan") if v_goal is None else v_goal.mean().item()
            vg_abs = float("nan") if v_goal is None else v_goal.abs().mean().item()
            print(
                f" step {k:>6}/{num_steps}"
                f" loss={loss.item():.4f}"
                f" Lnear={loss_near.item():.4f} Lgoal={loss_goal.item():.4f} Lall={loss_all.item():.4f}"
                f" | near_y {yn[0]:.3f}/{yn[1]:.3f} [{yn[2]:.3f},{yn[3]:.3f}]"
                f" | v_near {pn[0]:.3f}/{pn[1]:.3f}"
                f" | v_goal {vg_mean:.3f} abs={vg_abs:.3f}"
                f" | far_y {ya[0]:.3f}/{ya[1]:.3f} (w_all={w_all})"
                f" | pools near={len(near_pool)} goal={len(goal_pool)}"
                f" | sample={sampling_seconds:.2f}s"
            )
            wandb_log({
                wandb_step_metric: wandb_step_offset + k,
                f"{wandb_prefix}/loss": loss.item(),
                f"{wandb_prefix}/loss_near": loss_near.item(),
                f"{wandb_prefix}/loss_goal": loss_goal.item(),
                f"{wandb_prefix}/loss_all": loss_all.item(),
                f"{wandb_prefix}/near_y_mean": yn[0],
                f"{wandb_prefix}/near_y_std": yn[1],
                f"{wandb_prefix}/near_y_min": yn[2],
                f"{wandb_prefix}/near_y_max": yn[3],
                f"{wandb_prefix}/v_near_mean": pn[0],
                f"{wandb_prefix}/v_near_std": pn[1],
                f"{wandb_prefix}/v_goal_mean": vg_mean,
                f"{wandb_prefix}/v_goal_abs": vg_abs,
                f"{wandb_prefix}/far_y_mean": ya[0],
                f"{wandb_prefix}/far_y_std": ya[1],
                f"{wandb_prefix}/n_near": int(len(near_pool)),
                f"{wandb_prefix}/n_goal": int(len(goal_pool)),
                f"{wandb_prefix}/sampling_seconds": sampling_seconds,
            })
    
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)
        print("critic saved.")
    accelerator.wait_for_everyone()
    J_by_state = compute_j_by_state(
            dataset_name, specific_dataset, task_id,
            planner_checkpoint, reward_checkpoint, new_step,
            backbone_layers, hidden_layers, hidden_dim,
            reward_hidden_layers, reward_hidden_dim, all_trajs,
            accelerator=accelerator,
    )
    accelerator.wait_for_everyone()
    if is_main:
        print("testing critic quality droping the failed episodes")
        """
        evaluate_critic(
                    dataset_name, specific_dataset, task_id, new_step,
                    hidden_layers, hidden_dim, all_trajs, J_by_state, gamma, 
                    drop_timeouts=True, value_decode="symlog", reward_scale=500.0,
        )
        """
        evaluate_critic_hat_return(
                 dataset_name, specific_dataset, task_id,
                 new_step, reward_checkpoint,  hidden_layers, hidden_dim, 
                 reward_hidden_layers, reward_hidden_dim,
                 all_trajs, gamma, drop_timeouts=True, value_decode="symlog",
        )
        print("testing critic quality keeping the failed episodes")
        """
        evaluate_critic(
                    dataset_name, specific_dataset, task_id, new_step,
                    hidden_layers, hidden_dim, all_trajs, J_by_state, gamma, 
                    drop_timeouts=False, value_decode="symlog", reward_scale=500.0,
        )
        """
        evaluate_critic_hat_return(
                 dataset_name, specific_dataset, task_id,
                 new_step, reward_checkpoint,  hidden_layers, hidden_dim, 
                 reward_hidden_layers, reward_hidden_dim,
                 all_trajs, gamma, drop_timeouts=False, value_decode="symlog",
        )
        print()
        td_residual_stats(
                   dataset_name, specific_dataset, task_id, hidden_layers, hidden_dim, new_step,
                   all_trajs, gamma,
        )
        print()
        value_grad_stats(
                   dataset_name, specific_dataset, task_id, hidden_layers, hidden_dim, new_step,
                   all_trajs, batch_size = 256,
        )
    return 0.0, 1.0

