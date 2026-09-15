from pkgutil import get_data
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import ogbench as og
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import minari
import sys
import numpy as np
import torch
from typing import Optional
from sympy.calculus.util import continuous_domain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from collections import deque
import torch.distributed as dist
from accelerate import Accelerator
import gymnasium as gym
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Pretrain.utils import ema_smooth
from Pretrain.Dataset import get_dataset, get_env, Planner_Processor
from Pretrain.Planners.Backbone.Dit import DiT1d
import ogbench
from Finetuning.utils import (  
    DiT1d,
    Critic,
    SimpleReward,
    Planner_Processor,
    get_planner,
    get_reward_model,
    get_reward_stats,
    get_critic_model,
    get_critic_stats,
    get_Q_scale,
    sample_euler_karras,
    planner_karras_beta_schedule,
    planner_cosine_beta,
    symexp,
)
from typing import Optional, List, Union
from torch.utils.data import Dataset
import torch.nn as nn
from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Planners.Backbone.Sampler import (
    sample_euler_karras,
    clip_actions,
    karras_beta_schedule,
    cosine_beta,
)
from Finetuning.traj_reward5 import TotalReward_Critic, RewardConfig, TotalReward
from Pretrain.Transition_Kernel.Kernel_Backbone import (
    compute_log_density_mog,
)
import math
import torch
import torch.nn.functional as F
from Pretrain.utils import wandb_log



class Selector:
   
    def __init__(
        self,
        env_name,
        specific_env,
        RConfig: RewardConfig,
        reward_checkpoint: int,
        kernel_checkpoint: int,
        critic_checkpoint: Optional[int] = None,
        task_id: Optional[int] = None,
        lam: float = 0.0,
        n_candidates: int = 30,
    ):
        self.env_name = env_name
        self.specific_env = specific_env
        self.RConfig = RConfig
        self.task_id = task_id
        self.lam = lam
        self.n_candidates = n_candidates
        self.device = check_device()

        if critic_checkpoint is not None:
            self.model = TotalReward_Critic(
                self.device,
                RConfig,
                env_name,
                specific_env,
                reward_checkpoint,
                kernel_checkpoint,
                critic_checkpoint,
                task_id,
            )
        else:
            self.model = TotalReward(
                self.device,
                RConfig,
                env_name,
                specific_env,
                reward_checkpoint,
                kernel_checkpoint,
                task_id,
            )
        self.model.eval()

    def _flatten_sa(self, s: torch.Tensor, a: torch.Tensor, s_next: Optional[torch.Tensor] = None):
        N, T, d_s = s.shape
        s_f = s.reshape(N * T, d_s)
        a_f = a.reshape(N * T, a.shape[-1])
        if s_next is None:
            return s_f, a_f, N, T
        return s_f, a_f, s_next.reshape(N * T, d_s), N, T

    def _norm_obs_stat(self, s: torch.Tensor, stat) -> torch.Tensor:
        s_n = stat.norm_obs(s.detach().cpu().numpy())
        return torch.from_numpy(np.ascontiguousarray(s_n, dtype=np.float32)).to(s.device)

    def _constraint_c(self, s_norm: torch.Tensor, a: torch.Tensor, s_next_norm: torch.Tensor) -> torch.Tensor:
        model = self.model
        if model.config.type_kernel == "robust":
            total = None
            for kernel in model.kernels:
                mu, log_std = kernel(s_norm, a)
                lp = kernel.log_prob(s_next_norm, mu, log_std)
                total = lp if total is None else total + lp
            avg = total / len(model.kernels)
        else:
            avg = compute_log_density_mog(model.kernels, s_norm, a, s_next_norm)
        return F.softplus(model.config.min_log_prob - avg, beta=model.config.beta)

    def _decode_critic_value(self, v_raw: torch.Tensor) -> torch.Tensor:
        q_stats = getattr(self.model, "q_stats", None)
        if q_stats is not None:
            q_std = torch.as_tensor(q_stats.Q_std, device=v_raw.device, dtype=v_raw.dtype)
            q_mean = torch.as_tensor(q_stats.Q_mean, device=v_raw.device, dtype=v_raw.dtype)
            return q_std * v_raw + q_mean
        return symexp(v_raw)

    def _predict_total_reward(self, x: torch.Tensor) -> torch.Tensor:
        N, H, _ = x.shape
        d_s = self.model.config.d_s
        i = torch.arange(H - 1, device=x.device, dtype=x.dtype)
        gamma = self.model.config.critic_gamma
        lam = self.lam

        s = x[..., :d_s]
        a = x[..., d_s:]
        r = self.model.reward_net(self._norm_obs_stat(s, self.model.reward_stat), a)
        total = ((gamma ** i) * r[:, :-1]).sum(dim=-1) / H
        total = total + ((gamma ** (H - 1)) * r[:, -1]) / H

        s_k = self._norm_obs_stat(s, self.model.kernel_stat)
        s_t, a_t, s_tp, _, _ = self._flatten_sa(s_k[:, :-1], a[:, :-1], s_k[:, 1:])
        c = self._constraint_c(s_t, a_t, s_tp).view(N, H - 1)
        total = total - lam * (c.sum(dim=-1) / (H - 1))
        total = total + lam * self.model.config.delta
        return total

    def _predict_total_reward_critic(self, x: torch.Tensor) -> torch.Tensor:
        N, H, _ = x.shape
        d_s, d_c = self.model.config.d_s, self.model.config.critic_d_s
        i = torch.arange(H - 1, device=x.device, dtype=x.dtype)
        gamma = self.model.config.critic_gamma
        lam = self.lam

        s = x[..., :d_s]
        a = x[..., d_s:].clamp(-1.0, 1.0)

        r = self.model.reward_net(
            self._norm_obs_stat(s, self.model.reward_stat)[:, :-1], a[:, :-1]
        )
        total = (r * ((H - 1 - i) * (gamma ** i))).sum(dim=-1)

        s_c = self._norm_obs_stat(s[..., :d_c], self.model.critic_stat)[:, 1:]
        v_raw = self.model.critic(s_c.reshape(N * (H - 1), d_c)).reshape(N, H - 1)
        v = self._decode_critic_value(v_raw)
        total = total + (v * (gamma ** (i + 1))).sum(dim=-1)
        total = total / (H - 1)

        s_k = self._norm_obs_stat(s, self.model.kernel_stat)
        s_t, a_t, s_tp, _, _ = self._flatten_sa(s_k[:, :-1], a[:, :-1], s_k[:, 1:])
        c = self._constraint_c(s_t, a_t, s_tp).view(N, H - 1)
        total = total - lam * c.sum(dim=-1)
        total = total + lam * self.model.config.delta
        return total

    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, TotalReward_Critic):
            return self._predict_total_reward_critic(x)
        if isinstance(self.model, TotalReward):
            return self._predict_total_reward(x)
        raise TypeError(f"Unsupported selector model: {type(self.model)}")

    def select_plan(self, plans: Union[torch.Tensor, List[np.ndarray], np.ndarray]) -> np.ndarray:
        if isinstance(plans, list):
            if len(plans) == 0:
                raise ValueError("select_plan received an empty plan list")
            plans = torch.stack(
                [
                    p.detach().float() if isinstance(p, torch.Tensor)
                    else torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))
                    for p in plans
                ],
                dim=0,
            )
        elif isinstance(plans, np.ndarray):
            plans = torch.from_numpy(np.ascontiguousarray(plans, dtype=np.float32))

        plans = plans.detach().float().to(self.device)
        if plans.dim() == 2:
            plans = plans.unsqueeze(0)
        if plans.numel() == 0:
            raise ValueError("select_plan received an empty plan list")

        with torch.no_grad():
            rewards = self.predict_batch(plans)
            idx = int(torch.argmax(rewards).item())
        return plans[idx].detach().cpu().numpy().astype(np.float32, copy=True)

    @torch.no_grad()
    def sample_batch(
        self,
        s0: np.ndarray,
        score_model: torch.nn.Module,
        d_s: int,
        d_a: int,
        horizon: int,
        num_steps: int = 50,
        num_karras: int = 5,
        eta: float = 1.0,
        n_samples: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        device = device or self.device
        s0_t = torch.tensor(s0, device=device, dtype=torch.float32)
        if s0_t.shape[0] != d_s:
            raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")

        B = int(n_samples if n_samples is not None else self.n_candidates)
        dim = d_s + d_a

        t_grid, beta_1, sigma_grid = karras_beta_schedule(num_steps, device=device)
        beta_2 = cosine_beta(t_grid, s=0.008)

        x = torch.cat(
            [torch.randn(1, horizon, dim, device=device) * sigma_grid[0] for _ in range(B)],
            dim=0,
        )
        mask = torch.zeros(B, horizon, dim, device=device)
        mask[:, 0, :d_s] = 1.0
        y = torch.zeros_like(x)
        y[:, 0, :d_s] = s0_t
        x = mask * y + (1 - mask) * x

        for i in range(num_steps):
            t_now = t_grid[i]
            t_next = t_grid[i + 1] if i < num_steps - 1 else 0.0
            dt = (t_next - t_now).item()
            beta_now = (beta_1[i] if i < num_karras else beta_2[i]).item()

            drift = -0.5 * beta_now * x
            score = score_model(x, t_now.unsqueeze(0))

            if eta > 0:
                noise = torch.cat(
                    [torch.randn(1, horizon, dim, device=device) for _ in range(B)],
                    dim=0,
                )
                noise_scale = eta * math.sqrt(beta_now * (-dt))
                x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
            else:
                x = x + (drift - beta_now * score) * dt

            x = mask * y + (1 - mask) * x
            x = clip_actions(x, d_s)

        return x

    @torch.no_grad()
    def sample_selected_plan(
        self,
        current_state_norm,
        score_model,
        d_s: int,
        d_a: int,
        horizon: int,
        steps_T: int,
        num_karras: int,
        eta: float,
        device=None,
    ) -> np.ndarray:
        plans = self.sample_batch(
            current_state_norm,
            score_model,
            d_s,
            d_a,
            horizon,
            num_steps=steps_T,
            num_karras=num_karras,
            eta=eta,
            n_samples=self.n_candidates,
            device=device or self.device,
        )
        return self.select_plan(plans)




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
    rho: float = 1.0,          # conservatism: R_target = R_mean - rho * R_std (used when lam is None)
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
    log_every: int = 0,
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
    
    def _split(traj_list, suffix_only=False):
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
                      masks = np.concatenate([masks, np.ones(n - len(masks), np.float32)])
              play.append(obs[masks != 0.0])
              if not suffix_only:
                  goal.append(obs[masks == 0.0])
        play_pool = np.concatenate(play, 0) if any(len(x) for x in play) else np.zeros((0, obs_dim), np.float32)
        if suffix_only:
              return play_pool
        goal_pool = np.concatenate(goal, 0) if any(len(x) for x in goal) else np.zeros((0, obs_dim), np.float32)
        return play_pool, goal_pool
  
    # === NEW === train resets; seeds 10000+ disjoint from eval 0..999
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

 


    # ------------------------------------------------------------------ setup
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
    # critic
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

    # planner
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

    # reward
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

    # kernel
    kernels, k_mean, k_std = load_kernel_ensemble(
        dataset_name, specific_dataset, kernel_config, obs_dim, act_dim, device,
    )

    # critic stats
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

   
    """
    play_pool = np.concatenate(
            [t['observations'] for t in trajs], axis=0,
    ).astype(np.float32)
    """

    """
    play_rows, goal_rows = [], []
    for traj in trajs:
            obs = np.asarray(traj["observations"], dtype=np.float32)
            n = len(obs)
            raw = traj.get("masks", None)
            if raw is None:
                masks = np.ones(n, dtype=np.float32)
            else:
                masks = np.asarray(raw[:n], dtype=np.float32)
                if len(masks) < n:
                    masks = np.concatenate(
                             [masks, np.ones(n - len(masks), dtype=np.float32)],
                             axis=0,
                    )
            play_rows.append(obs[masks != 0.0])
            goal_rows.append(obs[masks == 0.0])

    play_pool = (
                  np.concatenate(play_rows, axis=0)
                  if any(len(x) for x in play_rows)
                  else np.zeros((0, obs_dim), dtype=np.float32)
    )
    goal_pool = (
                np.concatenate(goal_rows, axis=0)
                if any(len(x) for x in goal_rows)
                else np.zeros((0, obs_dim), dtype=np.float32)
    )
    """
    

    all_pool, goal_pool = _split(all_trajs)
    near_pool = _split(near_trajs, suffix_only=True) if len(near_trajs) else np.zeros((0, obs_dim), np.float32)


    reset_pool = (
            _train_reset_pool(dataset_name, specific_dataset, task_id, n=n_reset)
            if mix_reset else None
    )
   

    Scale = get_Q_scale(dataset_name, specific_dataset, task_id)
    running_tgt_mean = torch.zeros(1, device=device)
    running_tgt_std = torch.ones(1, device=device)

    alpha = 0.99

    # optim
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=min_lr,
    )

    # prepare only trainable parts
    critic, optimizer, scheduler = accelerator.prepare(critic, optimizer, scheduler)

    n = horizon - 1
    gamma_pow_t = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )
    def plans_to_batch(plans):
           s_planner = plans[..., :obs_dim]
           actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
           s_raw = s_planner * planner_std + planner_mean

           N, H, _ = s_raw.shape
           n = H - 1

           # rewards for t = 0 .. n-1
           s_for_r = (s_raw[:, :n] - r_mean) / r_std
           r_hat = reward_net(
                 s_for_r.reshape(N * n, -1),
                 actions[:, :n].reshape(N * n, -1),
           ).reshape(N, n)  # (N, n)

               # reward clipping -----------------------------------------------------
               #r_hat = torch.clamp(r_hat, float('-inf'), 0.0)
               #r_hat = torch.clamp(r_hat, 0.0, float('inf'))
           r_hat = r_hat / Scale.Q_scale

           plan_targets = torch.zeros(N, device=device)

           if lam is not None:
                 # λ-return (unchanged)
                 w = 1.0 - lam
                 weight_sum = 0.0

                 for L in range(1, n + 1):  # L = 1 .. n
                       discounts = gamma_pow_t[:L]
                       disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                       s_L = (s_raw[:, L] - c_mean) / c_std
                       v_boot = target_critic(s_L)
                       v_boot = symexp(v_boot)
                       #v_boot = torch.clamp(v_boot, float('-inf'), 0.0)
                       #v_boot = (v_boot * running_tgt_std) + running_tgt_mean
                       partial = disc_return + (gamma ** L) * v_boot
                       plan_targets += w * partial
                       weight_sum += w
                       w *= lam

                 plan_targets = plan_targets / max(weight_sum, 1e-8)

           else:
                 r_list = []
                 for L in range(1, n + 1):
                       discounts = gamma_pow_t[:L]
                       disc_return = (discounts.unsqueeze(0) * r_hat[:, :L]).sum(dim=1)
                       s_L = (s_raw[:, L] - c_mean) / c_std
                       v_boot = target_critic(s_L)
                       #print(f"critic value normalized: {v_boot.mean().item()}")
                       #v_boot = (v_boot * running_tgt_std) + running_tgt_mean
                       #print(f"critic value denormalized: {v_boot.mean().item()}")
                       v_boot = symexp(v_boot)
                       #v_boot = torch.clamp(v_boot, float('-inf'), 0.0)
                       partial = disc_return + (gamma ** L) * v_boot
                       r_list.append(partial)

                 R = torch.stack(r_list, dim=1)  # (N, n-1)
                 R_mean = R.mean(dim=1)          # (N,)
                 R_std = R.std(dim=1, unbiased=False).clamp(min=0.0)  # (N,)
                 plan_targets = R_mean - rho * R_std

           # ----- average targets per unique s0 -----
           s0_raw = s_raw[:, 0]
           s0_key = torch.round(s0_raw * 1e5) / 1e5

           unique_s0, inverse_indices = torch.unique(
               s0_key, dim=0, return_inverse=True
           )

           U = unique_s0.shape[0]
           averaged_targets = torch.zeros(U, device=device)
           counts = torch.zeros(U, device=device)

           averaged_targets.index_add_(0, inverse_indices, plan_targets)
           counts.index_add_(0, inverse_indices, torch.ones_like(plan_targets))
           averaged_targets = averaged_targets / counts.clamp(min=1.0)

           averaged_targets = averaged_targets.detach()
           #averaged_targets = averaged_targets.clamp(float('-inf'), 0.0)
           averaged_targets = symlog(averaged_targets)

           # running normalization
           batch_mean = averaged_targets.mean()
           batch_std = averaged_targets.std(unbiased=False) + 1e-8
           nonlocal running_tgt_mean, running_tgt_std
           running_tgt_mean = alpha * running_tgt_mean + (1 - alpha) * batch_mean
           running_tgt_std = alpha * running_tgt_std + (1 - alpha) * batch_std
           #normalized_target = (averaged_targets - running_tgt_mean) / running_tgt_std

            # critic input
           s0_critic = (unique_s0 - c_mean) / c_std
           s0_critic = s0_critic.detach()
           return s0_critic, averaged_targets
    
    critic.train()
    running = 0.0
    total_mae = 0.0
    total_bias = 0.0
    sampling_seconds = 0.0
    #n_resamples = max(1, num_steps // resample_every)
    #increments = max(1, (max_length + n_resamples - 1) // n_resamples)  # ceil
    for k in range(1, num_steps + 1):
        if (k - 1) % resample_every == 0:

            #n_resamples = max(1, num_steps // resample_every)
            #increments = max(1, (max_length + n_resamples - 1) // n_resamples)  # ceil

            with torch.no_grad():
                 sampling_started = time.perf_counter()
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
                        batch_size=batch_size,
                        training_step=k,
                        vectorized_sampling=vectorized_sampling,
                        plan_chunk_size=plan_chunk_size,
                        device=device,
                        accelerator=accelerator,
                        use_mix_reset=mix_reset,
                 )
                 s_all, y_all = plans_to_batch(plans_all)
                 B_eff = plans_all.shape[0]
                 U = s_all.shape[0]

                 if len(near_pool) > 0:
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
                                batch_size=batch_size,
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

                 sampling_seconds = time.perf_counter() - sampling_started

                 if len(goal_pool) > 0:
                         rng_g = np.random.RandomState(k + 20011)
                         n_goal = min(len(goal_pool), batch_size)
                         g_idx = rng_g.randint(0, len(goal_pool), size=n_goal)
                         g_raw = torch.as_tensor(
                                 goal_pool[g_idx], device=device, dtype=torch.float32,
                         )
                         g_critic = ((g_raw - c_mean) / c_std).detach()
                 else:
                         g_critic = None
              

        # gradient step
        """
        v_pred = critic(s0_critic)
        with torch.no_grad():
            pred_mean = v_pred.detach().mean()
            pred_std = v_pred.detach().std(unbiased=False)
            bias = (v_pred - averaged_targets).mean()
            mae = (v_pred - averaged_targets).abs().mean()
        #loss = F.smooth_l1_loss(v_pred, normalized_target, beta=1.0)
        loss = F.smooth_l1_loss(v_pred, averaged_targets, beta=1.0)
        #loss = F.mse_loss(v_pred, averaged_targets)
        """

        """
        # gradient step
        v_pred = critic(s0_critic)
        loss_play = F.smooth_l1_loss(v_pred, averaged_targets, beta=1.0)

        if g_critic is not None:
            v_goal = critic(g_critic)
            loss_goal = F.smooth_l1_loss(
                    v_goal, torch.zeros_like(v_goal), beta=1.0,
            )
        else:
            loss_goal = torch.zeros((), device=device, dtype=loss_play.dtype)

        loss = loss_play + loss_goal
        """
        v_pred = critic(s_all)
        loss_all = F.smooth_l1_loss(v_pred, y_all, beta=1.0)

        if s_near is not None:
               loss_near = F.smooth_l1_loss(critic(s_near), y_near, beta=1.0)
        else:
               loss_near = torch.zeros((), device=device, dtype=loss_all.dtype)

        if g_critic is not None:
               v_goal = critic(g_critic)
               loss_goal = F.smooth_l1_loss(
                     v_goal, torch.zeros_like(v_goal), beta=1.0,
                )
        else:
               loss_goal = torch.zeros((), device=device, dtype=loss_all.dtype)

        loss = loss_all + loss_near + loss_goal

        with torch.no_grad():
             pred_mean = v_pred.detach().mean()
             pred_std = v_pred.detach().std(unbiased=False)
             bias = (v_pred - y_all).mean()
             mae = (v_pred - y_all).abs().mean()




        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Polyak update
        with torch.no_grad():
            unwrapped = accelerator.unwrap_model(critic)
            for p, tp in zip(unwrapped.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        running += loss.item()
        total_mae += mae.item()
        total_bias += bias.item()
        
        
        if log_every > 0 and k % log_every == 0 and is_main:
            avg_loss = running / log_every
            avg_mae = total_mae / log_every
            avg_bias = total_bias / log_every
            
            """
            with torch.no_grad():
                 logged_targets = symexp(y_all)
            print(f"tgt_mean: {running_tgt_mean.item()}, tgt_std: {running_tgt_std.item()}")
            wandb_log({
                    wandb_step_metric: wandb_step_offset + k,
                    f"{wandb_prefix}/loss": avg_loss,
                    f"{wandb_prefix}/pred_mean": pred_mean.item(),
                    f"{wandb_prefix}/pred_std": pred_std.item(),
                    f"{wandb_prefix}/target_mean": logged_targets.mean().item(),
                    f"{wandb_prefix}/target_std": logged_targets.std().item(),
                    f"{wandb_prefix}/target_min": logged_targets.min().item(),
                    f"{wandb_prefix}/target_max": logged_targets.max().item(),
                    f"{wandb_prefix}/bias": avg_bias,
                    f"{wandb_prefix}/mae": avg_mae,
                    f"{wandb_prefix}/sampling_seconds": sampling_seconds,
                    f"{wandb_prefix}/plans_per_second": B_eff / max(sampling_seconds, 1e-8),
            })
            """
            with torch.no_grad():
                  parts = [y_all]
                  if y_near is not None:
                         parts.append(y_near)
                  if g_critic is not None:
                         parts.append(torch.zeros(g_critic.shape[0], device=device, dtype=y_all.dtype))
                  y_head = torch.cat(parts, dim=0)
                  logged_targets = symexp(y_head)
                  logged_all = symexp(y_all)
                  logged_near = symexp(y_near) if y_near is not None else None
                  print(f"tgt_mean: {running_tgt_mean.item()}, tgt_std: {running_tgt_std.item()}")
                  wandb_log({
                            wandb_step_metric: wandb_step_offset + k,
                            f"{wandb_prefix}/loss": avg_loss,
                            f"{wandb_prefix}/loss_all": loss_all.item(),
                            f"{wandb_prefix}/loss_near": loss_near.item(),
                            f"{wandb_prefix}/loss_goal": loss_goal.item(),
                            f"{wandb_prefix}/pred_mean": pred_mean.item(),
                            f"{wandb_prefix}/pred_std": pred_std.item(),
                            f"{wandb_prefix}/target_mean": logged_targets.mean().item(),
                            f"{wandb_prefix}/target_std": logged_targets.std().item(),
                            f"{wandb_prefix}/target_min": logged_targets.min().item(),
                            f"{wandb_prefix}/target_max": logged_targets.max().item(),
                            f"{wandb_prefix}/target_all_mean": logged_all.mean().item(),
                            f"{wandb_prefix}/target_near_mean": (
                                logged_near.mean().item() if logged_near is not None else 0.0
                             ),
                            f"{wandb_prefix}/bias": avg_bias,
                            f"{wandb_prefix}/mae": avg_mae,
                            f"{wandb_prefix}/sampling_seconds": sampling_seconds,
                            f"{wandb_prefix}/plans_per_second": B_eff / max(sampling_seconds, 1e-8),
                            })
            print(
                f" step {k:>6}/{num_steps} "
                f"loss = {avg_loss:.10f}  "
                f"B_eff={B_eff}  U={U}  "
                f"pred_mean={pred_mean.item():.3f}  "
                f"pred_std={pred_std.item():.3f}  "
                f"tgt_mean={y_all.mean().item():.3f}  "
                f"tgt_std={y_all.std().item():.3f}  "
                f"tgt_min={y_all.min().item():.3f}  "
                f"tgt_max={y_all.max().item():.3f}  "
                f"bias={avg_bias:.3f}  "
                f"mae={avg_mae:.3f}"
                f"  sampling={sampling_seconds:.2f}s"
            )
            running = 0.0
            total_bias = 0.0
            total_mae = 0.0

    # final save
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped_critic = accelerator.unwrap_model(critic)
        target_critic.load_state_dict(unwrapped_critic.state_dict())
        target_critic.eval()
        save_critic(target_critic, dataset_name, specific_dataset, task_id, new_step)

        
        print("critic saved.")

    return running_tgt_mean.item(), running_tgt_std.item()







@torch.no_grad()
def probe_multi_horizon_bellman(
    trajs,
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    critic_checkpoint: int,
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    reward_hidden_layers: int = 1,
    reward_hidden_dim: int = 128,
    n_s0: int = 64,
    n_plans_per_s0: int = 16,
    horizon: int = 32,
    gamma: float = 0.99,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    task_id: Optional[int] = None,
    mix_reset: bool = True,
    n_reset: int = 64,
    plan_chunk_size: int = 256,
    eps: float = 1e-8,
    accelerator=None,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    L = n_plans_per_s0

    _, d_s, d_a = get_env(dataset_name, specific_dataset)

    planner = DiT1d(
        in_dim=(d_s + d_a), emb_dim=128, d_model=256,
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
        dataset_name, specific_dataset, reward_checkpoint, task_id
    )
    reward_net = SimpleReward(
        d_s, d_a, reward_hidden_dim, reward_hidden_layers
    ).to(device)
    reward_net.load_state_dict(reward_state)
    reward_net.eval()
    for p in reward_net.parameters():
        p.requires_grad_(False)
    reward_stat = get_reward_stats(
        dataset_name, specific_dataset, reward_checkpoint, task_id
    )
    r_mean = torch.as_tensor(reward_stat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(
        np.maximum(reward_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    critic = Critic(d_s, hidden_dim, hidden_layers).to(device)
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=critic_checkpoint
    )
    q_scale = get_Q_scale(dataset_name, specific_dataset, task_id)
    critic.load_state_dict(critic_state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset, task_id=task_id, step=0
    )
    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    play = np.concatenate([t["observations"] for t in trajs], 0).astype(np.float32)
    if is_main:
        rng = np.random.RandomState(0)
        if mix_reset and task_id is not None:
            env, _, _ = get_env(dataset_name, specific_dataset, task_id=task_id)
            reset = np.stack(
                [
                    np.asarray(
                        env.reset(seed=10_000 + i, options=dict(task_id=task_id))[0],
                        dtype=np.float32,
                    )
                    for i in range(n_reset)
                ],
                0,
            )
            n_r = n_s0 // 2
            s0 = np.concatenate(
                [
                    play[rng.randint(0, len(play), size=n_s0 - n_r)],
                    reset[rng.randint(0, len(reset), size=n_r)],
                ],
                0,
            )
        else:
            s0 = play[rng.randint(0, len(play), size=n_s0)]
    else:
        s0 = np.empty((n_s0, play.shape[1]), dtype=np.float32)

    s0_t = torch.from_numpy(s0).to(device)
    if accelerator.num_processes > 1:
        dist.broadcast(s0_t, src=0)
    s0 = s0_t.cpu().numpy()

    local_idx = np.array_split(np.arange(n_s0), accelerator.num_processes)[
        accelerator.process_index
    ]
    local_s0 = s0[local_idx]
    M_loc = int(local_s0.shape[0])

    if M_loc == 0:
        R_s_loc = torch.zeros(0, horizon - 1, device="cpu")
    else:
        s0_norm = torch.as_tensor(
            np.stack([planner_proc.preprocess(s) for s in local_s0]),
            dtype=torch.float32, device=device,
        )
        cond = s0_norm.repeat_interleave(L, dim=0)
        dim = d_s + d_a
        t_grid, beta_1, sigma_grid = planner_karras_beta_schedule(steps_T, device=device)
        beta_2 = planner_cosine_beta(t_grid, s=0.008)
        chunks = []
        for start in range(0, cond.shape[0], plan_chunk_size):
            c = cond[start : start + plan_chunk_size]
            b = c.shape[0]
            x = torch.randn(b, horizon, dim, device=device) * sigma_grid[0]
            mask = torch.zeros_like(x)
            mask[:, 0, :d_s] = 1.0
            cond_x = torch.zeros_like(x)
            cond_x[:, 0, :d_s] = c
            x = mask * cond_x + (1.0 - mask) * x
            for i in range(steps_T):
                t_now = t_grid[i]
                t_next = t_grid[i + 1] if i < steps_T - 1 else 0.0
                dt = (t_next - t_now).item()
                beta_now = beta_1[i].item() if i < num_karras else beta_2[i].item()
                drift = -0.5 * beta_now * x
                score = planner(x, t_now.expand(b))
                if eta > 0:
                    ns = eta * math.sqrt(beta_now * (-dt))
                    x = x + (drift - beta_now * score) * dt + ns * torch.randn_like(x)
                else:
                    x = x + (drift - beta_now * score) * dt
                x = mask * cond_x + (1.0 - mask) * x
                x[..., d_s:] = torch.clamp(x[..., d_s:], -1.0, 1.0)
            chunks.append(x)
        plans = torch.cat(chunks, dim=0)

        s_raw = plans[..., :d_s] * planner_std + planner_mean
        actions = torch.clamp(plans[..., d_s:], -1.0, 1.0)
        P, H, _ = s_raw.shape
        n = H - 1
        r_hat = reward_net(
            ((s_raw[:, :n] - r_mean) / r_std).reshape(P * n, -1),
            actions[:, :n].reshape(P * n, -1),
        ).reshape(P, n)
        #r_hat = torch.clamp(r_hat, float('-inf'), 0.0)
        r_hat = r_hat / q_scale.get_Q_scale()
        V = symexp(
            critic(((s_raw - c_mean) / c_std).reshape(P * H, -1)).reshape(P, H)
        )
        gpow = torch.tensor(
            [gamma ** t for t in range(n)], device=device, dtype=torch.float32
        )
        cuts = []
        for K in range(1, n+1):  # R^(L) = sum_{t=0}^{L-1} γ^t r_t + γ^L V(s_L)
               disc = (gpow[:K].unsqueeze(0) * r_hat[:, :K]).sum(1)
               cuts.append(disc + (gamma ** K) * V[:, K])
        R_tau = torch.stack(cuts, dim=1).view(M_loc, L, -1)
        R_s_loc = R_tau.mean(dim=1).cpu()  # (M_loc, nK)  E_τ first

    if accelerator.num_processes > 1:
        gathered = [None] * accelerator.num_processes
        dist.all_gather_object(gathered, R_s_loc)
        R_s = torch.cat(gathered, dim=0) if is_main else None
    else:
        R_s = R_s_loc

    stats = None
    if is_main:
        R_s = R_s.to(device)
        m_s = R_s.mean(dim=1)
        std_s = R_s.std(dim=1, unbiased=False)
        R1, RNm1 = R_s[:, 0], R_s[:, -1]  # R^(1), R^(n-1)
        denom = R1.sign().clamp(min=0) * 2 - 1  # +1 if R1>=0, -1 if R1<0
        denom = denom * R1.abs().clamp(min=eps)
        ratio_s = RNm1 / denom
        ok = R1.abs() > eps
        M = int(R_s.shape[0])
        ratio_ok = ratio_s[ok]
        stats = {
               "n_s0": M,
               "n_plans_per_s0": L,
               "mean_of_RK": float(m_s.mean()),
               "mean_of_STD": float(std_s.mean()),
               "ratio": float(ratio_ok.mean()) if ok.any() else float("nan"),
               "se_mean_of_RK": float(m_s.std(unbiased=True) / math.sqrt(M)),
               "se_mean_of_STD": float(std_s.std(unbiased=True) / math.sqrt(M)),
               "se_ratio": float(ratio_ok.std(unbiased=True) / math.sqrt(int(ok.sum().clamp(min=1)))) if ok.any() else float("nan"),
               "median_ratio": float(ratio_ok.median()) if ok.any() else float("nan"),
               "n_ratio": int(ok.sum().item()),
               "E_RNm1_div_E_R1": float(
                  (RNm1.mean() / (R1.mean().sign() * R1.mean().abs().clamp(min=eps))).item()
                ),
         }
        print("=== slide 4.1–4.3 (E_τ per s, then s) ===")
        print(f"M={M}  L={L}")
        print(f"mean_of_RK         = {stats['mean_of_RK']:.4f}  se={stats['se_mean_of_RK']:.4f}")
        print(f"mean_of_STD        = {stats['mean_of_STD']:.4f}  se={stats['se_mean_of_STD']:.4f}")
        print(f"ratio              = {stats['ratio']:.4f}  se={stats['se_ratio']:.4f}")
        wandb_log({
                 "checkpoint": critic_checkpoint,
                 "mean_of_RK": stats["mean_of_RK"],
                 "mean_of_STD": stats["mean_of_STD"],
                 "ratio": stats["ratio"],
        })

    accelerator.wait_for_everyone()
    return stats




import pickle
import torch
from Pretrain.Critic.nets import Critic
from Finetuning.utils import symexp
from Pretrain.Rewards.nets import SimpleReward

def critic_heatmap(checkpoint: int, show: bool = True):
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle

    ckpt = f"Finetuning/Critics/antmaze/large/Models/AntMaze_Large_task4_Critic_{checkpoint}.pkl"
    stats_path = "Finetuning/Critics/antmaze/large/Stats/AntMaze_Large_task4_Critic_stats_0.pkl"

    critic = Critic(obs_dim=29, hidden_dim=512, hidden_layers=4)
    critic.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    critic.eval()

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    xs = np.linspace(-6, 42, 200)
    ys = np.linspace(-6, 30, 160)
    XX, YY = np.meshgrid(xs, ys)
    obs = np.broadcast_to(stats.obs_mean, (XX.size, 29)).copy()
    obs[:, 0] = XX.ravel()
    obs[:, 1] = YY.ravel()
    s = stats.norm_obs(obs)
    with torch.no_grad():
        V = symexp(critic(torch.as_tensor(s, dtype=torch.float32))).numpy().reshape(XX.shape)
    V = np.maximum(V, 0.0)

    MAZE = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    UNIT, OFF = 4.0, 4.0

    def ij_to_xy(ij):
        i, j = ij
        return j * UNIT - OFF, i * UNIT - OFF

    def xy_to_ij(xy):
        return (int((xy[1] + OFF + 0.5 * UNIT) / UNIT),
                int((xy[0] + OFF + 0.5 * UNIT) / UNIT))

    wall = np.zeros_like(V, dtype=bool)
    for r in range(XX.shape[0]):
        for c in range(XX.shape[1]):
            i, j = xy_to_ij((XX[r, c], YY[r, c]))
            if not (0 <= i < MAZE.shape[0] and 0 <= j < MAZE.shape[1]) or MAZE[i, j] == 1:
                wall[r, c] = True

    V_plot = np.ma.array(V, mask=wall)
    vmax = float(np.nanpercentile(V[~wall], 99.5))
    vmax = max(vmax, 1e-3)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("0.85")  # free cells at V=0 stay visible vs walls
    im = ax.pcolormesh(
           XX, YY, V_plot, shading="auto", cmap="magma",
           norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
    )
    plt.colorbar(im, ax=ax, label=r"$V(s)$")

    for i in range(MAZE.shape[0]):
        for j in range(MAZE.shape[1]):
            if MAZE[i, j] == 1:
                cx, cy = ij_to_xy((i, j))
                ax.add_patch(Rectangle(
                    (cx - UNIT / 2, cy - UNIT / 2), UNIT, UNIT,
                    facecolor="0.25", edgecolor="none", zorder=2,
                ))

    ax.scatter(*ij_to_xy((3, 8)), c="lime", s=60, zorder=3, label="start")
    ax.scatter(*ij_to_xy((5, 4)), c="cyan", s=80, marker="*", zorder=3, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"task4 critic @ {checkpoint}")
    ax.legend()
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    ax.legend(loc="upper left", bbox_to_anchor=(1.28, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    out = f"critic_heatmap_task4_{checkpoint}.png"
    plt.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out

def reward_heatmap(checkpoint: int = 0, show: bool = True):
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle

    ckpt = f"Finetuning/Rewards/antmaze/large/Models/AntMaze_Large_Task4_Reward_{checkpoint}.pkl"
    stats_path = "Finetuning/Rewards/antmaze/large/Stats/AntMaze_Large_Task4_Reward_stats_0.pkl"

    reward_net = SimpleReward(obs_dim=29, act_dim=8, hidden_dim=512, hidden_layers=4)
    reward_net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    reward_net.eval()

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    xs = np.linspace(-6, 42, 200)
    ys = np.linspace(-6, 30, 160)
    XX, YY = np.meshgrid(xs, ys)
    obs = np.broadcast_to(stats.obs_mean, (XX.size, 29)).copy()
    obs[:, 0] = XX.ravel()
    obs[:, 1] = YY.ravel()
    s = torch.as_tensor(stats.norm_obs(obs), dtype=torch.float32)
    a = torch.zeros(s.shape[0], 8)
    with torch.no_grad():
        R = reward_net(s, a).numpy().reshape(XX.shape)
    R = np.maximum(R, 0.0)

    MAZE = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    UNIT, OFF = 4.0, 4.0

    def ij_to_xy(ij):
        i, j = ij
        return j * UNIT - OFF, i * UNIT - OFF

    def xy_to_ij(xy):
        return (int((xy[1] + OFF + 0.5 * UNIT) / UNIT),
                int((xy[0] + OFF + 0.5 * UNIT) / UNIT))

    wall = np.zeros_like(R, dtype=bool)
    for r in range(XX.shape[0]):
        for c in range(XX.shape[1]):
            i, j = xy_to_ij((XX[r, c], YY[r, c]))
            if not (0 <= i < MAZE.shape[0] and 0 <= j < MAZE.shape[1]) or MAZE[i, j] == 1:
                wall[r, c] = True

    R_plot = np.ma.array(R, mask=wall)
    vmax = float(np.nanpercentile(R[~wall], 99.5))
    vmax = max(vmax, 1e-3)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("0.85")
    im = ax.pcolormesh(
        XX, YY, R_plot, shading="auto", cmap="magma",
        norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
    )
    plt.colorbar(im, ax=ax, label=r"$r(s, a=0)$")

    for i in range(MAZE.shape[0]):
        for j in range(MAZE.shape[1]):
            if MAZE[i, j] == 1:
                cx, cy = ij_to_xy((i, j))
                ax.add_patch(Rectangle(
                    (cx - UNIT / 2, cy - UNIT / 2), UNIT, UNIT,
                    facecolor="0.25", edgecolor="none", zorder=2,
                ))

    ax.scatter(*ij_to_xy((3, 8)), c="lime", s=60, zorder=3, label="start")
    ax.scatter(*ij_to_xy((5, 4)), c="cyan", s=80, marker="*", zorder=3, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"task4 reward @ {checkpoint}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.28, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    out = f"reward_heatmap_task4_{checkpoint}.png"
    plt.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out




"""
if __name__ == "__main__":
    critic_heatmap(0)
    #reward_heatmap(0)
"""



"""
env, dataset, eval_dataset = ogbench.make_env_and_datasets(
                 "cube-single-play-singletask-task4-v0", render_mode="rgb_array"
            )

Dict = {}

temp = 0
for i in range(len(dataset['observations'])):
     if(dataset['rewards'][i] == 0):
        if(temp > 0):
             continue 
        else:
             temp = i
             
     else:
         if(dataset['rewards'][i-1] == 0):
              print(dataset['terminals'][i-1])
              if( (i - temp) not in Dict.keys()):
                  Dict[(i - temp)] = 1
              else:
                  Dict[(i - temp)] += 1
              temp = 0
                   
"""
          


