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


"""
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
    batch_size: int = 64,
    oversample: int = 8,
    horizon: int = 32,
    gamma: float = 0.99,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    task_id: Optional[int] = None,
    mix_reset: bool = True,
    n_reset: int = 64,
    device: Optional[torch.device] = None,
):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    )
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval().to(device)
    for p in planner.parameters():
        p.requires_grad_(False)

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
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers
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

    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=critic_checkpoint
    )
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

    play_pool = np.concatenate(
        [t["observations"] for t in trajs], axis=0
    ).astype(np.float32)

    if mix_reset and task_id is not None:
        env, _, _ = get_env(dataset_name, specific_dataset, task_id=task_id)
        rows = []
        for i in range(n_reset):
            ob, _ = env.reset(seed=10_000 + i, options=dict(task_id=task_id))
            rows.append(np.asarray(ob, dtype=np.float32))
        reset_pool = np.stack(rows, 0)
        n_r = batch_size // 2
        s0 = np.concatenate(
            [
                play_pool[np.random.randint(0, len(play_pool), batch_size - n_r)],
                reset_pool[np.random.randint(0, len(reset_pool), n_r)],
            ],
            0,
        )
    else:
        s0 = play_pool[np.random.randint(0, len(play_pool), batch_size)]

    plans = []
    for s in s0:
        s_p = planner_proc.preprocess(s)
        for _ in range(oversample):
            x = sample_euler_karras(
                s_p, planner, obs_dim, act_dim, horizon,
                num_steps=steps_T, num_karras=num_karras, eta=eta, device=device,
            )
            plans.append(torch.from_numpy(np.asarray(x)).float())
    plans = torch.stack(plans, 0).to(device)          # (M, H, ds+da)

    s_raw = plans[..., :obs_dim] * planner_std + planner_mean
    actions = torch.clamp(plans[..., obs_dim:], -1.0, 1.0)
    M, H, _ = s_raw.shape
    n = H - 1                                         # K = 1 .. n

    s_for_r = (s_raw[:, :n] - r_mean) / r_std
    r_hat = reward_net(
        s_for_r.reshape(M * n, -1),
        actions[:, :n].reshape(M * n, -1),
    ).reshape(M, n)                                   # raw r̂, no clamp / Q_scale

    gamma_pow = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )
    s_norm_c = (s_raw - c_mean) / c_std
    V = symexp(critic(s_norm_c.reshape(M * H, -1)).reshape(M, H))

    R = []
    for K in range(1, n + 1):
        disc = (gamma_pow[:K].unsqueeze(0) * r_hat[:, :K]).sum(dim=1)
        RK = disc + (gamma ** K) * V[:, K]
        R.append(RK)
    R = torch.stack(R, dim=1)                         # (M, n)  columns = K=1..n

    mean_K = R.mean(dim=0)                            # (n,)
    std_over_plans = R.std(dim=0, unbiased=False)     # (n,)
    std_over_K = R.std(dim=1, unbiased=False)         # (M,)  planner7's R_std
    mean_R = R.mean(dim=1)

    stats = {
        "n_plans": int(M),
        "K": list(range(1, n + 1)),
        "mean_R_per_K": mean_K.cpu().tolist(),
        "std_R_over_plans_per_K": std_over_plans.cpu().tolist(),
        "mean_std_K": float(std_over_K.mean()),
        "median_std_K": float(std_over_K.median()),
        "p90_std_K": float(std_over_K.quantile(0.90)),
        "mean_R_mean": float(mean_R.mean()),
        "mean_R_std": float(mean_R.std(unbiased=False)),
        "R_min": float(R.min()),
        "R_max": float(R.max()),
    }

    print("=== multi-horizon Bellman probe ===")
    print(f"plans={M}  mean_K std(R^K)={stats['mean_std_K']:.4f}  "
          f"p90 std_K={stats['p90_std_K']:.4f}")
    for K, m, s in zip(stats["K"], stats["mean_R_per_K"], stats["std_R_over_plans_per_K"]):
        print(f"  K={K:02d}  E[R^K]={m:.4f}  std_τ(R^K)={s:.4f}")

    return stats, R.cpu()

"""


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
    batch_size: int = 64,
    oversample: int = 8,
    horizon: int = 32,
    gamma: float = 0.99,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    task_id: Optional[int] = None,
    mix_reset: bool = True,
    n_reset: int = 64,
    plan_chunk_size: int = 256,
    accelerator=None,
):
    from accelerate import Accelerator
    import math
    import torch.distributed as dist

    if accelerator is None:
        accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process

    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset)

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
        dataset_name, specific_dataset, reward_checkpoint, task_id
    )
    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers
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

    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id=task_id, step=critic_checkpoint,
    )
    critic.load_state_dict(critic_state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    critic_stat = get_critic_stats(
        dataset_name, specific_dataset, task_id=task_id, step=critic_checkpoint,
    )
    c_mean = torch.as_tensor(critic_stat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(
        np.maximum(critic_stat.obs_std, 1e-3), device=device, dtype=torch.float32
    )

    play_pool = np.concatenate(
        [t["observations"] for t in trajs], axis=0
    ).astype(np.float32)

    if is_main:
        rng = np.random.RandomState(0)
        if mix_reset and task_id is not None:
            env, _, _ = get_env(dataset_name, specific_dataset, task_id=task_id)
            rows = [
                np.asarray(
                    env.reset(seed=10_000 + i, options=dict(task_id=task_id))[0],
                    dtype=np.float32,
                )
                for i in range(n_reset)
            ]
            reset_pool = np.stack(rows, 0)
            n_r = batch_size // 2
            s0 = np.concatenate(
                [
                    play_pool[rng.randint(0, len(play_pool), size=batch_size - n_r)],
                    reset_pool[rng.randint(0, len(reset_pool), size=n_r)],
                ],
                0,
            )
        else:
            s0 = play_pool[rng.randint(0, len(play_pool), size=batch_size)]
    else:
        s0 = np.empty((batch_size, play_pool.shape[1]), dtype=np.float32)

    s0_t = torch.from_numpy(s0).to(device)
    if accelerator.num_processes > 1:
        dist.broadcast(s0_t, src=0)
    s0 = s0_t.cpu().numpy()

    local_idx = np.array_split(
        np.arange(batch_size), accelerator.num_processes
    )[accelerator.process_index]
    local_s0 = s0[local_idx]

    norm_s0 = torch.as_tensor(
        np.stack([planner_proc.preprocess(s) for s in local_s0]),
        dtype=torch.float32,
        device=device,
    )
    conditions = norm_s0.repeat_interleave(oversample, dim=0)
    M_loc = conditions.shape[0]
    dim = obs_dim + act_dim
    t_grid, beta_1, sigma_grid = planner_karras_beta_schedule(steps_T, device=device)
    beta_2 = planner_cosine_beta(t_grid, s=0.008)

    chunks = []
    for start in range(0, M_loc, plan_chunk_size):
        stop = min(start + plan_chunk_size, M_loc)
        cond = conditions[start:stop]
        B = cond.shape[0]
        x = torch.randn(B, horizon, dim, device=device) * sigma_grid[0]
        mask = torch.zeros_like(x)
        mask[:, 0, :obs_dim] = 1.0
        cond_x = torch.zeros_like(x)
        cond_x[:, 0, :obs_dim] = cond
        x = mask * cond_x + (1 - mask) * x
        for i in range(steps_T):
            t_now = t_grid[i]
            t_next = t_grid[i + 1] if i < steps_T - 1 else 0.0
            dt = (t_next - t_now).item()
            beta_now = beta_1[i].item() if i < num_karras else beta_2[i].item()
            drift = -0.5 * beta_now * x
            score = planner(x, t_now.expand(B))
            if eta > 0:
                ns = eta * math.sqrt(beta_now * (-dt))
                x = x + (drift - beta_now * score) * dt + ns * torch.randn_like(x)
            else:
                x = x + (drift - beta_now * score) * dt
            x = mask * cond_x + (1 - mask) * x
            x[..., obs_dim:] = torch.clamp(x[..., obs_dim:], -1.0, 1.0)
        chunks.append(x)
    local_plans = torch.cat(chunks, dim=0)

    s_raw = local_plans[..., :obs_dim] * planner_std + planner_mean
    actions = torch.clamp(local_plans[..., obs_dim:], -1.0, 1.0)
    M, H, _ = s_raw.shape
    n = H - 1

    r_hat = reward_net(
        ((s_raw[:, :n] - r_mean) / r_std).reshape(M * n, -1),
        actions[:, :n].reshape(M * n, -1),
    ).reshape(M, n)

    gamma_pow = torch.tensor(
        [gamma ** t for t in range(n)], device=device, dtype=torch.float32
    )
    V = symexp(
        critic(((s_raw - c_mean) / c_std).reshape(M * H, -1)).reshape(M, H)
    )

    R_loc = torch.stack(
        [
            (gamma_pow[:K].unsqueeze(0) * r_hat[:, :K]).sum(1)
            + (gamma ** K) * V[:, K]
            for K in range(1, n + 1)
        ],
        dim=1,
    )

    if accelerator.num_processes > 1:
        gathered = [None] * accelerator.num_processes
        dist.all_gather_object(gathered, R_loc.cpu())
        R = torch.cat(gathered, dim=0) if is_main else None
    else:
        R = R_loc

    stats = None
    if is_main:
        R = R.to(device)
        std_K = R.std(dim=1, unbiased=False)
        mean_R = R.mean(dim=1)
        R1 = R[:, 0]
        RN = R[:, -1]
        ratio = RN / R1.clamp(min=1e-8)
        stats = {
            "n_plans": int(R.shape[0]),
            "mean_std_K": float(std_K.mean()),
            "mean_R_mean": float(mean_R.mean()),
            "E_R31_over_R1": float(ratio.mean()),
            "median_R31_over_R1": float(ratio.median()),
            "E_R31": float(RN.mean()),
            "E_R1": float(R1.mean()),
            "E_R31_div_E_R1": float((RN.mean() / R1.mean().clamp(min=1e-8)).item()),
        }
        print(f"plans={stats['n_plans']}")
        print(f"mean_std_K          = {stats['mean_std_K']:.4f}")
        print(f"mean_R_mean         = {stats['mean_R_mean']:.4f}")
        print(f"E[R^31 / R^1]       = {stats['E_R31_over_R1']:.4f}")
        print(f"median(R^31 / R^1)  = {stats['median_R31_over_R1']:.4f}")
        print(f"E[R^31] / E[R^1]    = {stats['E_R31_div_E_R1']:.4f}")

    accelerator.wait_for_everyone()
    #return stats, (R.cpu() if is_main else None)

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
          


