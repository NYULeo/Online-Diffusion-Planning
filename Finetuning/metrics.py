
from typing import List, Optional
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import Dataset, DataLoader
from Pretrain.Critic.nets import Critic
from accelerate import Accelerator


# metrics for critic quality evaluation
def explained_variance(g: np.ndarray, v: np.ndarray) -> float:
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    var_g = float(np.var(g))
    if var_g < 1e-12:
        return float("nan")
    return float(1.0 - np.var(g - v) / var_g)

def spearman_correlation(g: np.ndarray, v: np.ndarray) -> float:
    return float(spearmanr(g, v).correlation)

def within_state_j_dispersion(J_by_state: np.ndarray, eps: float = 1e-8):
    
    J = np.asarray(J_by_state, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] < 2:
        raise ValueError("J_by_state must be (n_s0, K) with K>=2")
    sig = J.std(axis=1, ddof=0)
    mu = J.mean(axis=1)
    wsjd = float(sig.mean())
    btw = float(mu.std(ddof=0))
    alive = float((sig > eps).mean())
    print(
        f"WSJD (mean_s std_k J)={wsjd:.4f}  "
        f"between-s std(E[J|s])={btw:.4f}  "
        f"frac_alive={alive:.3f}  "
        f"n_s0={J.shape[0]} K={J.shape[1]}"
    )
    return {
        "wsjd": wsjd,
        "between_std": btw,
        "frac_alive": alive,
        "sigma_s": sig,
        "mu_s": mu,
    }







@torch.no_grad()
def evaluate_critic(
    dataset_name: str,
    specific_dataset: str,
    task_id: int,
    planner_checkpoint: int,
    reward_checkpoint: int,
    critic_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    reward_hidden_layers: int,
    reward_hidden_dim: int,
    backbone_layers: int,
    trajs: List[dict],
    gamma: float = 0.99,
    drop_timeouts: bool = True,
    value_decode: str = "symlog",
    accelerator: Accelerator = None,
):
    from Finetuning.utils import (
          check_device,
          get_critic_model,
          get_critic_stats,
          get_Q_scale,
          symexp,
    )
    def align_reward_mask(traj: dict):
        obs = np.asarray(traj["observations"], dtype=np.float32)
        n = len(obs)
        raw_m = traj.get("masks", None)
        if raw_m is None:
            masks = np.ones(n, dtype=np.float32)
        else:
            masks = np.asarray(raw_m, dtype=np.float32).reshape(-1)
            if len(masks) < n:
                masks = np.concatenate(
                    [masks, np.ones(n - len(masks), dtype=np.float32)]
                )
            masks = masks[:n]
        r = np.asarray(traj["rewards"], dtype=np.float64).reshape(-1)
        if len(r) == n - 1:
            trans_r = r
        elif len(r) == n:
            trans_r = r[1:]
        elif len(r) > n - 1:
            trans_r = r[: n - 1]
        else:
            trans_r = np.concatenate(
                [r, np.zeros(n - 1 - len(r), dtype=np.float64)]
            )
        return obs, masks, trans_r

    def traj_cost_to_go(traj: dict, gamma: float, drop_timeouts: bool = True):
        obs, masks, trans_r = align_reward_mask(traj)
        n = len(obs)
        if n == 0:
            return None, None
        goal = np.where(masks == 0.0)[0]
        G = np.zeros(n, dtype=np.float64)
        if len(goal) == 0:
            if drop_timeouts:
                return None, None
            acc = 0.0
            for t in range(n - 2, -1, -1):
                acc = float(trans_r[t]) + gamma * acc
                G[t] = acc
            G[n - 1] = 0.0
            return obs, G
        T = int(goal[0])
        G[T:] = 0.0
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc = float(trans_r[t]) + gamma * acc
            G[t] = acc
        return obs, G

    class CostToGoDataset(Dataset):
        def __init__(self, trajs, stats, gamma=0.99, drop_timeouts=True):
            xs, gs = [], []
            n_traj, n_drop = 0, 0
            for traj in trajs:
                obs, G = traj_cost_to_go(traj, gamma, drop_timeouts)
                if G is None:
                    n_drop += 1
                    continue
                n_traj += 1
                for t in range(len(obs)):
                    xs.append(stats.norm_obs(obs[t]))
                    gs.append(G[t])
            self.x = np.asarray(xs, dtype=np.float32)
            self.g = np.asarray(gs, dtype=np.float32)
            gmin = float(self.g.min()) if len(self.g) else float("nan")
            gmax = float(self.g.max()) if len(self.g) else float("nan")
            print(
                f"cost-to-go dataset: {len(self.x)} states  "
                f"(trajs kept={n_traj} dropped={n_drop})  "
                f"G min/max={gmin:.3f}/{gmax:.3f}"
            )

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return torch.from_numpy(self.x[i]), torch.tensor(self.g[i])

    def decode_v(v, value_decode, q_mean, q_std):
        if value_decode == "symlog":
            return symexp(v)
        if value_decode == "zscore":
            return v * q_std + q_mean
        if value_decode == "raw":
            return v
        raise ValueError(f"unknown value_decode={value_decode}")

    def sample_euler_karras_batch(
        s0_b: torch.Tensor,
        score_model,
        d_s: int,
        d_a: int,
        horizon: int,
        num_steps: int = 10,
        num_karras: int = 1,
        eta: float = 0.0,
    ):
       
        import math
        try:
            from Pretrain.Planners.Backbone.Sampler import (
                karras_beta_schedule, clip_actions,
            )
            from Pretrain.Planners.Backbone.utils import cosine_beta
        except Exception:
            from Sampler import karras_beta_schedule, clip_actions
            from utils import cosine_beta

        B = s0_b.shape[0]
        device = s0_b.device
        dim = d_s + d_a
        t_grid, beta_1, sigma_grid = karras_beta_schedule(num_steps, device=device)
        beta_2 = cosine_beta(t_grid, s=0.008)
        x = torch.randn(B, horizon, dim, device=device) * sigma_grid[0]
        mask = torch.zeros(B, horizon, dim, device=device)
        mask[:, 0, :d_s] = 1.0
        y = torch.zeros_like(x)
        y[:, 0, :d_s] = s0_b
        x = mask * y + (1.0 - mask) * x
        for i in range(num_steps):
            t_now = t_grid[i]
            t_next = t_grid[i + 1] if i < num_steps - 1 else torch.zeros((), device=device)
            dt = (t_next - t_now).item()
            beta_now = (beta_1[i] if i < num_karras else beta_2[i]).item()
            drift = -0.5 * beta_now * x
            t_in = t_now.repeat(B)
            score = score_model(x, t_in)
            if eta > 0:
                noise = torch.randn_like(x)
                x = x + (drift - beta_now * score) * dt + eta * math.sqrt(beta_now * (-dt)) * noise
            else:
                x = x + (drift - beta_now * score) * dt
            x = mask * y + (1.0 - mask) * x
            x = clip_actions(x, d_s)
        return x

    def compute_j_by_state(
        dataset_name: str,
        specific_dataset: str,
        task_id: int,
        planner_checkpoint: int,
        reward_checkpoint: int,
        critic_checkpoint: int,
        backbone_layers: int,
        hidden_layers: int,
        hidden_dim: int,
        reward_hidden_layers: int,
        reward_hidden_dim: int,
        trajs: List[dict],
        horizon: int = 32,
        gamma: float = 0.99,
        n_s0: int = 64,
        n_plans: int = 8,
        steps_T: int = 10,
        num_karras: int = 1,
        eta: float = 0.0,
        suffix_length: Optional[int] = None,
        seed: int = 0,
        value_decode: str = "symlog",
        type_a_only: bool = True,
        accelerator=None,
        plan_batch: int = 64,
    ):
        
        from accelerate import Accelerator
        from Pretrain.Dataset import get_env, Planner_Processor
        from Pretrain.Planners.Backbone.Dit import DiT1d
        from Pretrain.Rewards.nets import SimpleReward
        from Finetuning.utils import (
            get_planner, get_reward_model, get_reward_stats,
        )

        if accelerator is None:
            accelerator = Accelerator()
        device = accelerator.device
        rng = np.random.RandomState(seed + accelerator.process_index)
        _, obs_dim, act_dim = get_env(dataset_name, specific_dataset, task_id=task_id)

        s0_raw = []
        for traj in trajs:
            obs, masks, _ = align_reward_mask(traj)
            if type_a_only and (len(masks) == 0 or masks.min() != 0.0):
                continue
            play = np.where(masks != 0.0)[0]
            if suffix_length is not None:
                play = play[-suffix_length:]
            for t in play:
                s0_raw.append(obs[t])
        if not s0_raw:
            raise RuntimeError("no s0 for J_by_state")
        idx = rng.choice(len(s0_raw), size=min(n_s0, len(s0_raw)), replace=False)
        s0_raw = np.stack([s0_raw[i] for i in idx], axis=0)

        planner = DiT1d(
            in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
            n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
        )
        planner.load_state_dict(
            get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
        )
        planner.eval()
        proc = Planner_Processor(dataset_name, specific_dataset, task_id)

        rh, rhd = reward_hidden_layers or hidden_layers, reward_hidden_dim or hidden_dim
        rew_state, _, _ = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id,
        )
        reward_net = SimpleReward(obs_dim, act_dim, rhd, rh)
        reward_net.load_state_dict(rew_state)
        reward_net.eval()
        rstat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)

        ns = 0 if critic_checkpoint == -1 else critic_checkpoint
        stats_c = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
        c_state, _ = get_critic_model(
            dataset_name, specific_dataset, task_id, critic_checkpoint,
        )
        critic = Critic(obs_dim, hidden_dim, hidden_layers)
        critic.load_state_dict(c_state)
        critic.eval()

        planner, reward_net, critic = accelerator.prepare(planner, reward_net, critic)
        p_mean = torch.as_tensor(proc.stats.obs_mean, device=device, dtype=torch.float32)
        p_std = torch.as_tensor(np.maximum(proc.stats.obs_std, 1e-3), device=device)
        r_mean = torch.as_tensor(rstat.obs_mean, device=device, dtype=torch.float32)
        r_std = torch.as_tensor(np.maximum(rstat.obs_std, 1e-3), device=device)
        c_mean = torch.as_tensor(stats_c.obs_mean, device=device, dtype=torch.float32)
        c_std = torch.as_tensor(np.maximum(stats_c.obs_std, 1e-3), device=device)

        q_mean_t = torch.tensor(0.0, device=device)
        q_std_t = torch.tensor(1.0, device=device)
        scale = 1.0
        try:
            qs = get_Q_scale(dataset_name, specific_dataset, task_id)
            scale = float(getattr(qs, "Q_scale", 1.0) or 1.0)
            q_mean_t = torch.tensor(float(getattr(qs, "Q_mean", 0.0) or 0.0), device=device)
            q_std_t = torch.tensor(float(getattr(qs, "Q_std", 1.0) or 1.0), device=device)
        except Exception:
            pass

        n_s, K = len(s0_raw), n_plans
        s0_norm = np.stack([proc.preprocess(s) for s in s0_raw], axis=0).astype(np.float32)

        world = accelerator.num_processes
        rank = accelerator.process_index
        shard = np.arange(n_s)[rank::world]
        n_local = len(shard)
        J_local = np.zeros((n_local, K), dtype=np.float64)

        unwrap_p = accelerator.unwrap_model(planner)
        unwrap_r = accelerator.unwrap_model(reward_net)
        unwrap_c = accelerator.unwrap_model(critic)
        discounts = torch.tensor(
            [gamma ** t for t in range(horizon - 1)], device=device, dtype=torch.float32
        )

        pair_s = np.repeat(shard, K)
        pair_k = np.tile(np.arange(K), n_local)
        n_pairs = len(pair_s)
        for start in range(0, n_pairs, plan_batch):
            sl = slice(start, min(start + plan_batch, n_pairs))
            idx_s = pair_s[sl]
            idx_k = pair_k[sl]
            s0_b = torch.as_tensor(s0_norm[idx_s], device=device)
            xt = sample_euler_karras_batch(
                s0_b, unwrap_p, obs_dim, act_dim, horizon,
                num_steps=steps_T, num_karras=num_karras, eta=eta,
            )
            a = torch.clamp(xt[..., obs_dim:], -1.0, 1.0)
            s_raw = xt[..., :obs_dim] * p_std + p_mean
            nstep = s_raw.shape[1] - 1
            B = s_raw.shape[0]
            s_r = (s_raw[:, :nstep] - r_mean) / r_std
            r_hat = unwrap_r(
                s_r.reshape(B * nstep, -1), a[:, :nstep].reshape(B * nstep, -1),
            ).reshape(B, nstep)
            if value_decode == "symlog":
                r_hat = r_hat / max(scale, 1e-8)
            sH = (s_raw[:, -1] - c_mean) / c_std
            vH = decode_v(unwrap_c(sH).squeeze(-1), value_decode, q_mean_t, q_std_t)
            Jb = (r_hat * discounts[:nstep]).sum(dim=1) + (gamma ** nstep) * vH
            local_row = np.searchsorted(shard, idx_s)
            J_local[local_row, idx_k] = Jb.detach().float().cpu().numpy()

        max_local = int(accelerator.gather(
            torch.tensor([n_local], device=device, dtype=torch.long)
        ).max().item())
        J_pad = np.zeros((max_local, K), dtype=np.float64)
        J_pad[:n_local] = J_local
        mask_pad = np.zeros((max_local,), dtype=np.float32)
        mask_pad[:n_local] = 1.0
        J_g = accelerator.gather(torch.as_tensor(J_pad, device=device)).cpu().numpy()
        m_g = accelerator.gather(torch.as_tensor(mask_pad, device=device)).cpu().numpy()
        J_by_state = J_g.reshape(world, max_local, K)[m_g.reshape(world, max_local) > 0.5]
        if J_by_state.shape[0] > n_s:
            J_by_state = J_by_state[:n_s]
        if accelerator.is_main_process:
            print(
                f"J_by_state shape={J_by_state.shape}  "
                f"J mean/std={J_by_state.mean():.3f}/{J_by_state.std():.3f}"
            )
        return J_by_state

    def test_wsjd(
        dataset_name: str,
        specific_dataset: str,
        task_id: int,
        planner_checkpoint: int,
        reward_checkpoint: int,
        critic_checkpoint: int,
        backbone_layers: int,
        hidden_layers: int,
        hidden_dim: int,
        reward_hidden_layers: int,
        reward_hidden_dim: int,
        trajs: List[dict],
        accelerator: Accelerator,
    ):
        J = compute_j_by_state(
            dataset_name, specific_dataset, task_id, planner_checkpoint, reward_checkpoint, critic_checkpoint,
            backbone_layers, hidden_layers, hidden_dim, reward_hidden_layers, reward_hidden_dim, trajs,
            accelerator = accelerator,
        )
        stats = within_state_j_dispersion(J)
        stats["J_by_state"] = J
        return stats

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    stats = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    data = CostToGoDataset(trajs, stats, gamma, drop_timeouts)
    if len(data) == 0:
        raise RuntimeError("cost-to-go dataset empty — check masks/rewards on trajs")
    loader = DataLoader(data, batch_size=512, shuffle=False)

    state, obs_dim = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(state)
    model.eval()

    q_mean = q_std = 0.0
    if value_decode == "zscore":
        try:
            qs = get_Q_scale(dataset_name, specific_dataset, task_id)
            q_mean = float(getattr(qs, "Q_mean", 0.0) or 0.0)
            q_std = float(getattr(qs, "Q_std", 1.0) or 1.0)
        except Exception:
            q_mean, q_std = 0.0, 1.0
        q_mean = torch.tensor(q_mean, device=device)
        q_std = torch.tensor(q_std, device=device)

    preds, targets = [], []
    for s, g in loader:
        s = s.to(device)
        v = decode_v(model(s).squeeze(-1), value_decode, q_mean, q_std)
        preds.append(v.detach().cpu().numpy())
        targets.append(g.numpy())
    pred = np.concatenate(preds)
    tgt = np.concatenate(targets)
    ic = spearman_correlation(pred, tgt)
    ev = explained_variance(tgt, pred)
    WSJD = test_wsjd(
         dataset_name, specific_dataset, task_id, planner_checkpoint, reward_checkpoint, critic_checkpoint,
         backbone_layers, hidden_layers, hidden_dim, reward_hidden_layers, reward_hidden_dim,
         trajs, accelerator)

    mae = float(np.mean(np.abs(pred - tgt)))
    print(
        f"  cost-to-go test ckpt={critic_checkpoint} decode={value_decode}\n"
        f"  n={len(pred)}  IC={ic:.3f}  EV={ev:.3f}  MAE={mae:.3f}\n"
        f"  WSJD = {WSJD['wsjd']:.3f}\n"
        f"  pred mean/std={pred.mean():.3f}/{pred.std():.3f}\n"
        f"  G    mean/std={tgt.mean():.3f}/{tgt.std():.3f}\n"
        f"  G    min/max={tgt.min():.3f}/{tgt.max():.3f}"
    )
    return {"ic": ic, "ev": ev, "WSJD": WSJD['wsjd'], "mae": mae, "pred": pred, "G": tgt}










