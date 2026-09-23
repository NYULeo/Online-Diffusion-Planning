
from typing import List, Optional
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import Dataset, DataLoader
from Pretrain.Critic.nets import Critic
import math


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

def _as_2d(R_s) -> np.ndarray:
    R = np.asarray(R_s, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("R_s must be (n_s0, n_horizons) with n_horizons>=2")
    return R

def mean_of_RK(R_s) -> float:
    """E_s[ mean_K R^{(K)}(s) ]."""
    R = _as_2d(R_s)
    return float(R.mean(axis=1).mean())

def mean_of_STD(R_s) -> float:
    """E_s[ std_K R^{(K)}(s) ]  (population std over horizons)."""
    R = _as_2d(R_s)
    return float(R.std(axis=1, ddof=0).mean())

def ratio_Rn_over_R1(R_s, eps: float = 1e-8, reduce: str = "mean") -> float:
    """E_s[ R^{(n)}(s) / R^{(1)}(s) ] on states with |R1|>eps.

    Sign-preserving divide: denom = sign(R1) * max(|R1|, eps) in the probe;
    here we drop |R1|<=eps instead (same as their ratio_ok mask).
    reduce: 'mean' | 'median'
    """
    R = _as_2d(R_s)
    R1, Rn = R[:, 0], R[:, -1]
    ok = np.abs(R1) > eps
    if not np.any(ok):
        return float("nan")
    rat = Rn[ok] / R1[ok]
    if reduce == "median":
        return float(np.median(rat))
    return float(rat.mean())

def E_RNm1_div_E_R1(R_s, eps: float = 1e-8) -> float:
    """(E_s R^{(n)}) / (E_s R^{(1)}). Ratio of means, not mean of ratios."""
    R = _as_2d(R_s)
    R1 = float(R[:, 0].mean())
    Rn = float(R[:, -1].mean())
    if abs(R1) < eps:
        return float("nan")
    return Rn / R1

def bellman_cut_stats(R_s, eps: float = 1e-8) -> dict:
    """All four + SEs, matching Raw.probe_multi_horizon_bellman print keys."""
    R = _as_2d(R_s)
    M = R.shape[0]
    m_s = R.mean(axis=1)
    std_s = R.std(axis=1, ddof=0)
    R1, Rn = R[:, 0], R[:, -1]
    ok = np.abs(R1) > eps
    rat = Rn[ok] / R1[ok] if np.any(ok) else np.array([])

    def se(x):
        x = np.asarray(x, dtype=np.float64)
        if x.size < 2:
            return float("nan")
        return float(x.std(ddof=1) / math.sqrt(x.size))

    stats = {
        "n_s0": int(M),
        "n_horizons": int(R.shape[1]),
        "mean_of_RK": mean_of_RK(R),
        "mean_of_STD": mean_of_STD(R),
        "ratio": ratio_Rn_over_R1(R, eps, "mean"),
        "median_ratio": ratio_Rn_over_R1(R, eps, "median"),
        "E_RNm1_div_E_R1": E_RNm1_div_E_R1(R, eps),
        "se_mean_of_RK": se(m_s),
        "se_mean_of_STD": se(std_s),
        "se_ratio": se(rat) if rat.size else float("nan"),
        "n_ratio": int(ok.sum()),
    }
    print(
        f"bellman cuts  M={M} K={R.shape[1]}\n"
        f"  mean_of_RK        = {stats['mean_of_RK']:.4f}  se={stats['se_mean_of_RK']:.4f}\n"
        f"  mean_of_STD       = {stats['mean_of_STD']:.4f}  se={stats['se_mean_of_STD']:.4f}\n"
        f"  ratio R^n/R^1     = {stats['ratio']:.4f}  se={stats['se_ratio']:.4f}\n"
        f"  E[R^n]/E[R^1]     = {stats['E_RNm1_div_E_R1']:.4f}"
    )
    return stats

def R_s_from_cuts(R_tau: torch.Tensor) -> np.ndarray:
    """R_tau: (n_s0, n_plans, n_horizons) -> (n_s0, n_horizons) = E_tau first."""
    if R_tau.dim() != 3:
        raise ValueError("R_tau must be (M, L, K)")
    return R_tau.mean(dim=1).detach().cpu().numpy()




def decode_v(v, value_decode, q_mean, q_std):
        from Finetuning.utils import symexp
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
            get_planner, get_reward_model, get_reward_stats, get_critic_model, get_critic_stats, get_Q_scale,
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

def two_bellman_metrics(R_s: np.ndarray, eps: float = 1e-8) -> dict:
    R = np.asarray(R_s, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("R_s must be (n_s0, n_horizons) with n_horizons>=2")
    R1, RN = R[:, 0], R[:, -1]
    ok = np.abs(R1) > eps
    rho = float((RN[ok] / R1[ok]).mean()) if np.any(ok) else float("nan")
    sig2 = float(np.var(R, axis=1).mean())
    return {
        "expected_ratio_EN_over_E1": rho,
        "expected_var_k": sig2,
    }



@torch.no_grad()
def evaluate_critic(
    dataset_name: str,
    specific_dataset: str,
    task_id: int,
    critic_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    trajs: List[dict],
    J_by_state: np.ndarray,
    gamma: float = 0.99,
    drop_timeouts: bool = True,
    value_decode: str = "symlog",
    reward_scale: float = 500.0,
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
        def __init__(self, trajs, stats, gamma=0.99, drop_timeouts=True, reward_scale=500.0):
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
            #self.g = np.asarray(gs, dtype=np.float32)
            self.g = (np.asarray(gs, dtype=np.float64) * float(reward_scale)).astype(np.float32)
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

    def test_wsjd( 
        J_by_state: np.ndarray,
    ):
        
        stats = within_state_j_dispersion(J_by_state)
        stats["J_by_state"] = J_by_state
        return stats

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    stats = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    #data = CostToGoDataset(trajs, stats, gamma, drop_timeouts)
    data = CostToGoDataset(
        trajs, stats, gamma, drop_timeouts, reward_scale=reward_scale,
    )
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
    WSJD = test_wsjd(J_by_state)
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

@torch.no_grad()
def evaluate_critic_hat_return(
    dataset_name: str,
    specific_dataset: str,
    task_id: int,
    critic_checkpoint: int,
    reward_checkpoint: int,
    hidden_layers: int,
    hidden_dim: int,
    reward_hidden_layers: int,
    reward_hidden_dim: int,
    trajs: List[dict],
    J_by_state: np.ndarray,
    gamma: float = 0.99,
    drop_timeouts: bool = True,
    value_decode: str = "symlog",
    batch_size: int = 4096,
    max_trajs: Optional[int] = None,
):
    from Pretrain.Rewards.nets import SimpleReward
    from Pretrain.Critic.nets import Critic
    from Finetuning.utils import (
        check_device, get_critic_model, get_critic_stats, get_Q_scale,
        get_reward_model, get_reward_stats,
    )
    def test_wsjd( 
        J_by_state: np.ndarray,
    ):
        stats = within_state_j_dispersion(J_by_state)
        stats["J_by_state"] = J_by_state
        return stats

    

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    cstat = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    c_state, obs_dim = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic.load_state_dict(c_state)
    critic.eval()

    rew_state, extra, extra2 = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    # get_reward_model -> (state, obs_dim, act_dim) on main
    act_dim = extra2 if isinstance(extra2, int) else extra
    if not isinstance(act_dim, int):
        act_dim = int(np.asarray(trajs[0]["actions"]).shape[-1])

    reward_net = SimpleReward(
        obs_dim, act_dim, reward_hidden_dim, reward_hidden_layers,
    ).to(device)
    reward_net.load_state_dict(rew_state)
    reward_net.eval()
    rstat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)

    scale = 1.0
    q_mean = torch.tensor(0.0, device=device)
    q_std = torch.tensor(1.0, device=device)
    try:
        qs = get_Q_scale(dataset_name, specific_dataset, task_id)
        scale = float(getattr(qs, "Q_scale", 1.0) or 1.0)
        q_mean = torch.tensor(float(getattr(qs, "Q_mean", 0.0) or 0.0), device=device)
        q_std = torch.tensor(float(getattr(qs, "Q_std", 1.0) or 1.0), device=device)
    except Exception:
        pass

    r_mean = torch.as_tensor(rstat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(np.maximum(rstat.obs_std, 1e-3), device=device)

    # ---- gather Type A (or all) tapes ----
    segs = []  # list of (obs[0:T+1], act[0:T])
    n_drop = 0
    used = 0
    for traj in trajs:
        obs, masks, _ = align_reward_mask(traj)
        n = len(obs)
        if n < 2:
            continue
        acts = np.asarray(traj.get("actions", np.zeros((0, act_dim))), dtype=np.float32)
        if acts.ndim == 1:
            acts = acts.reshape(-1, 1)
        if len(acts) >= n:
            acts = acts[: n - 1]
        elif len(acts) < n - 1:
            pad = np.zeros((n - 1 - len(acts), acts.shape[-1]), dtype=np.float32)
            acts = np.concatenate([acts, pad], axis=0)

        goal = np.where(masks == 0.0)[0]
        if len(goal) == 0:
            if drop_timeouts:
                n_drop += 1
                continue
            T = n - 1
        else:
            T = int(goal[0])
        if T < 1:
            continue
        segs.append((obs[: T + 1], acts[:T]))
        used += 1
        if max_trajs is not None and used >= max_trajs:
            break

    if not segs:
        raise RuntimeError("hat-return: no tapes")

    all_s, all_a = [], []
    for obs, acts in segs:
        all_s.append(obs[:-1])
        all_a.append(acts)
    S = np.concatenate(all_s, axis=0).astype(np.float32)
    A = np.concatenate(all_a, axis=0).astype(np.float32)
    A = np.clip(A, -1.0, 1.0)

    # ---- one batched reward_net ----
    r_hat = np.empty((len(S),), dtype=np.float64)
    for i in range(0, len(S), batch_size):
        sl = slice(i, min(i + batch_size, len(S)))
        s = torch.as_tensor(S[sl], device=device)
        a = torch.as_tensor(A[sl], device=device)
        sn = (s - r_mean) / r_std
        r = reward_net(sn, a) / max(scale, 1e-8)
        r_hat[sl] = r.detach().float().cpu().numpy().reshape(-1)

    # ---- NumPy backup per tape ----
    xs, gs = [], []
    off = 0
    for obs, acts in segs:
        T = len(acts)
        rr = r_hat[off : off + T]
        off += T
        G = np.zeros(T + 1, dtype=np.float64)  # G[T]=0 absorb
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc = float(rr[t]) + gamma * acc
            G[t] = acc
        for t in range(T + 1):
            xs.append(cstat.norm_obs(obs[t]))
            gs.append(G[t])

    X = np.asarray(xs, dtype=np.float32)
    Gv = np.asarray(gs, dtype=np.float32)
    print(
        f"hat-return G: states={len(X)} trajs={len(segs)} dropped={n_drop} "
        f"G min/max={Gv.min():.3f}/{Gv.max():.3f} mean/std={Gv.mean():.3f}/{Gv.std():.3f}"
    )

    # ---- one batched critic ----
    pred = np.empty((len(X),), dtype=np.float64)
    for i in range(0, len(X), batch_size):
        sl = slice(i, min(i + batch_size, len(X)))
        s = torch.as_tensor(X[sl], device=device)
        v = decode_v(critic(s).squeeze(-1), value_decode, q_mean, q_std)
        pred[sl] = v.detach().float().cpu().numpy().reshape(-1)
    

    R1_list, Rn_list = [], []
    off_r = off_v = 0
    for obs, acts in segs:
        T = len(acts)
        rr = r_hat[off_r : off_r + T]
        vv = pred[off_v : off_v + T + 1].copy()
        gg = Gv[off_v : off_v + T + 1]
        vv[-1] = 0.0
        off_r += T
        off_v += T + 1
        if T < 1:
            continue
        R1_list.append(rr + gamma * vv[1:])
        Rn_list.append(np.asarray(gg[:T], dtype=np.float64))
    R_s = np.stack(
        [np.concatenate(R1_list), np.concatenate(Rn_list)], axis=1
    )
    #cuts = bellman_cut_stats(R_s)
    bellman_metrics =  two_bellman_metrics(R_s)
    ic = float(spearmanr(pred, Gv).correlation)
    var_g = float(np.var(Gv))
    ev = float("nan") if var_g < 1e-12 else float(1.0 - np.var(Gv - pred) / var_g)
    mae = float(np.mean(np.abs(pred - Gv)))
    WSJD = test_wsjd(J_by_state)
    print(
        f"hat-return test ckpt={critic_checkpoint}\n"
        f"  n={len(pred)}  IC={ic:.3f}  EV={ev:.3f}  MAE={mae:.3f}\n"
        f"  pred mean/std={pred.mean():.3f}/{pred.std():.3f}\n"
        f"  Ghat mean/std={Gv.mean():.3f}/{Gv.std():.3f}\n"
        f"  WSJD = {WSJD['wsjd']:.3f}\n"
        f"  E_s[ E R^N / E R^1] = {bellman_metrics['expected_ratio_EN_over_E1']:.4f}\n"\
        f"  E_s[Var_k E R^k] = {bellman_metrics["expected_var_k"]:.4f}\n"
        #f"  mean_of_RK={cuts['mean_of_RK']:.4f}  mean_of_STD={cuts['mean_of_STD']:.4f}\n"
        #f"  ratio={cuts['ratio']:.4f}  E[Rn]/E[R1]={cuts['E_RNm1_div_E_R1']:.4f}"
    )
    return {
        "ic": ic, "ev": ev, "mae": mae,
        "pred": pred, "G": Gv,
        "WSJD": WSJD["wsjd"],
        #"mean_of_RK": cuts["mean_of_RK"],
        #"mean_of_STD": cuts["mean_of_STD"],
        #"ratio": cuts["ratio"],
        #"E_RNm1_div_E_R1": cuts["E_RNm1_div_E_R1"],
    }

@torch.no_grad()
def td_residual_stats(
    dataset_name: str,
    specific_dataset: str,
    task_id: int,
    hidden_layers: int,
    hidden_dim: int,
    critic_checkpoint: int,
    trajs: List[dict],
    gamma: float = 0.99,
    value_decode: str = "symlog",
    suffix_length: Optional[int] = None,
):
    from Finetuning.utils import (
          check_device,
          get_critic_model,
          get_critic_stats,
          get_Q_scale,
          symexp,
    )

    def decode_v(v, value_decode, q_mean, q_std):
        if value_decode == "symlog":
            return symexp(v)
        if value_decode == "zscore":
            return v * q_std + q_mean
        if value_decode == "raw":
            return v
        raise ValueError(f"unknown value_decode={value_decode}")

    def _V(obs_np):
        x = torch.as_tensor(
            np.stack([stats.norm_obs(o) for o in obs_np], axis=0),
            device=device, dtype=torch.float32,
        )
        v = model(x).squeeze(-1)
        return decode_v(v, value_decode, q_mean, q_std).detach().cpu().numpy()
    
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

    def _summ(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return dict(n=0, mean=float("nan"), std=float("nan"), mse=float("nan"))
        return dict(
            n=int(a.size),
            mean=float(a.mean()),
            std=float(a.std()),
            mse=float(np.mean(a ** 2)),
        )

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    stats = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    state, obs_dim = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(state)
    model.eval()

    q_mean = torch.tensor(0.0, device=device)
    q_std = torch.tensor(1.0, device=device)
    if value_decode == "zscore":
        try:
            qs = get_Q_scale(dataset_name, specific_dataset, task_id)
            q_mean = torch.tensor(float(getattr(qs, "Q_mean", 0.0) or 0.0), device=device)
            q_std = torch.tensor(float(getattr(qs, "Q_std", 1.0) or 1.0), device=device)
        except Exception:
            pass

    
    d_all, d_near, d_far, d_goal = [], [], [], []
    for traj in trajs:
        obs, masks, trans_r = align_reward_mask(traj)
        n = len(obs)
        if n < 2:
            continue
        V = _V(obs)
        goal = np.where(masks == 0.0)[0]
        T = int(goal[0]) if len(goal) else None
        play = np.where(masks != 0.0)[0]
        near_set = set(play[-suffix_length:]) if suffix_length is not None else set(play)

        for t in range(n - 1):
            if T is not None and t >= T:
                break
            if T is not None and t + 1 == T:
                delta = float(trans_r[t] + 0.0 - V[t])
                d_goal.append(delta)
            else:
                delta = float(trans_r[t] + gamma * V[t + 1] - V[t])
            d_all.append(delta)
            if t in near_set:
                d_near.append(delta)
            elif T is None or t < T:
                d_far.append(delta)

    out = {
        "all": _summ(d_all),
        "near": _summ(d_near),
        "far": _summ(d_far),
        "goal_arrive": _summ(d_goal),
    }
    print(
        "TD residual  delta = r + gamma V(s') - V(s)\n"
        f"  all  n={out['all']['n']}  mean={out['all']['mean']:.4f}  "
        f"std={out['all']['std']:.4f}  mse={out['all']['mse']:.4f}\n"
        f"  near n={out['near']['n']}  mean={out['near']['mean']:.4f}  "
        f"std={out['near']['std']:.4f}\n"
        f"  far  n={out['far']['n']}  mean={out['far']['mean']:.4f}  "
        f"std={out['far']['std']:.4f}\n"
        f"  arrive-goal n={out['goal_arrive']['n']}  "
        f"mean={out['goal_arrive']['mean']:.4f}  std={out['goal_arrive']['std']:.4f}"
    )
    return out

def value_grad_stats(
    dataset_name: str,
    specific_dataset: str,
    task_id: int,
    hidden_layers: int,
    hidden_dim: int,
    critic_checkpoint: int,
    trajs: List[dict],
    value_decode: str = "symlog",
    suffix_length: Optional[int] = None,
    batch_size: int = 256,
    max_states: int = 8192,
    seed: int = 0,
):
    from Finetuning.utils import (
          check_device,
          get_critic_model,
          get_critic_stats,
          get_Q_scale,
          symexp,
    )
    def _v_and_grad(model, x, value_decode, q_mean, q_std):
        x = x.detach().requires_grad_(True)
        raw = model(x).squeeze(-1)
        if value_decode == "symlog":
              V = symexp(raw)
        elif value_decode == "zscore":
              V = raw * q_std + q_mean
        else:
              V = raw
        g = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=False)[0]
        gn = g.flatten(1).norm(dim=1)
        return V.detach(), gn.detach(), g.detach()
    
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
    
    def _summ(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return dict(n=0, mean=float("nan"), std=float("nan"), p90=float("nan"))
        return dict(
            n=int(a.size),
            mean=float(a.mean()),
            std=float(a.std()),
            p90=float(np.quantile(a, 0.90)),
        )

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    stats = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    state, obs_dim = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(state)
    model.eval()

    q_mean = torch.tensor(0.0, device=device)
    q_std = torch.tensor(1.0, device=device)
    if value_decode == "zscore":
        try:
            qs = get_Q_scale(dataset_name, specific_dataset, task_id)
            q_mean = torch.tensor(float(getattr(qs, "Q_mean", 0.0) or 0.0), device=device)
            q_std = torch.tensor(float(getattr(qs, "Q_std", 1.0) or 1.0), device=device)
        except Exception:
            pass

    buckets = {k: [] for k in ("all", "near", "far", "goal")}
    cos_near = []
    rng = np.random.RandomState(seed)
    xs_all, tag, xg_ref = [], [], []
    for traj in trajs:
        obs, masks, _ = align_reward_mask(traj)
        if len(obs) == 0:
            continue
        play = np.where(masks != 0.0)[0]
        goals = np.where(masks == 0.0)[0]
        near = set(play[-suffix_length:]) if suffix_length is not None else set(play)
        gvec = stats.norm_obs(obs[int(goals[0])]) if len(goals) else None
        for t in range(len(obs)):
            xs_all.append(stats.norm_obs(obs[t]))
            if masks[t] == 0.0:
                tag.append("goal")
            elif t in near:
                tag.append("near")
            else:
                tag.append("far")
            xg_ref.append(gvec)
    if not xs_all:
        raise RuntimeError("no states for value_grad_stats")
    if len(xs_all) > max_states:
        pick = rng.choice(len(xs_all), size=max_states, replace=False)
        xs_all = [xs_all[i] for i in pick]
        tag = [tag[i] for i in pick]
        xg_ref = [xg_ref[i] for i in pick]

    X = torch.as_tensor(np.stack(xs_all, axis=0), device=device, dtype=torch.float32)
    for start in range(0, len(X), batch_size):
        sl = slice(start, min(start + batch_size, len(X)))
        xb = X[sl]
        V, gn, g = _v_and_grad(model, xb, value_decode, q_mean, q_std)
        gn_np = gn.cpu().numpy()
        g_np = g.cpu().numpy()
        for i, lab in enumerate(tag[sl]):
            buckets["all"].append(gn_np[i])
            buckets[lab].append(gn_np[i])
            ref = xg_ref[start + i]
            if lab == "near" and ref is not None:
                d = ref - xb[i].detach().cpu().numpy()
                dn = float(np.linalg.norm(d) * np.linalg.norm(g_np[i]) + 1e-8)
                cos_near.append(float(np.dot(g_np[i], d) / dn))

    out = {k: _summ(v) for k, v in buckets.items()}
    out["cos_to_goal_near"] = float(np.mean(cos_near)) if cos_near else float("nan")
    print(
        "value grad  g=||dV/dx||  x=norm_obs(s)\n"
        f"  all  n={out['all']['n']}  mean={out['all']['mean']:.4f}  "
        f"std={out['all']['std']:.4f}  p90={out['all']['p90']:.4f}\n"
        f"  near mean={out['near']['mean']:.4f}  std={out['near']['std']:.4f}  "
        f"p90={out['near']['p90']:.4f}\n"
        f"  far  mean={out['far']['mean']:.4f}  std={out['far']['std']:.4f}\n"
        f"  goal mean={out['goal']['mean']:.4f}\n"
        f"  mean cos(dV, x_g-x) on near={out['cos_to_goal_near']:.3f}"
    )
    return out



