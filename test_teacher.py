"""
Teacher tests for Online-Diffusion-Planning.

Paste these into Finetuning/utils.py (or import from this file).

1) test_critic_cost_to_go
   G(s_t) = discounted dataset return to the FIRST goal (masks==0), V(g)=0.
   Type B (no goal) can be dropped or truncated.
   Reports Spearman IC and explained variance EV = 1 - Var(G-V)/Var(G).

2) test_teacher_ic
   Frozen planner plans from s0, J = sum hat r + gamma^{H-1} V(s_H).
   G = env success or -T if roll_env=True; else dataset G(s0) if s0 came from Type A.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def explained_variance(g: np.ndarray, v: np.ndarray) -> float:
    """EV = 1 - Var(G-V) / Var(G). nan if G has no variance."""
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    var_g = float(np.var(g))
    if var_g < 1e-12:
        return float("nan")
    return float(1.0 - np.var(g - v) / var_g)


def _traj_cost_to_go(
    rews: np.ndarray,
    masks: np.ndarray,
    gamma: float,
    drop_timeouts: bool = True,
) -> Optional[np.ndarray]:
    """G[t] = sum_{k=0}^{T-t-1} gamma^k r[t+k], T = first masks==0.
    After the goal, G=0. Timeouts (no goal) return None if drop_timeouts."""
    n = len(rews)
    if n == 0:
        return None
    goal = np.where(np.asarray(masks[:n]) == 0.0)[0]
    if len(goal) == 0:
        if drop_timeouts:
            return None
        T = n
        g = np.zeros(n, dtype=np.float64)
        acc = 0.0
        for t in range(n - 1, -1, -1):
            acc = float(rews[t]) + gamma * acc
            g[t] = acc
        return g
    T = int(goal[0])
    g = np.zeros(n, dtype=np.float64)
    acc = 0.0
    # absorb: G[T] = 0 (reward at goal already in r[T-1] if you shifted)
    for t in range(T - 1, -1, -1):
        acc = float(rews[t]) + gamma * acc
        g[t] = acc
    return g


class CostToGoDataset(Dataset):
    def __init__(
        self,
        trajs: List[dict],
        stats,
        gamma: float = 0.99,
        drop_timeouts: bool = True,
    ):
        xs, gs = [], []
        for traj in trajs:
            obs = np.asarray(traj["observations"], dtype=np.float32)
            rews = np.asarray(traj["rewards"], dtype=np.float64)
            masks = np.asarray(traj.get("masks", np.ones(len(obs))), dtype=np.float32)
            n = min(len(obs), len(rews) + 1, len(masks))
            obs, masks = obs[:n], masks[:n]
            # rewards are next-state / transition length n-1 in your Dataset.py
            if len(rews) == n:
                r = rews
            elif len(rews) == n - 1:
                r = np.concatenate([rews, [0.0]])
            else:
                r = rews[:n]
            G = _traj_cost_to_go(r, masks, gamma, drop_timeouts=drop_timeouts)
            if G is None:
                continue
            for t in range(n):
                xs.append(stats.norm_obs(obs[t]))
                gs.append(G[t])
        self.x = np.asarray(xs, dtype=np.float32)
        self.g = np.asarray(gs, dtype=np.float32)
        print(f"cost-to-go dataset: {len(self.x)} states")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.g[i])


@torch.no_grad()
def test_critic_cost_to_go(
    dataset_name: str,
    specific_dataset: str,
    hidden_layers: int,
    hidden_dim: int,
    critic_checkpoint: int,
    trajs: List[dict],
    gamma: float = 0.99,
    task_id: Optional[int] = None,
    drop_timeouts: bool = True,
):
    """V(s) vs dataset cost-to-go G(s). Returns Spearman IC and MAE in raw units."""
    from scipy.stats import spearmanr

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    stats = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    data = CostToGoDataset(trajs, stats, gamma, drop_timeouts)
    loader = DataLoader(data, batch_size=512, shuffle=False)

    state, obs_dim = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    model = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(state)
    model.eval()

    preds, targets = [], []
    for s, g in loader:
        s = s.to(device)
        v = symexp(model(s).squeeze(-1)).cpu().numpy()
        preds.append(v)
        targets.append(g.numpy())
    pred = np.concatenate(preds)
    tgt = np.concatenate(targets)
    ic = float(spearmanr(pred, tgt).correlation)
    ev = explained_variance(tgt, pred)
    mae = float(np.mean(np.abs(pred - tgt)))
    print(
        f"cost-to-go test ckpt={critic_checkpoint}\n"
        f"  n={len(pred)}  IC={ic:.3f}  EV={ev:.3f}  MAE={mae:.3f}\n"
        f"  pred mean/std={pred.mean():.3f}/{pred.std():.3f}\n"
        f"  G    mean/std={tgt.mean():.3f}/{tgt.std():.3f}\n"
        f"  G    min/max={tgt.min():.3f}/{tgt.max():.3f}"
    )
    return {"ic": ic, "ev": ev, "mae": mae, "pred": pred, "G": tgt}


@torch.no_grad()
def test_teacher_ic(
    dataset_name: str,
    specific_dataset: str,
    planner_checkpoint: int,
    reward_checkpoint: int,
    critic_checkpoint: int,
    backbone_layers: int,
    hidden_layers: int,
    hidden_dim: int,
    trajs: List[dict],
    task_id: Optional[int] = None,
    horizon: int = 32,
    gamma: float = 0.99,
    n_s0: int = 64,
    n_plans: int = 8,
    steps_T: int = 10,
    num_karras: int = 1,
    eta: float = 0.0,
    roll_env: bool = False,
    max_env_steps: Optional[int] = None,
    suffix_length: Optional[int] = 32,
    seed: int = 0,
):
    """
    For each s0 (last-suffix play frames by default):
      sample n_plans, compute J, and G.
    G:
      roll_env=False -> dataset cost-to-go at that s0 (Type A only).
      roll_env=True  -> execute planned actions in OGBench, G = success or -T.
    IC = mean over s0 of Spearman(J, G) if n_plans>=3 and G varies;
         else global Spearman over all (s0, plan) pairs.
    """
    from scipy.stats import spearmanr

    device = check_device()
    rng = np.random.RandomState(seed)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset, task_id=task_id)

    # ---- s0 from last-N play on Type A ----
    s0_raw, s0_G = [], []
    stats_c = get_critic_stats(
        dataset_name, specific_dataset, task_id,
        0 if critic_checkpoint == -1 else critic_checkpoint,
    )
    for traj in trajs:
        obs = np.asarray(traj["observations"], dtype=np.float32)
        rews = np.asarray(traj["rewards"], dtype=np.float64)
        masks = np.asarray(traj.get("masks", np.ones(len(obs))), dtype=np.float32)
        n = len(obs)
        if n == 0 or masks[-1] != 0.0:
            continue
        if len(rews) == n - 1:
            r = np.concatenate([rews, [0.0]])
        else:
            r = rews[:n]
        G = _traj_cost_to_go(r, masks, gamma, drop_timeouts=True)
        if G is None:
            continue
        play = np.where(masks != 0.0)[0]
        if suffix_length is not None:
            play = play[-suffix_length:]
        for t in play:
            s0_raw.append(obs[t])
            s0_G.append(G[t])
    if not s0_raw:
        raise RuntimeError("no Type A s0 for teacher IC")
    idx = rng.choice(len(s0_raw), size=min(n_s0, len(s0_raw)), replace=False)
    s0_raw = np.stack([s0_raw[i] for i in idx], axis=0)
    s0_G = np.asarray([s0_G[i] for i in idx], dtype=np.float64)

    # ---- models ----
    planner = DiT1d(
        in_dim=(obs_dim + act_dim), emb_dim=128, d_model=256,
        n_heads=256 // 64, depth=backbone_layers, timestep_emb_type="fourier",
    ).to(device)
    planner.load_state_dict(
        get_planner(dataset_name, specific_dataset, planner_checkpoint, task_id)
    )
    planner.eval()
    proc = Planner_Processor(dataset_name, specific_dataset, task_id)
    p_mean = torch.as_tensor(proc.stats.obs_mean, device=device, dtype=torch.float32)
    p_std = torch.as_tensor(np.maximum(proc.stats.obs_std, 1e-3), device=device)

    rew_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    # hidden sizes must match your reward ckpt
    reward_net = SimpleReward(obs_dim, act_dim, hidden_dim, hidden_layers).to(device)
    reward_net.load_state_dict(rew_state)
    reward_net.eval()
    rstat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
    r_mean = torch.as_tensor(rstat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(np.maximum(rstat.obs_std, 1e-3), device=device)

    c_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic.load_state_dict(c_state)
    critic.eval()
    c_mean = torch.as_tensor(stats_c.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(np.maximum(stats_c.obs_std, 1e-3), device=device)
    q_scale = get_Q_scale(dataset_name, specific_dataset, task_id)
    scale = float(getattr(q_scale, "Q_scale", 1.0) or 1.0)

    env = None
    if roll_env:
        env, _, _ = get_env(dataset_name, specific_dataset, task_id=task_id)

    Js, Gs = [], []
    for i in range(len(s0_raw)):
        s0 = s0_raw[i]
        s0_p = proc.preprocess(s0)
        j_s, g_s = [], []
        for _ in range(n_plans):
            x = sample_euler_karras(
                s0_p, planner, obs_dim, act_dim, horizon,
                num_steps=steps_T, num_karras=num_karras, eta=eta, device=device,
            )
            xt = torch.from_numpy(np.asarray(x)).float().to(device)
            if xt.dim() == 2:
                xt = xt.unsqueeze(0)
            s_pl = xt[..., :obs_dim]
            a = torch.clamp(xt[..., obs_dim:], -1.0, 1.0)
            s_raw = s_pl * p_std + p_mean
            H = s_raw.shape[1]
            n = H - 1
            s_r = (s_raw[:, :n] - r_mean) / r_std
            r_hat = reward_net(
                s_r.reshape(n, -1), a[:, :n].reshape(n, -1),
            ).reshape(n) / scale
            discounts = torch.tensor(
                [gamma ** t for t in range(n)], device=device, dtype=torch.float32
            )
            sH = (s_raw[0, -1] - c_mean) / c_std
            vH = symexp(critic(sH.unsqueeze(0)).squeeze())
            J = (discounts * r_hat).sum() + (gamma ** (n)) * vH
            j_s.append(float(J.item()))

            if roll_env:
                # execute actions; G = +1 success else -T
                Tlim = max_env_steps or horizon
                # OGBench reset-to-state is task-specific; fallback: cost-to-go
                try:
                    ob, _ = env.reset(options=dict(task_id=task_id))
                    # if env supports set_state, hook it here
                    success, t_done = 0.0, Tlim
                    acts = a[0, :n].cpu().numpy()
                    for t in range(min(n, Tlim)):
                        ob, r, term, trunc, info = env.step(acts[t])
                        if bool(info.get("success", False)) or term:
                            success, t_done = 1.0, t + 1
                            break
                    g_s.append(success if success else -float(t_done))
                except Exception:
                    g_s.append(float(s0_G[i]))
            else:
                g_s.append(float(s0_G[i]))
        Js.extend(j_s)
        Gs.extend(g_s)

    Js = np.asarray(Js)
    Gs = np.asarray(Gs)
    ic = float(spearmanr(Js, Gs).correlation) if np.std(Gs) > 1e-8 else float("nan")
    ev = explained_variance(Gs, Js)
    print(
        f"teacher IC ckpt={critic_checkpoint} roll_env={roll_env}\n"
        f"  pairs={len(Js)}  IC={ic:.3f}  EV={ev:.3f}\n"
        f"  J mean/std={Js.mean():.3f}/{Js.std():.3f}\n"
        f"  G mean/std={Gs.mean():.3f}/{Gs.std():.3f}"
    )
    return {"ic": ic, "ev": ev, "J": Js, "G": Gs}

