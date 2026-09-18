"""
Teacher tests that follow rewards/masks as stored on trajs.

Main cube get_trajectories() (current Dataset.py):
  observations[t], masks[t] aligned, length n
  rewards[t] = processed r for transition s_t -> s_{t+1}, length n-1
  first goal = first masks==0 (usually last frame of Type A)

G is the discounted sum of THOSE stored rewards to the first goal
(or to the end if drop_timeouts=False). No assumed scale (c=-1 vs +500).

V decode (critic training space):
  value_decode='symlog'  -> V = symexp(v)           # current main
  value_decode='zscore'  -> V = v * q_std + q_mean  # Debugger
  value_decode='raw'     -> V = v
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset



def _align_reward_mask(traj: dict):
    """obs, masks (len n), trans_r[t] for s_t -> s_{t+1} (len n-1)."""
    obs = np.asarray(traj["observations"], dtype=np.float32)
    n = len(obs)
    raw_m = traj.get("masks", None)
    if raw_m is None:
        masks = np.ones(n, dtype=np.float32)
    else:
        masks = np.asarray(raw_m, dtype=np.float32).reshape(-1)
        if len(masks) < n:
            masks = np.concatenate([masks, np.ones(n - len(masks), dtype=np.float32)])
        masks = masks[:n]
    r = np.asarray(traj["rewards"], dtype=np.float64).reshape(-1)
    if len(r) == n - 1:
        trans_r = r
    elif len(r) == n:
        trans_r = r[1:]
    elif len(r) > n - 1:
        trans_r = r[: n - 1]
    else:
        trans_r = np.concatenate([r, np.zeros(n - 1 - len(r), dtype=np.float64)])
    return obs, masks, trans_r


def traj_cost_to_go(traj: dict, gamma: float, drop_timeouts: bool = True):
    """G[t] for every state. None if timeout and drop_timeouts."""
    obs, masks, trans_r = _align_reward_mask(traj)
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


def _decode_v(v, value_decode, q_mean, q_std):
    if value_decode == "symlog":
        return symexp(v)
    if value_decode == "zscore":
        return v * q_std + q_mean
    if value_decode == "raw":
        return v
    raise ValueError(f"unknown value_decode={value_decode}")


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
    value_decode: str = "symlog",
):
    from scipy.stats import spearmanr

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
        v = _decode_v(model(s).squeeze(-1), value_decode, q_mean, q_std)
        preds.append(v.detach().cpu().numpy())
        targets.append(g.numpy())
    pred = np.concatenate(preds)
    tgt = np.concatenate(targets)
    ic = float(spearmanr(pred, tgt).correlation)
    ev = explained_variance(tgt, pred)
    mae = float(np.mean(np.abs(pred - tgt)))
    print(
        f"cost-to-go test ckpt={critic_checkpoint} decode={value_decode}\n"
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
    value_decode: str = "symlog",
    reward_hidden_layers: Optional[int] = None,
    reward_hidden_dim: Optional[int] = None,
):
    from scipy.stats import spearmanr

    device = check_device()
    rng = np.random.RandomState(seed)
    _, obs_dim, act_dim = get_env(dataset_name, specific_dataset, task_id=task_id)

    s0_raw, s0_G = [], []
    for traj in trajs:
        obs, G = traj_cost_to_go(traj, gamma, drop_timeouts=True)
        if G is None:
            continue
        _, masks, _ = _align_reward_mask(traj)
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

    rh, rhd = reward_hidden_layers or hidden_layers, reward_hidden_dim or hidden_dim
    rew_state, _, _ = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    reward_net = SimpleReward(obs_dim, act_dim, rhd, rh).to(device)
    reward_net.load_state_dict(rew_state)
    reward_net.eval()
    rstat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
    r_mean = torch.as_tensor(rstat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(np.maximum(rstat.obs_std, 1e-3), device=device)

    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    stats_c = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    c_state, _ = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic.load_state_dict(c_state)
    critic.eval()
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

    env = None
    if roll_env:
        env, _, _ = get_env(dataset_name, specific_dataset, task_id=task_id)

    Js, Gs = [], []
    for i in range(len(s0_raw)):
        s0 = s0_raw[i]
        s0_p = proc.preprocess(s0)
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
            nstep = s_raw.shape[1] - 1
            s_r = (s_raw[:, :nstep] - r_mean) / r_std
            r_hat = reward_net(
                s_r.reshape(nstep, -1), a[:, :nstep].reshape(nstep, -1),
            ).reshape(nstep)
            if value_decode == "symlog":
                r_hat = r_hat / max(scale, 1e-8)
            discounts = torch.tensor(
                [gamma ** t for t in range(nstep)], device=device, dtype=torch.float32
            )
            sH = (s_raw[0, -1] - c_mean) / c_std
            vH = _decode_v(critic(sH.unsqueeze(0)).squeeze(), value_decode, q_mean_t, q_std_t)
            J = (discounts * r_hat).sum() + (gamma ** nstep) * vH
            Js.append(float(J.item()))
            if roll_env:
                try:
                    Tlim = max_env_steps or horizon
                    env.reset(options=dict(task_id=task_id))
                    success, t_done = 0.0, Tlim
                    acts = a[0, :nstep].cpu().numpy()
                    for t in range(min(nstep, Tlim)):
                        _ob, _r, term, _trunc, info = env.step(acts[t])
                        if bool(info.get("success", False)) or term:
                            success, t_done = 1.0, t + 1
                            break
                    Gs.append(success if success else -float(t_done))
                except Exception:
                    Gs.append(float(s0_G[i]))
            else:
                Gs.append(float(s0_G[i]))

    Js = np.asarray(Js)
    Gs = np.asarray(Gs)
    ic = float(spearmanr(Js, Gs).correlation) if np.std(Gs) > 1e-8 else float("nan")
    ev = explained_variance(Gs, Js)
    print(
        f"teacher IC ckpt={critic_checkpoint} roll_env={roll_env} decode={value_decode}\n"
        f"  pairs={len(Js)}  IC={ic:.3f}  EV={ev:.3f}\n"
        f"  J mean/std={Js.mean():.3f}/{Js.std():.3f}\n"
        f"  G mean/std={Gs.mean():.3f}/{Gs.std():.3f}"
    )
    return {"ic": ic, "ev": ev, "J": Js, "G": Gs}
