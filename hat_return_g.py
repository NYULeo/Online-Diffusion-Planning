"""
G as planner7 / AM J, but along the DATASET tape (logged a), not a sampled plan.

G(s_t) = sum_{k=0}^{L-1} γ^k r̂(s_{t+k}, a_{t+k}) + γ^L V(s_{t+L})
L = steps to first goal, or to end of tape if drop_timeouts=False (then V(s_last)=0
if that last state is treated as absorb; else bootstrap V).

r̂ = reward_net(s,a) / Q_scale
V = decode(critic)

Paste into Finetuning/metrics.py and call instead of (or next to) evaluate_critic.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
    gamma: float = 0.99,
    drop_timeouts: bool = True,
    value_decode: str = "symlog",
    bootstrap_timeout: bool = False,
    max_states: int = 200000,
):
    """IC/EV/MAE of V(s) vs G_hat(s) = J-style return on logged (s,a)."""
    from scipy.stats import spearmanr
    from Pretrain.Rewards.nets import SimpleReward
    from Pretrain.Critic.nets import Critic
    from Finetuning.utils import (
        check_device, get_critic_model, get_critic_stats, get_Q_scale,
        get_reward_model, get_reward_stats, symexp,
    )
    from Finetuning.metrics import (
        align_reward_mask, decode_v, explained_variance, spearman_correlation,
    )

    device = check_device()
    ns = 0 if critic_checkpoint == -1 else critic_checkpoint
    cstat = get_critic_stats(dataset_name, specific_dataset, task_id, ns)
    c_state, obs_dim = get_critic_model(
        dataset_name, specific_dataset, task_id, critic_checkpoint,
    )
    critic = Critic(obs_dim, hidden_dim, hidden_layers).to(device)
    critic.load_state_dict(c_state)
    critic.eval()

    rew_state, _, act_dim = get_reward_model(
        dataset_name, specific_dataset, reward_checkpoint, task_id,
    )
    rh, rhd = reward_hidden_layers, reward_hidden_dim
    reward_net = SimpleReward(obs_dim, act_dim, rhd, rh).to(device)
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

    c_mean = torch.as_tensor(cstat.obs_mean, device=device, dtype=torch.float32)
    c_std = torch.as_tensor(np.maximum(cstat.obs_std, 1e-3), device=device)
    r_mean = torch.as_tensor(rstat.obs_mean, device=device, dtype=torch.float32)
    r_std = torch.as_tensor(np.maximum(rstat.obs_std, 1e-3), device=device)

    def V_np(obs_batch):
        x = torch.as_tensor(
            np.stack([cstat.norm_obs(o) for o in obs_batch], axis=0),
            device=device, dtype=torch.float32,
        )
        v = critic(x).squeeze(-1)
        return decode_v(v, value_decode, q_mean, q_std).detach().cpu().numpy()

    def rhat_np(obs_t, act_t):
        if len(obs_t) == 0:
            return np.zeros((0,), dtype=np.float64)
        s = torch.as_tensor(np.stack(obs_t, axis=0), device=device, dtype=torch.float32)
        a = torch.as_tensor(np.stack(act_t, axis=0), device=device, dtype=torch.float32)
        a = torch.clamp(a, -1.0, 1.0)
        sn = (s - r_mean) / r_std
        r = reward_net(sn, a).squeeze(-1) / max(scale, 1e-8)
        return r.detach().cpu().numpy().astype(np.float64)

    xs, gs = [], []
    n_keep = n_drop = 0
    for traj in trajs:
        obs, masks, _ = align_reward_mask(traj)
        n = len(obs)
        acts = np.asarray(traj.get("actions", np.zeros((0, 1))), dtype=np.float32)
        if n == 0:
            continue
        # actions align with transitions: prefer len n-1
        if len(acts) >= n:
            acts = acts[: n - 1]
        elif len(acts) < n - 1:
            pad = np.zeros((n - 1 - len(acts), acts.shape[-1] if acts.ndim == 2 else 1), dtype=np.float32)
            if acts.ndim == 1:
                acts = acts.reshape(-1, 1)
            acts = np.concatenate([acts.reshape(len(acts), -1), pad], axis=0)

        goal = np.where(masks == 0.0)[0]
        if len(goal) == 0:
            if drop_timeouts:
                n_drop += 1
                continue
            T = n - 1
            v_end = float(V_np(obs[T:T + 1])[0]) if bootstrap_timeout else 0.0
        else:
            T = int(goal[0])
            v_end = 0.0

        n_keep += 1
        r_seq = rhat_np(obs[:T], [acts[i] for i in range(T)]) if T > 0 else np.zeros((0,))
        G = np.zeros(T + 1, dtype=np.float64)
        G[T] = v_end
        acc = v_end
        for t in range(T - 1, -1, -1):
            acc = float(r_seq[t]) + gamma * acc
            G[t] = acc
        for t in range(T + 1):
            xs.append(cstat.norm_obs(obs[t]))
            gs.append(G[t])
        if len(xs) >= max_states:
            break

    x = np.asarray(xs, dtype=np.float32)
    g = np.asarray(gs, dtype=np.float32)
    print(
        f"hat-return G: states={len(x)} trajs kept={n_keep} dropped={n_drop} "
        f"G min/max={g.min():.3f}/{g.max():.3f} mean/std={g.mean():.3f}/{g.std():.3f}"
    )

    loader = DataLoader(
        list(zip(torch.from_numpy(x), torch.from_numpy(g))),
        batch_size=512, shuffle=False,
    )
    preds, tgts = [], []
    for s, gt in loader:
        s = s.to(device)
        v = decode_v(critic(s).squeeze(-1), value_decode, q_mean, q_std)
        preds.append(v.detach().cpu().numpy())
        tgts.append(gt.numpy())
    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)
    ic = float(spearmanr(pred, tgt).correlation)
    ev = explained_variance(tgt, pred)
    mae = float(np.mean(np.abs(pred - tgt)))
    print(
        f"hat-return test ckpt={critic_checkpoint}\n"
        f"  n={len(pred)}  IC={ic:.3f}  EV={ev:.3f}  MAE={mae:.3f}\n"
        f"  pred mean/std={pred.mean():.3f}/{pred.std():.3f}\n"
        f"  Ghat mean/std={tgt.mean():.3f}/{tgt.std():.3f}"
    )
    return {"ic": ic, "ev": ev, "mae": mae, "pred": pred, "G": tgt}
