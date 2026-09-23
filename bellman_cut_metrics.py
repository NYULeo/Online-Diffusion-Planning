"""Multi-horizon Bellman-cut stats (same as Finetuning/Raw.py probe).

R_s[i, j] = E_tau[ R^{(K_j)}(tau | s0_i) ]
K = 1..n,  R^{(K)} = sum_{t<K} γ^t r_hat_t + γ^K V(s_K)

Paste into Finetuning/metrics.py.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


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
