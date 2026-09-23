"""Exact slide metrics on SAMPLED plans.

    rho  = E_s[ E_tau R^N(s,tau) / E_tau R^1(s,tau) ]
    sig2 = E_s[ Var_k  E_tau R^k(s,tau) ]

R_s[i, j] must already be E_tau R^{(j+1)}(s_i), shape (n_s0, n_K).
Get R_s from probe_multi_horizon_bellman (return R_s after gather).
"""

from __future__ import annotations

import numpy as np


def expected_ratio_EN_over_E1(R_s, eps: float = 1e-8) -> float:
    """E_s[ E_tau R^N / E_tau R^1 ]."""
    R = np.asarray(R_s, dtype=np.float64)
    R1, RN = R[:, 0], R[:, -1]
    ok = np.abs(R1) > eps
    if not np.any(ok):
        return float("nan")
    return float((RN[ok] / R1[ok]).mean())


def expected_var_k(R_s) -> float:
    """E_s[ Var_k E_tau R^k ]  (population var over k)."""
    R = np.asarray(R_s, dtype=np.float64)
    return float(R.var(axis=1).mean())


def sampled_plan_bellman_metrics(R_s, eps: float = 1e-8) -> dict:
    rho = expected_ratio_EN_over_E1(R_s, eps)
    sig2 = expected_var_k(R_s)
    print(
        f"sampled-plan Bellman  M={len(R_s)} K={R_s.shape[1]}\n"
        f"  E_s[ E_tau R^N / E_tau R^1 ] = {rho:.4f}\n"
        f"  E_s[ Var_k E_tau R^k ]       = {sig2:.4f}"
    )
    return {
        "expected_ratio_EN_over_E1": rho,
        "expected_var_k": sig2,
    }
