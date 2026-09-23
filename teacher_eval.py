"""Teacher metrics — two tensors only.

J_by_state : (n_s0, n_plans)     full-horizon J = R^{(N)}(tau)
R_s        : (n_s0, n_horizons)  E_tau R^{(k)}(s), k=1..N

J_by_state alone → WSJD / between_std / mean_J
R_s alone        → E_s[E_tau R^N / E_tau R^1], E_s[Var_k E_tau R^k]
"""

from __future__ import annotations

import numpy as np


def metrics_from_j_by_state(J_by_state: np.ndarray, eps: float = 1e-8) -> dict:
    J = np.asarray(J_by_state, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] < 2:
        raise ValueError("J_by_state must be (n_s0, n_plans) with n_plans>=2")
    mu = J.mean(axis=1)
    sig = J.std(axis=1, ddof=0)
    out = {
        "n_s0": int(J.shape[0]),
        "n_plans": int(J.shape[1]),
        "mean_J": float(J.mean()),
        "std_J": float(J.std()),
        "WSJD": float(sig.mean()),
        "between_std": float(mu.std(ddof=0)),
        "frac_alive": float((sig > eps).mean()),
    }
    print(
        f"J_by_state {J.shape}  mean/std={out['mean_J']:.3f}/{out['std_J']:.3f}\n"
        f"  WSJD={out['WSJD']:.4f}  between={out['between_std']:.4f}  "
        f"alive={out['frac_alive']:.3f}"
    )
    return out


def sampled_plan_bellman_metrics(R_s: np.ndarray, eps: float = 1e-8) -> dict:
    R = np.asarray(R_s, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("R_s must be (n_s0, n_horizons) with n_horizons>=2")
    R1, RN = R[:, 0], R[:, -1]
    ok = np.abs(R1) > eps
    rho = float((RN[ok] / R1[ok]).mean()) if np.any(ok) else float("nan")
    sig2 = float(R.var(axis=1).mean())
    out = {
        "expected_ratio_EN_over_E1": rho,
        "expected_var_k": sig2,
        "n_s0": int(R.shape[0]),
        "n_horizons": int(R.shape[1]),
    }
    print(
        f"R_s {R.shape}\n"
        f"  E_s[ E_tau R^N / E_tau R^1 ] = {rho:.4f}\n"
        f"  E_s[ Var_k E_tau R^k ]       = {sig2:.4f}"
    )
    return out


def evaluate_plan_teachers(J_by_state, R_s=None) -> dict:
    out = metrics_from_j_by_state(J_by_state)
    if R_s is not None:
        out.update(sampled_plan_bellman_metrics(R_s))
    return out
