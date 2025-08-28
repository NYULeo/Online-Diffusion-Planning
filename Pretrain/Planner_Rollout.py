# sampler_mask_initial.py
# pip install torch numpy
import math
import numpy as np
import torch
from dataclasses import dataclass
from Backbone import UNet1D
from env import get_env
import pickle
from typing import Optional
import random
import os
from train_planner import train_planner
from utils import set_seed

# ---- 2) Cosine schedule (Nichol & Dhariwal) ----
def cosine_alphas_bar(T: int, s: float = 0.008):
    ts = torch.linspace(0, T, T + 1)
    f = torch.cos((math.pi/2)*((ts/T)+s)/(1+s))**2
    ab = f / f[0]                 # \bar{alpha}_t, t=0..T
    alpha = ab[1:] / ab[:-1]      # alpha_t, t=1..T
    beta  = 1.0 - alpha           # beta_t
    return {"ab": ab, "alpha": alpha, "beta": beta}

@dataclass
class VPDynamics:  # VP SDE
    betas: torch.Tensor             # (T,)
    def f(self, x, beta_t): return -0.5 * beta_t * x
    def g(self, beta_t):        return torch.sqrt(beta_t.clamp_min(1e-12))

# ---- 3) Reverse sampler with mask on first D1 coords ----
@torch.no_grad()
def sample_flat_with_initial(
    s0: np.ndarray,             # (D1,)
    score_model: UNet1D,
    d_s: int, 
    d_a: int, 
    horizon: int,   # dims & horizon used in training
    steps_T: int = 1000,
    eta: float = 1.0, 
    device: Optional[str] =  None          # 1.0: SDE (stochastic), 0.0: ODE (deterministic)
):
    D_tot = (d_s + d_a) * horizon
    device = device or("cuda" if torch.cuda.is_available() else "cpu")

    # schedule & dynamics
    sch = cosine_alphas_bar(steps_T)
    betas = sch["beta"].to(device)
    dyn   = VPDynamics(betas=betas)

    # reverse integration params (continuous time in [0,1])
    dt = -1.0 / steps_T
    sqrt_neg_dt = math.sqrt(-dt)

    # start from N(0, I) in data space
    x = torch.randn(D_tot, device=device)

    # mask for first D1 entries; fixed vector holding s0 in those slots
    M = torch.zeros(D_tot, device=device); M[:d_s] = 1.0
    Xfix = torch.zeros(D_tot, device=device, dtype=x.dtype)
    Xfix[:d_s] = torch.as_tensor(s0, device=device, dtype=x.dtype)

    for k in reversed(range(steps_T)):
        t_float = (k + 0.5) / steps_T
        beta_t  = betas[k]
        fx = dyn.f(x, beta_t)                 # -0.5*beta_t*x
        gt = dyn.g(beta_t)                    # sqrt(beta_t)
        t_float = torch.tensor(t_float)
        score = score_model(x, t_float)       # s_theta(x,t) -> (D_tot,)

        drift = (-fx + 0.5*(1.0 + eta**2)*(gt**2)*score)
        noise = eta * gt * torch.randn_like(x)

        x = x + drift * dt + noise * sqrt_neg_dt
        # --- constrain initial D1 coords to s0 (inpainting) ---
        x = M * Xfix + (1.0 - M) * x

    return x.detach().cpu().numpy()  # shape (D_tot,)

def rollout(model, env_name, d_a, d_s, horizon, steps_T, eta, specific_env: Optional[str] = None, episode_length : int = 1000):
     env = get_env(env_name, specific_env)
     s0 = env.reset()
     current_state = s0
     trajectory = []
     trajectory.append(current_state)
     for i in range(episode_length):
           X = sample_flat_with_initial(current_state, model, d_s, d_a, horizon, steps_T, eta)
           action = X[d_s:d_s+d_a].copy()
           obs, reward, terminated, truncated, info = env.step(action)
           trajectory.append(action)
           trajectory.append(obs.copy())
           trajectory.append(reward.copy())
           current_state = obs.copy()
           print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                print(f"Episode {i} terminated or truncated")
                break
     return trajectory





# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    set_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_planner(dataset_name = 'kitchen', specific_dataset = 'complete', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    print(model(torch.rand(4, 2176), torch.tensor(0.5)))
    s0 = np.zeros(59, dtype=np.float32)  # put the real current state here
    #X = sample_flat_with_initial(s0, model, d_s = 59, d_a = 9, horizon = 32, steps_T = 1000, eta = 1.0, device = device)
    
    #rollout(model, 'kitchen', d_a, d_s, horizon, steps_T = 1000, eta = 1.0, episode_length = 1000, device = device)
      