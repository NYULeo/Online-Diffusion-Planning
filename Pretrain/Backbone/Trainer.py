from sqlite3 import DatabaseError
import torch
from typing import Optional
from .utils import cosine_alpha_sigma, cosine_beta
import torch.nn.functional as F



class SDETrainer:
    def __init__(
        self,
        model,                             # TemporalUnet returning (B,H,D)
        s: float = 0.008,                  # cosine offset
        weight_type: str = 'sigma2',         # {"one", "sigma2", "beta"}
        device: Optional[torch.device] = None,
        eps: float = 1e-5,                 # clamp for t, ᾱ stability
    ):
        self.model = model
        self.s = s
        self.weight_type = weight_type
        self.eps = eps
        self.device = device or next(model.parameters()).device

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        x0 = x0.to(self.device)                          # (B,H,D)
        B, H, D = x0.shape

        # 1) sample time t ~ U(eps, 1 - eps), per sample (shape: (B,))
        t = torch.rand(B, device=self.device) * (1.0 - 2*self.eps) + self.eps

        # 2) α(t), σ(t) from cosine schedule (return 1D tensors, then expand to (B,1,1))
        alpha, sigma = cosine_alpha_sigma(t, self.s)     # (B,), (B,)
        alpha_b = alpha.view(B, 1, 1)                    # -> (B,1,1) for broadcasting
        sigma_b = sigma.view(B, 1, 1)                    # -> (B,1,1)

        # 3) perturbation
        eps = torch.randn_like(x0, dtype = x0.dtype)                       # (B,H,D)
        x_t = alpha_b * x0 + sigma_b * eps               # (B,H,D)

        # 4) analytic Gaussian score target for VP
        target = -(x_t - alpha_b * x0) / ( sigma_b**2 + 1e-8)   # (B,H,D)  (Song et al.) :contentReference[oaicite:2]{index=2}

        # 5) model prediction (must match (B,H,D)); pass per-sample t
        pred = self.model(x_t, t)                        # (B,H,D)

        # 6) loss weighting λ(t)
        if self.weight_type == "one":
            lam = torch.ones(B, device=self.device)      # classic VP choice
        elif self.weight_type == "sigma2":
            lam = sigma.pow(2)                           # common balancing heuristic (more VE-like)
        elif self.weight_type == "beta":
            beta = cosine_beta(t, self.s)                # g(t)^2 = β(t) for VP-SDE
            lam = beta
        else:
            raise ValueError(f"Unsupported weight_type {self.weight_type}")

        # 7) weighted MSE; λ(t) is per-sample => apply after summing over (H,D)
        mse = (pred - target).pow(2).sum(dim=(1, 2))      # (B,)
        loss = (lam * mse).mean()
        loss = loss/(H*D)
        return loss


