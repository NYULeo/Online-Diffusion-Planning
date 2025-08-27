"""
diffusion_transformer.py
-----------------------

This module implements a simple Score‑based diffusion training loop using
stochastic differential equations (SDEs) as described in the provided
algorithm.  The forward SDE gradually adds noise to clean data points
and the model is trained to predict the conditional score of the noisy
samples given the original data.  The backbone of the model is a
transformer encoder which operates on sequence data (e.g. trajectories
for a planner).  Continuous time is used and the popular cosine noise
schedule from Nichol & Dhariwal (2021) is employed.  During training
a random time step is sampled from a uniform distribution on ``[0,1]``
for each element in the batch, the corresponding noisy version of
``x0`` is constructed, and the model is optimised to minimise a weighted
mean‑squared error between its prediction and the true conditional
score

  ∇ₓₜ log pₜ(xₜ|x₀) = −(xₜ − α(t)x₀) / σ(t)².

The derivation of the conditional score and the choice of weight
functions are described in the training objective from the provided
document【332371029465962†L0-L1】.  The same document recommends using
either ``σ(t)²`` or ``β(t)`` as a weighting term at each time step.

Example
-------
>>> import torch
>>> from diffusion_transformer import ScoreModel, DiffusionSDETrainer
>>> # create dummy sequence data (batch, seq_len, state_dim)
>>> x0 = torch.randn(32, 10, 4)
>>> model = ScoreModel(state_dim=4, model_dim=128, num_layers=4, num_heads=4)
>>> trainer = DiffusionSDETrainer(model, max_timesteps=1000)
>>> # perform one training step
>>> loss = trainer.train_step(x0)

Note
----
This implementation is intended as a starting point for a diffusion
planner.  Depending on the application you may need to extend the
architecture (e.g. with cross‑attention for conditioning) or modify
the time schedule.  The code is written for PyTorch 2.0+.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Dataset import KitchenDataset


def positional_encoding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Generate a sinusoidal positional/time encoding for a batch of times.

    Args:
        t: Tensor of shape ``(batch,)`` with time steps in the range ``[0,1]``.
        embed_dim: Dimension of the positional embedding (must be even).

    Returns:
        Tensor of shape ``(batch, embed_dim)`` containing sinusoidal features.
    """
    # ensure embed_dim is even for splitting into sin and cos pairs
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sinusoidal encoding")
    half_dim = embed_dim // 2
    device = t.device
    # compute log‑scale frequencies as in Vaswani et al. (2017)
    # We follow the common DDPM convention of using 10000 as the base.
    exp_term = torch.arange(half_dim, dtype=torch.float32, device=device)
    exp_term = exp_term / (half_dim - 1)
    # exponentiate to create exponentially increasing frequencies
    frequencies = torch.exp(-math.log(10000.0) * exp_term)
    # (batch, half_dim)
    args = t.unsqueeze(1) * frequencies.unsqueeze(0)
    # stack sine and cosine
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb


def cosine_alpha_sigma(t: torch.Tensor, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the continuous cosine schedule α(t) and σ(t).

    The schedule is based on the cosine schedule introduced in
    Nichol & Dhariwal (2021).  The squared alpha "bar" is defined as

    ᾱ(t) = cos²(((t + s)/(1 + s)) * π/2) / cos²(s/(1 + s) * π/2)

    from which we recover α(t) = sqrt(ᾱ(t)) and σ(t) = sqrt(1 − ᾱ(t)).

    Args:
        t: Tensor of shape ``(batch,)`` with values in ``[0,1]`` representing
           the fractional time step.  ``t=0`` corresponds to the data and
           ``t=1`` corresponds to pure noise.
        s: Small offset to avoid singularities at the endpoints.  The
           original paper uses ``s=0.008``.

    Returns:
        Tuple (alpha, sigma) where both are tensors of shape ``(batch,)``.
    """
    # ensure t is in [0,1]
    t = t.clamp(0.0, 1.0)
    # compute scaled cosine term
    factor = (t + s) / (1.0 + s)
    # f(t) = cos^2(pi/2 * factor)
    f_t = torch.cos(   torch.tensor(factor * math.pi / 2)) ** 2
    # f(0)
    f0 = torch.cos(  torch.tensor((s / (1.0 + s)) * math.pi / 2)) ** 2
    alpha_bar = f_t / f0
    # clamp alpha_bar to avoid sqrt of negative due to numerical issues
    alpha_bar = alpha_bar.clamp(min=0.0, max=1.0)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt(1.0 - alpha_bar)
    return alpha, sigma


class ScoreModel(nn.Module):
    """A simple transformer‑based score network for diffusion models.

    The network takes as input a sequence of states ``x_t`` at a given time
    ``t`` and predicts the corresponding score sθ(x_t, t).  Each state in
    the sequence is treated as a token and is embedded into a common
    model dimension.  A sinusoidal time embedding is added to every token
    to provide the network with knowledge of the current time.  The
    sequence is processed by a standard transformer encoder and a final
    linear layer maps the latent representation back to the state
    dimension.
    """

    def __init__(
        self,
        state_dim: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        time_embed_dim: Optional[int] = None,
    ):
        """Initialise the score model.

        Args:
            state_dim: Dimensionality of each state (features per time step in
                the sequence).
            model_dim: Dimension of the transformer latent space.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads per layer.
            ff_multiplier: Factor by which to expand the feedforward network
                dimensionality inside the transformer.
            time_embed_dim: Dimension used for time embeddings.  If ``None``
                it defaults to ``model_dim``.
        """
        super().__init__()
        if time_embed_dim is None:
            time_embed_dim = model_dim
        # Project the raw state vector into the model dimension.
        self.state_proj = nn.Linear(state_dim, model_dim)
        # Time embedding module: sinusoidal encoding followed by an MLP.
        # The MLP lifts the raw sinusoidal encoding up to ``model_dim``.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        # Transformer encoder configuration.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_multiplier,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Output projection back to state dimension.
        self.output_proj = nn.Linear(model_dim, state_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the score for each element in a sequence.

        Args:
            x_t: Tensor of shape ``(batch, seq_len, state_dim)`` representing
                the noisy sequence at time ``t``.
            t: Tensor of shape ``(batch,)`` with time steps in ``[0,1]``.

        Returns:
            Tensor of shape ``(batch, seq_len, state_dim)`` containing the
            estimated score ∇ₓₜ log pₜ(xₜ).
        """
        B, L, D = x_t.shape
        # project state to model_dim
        h = self.state_proj(x_t)
        # compute time embedding and broadcast to sequence length
        time_emb = positional_encoding(t, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)
        # (batch, 1, model_dim) => broadcast
        time_emb = time_emb[:, None, :]
        h = h + time_emb
        # process with transformer
        h = self.encoder(h)
        # project back to state dimension
        out = self.output_proj(h)
        return out


class DiffusionSDETrainer:
    """Trainer for score‑based diffusion models using SDEs.

    This class encapsulates the logic for sampling times, constructing
    noisy inputs, computing the target conditional score, applying the
    weighting function, and returning the mean squared error loss.  It
    does not perform any parameter updates by itself; the user should
    create an optimiser and call ``loss.backward()`` followed by
    ``optim.step()`` externally.
    """

    def __init__(
        self,
        model: ScoreModel,
        max_timesteps: int = 1000,
        s: float = 0.008,
        weight_type: str = "sigma2",
    ):
        """Initialise the trainer.

        Args:
            model: The score network to be trained.
            max_timesteps: Number of diffusion steps used during generation
                (discrete).  Continuous times are sampled uniformly from
                ``[0,1]``, so this mainly controls the number of discrete
                steps to precompute if you later wish to perform sampling.
            s: Offset for the cosine schedule.  See ``cosine_alpha_sigma``.
            weight_type: Which weight to use in the loss.  Must be one of
                ``"sigma2"`` (σ(t)²) or ``"beta"``.  Using β requires
                computing a discretised β(t) schedule.  For continuous
                training ``σ(t)²`` is generally preferred【332371029465962†L0-L1】.
            device: Torch device on which to perform computations.  If
                ``None``, defaults to the device of the model parameters.
        """
        self.model = model
        self.max_timesteps = max_timesteps
        self.s = s
        self.weight_type = weight_type
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Precompute a discrete beta schedule for weighting if required
        if weight_type == "beta":
            betas, alphas, alpha_bars = self._compute_discrete_cosine_schedule(max_timesteps, s)
            # Convert to tensors on the correct device
            self.betas = torch.tensor(betas, dtype=torch.float32, device= self.device)
            # store α_bar as it may be used later
            self.alpha_bars = torch.tensor(alpha_bars, dtype=torch.float32, device = self.device)

    def _compute_discrete_cosine_schedule(self, T: int, s: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a discrete cosine schedule of length ``T``.

        Returns three arrays: betas (shape T), alphas (shape T) and
        alpha_bars (shape T+1).  The squared cumulative alphas
        ``alpha_bar[t]`` correspond to the standard notation in DDPMs.
        """
        # Create discrete steps from 0 to T inclusive
        steps = torch.arange(T + 1, dtype=torch.float64)
        # f(t) = cos^2(((t/T + s)/(1 + s)) * pi/2)
        f_t = torch.cos(((steps / T + s) / (1.0 + s)) * math.pi / 2) ** 2
        f0 = f_t[0]
        alpha_bar = f_t / f0
        # α_t = ᾱ_t / ᾱ_{t-1}
        alphas = alpha_bar[1:] / alpha_bar[:-1]
        betas = 1.0 - alphas
        return betas.to(dtype=torch.float32).numpy(), alphas.to(dtype=torch.float32).numpy(), alpha_bar.to(dtype=torch.float32).numpy()

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        """Perform one training step and return the loss.

        Args:
            x0: Batch of clean sequences of shape ``(batch, seq_len, state_dim)``.

        Returns:
            Scalar tensor representing the weighted mean squared error loss.
        """
        x0 = x0.to(self.device)
        B, L, D = x0.shape
        # Sample random continuous time steps in [0,1]
        # We use a uniform distribution as specified in the objective【332371029465962†L0-L1】.
        t = torch.rand(B, device=self.device)
        # Compute α(t) and σ(t) for each element
        alpha, sigma = cosine_alpha_sigma(t, self.s)
        # Reshape for broadcasting: (batch, 1, 1)
        alpha_b = alpha.view(B, 1, 1)
        sigma_b = sigma.view(B, 1, 1)
        # Sample standard normal noise of the same shape as x0
        eps = torch.randn_like(x0)
        # Construct the noisy input x_t = α(t) x0 + σ(t) ε
        x_t = alpha_b * x0 + sigma_b * eps
        # Compute the target conditional score ∇ₓₜ log pₜ(xₜ|x₀)
        # From the document: ∇ₓ log p_t(x_t|x0) = −(x_t − α(t)x0) / σ(t)^2【332371029465962†L0-L1】
        target = - (x_t - alpha_b * x0) / (sigma_b ** 2 + 1e-8)
        # Compute the model prediction
        pred = self.model(x_t, t)
        # Weighting term ω(t).  Choose based on configuration.
        if self.weight_type == "sigma2":
            weight = sigma.view(B, 1, 1) ** 2
        elif self.weight_type == "beta":
            # Map continuous t to discrete indices in [0, max_timesteps-1]
            indices = (t * (self.max_timesteps - 1)).long()
            weight = self.betas[indices].view(B, 1, 1)
        else:
            raise ValueError(f"unknown weight_type {self.weight_type}")
        # Compute mean squared error and apply the weight
        mse = F.mse_loss(pred, target, reduction='none')  # shape (B, L, D)
        # Multiply by weight and then average over batch and sequence/state dims
        loss = (weight * mse).mean()
        return loss


