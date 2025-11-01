import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve
from torch.distributions import Beta
from typing import Optional
class CategoricalReward(nn.Module):
    """
    PyTorch implementation of the DreamSmooth reward prediction model.
    
    Architecture:
    - Input: [deter, stoch] latent states
    - 5-layer MLP with 1024 units per layer
    - SiLU activation and layer normalization
    - Output: 255 discrete bins for reward prediction
    """
    
    def __init__(self, deter_dim, stoch_dim, hidden_units=1024, 
                 num_layers=5, num_bins=255, symlog_range=20.0):
        super().__init__()
        
        # Input dimensions
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.input_dim = deter_dim + stoch_dim
        
        # Architecture parameters
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_bins = num_bins
        self.symlog_range = symlog_range
        
        # Create the MLP layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(self.input_dim, hidden_units))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_units, num_bins))
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_units))
        
        # Initialize reward bins
        self.register_buffer('reward_bins', self._create_reward_bins())
        
    def _create_reward_bins(self):
        """Create the discrete reward bins in symlog space."""
        bins = torch.linspace(-self.symlog_range, self.symlog_range, self.num_bins)
        return bins
    
    def symlog(self, x):
        """Symmetric log transformation."""
        return torch.sign(x) * torch.log(1 + torch.abs(x))
    
    def symexp(self, x):
        """Symmetric exponential transformation."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
    def forward(self, deter, stoch):
        """
        Forward pass of the reward model.
        
        Args:
            deter: Deterministic latent state [batch_size, deter_dim]
            stoch: Stochastic latent state [batch_size, stoch_dim]
            
        Returns:
            reward_logits: Logits for reward prediction [batch_size, num_bins]
            reward_probs: Probability distribution over bins
            reward_mean: Expected reward value
        """
        # Concatenate inputs
        x = torch.cat([deter, stoch], dim=-1)
        
        # Forward through MLP layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            if i < len(self.layers) - 1:  # Hidden layers
                x = layer(x)
                x = norm(x)
                x = F.silu(x)  # SiLU activation
            else:  # Output layer
                x = layer(x)
        
        # Get logits for reward bins
        reward_logits = x
        
        # Convert to probabilities
        reward_probs = F.softmax(reward_logits, dim=-1)
        
        # Compute expected reward value
        reward_mean = torch.sum(reward_probs * self.reward_bins.unsqueeze(0), dim=-1)
        
        return reward_logits, reward_probs, reward_mean
    
    def compute_loss(self, reward_logits, target_rewards):
        """
        Compute the reward prediction loss.
        
        Args:
            reward_logits: Predicted logits [batch_size, num_bins]
            target_rewards: Target reward values [batch_size]
            
        Returns:
            loss: Cross-entropy loss
        """
        # Convert target rewards to symlog space
        target_symlog = self.symlog(target_rewards)
        
        # Create target bin indices using two-hot encoding
        target_bins = self._rewards_to_bins(target_symlog)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(reward_logits, target_bins)
        
        return loss
    
    def _rewards_to_bins(self, rewards):
        """
        Convert continuous rewards to discrete bin indices using two-hot encoding.
        
        Args:
            rewards: Continuous reward values [batch_size]
            
        Returns:
            bin_indices: Discrete bin indices [batch_size]
        """
        # Clamp rewards to bin range
        rewards = torch.clamp(rewards, -self.symlog_range, self.symlog_range)
        
        # Find the closest bin for each reward
        distances = torch.abs(rewards.unsqueeze(-1) - self.reward_bins.unsqueeze(0))
        bin_indices = torch.argmin(distances, dim=-1)
        
        return bin_indices
    
    def predict_reward(self, deter, stoch):
        """
        Predict reward from latent states.
        
        Args:
            deter: Deterministic latent state [batch_size, deter_dim]
            stoch: Stochastic latent state [batch_size, stoch_dim]
            
        Returns:
            predicted_rewards: Predicted reward values [batch_size]
        """
        with torch.no_grad():
            _, _, reward_mean = self.forward(deter, stoch)
            # Convert from symlog space back to original reward space
            predicted_rewards = self.symexp(reward_mean)
            return predicted_rewards




class ScalarReward(nn.Module):
  
    
    def __init__(self, obs_dim, act_dim, hidden_units=1024, 
                 num_layers=5, eps=1e-4):
        super().__init__()
        
        # Input dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.eps = eps
        
        # Architecture parameters
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        #self.output_activation = output_activation
        
        # Create the MLP layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(self.input_dim, hidden_units))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
        
        # Output layer - single scalar output
        self.layers.append(nn.Linear(hidden_units, 2))
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_units))
        
    def forward(self, obs, act):
        
        # Concatenate inputs
        x = torch.cat([obs, act], dim=-1)
        
        # Forward through MLP layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            if i < len(self.layers) - 1:  # Hidden layers
                x = layer(x)
                x = norm(x)
                x = F.silu(x)  # SiLU activation
            else:  # Output layer
                x = layer(x)
        
        raw_alpha = x[:, 0]  # shape (B,)
        raw_beta  = x[:, 1]  # shape (B,)
        # Transform to positive
        alpha = F.softplus(raw_alpha) + 1e-4
        beta  = F.softplus(raw_beta)  + 1e-4
        
        return alpha, beta
    
    def predict(self, obs, act, agg: str = "mean", ci:  Optional[float] = None):
        """
        Prediction head.
        - agg: "mean" (default), "mode", or "median_approx".
        - ci: if set (e.g., 0.95), also returns (lo, hi) credible interval.
        Returns:
            pred  : [B] point estimate in [0,1]
            (lo,hi): [B],[B] CI if ci is not None
        """
        alpha, beta = self.forward(obs, act)
        dist = Beta(alpha, beta)  # PyTorch Beta distribution

        if agg == "mean":
            pred = alpha / (alpha + beta)               # E[R]
        elif agg == "mode":
            # Only valid if alpha>1 and beta>1; fall back to mean otherwise
            mask = (alpha > 1) & (beta > 1)
            mode = (alpha - 1) / (alpha + beta - 2)
            mean = alpha / (alpha + beta)
            pred = torch.where(mask, mode, mean)
        elif agg == "median_approx":
            # Kerman (2011) approx: (α-1/3)/(α+β-2/3) for α,β≥1
            pred = (alpha - 1/3) / (alpha + beta - 2/3)
            pred = pred.clamp(0.0, 1.0)                 # guard numerics
        else:
            raise ValueError("agg must be 'mean', 'mode', or 'median_approx'")

        if ci is None:
            return pred
        qlo = (1 - ci) / 2
        qhi = 1 - qlo
        lo = dist.icdf(torch.full_like(alpha, qlo))
        hi = dist.icdf(torch.full_like(alpha, qhi))
        return pred, (lo, hi)

    def loss(self, obs, act, r):
        """
        Beta negative log-likelihood for targets r in [0,1].
        Clamps r into (eps, 1-eps) to avoid log_prob at the boundaries 0 or 1.
        Returns scalar loss.
        """
        alpha, beta = self.forward(obs, act)
        dist = Beta(alpha, beta)
        r = r.clamp(self.eps, 1 - self.eps)             # keep inside (0,1)
        nll = -dist.log_prob(r)                         # [B]
        return nll.mean()
    
    def variance(self, obs, act):
        alpha, beta = self.forward(obs, act)
        var = (alpha * beta) / ( ((alpha + beta)**2) + (alpha + beta + 1) )
        return var

    def compute_reward_gradients(self, obs, act, agg: str = "mean", return_pred: bool = True):
       """
         Compute the gradient of the reward prediction with respect to concatenated [obs, act].
    
         Args:
            reward_net: ScalarReward model
            obs: Observations tensor [batch_size, obs_dim]
            act: Actions tensor [batch_size, act_dim]
            agg: Aggregation method - "mean", "mode", or "median_approx" (default: "mean")
            return_pred: Whether to return the predicted values (default: True)
    
         Returns:
            grad_input: Gradient with respect to [obs, act] [batch_size, obs_dim + act_dim]
            pred: The predicted reward values [batch_size] (only if return_pred=True)
       """
       # Concatenate and ensure requires_grad
       x = torch.cat([obs, act], dim=-1).detach().requires_grad_(True)
    
       # Forward pass through the network
       # We need to manually pass through the network since it expects separate obs, act
       # So we'll split and call forward
       obs_split = x[..., :self.obs_dim]
       act_split = x[..., self.obs_dim:]
    
       alpha, beta = self.forward(obs_split, act_split)
    
       # Compute prediction based on aggregation method
       if agg == "mean":
            pred = alpha / (alpha + beta)
       elif agg == "mode":
            mask = (alpha > 1) & (beta > 1)
            mode = (alpha - 1) / (alpha + beta - 2)
            mean = alpha / (alpha + beta)
            pred = torch.where(mask, mode, mean)
       elif agg == "median_approx":
            pred = (alpha - 1/3) / (alpha + beta - 2/3)
            pred = pred.clamp(0.0, 1.0)
       else:
            raise ValueError("agg must be 'mean', 'mode', or 'median_approx'")
    
       # Compute gradient of sum (vectorized and efficient)
       pred_sum = pred.sum()
    
       grad_input = torch.autograd.grad(
            outputs=pred_sum,
            inputs=x,
            create_graph=True,
            retain_graph=True)[0]
    
       if return_pred:
           return grad_input, pred
       else:
           return grad_input


def compute_reward_gradients_per_sample(reward_net, obs, act, agg: str = "mean"):
      """
        Compute per-sample gradients with respect to concatenated [obs, act].
        Useful when you need independent gradients for each sample in the batch.
    
        Args:
        reward_net: ScalarReward model
        obs: Observations tensor [batch_size, obs_dim]
        act: Actions tensor [batch_size, act_dim]
        agg: Aggregation method - "mean", "mode", or "median_approx" (default: "mean")
    
        Returns:
        grad_input: Gradient with respect to [obs, act] [batch_size, obs_dim + act_dim]
        pred: The predicted reward values [batch_size]
      """
      # Concatenate and ensure requires_grad
      x = torch.cat([obs, act], dim=-1).detach().requires_grad_(True)
    
    # Split for forward pass
      obs_split = x[..., :reward_net.obs_dim]
      act_split = x[..., reward_net.obs_dim:]
    
      alpha, beta = reward_net.forward(obs_split, act_split)
    
    # Compute prediction based on aggregation method
      if agg == "mean":
           pred = alpha / (alpha + beta)
      elif agg == "mode":
           mask = (alpha > 1) & (beta > 1)
           mode = (alpha - 1) / (alpha + beta - 2)
           mean = alpha / (alpha + beta)
           pred = torch.where(mask, mode, mean)
      elif agg == "median_approx":
           pred = (alpha - 1/3) / (alpha + beta - 2/3)
           pred = pred.clamp(0.0, 1.0)
      else:
          raise ValueError("agg must be 'mean', 'mode', or 'median_approx'")
    
    # Compute per-sample gradients
      batch_size = pred.shape[0]
      grad_input = torch.zeros_like(x)
    
      for i in range(batch_size):
          grad = torch.autograd.grad(
              outputs=pred[i],
              inputs=x,
              retain_graph=(i < batch_size - 1),
              create_graph=False,
              allow_unused=False
          )[0]
          grad_input[i] = grad[i]
    
      return grad_input, pred


class LargeScalarReward(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int = 0,
        hidden_dims=(256, 256, 128),
        embedding_dim: int = 64,
        output_scale: float = 5.0,
        grad_reg_coeff: float = 1e-4,
    ):
        """
        Reward-model head: (s, a[, g]) → r_hat in [0, output_scale]
        """
        super().__init__()
        in_dim = state_dim + action_dim + (goal_dim if goal_dim > 0 else 0)
        
        # Feature extractor MLP with residual blocks
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        
        # Residual connection optionally
        self.residual = nn.Linear(in_dim, hidden_dims[2]) if in_dim != hidden_dims[2] else nn.Identity()
        
        self.emb = nn.Linear(hidden_dims[2], embedding_dim)
        
        # Final head to scalar reward
        self.head = nn.Linear(embedding_dim, 1)
        
        self.output_scale = output_scale
        self.act = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dims[0])
        self.norm2 = nn.LayerNorm(hidden_dims[1])
        self.norm3 = nn.LayerNorm(hidden_dims[2])
        self.grad_reg_coeff = grad_reg_coeff
        
    def forward(self, s: torch.Tensor, a: torch.Tensor, g: torch.Tensor = None):
        if g is not None:
            x = torch.cat([s, a, g], dim=-1)
        else:
            x = torch.cat([s, a], dim=-1)
        
        x0 = self.act(self.norm1(self.fc1(x)))
        x1 = self.act(self.norm2(self.fc2(x0)))
        x2 = self.act(self.norm3(self.fc3(x1)))
        
        # add residual
        x2 = x2 + self.residual(x)
        
        h = self.act(self.emb(x2))
        z = self.head(h).squeeze(-1)
        
        # Bound output to [0, output_scale] using sigmoid then scaling
        r_hat = self.output_scale * torch.sigmoid(z)
        return r_hat
    
    def loss(self, s, a, r):
        r_hat = self.forward(s, a)
        loss_reg = F.mse_loss(r_hat, r)
        s.requires_grad_(True)
        r_hat_for_grad = self.forward(s, a)
        grads = torch.autograd.grad(
            outputs=r_hat_for_grad.sum(),
            inputs=s,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm2 = grads.pow(2).sum(dim=-1).mean()
        loss_grad_reg = self.grad_reg_coeff * grad_norm2
        loss = loss_reg + loss_grad_reg
        return loss
        

class Reward(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)




def gaussian_rewards(episode, sigma):
    if sigma > 0:
        reward_raw = episode["rewards"]
        reward_smooth = gaussian_filter1d(reward_raw, sigma, mode="nearest")
        episode.update({"rewards_raw": reward_raw, "rewards": reward_smooth})
        return episode