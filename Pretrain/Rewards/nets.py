import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve
from torch.distributions import Beta
from typing import Optional
import warnings


class CategoricalReward(nn.Module):
    
    
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
       
        bins = torch.linspace(-self.symlog_range, self.symlog_range, self.num_bins)
        return bins
    
    def symlog(self, x):
       
        return torch.sign(x) * torch.log(1 + torch.abs(x))
    
    def symexp(self, x):
       
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
    def forward(self, deter, stoch):
       
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
        
        # Convert target rewards to symlog space
        target_symlog = self.symlog(target_rewards)
        
        # Create target bin indices using two-hot encoding
        target_bins = self._rewards_to_bins(target_symlog)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(reward_logits, target_bins)
        
        return loss
    
    def _rewards_to_bins(self, rewards):
       
        # Clamp rewards to bin range
        rewards = torch.clamp(rewards, -self.symlog_range, self.symlog_range)
        
        # Find the closest bin for each reward
        distances = torch.abs(rewards.unsqueeze(-1) - self.reward_bins.unsqueeze(0))
        bin_indices = torch.argmin(distances, dim=-1)
        
        return bin_indices
    
    def predict_reward(self, deter, stoch):
        
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



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
#from common import util


def get_network(param_shape, deconv=False):
    """
    Parameters
    ----------
    param_shape: tuple, length:[(4, ), (2, )], optional

    deconv: boolean
        Only work when len(param_shape) == 4.
    """

    if len(param_shape) == 4:
        if deconv:
            in_channel, kernel_size, stride, out_channel = param_shape
            return torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        else:
            in_channel, kernel_size, stride, out_channel = param_shape
            return torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
    elif len(param_shape) == 2:
        in_dim, out_dim = param_shape
        return torch.nn.Linear(in_dim, out_dim)
    else:
        raise ValueError(f"Network shape {param_shape} illegal.")


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def get_act_cls(act_fn_name):
    act_fn_name = act_fn_name.lower()
    if act_fn_name == "tanh":
        act_cls = torch.nn.Tanh
    elif act_fn_name == "sigmoid":
        act_cls = torch.nn.Sigmoid
    elif act_fn_name == 'relu':
        act_cls = torch.nn.ReLU
    elif act_fn_name == 'identity':
        act_cls = torch.nn.Identity
    elif act_fn_name == 'swish':
        act_cls = Swish
    else:
        raise NotImplementedError(f"Activation functtion {act_fn_name} is not implemented. \
            Possible choice: ['tanh', 'sigmoid', 'relu', 'identity'].")
    return act_cls


class MLPNetwork(nn.Module):
    def __init__(
            self, input_dim: int,
            out_dim: int,
            hidden_dims: Union[int, list],
            act_fn="relu",
            out_act_fn="identity",
            **kwargs
    ):
        super(MLPNetwork, self).__init__()
        if len(kwargs.keys()) > 0:
            warn_str = "Redundant parameters for MLP network {}.".format(kwargs)
            warnings.warn(warn_str)

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_cls = get_act_cls(act_fn)
        out_act_cls = get_act_cls(out_act_fn)

        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = get_network([curr_shape, next_shape])
            self.networks.extend([curr_network, act_cls()])
        final_network = get_network([hidden_dims[-1], out_dim])
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)

    def forward(self, input):
        return self.networks(input)

    @property
    def weights(self):
        return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear)]


"""
class EnsembleModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, device, ensemble_size=7, num_elite=5, decay_weights=None,
                 act_fn="swish", out_act_fn="identity", reward_dim=1, **kwargs):
        super(EnsembleModel, self).__init__()
        assert (decay_weights is None or len(decay_weights) == len(hidden_dims) + 1)
        self.out_dim = obs_dim + reward_dim

        self.ensemble_models = [
            MLPNetwork(input_dim=obs_dim + action_dim, out_dim=self.out_dim * 2, hidden_dims=hidden_dims, act_fn=act_fn,
                       out_act_fn=out_act_fn) for _ in range(ensemble_size)]
        for i in range(ensemble_size):
            self.add_module("model_{}".format(i), self.ensemble_models[i])

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_elite = num_elite
        self.ensemble_size = ensemble_size
        self.decay_weights = decay_weights
        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])
        self.max_logvar = nn.Parameter((torch.ones((1, self.out_dim)).float() / 2).to(device), requires_grad=True)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.out_dim)).float() * 10).to(device), requires_grad=True)
        self.register_parameter("max_logvar", self.max_logvar)
        self.register_parameter("min_logvar", self.min_logvar)
        self.to(device)

    def predict(self, input):
        # convert input to tensors
        if type(input) != torch.Tensor:
            if len(input.shape) == 1:
                input = torch.FloatTensor([input]).to(util.device)
            else:
                input = torch.FloatTensor(input).to(util.device)
        # predict
        if len(input.shape) == 3:
            model_outputs = [net(ip) for ip, net in zip(torch.unbind(input), self.ensemble_models)]
        elif len(input.shape) == 2:
            model_outputs = [net(input) for net in self.ensemble_models]
        predictions = torch.stack(model_outputs)

        mean = predictions[:, :, :self.out_dim]
        logvar = predictions[:, :, self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def get_decay_loss(self):
        decay_losses = []
        for model_net in self.ensemble_models:
            curr_net_decay_losses = [decay_weight * torch.sum(torch.square(weight)) for decay_weight, weight in
                                     zip(self.decay_weights, model_net.weights)]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))

    def load_state_dicts(self, state_dicts):
        for i in range(self.ensemble_size):
            self.ensemble_models[i].load_state_dict(state_dicts[i])

"""
