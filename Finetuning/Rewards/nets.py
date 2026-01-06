import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve
from torch.distributions import Beta
from typing import Optional
import warnings



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


class SimpleReward(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()                              
        )
        #self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        #return self.net(x).squeeze(-1) * self.scale
        return self.net(x).squeeze(-1)