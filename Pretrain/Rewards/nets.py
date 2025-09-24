import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    """
    PyTorch implementation of reward prediction model with scalar output.
    
    Architecture:
    - Input: [deter, stoch] latent states
    - 5-layer MLP with 1024 units per layer
    - SiLU activation and layer normalization
    - Output: Single scalar reward value
    """
    
    def __init__(self, deter_dim, stoch_dim, hidden_units=1024, 
                 num_layers=5, output_activation='tanh'):
        super().__init__()
        
        # Input dimensions
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.input_dim = deter_dim + stoch_dim
        
        # Architecture parameters
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_activation = output_activation
        
        # Create the MLP layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(self.input_dim, hidden_units))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
        
        # Output layer - single scalar output
        self.layers.append(nn.Linear(hidden_units, 1))
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_units))
        
    def forward(self, deter, stoch):
        """
        Forward pass of the reward model.
        
        Args:
            deter: Deterministic latent state [batch_size, deter_dim]
            stoch: Stochastic latent state [batch_size, stoch_dim]
            
        Returns:
            reward: Scalar reward prediction [batch_size, 1]
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
        
        # Apply output activation
        if self.output_activation == 'tanh':
            reward = torch.tanh(x)
        elif self.output_activation == 'sigmoid':
            reward = torch.sigmoid(x)
        elif self.output_activation == 'none':
            reward = x
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")
        
        return reward.squeeze(-1)  # Remove last dimension to get [batch_size]
    
    def compute_loss(self, predicted_rewards, target_rewards, loss_type='mse'):
        """
        Compute the reward prediction loss.
        
        Args:
            predicted_rewards: Predicted reward values [batch_size]
            target_rewards: Target reward values [batch_size]
            loss_type: Type of loss function ('mse', 'mae', 'huber')
            
        Returns:
            loss: Scalar loss value
        """
        if loss_type == 'mse':
            loss = F.mse_loss(predicted_rewards, target_rewards)
        elif loss_type == 'mae':
            loss = F.l1_loss(predicted_rewards, target_rewards)
        elif loss_type == 'huber':
            loss = F.huber_loss(predicted_rewards, target_rewards)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
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
            predicted_rewards = self.forward(deter, stoch)
            return predicted_rewards



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

