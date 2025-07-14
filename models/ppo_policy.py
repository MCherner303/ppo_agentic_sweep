import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class PPOPolicy(nn.Module):
    """
    Policy network for PPO that includes both actor and critic.
    
    This is a clean implementation specifically designed for PPO, without any A2C-specific logic.
    It includes proper support for mixed precision training and is optimized for PPO's requirements.
    """
    
    def __init__(self, 
                 grid_size: int = 10, 
                 num_actions: int = 4, 
                 hidden_size: int = 256,
                 use_lstm: bool = True):
        """
        Initialize the PPO Policy network.
        
        Args:
            grid_size: Size of the grid environment
            num_actions: Number of possible actions
            hidden_size: Size of hidden layers
            use_lstm: Whether to use LSTM for temporal dependencies
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.use_lstm = use_lstm
        
        # Shared feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, grid_size, grid_size]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, grid_size, grid_size]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Flatten()
        )
        
        # Calculate the size of the flattened conv output
        conv_out_size = 64 * grid_size * grid_size
        
        # Feature extractor before LSTM/MLP
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # LSTM for temporal dependencies
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
            self.hidden = None
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Critic head (value function) with enhanced stability
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Value function normalization
        self.value_normalizer = nn.BatchNorm1d(1, affine=False)
        self.value_scale = nn.Parameter(torch.ones(1))
        self.value_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with orthogonal initialization and scaled final layer."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if isinstance(module, nn.Linear) and module == self.critic[-1]:
                # Special initialization for final critic layer
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                # Orthogonal initialization for all other linear layers
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize LayerNorms
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def init_hidden(self, batch_size: int = 1):
        """Initialize hidden state for LSTM."""
        if not self.use_lstm:
            return None
            
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device)
        )
    
    def forward(self, 
               x: torch.Tensor, 
               hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
               return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4, grid_size, grid_size)
            hidden: Optional LSTM hidden state (h, c)
            return_hidden: Whether to return the hidden state
            
        Returns:
            Dictionary containing:
                - 'logits': Action logits
                - 'value': State value
                - 'hidden': LSTM hidden state (if return_hidden=True)
        """
        # Get the current dtype for mixed precision
        current_dtype = next(self.parameters()).dtype
        
        # Ensure input is in the correct dtype for the model
        x = x.to(dtype=current_dtype)
        
        # Extract features
        with torch.autocast(device_type='cuda' if x.is_cuda else 'cpu', 
                          dtype=current_dtype, enabled=torch.is_autocast_enabled()):
            features = self.conv(x)
            features = self.feature_extractor(features)
            
            # Add sequence dimension if needed for LSTM
            if self.use_lstm:
                if features.dim() == 2:  # Add sequence dimension
                    features = features.unsqueeze(1)
                
                # Ensure hidden states are in the same dtype as features
                if hidden is not None:
                    hidden = (hidden[0].to(dtype=current_dtype), 
                             hidden[1].to(dtype=current_dtype))
                else:
                    if self.hidden is None or self.hidden[0].size(1) != features.size(0):
                        self.init_hidden(features.size(0))
                    self.hidden = (self.hidden[0].to(dtype=current_dtype), 
                                 self.hidden[1].to(dtype=current_dtype))
                    hidden = self.hidden
                
                # Process through LSTM
                lstm_out, new_hidden = self.lstm(features, hidden)
                
                # Update hidden state
                self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
                
                # Get the last output from LSTM
                features = lstm_out.squeeze(1)
            
            # Get action logits
            logits = self.actor(features)
            
            # Get value with normalization
            value = self.critic(features).squeeze(-1)
            if self.training and value.dim() > 1:  # Only normalize during training and if batch size > 1
                value = self.value_normalizer(value.unsqueeze(1)).squeeze(1)
            value = value * self.value_scale + self.value_bias
            
            result = {
                'logits': logits,
                'value': value
            }
            
            if return_hidden and self.use_lstm:
                result['hidden'] = new_hidden
                
        return result
    
    def get_action_and_value(self, 
                           x: torch.Tensor, 
                           action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for a given state.
        
        Args:
            x: Input state
            action: If provided, use this action instead of sampling
            
        Returns:
            Dictionary containing:
                - 'action': Selected action
                - 'log_prob': Log probability of the action
                - 'entropy': Entropy of the action distribution
                - 'value': State value
        """
        # Forward pass
        output = self.forward(x, return_hidden=False)
        logits = output['logits']
        value = output['value']
        
        # Create action distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return {
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value
        }
