import random
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import deque, defaultdict
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.cuda.amp import GradScaler, autocast

# For checkpointing
from datetime import datetime

# Import from local modules
from models.a2c_policy import A2CPolicy
from models.llama_mock import create_mock_llama

class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([channels, 10, 10])
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([channels, 10, 10])
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        x = self.dropout(x)
        x = self.ln2(self.conv2(x))
        return F.leaky_relu(x + residual)

def init_weights(m):
    """Initialize weights with orthogonal initialization"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.orthogonal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class RunningMeanStd:
    """Tracks the mean and variance of values."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        """Update the running statistics."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch mean, variance, and count."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

class A2CPolicy(nn.Module):
    def __init__(self, grid_size: int = 10, num_actions: int = 4):
        """
        Enhanced A2C Policy Network with residual connections and layer norm
        
        Args:
            grid_size: Size of the grid environment
            num_actions: Number of possible actions (up, down, left, right)
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = num_actions  # Store number of actions as a class attribute
        
        # Apply orthogonal initialization to all layers
        self.apply(init_weights)
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.LayerNorm([64, grid_size, grid_size]),
            nn.LeakyReLU(0.1)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            nn.Dropout2d(0.1),
            ResidualBlock(64),
            nn.Dropout2d(0.1)
        )
        
        # Calculate the size of the flattened conv output
        self.conv_out_size = 64 * grid_size * grid_size
        
        # Feature processing for actor and critic
        self.feature_net = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 4, grid_size, grid_size)
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Feature extraction
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        
        # Process features
        features = x.view(x.size(0), -1)
        features = self.feature_net(features)
        
        # Get action logits and state value
        action_logits = self.actor(features)
        state_value = self.critic(features).squeeze(-1)
        
        return action_logits, state_value


class A2CAgent:
    def __init__(self, 
                 agent_id: str,
                 grid_size: int = 10, 
                 num_actions: int = 4, 
                 actor_lr: float = 1e-4,     # Increased from 2.5e-5
                 critic_lr: float = 2e-4,    # Increased from 5e-5
                 gamma: float = 0.99, 
                 epsilon_start: float = 1.0, 
                 epsilon_end: float = 0.05,  # Increased from 0.01 for more exploration
                 epsilon_decay: float = 0.997, # Slower decay (0.997^1000 ≈ 0.05)
                 entropy_coef: float = 0.02,  # Increased from 0.01 for better exploration
                 value_loss_coef: float = 0.5, 
                 max_grad_norm: float = 0.5, 
                 use_gae: bool = True, 
                 gae_lambda: float = 0.9,    # Reduced from 0.95 for less bias
                 value_loss_clip: float = 0.5, # Increased from 0.2 for more stable updates
                 grad_clip_value: float = 5.0, # Reduced from 10.0 for more stable training
                 checkpoint_dir: str = 'checkpoints',
                 batch_size: int = 64,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_llama: bool = False,
                 clip_param: float = 0.2,  # For PPO-style clipping
                 value_clip_param: float = 0.2,  # For value function clipping
                 normalize_advantages: bool = True,  # Whether to normalize advantages
                 clip_value_loss: bool = True,  # Whether to clip value function loss
                 normalize_returns: bool = True):    
        """
        A2C Agent for the treasure collection environment with improved exploration
        
        Args:
            agent_id: Unique identifier for the agent
            grid_size: Size of the grid environment
            num_actions: Number of possible actions
            actor_lr: Learning rate for the actor optimizer (default: 1e-4)
            critic_lr: Learning rate for the critic optimizer (default: 2e-4)
            gamma: Discount factor
            epsilon_start: Initial exploration rate (0.0-1.0)
            epsilon_end: Minimum exploration rate (0.0-1.0)
            epsilon_decay: Decay rate for epsilon
            entropy_coef: Coefficient for entropy regularization
            value_loss_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for gradient clipping
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: Lambda parameter for GAE
            value_loss_clip: Clip value loss to prevent large updates
            grad_clip_value: Clip gradients by value to prevent exploding gradients
            checkpoint_dir: Directory to save checkpoints
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.agent_id = agent_id
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.value_loss_clip = value_loss_clip
        self.grad_clip_value = grad_clip_value
        
        # Initialize policy network or LLaMA
        self.use_llama = use_llama
        
        if self.use_llama:
            # Initialize mock LLaMA
            self.policy = create_mock_llama(num_actions=num_actions, device=device)
            self.optimizer = None  # No optimizer needed for mock LLaMA
        else:
            # Initialize A2C policy
            self.policy = A2CPolicy(grid_size, num_actions).to(device)
            
            # Separate optimizers for actor and critic
            self.actor_params = []
            self.critic_params = []
            
            # Split parameters into actor and critic
            for name, param in self.policy.named_parameters():
                if 'critic' in name:
                    self.critic_params.append(param)
                else:
                    self.actor_params.append(param)
            
            # Separate optimizers with different learning rates and momentum
            self.actor_optimizer = optim.AdamW(
                self.actor_params,
                lr=actor_lr,
                weight_decay=1e-5,
                eps=1e-6,
                amsgrad=True,
                betas=(0.9, 0.98)  # Higher momentum for stability
            )
            
            self.critic_optimizer = optim.AdamW(
                self.critic_params,
                lr=critic_lr,
                weight_decay=1e-5,
                eps=1e-6,
                amsgrad=True,
                betas=(0.9, 0.98)  # Higher momentum for stability
            )
            
            # Warmup and decay learning rate schedule
            self.warmup_steps = 1000
            self.total_steps = 100000
            
            def lr_lambda(step):
                # Linear warmup and linear decay
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                # Linear decay to 10% of initial LR
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                return max(0.1, 1.0 - 0.9 * progress)
            
            # Learning rate schedulers with warmup
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(
                self.actor_optimizer, 
                lr_lambda=lr_lambda
            )
            
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(
                self.critic_optimizer,
                lr_lambda=lr_lambda
            )
        
        # Store batch size and checkpoint directory
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize reward normalizer
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0
        
        # For mixed precision training
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Track training statistics
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy = 0.0
        self.approx_kl = 0.0
        self.grad_norm = 0.0
        
        # Experience buffer with additional tracking
        self.memory = []
        self.episode_count = 0
        self.best_reward = -np.inf
        self.last_checkpoint_step = 0
        self.checkpoint_interval = 1000  # Save checkpoint every N steps
        
        # For monitoring and early stopping
        self.entropy_history = []
        self.kl_history = []
        self.reward_history = []
        self.patience = 20  # For early stopping
        self.best_metric = -np.inf
        self.bad_epochs = 0
        self.global_step = 0  # Track total number of training steps
        
        # Gradient clipping and logging
        self.grad_norms = defaultdict(list)
        self.value_losses = []
        self.policy_losses = []
        self.entropies = []
        
        # For gradient statistics
        self.total_grad_steps = 0
        
        # PPO-style parameters
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.normalize_advantages = normalize_advantages
        self.clip_value_loss = clip_value_loss
        self.normalize_returns = normalize_returns
        
        # For advantage normalization
        self.advantage_mean = 0.0
        self.advantage_var = 1.0
        self.advantage_count = 1e-4
        
        # For return normalization
        self.return_mean = 0.0
        self.return_var = 1.0
        self.return_count = 1e-4
        
    def get_observation(self, obs: Dict) -> torch.Tensor:
        """
        Convert environment observation to model input
        
        Args:
            obs: Observation dictionary from the environment
            
        Returns:
            torch.Tensor: Processed observation tensor of shape (4, grid_size, grid_size)
        """
        # Stack the observation channels
        grid = obs["grid"]  # (grid_size, grid_size)
        agent_pos = obs["agent_pos"]  # (2,)
        grid_size = grid.shape[0]  # Get the actual grid size
        
        # Create one-hot encoded position for current agent
        agent_pos_grid = np.zeros_like(grid)
        x, y = agent_pos
        
        # Ensure positions are within bounds
        if 0 <= x < grid_size and 0 <= y < grid_size:
            agent_pos_grid[x, y] = 1
        
        # Stack all channels (C, H, W) = (4, grid_size, grid_size)
        obs_tensor = np.stack([
            (grid == 1).astype(float),  # Other agents
            (grid == 2).astype(float),  # Treasures
            (grid == 3).astype(float),  # Base
            agent_pos_grid              # Current agent position
        ]).astype(np.float32)  # Ensure float32 dtype
        
        # Convert to tensor and ensure correct shape (C, H, W) = (4, grid_size, grid_size)
        obs_tensor = torch.from_numpy(obs_tensor)
        
        # Ensure the tensor is on the correct device
        return obs_tensor.to(self.device)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        if not hasattr(self, 'memory'):
            self.memory = []
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self, step: int):
        """
        Update exploration rate using cosine decay with warmup
        
        Args:
            step: Current training step (not episode)
        """
        # Linear warmup for first 10% of training, then cosine decay
        warmup_steps = 1000
        decay_steps = 10000
        
        if step < warmup_steps:
            # Linear warmup
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (step / warmup_steps)
        else:
            # Cosine decay after warmup
            progress = min(1.0, (step - warmup_steps) / (decay_steps - warmup_steps))
            self.epsilon = self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * (1 + math.cos(math.pi * progress))
            
        # Ensure epsilon doesn't go below minimum
        self.epsilon = max(self.epsilon_end, self.epsilon)
            
    def on_episode_end(self, episode_reward: float = None):
        """
        Called at the end of each episode to update exploration parameters and learning rate
        
        Args:
            episode_reward: The total reward obtained in the episode (for LR scheduling)
        """
        self.episode_count += 1
        self.update_epsilon(self.episode_count)
        
        # Update learning rate scheduler if enabled
        if self.actor_scheduler is not None and episode_reward is not None:
            self.actor_scheduler.step(episode_reward)
        
        # Update learning rate scheduler if enabled
        if self.critic_scheduler is not None and episode_reward is not None:
            self.critic_scheduler.step(-episode_reward)
        
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy distribution"""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()
        
    def select_action(self, obs: Dict, explore: bool = True) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Select an action using the current policy with epsilon-greedy exploration
        
        Args:
            obs: Observation dictionary from the environment
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Tuple of (selected action index, action log probability if not exploring else None)
        """
        # Epsilon-greedy exploration (only for A2C)
        if not self.use_llama and explore and random.random() < self.epsilon:
            action = random.randint(0, self.policy.num_actions - 1)
            return action, None
        
        # For LLaMA, we pass the observation directly
        if self.use_llama:
            with torch.no_grad():
                action_logits, _ = self.policy(obs)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs).item()
                return action, None
        
        # For A2C, process the observation as before
        obs_tensor = self.get_observation(obs)
        
        # Ensure we have the right shape: (batch_size=1, channels=4, height, width)
        if len(obs_tensor.shape) == 3:  # (C, H, W) -> add batch dim
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            # Forward pass through the network
            action_logits, _ = self.policy(obs_tensor)
            print(f"action_logits shape: {action_logits.shape}")  # Debug print
            print(f"action_logits: {action_logits}")  # Debug print
            
            # Remove batch dimension if present
            if len(action_logits.shape) > 1:
                action_logits = action_logits.squeeze(0)
                print(f"action_logits after squeeze: {action_logits.shape}")  # Debug print
            
            action_probs = F.softmax(action_logits, dim=-1)
            print(f"action_probs: {action_probs}")  # Debug print
            
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            print(f"action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")  # Debug print
            print(f"action: {action}")  # Debug print
            
            # Ensure we return a Python scalar
            if torch.is_tensor(action):
                if action.numel() == 1:
                    action = action.item()
                else:
                    # If we have multiple actions, take the first one (this might need to be adjusted)
                    action = action[0].item()
            
            print(f"Final action: {action}")  # Debug print
                
            if explore:
                return int(action), None
            else:
                log_prob = action_dist.log_prob(torch.tensor([action], device=self.device))
                return int(action), log_prob
    
    def _compute_returns_and_advantages(self, rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
        """Compute returns and advantages using GAE with value normalization."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Last advantage is just the TD error
        last_gae_lam = 0
        next_non_terminal = 1.0 - dones[-1]
        next_value = next_value.item()
        
        # Compute advantages using GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1].item()
                
            # Compute TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t].item()
            
            # Update GAE
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
            # Compute returns
            returns[t] = advantages[t] + values[t].item()
        
        # Clip advantages to prevent outliers
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        return returns, advantages
        
    def _normalize_advantages(self, advantages):
        """Normalize advantages to have zero mean and unit variance."""
        if self.normalize_advantages:
            # Update running statistics
            self._update_advantage_stats(advantages)
            
            # Normalize advantages
            advantages = (advantages - self.advantage_mean) / (torch.sqrt(self.advantage_var) + 1e-8)
            
            # Clip to prevent extreme values
            advantages = torch.clamp(advantages, -10.0, 10.0)
            
        return advantages
        
    def _normalize_returns(self, returns):
        """Normalize returns to have zero mean and unit variance."""
        if self.normalize_returns:
            # Update running statistics
            self._update_return_stats(returns)
            
            # Normalize returns
            returns = (returns - self.return_mean) / (torch.sqrt(self.return_var) + 1e-8)
            
            # Clip to prevent extreme values
            returns = torch.clamp(returns, -10.0, 10.0)
            
        return returns
        
    def _update_advantage_stats(self, advantages):
        """Update running statistics for advantages."""
        batch_mean = advantages.mean().item()
        batch_var = advantages.var().item()
        batch_count = advantages.numel()
        
        # Update running mean and variance
        delta = batch_mean - self.advantage_mean
        total_count = self.advantage_count + batch_count
        
        new_mean = self.advantage_mean + delta * batch_count / total_count
        m_a = self.advantage_var * self.advantage_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.advantage_count * batch_count / total_count
        new_var = M2 / total_count
        
        self.advantage_mean = new_mean
        self.advantage_var = new_var
        self.advantage_count = min(total_count, 1e6)  # Prevent infinite growth
        
    def _update_return_stats(self, returns):
        """Update running statistics for returns."""
        batch_mean = returns.mean().item()
        batch_var = returns.var().item()
        batch_count = returns.numel()
        
        # Update running mean and variance
        delta = batch_mean - self.return_mean
        total_count = self.return_count + batch_count
        
        new_mean = self.return_mean + delta * batch_count / total_count
        m_a = self.return_var * self.return_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.return_count * batch_count / total_count
        new_var = M2 / total_count
        
        self.return_mean = new_mean
        self.return_var = new_var
        self.return_count = min(total_count, 1e6)  # Prevent infinite growth
    
    def _clip_value_loss(self, values, old_values, returns):
        """Clip value function update to prevent large updates."""
        values_clipped = old_values + torch.clamp(values - old_values, -self.value_loss_clip, self.value_loss_clip)
        value_loss1 = (values - returns).pow(2)
        value_loss2 = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        return value_loss
    
    def _log_gradients(self, model):
        """Log gradient statistics for debugging."""
        stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                self.grad_norms[name].append(grad_norm)
                stats[f'grad_norm/{name}'] = grad_norm
        return stats
    
    def _check_nan_inf(self, tensor, name):
        """Check for NaN or Inf in tensor."""
        if torch.isnan(tensor).any():
            print(f"Warning: NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            print(f"Warning: Inf detected in {name}")
            return True
        return False
    
    def normalize_rewards(self, rewards):
        """Normalize rewards using running mean and std."""
        if not hasattr(self, 'ret_rms'):
            self.ret_rms = RunningMeanStd(shape=())
            self.ret = 0
            
        # Ensure rewards is a numpy array
        if torch.is_tensor(rewards):
            rewards_np = rewards.detach().cpu().numpy()
        else:
            rewards_np = np.array(rewards)
            
        # Update running statistics
        self.ret = self.ret * self.gamma + rewards_np
        self.ret_rms.update(np.array([self.ret]))
        
        # Normalize rewards and ensure same type as input
        normalized = rewards_np / (np.sqrt(self.ret_rms.var) + 1e-8)
        
        # Return same type as input
        if torch.is_tensor(rewards):
            return torch.tensor(normalized, device=rewards.device, dtype=rewards.dtype)
        return normalized
    
    def check_early_stopping(self, current_metric, min_delta=1e-4):
        """Check if training should stop early based on validation metric.
        
        Note: Early stopping is currently disabled to allow full training.
        We only track the best model based on the current metric.
        """
        # Track best model
        if current_metric > self.best_metric + min_delta:
            self.best_metric = current_metric
            return False, True  # (should_stop, is_best)
        return False, False  # Never stop early
    
    def update(self, batch):
        if not batch:
            return {}
            
        # Extract tensors from batch dictionary
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        old_values = batch['values'].to(self.device)
        
        # Normalize rewards
        with torch.no_grad():
            rewards = self.normalize_rewards(rewards)
        
        # Enable mixed precision training
        with autocast(enabled=torch.cuda.is_available()):
            # Get current policy and value
            logits, values = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            
            # Compute policy loss (with clipping)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Compute advantages with GAE
            with torch.no_grad():
                # Get next value
                _, next_value = self.policy(next_states[-1].unsqueeze(0))
                next_value = next_value.squeeze(-1)
                
                # Compute returns and advantages with GAE
                returns, advantages = self._compute_returns_and_advantages(
                    rewards, old_values, dones, next_value, 
                    gamma=self.gamma, gae_lambda=0.95
                )
                
                # Normalize advantages (zero mean, unit variance)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO-style policy loss with clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            values = values.squeeze(-1)
            value_loss = self._clip_value_loss(values, old_values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # KL divergence for monitoring
            with torch.no_grad():
                approx_kl = (old_log_probs - log_probs).mean().item()
                self.kl_history.append(approx_kl)
            
            # Total loss with entropy regularization
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Zero gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # Scale loss and backpropagate
        self.scaler.scale(loss).backward()
        
        # Unscale gradients
        self.scaler.unscale_(self.actor_optimizer)
        self.scaler.unscale_(self.critic_optimizer)
        
        # Log gradient statistics
        grad_stats = self._log_gradients(self.policy)
        
        # Gradient clipping by norm (prevents exploding gradients)
        actor_grad_norm = clip_grad_norm_(
            self.actor_params,
            max_norm=self.max_grad_norm,
            norm_type=2.0
        )
        
        critic_grad_norm = clip_grad_norm_(
            self.critic_params,
            max_norm=self.max_grad_norm,
            norm_type=2.0
        )
        
        # Gradient clipping by value (prevents exploding gradients)
        clip_grad_value_(self.policy.parameters(), clip_value=self.grad_clip_value)
        
        # Step optimizers with scaling
        self.scaler.step(self.actor_optimizer)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        
        # Get current policy and value
        logits, values = self.policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Compute policy loss (with clipping)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages  # PPO-style clipping
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss (with clipping)
        values = values.squeeze(-1)
        value_loss = self._clip_value_loss(values, old_values, returns)
        
        # Compute entropy and KL divergence
        entropy = dist.entropy().mean()
        approx_kl = (old_log_probs - log_probs).mean().item()
        
        # Total loss with entropy regularization
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Check for NaN/Inf in loss components
        self._check_nan_inf(policy_loss, 'policy_loss')
        self._check_nan_inf(value_loss, 'value_loss')
        self._check_nan_inf(entropy, 'entropy')
        
        # Zero gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # Backward pass with gradient scaling for mixed precision
        self.scaler.scale(loss).backward()
        
        # Log gradient statistics
        grad_stats = self._log_gradients(self.policy)
        
        # Gradient clipping by norm (prevents exploding gradients)
        actor_grad_norm = clip_grad_norm_(self.actor_params, max_norm=self.max_grad_norm, norm_type=2)
        critic_grad_norm = clip_grad_norm_(self.critic_params, max_norm=self.max_grad_norm, norm_type=2)
        clip_grad_value_(self.policy.parameters(), clip_value=self.grad_clip_value)
        
        # Update parameters with gradient scaling
        self.scaler.step(self.actor_optimizer)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        
        # Update learning rates (LambdaLR doesn't accept metrics parameter)
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # Track metrics
        self.total_grad_steps += 1
        self.entropy_history.append(entropy.item())
        self.reward_history.append(returns.mean().item())
        self.global_step += 1
        
        # Log metrics
        with torch.no_grad():
            # Compute explained variance
            y_true = returns.cpu().numpy()
            y_pred = values.detach().cpu().numpy()
            if len(y_true) > 1 and len(y_pred) > 1 and y_true.var() > 1e-8:
                explained_var = 1 - (y_true - y_pred).var() / y_true.var()
            else:
                explained_var = 0.0
            
            # Compute value function statistics
            value_pred_clipped = old_values + (values - old_values).clamp(-self.value_loss_clip, self.value_loss_clip)
            value_frac_clipped = float((value_pred_clipped != values).float().mean().item())
            
            # Compute ratio statistics
            ratio = (log_probs - old_log_probs).exp()
            ratio_clip_frac = float(((ratio < 0.8) | (ratio > 1.2)).float().mean().item())
            
            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'approx_kl': approx_kl,
                'actor_grad_norm': actor_grad_norm.item() if not torch.isinf(actor_grad_norm) and not torch.isnan(actor_grad_norm) else 0.0,
                'critic_grad_norm': critic_grad_norm.item() if not torch.isinf(critic_grad_norm) and not torch.isnan(critic_grad_norm) else 0.0,
                'epsilon': self.epsilon,
                'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
                'explained_variance': explained_var,
                'value_clip_frac': value_frac_clipped,
                'ratio_clip_frac': ratio_clip_frac,
                'value_mean': values.mean().item(),
                'value_var': values.var().item(),
                'adv_mean': advantages.mean().item(),
                'adv_var': advantages.var().item(),
                'return_mean': returns.mean().item(),
                'return_var': returns.var().item()
            }
        
        # Checkpointing
        if self.global_step - self.last_checkpoint_step >= self.checkpoint_interval:
            self.save_checkpoint(self.global_step)
            self.last_checkpoint_step = self.global_step
        
        # Check for early stopping and save best model
        should_stop, is_best = self.check_early_stopping(np.mean(self.reward_history[-10:]) if self.reward_history else 0)
        if is_best:
            self.save_checkpoint(self.global_step, is_best=True)
            
        return metrics
        
        # Update parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Update learning rate based on performance
        if hasattr(self, 'actor_scheduler') and self.actor_scheduler is not None:
            self.actor_scheduler.step(metrics=policy_loss.item())
        
        if hasattr(self, 'critic_scheduler') and self.critic_scheduler is not None:
            self.critic_scheduler.step(metrics=value_loss.item())
        
        # Update epsilon for exploration
        # Ensure learning rate doesn't get too small
        for param_group in self.actor_optimizer.param_groups + self.critic_optimizer.param_groups:
            if param_group['lr'] < 1e-6:
                param_group['lr'] = 1e-6
        
        # Update training statistics
        self.policy_loss = policy_loss.item()
        self.value_loss = value_loss.item()
        self.entropy = entropy.item()
        self.approx_kl = (old_log_probs - log_probs.detach()).mean().item()
        self.grad_norm = grad_norm
        
        # Log metrics with additional stability metrics
        with torch.no_grad():
            # Compute explained variance
            y_true = returns.cpu().numpy()
            y_pred = values.detach().cpu().numpy()
            if len(y_true) > 1 and len(y_pred) > 1 and y_true.var() > 1e-8:
                explained_var = 1 - (y_true - y_pred).var() / y_true.var()
            else:
                explained_var = 0.0
                
            # Compute value function statistics
            value_pred_clipped = old_values + (values - old_values).clamp(-self.value_loss_clip, self.value_loss_clip)
            value_frac_clipped = float((value_pred_clipped != values).float().mean().item())
            
            # Compute ratio statistics
            ratio = (log_probs - old_log_probs).exp()
            ratio_clip_frac = float(((ratio < 0.8) | (ratio > 1.2)).float().mean().item())
            
            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'grad_norm': total_norm.item() if not torch.isinf(total_norm) and not torch.isnan(total_norm) else 0.0,
                'epsilon': self.epsilon,
                'lr': self.optimizer.param_groups[0]['lr'],
                'explained_variance': explained_var,
                'value_clip_frac': value_frac_clipped,
                'ratio_clip_frac': ratio_clip_frac,
                'value_mean': values.mean().item(),
                'value_var': values.var().item(),
                'adv_mean': advantages.mean().item(),
                'adv_var': advantages.var().item(),
                'return_mean': returns.mean().item(),
                'return_var': returns.var().item()
            }
        
        # Clear memory
        self.memory = []
        self.global_step += 1
        
        return metrics
    
    def save(self, path: str):
        """Save the model weights to a file"""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict() if hasattr(self, 'actor_optimizer') else None,
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict() if hasattr(self, 'critic_optimizer') else None,
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict() if hasattr(self, 'actor_scheduler') else None,
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict() if hasattr(self, 'critic_scheduler') else None,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'entropy_history': self.entropy_history,
            'kl_history': self.kl_history,
            'reward_history': self.reward_history,
        }, path)

    def _update_network(self, states, actions, old_log_probs, returns, advantages, old_values):
        """Update the actor-critic network with a batch of experiences"""
        # Normalize advantages if enabled
        if self.normalize_advantages:
            advantages = self._normalize_advantages(advantages)
        
        # Normalize returns if enabled
        if self.normalize_returns:
            returns = self._normalize_returns(returns)
        
        # Get current policy and value
        logits, values = self.policy(states)
        dist = torch.distributions.Categorical(logits=logits)

        # Compute policy loss (with PPO-style clipping)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs.detach())
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss (with optional clipping)
        if self.clip_value_loss:
            # PPO-style value clipping
            value_pred_clipped = old_values + (values - old_values).clamp(-self.value_clip_param, self.value_clip_param)
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            # Standard MSE loss
            value_loss = 0.5 * F.mse_loss(values, returns)

        # Compute entropy bonus
        entropy = dist.entropy().mean()

        # Total loss with entropy regularization
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Backpropagate with gradient scaling for mixed precision
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()

        # Clip gradients by norm
        actor_grad_norm = clip_grad_norm_(self.actor_params, self.max_grad_norm)
        critic_grad_norm = clip_grad_norm_(self.critic_params, self.max_grad_norm)
        
        # Clip gradients by value to prevent exploding gradients
        clip_grad_value_(self.actor_params, self.grad_clip_value)
        clip_grad_value_(self.critic_params, self.grad_clip_value)
        
        # Step optimizers
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Update learning rate schedulers
        if hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.step()
        if hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            # Compute explained variance
            y_true = returns.cpu().numpy()
            y_pred = values.detach().cpu().numpy()
            if len(y_true) > 1 and len(y_pred) > 1 and y_true.var() > 1e-8:
                explained_var = 1 - (y_true - y_pred).var() / y_true.var()
            else:
                explained_var = 0.0

            # Compute value function statistics
            value_pred_clipped = old_values + (values - old_values).clamp(-self.value_loss_clip, self.value_loss_clip)
            value_frac_clipped = float((value_pred_clipped != values).float().mean().item())

            # Compute ratio statistics
            ratio = (log_probs - old_log_probs).exp()
            ratio_clip_frac = float(((ratio < 0.8) | (ratio > 1.2)).float().mean().item())

            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'approx_kl': (old_log_probs - log_probs.detach()).mean().item(),
                'actor_grad_norm': actor_grad_norm.item() if not torch.isinf(actor_grad_norm) and not torch.isnan(actor_grad_norm) else 0.0,
                'critic_grad_norm': critic_grad_norm.item() if not torch.isinf(critic_grad_norm) and not torch.isnan(critic_grad_norm) else 0.0,
                'epsilon': self.epsilon,
                'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
                'explained_variance': explained_var,
                'value_clip_frac': value_frac_clipped,
                'ratio_clip_frac': ratio_clip_frac,
                'value_mean': values.mean().item(),
                'value_var': values.var().item(),
                'adv_mean': advantages.mean().item(),
                'adv_var': advantages.var().item(),
                'return_mean': returns.mean().item(),
                'return_var': returns.var().item()
            }

        return metrics

    def save(self, path: str):
        """Save the model weights to a file"""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict() if hasattr(self, 'actor_optimizer') else None,
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict() if hasattr(self, 'critic_optimizer') else None,
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict() if hasattr(self, 'actor_scheduler') else None,
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict() if hasattr(self, 'critic_scheduler') else None,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'entropy_history': self.entropy_history,
            'kl_history': self.kl_history,
            'reward_history': self.reward_history,
            'ret_rms_mean': self.ret_rms.mean if hasattr(self, 'ret_rms') else 0.0,
            'ret_rms_var': self.ret_rms.var if hasattr(self, 'ret_rms') else 1.0,
            'ret': self.ret if hasattr(self, 'ret') else 0.0
        }, path)

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save a checkpoint of the agent's state.
        
        Args:
            step: Current training step
            is_best: Whether this is the best model so far
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pt')
        self.save(checkpoint_path)
        
        # If this is the best model, save it separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            shutil.copy2(checkpoint_path, best_path)
            print(f"New best model saved at step {step} with reward {self.best_reward:.2f}")
        
        return checkpoint_path

def create_agents(grid_size: int = 10, num_agents: int = 2, 
                  epsilon_start: float = 0.5, epsilon_end: float = 0.05, 
                  epsilon_decay: float = 0.995, batch_size: int = 64, 
                  use_llama: bool = False) -> Dict[str, A2CAgent]:
    """
    Create a dictionary of A2C agents with improved exploration
    
    Args:
        grid_size: Size of the grid environment
        num_agents: Number of agents to create
        epsilon_start: Initial exploration rate (0.0-1.0)
        epsilon_end: Minimum exploration rate (0.0-1.0)
        epsilon_decay: Rate at which to decay epsilon (0.0-1.0)
        batch_size: Batch size for training
        use_llama: Whether to use LLaMA model
        
    Returns:
        Dictionary mapping agent IDs to A2CAgent instances
    """
    agents = {}
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        agents[agent_id] = A2CAgent(
            agent_id=agent_id,
            grid_size=grid_size,
            num_actions=4,  # Up, Down, Left, Right
            actor_lr=2.5e-5,
            critic_lr=5e-5,
            gamma=0.99,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=0.995,  # Using decay rate instead of steps
            entropy_coef=0.01,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            use_gae=True,
            gae_lambda=0.95,
            value_loss_clip=0.2,
            grad_clip_value=10.0,
            checkpoint_dir='checkpoints',
            use_llama=use_llama,
            batch_size=batch_size
        )
    return agents
