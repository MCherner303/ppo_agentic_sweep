import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

# Import the policy network
from models.ppo_policy import PPOPolicy
from models.llama_mock import create_mock_llama

# Function to calculate explained variance
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

class PPOAgent:
    def __init__(self, 
                 agent_id: str,
                 grid_size: int = 10,
                 num_actions: int = 4,
                 actor_lr: float = 3e-4,  # Standard learning rate
                 critic_lr: float = 3e-4,  # Same as actor for simplicity
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,  # Standard GAE lambda
                 clip_param: float = 0.2,  # Standard PPO clip parameter
                 ppo_epochs: int = 4,
                 num_mini_batches: int = 4,
                 entropy_coef: float = 0.01,  # Standard entropy coefficient
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,  # Standard gradient clipping
                 max_grad_value: float = 10.0,  # Maximum gradient value for clipping
                 clip_value_loss: bool = True,  # Whether to clip the value function loss
                 use_llama: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: str = 'checkpoints',
                 mixed_precision: bool = True):  # Enable mixed precision training
        """
        PPO Agent implementation with support for both standard policy networks and LLaMA.
        
        Args:
            agent_id: Unique identifier for the agent
            grid_size: Size of the grid environment
            num_actions: Number of possible actions
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clip parameter (epsilon)
            ppo_epochs: Number of PPO epochs per update
            num_mini_batches: Number of mini-batches per PPO epoch
            entropy_coef: Coefficient for entropy bonus
            value_loss_coef: Coefficient for value function loss
            max_grad_norm: Maximum gradient norm for clipping
            use_llama: Whether to use LLaMA model
            device: Device to run the model on
            checkpoint_dir: Directory to save checkpoints
        """
        self.agent_id = agent_id
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize memory buffer and batch size
        self.memory = []
        self.batch_size = 64  # Default batch size, can be overridden by training script
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize policy network or LLaMA
        self.use_llama = use_llama
        
        if self.use_llama:
            self.policy = create_mock_llama(num_actions=num_actions, device=device)
            self.optimizer = None  # No optimizer needed for mock LLaMA
        else:
            # Initialize PPO policy network
            self.policy = PPOPolicy(
                grid_size=grid_size,
                num_actions=num_actions,
                hidden_size=256,  # Match the default from run_baseline.py
                use_lstm=True
            ).to(device)
            
            # Separate optimizers for actor and critic with improved stability settings
            actor_params = [p for n, p in self.policy.named_parameters() if 'critic' not in n]
            critic_params = [p for n, p in self.policy.named_parameters() if 'critic' in n]
            
            self.actor_optimizer = optim.AdamW(
                actor_params, 
                lr=actor_lr, 
                weight_decay=1e-4,
                eps=1e-5  # Add epsilon for numerical stability
            )
            
            self.critic_optimizer = optim.AdamW(
                critic_params, 
                lr=critic_lr, 
                weight_decay=1e-4,
                eps=1e-5  # Add epsilon for numerical stability
            )
            
            # More conservative learning rate schedule with cosine decay
            def lr_lambda(step):
                warmup_steps = 10000  # Longer warmup for stability
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                # Cosine decay after warmup
                decay_steps = 200000
                progress = min(1.0, (step - warmup_steps) / decay_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress)) * 0.1 + 0.1  # Decay to 10% of initial LR
            
            self.actor_scheduler = LambdaLR(
                self.actor_optimizer,
                lr_lambda=lr_lambda
            )
            
            self.critic_scheduler = LambdaLR(
                self.critic_optimizer,
                lr_lambda=lr_lambda
            )
        
        # Mixed precision training
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.mixed_precision)
        
        # Track training statistics
        self.global_step = 0
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_frac': [],
            'explained_variance': []
        }
        
        # For advantage normalization with running stats
        self.advantage_mean = 0.0
        self.advantage_var = 1.0
        self.advantage_count = 1e-4
        self.advantage_min = -10.0  # Clip advantages to prevent extreme values
        self.advantage_max = 10.0
        
        # For return normalization with running stats
        self.return_mean = 0.0
        self.return_var = 1.0
        self.return_count = 1e-4
        self.return_min = -20.0  # Clip returns to prevent extreme values
        self.return_max = 20.0
        
        # For reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4
        self.reward_min = -5.0  # Clip rewards to prevent extreme values
        self.reward_max = 5.0
        
        # For gradient clipping
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self.clip_value_loss = clip_value_loss
        
        # For compatibility with training loop (PPO doesn't use epsilon-greedy)
        self.epsilon = 0.0

    def get_observation(self, obs) -> torch.Tensor:
        """
        Convert environment observation to model input with shape [batch, channels, height, width].
        
        Args:
            obs: Observation from environment, can be dict with 'grid' key or tensor
            
        Returns:
            torch.Tensor: Observation tensor with shape [batch_size, 4, grid_size, grid_size]
        """
        # print("\n" + "="*50)
        # print("[DEBUG] get_observation called")
        # print(f"Input type: {type(obs)}")
        if hasattr(obs, 'shape'):
            # print(f"Input shape: {obs.shape}")
            pass
        elif isinstance(obs, dict):
            # print(f"Dict keys: {list(obs.keys())}")
            for k, v in obs.items():
                if hasattr(v, 'shape'):
                    # print(f"  - {k} shape: {v.shape}")
                    pass
                else:
                    # print(f"  - {k} type: {type(v)}")
                    pass
        
        # Extract grid from observation dict if needed
        if isinstance(obs, dict) and 'grid' in obs:
            # print("Extracting 'grid' from observation dict")
            grid = obs['grid']
        else:
            # print("Using observation directly")
            grid = obs
            pass
        
        # Debug grid before conversion
        # print("\n[DEBUG] Grid before conversion:")
        # print(f"Type: {type(grid)}")
        if hasattr(grid, 'shape'):
            # print(f"Shape: {grid.shape}")
            pass
        if hasattr(grid, 'dtype'):
            # print(f"Dtype: {grid.dtype}")
            pass
        
        # Convert to tensor if not already
        if not isinstance(grid, torch.Tensor):
            # print("Converting to tensor...")
            try:
                grid = torch.from_numpy(np.asarray(grid, dtype=np.float32))
                # print(f"Converted to tensor, new shape: {grid.shape}")
            except Exception as e:
                # print(f"Error converting to tensor: {e}")
                # print(f"Grid type: {type(grid)}")
                # print(f"Grid content: {grid}")
                raise
        
        # Debug tensor properties
        # print("\n[DEBUG] Tensor properties:")
        # print(f"Shape: {grid.shape}")
        # print(f"Dtype: {grid.dtype}")
        # print(f"Device: {grid.device if hasattr(grid, 'device') else 'N/A'}")
        
        # Handle different input shapes
        # print("\n[DEBUG] Processing tensor shape:")
        if len(grid.shape) == 3:
            # [C, H, W] -> [1, C, H, W]
            # print(f"Adding batch dimension to shape {grid.shape}")
            grid = grid.unsqueeze(0)
        elif len(grid.shape) == 5:
            # [B, 1, C, H, W] -> [B, C, H, W]
            # print(f"Removing extra dimension from shape {grid.shape}")
            grid = grid.squeeze(1)
        
        # print(f"Shape after processing: {grid.shape}")
        
        # Ensure we have the correct number of channels (4)
        if len(grid.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape {grid.shape}")
            
        if grid.size(1) != 4:
            raise ValueError(f"Expected 4 channels in observation, got {grid.size(1)}")
        
        # Move to device and ensure correct dtype
        # print("\n[DEBUG] Final processing:")
        if not grid.is_floating_point():
            # print("Converting to float32")
            grid = grid.float()
            
        if hasattr(self, 'device') and str(grid.device) != self.device:
            # print(f"Moving tensor to device {self.device}")
            grid = grid.to(device=self.device)
        
        # Final debug output
        # print("\n[DEBUG] Final observation:")
        # print(f"Shape: {grid.shape}")
        # print(f"Dtype: {grid.dtype}")
        # print(f"Device: {grid.device}")
        # print("="*50 + "\n")
        
        return grid

    def select_action(self, obs: Dict, deterministic: bool = False):
        """
        Select an action using the current policy.
        
        Args:
            obs: Observation dictionary from the environment
            deterministic: If True, use deterministic action selection
            
        Returns:
            Tuple of (action, action_log_prob, value_estimate)
        """
        with torch.no_grad():
            # Get observation tensor with shape [1, C, H, W]
            obs_tensor = self.get_observation(obs)
            
            # Debug print the observation shape before policy
            # print(f"[DEBUG] select_action - obs_tensor shape: {obs_tensor.shape}")
            
            # Get action and value from policy
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                                 enabled=torch.cuda.is_available() and not self.use_llama):
                # Get action distribution and value from policy
                # Note: get_observation already returns [1, C, H, W] for a single observation
                output = self.policy(obs_tensor)
                
                # Debug print the output shapes
                # print(f"[DEBUG] select_action - logits shape: {output['logits'].shape}")
                # print(f"[DEBUG] select_action - value shape: {output['value'].shape if isinstance(output['value'], torch.Tensor) else 'N/A'}")
                
                # Create action distribution
                dist = torch.distributions.Categorical(logits=output['logits'])
                
                if deterministic:
                    action = torch.argmax(output['logits'], dim=-1)
                else:
                    action = dist.sample()
                
                # Compute log probability of the action
                log_prob = dist.log_prob(action)
                
                # Debug print the action and log prob
                # print(f"[DEBUG] select_action - action: {action}, log_prob: {log_prob}")
                
                return action.item(), log_prob.item(), output['value'].item()

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) with improved numerical stability.
        
        Args:
            next_value: Tensor of shape [1] representing the value of the next state
            rewards: Tensor of shape [T] containing rewards for each timestep
            masks: Tensor of shape [T] containing masks (1 for not done, 0 for done)
            values: Tensor of shape [T] containing value estimates for each timestep
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (returns, advantages)
        """
        advantages = torch.zeros_like(rewards, device=rewards.device)
        gae = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            gae = delta + gamma * gae_lambda * masks[t] * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        if len(advantages) > 1:  # Only normalize if we have more than one sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return returns, advantages
        advantages = []
        gae = 0
        next_value = next_value.detach()

        # Iterate backwards through time steps
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - (1.0 - masks[-1])
                next_values = next_value
            else:
                next_non_terminal = 1.0 - (1.0 - masks[t + 1])
                next_values = values[t + 1]

            # Clip value targets to prevent extreme values
            if hasattr(self, 'value_clip') and self.value_clip > 0:
                next_values = torch.clamp(next_values, -self.value_clip, self.value_clip)

            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        # Convert to tensor
        advantages = torch.stack(advantages)

        # Clip advantages to prevent extreme values
        if hasattr(self, 'value_clip') and self.value_clip > 0:
            advantages = torch.clamp(advantages, -self.value_clip, self.value_clip)

        # Compute returns
        returns = advantages + values

        # Normalize advantages (with numerical stability)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    def _update_network(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update network parameters using PPO loss with gradient clipping and mixed precision."""
        # Initialize metrics tracking with default values
        metrics = {
            'loss/total': 0.0,
            'loss/policy': 0.0,
            'loss/value': 0.0,
            'loss/entropy': 0.0,
            'policy/ratio': 0.0,
            'policy/clip_frac': 0.0,
            'value/explained_variance': 0.0,
            'lr/actor': self.actor_optimizer.param_groups[0]['lr'] if hasattr(self, 'actor_optimizer') else 0.0,
            'lr/critic': self.critic_optimizer.param_groups[0]['lr'] if hasattr(self, 'critic_optimizer') else 0.0,
            'num_updates': 0,
            'error': None
        }
        
        try:
            # Debug: Print batch keys for verification
            # print(f"[DEBUG] Batch keys: {list(batch.keys())}")
            
            # Unpack batch with error checking
            required_keys = ['states', 'actions', 'old_log_probs', 'old_values', 'returns', 'advantages']
            for key in required_keys:
                if key not in batch:
                    raise KeyError(f"Missing required key in batch: {key}")
            
            states = batch['states']
            actions = batch['actions']
            old_log_probs = batch['old_log_probs']
            old_values = batch['old_values']
            returns = batch['returns']
            advantages = batch['advantages']
            
            # Debug: Print batch content for verification
            # print(f"[DEBUG] Batch content:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # print(f"  {k}: {v.shape} {v.dtype} {v.device}")
                    pass
                else:
                    # print(f"  {k}: {type(v)}")
                    pass
            
            # Create indices for minibatch sampling
            batch_size = states.size(0)
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            # Calculate batch sizes
            mini_batch_size = batch_size // self.num_mini_batches
            if mini_batch_size == 0:
                mini_batch_size = 1
                self.num_mini_batches = batch_size
            
            # Process minibatches
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                if end > batch_size:
                    end = batch_size
                
                # Get minibatch indices
                idx = indices[start:end]
                
                try:
                    # Forward pass with mixed precision
                    with autocast(dtype=torch.float16, enabled=self.mixed_precision):
                        # Get current policy outputs
                        policy_outputs = self.policy(states[idx])
                        
                        # Check for valid policy outputs
                        if 'logits' not in policy_outputs or 'value' not in policy_outputs:
                            raise ValueError("Policy outputs must contain 'logits' and 'value' keys")
                        
                        # Compute action distribution and sample
                        dist = Categorical(logits=policy_outputs['logits'])
                        new_log_probs = dist.log_prob(actions[idx].squeeze(-1)).unsqueeze(-1)
                        entropy = dist.entropy().mean()
                        
                        # Compute ratio (pi_theta / pi_theta_old)
                        ratio = (new_log_probs - old_log_probs[idx]).exp()
                        
                        # Clipped surrogate objective
                        surr1 = ratio * advantages[idx]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[idx]
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss with optional clipping
                        values = policy_outputs['value'].squeeze(-1)
                        returns_batch = returns[idx]
                        
                        if self.clip_value_loss:
                            values_clipped = old_values[idx] + torch.clamp(
                                values - old_values[idx], -self.clip_param, self.clip_param
                            )
                            value_loss1 = (values - returns_batch).pow(2)
                            value_loss2 = (values_clipped - returns_batch).pow(2)
                            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                        else:
                            value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                        
                        # Total loss with entropy bonus
                        loss = (policy_loss 
                              + self.value_loss_coef * value_loss 
                              - self.entropy_coef * entropy)

                        # === NaN/Inf Guard ===
                        # Check all relevant losses/metrics for NaN or Inf and abort if found
                        to_check = {
                            'policy_loss': policy_loss,
                            'value_loss': value_loss,
                            'entropy': entropy,
                            'loss': loss,
                        }
                        for name, tensor in to_check.items():
                            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                msg = f"[ABORT] NaN/Inf detected in {name} during PPO update. Aborting run."
                                print(msg)
                                raise RuntimeError(msg)
                    
                    # Backward pass with gradient scaling for mixed precision
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    
                    # Scale loss and call backward
                    self.scaler.scale(loss).backward()
                    
                    # Unscale the gradients before clipping
                    self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm
                    )
                    
                    # Update parameters
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                    
                    # Update learning rates
                    if hasattr(self, 'actor_scheduler') and hasattr(self, 'critic_scheduler'):
                        self.actor_scheduler.step()
                        self.critic_scheduler.step()
                    
                    # Calculate metrics
                    with torch.no_grad():
                        # Calculate clip fraction
                        clipped = (ratio < (1.0 - self.clip_param)) | (ratio > (1.0 + self.clip_param))
                        clip_frac = torch.as_tensor(
                            clipped, dtype=torch.float32).mean().item()
                        
                        # Calculate explained variance
                        y_pred = values.detach().cpu().numpy()
                        y_true = returns_batch.detach().cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
                        
                        # Update metrics
                        metrics['loss/total'] += loss.item()
                        metrics['loss/policy'] += policy_loss.item()
                        metrics['loss/value'] += value_loss.item()
                        metrics['loss/entropy'] += entropy.item()
                        metrics['policy/ratio'] += ratio.mean().item()
                        metrics['policy/clip_frac'] += clip_frac
                        metrics['value/explained_variance'] += explained_var
                        metrics['num_updates'] += 1
                        
                        # Log detailed metrics periodically
                        if self.global_step % 10 == 0:
                            # print(f"[DEBUG] Step {self.global_step}:")
                            # print(f"  Policy loss: {policy_loss.item():.4f}")
                            # print(f"  Value loss: {value_loss.item():.4f}")
                            # print(f"  Entropy: {entropy.item():.4f}")
                            # print(f"  Explained variance: {explained_var:.4f}")
                            # print(f"  Clip fraction: {clip_frac:.4f}")
                            # print(f"  Ratio mean: {ratio.mean().item():.4f} ± {ratio.std().item():.4f}")
                            # print(f"  Advantages mean: {advantages[idx].mean().item():.4f} ± {advantages[idx].std().item():.4f}")
                            # print(f"  Returns mean: {returns_batch.mean().item():.4f} ± {returns_batch.std().item():.4f}")
                            # print(f"  Values mean: {values.mean().item():.4f} ± {values.std().item():.4f}")
                            pass
                    
                    # Update global step counter
                    self.global_step += 1
                    
                except Exception as e:
                    # print(f"[ERROR] Error in minibatch processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Average metrics over all minibatches
            if metrics['num_updates'] > 0:
                for k in ['loss/total', 'loss/policy', 'loss/value', 'loss/entropy', 
                         'policy/ratio', 'policy/clip_frac', 'value/explained_variance']:
                    if k in metrics and isinstance(metrics[k], (int, float)):
                        metrics[k] /= metrics['num_updates']
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error in _update_network: {str(e)}"
            # print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            metrics['error'] = error_msg
            # Calculate final metrics
            if metrics['num_updates'] > 0:
                for k in ['loss/total', 'loss/policy', 'loss/value', 'loss/entropy', 
                         'policy/ratio', 'policy/clip_frac', 'value/explained_variance']:
                    if k in metrics and isinstance(metrics[k], (int, float)):
                        metrics[k] /= metrics['num_updates']
            
            # Log metrics to console for debugging
            if self.global_step % 10 == 0:
                # print("\n=== Training Metrics ===")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        # print(f"{k}: {v:.6f}")
                        pass
                # print("======================\n")
            
            return metrics
        except Exception as e:
            error_msg = f"Error in ppo_update: {str(e)}"
            # print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                'loss/total': 0.0,
                'loss/policy': 0.0,
                'loss/value': 0.0,
                'loss/entropy': 0.0,
                'policy/ratio': 0.0,
                'policy/clip_frac': 0.0,
                'value/explained_variance': 0.0,
                'lr/actor': 0.0,
                'lr/critic': 0.0,
                'num_updates': 0,
                'returns_std': 0.0,
                'error': error_msg
            }
    
    def ppo_update(self, batch: Dict[str, torch.Tensor], clip_param: float = 0.2) -> Dict[str, float]:
        """
        Perform a PPO update step using the provided batch of experiences.
        
        Args:
            batch: Dictionary containing:
                - states: batch of states
                - actions: batch of actions
                - old_log_probs: log probabilities from the old policy
                - old_values: value estimates from the old policy
                - returns: computed returns
                - advantages: computed advantages
            clip_param: PPO clip parameter (epsilon)
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'loss/total': 0.0,
            'loss/policy': 0.0,
            'loss/value': 0.0,
            'loss/entropy': 0.0,
            'policy/ratio': 0.0,
            'policy/clip_frac': 0.0,
            'value/explained_variance': 0.0,
            'lr/actor': self.actor_optimizer.param_groups[0]['lr'] if hasattr(self, 'actor_optimizer') else 0.0,
            'lr/critic': self.critic_optimizer.param_groups[0]['lr'] if hasattr(self, 'critic_optimizer') else 0.0,
            'num_updates': 0
        }
        """
        Perform a PPO update step using the provided batch of experiences.
        
        Args:
            batch: Dictionary containing:
                - states: batch of states
                - actions: batch of actions
                - old_log_probs: log probabilities from the old policy
                - returns: computed returns
                - advantages: computed advantages
                - values: value estimates from the old policy
            clip_param: PPO clip parameter (epsilon)
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Convert batch to tensors if they aren't already
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            old_log_probs = batch['old_log_probs'].to(self.device)
            returns = batch['returns'].to(self.device)
            advantages = batch['advantages'].to(self.device)
            old_values = batch['values'].to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update the network using PPO
            metrics = self._update_network({
                'states': states,
                'actions': actions,
                'old_log_probs': old_log_probs,
                'returns': returns,
                'advantages': advantages,
                'old_values': old_values
            })
            
            return metrics
            
        except Exception as e:
            # print(f"[ERROR] in ppo_update: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def save(self, path: str) -> None:
        """
        Save the agent's model and optimizer states to the specified path.
        
        Args:
            path: Directory path where to save the model
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save policy state
            policy_path = os.path.join(path, f"{self.agent_id}_policy.pth")
            torch.save(self.policy.state_dict(), policy_path)
            
            # Save optimizers if they exist
            if hasattr(self, 'actor_optimizer') and self.actor_optimizer is not None:
                actor_optim_path = os.path.join(path, f"{self.agent_id}_actor_optimizer.pth")
                torch.save(self.actor_optimizer.state_dict(), actor_optim_path)
                
            if hasattr(self, 'critic_optimizer') and self.critic_optimizer is not None:
                critic_optim_path = os.path.join(path, f"{self.agent_id}_critic_optimizer.pth")
                torch.save(self.critic_optimizer.state_dict(), critic_optim_path)
                
            # Save schedulers if they exist
            if hasattr(self, 'actor_scheduler') and self.actor_scheduler is not None:
                actor_sched_path = os.path.join(path, f"{self.agent_id}_actor_scheduler.pth")
                torch.save(self.actor_scheduler.state_dict(), actor_sched_path)
                
            if hasattr(self, 'critic_scheduler') and self.critic_scheduler is not None:
                critic_sched_path = os.path.join(path, f"{self.agent_id}_critic_scheduler.pth")
                torch.save(self.critic_scheduler.state_dict(), critic_sched_path)
                
            # print(f"[INFO] Saved model for agent {self.agent_id} to {path}")
            
        except Exception as e:
            # print(f"[ERROR] Failed to save model for agent {self.agent_id}: {str(e)}")
            raise

def create_ppo_agents(grid_size: int = 10, 
                     num_agents: int = 2,
                     actor_lr: float = 3e-4,
                     critic_lr: float = 3e-4,
                     gamma: float = 0.99,
                     gae_lambda: float = 0.95,
                     clip_param: float = 0.2,
                     ppo_epochs: int = 4,
                     num_mini_batch: int = 4,
                     entropy_coef: float = 0.01,
                     value_loss_coef: float = 0.5,
                     max_grad_norm: float = 0.5,
                     use_gae: bool = True,
                     use_clipped_value_loss: bool = True,
                     reward_clip: float = 10.0,  # Clip rewards to this value
                     value_clip: float = 10.0,  # Clip value function updates
                     device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     checkpoint_dir: str = 'checkpoints') -> Dict[str, PPOAgent]:
    """
    Create a dictionary of PPO agents.
    
    Args:
        grid_size: Size of the grid environment
        num_agents: Number of agents to create
        actor_lr: Learning rate for the actor
        critic_lr: Learning rate for the critic
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_param: PPO clip parameter (epsilon)
        ppo_epochs: Number of PPO epochs per update
        num_mini_batches: Number of mini-batches per PPO epoch
        entropy_coef: Coefficient for entropy bonus
        value_loss_coef: Coefficient for value function loss
        max_grad_norm: Maximum gradient norm for clipping
        use_llama: Whether to use LLaMA model
        device: Device to run the model on
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Dictionary mapping agent IDs to PPOAgent instances
    """
    agents = {}
    for i in range(num_agents):
        agent_id = f'agent_{i}'
        agent_checkpoint_dir = os.path.join(checkpoint_dir, agent_id)
        os.makedirs(agent_checkpoint_dir, exist_ok=True)
        
        agents[agent_id] = PPOAgent(
            agent_id=agent_id,
            grid_size=grid_size,
            num_actions=4,  # 4 directions (up, down, left, right)
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_param=clip_param,
            ppo_epochs=ppo_epochs,
            num_mini_batches=num_mini_batch,  # Using the correct parameter name (plural for PPOAgent)
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            # Removed use_llama parameter as it's not supported
            device=device,
            checkpoint_dir=agent_checkpoint_dir
        )
    
    return agents
