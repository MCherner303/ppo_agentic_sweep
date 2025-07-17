import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple, defaultdict
import json
from datetime import datetime
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from our package structure
from environment.pettingzoo_env_wrapper import SimpleGridEnv
from agents.ppo_agent import PPOAgent, create_ppo_agents

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training with Hyperparameter Sweep Support')
    
    # Environment
    parser.add_argument('--grid-size', type=int, default=10, help='Size of the grid environment')
    parser.add_argument('--num-treasures', type=int, default=3, help='Number of treasures in the environment')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    
    # Training
    parser.add_argument('--num-episodes', type=int, default=10000, help='Total number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.8, help='Lambda for GAE')
    parser.add_argument('--clip-param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='Number of PPO epochs per update')
    parser.add_argument('--num-mini-batches', type=int, default=16, help='Number of mini-batches')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--actor-lr', type=float, default=1e-5, help='Learning rate for actor')
    parser.add_argument('--critic-lr', type=float, default=1e-5, help='Learning rate for critic')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    parser.add_argument('--num-steps', type=int, default=4096, help='Number of steps per update')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--eval-interval', type=int, default=100, help='Interval between evaluations (episodes)')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--eval-eps', type=float, default=0.0, help='Epsilon value for evaluation (0.0 for greedy)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--run-id', type=str, default=None, help='Unique identifier for the run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='ppo-gridworld', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity name')
    
    # Rendering
    parser.add_argument('--render-mode', type=str, default=None, help='Rendering mode')
    parser.add_argument('--render-interval', type=int, default=10, help='Render every N episodes')
    parser.add_argument('--save-video', action='store_true', help='Save video of episodes')
    
    return parser.parse_args()


class PPOTrainConfig:
    # Curriculum learning stages - Slower progression
    CURRICULUM_STAGES = [
        {'treasures': 1, 'episodes': 2000},  # Stage 1: 1 treasure (longer training)
        {'treasures': 2, 'episodes': 4000},  # Stage 2: 2 treasures
        {'treasures': 3, 'episodes': 6000},  # Stage 3: 3 treasures
        {'treasures': 4, 'episodes': 8000},  # Stage 4: 4 treasures
        {'treasures': 5, 'episodes': 10000}  # Stage 5: 5 treasures
    ]
    
    # Default logging and saving paths
    DEFAULT_LOG_DIR = "logs/ppo_stable"
    DEFAULT_SAVE_DIR = "saved_models/ppo_stable"
    SAVE_INTERVAL = 100     # Save models every N episodes
    LOG_INTERVAL = 10       # Print logs every N episodes
    DEBUG_MODE = True      # Enable additional debug logging
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, *args, **kwargs):
        # Handle different ways of passing arguments
        if not args and not kwargs:
            # No arguments provided, parse from command line
            parsed_args = parse_args()
            args_dict = vars(parsed_args)
        elif args and isinstance(args[0], dict):
            # If a dictionary is passed as the first argument
            args_dict = args[0]
        elif args and hasattr(args[0], '__dict__'):
            # If an argparse.Namespace object is passed
            args_dict = vars(args[0])
        else:
            # If keyword arguments are passed directly
            args_dict = kwargs
            
        # Store args for later use
        self.args = args_dict
            
        # Environment
        self.GRID_SIZE = args_dict.get('grid_size', 10)
        self.NUM_TREASURES = args_dict.get('num_treasures', 3)
        self.MAX_STEPS = args_dict.get('max_steps', 1000)
        
        # Training
        self.NUM_EPISODES = args_dict.get('num_episodes', 10000)
        self.GAMMA = args_dict.get('gamma', 0.99)
        self.GAE_LAMBDA = args_dict.get('gae_lambda', 0.8)
        self.CLIP_PARAM = args_dict.get('clip_param', 0.2)
        self.PPO_EPOCHS = args_dict.get('ppo_epochs', 4)
        self.NUM_MINI_BATCH = args_dict.get('num_mini_batch', 16)  # Updated to match function signature
        self.ENTROPY_COEF = args_dict.get('entropy_coef', 0.01)
        self.VALUE_LOSS_COEF = args_dict.get('value_loss_coef', 0.5)
        self.MAX_GRAD_NORM = args_dict.get('max_grad_norm', 0.5)
        self.ACTOR_LR = args_dict.get('actor_lr', 1e-5)
        self.CRITIC_LR = args_dict.get('critic_lr', 1e-5)
        self.BATCH_SIZE = args_dict.get('batch_size', 2048)
        self.NUM_STEPS = args_dict.get('num_steps', 4096)
        
        # Logging and saving
        self.LOG_DIR = args_dict.get('log_dir', self.DEFAULT_LOG_DIR)
        self.SAVE_DIR = args_dict.get('save_dir', self.DEFAULT_SAVE_DIR)
        self.RUN_ID = str(args_dict.get('run_id', f"run_{int(time.time())}"))  # Ensure RUN_ID is always a string
        self.SEED = args_dict.get('seed', 42)
        
        # Experiment tracking
        self.USE_WANDB = args_dict.get('use_wandb', False)
        self.WANDB_PROJECT = args_dict.get('wandb_project', 'ppo-gridworld')
        self.WANDB_ENTITY = args_dict.get('wandb_entity', None)
        
        # Rendering
        self.RENDER_MODE = args_dict.get('render_mode', None)
        self.RENDER_INTERVAL = args_dict.get('render_interval', 10)
        self.SAVE_VIDEO = args_dict.get('save_video', False)
        
        # Create necessary directories
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.LOG_DIR, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.LOG_DIR, "plots"), exist_ok=True)

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_environment()
        self.setup_agents()
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.start_time = time.time()
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_treasure_counts = []
        self.best_avg_reward = -np.inf
        
        # Track evaluation statistics
        self.eval_rewards = []
        self.eval_success_rates = []
        
        # For curriculum learning
        self.current_stage = 0
        self.stage_start_episode = 0
        
    def setup_directories(self):
        """Create necessary directories for logging and saving models."""
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.config.LOG_DIR, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.config.LOG_DIR, "plots"), exist_ok=True)
        
    def setup_environment(self, num_treasures=None):
        """Initialize the environment with optional treasure count for curriculum."""
        treasures = num_treasures if num_treasures is not None else self.config.NUM_TREASURES
        self.env = SimpleGridEnv(
            size=self.config.GRID_SIZE,
            num_treasures=treasures,
            max_steps=self.config.MAX_STEPS,
            render_mode=self.config.RENDER_MODE,
            debug=False  # Disable debug in training for better performance
        )
        
    def setup_agents(self):
        """Initialize PPO agents with the current configuration."""
        from agents.ppo_agent import create_ppo_agents
        
        # Create agents with current configuration
        self.agents = create_ppo_agents(
            grid_size=self.config.GRID_SIZE,
            num_agents=2,  # Fixed at 2 agents for this environment
            actor_lr=self.config.ACTOR_LR,
            critic_lr=self.config.CRITIC_LR,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            clip_param=self.config.CLIP_PARAM,
            ppo_epochs=self.config.PPO_EPOCHS,
            num_mini_batch=self.config.NUM_MINI_BATCH,  # Using the correct config parameter name
            entropy_coef=self.config.ENTROPY_COEF,
            value_loss_coef=self.config.VALUE_LOSS_COEF,
            max_grad_norm=self.config.MAX_GRAD_NORM,
            use_gae=True,  # Enable GAE by default
            use_clipped_value_loss=True,  # Enable clipped value loss by default
            reward_clip=10.0,  # Clip rewards
            value_clip=10.0,  # Clip value function updates
            device=self.config.DEVICE,
            checkpoint_dir=os.path.join(self.config.SAVE_DIR, self.config.RUN_ID)
        )
        
    def setup_logging(self):
        """Initialize logging."""
        self.log_file = os.path.join(self.config.LOG_DIR, f"training_{int(time.time())}.log")
        self.metrics_file = os.path.join(self.config.LOG_DIR, "metrics.json")
        self.metrics = {
            'episode': [],
            'reward': [],
            'length': [],
            'treasures': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_frac': [],
            'explained_variance': []
        }
        
    def log_metrics(self, episode, step, metrics):
        """Log training metrics to file and console."""
        # Update metrics
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(metrics.get('episode_reward', 0))
        self.metrics['length'].append(metrics.get('episode_length', 0))
        self.metrics['treasures'].append(metrics.get('treasures_collected', 0))
        
        # Process PPO metrics
        ppo_metrics = {}
        for key, value in metrics.items():
            # Skip non-PPO metrics
            if key in ['episode_reward', 'episode_length', 'treasures_collected']:
                continue
                
            # Add to PPO metrics
            if isinstance(value, dict):
                for k, v in value.items():
                    ppo_metrics[f"{key}/{k}"] = v
            else:
                ppo_metrics[key] = value
        
        # Log to console
        if episode % self.config.LOG_INTERVAL == 0:
            print(f"\n=== Episode {episode} ===")
            print(f"Steps: {step}, Treasures: {metrics.get('treasures_collected', 0)}/{getattr(self.env, 'num_treasures', 0)}")
            print(f"Reward: {metrics.get('episode_reward', 0):.2f}, Length: {metrics.get('episode_length', 0)}")
            
            if ppo_metrics:
                print("\nPPO Metrics:")
                # Sort metrics by key for better readability
                for k in sorted(ppo_metrics.keys()):
                    v = ppo_metrics[k]
                    # Only format numeric values with :.4f
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.6f}")
                    else:
                        print(f"  {k}: {v}")
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f)
            
        # Also save PPO metrics to a separate file
        ppo_metrics_file = os.path.join(self.config.LOG_DIR, "ppo_metrics.json")
        with open(ppo_metrics_file, 'a') as f:
            log_entry = {
                'episode': episode,
                'step': step,
                'timestamp': time.time(),
                'metrics': ppo_metrics
            }
            # Convert all float32 (and numpy) to Python float for JSON serialization
            for k, v in log_entry.items():
                if isinstance(v, (np.floating, np.float32, np.float64)):
                    log_entry[k] = float(v)
                elif isinstance(v, torch.Tensor):
                    log_entry[k] = float(v.item())
                elif isinstance(v, dict):
                    # Recursively convert nested dicts (e.g., 'metrics')
                    for mk, mv in v.items():
                        if isinstance(mv, (np.floating, np.float32, np.float64)):
                            v[mk] = float(mv)
                        elif isinstance(mv, torch.Tensor):
                            v[mk] = float(mv.item())
            f.write(json.dumps(log_entry) + '\n')
    
    def collect_experience(self):
        """Collect experience from the environment using current policy."""
        # Reset environment
        obs, _ = self.env.reset()
        dones = {agent_id: False for agent_id in self.agents}
        
        # Initialize storage
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents}
        episode_lengths = {agent_id: 0 for agent_id in self.agents}
        
        # Collect experience
        step = 0
        while step < self.config.NUM_STEPS and not all(dones.values()):
            actions = {}
            values = {}
            log_probs = {}
            
            # Get actions from all agents
            for agent_id, agent in self.agents.items():
                if dones[agent_id]:
                    continue
                    
                # Get action, log_prob, and value from agent
                action, log_prob, value = agent.select_action(obs[agent_id])
                actions[agent_id] = action
                values[agent_id] = value
                log_probs[agent_id] = log_prob
            
            # Convert actions to integers before passing to environment
            env_actions = {agent_id: int(action) for agent_id, action in actions.items()}
            
            # Step the environment
            next_obs, rewards, terminations, truncations, _ = self.env.step(env_actions)
            
            # Store experience
            for agent_id in self.agents:
                if dones[agent_id]:
                    continue
                    
                # Extract grid observation if it's a dict
                if isinstance(obs[agent_id], dict):
                    state = obs[agent_id]['grid']
                else:
                    state = obs[agent_id]
                    
                batch['states'].append(state)
                batch['actions'].append(actions[agent_id])
                batch['rewards'].append(float(rewards[agent_id]))
                batch['dones'].append(terminations[agent_id] or truncations[agent_id])
                batch['values'].append(float(values[agent_id]))
                batch['log_probs'].append(float(log_probs[agent_id]))
                
                # Update episode stats
                episode_rewards[agent_id] += float(rewards[agent_id])
                episode_lengths[agent_id] += 1
            
            # Update observations and check dones
            obs = next_obs
            dones = {agent_id: terminations[agent_id] or truncations[agent_id] 
                    for agent_id in self.agents}
            
            step += 1
        
        # Convert all batch data to tensors
        for k in ['actions', 'rewards', 'dones', 'values', 'log_probs']:
            batch[k] = torch.tensor(batch[k], device=self.config.DEVICE, dtype=torch.float32)
            
        # Process states to ensure correct shape for PPO [batch_size, C, H, W]
        states = []
        for s in batch['states']:
            # Convert to tensor
            if isinstance(s, np.ndarray):
                s_tensor = torch.from_numpy(s).to(device=self.config.DEVICE, dtype=torch.float32)
            else:
                s_tensor = torch.tensor(s, device=self.config.DEVICE, dtype=torch.float32)
            
            # Ensure correct shape [C, H, W]
            if len(s_tensor.shape) == 2:  # [H, W]
                s_tensor = s_tensor.unsqueeze(0)  # [1, H, W]
            
            # Ensure we have 4 channels (repeat if needed)
            if s_tensor.shape[0] == 1:
                s_tensor = s_tensor.repeat(4, 1, 1)  # [4, H, W]
            
            states.append(s_tensor)
        
        # Stack to create [batch_size, C, H, W]
        if states:  # Only stack if we have states
            batch['states'] = torch.stack(states).to(device=self.config.DEVICE, dtype=torch.float32)
        else:
            # Handle empty batch case (shouldn't happen but just in case)
            batch['states'] = torch.zeros((0, 4, self.config.GRID_SIZE, self.config.GRID_SIZE), 
                                        device=self.config.DEVICE, dtype=torch.float32)
        
        # Debug prints
        print(f"[DEBUG] Batch sizes - states: {len(batch['states'])}")
        print(f"[DEBUG] Final batch states shape: {batch['states'].shape}")
        print(f"[DEBUG] Batch states dtype: {batch['states'].dtype}")
        print(f"[DEBUG] Batch states device: {batch['states'].device}")
        
        # Calculate advantages and returns
        with torch.no_grad():
            # Get value estimates for the last state
            next_values = {}
            for agent_id, agent in self.agents.items():
                if not dones[agent_id]:
                    # Get the grid observation and ensure it has the right shape
                    obs_grid = torch.FloatTensor(obs[agent_id]['grid']).unsqueeze(0).to(self.config.DEVICE)
                    # Get the policy output (logits, value, _)
                    logits, next_value, _ = agent.policy(obs_grid)
                    next_values[agent_id] = next_value.item()
                else:
                    next_values[agent_id] = 0.0
            
            # Compute advantages and returns
            batch['returns'] = []
            batch['advantages'] = []
            
            # For simplicity, we'll use the same GAE for all agents
            # In a more complex setup, you might want to compute this per agent
            gae = 0
            returns = []
            values = batch['values'].cpu().numpy()
            rewards = batch['rewards'].cpu().numpy()
            dones = batch['dones'].cpu().numpy()
            
            next_value = np.mean(list(next_values.values()))
            values = np.append(values, next_value)
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    next_return = returns[0]
                
                delta = rewards[t] + self.config.GAMMA * next_return * next_non_terminal - values[t]
                gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * gae
                returns.insert(0, gae + values[t])
            
            # Convert to tensors and normalize advantages
            returns = torch.tensor(returns, device=self.config.DEVICE, dtype=torch.float32)
            advantages = returns - batch['values']
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            batch['returns'] = returns
            batch['advantages'] = advantages
        
        # Calculate average episode stats
        avg_episode_reward = np.mean(list(episode_rewards.values()))
        avg_episode_length = np.mean(list(episode_lengths.values()))
        
        # Get number of treasures collected (from environment if available)
        treasures_collected = 0
        if hasattr(self.env, 'collected_count'):
            treasures_collected = self.env.collected_count
        
        return batch, {
            'episode_reward': avg_episode_reward,
            'episode_length': avg_episode_length,
            'treasures_collected': treasures_collected
        }
    
    def update_agents(self, batch):
        """Update agents using PPO."""
        # Convert batch to tensors if not already
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        returns = batch['returns']
        advantages = batch['advantages']
        old_values = batch['values']
        
        # Initialize metrics aggregation
        all_metrics = {}
        
        # For each agent, update with the batch
        for agent_id, agent in self.agents.items():
            # Prepare batch for this agent
            agent_batch = {
                'states': states,
                'actions': actions,
                'old_log_probs': old_log_probs,
                'returns': returns,
                'advantages': advantages,
                'values': old_values
            }
            
            try:
                # Update agent and get metrics
                metrics = agent.ppo_update(agent_batch, self.config.CLIP_PARAM)
                
                # Log agent-specific metrics
                if metrics:
                    for key, value in metrics.items():
                        # Skip None values
                        if value is not None:
                            all_metrics[f'{agent_id}/{key}'] = value
                    
                    # Add debug print for metrics
                    print(f"[DEBUG] Agent {agent_id} metrics:")
                    for k, v in metrics.items():
                        if v is not None:
                            print(f"  {k}: {v:.6f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
                
                # Log learning rates
                if hasattr(agent, 'actor_optimizer'):
                    lr_actor = agent.actor_optimizer.param_groups[0]['lr']
                    all_metrics[f'{agent_id}/lr/actor'] = lr_actor
                    print(f"[DEBUG] Agent {agent_id} actor LR: {lr_actor:.6f}")
                
                if hasattr(agent, 'critic_optimizer'):
                    lr_critic = agent.critic_optimizer.param_groups[0]['lr']
                    all_metrics[f'{agent_id}/lr/critic'] = lr_critic
                    print(f"[DEBUG] Agent {agent_id} critic LR: {lr_critic:.6f}")
                
                # Log gradient norms if available
                if hasattr(agent, 'last_grad_norm'):
                    all_metrics[f'{agent_id}/grad_norm'] = agent.last_grad_norm
                    print(f"[DEBUG] Agent {agent_id} grad norm: {agent.last_grad_norm:.6f}")
                    
            except Exception as e:
                print(f"[ERROR] Error updating agent {agent_id}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Calculate mean metrics across all agents
        mean_metrics = {}
        if all_metrics:
            # Group metrics by type (e.g., 'loss/total', 'policy/ratio')
            metric_groups = {}
            for key, value in all_metrics.items():
                # Extract metric name without agent ID
                metric_name = '/'.join(key.split('/')[1:]) if '/' in key else key
                if metric_name not in metric_groups:
                    metric_groups[metric_name] = []
                metric_groups[metric_name].append(value)
            
            # Calculate mean for each metric group
            for metric_name, values in metric_groups.items():
                if values:
                    mean_metrics[f'mean/{metric_name}'] = np.mean(values)
        
        return {**all_metrics, **mean_metrics}
    
    def train(self):
        """Main training loop with curriculum learning."""
        print("Starting PPO training...")
        print(f"Device: {self.config.DEVICE}")
        print(f"Agents: {list(self.agents.keys())}")
        
        # Training loop
        for episode in range(1, self.config.NUM_EPISODES + 1):
            self.episode_count = episode
            
            # Update curriculum stage if needed
            self.update_curriculum(episode)
            
            try:
                # Collect experience
                batch, episode_metrics = self.collect_experience()
                
                # Skip update if batch is empty
                if not batch or len(batch.get('states', [])) == 0:
                    print(f"[WARNING] Empty batch in episode {episode}, skipping update")
                    continue
                    
                # Update agents
                ppo_metrics = self.update_agents(batch)
                
                # Log metrics
                metrics = {
                    'episode_reward': episode_metrics.get('episode_reward', 0),
                    'episode_length': episode_metrics.get('episode_length', 0),
                    'treasures_collected': episode_metrics.get('treasures_collected', 0),
                    **ppo_metrics  # Include all PPO metrics directly
                }
                
                self.log_metrics(episode, self.global_step, metrics)
                # Print summary line for sweep parsing
                print(f"=== Episode {episode} Reward: {metrics.get('episode_reward', 0):.2f}, Length: {metrics.get('episode_length', 0)}, Treasures: {metrics.get('treasures_collected', 0)}")
                
            except Exception as e:
                print(f"[ERROR] Error in training loop at episode {episode}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue  # Continue to next episode
            
            # Save models periodically
            if episode % self.config.SAVE_INTERVAL == 0:
                self.save_models(episode)
            
            # Update global step - get batch size from the states tensor
            self.global_step += batch['states'].size(0)
            
            # Early stopping if needed
            if self.check_early_stopping():
                print("Early stopping triggered.")
                break
        
        # Save final models
        self.save_models(self.episode_count, is_final=True)
        print("Training completed!")
    
    def update_curriculum(self, episode):
        """Update curriculum stage if needed."""
        for i, stage in enumerate(self.config.CURRICULUM_STAGES):
            if episode < stage['episodes']:
                if i != self.current_stage:
                    print(f"\n=== Advancing to curriculum stage {i+1}: {stage['treasures']} treasures ===")
                    self.current_stage = i
                    self.stage_start_episode = episode
                    self.setup_environment(num_treasures=stage['treasures'])
                break
    
    def check_early_stopping(self):
        """Check if we should stop training early."""
        # For now, we'll just use a simple check based on episode count
        # You might want to implement more sophisticated early stopping
        return False
    
    def save_models(self, episode, is_final=False):
        """Save agent models."""
        for agent_id, agent in self.agents.items():
            if is_final:
                path = os.path.join(self.config.SAVE_DIR, f"{agent_id}_final.pt")
            else:
                path = os.path.join(self.config.SAVE_DIR, f"{agent_id}_episode_{episode}.pt")
            
            agent.save(path)

        print(f"\nSaved models at episode {episode}")

def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set random seeds for reproducibility
    setup_seed(args.seed)

    # Initialize config with command-line arguments
    config = PPOTrainConfig(vars(args))

    # Initialize Weights & Biases if enabled
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config=vars(args),
            name=config.RUN_ID,
            sync_tensorboard=True
        )

    # Create and run trainer
    trainer = PPOTrainer(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...")
    finally:
        # Save final models and clean up
        trainer.save_models(trainer.episode_count, is_final=True)
        if config.USE_WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()
