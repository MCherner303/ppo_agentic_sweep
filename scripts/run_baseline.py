#!/usr/bin/env python3
"""Baseline training run for the multi-agent treasure collection environment.

This script runs a training loop to establish a baseline for learning performance.
It tracks and logs key metrics to evaluate if the agents are learning effectively.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.pettingzoo_env_wrapper import SimpleGridEnv
from agents.ppo_agent import create_ppo_agents as create_agents
from utils.logger import setup_logging

class BaselineTrainer:
    def __init__(self, config):
        """Initialize the baseline trainer with configuration."""
        self.config = config
        self.setup_directories()
        self.setup_logging()  # Setup logging before agents to ensure logger is available
        self.setup_environment()
        self.setup_agents()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.treasures_collected = []
        self.start_time = time.time()
        
        self.logger.info("Baseline trainer initialized")

    def setup_directories(self):
        """Create necessary directories for logging and saving models."""
        self.log_dir = Path(self.config.get('log_dir', 'logs/baseline'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamped directory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / f'run_{timestamp}'
        self.run_dir.mkdir()
        
        # Save config
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_environment(self):
        """Initialize the environment with configured settings."""
        self.env = SimpleGridEnv(
            size=self.config['grid_size'],
            num_treasures=self.config['num_treasures'],
            max_steps=self.config['max_steps'],
            early_stop_threshold=1.0,  # Disable early stopping by setting threshold to 1.0
            debug=self.config.get('debug', False)
        )
        self.agents = self.env.possible_agents

    def setup_agents(self):
        """Initialize the agents with the specified configuration."""
        # Initialize agents list
        self.agents = [f'agent_{i}' for i in range(self.config['num_agents'])]
        
        self.logger.info(f"Initializing {self.config['num_agents']} agents")
        self.agents_dict = create_agents(
            grid_size=self.config['grid_size'],
            num_agents=self.config['num_agents'],
            actor_lr=self.config.get('actor_lr', 3e-4),
            critic_lr=self.config.get('critic_lr', 3e-4),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_param=self.config.get('clip_param', 0.2),
            ppo_epochs=self.config.get('ppo_epochs', 4),
            num_mini_batches=self.config.get('num_mini_batches', 4),
            entropy_coef=self.config.get('entropy_coef', 0.01),
            value_loss_coef=self.config.get('value_loss_coef', 0.5),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            use_llama=self.config.get('use_llama', False),
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            checkpoint_dir=str(self.run_dir / 'checkpoints')
        )
        self.logger.info(f"Agents initialized: {list(self.agents_dict.keys())}")
        if self.config.get('use_llama', False):
            self.logger.info("Using LLaMA model for decision making")
        else:
            self.logger.info("Using PPO policy for decision making")
        
        # Set device for all agents
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        for agent in self.agents_dict.values():
            agent.device = device
            agent.policy = agent.policy.to(device)
            
        self.logger.info(f"Initialized {len(self.agents_dict)} PPO agents")
        self.logger.info(f"Using device: {device}")
        self.logger.info(f"PPO configuration: clip_param={self.config.get('clip_param', 0.2)}, ppo_epochs={self.config.get('ppo_epochs', 4)}, num_mini_batches={self.config.get('num_mini_batches', 4)}")
        self.logger.info(f"Learning rates: actor={self.config.get('actor_lr', 3e-4)}, critic={self.config.get('critic_lr', 3e-4)}")

    def setup_logging(self):
        """Initialize logging."""
        self.logger = setup_logging(self.run_dir)
        
        # Initialize TensorBoard writer if available
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.run_dir / 'tensorboard')
            self.logger.info("TensorBoard logging enabled")
        except ImportError:
            self.logger.warning("TensorBoard not available, logging to console and file only")
        
        self.logger.info("Starting baseline training run")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

    def log_metrics(self, episode, episode_rewards, step_count, treasures_collected):
        """Log training metrics with detailed treasure collection information."""
        # Convert episode_rewards to a list of values if it's a dictionary
        rewards_list = list(episode_rewards.values()) if isinstance(episode_rewards, dict) else episode_rewards
        
        # Calculate metrics
        mean_reward = np.mean(rewards_list) if rewards_list else 0
        max_reward = np.max(rewards_list) if rewards_list else 0
        min_reward = np.min(rewards_list) if rewards_list else 0
        
        # Get number of agents
        num_agents = len(episode_rewards) if isinstance(episode_rewards, dict) else 1
        
        # Get remaining treasures if available
        remaining_treasures = 0
        if hasattr(self.env, 'treasures'):
            remaining_treasures = len(self.env.treasures)
        elif hasattr(self.env, 'num_treasures'):
            remaining_treasures = self.env.num_treasures
            
        # Calculate treasure collection rate
        total_treasures = treasures_collected + remaining_treasures
        collection_rate = (treasures_collected / total_treasures * 100) if total_treasures > 0 else 0
        
        # Log to console with color for better visibility
        print(f"\n=== Episode {episode + 1:4d}/{self.config['num_episodes']} ===")
        print(f"Steps: {step_count:3d} | Agents: {num_agents}")
        print(f"Rewards - Mean: {mean_reward:6.2f} | Max: {max_reward:6.2f} | Min: {min_reward:6.2f}")
        print(f"Treasures - Collected: {treasures_collected:2d} | Remaining: {remaining_treasures:2d} "
              f"| Rate: {collection_rate:5.1f}%")
        
        # Log individual agent rewards if available
        if isinstance(episode_rewards, dict):
            print("Agent Rewards:")
            for agent_id, reward in episode_rewards.items():
                print(f"  {agent_id}: {reward:6.2f}")
        
        # Log to file (without color codes)
        log_message = (
            f"Episode {episode + 1}/{self.config['num_episodes']}, "
            f"Steps: {step_count}, Agents: {num_agents}, "
            f"Mean Reward: {mean_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}, "
            f"Treasures: {treasures_collected}/{total_treasures} ({collection_rate:.1f}%)"
        )
        self.logger.info(log_message)
        
        # Log to TensorBoard with more detailed metrics
        if self.writer is not None:
            self.writer.add_scalar('Reward/mean', mean_reward, episode)
            self.writer.add_scalar('Reward/max', max_reward, episode)
            self.writer.add_scalar('Reward/min', min_reward, episode)
            self.writer.add_scalar('Stats/agents', num_agents, episode)
            self.writer.add_scalar('Stats/steps', step_count, episode)
            self.writer.add_scalar('Treasures/collected', treasures_collected, episode)
            self.writer.add_scalar('Treasures/remaining', remaining_treasures, episode)
            self.writer.add_scalar('Treasures/collection_rate', collection_rate, episode)
            
            # Log individual agent rewards if available
            if isinstance(episode_rewards, dict):
                for agent_id, reward in episode_rewards.items():
                    self.writer.add_scalar(f'Agent_Rewards/{agent_id}', reward, episode)
        
        # Save metrics
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(step_count)
        self.treasures_collected.append(treasures_collected)
        
        # Save metrics to file
        metrics = {
            'episode': episode + 1,
            'mean_reward': float(mean_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
            'steps': int(step_count),
            'agents': int(num_agents),
            'treasures_collected': int(treasures_collected),
            'timestamp': datetime.now().isoformat(),
            'agent_rewards': {k: float(v) for k, v in episode_rewards.items()}
        }
        
        with open(self.run_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def plot_metrics(self):
        """Plot training metrics."""
        plt.figure(figsize=(12, 4))
        
        # Plot mean reward
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Mean Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        
        # Plot treasures collected
        plt.subplot(1, 2, 2)
        plt.plot(self.treasures_collected)
        plt.title('Treasures Collected per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Treasures Collected')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_metrics.png')
        plt.close()

    def train(self):
        """Run the training loop."""
        num_episodes = self.config['num_episodes']
        
        for episode in range(num_episodes):
            # Reset environment and get initial observations
            observations, _ = self.env.reset()  # Unpack the tuple (observations, infos)
            dones = {agent: False for agent in self.agents}
            episode_rewards = {agent: 0 for agent in self.agents}
            step_count = 0
            total_treasures_collected = 0
            
            # Handle different types of dones (dict or tuple)
            def all_done(dones):
                if isinstance(dones, dict):
                    return all(dones.values())
                elif isinstance(dones, (tuple, list)):
                    return all(dones)
                return False
                
            while not all_done(dones):
                actions = {}
                
                # Initialize dictionaries to store agent experiences for this step
                step_experiences = {agent_id: None for agent_id in self.agents}
                
                # Get actions from each agent
                for agent_id, obs in observations.items():
                    # Check if agent is done
                    agent_done = False
                    if isinstance(dones, dict):
                        agent_done = dones.get(agent_id, False)
                    elif isinstance(dones, (tuple, list)) and len(dones) > 0:
                        agent_index = list(self.agents).index(agent_id) if agent_id in self.agents else -1
                        if 0 <= agent_index < len(dones):
                            agent_done = dones[agent_index]
                    
                    if agent_done:
                        continue
                    
                    # Get action, log_prob, and value from agent
                    action, log_prob, value_estimate = self.agents_dict[agent_id].select_action(obs)
                    actions[agent_id] = action
                    
                    # Store experience for this agent
                    step_experiences[agent_id] = {
                        'obs': obs,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value_estimate
                    }
                
                # Step the environment
                step_returns = self.env.step(actions)
                if len(step_returns) == 5:
                    next_observations, rewards_dict, terminations, truncations, infos = step_returns
                    # Convert to the old format for compatibility
                    dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                            for agent in self.agents}
                else:
                    # Handle case where step returns a different number of values
                    next_observations, rewards_dict, dones, infos = step_returns
                
                # Store experiences and update agents
                for agent_id in self.agents:
                    # Skip if agent is done
                    agent_done = False
                    if isinstance(dones, dict):
                        agent_done = dones.get(agent_id, False)
                    elif isinstance(dones, (tuple, list)) and len(dones) > 0:
                        agent_index = list(self.agents).index(agent_id) if agent_id in self.agents else -1
                        if 0 <= agent_index < len(dones):
                            agent_done = dones[agent_index]
                    
                    if agent_done:
                        continue
                    
                    # Get reward for this agent
                    reward = 0
                    if isinstance(rewards_dict, dict):
                        reward = rewards_dict.get(agent_id, 0)
                    elif isinstance(rewards_dict, (tuple, list)) and len(rewards_dict) > 0:
                        agent_index = list(self.agents).index(agent_id) if agent_id in self.agents else -1
                        if 0 <= agent_index < len(rewards_dict):
                            reward = rewards_dict[agent_index]
                    
                    # Get the experience for this agent from the current step
                    if step_experiences.get(agent_id) is None:
                        continue
                        
                    exp = step_experiences[agent_id]
                    obs = exp['obs']
                    action = exp['action']
                    log_prob = exp['log_prob']
                    value = exp['value']
                    next_obs = next_observations[agent_id] if agent_id in next_observations else obs
                    
                    # Store the experience in the agent's memory buffer
                    if not hasattr(self.agents_dict[agent_id], 'memory'):
                        self.agents_dict[agent_id].memory = []
                        
                    self.agents_dict[agent_id].memory.append({
                        'states': obs,
                        'actions': action,
                        'rewards': reward,
                        'next_states': next_obs,
                        'dones': float(agent_done),
                        'log_probs': log_prob,
                        'values': value
                    })
                    
                    # Check if we have enough samples to update
                    if len(self.agents_dict[agent_id].memory) >= self.agents_dict[agent_id].batch_size:
                        # Prepare batch for PPO update
                        # We need to compute returns and advantages for PPO
                        # First, collect all experiences
                        experiences = self.agents_dict[agent_id].memory
                        
                        # Get the next value for GAE calculation
                        with torch.no_grad():
                            last_obs = experiences[-1]['next_states']
                            last_obs_tensor = self.agents_dict[agent_id].get_observation(last_obs)
                            _, last_value, _ = self.agents_dict[agent_id].policy(last_obs_tensor)
                            last_value = last_value.squeeze()
                        
                        # Convert experiences to lists for processing
                        states = []
                        actions = []
                        rewards = []
                        dones = []
                        log_probs = []
                        values = []
                        
                        for exp in experiences:
                            # Get observation tensors
                            obs_tensor = self.agents_dict[agent_id].get_observation(exp['states'])
                            
                            # Add to batch
                            states.append(obs_tensor.squeeze(0))  # Remove batch dim for stacking
                            actions.append(exp['actions'])
                            rewards.append(exp['rewards'])
                            dones.append(exp['dones'])
                            log_probs.append(exp['log_probs'])
                            values.append(exp['values'])
                        
                        # Convert to tensors
                        states = torch.stack(states).to(self.config['device'])
                        actions = torch.tensor(actions, device=self.config['device'])
                        rewards = torch.tensor(rewards, device=self.config['device'])
                        dones = torch.tensor(dones, device=self.config['device'])
                        log_probs = torch.stack(log_probs).to(self.config['device'])
                        values = torch.stack(values).to(self.config['device'])
                        
                        # Compute returns and advantages using GAE
                        returns, advantages = self.agents_dict[agent_id].compute_gae(
                            next_value=last_value,
                            rewards=rewards,
                            masks=1 - dones.float(),  # 1 for not done, 0 for done
                            values=values,
                            gamma=self.config['gamma'],
                            gae_lambda=self.config.get('gae_lambda', 0.95)
                        )
                        
                        # Prepare the batch for PPO update
                        # Note: states, actions, log_probs, returns, advantages, and values are already tensors
                        # and have been moved to the correct device in the previous steps
                        batch = {
                            'states': states,  # Already a tensor [T, C, H, W]
                            'actions': actions,  # Already a tensor [T]
                            'old_log_probs': log_probs,  # Already a tensor [T]
                            'returns': returns,  # Already a tensor [T]
                            'advantages': advantages,  # Already a tensor [T]
                            'values': values  # Already a tensor [T]
                        }
                        
                        # Perform PPO update
                        metrics = self.agents_dict[agent_id].ppo_update(batch)
                        
                        # Log metrics if available
                        if metrics and hasattr(self, 'writer') and self.writer is not None:
                            for metric_name, metric_value in metrics.items():
                                self.writer.add_scalar(f'agent_{agent_id}/{metric_name}', metric_value, step_count)
                        
                        # Clear memory after update
                        self.agents_dict[agent_id].memory = []
                    
                    # Update episode rewards
                    episode_rewards[agent_id] += reward
                
                observations = next_observations
                step_count += 1
                
                # Optional: Render the environment
                if self.config.get('render', False) and episode % self.config.get('render_every', 10) == 0:
                    self.env.render()
                    time.sleep(0.1)  # Slow down rendering
            
            # Update exploration parameters at the end of the episode
            for agent in self.agents_dict.values():
                agent.on_episode_end()
            
            # Get the actual number of treasures collected from the environment
            treasures_collected = getattr(self.env, 'collected_count', 0)
            
            # Log metrics at the end of the episode
            self.log_metrics(episode, episode_rewards, step_count, treasures_collected)
            
            # Log exploration metrics
            if self.writer is not None:
                # Log epsilon from the first agent (all agents share the same schedule)
                first_agent = self.agents_dict[self.agents[0]]
                self.writer.add_scalar('exploration/epsilon', first_agent.epsilon, episode)
                
                # Log average entropy if available
                if hasattr(first_agent, 'last_entropy'):
                    avg_entropy = np.mean([agent.last_entropy for agent in self.agents_dict.values() 
                                         if hasattr(agent, 'last_entropy')])
                    self.writer.add_scalar('exploration/avg_entropy', avg_entropy, episode)
                
                # Log treasures collected
                self.writer.add_scalar('treasures/collected', treasures_collected, episode)
            
            # Save model checkpoints periodically
            if (episode + 1) % self.config.get('save_interval', 10) == 0:
                self.save_models(episode + 1)
                
            # Log exploration progress
            if (episode + 1) % 10 == 0:
                first_agent = self.agents_dict[self.agents[0]]
                self.logger.info(f"Episode {episode + 1}: Epsilon = {first_agent.epsilon:.3f}, "
                               f"Avg Reward = {np.mean(list(episode_rewards.values())):.2f}, "
                               f"Treasures collected: {treasures_collected}")
                
                # Log remaining treasures for debugging
                if hasattr(self.env, 'treasures'):
                    self.logger.debug(f"Remaining treasures: {len(self.env.treasures)}")
            
            # Plot metrics every 10 episodes
            if (episode + 1) % 10 == 0:
                self.plot_metrics()
            
            # Run greedy evaluation every 100 episodes
            if (episode + 1) % 100 == 0:
                self.run_greedy_evaluation(num_episodes=10)
    
    def run_greedy_evaluation(self, num_episodes=10):
        """Run evaluation with ε=0 (no exploration)."""
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Running greedy evaluation for {num_episodes} episodes (ε=0)")
        self.logger.info("="*50)
        
        # Save current epsilons
        original_epsilons = {agent_id: agent.epsilon for agent_id, agent in self.agents_dict.items()}
        
        # Set ε=0 for all agents
        for agent in self.agents_dict.values():
            agent.epsilon = 0.0
        
        # Run evaluation episodes
        eval_results = []
        for ep in range(num_episodes):
            observations, _ = self.env.reset()
            dones = {agent: False for agent in self.agents}
            episode_rewards = {agent: 0 for agent in self.agents}
            step_count = 0
            treasures_collected = 0
            
            while not all(dones.values()):
                actions = {}
                for agent_id, obs in observations.items():
                    if dones[agent_id]:
                        continue
                    # Use explore=False to ensure greedy action selection
                    action, _ = self.agents_dict[agent_id].select_action(obs, explore=False)
                    actions[agent_id] = action
                
                # Step the environment
                next_observations, rewards_dict, terminations, truncations, _ = self.env.step(actions)
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in self.agents}
                
                # Update rewards and step count
                for agent_id, reward in rewards_dict.items():
                    episode_rewards[agent_id] += reward
                
                observations = next_observations
                step_count += 1
                
                # Track treasures collected
                if hasattr(self.env, 'collected_count'):
                    treasures_collected = self.env.collected_count
                
                if step_count >= self.config['max_steps']:
                    break
            
            # Log episode results
            mean_reward = np.mean(list(episode_rewards.values()))
            max_reward = max(episode_rewards.values())
            min_reward = min(episode_rewards.values())
            
            self.logger.info(f"Eval Episode {ep+1}/{num_episodes}:")
            self.logger.info(f"  Steps: {step_count}, Treasures: {treasures_collected}/{self.config['num_treasures']}")
            self.logger.info(f"  Rewards - Mean: {mean_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}")
            
            eval_results.append({
                'steps': step_count,
                'treasures_collected': treasures_collected,
                'mean_reward': mean_reward,
                'max_reward': max_reward,
                'min_reward': min_reward
            })
        
        # Calculate and log summary statistics
        success_rate = np.mean([1 if r['treasures_collected'] == self.config['num_treasures'] else 0 
                              for r in eval_results])
        avg_steps = np.mean([r['steps'] for r in eval_results])
        avg_reward = np.mean([r['mean_reward'] for r in eval_results])
        
        self.logger.info("\n" + "="*50)
        self.logger.info("Greedy Evaluation Summary:")
        self.logger.info(f"  Success Rate: {success_rate*100:.1f}%")
        self.logger.info(f"  Avg Steps: {avg_steps:.1f}")
        self.logger.info(f"  Avg Reward: {avg_reward:.2f}")
        self.logger.info("="*50 + "\n")
        
        # Restore original epsilons
        for agent_id, epsilon in original_epsilons.items():
            self.agents_dict[agent_id].epsilon = epsilon
        
        return {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward
        }

    def save_models(self, episode):
        """Save model checkpoints."""
        checkpoint_dir = self.run_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        for agent_id, agent in self.agents_dict.items():
            checkpoint_path = checkpoint_dir / f'agent_{agent_id}_episode_{episode}.pt'
            agent.save(checkpoint_path)
            
        self.logger.info(f"Saved checkpoints for episode {episode}")

def main():
    """Run the baseline training."""
    # Configuration
    config = {
        'env_name': 'SimpleGrid-v0',
        'grid_size': 10,
        'num_agents': 2,
        'num_treasures': 3,
        'max_steps': 2000,  # Increased to allow more exploration
        'num_episodes': 1000,
        'batch_size': 64,
        'hidden_size': 256,
        
        # PPO-specific parameters
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_param': 0.2,
        'ppo_epochs': 4,
        'num_mini_batches': 4,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Evaluation and logging
        'eval_interval': 10,
        'log_interval': 10,
        'save_interval': 50,
        'log_dir': 'logs/baseline',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'use_llama': False  # Set to True to use LLaMA model instead of A2C
    }
    
    # Run training
    trainer = BaselineTrainer(config)
    trainer.train()

if __name__ == "__main__":
    import torch  # Import here to avoid circular imports
    main()
