import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque, namedtuple
import json
from datetime import datetime

# Import from our package structure
from environment.pettingzoo_env_wrapper import SimpleGridEnv
from agents.a2c_agent import A2CAgent, create_agents

# Configuration
class Config:
    # Environment
    GRID_SIZE = 10
    NUM_TREASURES = 3
    MAX_STEPS = 200
    RENDER_MODE = None  # Set to "human" to visualize training
    
    # Rendering settings
    RENDER_MODE = "human"  # Options: None, "human", "rgb_array"
    RENDER_INTERVAL = 10    # Render every N episodes (0 for no rendering)
    SAVE_VIDEO = True      # Save video of rendered episodes
    
    # Training settings
    NUM_EPISODES = 5000
    BATCH_SIZE = 128
    GAMMA = 0.99
    LR = 2.5e-4
    
    # Exploration settings
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.9995
    
    # Curriculum learning stages
    CURRICULUM_STAGES = [
        {'treasures': 1, 'episodes': 1000},  # Stage 1: 1 treasure (extended)
        {'treasures': 2, 'episodes': 2000},  # Stage 2: 2 treasures (extended)
        {'treasures': 3, 'episodes': 3000},  # Stage 3: 3 treasures (extended)
        {'treasures': 4, 'episodes': 4000},  # Stage 4: 4 treasures (new)
        {'treasures': 5, 'episodes': 5000}   # Stage 5: 5 treasures (new)
    ]
    
    # Logging and saving
    LOG_DIR = "logs"
    SAVE_DIR = "saved_models"
    SAVE_INTERVAL = 100     # Save models every N episodes
    LOG_INTERVAL = 10       # Print logs every N episodes
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        # Create necessary directories
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.LOG_DIR, "videos"), exist_ok=True)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_environment()
        self.setup_agents()
        self.setup_logging()
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_treasure_counts = []
        self.best_avg_reward = -np.inf
        
    def setup_directories(self):
        """Create necessary directories for logging and saving models."""
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        
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
        """Initialize the agents with stability improvements."""
        self.agents = create_agents(
            grid_size=self.config.GRID_SIZE,
            num_agents=2,
            epsilon_start=self.config.EPSILON_START,
            epsilon_end=self.config.EPSILON_END,
            epsilon_decay=300,  # Decay over 300 episodes
            batch_size=self.config.BATCH_SIZE,
            # Stability improvements
            actor_lr=1e-4,  # Lower learning rate for actor
            critic_lr=2e-4,  # Slightly higher learning rate for critic
            entropy_coef=0.02,  # Increased entropy for better exploration
            max_grad_norm=0.5,  # Gradient clipping
            value_loss_clip=0.2,  # Clip value updates
            grad_clip_value=5.0,  # Clip gradients by value
            # PPO-style improvements
            clip_param=0.2,  # PPO clip parameter
            value_clip_param=0.2,  # Value function clip parameter
            normalize_advantages=True,  # Normalize advantages
            clip_value_loss=True,  # Clip value loss
            normalize_returns=True  # Normalize returns
        )
        self.epsilon = self.config.EPSILON_START
        
    def setup_logging(self):
        """Initialize logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.config.LOG_DIR, f"training_{timestamp}.csv")
        
        # Write CSV header
        with open(self.log_file, 'w') as f:
            f.write("episode,step,agent,reward,epsilon,total_reward,policy_loss,value_loss,entropy\n")
    
    def log_metrics(self, episode, step, agent_id, metrics):
        """Log training metrics to file."""
        with open(self.log_file, 'a') as f:
            f.write(
                f"{episode},{step},{agent_id},{metrics.get('reward', 0):.4f},"
                f"{self.epsilon:.4f},{metrics.get('total_reward', 0):.4f},"
                f"{metrics.get('policy_loss', 0):.4f},"
                f"{metrics.get('value_loss', 0):.4f},"
                f"{metrics.get('entropy', 0):.4f}\n"
            )
    
    def update_agent_parameters(self, global_step):
        """Update agent parameters like epsilon and learning rates with stability checks."""
        # Update epsilon for all agents using step-based scheduling
        for agent in self.agents.values():
            agent.update_epsilon(global_step)
            
            # Step the learning rate schedulers with gradient statistics
            if not agent.use_llama and hasattr(agent, 'actor_scheduler'):
                # Get gradient statistics before stepping
                actor_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in agent.actor_params if p.grad is not None]), 2).item()
                critic_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in agent.critic_params if p.grad is not None]), 2).item()
                
                # Only step if gradients are stable
                if not np.isnan(actor_grad_norm) and not np.isinf(actor_grad_norm):
                    agent.actor_scheduler.step()
                if not np.isnan(critic_grad_norm) and not np.isinf(critic_grad_norm):
                    agent.critic_scheduler.step()
        
        # Update global epsilon for logging
        self.epsilon = next(iter(self.agents.values())).epsilon if self.agents else self.config.EPSILON_END
    
    def run_episode(self, episode_num, num_treasures):
        """
        Run a single training episode with optional rendering and logging.
        
        Args:
            episode_num: Current episode number
            num_treasures: Number of treasures for this episode
            
        Returns:
            Dictionary containing episode statistics
        """
        # Enable rendering for visualization if needed
        render_episode = (episode_num % self.config.RENDER_INTERVAL == 0) or (episode_num < 5)
        
        # Reset environment and get initial observations
        self.env.render_mode = "human" if render_episode and self.config.RENDER_MODE == "human" else None
        observations, _ = self.env.reset()
        episode_rewards = {agent_id: 0 for agent_id in self.env.agents}
        
        # Store frames for video creation if needed
        frames = [] if render_episode and self.config.SAVE_VIDEO else None
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }
        
        for step in range(self.config.MAX_STEPS):
            actions = {}
            
            # Select actions for each agent
            for agent_id in self.env.agents:
                agent = self.agents[agent_id]
                # select_action takes obs and explore flag, not training
                action, _ = agent.select_action(observations[agent_id], explore=True)
                actions[agent_id] = action
            
            # Render current state if needed
            if render_episode and self.config.RENDER_MODE == "human":
                self.env.render()
                time.sleep(0.05)  # Slow down rendering for human visibility
                
                # Save frame if creating video
                if frames is not None:
                    frame = self.env.render(mode="rgb_array")
                    frames.append(frame)
            
            # Take a step in the environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Store step data for logging
            if render_episode:
                episode_data['observations'].append(observations)
                episode_data['actions'].append(actions)
                episode_data['rewards'].append(rewards)
                episode_data['dones'].append(terminations or truncations)
                episode_data['infos'].append(infos)
            
            # Store experiences and update agents
            for agent_id in self.env.agents:
                agent = self.agents[agent_id]
                
                # Process observations before storing in memory
                state_tensor = agent.get_observation(observations[agent_id])
                next_state_tensor = agent.get_observation(next_observations[agent_id])
                
                # Store the experience in the agent's memory
                agent.remember(
                    state_tensor,  # Store processed tensor
                    actions[agent_id],
                    rewards[agent_id],
                    next_state_tensor,  # Store processed tensor
                    terminations[agent_id] or truncations[agent_id]
                )
                
                # Update agent parameters (epsilon, learning rates)
                self.update_agent_parameters(global_step=episode_num * self.config.MAX_STEPS + step)
                
                # Update the agent with the batch of experiences
                if len(agent.memory) >= self.config.BATCH_SIZE:
                    # Get a batch of experiences
                    batch = random.sample(agent.memory, self.config.BATCH_SIZE)
                    states, actions_batch, rewards_batch, next_states, dones = zip(*batch)
                    
                    # Convert to tensors (states and next_states already have shape [C,H,W])
                    # Stack them to get [B,C,H,W]
                    states_tensor = torch.stack(states).to(self.config.DEVICE)
                    next_states_tensor = torch.stack(next_states).to(self.config.DEVICE)
                    actions_tensor = torch.tensor(actions_batch, device=self.config.DEVICE)
                    
                    # Convert rewards to tensor and move to device
                    rewards_tensor = torch.tensor(rewards_batch, device=self.config.DEVICE, dtype=torch.float32)
                    dones_tensor = torch.tensor(dones, device=self.config.DEVICE, dtype=torch.float32)
                    
                    # Clip rewards to prevent outliers
                    rewards_tensor = torch.clamp(rewards_tensor, -10.0, 10.0)
                    
                    # Apply reward normalization (handled by the agent's reward normalizer)
                    
                    # Get log_probs and values for the current states
                    with torch.no_grad():
                        logits, values = agent.policy(states_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_probs = dist.log_prob(actions_tensor)
                    
                    # Prepare the batch of experiences
                    batch = {
                        'states': states_tensor,
                        'actions': actions_tensor,
                        'rewards': rewards_tensor,
                        'next_states': next_states_tensor,
                        'dones': dones_tensor,
                        'log_probs': log_probs,
                        'values': values.squeeze()
                    }
                    
                    try:
                        # Update the agent with the batch of experiences
                        stats = agent.update(batch)
                        
                        # Clear memory after successful update
                        agent.memory = []
                        
                        # Log metrics with additional stability information
                        if stats:
                            # Add gradient statistics
                            stats['reward'] = rewards[agent_id]
                            stats['total_reward'] = episode_rewards[agent_id] + rewards[agent_id]
                            
                            # Track value and advantage statistics
                            if 'value_mean' in stats and 'value_var' in stats:
                                stats['value_std'] = np.sqrt(max(0, stats['value_var']))
                            if 'adv_mean' in stats and 'adv_var' in stats:
                                stats['adv_std'] = np.sqrt(max(0, stats['adv_var']))
                            
                            # Log the metrics
                            self.log_metrics(episode_num, step, agent_id, stats)
                            
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print("CUDA out of memory, skipping batch")
                            torch.cuda.empty_cache()
                        elif 'NaN' in str(e) or 'Inf' in str(e):
                            print(f"Numerical instability detected: {e}")
                            # Reset gradients and clear memory
                            agent.actor_optimizer.zero_grad()
                            agent.critic_optimizer.zero_grad()
                            agent.memory = []
                        else:
                            raise
                
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Update observations
            observations = next_observations
            
            # Check if all agents are done
            if all(terminations.values()) or all(truncations.values()):
                break
        
        # Decay exploration rate
        self.decay_epsilon()
        
        # Save models periodically
        if episode_num % self.config.SAVE_INTERVAL == 0:
            self.save_models(episode_num)
        
        # Save episode data if this was a rendered episode
        if render_episode:
            self._save_episode_data(episode_num, episode_data, frames)
            
            # Save video if we captured frames
            if frames and len(frames) > 0:
                self._save_episode_video(episode_num, frames)
        
        # Calculate episode statistics
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
        max_reward = max(episode_rewards.values())
        min_reward = min(episode_rewards.values())
        
        # Get treasure collection stats from environment
        treasures_collected = getattr(self.env, 'collected_count', 0)
        total_treasures = getattr(self.env, 'initial_treasures', 0)
        collection_rate = (treasures_collected / total_treasures * 100) if total_treasures > 0 else 0
        
        # Print episode summary with treasure collection info
        if episode_num % self.config.LOG_INTERVAL == 0:
            print(f"\n--- Episode {episode_num} Summary ---")
            print(f"Stage: {num_treasures} treasures")
            print(f"Steps: {step + 1}/{self.config.MAX_STEPS}")
            print(f"Treasures: {treasures_collected}/{total_treasures} ({collection_rate:.1f}%)")
            print(f"Total Reward: {sum(episode_rewards.values()):.2f} (Avg: {avg_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f})")
            print(f"Exploration Rate: {self.epsilon:.4f}")
            
            # Log additional metrics if available
            if hasattr(self, 'last_metrics'):
                metrics = self.last_metrics
                print(f"Policy Loss: {metrics.get('policy_loss', 0):.4f}")
                print(f"Value Loss: {metrics.get('value_loss', 0):.4f}")
                print(f"Entropy: {metrics.get('entropy', 0):.4f}")
            
            # Log reward components if available
            if hasattr(self.env, 'reward_components'):
                print("\nReward Components (avg per step):")
                for comp, value in self.env.reward_components.items():
                    print(f"- {comp}: {value:.4f}")
            
            print("-" * 40)
            print(f"Episode {episode_num}, Epsilon: {self.epsilon:.3f}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Treasures: {treasures_collected}/{total_treasures} ({collection_rate:.1f}%), "
                  f"Steps: {step+1}/{self.config.MAX_STEPS}")
            
            # Save visualization of agent paths if this was a rendered episode
            if render_episode and hasattr(self.env, 'render_paths'):
                try:
                    self.env.render_paths(episode_num)
                    print(f"[INFO] Saved path visualization for episode {episode_num}")
                except Exception as e:
                    print(f"[WARNING] Failed to save path visualization: {str(e)}")
        
        return episode_rewards
    
    def _save_episode_data(self, episode_num, episode_data, frames=None):
        """
        Save episode data for analysis and visualization.
        
        Args:
            episode_num: Episode number
            episode_data: Dictionary containing episode data
            frames: List of frames if video was captured
        """
        # Create episode directory
        episode_dir = os.path.join(self.config.LOG_DIR, f"episode_{episode_num}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save episode data
        episode_path = os.path.join(episode_dir, "episode_data.pkl")
        with open(episode_path, 'wb') as f:
            import pickle
            pickle.dump(episode_data, f)
        
        # Save frames if available
        if frames and len(frames) > 0:
            frames_dir = os.path.join(episode_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, frame in enumerate(frames):
                import imageio
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                imageio.imwrite(frame_path, frame)
    
    def _save_episode_video(self, episode_num, frames):
        """
        Save episode as a video file and optionally create a GIF and path visualization.
        
        Args:
            episode_num: Episode number
            frames: List of frames to save as video
        """
        if not frames:
            return
            
        try:
            import imageio
            from tqdm import tqdm
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.animation import FuncAnimation
            
            # Create output directories
            videos_dir = os.path.join(self.config.LOG_DIR, "videos")
            gifs_dir = os.path.join(self.config.LOG_DIR, "gifs")
            path_viz_dir = os.path.join(self.config.LOG_DIR, "path_visualizations")
            
            for dir_path in [videos_dir, gifs_dir, path_viz_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # 1. Save as MP4
            video_path = os.path.join(videos_dir, f"episode_{episode_num}.mp4")
            with imageio.get_writer(video_path, fps=15) as writer:
                for frame in tqdm(frames, desc=f"Saving video for episode {episode_num}"):
                    writer.append_data(frame)
            
            # 2. Save as GIF
            gif_path = os.path.join(gifs_dir, f"episode_{episode_num}.gif")
            with imageio.get_writer(gif_path, mode='I', fps=10, loop=0) as writer:
                for frame in tqdm(frames, desc=f"Creating GIF for episode {episode_num}"):
                    writer.append_data(frame)
            
            # 3. Create and save path visualization
            self._save_path_visualization(episode_num, frames, path_viz_dir)
            
            print(f"\n--- Saved visualizations for episode {episode_num} ---")
            print(f"Video: {video_path}")
            print(f"GIF: {gif_path}")
            print(f"Path visualization: {path_viz_dir}/episode_{episode_num}_paths.png")
            
        except ImportError as e:
            print(f"Warning: Could not create visualizations - {e}. Install imageio, matplotlib, and tqdm for full support.")
    
    def _save_path_visualization(self, episode_num, frames, output_dir):
        """
        Create and save a static visualization of agent paths.
        
        Args:
            episode_num: Episode number
            frames: List of frames from the episode
            output_dir: Directory to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Rectangle, Circle
            
            if not frames:
                return
                
            # Create a new figure
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Use the last frame as the base
            base_img = frames[-1].copy()
            
            # Convert to float for alpha blending
            base_img = base_img.astype(float) / 255.0
            
            # Create a transparent overlay for paths
            overlay = np.zeros_like(base_img)
            
            # Draw paths with fading effect
            num_frames = len(frames)
            for i, frame in enumerate(frames):
                # Convert frame to grayscale for path detection
                gray = np.mean(frame, axis=2) / 255.0
                
                # Find agent positions (red pixels)
                red_mask = (frame[..., 0] > 200) & (frame[..., 1] < 100) & (frame[..., 2] < 100)
                y_coords, x_coords = np.where(red_mask)
                
                # Add to overlay with fading effect
                alpha = 0.3 * (i / num_frames)  # Fade in effect
                for y, x in zip(y_coords, x_coords):
                    # Use a color that stands out (cyan for visibility)
                    overlay[y, x] = [0, 1, 1, alpha]  # RGBA
            
            # Combine base image with overlay
            for c in range(3):  # RGB channels
                base_img[..., c] = base_img[..., c] * (1 - overlay[..., 3]) + overlay[..., c] * overlay[..., 3]
            
            # Display the result
            ax.imshow(base_img)
            ax.axis('off')
            
            # Add title and info
            plt.title(f'Episode {episode_num} - Agent Paths', fontsize=14)
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.5, label='Agents'),
                plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.5, label='Base'),
                plt.Rectangle((0,0), 1, 1, fc=(1, 0.84, 0), alpha=0.5, label='Treasures'),
                plt.Line2D([0], [0], color='cyan', lw=4, alpha=0.5, label='Agent Paths')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Save the figure
            output_path = os.path.join(output_dir, f'episode_{episode_num}_paths.png')
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create path visualization - {e}")
    
    def save_models(self, episode_num):
        """Save agent models."""
        save_dir = os.path.join(self.config.SAVE_DIR, f"episode_{episode_num}")
        os.makedirs(save_dir, exist_ok=True)
        for agent_id, agent in self.agents.items():
            # Create a subdirectory for each agent
            agent_dir = os.path.join(save_dir, agent_id)
            os.makedirs(agent_dir, exist_ok=True)
            # Save the agent's model
            model_path = os.path.join(agent_dir, "model.pth")
            agent.save(model_path)
    
    def train(self):
        """Main training loop with curriculum learning."""
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Logging to {self.log_file}")
        
        # Initialize curriculum learning
        current_stage = 0
        current_treasures = self.config.CURRICULUM_STAGES[0]['treasures']
        best_avg_reward = float('-inf')
        
        # Track episode statistics
        episode_rewards_history = []
        episode_lengths = []
        
        try:
            for episode in range(1, self.config.NUM_EPISODES + 1):
                start_time = time.time()
                
                # Update environment with current number of treasures
                self.setup_environment(num_treasures=current_treasures)
                
                # Run one episode
                episode_data = self.run_episode(episode, current_treasures)
                
                # Update curriculum based on performance
                if episode > 100:  # Wait for some initial exploration
                    avg_reward = np.mean(episode_rewards_history[-100:])  # Last 100 episodes
                    if avg_reward > best_avg_reward * 1.1:  # 10% improvement
                        best_avg_reward = avg_reward
                        if current_stage < len(self.config.CURRICULUM_STAGES) - 1:
                            current_stage += 1
                            current_treasures = self.config.CURRICULUM_STAGES[current_stage]['treasures']
                            print(f"\nAdvancing to curriculum stage {current_stage + 1} with {current_treasures} treasures")
                
                # Get learning rates for logging
                actor_lr = critic_lr = 0
                if self.agents:
                    sample_agent = next(iter(self.agents.values()))
                    if not sample_agent.use_llama:
                        actor_lr = sample_agent.actor_optimizer.param_groups[0]['lr']
                        critic_lr = sample_agent.critic_optimizer.param_groups[0]['lr']
                
                # Log to console
                print(f"Episode {episode}/{self.config.NUM_EPISODES}, "
                      f"Avg Reward: {avg_reward if 'avg_reward' in locals() else 0:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Actor LR: {actor_lr:.2e}, "
                      f"Critic LR: {critic_lr:.2e}, "
                      f"Time: {time.time() - start_time:.1f}s")
                
                # Log per-agent rewards
                if 'episode_rewards' in locals():
                    for agent_id, reward in episode_rewards.items():
                        print(f"  {agent_id}: {reward:.4f}")
                
                # Log timing information
                elapsed_time = time.time() - start_time
                print(f"Episode time: {elapsed_time:.2f}s")
                
                # Flush output to ensure we see the logs
                import sys
                sys.stdout.flush()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving models...")
        
        finally:
            # Save final models
            self.save_models("final")
            print(f"Training complete. Models saved to {self.config.SAVE_DIR}")

def main():
    # Initialize and run training
    config = Config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
