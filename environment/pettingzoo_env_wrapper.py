import numpy as np
import random
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing import Dict, List, Optional, Tuple, Union, Any

class RunningMeanStd:
    """Tracks the running mean and variance of values using Welford's online algorithm."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        """Update the running statistics with a new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update the running statistics from batch statistics."""
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

class SimpleGridEnv(ParallelEnv):
    """
    A simple grid world environment for multi-agent treasure collection.
    
    Agents start at the base and must collect treasures in the environment.
    The environment ends when all treasures are collected or max_steps is reached.
    
    Agents:
    - Can move in 4 directions (up, down, left, right)
    - Receive +1 reward for collecting a treasure
    - Share the same observation and action spaces
    """
    
    metadata = {
        "name": "simple_grid_env_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
    
    def __init__(self, size: int = 10, num_treasures: int = 3, max_steps: int = 200, 
                 render_mode: Optional[str] = None, debug: bool = True, track_paths: bool = True,
                 early_stop_threshold: float = 0.8, min_treasures: Optional[int] = None):
        """Initialize the environment.
        
        Args:
            size: Size of the grid (size x size)
            num_treasures: Number of treasures to place
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', None)
            debug: Enable debug output
            track_paths: Whether to track agent paths for visualization
            early_stop_threshold: Stop episode if no progress after this fraction of max_steps
        """
        # Initialize base attributes first
        self.size = size
        self.num_treasures = num_treasures
        self.min_treasures = min_treasures or num_treasures  # For curriculum learning
        self.max_steps = max_steps
        self.steps_taken = 0  # Track steps in current episode
        self.render_mode = render_mode
        self.debug = debug
        self.track_paths = track_paths
        self.early_stop_threshold = early_stop_threshold
        self.episode_count = 0  # Track number of episodes
        
        # Define action spaces and possible agents
        self.possible_agents = ["agent_0", "agent_1"]
        self.possible_actions = 4  # Up, Down, Left, Right
        
        # Initialize path tracking
        self.agent_paths = {agent: [] for agent in self.possible_agents}
        self.episode_treasure_locations = []  # Store treasure locations at episode start
        
        # Reward structure with clipping and normalization
        self.treasure_reward = 5.0     # Base treasure value
        self.step_penalty = -0.01      # Small penalty per step
        self.proximity_bonus = 0.1     # Small proximity bonus
        self.completion_bonus = 5.0    # Completion bonus
        self.time_bonus = 0.02         # Bonus for faster completion
        
        # Reward clipping
        self.max_reward = 10.0         # Maximum reward per step
        self.min_reward = -1.0         # Minimum reward per step
        
        # For curriculum learning
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Normalization
        self.return_rms = RunningMeanStd(shape=())  # For reward normalization
        self.ret = np.zeros(1)         # For tracking returns
        self.gamma = 0.99              # Discount factor
        self.epsilon = 1e-8            # Small constant for numerical stability
        
        # For dynamic rewards
        self.initial_treasure_value = self.treasure_reward
        
        # For logging and debugging
        self.treasures_collected = 0
        self.proximity_count = 0
        
        if self.debug:
            print(f"[DEBUG] Environment initialized with {num_treasures} treasures, size {size}x{size}")
        
        # Define action and observation spaces
        self.possible_agents = ["agent_0", "agent_1"]
        self.possible_actions = 4  # Up, Down, Left, Right
        
        # Each agent can observe the full grid with 4 channels:
        # Channel 0: Agent positions (1 if agent is present, 0 otherwise)
        # Channel 1: Treasure positions (1 if treasure is present, 0 otherwise)
        # Channel 2: Base position (1 if base is present, 0 otherwise)
        # Channel 3: Current agent position (1 if current agent is present, 0 otherwise)
        self.observation_spaces = {
            agent: spaces.Dict({
                "grid": spaces.Box(low=0, high=1, shape=(4, size, size), dtype=np.float32),
                "agent_pos": spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
            })
            for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: spaces.Discrete(self.possible_actions) 
            for agent in self.possible_agents
        }
        
        # Initialize environment state
        self.agents = self.possible_agents[:]
        self.base_location = (size // 2, size // 2)
        self.treasures = []
        self.agent_positions = {}
        self.steps_taken = 0
        self.grid = np.zeros((4, size, size), dtype=np.float32)
        self.initial_treasures = 0
        self.agent_last_positions = {}
        self.collected_count = 0  # Track total treasures collected
        
    def _update_grid(self):
        """Update the grid representation based on current state.
        
        Creates a 4-channel grid where:
        - Channel 0: Agent positions (1 if any agent is present, 0 otherwise)
        - Channel 1: Treasure positions (1 if treasure is present, 0 otherwise)
        - Channel 2: Base position (1 if base is present, 0 otherwise)
        - Channel 3: Current agent position (1 if current agent is present, 0 otherwise)
        """
        # Initialize 4-channel grid
        self.grid = np.zeros((4, self.size, self.size), dtype=np.float32)
        
        # Channel 1: Mark treasures
        for treasure in self.treasures:
            self.grid[1, treasure[0], treasure[1]] = 1.0
            
        # Channel 2: Mark base
        self.grid[2, self.base_location[0], self.base_location[1]] = 1.0
        
        # Channel 0: Mark all agent positions
        for pos in self.agent_positions.values():
            self.grid[0, pos[0], pos[1]] = 1.0
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # Initialize the random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Reset path tracking
        if self.track_paths:
            self.agent_paths = {agent: [] for agent in self.possible_agents}
            self.episode_treasure_locations = []
        
        # Reset environment state
        self.grid.fill(0)
        self.steps_taken = 0
        self.treasures = []
        self.agents = self.possible_agents[:]
        
        # For early stopping and progress tracking
        self.steps_without_progress = 0
        self.best_treasures_found = 0
        self.last_treasure_step = 0  # Track when last treasure was found
        self.progress_history = []  # Track progress over time
        
        # Place agents at base
        for agent in self.agents:
            self.agent_positions[agent] = self.base_location
            if self.debug:
                print(f"[DEBUG] {agent} reset to position {self.base_location}")
            
        # Place treasures randomly
        self.treasures = []
        while len(self.treasures) < self.num_treasures:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos != self.base_location and pos not in self.treasures:
                self.treasures.append(pos)
                if self.debug:
                    print(f"[DEBUG] Treasure placed at {pos}")
        
        # Store initial treasure locations for path visualization
        if self.track_paths:
            self.episode_treasure_locations = self.treasures.copy()
        
        self.initial_treasures = len(self.treasures)
        if self.debug:
            print(f"[DEBUG] Reset complete - {self.initial_treasures} treasures placed, {len(self.agents)} agents ready")
                
        # Reset collected count
        self.collected_count = 0
        
        # Update grid
        self._update_grid()
        
        # Get initial observations
        observations = {
            agent: self._get_observation(agent) 
            for agent in self.agents
        }
        
        # Get dummy infos
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            actions: Dict mapping agent names to actions
            
        Returns:
            observations: Dict mapping agent names to their observations
            rewards: Dict mapping agent names to their rewards
            terminations: Dict mapping agent names to whether they are done
            truncations: Dict mapping agent names to whether they were truncated
            infos: Dict mapping agent names to additional info
        """
        # Enforce hard episode cap
        if self.steps_taken >= self.max_steps:
            return (
                {agent: self._get_observation(agent) for agent in self.agents},
                {agent: 0.0 for agent in self.agents},
                {agent: True for agent in self.agents},  # terminations
                {agent: True for agent in self.agents},  # truncations
                {agent: {} for agent in self.agents}     # infos
            )
        
        # Store the last positions for each agent before moving
        self.agent_last_positions = {agent: pos for agent, pos in self.agent_positions.items()}
        
        # Track paths if enabled
        if self.track_paths:
            for agent in self.agents:
                self.agent_paths[agent].append(self.agent_positions[agent])
        
        # Initialize rewards and terminations
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # Store previous positions for movement tracking
        prev_positions = {agent: self.agent_positions[agent] for agent in self.agents}
        
        # Execute actions
        for agent in self.agents:
            if agent in actions:
                self._move_agent(agent, actions[agent])
        
        # Calculate rewards for each agent
        for agent in self.agents:
            pos = self.agent_positions[agent]
            
            # Step penalty and movement bonus
            moved = (pos != prev_positions[agent])
            rewards[agent] += self.step_penalty * (0.5 if moved else 2.0)  # Stronger penalty for staying still
            
            # Check for treasure collection with dynamic value and time bonus
            if pos in self.treasures:
                # Fixed treasure value for consistent learning
                treasure_value = self.treasure_reward
                rewards[agent] += treasure_value
                
                # Remove treasure from the grid and tracking
                self.treasures.remove(pos)
                # Update the treasure channel (channel 1) at the treasure position
                self.grid[1, pos[0], pos[1]] = 0.0  # Clear the treasure from the grid
                self.collected_count += 1  # Increment collected count
                self.treasures_collected += 1
                
                # Log detailed treasure collection info
                if self.debug:
                    print(f"[TREASURE] {agent} collected treasure at {pos}! "
                          f"Total collected: {self.collected_count}/{self.initial_treasures} "
                          f"(Ep. {self.steps_taken} steps)")
                    print(f"[REWARD] Added treasure reward: {treasure_value:.2f} "
                          f"(Total reward: {rewards[agent]:.2f})")
                
                # Completion bonus when all treasures are collected
                if not self.treasures:
                    rewards[agent] += self.completion_bonus
                
                if self.debug:
                    print(f"[DEBUG] {agent} collected a treasure at {pos}!")
                    print(f"[DEBUG] Reward breakdown - Base: {treasure_value:.3f}, "
                          f"Total: {rewards[agent]:.3f}")
                    print(f"[DEBUG] Remaining treasures: {remaining_treasures} at positions: {self.treasures}")
                    print(f"[DEBUG] Agent positions: {self.agent_positions}")
                    print(f"[DEBUG] Total treasures collected: {self.collected_count}")
                
                # Ensure rewards don't explode
                rewards[agent] = float(np.clip(rewards[agent], -10.0, 10.0))
                
            # Simple proximity bonus for nearby treasures
            for treasure in self.treasures:
                dx = abs(pos[0] - treasure[0])
                dy = abs(pos[1] - treasure[1])
                distance = dx + dy
                
                # Give proximity bonus for treasures within 3 steps
                if distance <= 3:
                    proximity_bonus = self.proximity_bonus * (4 - distance)  # Linear decay
                    rewards[agent] += proximity_bonus
                    
                    # Add small randomness to the reward (0.9 to 1.1 scale)
                    rewards[agent] *= random.uniform(0.9, 1.1)
                    
                    # Debug logging for first few steps
                    if self.debug and self.proximity_count <= 10:
                        print(f"[DEBUG] Proximity bonus for {agent} at distance {distance}: "
                              f"bonus={proximity_bonus:.3f}, "
                              f"total_reward={rewards[agent]:.3f}")
                    
                    self.proximity_count += 1
                    
            # Clip rewards to prevent explosions
            rewards[agent] = np.clip(rewards[agent], self.min_reward, self.max_reward)
            
            # Apply reward normalization
            if hasattr(self, 'return_rms'):
                rewards[agent] = self._normalize_reward(rewards[agent])
            
            # Check if all treasures are collected
            if not self.treasures:
                rewards[agent] += self.completion_bonus
                if self.debug:
                    print(f"[DEBUG] All treasures collected! Agent {agent} receives completion bonus: {self.completion_bonus}")
                    print(f"[DEBUG] Total treasures collected: {self.treasures_collected}")
                    print(f"[DEBUG] Step: {self.steps_taken}/{self.max_steps}")
                
            # Store current position for next step
            self.agent_last_positions[agent] = pos
        
        # Update step count and check termination
        self.steps_taken += 1
        
        # Update progress tracking
        treasures_found = self.initial_treasures - len(self.treasures)
        self.progress_history.append(treasures_found / self.initial_treasures if self.initial_treasures > 0 else 0)
        
        # Check for progress
        if treasures_found > self.best_treasures_found:
            self.best_treasures_found = treasures_found
            self.steps_without_progress = 0
            self.last_treasure_step = self.steps_taken
        else:
            self.steps_without_progress += 1
        
        # Only end episode if all treasures are found or max steps reached
        done = False
        
        # 1. All treasures found
        if treasures_found >= self.initial_treasures:
            if self.debug:
                print(f"[DEBUG] All treasures found! Stopping episode at step {self.steps_taken}")
            done = True
        # 2. Max steps reached
        elif self.steps_taken >= self.max_steps:
            if self.debug:
                print(f"[DEBUG] Max steps ({self.max_steps}) reached. Stopping episode.")
            done = True
            
        # Log progress every 100 steps
        if self.debug and self.steps_taken % 100 == 0:
            print(f"[DEBUG] Step {self.steps_taken}: Treasures found: {treasures_found}/{self.initial_treasures}")
        
        # Update grid
        self._update_grid()
        
        # Get observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }
        
        # Dummy infos
        infos = {agent: {} for agent in self.agents}
        
        # Check for termination conditions
        if not self.treasures:
            terminations = {agent: True for agent in self.agents}
        
        # Handle rendering
        if self.render_mode == "human":
            self.render()
            
        return observations, rewards, terminations, truncations, infos
    
    def _move_agent(self, agent, action):
        """Move an agent based on the action."""
        moves = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]
        
        # Ensure action is a valid integer index
        try:
            action = int(action)
            if not (0 <= action < len(moves)):
                if self.debug:
                    print(f"[DEBUG] Invalid action {action} for {agent}, using NOOP")
                return  # No movement for invalid actions
                
            current_pos = self.agent_positions[agent]
            move = moves[action]
            new_pos = (
                max(0, min(self.size - 1, current_pos[0] + move[0])),
                max(0, min(self.size - 1, current_pos[1] + move[1]))
            )
            self.agent_positions[agent] = new_pos
            
        except (ValueError, TypeError) as e:
            if self.debug:
                print(f"[DEBUG] Invalid action type {action} for {agent} (type: {type(action)}), using NOOP. Error: {e}")
            return  # No movement for invalid actions
    
    def _get_observation(self, agent):
        """Get the observation for a specific agent.
        
        Returns:
            dict: Observation dictionary containing:
                - grid: 4-channel grid observation (4, size, size)
                - agent_pos: Current agent's position (2,)
        """
        # Create a copy of the grid to modify for this agent
        grid = self.grid.copy()
        
        # Channel 3: Mark current agent position (overrides other channels)
        current_agent_pos = self.agent_positions[agent]
        grid[3, :, :] = 0.0  # Reset channel 3
        grid[3, current_agent_pos[0], current_agent_pos[1]] = 1.0
        
        return {
            "grid": grid,
            "agent_pos": np.array(current_agent_pos, dtype=np.int32)
        }
    
    def _normalize_reward(self, reward):
        """Normalize rewards using running mean and std."""
        if not hasattr(self, 'return_rms'):
            return reward
            
        self.ret = self.ret * self.gamma + reward
        self.return_rms.update(self.ret)
        
        # Normalize the reward
        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return np.clip(normalized_reward, -10, 10)
        
    def _check_treasure_collection(self):
        """Check if any agent has collected a treasure and update rewards."""
        collected = set()
        for agent, pos in self.agent_positions.items():
            if pos in self.treasures and not self.terminations[agent]:
                collected.add(pos)
                self.rewards[agent] += self.treasure_reward
                self.treasures_collected += 1
                
                # Dynamic treasure value (decreases as more are collected)
                remaining = len(self.treasures) - 1  # -1 because we haven't removed this one yet
                dynamic_reward = self.treasure_reward * (1.0 + remaining * 0.1)  # Slight bonus for harder-to-get treasures
                self.rewards[agent] += dynamic_reward
                
                if self.debug:
                    print(f"[DEBUG] {agent} collected treasure at {pos} (reward: {self.treasure_reward + dynamic_reward:.2f})")
        
        # Remove collected treasures and update grid
        for pos in collected:
            self.treasures.discard(pos)
            self.grid[pos] = 0  # Clear treasure from grid
            
            # Remove from treasure locations for rendering
            if pos in self.treasure_locations:
                self.treasure_locations.remove(pos)
        
        # Check for proximity to treasures (even if not collected)
        self._check_proximity_to_treasures()
    
    def _check_proximity_to_treasures(self):
        """Check if agents are close to treasures and give proximity bonus."""
        for agent, pos in self.agent_positions.items():
            if not self.terminations[agent]:
                for tx, ty in self.treasures:
                    dist = abs(pos[0] - tx) + abs(pos[1] - ty)  # Manhattan distance
                    if dist <= 2:  # Within 2 cells
                        # Scale bonus by distance (closer = higher bonus)
                        bonus = self.proximity_bonus * (3 - dist) / 3.0
                        # Scale by remaining treasures (more valuable when fewer remain)
                        bonus *= 1.0 + (len(self.treasures) / self.initial_treasures)
                        self.rewards[agent] += bonus
                        self.proximity_count += 1
    
    def _log_episode_stats(self, success: bool):
        """Log statistics for the completed episode."""
        episode_length = self.steps_taken
        episode_reward = sum(self._cumulative_rewards.values()) / len(self.agents)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if self.debug:
            print(f"[EPISODE {self.episode_count}] "
                  f"Length: {episode_length}, "
                  f"Avg Reward: {episode_reward:.2f}, "
                  f"Treasures: {self.treasures_collected}/{self.initial_treasures}, "
                  f"Success: {success}")
    
    def render(self, mode: str = None):
        """Render the environment with optional path visualization."""
        if mode is None:
            mode = self.render_mode
        
        # Create a 2D grid for rendering by taking the maximum value across channels
        # This combines all channels into a single 2D grid where:
        # - 0: Empty
        # - 1: Agent positions (channel 0)
        # - 2: Treasure positions (channel 1)
        # - 3: Base position (channel 2)
        render_grid = np.argmax(self.grid, axis=0)  # Shape: (size, size)
        
        # Add agent paths if tracking
        if self.track_paths and hasattr(self, 'agent_paths'):
            for agent, path in self.agent_paths.items():
                for x, y in path:
                    if 0 <= x < self.size and 0 <= y < self.size and render_grid[x, y] == 0:
                        render_grid[x, y] = 4  # Use 4 for path
        
        if mode == "human":
            import matplotlib.pyplot as plt
            
            if not hasattr(self, 'fig'):
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                self.img = self.ax.imshow(render_grid, cmap='viridis', vmin=0, vmax=4)
                
                # Customize plot
                self.ax.set_title(f"Simple Grid Environment (Step: {getattr(self, 'steps_taken', 0)}/{self.max_steps})")
                
                # Create custom legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='black', label='Empty'),
                    Patch(facecolor='red', label='Agent'),
                    Patch(facecolor='blue', label='Base'),
                    Patch(facecolor='yellow', label='Treasure'),
                    Patch(facecolor='green', alpha=0.3, label='Agent Path')
                ]
                self.ax.legend(handles=legend_elements, loc='upper right')
                
                # Add grid lines
                self.ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
                self.ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
                self.ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
                
            # Update plot
            self.img.set_data(render_grid)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # Pause for rendering
            plt.pause(0.01)
        
        elif mode == 'rgb_array':
            # Create an RGB array (3 channels)
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            
            # Set empty cells to white
            img.fill(255)
            
            # Draw treasures (green)
            for treasure in self.treasures:
                img[treasure] = [0, 255, 0]  # Green for treasures
            
            # Draw base (blue)
            img[self.base_location] = [0, 0, 255]  # Blue for base
            
            # Draw agents (red)
            for pos in self.agent_positions.values():
                img[pos] = [255, 0, 0]  # Red for agents
            
            return img
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]


def env(**kwargs):
    """The env function wraps the environment in a wrapper that handles parallelization."""
    # Set default values if not provided
    if 'size' not in kwargs:
        kwargs['size'] = 10
    if 'num_treasures' not in kwargs:
        kwargs['num_treasures'] = 3
    if 'max_steps' not in kwargs:
        kwargs['max_steps'] = 200
    if 'render_mode' not in kwargs:
        kwargs['render_mode'] = None
    
    env = SimpleGridEnv(**kwargs)
    return env


if __name__ == "__main__":
    # Example usage
    env = SimpleGridEnv(size=5, num_treasures=3, render_mode="human")
    observations, infos = env.reset()
    
    print("Initial observations:")
    for agent, obs in observations.items():
        print(f"{agent} position: {obs['agent_pos']}")
    
    env.render()
    
    # Run a few steps with random actions
    for step in range(5):
        print(f"\nStep {step + 1}:")
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            print(f"{agent}:")
            print(f"  Action: {actions[agent]}")
            print(f"  Reward: {rewards[agent]}")
            print(f"  Position: {observations[agent]['agent_pos']}")
        
        if any(terminations.values()) or any(truncations.values()):
            print("Environment done!")
            break
