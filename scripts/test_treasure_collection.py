"""
Test script to verify treasure collection in the environment.

This script runs a series of test episodes to verify that:
1. Agents can collect treasures
2. Treasure collection is properly tracked
3. Rewards are correctly assigned
4. The environment state is updated correctly
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.pettingzoo_env_wrapper import SimpleGridEnv

def visualize_episode(episode_data, episode_num, grid_size):
    """Visualize agent paths and treasure collection during an episode."""
    plt.figure(figsize=(10, 8))
    
    # Create grid
    grid = np.zeros((grid_size, grid_size))
    
    # Plot treasure locations
    for treasure in episode_data['initial_treasures']:
        plt.scatter(treasure[1], grid_size - 1 - treasure[0], 
                   c='gold', s=500, marker='*', label='Treasure' if treasure == episode_data['initial_treasures'][0] else "")
    
    # Plot agent paths
    colors = ['red', 'blue']
    for i, (agent, path) in enumerate(episode_data['paths'].items()):
        if path:  # If path is not empty
            path = np.array(path)
            # Plot path
            plt.plot(path[:, 1], grid_size - 1 - path[:, 0], 
                    c=colors[i], alpha=0.5, label=f'{agent} path')
            # Mark start and end points
            plt.scatter(path[0, 1], grid_size - 1 - path[0, 0], 
                       c=colors[i], marker='o', s=100, label=f'{agent} start')
            plt.scatter(path[-1, 1], grid_size - 1 - path[-1, 0], 
                       c=colors[i], marker='X', s=100, label=f'{agent} end')
    
    # Plot treasure collection points
    for pos in episode_data['collected_treasures']:
        plt.scatter(pos[1], grid_size - 1 - pos[0], 
                   c='lime', s=200, marker='o', alpha=0.5, label='Collected' if pos == episode_data['collected_treasures'][0] else "")
    
    plt.title(f'Episode {episode_num} - Agent Paths and Treasure Collection')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Save the figure
    os.makedirs('test_visualizations', exist_ok=True)
    plt.savefig(f'test_visualizations/episode_{episode_num}_paths.png')
    plt.close()

def test_treasure_collection():
    """Test that agents can collect treasures and they are properly tracked."""
    # Create a small environment for testing with debug on
    env = SimpleGridEnv(size=5, num_treasures=3, max_steps=100, debug=True)
    
    # Test parameters
    num_episodes = 5
    success_count = 0
    total_steps = []
    
    print("\n=== Starting Treasure Collection Test ===")
    print(f"Running {num_episodes} test episodes...\n")
    
    # Run test episodes
    for episode in range(num_episodes):
        episode_data = {
            'initial_treasures': [],
            'collected_treasures': [],
            'paths': {agent: [] for agent in env.possible_agents},
            'rewards': defaultdict(list),
            'steps': 0
        }
        
        # Reset the environment
        obs, _ = env.reset()
        episode_data['initial_treasures'] = env.treasures.copy()
        
        print(f"\n=== Episode {episode + 1} ===")
        print(f"Starting treasures: {episode_data['initial_treasures']}")
        
        done = False
        step_count = 0
        
        while not done and step_count < 100:  # More steps to allow exploration
            # Store agent positions for visualization
            for agent in env.agents:
                episode_data['paths'][agent].append(env.agent_positions[agent])
            
            # Take random actions
            actions = {agent: env.action_space(agent).sample() 
                     for agent in env.agents}
            
            # Step the environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Track rewards and treasure collection
            for agent in env.agents:
                episode_data['rewards'][agent].append(rewards[agent])
                
                # Check for treasure collection
                if rewards[agent] > 0.5 * env.treasure_reward:  # Significant reward likely from treasure
                    pos = env.agent_positions[agent]
                    if pos in episode_data['initial_treasures'] and pos not in episode_data['collected_treasures']:
                        episode_data['collected_treasures'].append(pos)
                        print(f"Step {step_count}: {agent} collected treasure at {pos} "
                              f"(Reward: {rewards[agent]:.2f})")
            
            step_count += 1
            
            # Check if all treasures are collected
            if not env.treasures:
                print(f"✅ All treasures collected in {step_count} steps!")
                success_count += 1
                total_steps.append(step_count)
                done = True
            
            # Check if episode is done
            done = done or all(terminations.values()) or all(truncations.values())
        
        # Store episode data
        episode_data['steps'] = step_count
        
        # Print episode summary
        print(f"Episode {episode + 1} complete:")
        print(f"- Steps taken: {step_count}")
        print(f"- Treasures collected: {len(episode_data['collected_treasures'])}/{len(episode_data['initial_treasures'])}")
        print(f"- Total reward (agent_0): {sum(episode_data['rewards']['agent_0']):.2f}")
        print(f"- Total reward (agent_1): {sum(episode_data['rewards']['agent_1']):.2f}")
        
        # Visualize this episode
        visualize_episode(episode_data, episode + 1, env.size)
    
    # Print overall test results
    print("\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Successful episodes: {success_count} ({success_count/num_episodes*100:.1f}%)")
    if success_count > 0:
        print(f"Average steps to collect all treasures: {np.mean(total_steps):.1f}")
    
    return success_count == num_episodes
                

if __name__ == "__main__":
    test_treasure_collection()
