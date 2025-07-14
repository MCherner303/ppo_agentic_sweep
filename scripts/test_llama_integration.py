#!/usr/bin/env python3
"""Test script to verify LLaMA integration with the environment."""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.pettingzoo_env_wrapper import SimpleGridEnv
from agents.a2c_agent import create_agents

def test_llama_integration():
    """Test the LLaMA integration with the environment."""
    # Configuration
    config = {
        'grid_size': 10,
        'num_agents': 2,
        'num_treasures': 3,
        'max_steps': 100,
        'num_episodes': 5,
        'use_llama': True  # Enable LLaMA model
    }
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = SimpleGridEnv(
        size=config['grid_size'],
        num_treasures=config['num_treasures'],
        max_steps=config['max_steps'],
        render_mode=None
    )
    
    # Create agents with LLaMA model
    agents = create_agents(
        grid_size=config['grid_size'],
        num_agents=config['num_agents'],
        use_llama=config['use_llama']
    )
    
    print(f"Testing LLaMA integration with {len(agents)} agents")
    print(f"Environment: {config['grid_size']}x{config['grid_size']} grid, {config['num_treasures']} treasures")
    
    # Run test episodes
    for episode in range(config['num_episodes']):
        obs = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        done = False
        step = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done and step < config['max_steps']:
            actions = {}
            
            # Get actions from all agents
            for agent_id, agent in agents.items():
                if agent_id in obs:  # Agent is active
                    # Get action from LLaMA model
                    action, _ = agent.select_action(obs[agent_id], explore=False)
                    actions[agent_id] = action
            
            # Step the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
            done = all(dones.values())
            
            # Update rewards
            for agent_id in agents.keys():
                if agent_id in rewards:  # Agent is active
                    episode_rewards[agent_id] += rewards[agent_id]
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}: Rewards: {episode_rewards}")
                
            obs = next_obs
            step += 1
        
        # Print episode summary
        print(f"Episode {episode + 1} completed in {step} steps")
        print(f"Final rewards: {episode_rewards}")
        
        # Print treasure collection status
        for agent_id, agent in agents.items():
            if hasattr(agent, 'treasures_collected'):
                print(f"{agent_id} collected {agent.treasures_collected} treasures")
    
    print("\nLLaMA integration test completed!")

if __name__ == "__main__":
    test_llama_integration()
