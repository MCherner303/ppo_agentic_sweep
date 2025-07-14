#!/usr/bin/env python3
"""
Test script to manually verify treasure pickup mechanics in the environment.
"""
import sys
import os
import random
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.pettingzoo_env_wrapper import SimpleGridEnv

def print_environment_state(env, agent_positions):
    """Print a visual representation of the environment."""
    grid = np.zeros((env.size, env.size), dtype=str)
    grid.fill('.')
    
    # Mark treasures
    for treasure in env.treasures:
        grid[treasure] = 'T'
    
    # Mark agents
    for i, (agent, pos) in enumerate(agent_positions.items()):
        if grid[pos] == 'T':
            grid[pos] = f"{i+1}T"  # Agent on treasure
        else:
            grid[pos] = str(i+1)    # Agent on empty cell
    
    # Mark base
    if grid[env.base_location] == '.':
        grid[env.base_location] = 'B'
    
    # Print the grid
    print("\n" + "=" * (env.size * 2 + 3))
    print(" " + "-" * (env.size * 2 + 1))
    for row in grid:
        print("|", end=" ")
        print(" ".join(row), end=" ")
        print("|")
    print(" " + "-" * (env.size * 2 + 1))
    print("=" * (env.size * 2 + 3) + "\n")

def manual_test():
    """Run a manual test of the environment with user input."""
    # Create environment with debug output
    print("Creating environment...")
    env = SimpleGridEnv(
        size=5,  # Smaller grid for testing
        num_treasures=2,
        max_steps=50,
        render_mode="human",
        debug=True
    )
    
    # Reset environment
    print("\nResetting environment...")
    observations, _ = env.reset()
    
    # Print initial state
    print("\nInitial state:")
    print(f"Agent positions: {env.agent_positions}")
    print(f"Treasure positions: {env.treasures}")
    print_environment_state(env, env.agent_positions)
    
    # Manual control loop
    step = 0
    done = False
    
    print("\n=== Manual Control ===")
    print("Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=No-op")
    print("Enter 'q' to quit, 'r' to reset, 's' to step with random actions")
    
    while not done and step < 50:
        # Get user input
        user_input = input(f"\nStep {step} - Enter action (or 'h' for help): ").strip().lower()
        
        if user_input == 'q':
            print("Quitting...")
            break
            
        if user_input == 'h':
            print("\n=== Help ===")
            print("0: Up")
            print("1: Down")
            print("2: Left")
            print("3: Right")
            print("4: No-op")
            print("s: Step with random actions")
            print("r: Reset environment")
            print("q: Quit")
            print("h: Show this help")
            continue
            
        if user_input == 'r':
            print("\nResetting environment...")
            observations, _ = env.reset()
            step = 0
            print_environment_state(env, env.agent_positions)
            continue
            
        # Generate actions
        actions = {}
        for agent in env.agents:
            if user_input == 's':
                # Random action
                action = random.randint(0, 4)
            else:
                try:
                    action = int(user_input)
                    if action < 0 or action > 4:
                        print(f"Invalid action: {action}. Must be 0-4.")
                        continue
                except ValueError:
                    print(f"Invalid input: {user_input}. Try again.")
                    continue
            
            actions[agent] = action
            print(f"{agent} taking action: {action}")
        
        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Print results
        print(f"\n=== Step {step} ===")
        print(f"Agent positions: {env.agent_positions}")
        print(f"Treasure positions: {env.treasures}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
        
        # Print environment state
        print_environment_state(env, env.agent_positions)
        
        # Check if done
        done = all(terminations.values()) or all(truncations.values())
        step += 1
    
    print("\nTest complete!")

if __name__ == "__main__":
    manual_test()
