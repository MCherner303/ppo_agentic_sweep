#!/usr/bin/env python3
"""
Hyperparameter sweep script for PPO training.

This script performs a grid search over specified hyperparameter ranges
and launches training runs with different configurations.
"""
import os
import subprocess
import argparse
import itertools
from typing import List, Dict, Any
import random
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for PPO training')
    
    # Sweep configuration
    parser.add_argument('--sweep-name', type=str, default='sweep',
                      help='Base name for this sweep')
    parser.add_argument('--num-seeds', type=int, default=2,
                      help='Number of random seeds to try per config')
    parser.add_argument('--max-concurrent', type=int, default=2,
                      help='Maximum number of concurrent training runs')
    parser.add_argument('--gpu-ids', type=str, default='0',
                      help='Comma-separated list of GPU IDs to use')
    
    # Base command
    parser.add_argument('--base-cmd', type=str, default='python train/train_ppo.py',
                      help='Base command to run')
    
    # Training configuration
    parser.add_argument('--num-treasures', type=int, default=1,
                      help='Number of treasures in the environment')
    parser.add_argument('--num-episodes', type=int, default=100,
                      help='Number of episodes per training run')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='sweep_logs',
                      help='Directory to store sweep logs')
    
    return parser.parse_args()

def get_hyperparameter_grid():
    """Define the hyperparameter grid for the sweep."""
    return {
        # Learning rates - narrowed down based on previous sweeps
        '--actor-lr': [3e-5, 5e-5],
        '--critic-lr': [3e-5, 5e-5],
        
        # PPO parameters - more stable ranges
        '--clip-param': [0.1, 0.2],
        '--gae-lambda': [0.95, 0.97],
        '--entropy-coef': [0.01, 0.02],
        '--num-mini-batches': [4, 8],
        '--ppo-epochs': [4],  # Fixed based on PPO paper
        '--batch-size': [2048],  # Fixed for stability
        '--num-steps': [2048],  # Fixed for consistency
        '--max-grad-norm': [0.5],  # Add gradient clipping
        '--value-loss-coef': [0.5],  # Add value loss coefficient
        
        # Environment
        '--num-treasures': [3],  # Fixed for consistency
        '--num-episodes': [500],  # More episodes for better convergence
    }

def sample_config(grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Sample a random configuration from the grid."""
    return {param: random.choice(values) for param, values in grid.items()}

# Remove the unused config_to_args function as we're handling arguments directly in run_sweep

def run_sweep():
    args = parse_args()
    
    # Create log directory
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    sweep_dir = os.path.join(args.log_dir, f"{args.sweep_name}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Get hyperparameter grid
    grid = get_hyperparameter_grid()
    
    # Get GPU IDs
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',') if gpu_id.strip()]
    if not gpu_ids:
        gpu_ids = [0]  # Default to GPU 0 if none specified
    
    # Create all possible configurations
    param_names = list(grid.keys())
    param_values = list(grid.values())
    configs = list(itertools.product(*param_values))
    
    print(f"Starting hyperparameter sweep with {len(configs)} configurations and {args.num_seeds} seeds each...")
    print(f"Logging to: {sweep_dir}")
    
    # Track running processes
    running_processes = []
    
    try:
        # For each configuration
        for i, config_values in enumerate(configs):
            config = dict(zip(param_names, config_values))
            
            # For each seed
            for seed in range(args.num_seeds):
                # Wait if we've reached max concurrent processes
                while len(running_processes) >= args.max_concurrent:
                    # Check for completed processes
                    running_processes = [p for p in running_processes if p.poll() is None]
                    time.sleep(1)
                
                # Get next available GPU
                gpu_id = gpu_ids[len(running_processes) % len(gpu_ids)]
                
                # Create a unique run ID
                run_id = f"sweep_{i}_s{seed}"
                
                # Build the base command
                cmd = [
                    'python', 'train/train_ppo.py',
                    '--num-treasures', str(args.num_treasures),
                    '--num-episodes', str(args.num_episodes),
                    '--seed', str(seed + 42),  # Start from 42 to be different from default
                    '--run-id', run_id
                ]
                
                # Add config parameters
                for param, value in config.items():
                    # Skip None values
                    if value is None:
                        continue
                        
                    # Remove any leading dashes from parameter names
                    clean_param = param.lstrip('-')
                    
                    # Handle boolean values (only include if True)
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{clean_param.replace('_', '-')}")
                    # Handle all other values
                    else:
                        cmd.extend([f"--{clean_param.replace('_', '-')}", str(value)])
                
                # Explicitly disable W&B by not including the flag at all
                # (default in train_ppo.py is use_wandb=False)
                
                # Create log file
                log_file = os.path.join(sweep_dir, f"{run_id}.log")
                
                # Prepare environment with CUDA_VISIBLE_DEVICES
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                print(f"\nLaunching {run_id} on GPU {gpu_id}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Log file: {log_file}")
                
                # Add debug output
                print(f"\nCommand to execute:")
                print(' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd))
                print(f"Environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
                
                # Run the command with output to log file
                with open(log_file, 'w') as f:
                    # Write the command to the log file for reference
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"Environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}\n")
                    f.write("-" * 80 + "\n")
                    f.flush()
                    
                    process = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env
                    )
                
                running_processes.append(process)
                time.sleep(5)  # Small delay between starts
        
        # Wait for all processes to complete
        print("\nWaiting for all runs to complete...")
        for process in running_processes:
            process.wait()
        
        print("\nSweep completed!")
        
    except KeyboardInterrupt:
        print("\nSweep interrupted. Terminating all processes...")
        for process in running_processes:
            process.terminate()
        print("All processes terminated.")
        sys.exit(1)

if __name__ == "__main__":
    run_sweep()
