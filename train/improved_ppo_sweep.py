#!/usr/bin/env python3
"""
Improved PPO Hyperparameter Sweep with Early Stopping and Multi-Seed Support.

Features:
- Early stopping based on reward improvement
- Multi-seed validation for each configuration
- Greedy evaluation (ε=0) for better performance assessment
- Progress monitoring and automatic pruning of underperforming runs
"""
import os
import sys
import time
import random
import argparse
import subprocess
import numpy as np
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class PPOConfig:
    """PPO hyperparameter configuration."""
    def __init__(self, **kwargs):
        # Default hyperparameters
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.clip_param = 0.2
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.num_mini_batches = 4
        self.ppo_epochs = 4
        self.batch_size = 2048
        self.num_steps = 2048
        self.max_grad_norm = 0.5
        self.value_loss_coef = 0.5
        self.gamma = 0.99
        self.eps = 0.2
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_args(self) -> List[str]:
        """Convert config to command-line arguments."""
        args = []
        for key, value in self.__dict__.items():
            if key == 'eps':
                args.extend([f'--eps', str(value)])
            else:
                arg_name = '--' + key.replace('_', '-')
                args.extend([arg_name, str(value)])
        return args
    
    def __str__(self):
        return "_".join(f"{k[:4]}{v:.0e}" for k, v in sorted(self.__dict__.items()))


def generate_hyperparameter_grid() -> List[Dict[str, Any]]:
    """Generate a grid of hyperparameter configurations to test."""
    grid = {
        # Narrowed learning rates based on previous results
        'actor_lr': [7e-5],
        'critic_lr': [7e-5, 1e-4],
        
        # Refine clip parameter
        'clip_param': [0.1, 0.15],
        
        # More precise GAE lambda and entropy tuning
        'gae_lambda': [0.92, 0.95],
        'entropy_coef': [0.0, 0.005],
        
        # Other parameters
        'num_mini_batches': [4],
        'batch_size': [2048],
        'ppo_epochs': [4],
        'num_steps': [2048],
    }
    
    # Generate all combinations
    from itertools import product
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in product(*values)]


class PPOSweep:
    """PPO Hyperparameter Sweep with Early Stopping and Multi-Seed Support."""
    
    def __init__(self, config: PPOConfig, log_dir: str, num_seeds: int = 3):
        self.config = config
        self.log_dir = Path(log_dir)
        self.num_seeds = num_seeds
        self.seeds = [random.randint(0, 10000) for _ in range(num_seeds)]
        self.results = []
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_seed(self, seed: int) -> Dict[str, Any]:
        """Run training for a single seed and return results."""
        log_file = self.log_dir / f"seed_{seed}.log"
        # Remove --eps and its value from args (not supported by train_ppo.py)
        raw_args = self.config.to_args()
        args = []
        skip = False
        for i, a in enumerate(raw_args):
            if skip:
                skip = False
                continue
            if a == '--eps':
                skip = True  # skip this and the next item (the value)
                continue
            args.append(a)
        cmd_python = str(Path(__file__).parent.parent / ".venv/bin/python")
        # Diagnostic: print python version and pip list before running train_ppo.py
        print(f"[DIAG] Using Python: {cmd_python}")
        os.system(f'{cmd_python} --version')
        os.system(f'{cmd_python} -m pip list')
        cmd = [
            cmd_python, "train/train_ppo.py",
            *args,
            "--seed", str(seed),
            "--log-dir", str(self.log_dir),
            "--num-episodes", "500",  # Will stop early if no improvement
            "--eval-interval", "100",  # Greedy evaluation every 100 episodes
            "--eval-episodes", "5",    # 5 evaluation episodes per eval
            "--eval-eps", "0.0",      # Set epsilon to 0 for evaluation
        ]
        
        print(f"Starting training with seed {seed}...")
        print(" ".join(cmd))
        
        with open(log_file, 'w') as f:
            venv_dir = str(Path(__file__).parent.parent / ".venv")
            venv_bin = str(Path(venv_dir) / "bin")
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = venv_dir
            env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
            # Optionally set PYTHONHOME and PYTHONPATH if needed (not usually required)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Monitor output for early stopping
            best_reward = -np.inf
            last_improvement = 0
            episode_rewards = []
            error_lines = []

            for line in process.stdout:
                f.write(line)
                f.flush()

                # Save error lines for later (last 10 lines)
                error_lines.append(line)
                if len(error_lines) > 10:
                    error_lines.pop(0)

                # Parse episode information
                if "=== Episode" in line and "Reward:" in line:
                    try:
                        ep_num = int(line.split("Episode")[1].split()[0])
                        reward = float(line.split("Reward:")[1].split(",")[0])
                        episode_rewards.append(reward)

                        # Check for improvement
                        if reward > best_reward + 0.1:  # Small threshold
                            best_reward = reward
                            last_improvement = ep_num

                        # Early stopping check
                        if (ep_num - last_improvement) >= 400 and ep_num >= 250:
                            print(f"\n⚠️  Early stopping at episode {ep_num} (no improvement for {ep_num - last_improvement} episodes)")
                            process.terminate()
                            break
                    except Exception:
                        print(f"Error parsing line: {line.strip()}")
                        continue
            process.wait()
            exit_code = process.returncode

            # If the process failed, mark as failed and capture error
            if exit_code != 0 or len(episode_rewards) == 0:
                error_msg = f"Process exited with code {exit_code}. Last log lines:\n" + ''.join(error_lines)
                # Guard: If import error, print and exit sweep
                if any('ModuleNotFoundError' in line or 'ImportError' in line for line in error_lines):
                    print("\n[SWEEP GUARD] Import error detected in subprocess. Stopping further launches.\n")
                    print(error_msg)
                    # Exit the sweep entirely
                    import sys
                    sys.exit(1)
                return {
                    'seed': seed,
                    'best_reward': -np.inf,
                    'final_reward': -np.inf,
                    'episodes': 0,
                    'avg_last_100': float('nan'),
                    'log_file': str(log_file),
                    'status': 'failed',
                    'error': error_msg
                }
            # Return training statistics
            return {
                'seed': seed,
                'best_reward': best_reward,
                'final_reward': episode_rewards[-1] if episode_rewards else -np.inf,
                'episodes': len(episode_rewards),
                'avg_last_100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                'log_file': str(log_file),
                'status': 'success',
                'error': None
            }
    
    def run(self) -> Dict[str, Any]:
        """Run the hyperparameter sweep across all seeds."""
        print(f"\n{'='*80}")
        print(f"Starting PPO Sweep for config: {self.config}")
        print(f"Logging to: {self.log_dir}")
        print(f"Seeds: {self.seeds}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Run training for each seed
        for seed in self.seeds:
            result = self.run_single_seed(seed)
            self.results.append(result)
            
            print(f"\n{'='*50}")
            print(f"Seed {seed} Results:")
            print(f"  Best Reward: {result['best_reward']:.2f}")
            print(f"  Final Reward: {result['final_reward']:.2f}")
            print(f"  Episodes: {result['episodes']}")
            print(f"  Avg Last 100: {result['avg_last_100']:.2f}")
            print(f"  Log: {result['log_file']}")
            print(f"{'='*50}\n")
        
        # Calculate summary statistics
        avg_best = np.mean([r['best_reward'] for r in self.results])
        avg_final = np.mean([r['final_reward'] for r in self.results])
        avg_episodes = np.mean([r['episodes'] for r in self.results])
        
        summary = {
            'config': self.config.__dict__,
            'avg_best_reward': avg_best,
            'avg_final_reward': avg_final,
            'avg_episodes': avg_episodes,
            'seeds': self.seeds,
            'results': self.results,
            'duration_seconds': time.time() - start_time
        }
        
        # Save summary to file
        import json
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Sweep Complete!")
        print(f"Average Best Reward: {avg_best:.2f}")
        print(f"Average Final Reward: {avg_final:.2f}")
        print(f"Average Episodes: {avg_episodes:.1f}")
        print(f"Duration: {(time.time() - start_time)/3600:.2f} hours")
        print(f"Summary saved to: {self.log_dir / 'summary.json'}")
        print(f"{'='*80}\n")
        
        return summary

def run_config_job(i, config, config_dir, total, num_seeds):
    config_str = str(config)
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*80}")
    print(f"Running Configuration {i+1}/{total}")
    print(f"Config: {config}")
    print(f"Logging to: {config_dir}")
    print(f"{'#'*80}\n")
    sweep = PPOSweep(config, config_dir, num_seeds=num_seeds)
    result = sweep.run()
    return result

def run_grid_search():
    import concurrent.futures
    parser = argparse.ArgumentParser(description='PPO Hyperparameter Sweep')
    parser.add_argument('--log-dir', type=str, default='sweep_logs',
                      help='Base directory for logs')
    parser.add_argument('--num-seeds', type=int, default=3,
                      help='Number of random seeds per config')
    parser.add_argument('--max-configs', type=int, default=None,
                      help='Maximum number of configurations to try')
    parser.add_argument('--resume', action='store_true', help='Resume sweep: skip configs with summary.json')
    parser.add_argument('--max-workers', type=int, default=2, help='Number of concurrent configs to run')
    args = parser.parse_args()

    # Generate all possible configurations
    configs = [PPOConfig(**params) for params in generate_hyperparameter_grid()]

    if args.max_configs and len(configs) > args.max_configs:
        print(f"Randomly sampling {args.max_configs} configurations from {len(configs)} possible...")
        configs = random.sample(configs, args.max_configs)

    # Prepare jobs (skip those already completed if resume)
    jobs = []
    for i, config in enumerate(configs):
        config_str = str(config)
        config_dir = Path(args.log_dir) / f"sweep_{config_str}"
        summary_file = config_dir / 'summary.json'
        if args.resume and summary_file.exists():
            print(f"[RESUME] Skipping config {i+1}/{len(configs)} ({config_str}) -- already complete.")
            continue
        jobs.append((i, config, config_dir, len(configs), args.num_seeds))

    results = []
    results_path = Path(args.log_dir) / 'all_results.json'

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(run_config_job, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                # Write summary after each config
            except Exception as exc:
                print(f"[ERROR] Config {job[1]} generated an exception: {exc}")

    # Filter out -inf/NaN results
    filtered_results = [result for result in results if not np.isinf(result['avg_best_reward']) and not np.isnan(result['avg_best_reward'])]

    # Sort by average best reward
    filtered_results.sort(key=lambda x: x['avg_best_reward'], reverse=True)

    # Write top 10 configs to file
    top_configs_path = Path(args.log_dir) / 'top_configs.json'
    with open(top_configs_path, 'w') as f:
        json.dump([result['config'] for result in filtered_results[:10]], f, indent=2)

    # Print final summary
    print("\n" + "="*80)
    print("Hyperparameter Sweep Complete!")
    print(f"Tested {len(results)} configurations with {args.num_seeds} seeds each")

    print("\nTop 10 Configurations:")
    for i, result in enumerate(filtered_results[:10]):
        print(f"{i+1}. {result['config']}")
        print(f"   Avg Best Reward: {result['avg_best_reward']:.2f}")
        print(f"   Logs: {Path(result['results'][0]['log_file']).parent}")
        print()

    print(f"\nTop 10 configs written to: {top_configs_path}")

    # Print failed runs summary
    print("\nFailed Runs Summary:")
    any_failed = False
    for result in results:
            if seed_result.get('status') == 'failed':
                any_failed = True
                print(f"Config: {result['config']}, Seed: {seed_result['seed']}")
                print(f"  Error: {seed_result.get('error', '')[:500]}")
                print(f"  Log file: {seed_result['log_file']}")
                print()
    if not any_failed:
        print("No failed runs detected!")
    print("="*80)
    return results

if __name__ == "__main__":
    run_grid_search()
