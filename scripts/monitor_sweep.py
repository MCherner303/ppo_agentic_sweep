#!/usr/bin/env python3
"""
Monitor the progress of a hyperparameter sweep.

This script tracks the training progress by monitoring the log files
and reports key metrics like reward trends and episode lengths.
"""
import os
import time
import glob
import re
from datetime import datetime
import numpy as np
from collections import deque

class SweepMonitor:
    def __init__(self, log_dir, window_size=50):
        self.log_dir = log_dir
        self.window_size = window_size
        self.last_episodes = {}
        self.reward_history = {}
        self.length_history = {}
        
    def find_latest_logs(self):
        """Find all log files from the current sweep."""
        return glob.glob(f"{self.log_dir}/sweep_*.log")
    
    def parse_log_file(self, log_file):
        """Parse a log file and extract episode information."""
        if log_file not in self.last_episodes:
            self.last_episodes[log_file] = 0
            self.reward_history[log_file] = deque(maxlen=1000)
            self.length_history[log_file] = deque(maxlen=1000)
            
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            episodes = []
            for i, line in enumerate(lines):
                if '=== Episode ' in line:
                    try:
                        ep_num = int(line.split('Episode ')[1].split(' ===')[0])
                        if i + 1 < len(lines) and 'Reward:' in lines[i+1]:
                            reward = float(lines[i+1].split('Reward: ')[1].split(',')[0])
                            length = float(lines[i+1].split('Length: ')[1].strip())
                            episodes.append((ep_num, reward, length))
                    except (IndexError, ValueError) as e:
                        continue
            
            # Only process new episodes
            new_episodes = [ep for ep in episodes if ep[0] > self.last_episodes[log_file]]
            if new_episodes:
                self.last_episodes[log_file] = max(ep[0] for ep in new_episodes)
                for ep_num, reward, length in new_episodes:
                    self.reward_history[log_file].append(reward)
                    self.length_history[log_file].append(length)
            
            return new_episodes
            
        except FileNotFoundError:
            print(f"Warning: Log file {log_file} not found")
            return []
    
    def analyze_progress(self):
        """Analyze the training progress across all log files."""
        all_rewards = []
        all_lengths = []
        
        for log_file in self.reward_history:
            if self.reward_history[log_file]:
                all_rewards.extend(self.reward_history[log_file])
                all_lengths.extend(self.length_history[log_file])
        
        if not all_rewards:
            return None
            
        stats = {
            'current_episode': max(self.last_episodes.values(), default=0),
            'avg_reward': np.mean(all_rewards[-self.window_size:]),
            'max_reward': max(all_rewards) if all_rewards else 0,
            'avg_length': np.mean(all_lengths[-self.window_size:]) if all_lengths else 0,
            'total_episodes': len(all_rewards)
        }
        
        # Check for improvement
        if len(all_rewards) > self.window_size:
            prev_avg = np.mean(all_rewards[-2*self.window_size:-self.window_size])
            stats['improvement'] = stats['avg_reward'] - prev_avg
        
        return stats

def monitor_sweep(log_dir, check_interval=300):
    """Monitor the sweep progress at regular intervals."""
    monitor = SweepMonitor(log_dir)
    last_check = time.time()
    
    print(f"Monitoring sweep in {log_dir}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Check for new log files
            log_files = monitor.find_latest_logs()
            
            # Parse all log files
            for log_file in log_files:
                monitor.parse_log_file(log_file)
            
            # Analyze progress
            stats = monitor.analyze_progress()
            if stats:
                print(f"\n=== Sweep Status at {datetime.now().strftime('%H:%M:%S')} ===")
                print(f"Current Episode: {stats['current_episode']}")
                print(f"Average Reward (last {monitor.window_size} eps): {stats['avg_reward']:.2f}")
                print(f"Max Reward: {stats['max_reward']:.2f}")
                print(f"Average Length: {stats['avg_length']:.1f} steps")
                
                if 'improvement' in stats:
                    trend = "↑" if stats['improvement'] > 0 else "↓"
                    print(f"Trend: {trend} {abs(stats['improvement']):.2f} (last {monitor.window_size} eps)")
                
                # Check for early stopping conditions
                if stats['current_episode'] >= 250 and stats['max_reward'] < 1.0:
                    print("\n⚠️  WARNING: Training may have plateaued. Consider early stopping.")
            
            # Wait for next check
            time.sleep(max(0, check_interval - (time.time() - last_check) % check_interval))
            last_check = time.time()
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor hyperparameter sweep progress')
    parser.add_argument('--log-dir', type=str, default='sweep_logs/sweep_20250710-162410',
                      help='Directory containing sweep logs')
    parser.add_argument('--interval', type=int, default=60,
                      help='Check interval in seconds (default: 60)')
    
    args = parser.parse_args()
    monitor_sweep(args.log_dir, args.interval)
