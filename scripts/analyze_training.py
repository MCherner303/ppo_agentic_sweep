import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_training_logs(log_dir="logs"):
    """Load all training logs and combine into a single DataFrame."""
    log_files = sorted(glob.glob(f"{log_dir}/training_*.csv"), reverse=True)
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {log_dir}")
    
    print(f"Loading training log: {log_files[0]}")
    df = pd.read_csv(log_files[0])
    return df

def analyze_training(df):
    """Analyze training metrics and generate insights."""
    # Basic statistics
    total_episodes = df['episode'].max()
    avg_reward = df.groupby('episode')['total_reward'].last().mean()
    max_reward = df.groupby('episode')['total_reward'].last().max()
    
    # Calculate curriculum stages
    curriculum_stages = [
        (1, 500, "Stage 1 (1 treasure)"),
        (501, 1000, "Stage 2 (2 treasures)"),
        (1001, 2000, "Stage 3 (3 treasures)")
    ]
    
    print("\n=== Training Analysis ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Average final reward: {avg_reward:.2f}")
    print(f"Maximum final reward: {max_reward:.2f}")
    
    # Analyze by curriculum stage
    print("\n=== Performance by Curriculum Stage ===")
    for start, end, name in curriculum_stages:
        stage_df = df[df['episode'].between(start, end)]
        if len(stage_df) == 0:
            continue
            
        avg_rew = stage_df.groupby('episode')['total_reward'].last().mean()
        max_rew = stage_df.groupby('episode')['total_reward'].last().max()
        print(f"{name} (Episodes {start}-{end}):")
        print(f"  Average reward: {avg_rew:.2f}")
        print(f"  Maximum reward: {max_rew:.2f}")
    
    return df

def plot_training_curves(df, save_dir="analysis"):
    """Plot training curves and save visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare episode-level metrics
    episode_metrics = df.groupby('episode').agg({
        'total_reward': 'last',
        'epsilon': 'first',
        'policy_loss': 'mean',
        'value_loss': 'mean',
        'entropy': 'mean'
    }).reset_index()
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_metrics['episode'], 
             episode_metrics['total_reward'].rolling(20).mean(),
             label='20-episode moving average')
    
    # Add curriculum stage indicators
    for x in [1, 501, 1001, 2000]:
        plt.axvline(x=x, color='r', linestyle='--', alpha=0.3)
    
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rewards.png")
    
    # Plot losses and entropy
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(episode_metrics['episode'], 
             episode_metrics['policy_loss'].rolling(20).mean(),
             label='Policy Loss')
    plt.plot(episode_metrics['episode'], 
             episode_metrics['value_loss'].rolling(20).mean(),
             label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(episode_metrics['episode'], 
             episode_metrics['entropy'].rolling(20).mean(),
             label='Entropy', color='green')
    plt.title('Policy Entropy')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/losses_entropy.png")
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 4))
    plt.plot(episode_metrics['episode'], 
             episode_metrics['epsilon'],
             label='Epsilon', color='purple')
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epsilon.png")

def main():
    try:
        # Load and analyze training data
        df = load_training_logs()
        df = analyze_training(df)
        
        # Generate visualizations
        plot_training_curves(df)
        
        print("\nAnalysis complete! Check the 'analysis' directory for visualizations.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
