#!/usr/bin/env python3
"""Main entry point for the multi-agent treasure collection training."""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from train.train_loop import Config, Trainer
from utils.logger import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-agent A2C for treasure collection')
    
    # Environment arguments
    parser.add_argument('--grid-size', type=int, default=10,
                      help='Size of the grid environment')
    parser.add_argument('--num-treasures', type=int, default=3,
                      help='Number of treasures to collect')
    parser.add_argument('--max-steps', type=int, default=200,
                      help='Maximum number of steps per episode')
    
    # Training arguments
    parser.add_argument('--num-episodes', type=int, default=1000,
                      help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training (cuda or cpu)')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory to save logs and models')
    parser.add_argument('--save-interval', type=int, default=100,
                      help='Save model every N episodes')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting training with the following configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Create configuration
    config = Config()
    
    # Update config with command line arguments
    config.GRID_SIZE = args.grid_size
    config.NUM_TREASURES = args.num_treasures
    config.MAX_STEPS = args.max_steps
    config.NUM_EPISODES = args.num_episodes
    config.BATCH_SIZE = args.batch_size
    config.GAMMA = args.gamma
    config.LEARNING_RATE = args.lr
    config.DEVICE = args.device
    config.SAVE_INTERVAL = args.save_interval
    config.LOG_DIR = args.log_dir
    
    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
