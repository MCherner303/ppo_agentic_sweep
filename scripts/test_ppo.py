import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train.train_ppo import PPOTrainConfig, PPOTrainer

def test_ppo_training():
    # Create a test configuration
    class TestConfig(PPOTrainConfig):
        # Reduce training parameters for testing
        NUM_EPISODES = 5  # Very small number for quick test
        NUM_STEPS = 64    # Small number of steps per update
        BATCH_SIZE = 32   # Small batch size
        
        # Disable video saving for test
        SAVE_VIDEO = False
        RENDER_MODE = None
        
        # Use CPU for consistent testing
        DEVICE = "cpu"
        
        # Simplify curriculum for testing
        CURRICULUM_STAGES = [
            {'treasures': 1, 'episodes': 5}
        ]
    
    print("Starting PPO test...")
    
    # Initialize and run trainer
    config = TestConfig()
    trainer = PPOTrainer(config)
    
    try:
        trainer.train()
        print("\nPPO test completed successfully!")
        return True
    except Exception as e:
        print(f"\nError during PPO test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ppo_training()
    sys.exit(0 if success else 1)
