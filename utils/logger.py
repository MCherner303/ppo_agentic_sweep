"""Logging utilities for the project."""

import os
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir='logs'):
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create and return logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file.absolute()}")
    
    return logger

def log_metrics(logger, episode, metrics):
    """Log training metrics in a consistent format.
    
    Args:
        logger: Logger instance
        episode: Current episode number
        metrics: Dictionary of metrics to log
    """
    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Episode {episode}: {metric_str}")
