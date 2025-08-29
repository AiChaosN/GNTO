"""Utility functions for training."""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def setup_training(output_dir: str, model_name: str) -> Path:
    """Setup training environment.
    
    Args:
        output_dir: Output directory for training artifacts
        model_name: Name of the model being trained
        
    Returns:
        Path to the created output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_path / f"{model_name}_training.log"
    
    # Create file handler if not already exists
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) 
              for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Training environment setup at {output_path}")
    return output_path


def save_model(model, filepath: str, metadata: Optional[Dict[str, Any]] = None):
    """Save a trained model to disk.
    
    Args:
        model: Model object to save
        filepath: Path to save the model
        metadata: Optional metadata to save with model
    """
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object and metadata
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data.get('model')
    metadata = model_data.get('metadata', {})
    
    logger.info(f"Model loaded from {filepath}")
    return model, metadata


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable way.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(num: float, precision: int = 4) -> str:
    """Format a number for display.
    
    Args:
        num: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision-2}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision-2}f}K"
    else:
        return f"{num:.{precision}f}"


def create_config_summary(config) -> str:
    """Create a summary string of training configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        Formatted configuration summary
    """
    summary_lines = [
        "=== Training Configuration ===",
        f"Model Type: {config.model_type}",
        f"Node Encoder: {config.node_encoder_type}",
        f"Hidden Dim: {config.hidden_dim}",
        f"Learning Rate: {config.learning_rate}",
        f"Batch Size: {config.batch_size}",
        f"Epochs: {config.num_epochs}",
        f"Target: {config.target_column}",
        f"Output Dir: {config.output_dir}",
        "=" * 30
    ]
    
    return "\n".join(summary_lines)


def validate_config(config) -> bool:
    """Validate training configuration.
    
    Args:
        config: Training configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required attributes
    required_attrs = [
        'model_type', 'learning_rate', 'num_epochs', 
        'target_column', 'output_dir'
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Missing required configuration attribute: {attr}")
    
    # Validate values
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if config.num_epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if config.model_type not in ['statistical', 'gcn', 'gat']:
        raise ValueError(f"Invalid model type: {config.model_type}")
    
    logger.info("Configuration validation passed")
    return True
