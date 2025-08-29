"""Training module for the GNTO project.

This module provides classes and utilities for training query optimization models
on PostgreSQL execution plans. It supports both traditional statistical methods
and GNN-based approaches.
"""

from .dataset import PlanDataset, create_plan_dataset
from .trainer import GNTOTrainer, TrainingConfig
from .metrics import calculate_metrics, MetricsTracker
from .utils import setup_training, save_model, load_model, set_random_seed

__all__ = [
    "PlanDataset",
    "create_plan_dataset", 
    "GNTOTrainer",
    "TrainingConfig",
    "calculate_metrics",
    "MetricsTracker",
    "setup_training",
    "save_model",
    "load_model",
    "set_random_seed"
]

def get_training_info() -> dict:
    """Get information about the training module."""
    return {
        'version': '1.0.0',
        'supported_targets': ['execution_time', 'cost', 'rows'],
        'supported_models': ['statistical', 'gcn', 'gat'],
        'metrics': ['mse', 'mae', 'mape', 'r2']
    }
