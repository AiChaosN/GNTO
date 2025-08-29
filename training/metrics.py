"""Metrics and evaluation utilities for training."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metric names and values
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    metrics['mean_error'] = np.mean(y_pred - y_true)
    metrics['std_error'] = np.std(y_pred - y_true)
    
    # Q-error (geometric mean of ratios)
    ratios = np.maximum(y_pred / np.maximum(y_true, 1e-10), 
                       np.maximum(y_true, 1e-10) / y_pred)
    metrics['q_error_mean'] = np.mean(ratios)
    metrics['q_error_95th'] = np.percentile(ratios, 95)
    metrics['q_error_99th'] = np.percentile(ratios, 99)
    
    return metrics


class MetricsTracker:
    """Track metrics across training epochs and splits."""
    
    def __init__(self, metric_names: Optional[List[str]] = None):
        """Initialize metrics tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names or ['mse', 'mae', 'r2']
        self.history: Dict[str, List[float]] = {
            f'train_{name}': [] for name in self.metric_names
        }
        self.history.update({
            f'val_{name}': [] for name in self.metric_names
        })
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch = 0
    
    def update(self, 
               train_metrics: Dict[str, float], 
               val_metrics: Optional[Dict[str, float]] = None,
               epoch: int = 0):
        """Update metrics for current epoch.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            epoch: Current epoch number
        """
        # Update training metrics
        for name in self.metric_names:
            if name in train_metrics:
                self.history[f'train_{name}'].append(train_metrics[name])
        
        # Update validation metrics
        if val_metrics:
            for name in self.metric_names:
                if name in val_metrics:
                    self.history[f'val_{name}'].append(val_metrics[name])
        
        # Track best validation metrics (lower is better for mse, mae; higher for r2)
        if val_metrics:
            # Use validation R2 as primary metric (higher is better)
            current_r2 = val_metrics.get('r2', -float('inf'))
            best_r2 = self.best_metrics.get('val_r2', -float('inf'))
            
            if current_r2 > best_r2:
                self.best_epoch = epoch
                for name in self.metric_names:
                    if name in val_metrics:
                        self.best_metrics[f'val_{name}'] = val_metrics[name]
                    if name in train_metrics:
                        self.best_metrics[f'train_{name}'] = train_metrics[name]
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get most recent metrics."""
        current = {}
        for key, values in self.history.items():
            if values:
                current[key] = values[-1]
        return current
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best validation metrics."""
        return self.best_metrics.copy()
    
    def print_summary(self):
        """Print training summary."""
        current = self.get_current_metrics()
        best = self.get_best_metrics()
        
        logger.info("=== Training Summary ===")
        logger.info(f"Best epoch: {self.best_epoch}")
        
        if best:
            logger.info("Best validation metrics:")
            for name in self.metric_names:
                key = f'val_{name}'
                if key in best:
                    logger.info(f"  {name}: {best[key]:.6f}")
        
        if current:
            logger.info("Final metrics:")
            for name in self.metric_names:
                train_key = f'train_{name}'
                val_key = f'val_{name}'
                
                train_val = current.get(train_key, 'N/A')
                val_val = current.get(val_key, 'N/A')
                
                if isinstance(train_val, float):
                    train_val = f"{train_val:.6f}"
                if isinstance(val_val, float):
                    val_val = f"{val_val:.6f}"
                
                logger.info(f"  {name} - Train: {train_val}, Val: {val_val}")
    
    def save_history(self, filepath: str):
        """Save training history to file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                       for v in values]
        
        with open(filepath, 'w') as f:
            json.dump({
                'history': serializable_history,
                'best_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in self.best_metrics.items()},
                'best_epoch': self.best_epoch
            }, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")
    
    def load_history(self, filepath: str):
        """Load training history from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.history = data['history']
        self.best_metrics = data['best_metrics']
        self.best_epoch = data['best_epoch']
        
        logger.info(f"Training history loaded from {filepath}")
