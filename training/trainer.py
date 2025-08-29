"""Training utilities and trainer class for GNTO models."""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable

import numpy as np

# Try to import PyTorch components
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None

try:
    from ..models import (
        DataPreprocessor, NodeEncoder, TreeEncoder, PredictionHead,
        create_node_encoder, create_tree_encoder, is_gnn_available
    )
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import (
        DataPreprocessor, NodeEncoder, TreeEncoder, PredictionHead,
        create_node_encoder, create_tree_encoder, is_gnn_available
    )
from .dataset import PlanDataset, PlanSample
from .metrics import calculate_metrics, MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training GNTO models."""
    
    # Model configuration
    model_type: str = "statistical"  # "statistical", "gcn", "gat"
    node_encoder_type: str = "simple"  # "simple", "large"
    hidden_dim: int = 64
    output_dim: int = 1
    
    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # Data configuration
    target_column: str = "Actual Total Time"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    # Training behavior
    early_stopping_patience: int = 10
    save_best_model: bool = True
    log_interval: int = 10
    
    # Paths
    output_dir: str = "result/training"
    model_name: str = "gnto_model"
    
    # Advanced options
    normalize_targets: bool = True
    clip_gradient: Optional[float] = None
    scheduler_type: Optional[str] = None  # "step", "cosine", None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in ["statistical", "gcn", "gat"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        if self.model_type in ["gcn", "gat"] and not is_gnn_available():
            logger.warning(f"GNN model '{self.model_type}' requested but PyTorch Geometric not available. "
                          "Falling back to statistical model.")
            self.model_type = "statistical"


class GNTOModel:
    """Complete GNTO model combining all components."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize GNTO model.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        
        # Create encoders based on configuration
        if config.node_encoder_type == "simple":
            self.node_encoder = create_node_encoder(config.hidden_dim)
        else:
            from ..models import create_large_node_encoder
            self.node_encoder = create_large_node_encoder(config.hidden_dim)
        
        # Always use GNN TreeEncoder, map statistical to gcn for compatibility
        if config.model_type.lower() == "statistical":
            # Map statistical to gcn for GNN processing
            gnn_model_type = "gcn"
        else:
            gnn_model_type = config.model_type
        
        self.tree_encoder = create_tree_encoder(
            use_gnn=True,
            model_type=gnn_model_type,
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim
        )
        
        # Initialize prediction head with random weights
        # The existing PredictionHead expects weights as parameter
        initial_weights = np.random.normal(0, 0.1, config.hidden_dim)
        self.prediction_head = PredictionHead(weights=initial_weights)
        
        # Target normalization parameters
        self.target_mean = 0.0
        self.target_std = 1.0
    
    def forward(self, plan_tree) -> np.ndarray:
        """Forward pass through the model.
        
        Args:
            plan_tree: PlanNode tree structure
            
        Returns:
            Model predictions
        """
        # Encode nodes - need to encode each node in the tree
        def encode_tree_nodes(node):
            """Recursively encode all nodes in the tree."""
            # Encode current node
            node.node_vector = self.node_encoder.encode_node(node).detach().numpy()
            
            # Recursively encode children
            for child in node.children:
                encode_tree_nodes(child)
            
            return node
        
        encoded_tree = encode_tree_nodes(plan_tree)
        
        # GNNTreeEncoder can handle tree structures directly
        tree_embedding = self.tree_encoder.forward(encoded_tree)
        
        # Make prediction
        prediction = self.prediction_head.predict(tree_embedding)
        
        return prediction
    
    def fit_target_normalization(self, targets: np.ndarray):
        """Fit target normalization parameters.
        
        Args:
            targets: Target values for normalization
        """
        if self.config.normalize_targets:
            self.target_mean = np.mean(targets)
            self.target_std = np.std(targets)
            if self.target_std == 0:
                self.target_std = 1.0
            
            logger.info(f"Target normalization: mean={self.target_mean:.4f}, std={self.target_std:.4f}")
    
    def normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        """Normalize target values."""
        if self.config.normalize_targets:
            return (targets - self.target_mean) / self.target_std
        return targets
    
    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        if self.config.normalize_targets:
            return predictions * self.target_std + self.target_mean
        return predictions


class GNTOTrainer:
    """Trainer class for GNTO models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = GNTOModel(config)
        self.metrics_tracker = MetricsTracker()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for training."""
        log_file = self.output_dir / f"{self.config.model_name}_training.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.info(f"Training started with config: {self.config}")
    
    def train(self, dataset: PlanDataset) -> Dict[str, Any]:
        """Train the model on the provided dataset.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training with {len(dataset)} samples")
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = dataset.split(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio
        )
        
        # Fit target normalization on training data
        train_targets = train_dataset.get_targets(self.config.target_column)
        self.model.fit_target_normalization(train_targets)
        
        # Training loop
        best_val_score = -float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_dataset, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_dataset, epoch)
            
            # Update metrics tracker
            self.metrics_tracker.update(train_metrics, val_metrics, epoch)
            
            # Check for improvement
            current_val_score = val_metrics.get('r2', -float('inf'))
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                patience_counter = 0
                
                if self.config.save_best_model:
                    self._save_model(epoch, "best")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                self._log_epoch_results(epoch, train_metrics, val_metrics)
        
        # Final evaluation on test set
        test_metrics = self._validate_epoch(test_dataset, epoch, split_name="test")
        
        # Training summary
        training_time = time.time() - start_time
        self.metrics_tracker.print_summary()
        
        # Save final results
        results = {
            'config': self.config,
            'training_time': training_time,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_metrics': self.metrics_tracker.get_best_metrics(),
            'best_epoch': self.metrics_tracker.best_epoch
        }
        
        self._save_results(results)
        
        return results
    
    def _train_epoch(self, dataset: PlanDataset, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataset: Training dataset
            epoch: Current epoch number
            
        Returns:
            Training metrics for this epoch
        """
        predictions = []
        targets = []
        
        # Process all samples
        for sample in dataset.samples:
            try:
                # Forward pass
                pred = self.model.forward(sample.plan_tree)
                target = sample.targets[self.config.target_column]
                
                predictions.append(pred)
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("No valid predictions generated during training")
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets)
        
        # Denormalize predictions for metrics calculation
        predictions_denorm = self.model.denormalize_predictions(predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(targets, predictions_denorm)
        
        return metrics
    
    def _validate_epoch(self, dataset: PlanDataset, epoch: int, split_name: str = "validation") -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            dataset: Validation dataset
            epoch: Current epoch number
            split_name: Name of the split (for logging)
            
        Returns:
            Validation metrics for this epoch
        """
        predictions = []
        targets = []
        
        # Process all samples
        for sample in dataset.samples:
            try:
                # Forward pass
                pred = self.model.forward(sample.plan_tree)
                target = sample.targets[self.config.target_column]
                
                predictions.append(pred)
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Error processing sample during {split_name}: {e}")
                continue
        
        if not predictions:
            logger.warning(f"No valid predictions generated during {split_name}")
            return {}
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets)
        
        # Denormalize predictions for metrics calculation
        predictions_denorm = self.model.denormalize_predictions(predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(targets, predictions_denorm)
        
        return metrics
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log results for current epoch."""
        logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
        logger.info(f"  Train - MSE: {train_metrics.get('mse', 0):.6f}, "
                   f"MAE: {train_metrics.get('mae', 0):.6f}, "
                   f"R2: {train_metrics.get('r2', 0):.6f}")
        logger.info(f"  Val   - MSE: {val_metrics.get('mse', 0):.6f}, "
                   f"MAE: {val_metrics.get('mae', 0):.6f}, "
                   f"R2: {val_metrics.get('r2', 0):.6f}")
    
    def _save_model(self, epoch: int, suffix: str = ""):
        """Save model checkpoint."""
        model_path = self.output_dir / f"{self.config.model_name}_{suffix}.pkl"
        
        # For now, we'll save the model configuration and normalization parameters
        # In a full implementation, you'd serialize the actual model parameters
        model_data = {
            'config': self.config,
            'target_mean': self.model.target_mean,
            'target_std': self.model.target_std,
            'epoch': epoch
        }
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results."""
        import json
        
        # Save metrics history
        self.metrics_tracker.save_history(
            self.output_dir / f"{self.config.model_name}_history.json"
        )
        
        # Save results summary (convert non-serializable objects)
        serializable_results = {}
        for key, value in results.items():
            if key == 'config':
                # Convert dataclass to dict
                serializable_results[key] = value.__dict__
            elif isinstance(value, (np.floating, np.integer)):
                serializable_results[key] = float(value)
            elif isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        results_path = self.output_dir / f"{self.config.model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def predict(self, dataset: PlanDataset) -> np.ndarray:
        """Make predictions on a dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        for sample in dataset.samples:
            try:
                pred = self.model.forward(sample.plan_tree)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error predicting sample: {e}")
                predictions.append(0.0)
        
        predictions = np.array(predictions).flatten()
        return self.model.denormalize_predictions(predictions)
