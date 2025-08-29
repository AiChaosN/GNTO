"""Training configuration presets for different scenarios."""

from dataclasses import dataclass
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory to path for imports
if __name__ != "__main__":
    try:
        from ..training.trainer import TrainingConfig
    except ImportError:
        # Fallback for when running as script
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from training.trainer import TrainingConfig
else:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.trainer import TrainingConfig


# Preset configurations for different use cases
def get_quick_test_config() -> TrainingConfig:
    """Get configuration for quick testing/debugging."""
    return TrainingConfig(
        model_type="statistical",
        node_encoder_type="simple",
        hidden_dim=32,
        learning_rate=0.01,
        batch_size=16,
        num_epochs=10,
        early_stopping_patience=5,
        target_column="Actual Total Time",
        output_dir="result/training/quick_test",
        model_name="quick_test",
        log_interval=2
    )


def get_statistical_config() -> TrainingConfig:
    """Get configuration for statistical baseline model."""
    return TrainingConfig(
        model_type="statistical",
        node_encoder_type="large",
        hidden_dim=128,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        early_stopping_patience=15,
        target_column="Actual Total Time",
        output_dir="result/training/statistical",
        model_name="statistical_model",
        normalize_targets=True,
        weight_decay=1e-4
    )


def get_gcn_config() -> TrainingConfig:
    """Get configuration for GCN model."""
    return TrainingConfig(
        model_type="gcn",
        node_encoder_type="large",
        hidden_dim=128,
        learning_rate=0.001,
        batch_size=16,  # Smaller batch size for GNN
        num_epochs=200,
        early_stopping_patience=20,
        target_column="Actual Total Time",
        output_dir="result/training/gcn",
        model_name="gcn_model",
        normalize_targets=True,
        weight_decay=1e-4,
        clip_gradient=1.0,
        scheduler_type="step"
    )


def get_gat_config() -> TrainingConfig:
    """Get configuration for GAT model."""
    return TrainingConfig(
        model_type="gat",
        node_encoder_type="large",
        hidden_dim=128,
        learning_rate=0.0005,  # Slightly lower LR for GAT
        batch_size=16,
        num_epochs=200,
        early_stopping_patience=25,
        target_column="Actual Total Time",
        output_dir="result/training/gat",
        model_name="gat_model",
        normalize_targets=True,
        weight_decay=1e-4,
        clip_gradient=1.0,
        scheduler_type="cosine"
    )


def get_cost_prediction_config() -> TrainingConfig:
    """Get configuration for predicting query cost instead of execution time."""
    return TrainingConfig(
        model_type="statistical",
        node_encoder_type="large",
        hidden_dim=128,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        early_stopping_patience=15,
        target_column="Total Cost",  # Predict cost instead of time
        output_dir="result/training/cost_prediction",
        model_name="cost_prediction_model",
        normalize_targets=True,
        weight_decay=1e-4
    )


def get_multi_target_config() -> TrainingConfig:
    """Get configuration for multi-target prediction (time + cost)."""
    return TrainingConfig(
        model_type="statistical",
        node_encoder_type="large",
        hidden_dim=128,
        output_dim=2,  # Predict both time and cost
        learning_rate=0.001,
        batch_size=32,
        num_epochs=150,
        early_stopping_patience=20,
        target_column="Actual Total Time",  # Primary target
        output_dir="result/training/multi_target",
        model_name="multi_target_model",
        normalize_targets=True,
        weight_decay=1e-4
    )


# Configuration registry
CONFIG_REGISTRY = {
    "quick_test": get_quick_test_config,
    "statistical": get_statistical_config,
    "gcn": get_gcn_config,
    "gat": get_gat_config,
    "cost_prediction": get_cost_prediction_config,
    "multi_target": get_multi_target_config
}


def get_config(config_name: str) -> TrainingConfig:
    """Get a predefined training configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        TrainingConfig instance
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return CONFIG_REGISTRY[config_name]()


def list_available_configs() -> List[str]:
    """List all available configuration presets.
    
    Returns:
        List of configuration names
    """
    return list(CONFIG_REGISTRY.keys())


def print_available_configs():
    """Print information about available configurations."""
    print("Available Training Configurations:")
    print("=" * 40)
    
    configs_info = {
        "quick_test": "Fast configuration for testing (10 epochs, statistical)",
        "statistical": "Baseline statistical model (100 epochs)",
        "gcn": "Graph Convolutional Network model (200 epochs)",
        "gat": "Graph Attention Network model (200 epochs)",
        "cost_prediction": "Predict query cost instead of execution time",
        "multi_target": "Predict both execution time and cost"
    }
    
    for name, description in configs_info.items():
        print(f"  {name:15} - {description}")
    
    print("\nUsage:")
    print("  from config.training_config import get_config")
    print("  config = get_config('statistical')")


@dataclass
class ExperimentConfig:
    """Configuration for running multiple training experiments."""
    
    configs_to_run: List[str]
    data_path: str
    target_columns: List[str] = None
    random_seed: int = 42
    n_runs: int = 1  # Number of runs per configuration
    save_predictions: bool = False
    
    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = ["Actual Total Time"]


def create_experiment_config(data_path: str, 
                           configs: List[str] = None,
                           target_columns: List[str] = None) -> ExperimentConfig:
    """Create an experiment configuration.
    
    Args:
        data_path: Path to the training data
        configs: List of configuration names to run
        target_columns: Target columns to predict
        
    Returns:
        ExperimentConfig instance
    """
    if configs is None:
        configs = ["quick_test", "statistical"]
    
    return ExperimentConfig(
        configs_to_run=configs,
        data_path=data_path,
        target_columns=target_columns
    )
