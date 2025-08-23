"""
Configuration management for GNTO
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration"""
    
    # Node encoder config
    node_encoder: Dict[str, Any] = Field(default={
        "hidden_dims": [256, 128],
        "output_dim": 128,
        "dropout": 0.1,
        "activation": "relu",
        "batch_norm": True
    })
    
    # Structure encoder config  
    structure_encoder: Dict[str, Any] = Field(default={
        "hidden_dim": 128,
        "num_layers": 3,
        "num_edge_types": 10,
        "gnn_type": "gcn",
        "heads": 4,
        "dropout": 0.1,
        "pooling": "mean",
        "residual": True
    })
    
    # Prediction heads config
    heads: Dict[str, Any] = Field(default={
        "hidden_dims": [128, 64],
        "dropout": 0.1,
        "uncertainty": False,
        "loss_weights": None,
        "adaptive_weighting": True
    })
    
    # Tasks to train/evaluate
    tasks: list = Field(default=["cost", "latency", "ranking"])


class TrainingConfig(BaseModel):
    """Training configuration"""
    
    # Optimization
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-5)
    optimizer: str = Field(default="adam")
    scheduler: str = Field(default="cosine")
    
    # Training loop
    epochs: int = Field(default=100)
    batch_size: int = Field(default=32)
    eval_every: int = Field(default=5)
    save_every: int = Field(default=10)
    early_stopping_patience: int = Field(default=20)
    
    # Regularization
    grad_clip: float = Field(default=1.0)
    dropout: float = Field(default=0.1)
    
    # Data
    train_split: float = Field(default=0.8)
    val_split: float = Field(default=0.1)
    test_split: float = Field(default=0.1)


class DataConfig(BaseModel):
    """Data configuration"""
    
    # Paths
    data_dir: str = Field(default="data")
    train_file: Optional[str] = Field(default=None)
    val_file: Optional[str] = Field(default=None)
    test_file: Optional[str] = Field(default=None)
    
    # Processing
    max_nodes_per_plan: int = Field(default=100)
    max_plans_per_batch: int = Field(default=32)
    normalize_features: bool = Field(default=True)
    
    # Feature configuration
    continuous_features: list = Field(default=[
        "rows", "ndv", "selectivity", "io_cost", "cpu_cost", "parallel_degree"
    ])
    categorical_features: Dict[str, int] = Field(default={
        "operator_type": 50,
        "join_type": 10,
        "index_type": 20,
        "storage_format": 15,
        "hint": 30
    })
    structure_features: list = Field(default=[
        "is_blocking", "is_pipeline", "is_probe", "is_build", "stage_id"
    ])


class InferenceConfig(BaseModel):
    """Inference configuration"""
    
    # Service settings
    batch_size: int = Field(default=32)
    enable_monitoring: bool = Field(default=True)
    
    # Fallback thresholds
    fallback_threshold: Dict[str, float] = Field(default={
        "uncertainty_threshold": 0.5,
        "cost_confidence_threshold": 0.8,
        "latency_confidence_threshold": 0.8,
        "min_plan_score": 0.1
    })
    
    # Performance
    warmup_predictions: int = Field(default=10)
    max_concurrent_requests: int = Field(default=100)


class Config(BaseModel):
    """Main configuration class"""
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    
    # General settings
    experiment_name: str = Field(default="lqo_experiment")
    output_dir: str = Field(default="outputs")
    log_level: str = Field(default="INFO")
    seed: int = Field(default=42)
    device: str = Field(default="auto")  # auto, cpu, cuda
    
    # Monitoring
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="gnto")
    use_tensorboard: bool = Field(default=True)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file (yaml or json)
    
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return Config(**config_dict)


def save_config(config: Config, save_path: Union[str, Path]):
    """
    Save configuration to file
    
    Args:
        config: Config object to save
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.dict()
    
    # Save based on file extension
    if save_path.suffix.lower() in ['.yml', '.yaml']:
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif save_path.suffix.lower() == '.json':
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {save_path.suffix}")


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge configuration with overrides
    
    Args:
        base_config: Base configuration
        override_config: Dictionary of overrides
    
    Returns:
        Merged configuration
    """
    # Convert base config to dict
    base_dict = base_config.dict()
    
    # Deep merge override config
    def deep_merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_dict = deep_merge(base_dict, override_config)
    
    return Config(**merged_dict)
