"""Configuration module for GNTO training."""

from .training_config import (
    TrainingConfig,
    ExperimentConfig,
    get_config,
    list_available_configs,
    print_available_configs,
    create_experiment_config,
    get_quick_test_config,
    get_statistical_config,
    get_gcn_config,
    get_gat_config
)

__all__ = [
    "TrainingConfig",
    "ExperimentConfig", 
    "get_config",
    "list_available_configs",
    "print_available_configs",
    "create_experiment_config",
    "get_quick_test_config",
    "get_statistical_config",
    "get_gcn_config",
    "get_gat_config"
]
