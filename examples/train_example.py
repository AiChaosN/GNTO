#!/usr/bin/env python3
"""Example script showing how to use the training module.

This example demonstrates:
1. Loading a dataset from CSV
2. Training different types of models
3. Evaluating model performance
4. Saving and loading trained models
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training import PlanDataset, GNTOTrainer, create_plan_dataset
from config import get_config, list_available_configs
from training.utils import set_random_seed, format_time
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_training():
    """Example 1: Basic training with default settings."""
    print("\n" + "="*50)
    print("Example 1: Basic Training")
    print("="*50)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Load dataset
    data_path = project_root / "data" / "demo_plan_01.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure you have data in the data/ directory")
        return
    
    print(f"Loading dataset from {data_path}")
    dataset = create_plan_dataset(data_path)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Show dataset statistics
    print("\nDataset Statistics:")
    for key, stats in dataset.statistics.items():
        if isinstance(stats, dict):
            print(f"  {key}:")
            for stat_name, value in stats.items():
                if isinstance(value, float):
                    print(f"    {stat_name}: {value:.4f}")
                else:
                    print(f"    {stat_name}: {value}")
    
    # Get quick test configuration
    config = get_config("quick_test")
    print(f"\nUsing configuration: {config.model_type} model")
    print(f"Training for {config.num_epochs} epochs")
    
    # Create trainer and train
    trainer = GNTOTrainer(config)
    
    start_time = time.time()
    results = trainer.train(dataset)
    training_time = time.time() - start_time
    
    # Show results
    print(f"\nTraining completed in {format_time(training_time)}")
    
    test_metrics = results.get('test_metrics', {})
    if test_metrics:
        print("Test Performance:")
        print(f"  R² Score: {test_metrics.get('r2', 0):.4f}")
        print(f"  MAE: {test_metrics.get('mae', 0):.4f}")
        print(f"  RMSE: {test_metrics.get('rmse', 0):.4f}")


def example_model_comparison():
    """Example 2: Compare different model types."""
    print("\n" + "="*50)
    print("Example 2: Model Comparison")
    print("="*50)
    
    # Set random seed
    set_random_seed(42)
    
    # Load dataset
    data_path = project_root / "data" / "demo_plan_01.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    dataset = create_plan_dataset(data_path)
    print(f"Dataset: {len(dataset)} samples")
    
    # Models to compare
    models_to_test = ["quick_test", "statistical"]
    
    # Try to include GNN models if available
    try:
        from models import is_gnn_available
        if is_gnn_available():
            # For quick comparison, we'll use smaller configs
            models_to_test.append("gcn")
            print("GNN models available - including GCN")
        else:
            print("GNN models not available - using statistical models only")
    except:
        print("GNN availability check failed - using statistical models only")
    
    results_comparison = {}
    
    for model_name in models_to_test:
        print(f"\n--- Training {model_name} model ---")
        
        # Get configuration
        config = get_config(model_name)
        # Reduce epochs for quick comparison
        config.num_epochs = min(config.num_epochs, 20)
        config.output_dir = f"result/training/comparison/{model_name}"
        
        # Train model
        trainer = GNTOTrainer(config)
        
        start_time = time.time()
        try:
            results = trainer.train(dataset)
            training_time = time.time() - start_time
            
            # Store results
            test_metrics = results.get('test_metrics', {})
            results_comparison[model_name] = {
                'r2': test_metrics.get('r2', 0),
                'mae': test_metrics.get('mae', 0),
                'rmse': test_metrics.get('rmse', 0),
                'training_time': training_time
            }
            
            print(f"  Completed in {format_time(training_time)}")
            print(f"  R² Score: {test_metrics.get('r2', 0):.4f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    # Print comparison
    print("\n" + "="*30)
    print("Model Comparison Results")
    print("="*30)
    print(f"{'Model':<15} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Time':<10}")
    print("-" * 55)
    
    for model_name, metrics in results_comparison.items():
        print(f"{model_name:<15} "
              f"{metrics['r2']:<8.4f} "
              f"{metrics['mae']:<10.2f} "
              f"{metrics['rmse']:<10.2f} "
              f"{format_time(metrics['training_time']):<10}")


def example_custom_config():
    """Example 3: Create and use custom configuration."""
    print("\n" + "="*50)
    print("Example 3: Custom Configuration")
    print("="*50)
    
    # Import TrainingConfig for custom configuration
    from training.trainer import TrainingConfig
    
    # Create custom configuration
    custom_config = TrainingConfig(
        model_type="statistical",
        node_encoder_type="large",
        hidden_dim=96,
        learning_rate=0.005,
        batch_size=24,
        num_epochs=30,
        early_stopping_patience=8,
        target_column="Actual Total Time",
        output_dir="result/training/custom",
        model_name="custom_model",
        normalize_targets=True,
        weight_decay=1e-3,
        log_interval=5
    )
    
    print("Custom Configuration:")
    print(f"  Model Type: {custom_config.model_type}")
    print(f"  Hidden Dim: {custom_config.hidden_dim}")
    print(f"  Learning Rate: {custom_config.learning_rate}")
    print(f"  Batch Size: {custom_config.batch_size}")
    print(f"  Epochs: {custom_config.num_epochs}")
    
    # Load dataset
    data_path = project_root / "data" / "demo_plan_01.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    dataset = create_plan_dataset(data_path)
    
    # Train with custom config
    trainer = GNTOTrainer(custom_config)
    results = trainer.train(dataset)
    
    print("\nCustom model training completed!")
    test_metrics = results.get('test_metrics', {})
    if test_metrics:
        print(f"Test R² Score: {test_metrics.get('r2', 0):.4f}")


def main():
    """Run all examples."""
    print("GNTO Training Examples")
    print("This script demonstrates the training module capabilities")
    
    try:
        # Run examples
        example_basic_training()
        example_model_comparison()
        example_custom_config()
        
        print("\n" + "="*50)
        print("All examples completed!")
        print("Check the result/training/ directory for outputs")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
