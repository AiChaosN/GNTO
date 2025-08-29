#!/usr/bin/env python3
"""Main training script for GNTO models.

This script provides a command-line interface for training query optimization models
on PostgreSQL execution plans.

Examples:
    # Quick test with default settings
    python train.py --config quick_test --data data/demo_plan_01.csv
    
    # Train statistical baseline
    python train.py --config statistical --data data/demo_plan_01.csv
    
    # Train GCN model (requires PyTorch Geometric)
    python train.py --config gcn --data data/demo_plan_01.csv
    
    # Custom configuration
    python train.py --data data/demo_plan_01.csv --model-type statistical --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import GNTO components
try:
    from config import get_config, list_available_configs, print_available_configs
    from training import PlanDataset, GNTOTrainer, TrainingConfig, setup_training, set_random_seed
    from training.utils import validate_config, create_config_summary
except ImportError as e:
    logger.error(f"Failed to import GNTO components: {e}")
    logger.error("Make sure you're running from the GNTO project root directory")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNTO query optimization models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to CSV file containing query plans"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="Actual Total Time",
        help="Target column to predict (default: Actual Total Time)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Use predefined configuration (see --list-configs)"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available predefined configurations and exit"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["statistical", "gcn", "gat"],
        default="statistical",
        help="Type of model to train (default: statistical)"
    )
    
    parser.add_argument(
        "--node-encoder",
        type=str,
        choices=["simple", "large"],
        default="simple",
        help="Node encoder type (default: simple)"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension size (default: 64)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="result/training",
        help="Output directory for training results (default: result/training)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="gnto_model",
        help="Name for the trained model (default: gnto_model)"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without training"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    return TrainingConfig(
        model_type=args.model_type,
        node_encoder_type=args.node_encoder,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        target_column=args.target,
        output_dir=args.output_dir,
        model_name=args.model_name,
        normalize_targets=True,
        save_best_model=True,
        log_interval=max(1, args.epochs // 10)
    )


def main():
    """Main training function."""
    args = parse_args()
    
    # Handle special arguments
    if args.list_configs:
        print_available_configs()
        return
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create or load configuration
    if args.config:
        logger.info(f"Loading predefined configuration: {args.config}")
        try:
            config = get_config(args.config)
            # Override with command line arguments if provided
            if hasattr(args, 'data'):
                # Update paths and names based on args
                config.output_dir = args.output_dir
                config.model_name = args.model_name
                config.target_column = args.target
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            logger.info("Available configurations:")
            for config_name in list_available_configs():
                logger.info(f"  - {config_name}")
            return
    else:
        logger.info("Creating configuration from command line arguments")
        config = create_config_from_args(args)
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Print configuration summary
    logger.info("Training configuration:")
    print(create_config_summary(config))
    
    if args.dry_run:
        logger.info("Dry run mode - exiting without training")
        return
    
    # Check data file
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        dataset = PlanDataset(
            csv_path=data_path,
            target_columns=[config.target_column]
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        if hasattr(dataset, 'statistics'):
            logger.info(f"Dataset statistics: {dataset.statistics}")
        else:
            logger.info("Dataset statistics not available")
        
        # Create trainer
        trainer = GNTOTrainer(config)
        
        # Check model structure before training
        trainer._check_model_updates()
        
        # Train model
        logger.info("Starting training...")
        results = trainer.train(dataset)
        
        # Print results summary
        logger.info("Training completed!")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")
        
        best_metrics = results.get('best_metrics', {})
        if best_metrics:
            logger.info("Best validation metrics:")
            for metric, value in best_metrics.items():
                if metric.startswith('val_'):
                    logger.info(f"  {metric}: {value:.6f}")
        
        test_metrics = results.get('test_metrics', {})
        if test_metrics:
            logger.info("Test metrics:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.6f}")
        
        logger.info(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
