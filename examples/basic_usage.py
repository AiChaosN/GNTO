"""
Basic usage example for GNTO framework
"""

import torch
import json
from pathlib import Path

# Import GNTO components
from gnto import LQOModel, FeatureSpec, InferenceService
from gnto.core.feature_spec import NodeFeatureConfig
from gnto.utils.config import Config, get_default_config


def create_sample_plan():
    """Create a sample execution plan for testing"""
    return {
        "nodes": [
            {
                # Scan node
                "operator_type": "SeqScan",
                "rows": 10000.0,
                "ndv": 1000.0,
                "selectivity": 1.0,
                "io_cost": 100.0,
                "cpu_cost": 50.0,
                "parallel_degree": 1.0,
                "join_type": "none",
                "index_type": "none",
                "storage_format": "heap",
                "hint": "none",
                "is_blocking": 0.0,
                "is_pipeline": 1.0,
                "is_probe": 0.0,
                "is_build": 0.0,
                "stage_id": 0.0
            },
            {
                # Filter node
                "operator_type": "Filter",
                "rows": 5000.0,
                "ndv": 500.0,
                "selectivity": 0.5,
                "io_cost": 0.0,
                "cpu_cost": 25.0,
                "parallel_degree": 1.0,
                "join_type": "none",
                "index_type": "none",
                "storage_format": "none",
                "hint": "none",
                "is_blocking": 0.0,
                "is_pipeline": 1.0,
                "is_probe": 0.0,
                "is_build": 0.0,
                "stage_id": 0.0
            },
            {
                # Join node
                "operator_type": "HashJoin",
                "rows": 7500.0,
                "ndv": 750.0,
                "selectivity": 1.5,
                "io_cost": 50.0,
                "cpu_cost": 100.0,
                "parallel_degree": 1.0,
                "join_type": "inner",
                "index_type": "hash",
                "storage_format": "none",
                "hint": "none",
                "is_blocking": 1.0,
                "is_pipeline": 0.0,
                "is_probe": 1.0,
                "is_build": 0.0,
                "stage_id": 1.0
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "pipeline"},
            {"source": 1, "target": 2, "type": "pipeline"}
        ],
        "metadata": {
            "query_id": "sample_001",
            "database": "test_db"
        }
    }


def basic_model_usage():
    """Demonstrate basic model usage"""
    print("=== Basic Model Usage ===")
    
    # 1. Create feature specification
    feature_config = NodeFeatureConfig()
    feature_spec = FeatureSpec(feature_config)
    
    print(f"Feature dimensions: {feature_spec.feature_dims}")
    
    # 2. Create model with default configuration
    config = get_default_config()
    
    model = LQOModel(
        feature_spec=feature_spec,
        tasks=["cost", "latency", "ranking"],
        device=torch.device("cpu")  # Use CPU for demo
    )
    
    print(f"Model info: {model.get_model_info()}")
    
    # 3. Create sample plan
    sample_plan = create_sample_plan()
    
    # 4. Make prediction
    model.eval()
    predictions = model.predict(sample_plan)
    
    print("Predictions:")
    for task, pred in predictions.items():
        if isinstance(pred, torch.Tensor):
            print(f"  {task}: {pred.item():.4f}")
        else:
            print(f"  {task}: {pred}")
    
    # 5. Get plan embedding
    embedding = model.get_plan_embedding(sample_plan)
    print(f"Plan embedding shape: {embedding.shape}")
    
    return model, feature_spec


def inference_service_usage():
    """Demonstrate inference service usage"""
    print("\n=== Inference Service Usage ===")
    
    # Create model and feature spec
    model, feature_spec = basic_model_usage()
    
    # Create inference service
    service = InferenceService(
        model=model,
        enable_monitoring=True,
        batch_size=8
    )
    
    # Warmup the service
    service.warmup(num_warmup=3)
    
    # Create sample plans
    plans = [create_sample_plan() for _ in range(5)]
    
    # Make predictions
    results = service.batch_predict(plans, return_embeddings=True)
    
    print(f"Processed {len(results)} plans")
    
    for i, result in enumerate(results[:2]):  # Show first 2 results
        print(f"\nPlan {i+1}:")
        print(f"  Cost: {result.get('cost', 'N/A')}")
        print(f"  Latency: {result.get('latency', 'N/A')}")
        print(f"  Ranking: {result.get('ranking', 'N/A')}")
        print(f"  Confidence: {result.get('overall_confidence', 'N/A')}")
        print(f"  Fallback recommended: {result.get('fallback_recommended', 'N/A')}")
        
        if 'embedding' in result:
            print(f"  Embedding shape: {len(result['embedding'])}")
    
    # Get service statistics
    stats = service.get_stats()
    print(f"\nService stats: {stats}")
    
    return service


def batch_processing_example():
    """Demonstrate batch processing"""
    print("\n=== Batch Processing Example ===")
    
    # Create model
    model, _ = basic_model_usage()
    
    # Create batch of plans
    plans = [create_sample_plan() for _ in range(10)]
    
    # Add some variation to the plans
    for i, plan in enumerate(plans):
        # Vary the row counts
        for node in plan["nodes"]:
            node["rows"] *= (1.0 + i * 0.1)
        
        plan["metadata"]["query_id"] = f"batch_query_{i:03d}"
    
    # Process batch
    model.eval()
    batch_predictions = model.predict(plans)
    
    print(f"Batch predictions shape:")
    for task, preds in batch_predictions.items():
        if isinstance(preds, torch.Tensor):
            print(f"  {task}: {preds.shape}")
    
    # Show individual predictions
    print("\nIndividual predictions:")
    for i in range(min(3, len(plans))):
        print(f"  Plan {i+1}:")
        for task, preds in batch_predictions.items():
            if isinstance(preds, torch.Tensor):
                if preds.dim() == 1:
                    print(f"    {task}: {preds[i].item():.4f}")
                elif preds.dim() == 2:
                    print(f"    {task}: {preds[i].tolist()}")


def configuration_example():
    """Demonstrate configuration usage"""
    print("\n=== Configuration Example ===")
    
    # Create custom configuration
    config = Config()
    
    # Modify some settings
    config.model.tasks = ["cost", "latency"]
    config.model.node_encoder["hidden_dims"] = [512, 256, 128]
    config.model.structure_encoder["gnn_type"] = "gat"
    config.model.structure_encoder["heads"] = 8
    
    config.training.learning_rate = 5e-4
    config.training.batch_size = 64
    
    print(f"Model tasks: {config.model.tasks}")
    print(f"Node encoder dims: {config.model.node_encoder['hidden_dims']}")
    print(f"GNN type: {config.model.structure_encoder['gnn_type']}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Save configuration
    from gnto.utils.config import save_config
    config_path = Path("example_config.yaml")
    save_config(config, config_path)
    print(f"Saved configuration to {config_path}")
    
    # Load configuration
    from gnto.utils.config import load_config
    loaded_config = load_config(config_path)
    print(f"Loaded config tasks: {loaded_config.model.tasks}")
    
    # Clean up
    config_path.unlink()


def main():
    """Run all examples"""
    print("GNTO Framework - Basic Usage Examples")
    print("=" * 40)
    
    try:
        # Basic usage
        basic_model_usage()
        
        # Inference service
        inference_service_usage()
        
        # Batch processing
        batch_processing_example()
        
        # Configuration
        configuration_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
