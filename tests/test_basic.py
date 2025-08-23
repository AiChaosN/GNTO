"""
Basic tests for GNTO framework
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from gnto import LQOModel, FeatureSpec, InferenceService
from gnto.core.feature_spec import NodeFeatureConfig, PlanBatch
from gnto.utils.config import Config, get_default_config


@pytest.fixture
def sample_plan():
    """Sample execution plan for testing"""
    return {
        "nodes": [
            {
                "operator_type": "SeqScan",
                "rows": 1000.0,
                "ndv": 100.0,
                "selectivity": 1.0,
                "io_cost": 10.0,
                "cpu_cost": 5.0,
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
                "operator_type": "Filter",
                "rows": 500.0,
                "ndv": 50.0,
                "selectivity": 0.5,
                "io_cost": 0.0,
                "cpu_cost": 2.0,
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
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "pipeline"}
        ]
    }


@pytest.fixture
def feature_spec():
    """Feature specification for testing"""
    config = NodeFeatureConfig()
    return FeatureSpec(config)


@pytest.fixture
def model(feature_spec):
    """LQO model for testing"""
    return LQOModel(
        feature_spec=feature_spec,
        tasks=["cost", "latency"],
        device=torch.device("cpu")
    )


class TestFeatureSpec:
    """Test FeatureSpec functionality"""
    
    def test_feature_spec_creation(self, feature_spec):
        """Test feature spec creation"""
        assert feature_spec is not None
        assert "total" in feature_spec.feature_dims
        assert feature_spec.feature_dims["total"] > 0
    
    def test_plan_validation(self, feature_spec, sample_plan):
        """Test plan validation"""
        result = feature_spec.validate(sample_plan)
        
        assert result["is_valid"] is True
        assert "normalized_plan" in result
        assert len(result["errors"]) == 0
    
    def test_plan_tensorization(self, feature_spec, sample_plan):
        """Test plan tensorization"""
        batch = feature_spec.tensorize(sample_plan)
        
        assert isinstance(batch, PlanBatch)
        assert batch.batch_size == 1
        assert batch.total_nodes == 2
        assert batch.node_features.shape[1] == feature_spec.feature_dims["total"]
    
    def test_batch_tensorization(self, feature_spec, sample_plan):
        """Test batch tensorization"""
        plans = [sample_plan, sample_plan.copy()]
        batch = feature_spec.tensorize(plans)
        
        assert isinstance(batch, PlanBatch)
        assert batch.batch_size == 2
        assert batch.total_nodes == 4


class TestLQOModel:
    """Test LQOModel functionality"""
    
    def test_model_creation(self, model):
        """Test model creation"""
        assert model is not None
        assert len(model.tasks) == 2
        assert "cost" in model.tasks
        assert "latency" in model.tasks
    
    def test_model_forward(self, model, feature_spec, sample_plan):
        """Test model forward pass"""
        batch = feature_spec.tensorize(sample_plan)
        
        model.eval()
        with torch.no_grad():
            predictions = model.forward(batch)
        
        assert isinstance(predictions, dict)
        assert "cost" in predictions
        assert "latency" in predictions
        
        for task in model.tasks:
            if task in predictions:
                pred = predictions[task]
                assert isinstance(pred, torch.Tensor)
                assert pred.shape[0] == 1  # Batch size
    
    def test_model_predict(self, model, sample_plan):
        """Test model prediction interface"""
        predictions = model.predict(sample_plan)
        
        assert isinstance(predictions, dict)
        for task in model.tasks:
            if task in predictions:
                assert task in predictions
    
    def test_model_embedding(self, model, sample_plan):
        """Test plan embedding extraction"""
        embedding = model.get_plan_embedding(sample_plan)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape[0] == 1  # Batch size
        assert embedding.shape[1] > 0   # Embedding dimension
    
    def test_model_save_load(self, model, feature_spec):
        """Test model save and load"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Save model
            model.save_checkpoint(str(checkpoint_path), epoch=1)
            assert checkpoint_path.exists()
            
            # Load model
            loaded_model = LQOModel.load_checkpoint(str(checkpoint_path))
            assert loaded_model is not None
            assert loaded_model.tasks == model.tasks


class TestInferenceService:
    """Test InferenceService functionality"""
    
    def test_service_creation(self, model):
        """Test inference service creation"""
        service = InferenceService(model=model)
        assert service is not None
        assert service.model == model
    
    def test_service_predict(self, model, sample_plan):
        """Test inference service prediction"""
        service = InferenceService(model=model, enable_monitoring=True)
        
        result = service.predict(sample_plan)
        
        assert isinstance(result, dict)
        assert "inference_time" in result
        
        # Check for prediction results
        for task in model.tasks:
            if task in result:
                assert isinstance(result[task], (float, int, list))
    
    def test_service_batch_predict(self, model, sample_plan):
        """Test batch prediction"""
        service = InferenceService(model=model)
        
        plans = [sample_plan, sample_plan.copy()]
        results = service.batch_predict(plans)
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, dict)
    
    def test_service_stats(self, model, sample_plan):
        """Test service monitoring"""
        service = InferenceService(model=model, enable_monitoring=True)
        
        # Make some predictions
        service.predict(sample_plan)
        service.predict(sample_plan)
        
        stats = service.get_stats()
        assert isinstance(stats, dict)
        assert "total_predictions" in stats
        assert stats["total_predictions"] == 2


class TestConfig:
    """Test configuration functionality"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = get_default_config()
        
        assert isinstance(config, Config)
        assert len(config.model.tasks) > 0
        assert config.training.learning_rate > 0
        assert config.training.batch_size > 0
    
    def test_config_save_load(self):
        """Test configuration save and load"""
        config = get_default_config()
        config.experiment_name = "test_experiment"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Save config
            from gnto.utils.config import save_config, load_config
            save_config(config, config_path)
            assert config_path.exists()
            
            # Load config
            loaded_config = load_config(config_path)
            assert loaded_config.experiment_name == "test_experiment"


def test_integration():
    """Integration test for the complete pipeline"""
    # Create sample data
    sample_plan = {
        "nodes": [
            {
                "operator_type": "SeqScan",
                "rows": 1000.0,
                "ndv": 100.0,
                "selectivity": 1.0,
                "io_cost": 10.0,
                "cpu_cost": 5.0,
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
            }
        ],
        "edges": []
    }
    
    # Create feature spec
    feature_spec = FeatureSpec()
    
    # Create model
    model = LQOModel(
        feature_spec=feature_spec,
        tasks=["cost", "latency"],
        device=torch.device("cpu")
    )
    
    # Create inference service
    service = InferenceService(model=model)
    
    # Make prediction
    result = service.predict(sample_plan)
    
    # Verify results
    assert isinstance(result, dict)
    assert "cost" in result or "latency" in result
    assert "inference_time" in result
    
    print("✅ Integration test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
