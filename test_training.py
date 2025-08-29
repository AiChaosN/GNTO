#!/usr/bin/env python3
"""Simple test script for the training module."""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test if all training components can be imported."""
    print("Testing imports...")
    
    try:
        # Test models import
        from models import DataPreprocessor, PlanNode
        print("✓ Models imported successfully")
        
        # Test training dataset
        from training.dataset import PlanDataset
        print("✓ Dataset class imported successfully")
        
        # Test training metrics
        from training.metrics import calculate_metrics
        print("✓ Metrics imported successfully")
        
        # Test training trainer
        from training.trainer import GNTOTrainer, TrainingConfig
        print("✓ Trainer imported successfully")
        
        print("All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\nTesting basic functionality...")
    
    try:
        # Test DataPreprocessor
        from models import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Sample plan data (simplified)
        sample_plan = {
            "Node Type": "Seq Scan",
            "Total Cost": 100.0,
            "Actual Total Time": 50.0,
            "Plans": []
        }
        
        # Test preprocessing
        plan_node = preprocessor.preprocess(sample_plan)
        print(f"✓ Plan preprocessing works: {plan_node.node_type}")
        
        # Test configuration
        from training.trainer import TrainingConfig
        
        config = TrainingConfig(
            model_type="statistical",
            num_epochs=5,
            output_dir="test_output"
        )
        print(f"✓ Configuration created: {config.model_type}")
        
        print("Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_data_loading():
    """Test data loading if demo data exists."""
    print("\nTesting data loading...")
    
    data_path = Path("data/demo_plan_01.csv")
    if not data_path.exists():
        print("⚠ Demo data not found, skipping data loading test")
        return True
    
    try:
        from training.dataset import PlanDataset
        
        # Try to load just first few rows for testing
        import pandas as pd
        df = pd.read_csv(data_path)
        
        if len(df) == 0:
            print("⚠ Demo data is empty")
            return True
        
        # Create a small test dataset
        test_df = df.head(3)  # Just first 3 rows
        test_path = Path("test_data.csv")
        test_df.to_csv(test_path, index=False)
        
        # Test dataset creation
        dataset = PlanDataset(test_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Clean up
        test_path.unlink()
        
        print("Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        # Clean up on failure
        test_path = Path("test_data.csv")
        if test_path.exists():
            test_path.unlink()
        return False


def main():
    """Run all tests."""
    print("GNTO Training Module Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training module is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python train.py --config quick_test --data data/demo_plan_01.csv")
        print("3. Check examples: python examples/train_example.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
