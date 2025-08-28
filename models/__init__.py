"""Models package for the GNTO project.

This package provides a unified set of classes for learning-based query
optimization. All components support both traditional and GNN-enhanced
modes with automatic fallback when dependencies are not available.
"""

from .DataPreprocessor import DataPreprocessor, PlanNode
from .NodeEncoder import NodeEncoder, NodeFeatures, create_simple_node_encoder, create_rich_node_encoder
from .TreeEncoder import TreeEncoder, create_tree_encoder, is_gnn_available
from .PredictionHead import PredictionHead
from .Gnto import GNTO, create_traditional_gnto, create_gnn_gnto, create_auto_gnto

# Try to import GNN-specific classes
try:
    from .TreeEncoder import GNNTreeEncoder, GCNTreeEncoder, GATTreeEncoder, TreeToGraphConverter
    _GNN_CLASSES_AVAILABLE = True
except ImportError:
    _GNN_CLASSES_AVAILABLE = False
    # Create dummy classes for graceful fallback
    class GNNTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("GNN components require PyTorch and PyTorch Geometric")
    
    class GCNTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("GNN components require PyTorch and PyTorch Geometric")
    
    class GATTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("GNN components require PyTorch and PyTorch Geometric")
    
    class TreeToGraphConverter:
        def __init__(self, *args, **kwargs):
            raise ImportError("GNN components require PyTorch and PyTorch Geometric")

__all__ = [
    # Core components - Clean architecture
    "DataPreprocessor",
    "PlanNode",
    "NodeEncoder",
    "NodeFeatures", 
    "TreeEncoder",
    "PredictionHead",
    "GNTO",
    
    # Factory functions - Clean architecture
    "create_simple_node_encoder",
    "create_rich_node_encoder",
    "create_tree_encoder",
    "create_traditional_gnto",
    "create_gnn_gnto",
    "create_auto_gnto",
    
    # GNN-specific classes (may raise ImportError if dependencies missing)
    "GNNTreeEncoder",
    "GCNTreeEncoder", 
    "GATTreeEncoder",
    "TreeToGraphConverter",
    
    # Utility functions
    "is_gnn_available",
]

# Package information
def get_package_info() -> dict:
    """Get information about the package and available components."""
    return {
        'version': '1.0.0',
        'gnn_available': is_gnn_available(),
        'components': {
            'DataPreprocessor': 'Convert raw JSON plans to structured trees',
            'Encoder': 'Encode plan nodes to numerical vectors (traditional + GNN modes)',
            'TreeModel': 'Aggregate node vectors (traditional + GNN models)',
            'Predictioner': 'Linear prediction head',
            'GNTO': 'End-to-end pipeline (traditional + GNN modes)'
        },
        'gnn_models': ['GCN', 'GAT'] if is_gnn_available() else []
    }

def print_package_info():
    """Print package information."""
    info = get_package_info()
    print(f"GNTO Models Package v{info['version']}")
    print(f"GNN Support: {'Yes' if info['gnn_available'] else 'No'}")
    if info['gnn_available']:
        print(f"Available GNN Models: {', '.join(info['gnn_models'])}")
    print("\nComponents:")
    for name, desc in info['components'].items():
        print(f"  {name}: {desc}")

# Convenience function for quick setup
def quick_setup(use_gnn: bool = None) -> GNTO:
    """Quick setup for GNTO pipeline.
    
    Parameters
    ----------
    use_gnn:
        Whether to use GNN. If None, auto-detects based on availability.
        
    Returns
    -------
    GNTO:
        Configured GNTO instance
    """
    if use_gnn is None:
        return create_auto_gnto()
    elif use_gnn:
        return create_gnn_gnto()
    else:
        return create_traditional_gnto()