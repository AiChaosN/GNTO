"""Models package for the GNTO project.

This package provides a unified set of classes for learning-based query
optimization. All components support both traditional and GNN-enhanced
modes with automatic fallback when dependencies are not available.
"""

from .DataPreprocessor import DataPreprocessor, PlanNode
from .NodeEncoder import NodeEncoder, create_node_encoder, create_simple_node_encoder, create_large_node_encoder
from .TreeEncoder import TreeEncoder, create_tree_encoder, is_gnn_available
from .PredictionHead import PredictionHead


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
    "TreeEncoder",
    "PredictionHead",
    
    # Factory functions - Clean architecture
    "create_node_encoder",
    "create_simple_node_encoder",
    "create_large_node_encoder",
    "create_tree_encoder",
    
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
            'NodeEncoder': 'Encode plan nodes to numerical vectors using multi-view encoding',
            'TreeEncoder': 'Aggregate node vectors using statistical or GNN methods',
            'PredictionHead': 'Linear prediction head for final outputs'
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

