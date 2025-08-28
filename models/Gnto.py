"""Unified GNTO pipeline with GNN support.

This module provides a flexible GNTO pipeline that can use either traditional
components or GNN-enhanced components, with automatic fallback when GNN
dependencies are not available.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Union, List
import numpy as np

from .DataPreprocessor import DataPreprocessor
from .NodeEncoder import NodeEncoder, create_simple_node_encoder, create_rich_node_encoder
from .TreeEncoder import TreeEncoder, create_tree_encoder, is_gnn_available
from .PredictionHead import PredictionHead

# Try to import GNN components
try:
    from .TreeEncoder import GNNTreeEncoder
    _GNN_TREE_ENCODER_AVAILABLE = True
except ImportError:
    _GNN_TREE_ENCODER_AVAILABLE = False
    GNNTreeEncoder = None


class GNTO:
    """Unified end-to-end pipeline with optional GNN support.

    This class provides a flexible pipeline that can use either traditional
    components or GNN-enhanced components. It maintains full backward
    compatibility while offering enhanced capabilities when GNN dependencies
    are available.
    """

    def __init__(self,
                 preprocessor: Optional[DataPreprocessor] = None,
                 encoder: Optional[NodeEncoder] = None,
                 tree_model: Optional[Union[TreeEncoder, GNNTreeEncoder]] = None,
                 prediction_head: Optional[PredictionHead] = None,
                 use_gnn: bool = False,
                 gnn_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the GNTO pipeline.
        
        Parameters
        ----------
        preprocessor:
            Data preprocessor component
        encoder:
            Node encoder component
        tree_model:
            Tree encoder component
        prediction_head:
            Prediction head component
        use_gnn:
            Whether to use GNN components when available
        gnn_config:
            Configuration for GNN components
        """
        self.use_gnn = use_gnn and is_gnn_available()
        
        # Default GNN configuration
        default_gnn_config = {
            'encoder': {
                'feature_dim': 64,
                'include_numerical': True,
                'include_categorical': True,
                'normalize_features': True
            },
            'tree_model': {
                'model_type': 'gcn',
                'input_dim': 64,
                'hidden_dim': 128,
                'output_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'pooling': 'mean',
                'device': 'cpu'
            }
        }
        
        if gnn_config:
            # Merge user config with defaults
            for key in default_gnn_config:
                if key in gnn_config:
                    default_gnn_config[key].update(gnn_config[key])
        
        self.gnn_config = default_gnn_config
        
        # Initialize components
        self.preprocessor = preprocessor or DataPreprocessor()
        
        if encoder is not None:
            self.encoder = encoder
        elif self.use_gnn:
            self.encoder = create_rich_node_encoder(**self.gnn_config['encoder'])
        else:
            self.encoder = create_simple_node_encoder()
        
        if tree_model is not None:
            self.tree_model = tree_model
        elif self.use_gnn:
            # Pass encoder to GNN tree model for feature extraction
            self.tree_model = create_tree_encoder(
                use_gnn=True, 
                **self.gnn_config['tree_model']
            )
        else:
            self.tree_model = create_tree_encoder(use_gnn=False)
        
        self.prediction_head = prediction_head or PredictionHead()
        
        # Track component types
        self._using_gnn = (_GNN_TREE_ENCODER_AVAILABLE and 
                          self.use_gnn and 
                          isinstance(self.tree_model, type(GNNTreeEncoder)) if GNNTreeEncoder else False)

    def run(self, plan: Dict[str, Any]) -> float:
        """Execute the end-to-end pipeline on a query plan.
        
        Parameters
        ----------
        plan:
            Raw query plan dictionary
            
        Returns
        -------
        float:
            Performance prediction score
        """
        # Preprocess the plan
        structured = self.preprocessor.preprocess(plan)
        
        # Apply tree model
        if self._using_gnn:
            # GNN tree model can work with tree structures directly
            vector = self.tree_model.forward([structured])
        else:
            # Traditional approach: encode nodes then aggregate
            # Collect all nodes
            all_nodes = []
            def collect_nodes(node):
                all_nodes.append(node)
                for child in node.children:
                    collect_nodes(child)
            
            collect_nodes(structured)
            
            # Encode all nodes
            encoded_vectors = [self.encoder.encode_node(node) for node in all_nodes]
            
            # Apply tree model aggregation
            vector = self.tree_model.forward(encoded_vectors)
        
        # Make prediction
        return self.prediction_head.predict(vector)
    
    def run_batch(self, plans: List[Dict[str, Any]]) -> List[float]:
        """Process multiple plans in batch.
        
        Parameters
        ----------
        plans:
            List of raw query plan dictionaries
            
        Returns
        -------
        List[float]:
            List of performance predictions
        """
        return [self.run(plan) for plan in plans]
    
    def is_using_gnn(self) -> Dict[str, Any]:
        """Check which components are using GNN.
        
        Returns
        -------
        Dict[str, Any]:
            Information about GNN usage
        """
        return {
            'gnn_available': is_gnn_available(),
            'using_gnn': self._using_gnn,
            'encoder_gnn_mode': getattr(self.encoder, 'gnn_mode', False),
            'components': self.get_component_info()
        }
    
    def get_component_info(self) -> Dict[str, str]:
        """Get information about the components being used.
        
        Returns
        -------
        Dict[str, str]:
            Information about each component
        """
        return {
            'preprocessor': type(self.preprocessor).__name__,
            'encoder': type(self.encoder).__name__,
            'tree_model': type(self.tree_model).__name__,
            'prediction_head': type(self.prediction_head).__name__
        }
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about feature extraction.
        
        Returns
        -------
        Dict[str, Any]:
            Feature extraction information
        """
        if hasattr(self.encoder, 'get_feature_info'):
            return self.encoder.get_feature_info()
        return {'traditional_encoder': True}
    
    # Training support for GNN components
    def set_training_mode(self, training: bool = True):
        """Set training mode for GNN components.
        
        Parameters
        ----------
        training:
            Whether to set training mode (True) or evaluation mode (False)
        """
        if hasattr(self.tree_model, 'train_mode') and hasattr(self.tree_model, 'eval_mode'):
            if training:
                self.tree_model.train_mode()
            else:
                self.tree_model.eval_mode()
    
    def get_model_parameters(self):
        """Get trainable parameters from GNN components.
        
        Returns
        -------
        Iterator or None:
            Model parameters if GNN components are used, None otherwise
        """
        if hasattr(self.tree_model, 'get_parameters'):
            return self.tree_model.get_parameters()
        return None
    
    def save_components(self, path: str):
        """Save model components.
        
        Parameters
        ----------
        path:
            Base path to save components
        """
        # Save GNN model if available
        if hasattr(self.tree_model, 'save_model'):
            self.tree_model.save_model(f"{path}_tree_model.pth")
        
        # Save encoder vocabularies
        if hasattr(self.encoder, 'save_vocabularies'):
            self.encoder.save_vocabularies(f"{path}_encoder_vocab.json")
    
    def load_components(self, path: str):
        """Load model components.
        
        Parameters
        ----------
        path:
            Base path to load components from
        """
        # Load GNN model if available
        if hasattr(self.tree_model, 'load_model'):
            self.tree_model.load_model(f"{path}_tree_model.pth")
        
        # Load encoder vocabularies
        if hasattr(self.encoder, 'load_vocabularies'):
            self.encoder.load_vocabularies(f"{path}_encoder_vocab.json")
    
    def benchmark_performance(self, plans: List[Dict[str, Any]], 
                            iterations: int = 10) -> Dict[str, float]:
        """Benchmark the performance of the pipeline.
        
        Parameters
        ----------
        plans:
            List of plans to benchmark on
        iterations:
            Number of iterations for timing
            
        Returns
        -------
        Dict[str, float]:
            Timing results for each component
        """
        import time
        
        results = {}
        
        # Benchmark full pipeline
        start_time = time.time()
        for _ in range(iterations):
            for plan in plans:
                self.run(plan)
        results['full_pipeline'] = (time.time() - start_time) / (iterations * len(plans))
        
        # Benchmark individual components
        structured_plans = [self.preprocessor.preprocess(plan) for plan in plans]
        
        # Preprocessing
        start_time = time.time()
        for _ in range(iterations):
            for plan in plans:
                self.preprocessor.preprocess(plan)
        results['preprocessing'] = (time.time() - start_time) / (iterations * len(plans))
        
        # Encoding (for traditional approach)
        if not self._using_gnn:
            start_time = time.time()
            for _ in range(iterations):
                for structured in structured_plans:
                    # Collect all nodes
                    all_nodes = []
                    def collect_nodes(node):
                        all_nodes.append(node)
                        for child in node.children:
                            collect_nodes(child)
                    
                    collect_nodes(structured)
                    
                    # Encode all nodes
                    [self.encoder.encode_node(node) for node in all_nodes]
            results['encoding'] = (time.time() - start_time) / (iterations * len(plans))
        else:
            results['encoding'] = 0.0  # GNN handles encoding internally
        
        return results


# Convenience factory functions
def create_traditional_gnto() -> GNTO:
    """Create a traditional GNTO instance (original behavior)."""
    return GNTO(use_gnn=False)


def create_gnn_gnto(gnn_config: Optional[Dict[str, Any]] = None) -> GNTO:
    """Create a GNN-enhanced GNTO instance.
    
    Parameters
    ----------
    gnn_config:
        Custom GNN configuration
        
    Returns
    -------
    GNTO:
        GNN-enhanced GNTO instance (or traditional if GNN not available)
    """
    return GNTO(use_gnn=True, gnn_config=gnn_config)


def create_auto_gnto(**kwargs) -> GNTO:
    """Create GNTO with automatic GNN detection.
    
    Uses GNN if available, falls back to traditional otherwise.
    """
    use_gnn = is_gnn_available()
    if use_gnn:
        print("GNN components available - using enhanced pipeline")
    else:
        print("GNN components not available - using traditional pipeline")
    
    return GNTO(use_gnn=use_gnn, **kwargs)