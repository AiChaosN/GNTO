"""Pure Node-level Encoder for PlanNode objects.

This is the correct NODE ENCODER layer in the architecture:
📊 Architecture Position: Step 2 (Node-level Encoding)
- Input: Individual TreeNode with attributes (node_type, extra_info)
- Output: Node-level embedding vector
- Scope: ONLY single node feature extraction

⚠️  IMPORTANT: This encoder handles ONLY node-level features.
NO tree structure processing, NO recursive aggregation.
Structure-level encoding is handled by TreeModel.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class NodeFeatures:
    """Container for node features extracted from query plan nodes."""
    
    node_type_vector: np.ndarray  # One-hot encoding of node type
    numerical_features: np.ndarray  # Numerical features like cost, rows, etc.
    categorical_features: np.ndarray  # Encoded categorical features
    combined_features: np.ndarray  # Combined feature vector


class NodeEncoder:
    """Pure node-level encoder for converting individual PlanNodes to feature vectors.
    
    This is the NODE ENCODER layer in the correct architecture:
    📊 Architecture Position: Step 2 (Node-level Encoding)
    - Input: Individual TreeNode with attributes (node_type, extra_info)
    - Output: Node-level embedding vector
    - Scope: Single node feature extraction ONLY
    
    ⚠️  CRITICAL: This encoder handles ONLY node-level features.
    - NO recursive processing of children
    - NO tree structure aggregation
    - NO graph construction
    
    Structure-level encoding (tree/graph aggregation) is handled by TreeModel.
    
    Two encoding strategies:
    1. Simple mode: Only node type (one-hot)
    2. Rich mode: Node type + numerical + categorical features
    """
    
    def __init__(self, 
                 rich_features: bool = False,
                 feature_dim: Optional[int] = None,
                 include_numerical: bool = True,
                 include_categorical: bool = True,
                 normalize_features: bool = True) -> None:
        """Initialize the node encoder.
        
        Parameters
        ----------
        rich_features:
            If True, extracts comprehensive features beyond just node type
        feature_dim:
            Target dimension for output vectors (if None, uses dynamic size)
        include_numerical:
            Whether to include numerical features from extra_info
        include_categorical:
            Whether to include categorical features from extra_info
        normalize_features:
            Whether to normalize numerical features
        """
        # Core vocabulary for node types
        self.node_index: Dict[str, int] = {}
        
        # Feature extraction configuration
        self.rich_features = rich_features
        self.feature_dim = feature_dim
        self.include_numerical = include_numerical
        self.include_categorical = include_categorical
        self.normalize_features = normalize_features
        
        # Additional vocabularies for rich features
        self.categorical_vocabs: Dict[str, Dict[str, int]] = {}
        
        # Predefined feature keys for query plans
        self.numerical_keys = [
            'Total Cost', 'Startup Cost', 'Plan Rows', 'Plan Width',
            'Actual Total Time', 'Actual Rows', 'Actual Loops'
        ]
        
        self.categorical_keys = [
            'Join Type', 'Scan Direction', 'Strategy', 'Parent Relationship',
            'Relation Name', 'Index Name', 'Sort Method'
        ]
    
    # ------------------------------------------------------------------ utilities
    def _ensure_index(self, node_type: str) -> int:
        """Ensure a node type exists in vocabulary and return its index."""
        if node_type not in self.node_index:
            self.node_index[node_type] = len(self.node_index)
        return self.node_index[node_type]
    
    def _one_hot(self, idx: int) -> np.ndarray:
        """Create one-hot vector for given index."""
        vec = np.zeros(len(self.node_index), dtype=float)
        vec[idx] = 1.0
        return vec
    
    def _ensure_categorical_vocab(self, vocab_dict: Dict[str, int], key: str) -> int:
        """Ensure a key exists in categorical vocabulary and return its index."""
        if key not in vocab_dict:
            vocab_dict[key] = len(vocab_dict)
        return vocab_dict[key]
    
    # --------------------------------------------------------------------- encoder
    def encode_node(self, node) -> np.ndarray:
        """Encode a SINGLE PlanNode to feature vector and store it in node.node_vector.
        
        ⚠️  IMPORTANT: This method processes ONLY the given node.
        It does NOT process children or any tree structure.
        
        Parameters
        ----------
        node:
            PlanNode object to encode (children are IGNORED)
            
        Returns
        -------
        np.ndarray:
            Node-level feature vector (also stored in node.node_vector)
        """
        if not self.rich_features:
            # Simple mode: only node type
            vector = self._encode_simple(node)
        else:
            # Rich mode: comprehensive features
            vector = self._encode_rich(node)
        
        # Store the vector in the node
        node.node_vector = vector
        return vector
    
    def _encode_simple(self, node) -> np.ndarray:
        """Simple encoding: only node type (one-hot)."""
        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        return self._one_hot(idx)
    
    def _encode_rich(self, node) -> np.ndarray:
        """Rich encoding: node type + numerical + categorical features."""
        # Start with node type encoding
        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        node_type_vec = self._one_hot(idx)
        
        # Collect all feature components
        feature_components = [node_type_vec]
        
        # Add numerical features
        if self.include_numerical:
            numerical_features = self._extract_numerical_features(node)
            if len(numerical_features) > 0:
                feature_components.append(numerical_features)
        
        # Add categorical features
        if self.include_categorical:
            categorical_features = self._extract_categorical_features(node)
            if len(categorical_features) > 0:
                feature_components.append(categorical_features)
        
        # Combine all features
        combined_features = np.concatenate(feature_components)
        
        # Apply fixed dimension if specified
        if self.feature_dim is not None:
            combined_features = self._resize_vector(combined_features, self.feature_dim)
        
        return combined_features
    
    def encode_nodes(self, nodes: Iterable) -> List[np.ndarray]:
        """Encode multiple nodes into vectors and store them in each node.node_vector.
        
        Each node is processed independently and its vector is stored in node.node_vector.
        
        Parameters
        ----------
        nodes:
            Iterable of PlanNode objects to encode
            
        Returns
        -------
        List[np.ndarray]:
            List of node-level feature vectors (also stored in each node.node_vector)
        """
        return [self.encode_node(node) for node in nodes]
    
    # --------------------------------------------------------- feature extraction ---
    def _extract_numerical_features(self, node) -> np.ndarray:
        """Extract numerical features from node's extra_info."""
        extra_info = getattr(node, 'extra_info', {})
        features = []
        
        for key in self.numerical_keys:
            value = extra_info.get(key, 0.0)
            
            # Handle different value types
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                try:
                    features.append(float(value))
                except ValueError:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        if not features:
            return np.array([])
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize_features:
            features = self._normalize_numerical(features)
        
        return features
    
    def _extract_categorical_features(self, node) -> np.ndarray:
        """Extract categorical features from node's extra_info."""
        extra_info = getattr(node, 'extra_info', {})
        features = []
        
        for key in self.categorical_keys:
            value = extra_info.get(key, 'Unknown')
            
            # Ensure vocabulary exists for this key
            if key not in self.categorical_vocabs:
                self.categorical_vocabs[key] = {}
            
            # Convert value to string and get index
            str_value = str(value) if value is not None else 'Unknown'
            idx = self._ensure_categorical_vocab(self.categorical_vocabs[key], str_value)
            features.append(idx)
        
        return np.array(features, dtype=np.float32) if features else np.array([])
    
    def extract_node_features(self, node) -> NodeFeatures:
        """Extract comprehensive features from a single node.
        
        Parameters
        ----------
        node:
            PlanNode object to extract features from
            
        Returns
        -------
        NodeFeatures:
            Container with different types of extracted features
        """
        # Extract different types of features
        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        node_type_vec = self._one_hot(idx)
        numerical_vec = self._extract_numerical_features(node)
        categorical_vec = self._extract_categorical_features(node)
        
        # Combine features
        feature_components = [node_type_vec]
        if len(numerical_vec) > 0:
            feature_components.append(numerical_vec)
        if len(categorical_vec) > 0:
            feature_components.append(categorical_vec)
        
        combined_vec = np.concatenate(feature_components)
        if self.feature_dim is not None:
            combined_vec = self._resize_vector(combined_vec, self.feature_dim)
        
        return NodeFeatures(
            node_type_vector=node_type_vec,
            numerical_features=numerical_vec,
            categorical_features=categorical_vec,
            combined_features=combined_vec
        )
    
    # ---------------------------------------------------------------- internal ---
    def _normalize_numerical(self, features: np.ndarray) -> np.ndarray:
        """Normalize numerical features using min-max scaling."""
        if len(features) == 0:
            return features
        
        # Simple normalization - clip outliers and scale
        features = np.clip(features, 0, np.percentile(features, 95) if np.any(features > 0) else 1)
        max_val = np.maximum(features.max(), 1.0)
        return features / max_val
    
    def _resize_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize vector to target dimension."""
        if len(vector) == target_dim:
            return vector
        elif len(vector) < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:len(vector)] = vector
            return padded
        else:
            # Truncate
            return vector[:target_dim]
    
    # -------------------------------------------------------------- configuration ---
    def enable_rich_features(self, feature_dim: Optional[int] = None):
        """Enable rich feature extraction."""
        self.rich_features = True
        if feature_dim is not None:
            self.feature_dim = feature_dim
    
    def disable_rich_features(self):
        """Disable rich features (only node type)."""
        self.rich_features = False
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about current feature configuration."""
        return {
            'rich_features': self.rich_features,
            'feature_dim': self.feature_dim,
            'include_numerical': self.include_numerical,
            'include_categorical': self.include_categorical,
            'node_types_count': len(self.node_index),
            'categorical_vocabs': {k: len(v) for k, v in self.categorical_vocabs.items()}
        }
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get the sizes of different vocabularies."""
        vocab_sizes = {
            'node_type': len(self.node_index)
        }
        
        for key, vocab in self.categorical_vocabs.items():
            vocab_sizes[f'categorical_{key}'] = len(vocab)
        
        return vocab_sizes


# Convenience factory functions
def create_simple_node_encoder() -> NodeEncoder:
    """Create a simple node encoder (only node type)."""
    return NodeEncoder(rich_features=False)


def create_rich_node_encoder(feature_dim: int = 64, 
                           include_numerical: bool = True,
                           include_categorical: bool = True,
                           normalize_features: bool = True) -> NodeEncoder:
    """Create a rich node encoder with comprehensive features."""
    return NodeEncoder(
        rich_features=True,
        feature_dim=feature_dim,
        include_numerical=include_numerical,
        include_categorical=include_categorical,
        normalize_features=normalize_features
    )
