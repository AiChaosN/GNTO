"""Tree-level encoders for query plan structures.

This is the TREE ENCODER layer in the clean architecture:
📊 Architecture Position: Step 3 (Structure-level Encoding)
- Input: Tree/DAG with node-level embeddings
- Output: Global plan embedding vector
- Scope: Structural relationship modeling

Provides multiple tree encoding approaches:
1. Traditional: Simple statistical aggregation (mean/sum/max)
2. Tree-LSTM/Recursive NN: Hierarchical tree processing  
3. GNN: Graph neural networks for complex DAG relationships
4. Transformer: Sequential processing with positional encoding

⚠️  IMPORTANT: This handles tree-level encoding ONLY.
Node-level feature extraction is handled by the NodeEncoder class.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    _GNN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GNN_AVAILABLE = False


class TreeEncoder(nn.Module):
    """Tree-level encoder using simple statistical aggregation."""

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize tree model with reduction method.

        Parameters
        ----------
        reduction:
            Reduction method: 'mean' or 'sum'
        """
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, vectors: Iterable[Union[np.ndarray, 'torch.Tensor']]) -> 'torch.Tensor':
        """Reduce vectors into a single tensor."""
        vec_list = list(vectors)
        stacked = self._pad_and_stack(vec_list)
        if self.reduction == "mean":
            return stacked.mean(dim=0)
        return stacked.sum(dim=0)

    def _pad_and_stack(self, vecs: List[Union[np.ndarray, 'torch.Tensor']]) -> 'torch.Tensor':
        """Pad vectors to equal length and stack them using torch."""
        if not vecs:
            return torch.zeros(0)
        tensor_vecs = [torch.as_tensor(v, dtype=torch.float32) for v in vecs]
        max_len = max(v.shape[0] for v in tensor_vecs)
        stacked = torch.zeros(len(tensor_vecs), max_len, dtype=torch.float32)
        for i, v in enumerate(tensor_vecs):
            stacked[i, : v.shape[0]] = v
        return stacked


# GNN-based models (only available if PyTorch is installed)
if _GNN_AVAILABLE:
    
    class TreeToGraphConverter:
        """Convert tree structures to graph format for GNN processing."""
        
        def __init__(self, encoder=None):
            self.node_counter = 0
            self.encoder = encoder
        
        def tree_to_graph(self, node) -> Tuple[torch.Tensor, List[np.ndarray]]:
            """Convert a tree structure to graph edges and collect node features."""
            self.node_counter = 0
            edges = []
            features = []
            
            def _traverse(current_node, parent_id=None):
                current_id = self.node_counter
                self.node_counter += 1
                
                # Extract node features
                if hasattr(current_node, 'feature_vector'):
                    features.append(current_node.feature_vector)
                elif self.encoder is not None:
                    # Use encoder for feature extraction
                    node_type = getattr(current_node, 'node_type', 'Unknown')
                    if hasattr(self.encoder, 'gnn_mode') and self.encoder.gnn_mode:
                        # Create temporary node without children for individual encoding
                        temp_node = type('TempNode', (), {
                            'node_type': node_type,
                            'children': [],
                            'extra_info': getattr(current_node, 'extra_info', {})
                        })()
                        feature_vec = self.encoder._encode_gnn(temp_node)
                    else:
                        # Original encoder behavior
                        idx = self.encoder._ensure_index(node_type)
                        feature_vec = self.encoder._one_hot(idx)
                    features.append(feature_vec)
                else:
                    # Fallback: simple feature
                    node_type = getattr(current_node, 'node_type', 'Unknown')
                    features.append(np.array([hash(node_type) % 100], dtype=np.float32))
                
                # Add edges (bidirectional)
                if parent_id is not None:
                    edges.append([parent_id, current_id])
                    edges.append([current_id, parent_id])
                
                # Process children
                for child in getattr(current_node, 'children', []):
                    _traverse(child, current_id)
            
            _traverse(node)
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            return edge_index, features
    
    
    class GCNTreeEncoder(nn.Module):
        """Graph Convolutional Network for tree aggregation."""
        
        def __init__(self, 
                     input_dim: int = 64,
                     hidden_dim: int = 128,
                     output_dim: int = 64,
                     num_layers: int = 3,
                     dropout: float = 0.1,
                     pooling: str = "mean"):
            super(GCNTreeEncoder, self).__init__()
            
            self.num_layers = num_layers
            self.pooling = pooling
            self.dropout = dropout
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # GCN layers
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            
            # Batch normalization
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                   batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Input projection
            x = self.input_proj(x)
            x = F.relu(x)
            
            # GCN layers
            for i, conv in enumerate(self.convs):
                residual = x
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Residual connection
                if x.size() == residual.size():
                    x = x + residual
            
            # Global pooling
            if batch is None:
                if self.pooling == "mean":
                    x = torch.mean(x, dim=0, keepdim=True)
                else:
                    x = torch.max(x, dim=0, keepdim=True)[0]
            else:
                if self.pooling == "mean":
                    x = global_mean_pool(x, batch)
                else:
                    x = global_max_pool(x, batch)
            
            # Output projection
            x = self.output_proj(x)
            return x
    
    
    class GATTreeEncoder(nn.Module):
        """Graph Attention Network for tree aggregation."""
        
        def __init__(self, 
                     input_dim: int = 64,
                     hidden_dim: int = 128,
                     output_dim: int = 64,
                     num_layers: int = 3,
                     num_heads: int = 4,
                     dropout: float = 0.1,
                     pooling: str = "mean"):
            super(GATTreeEncoder, self).__init__()
            
            self.num_layers = num_layers
            self.pooling = pooling
            self.dropout = dropout
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # GAT layers
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                if i == num_layers - 1:
                    self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
                else:
                    self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, 
                                            heads=num_heads, dropout=dropout))
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                   batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Input projection
            x = self.input_proj(x)
            x = F.relu(x)
            
            # GAT layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling
            if batch is None:
                if self.pooling == "mean":
                    x = torch.mean(x, dim=0, keepdim=True)
                else:
                    x = torch.max(x, dim=0, keepdim=True)[0]
            else:
                if self.pooling == "mean":
                    x = global_mean_pool(x, batch)
                else:
                    x = global_max_pool(x, batch)
            
            # Output projection
            x = self.output_proj(x)
            return x
    
    
    class GNNTreeEncoder:
        """GNN-based tree model with traditional TreeModel interface."""
        
        def __init__(self, 
                     model_type: str = "gcn",
                     input_dim: int = 64,
                     hidden_dim: int = 128,
                     output_dim: int = 64,
                     num_layers: int = 3,
                     num_heads: int = 4,
                     dropout: float = 0.1,
                     pooling: str = "mean",
                     device: str = "cpu",
                     encoder = None):
            """Initialize GNN Tree Model."""
            self.device = torch.device(device)
            self.converter = TreeToGraphConverter(encoder=encoder)
            self.input_dim = input_dim
            
            # Initialize the GNN model
            if model_type.lower() == "gcn":
                self.model = GCNTreeEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    pooling=pooling
                )
            elif model_type.lower() == "gat":
                self.model = GATTreeEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    pooling=pooling
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model.to(self.device)
            self.model.eval()
        
        def forward(self, vectors: Union[Iterable[np.ndarray], List]) -> np.ndarray:
            """Process input using GNN (compatible with TreeModel interface)."""
            vectors = list(vectors)
            
            if not vectors:
                return np.zeros(self.model.output_proj.out_features)
            
            # Handle different input types
            if isinstance(vectors[0], np.ndarray):
                # Traditional vector input - use simple aggregation
                return self._process_vectors(vectors)
            elif hasattr(vectors[0], 'node_type'):
                # Tree structure input - use GNN
                return self._process_tree(vectors[0])
            else:
                raise ValueError("Input must be either numpy arrays or tree structures")
        
        def _process_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
            """Fallback to simple aggregation for vector input."""
            stacked = self._pad_and_stack(vectors)
            return stacked.mean(axis=0)
        
        def _process_tree(self, tree_root) -> np.ndarray:
            """Process tree structure using GNN."""
            with torch.no_grad():
                edge_index, features = self.converter.tree_to_graph(tree_root)
                
                if not features:
                    return np.zeros(self.model.output_proj.out_features)
                
                # Prepare input
                x = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)
                
                # Pad/truncate features to input_dim
                if x.size(1) < self.input_dim:
                    padding = torch.zeros(x.size(0), self.input_dim - x.size(1)).to(self.device)
                    x = torch.cat([x, padding], dim=1)
                elif x.size(1) > self.input_dim:
                    x = x[:, :self.input_dim]
                
                edge_index = edge_index.to(self.device)
                
                # Forward pass
                output = self.model(x, edge_index)
                return output.cpu().numpy().flatten()
        
        def _pad_and_stack(self, vecs: List[np.ndarray]) -> np.ndarray:
            """Helper method for vector padding."""
            if not vecs:
                return np.zeros(0)
            max_len = max(len(v) for v in vecs)
            stacked = np.zeros((len(vecs), max_len))
            for i, v in enumerate(vecs):
                stacked[i, :len(v)] = v
            return stacked
        
        # Training utilities
        def train_mode(self):
            """Set model to training mode."""
            self.model.train()
        
        def eval_mode(self):
            """Set model to evaluation mode."""
            self.model.eval()
        
        def get_parameters(self):
            """Get model parameters for optimization."""
            return self.model.parameters()
        
        def save_model(self, path: str):
            """Save model state dict."""
            torch.save(self.model.state_dict(), path)
        
        def load_model(self, path: str):
            """Load model state dict."""
            self.model.load_state_dict(torch.load(path, map_location=self.device))

else:
    # Dummy classes when GNN is not available
    class GNNTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("GNN TreeModel requires PyTorch and PyTorch Geometric. "
                            "Install with: pip install torch torch-geometric")
    
    class GCNTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("GCN TreeModel requires PyTorch and PyTorch Geometric")
    
    class GATTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("GAT TreeModel requires PyTorch and PyTorch Geometric")


# Convenience factory functions
def create_tree_encoder(use_gnn: bool = False, **kwargs) -> Union[TreeEncoder, 'GNNTreeEncoder']:
    """Create a tree encoder based on requirements.
    
    Parameters
    ----------
    use_gnn:
        Whether to use GNN-based model
    **kwargs:
        Additional arguments for the model
        
    Returns
    -------
    TreeEncoder or GNNTreeEncoder:
        Appropriate tree encoder instance
    """
    if use_gnn:
        if not _GNN_AVAILABLE:
            print("Warning: GNN not available, falling back to traditional TreeEncoder")
            return TreeEncoder(**{k: v for k, v in kwargs.items() if k in ['reduction']})
        return GNNTreeEncoder(**kwargs)
    else:
        return TreeEncoder(**{k: v for k, v in kwargs.items() if k in ['reduction']})


def is_gnn_available() -> bool:
    """Check if GNN components are available."""
    return _GNN_AVAILABLE
