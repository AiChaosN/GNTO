"""
Main LQO Model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging

from .feature_spec import FeatureSpec, PlanBatch
from .encoders import NodeEncoder, StructureEncoder
from .heads import Heads, MultiTaskLoss


logger = logging.getLogger(__name__)


class LQOModel(nn.Module):
    """
    Learned Query Optimizer Model
    
    Combines feature extraction, representation learning, and multi-task prediction
    for query plan optimization tasks including cost estimation, plan ranking,
    and cardinality estimation.
    """
    
    def __init__(
        self,
        feature_spec: FeatureSpec,
        node_encoder_config: Optional[Dict[str, Any]] = None,
        structure_encoder_config: Optional[Dict[str, Any]] = None,
        heads_config: Optional[Dict[str, Any]] = None,
        tasks: List[str] = ["cost", "latency", "ranking"],
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.feature_spec = feature_spec
        self.tasks = tasks
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default configurations
        node_encoder_config = node_encoder_config or {}
        structure_encoder_config = structure_encoder_config or {}
        heads_config = heads_config or {}
        
        # Node encoder
        node_input_dim = feature_spec.feature_dims["total"]
        node_output_dim = node_encoder_config.get("output_dim", 128)
        
        self.node_encoder = NodeEncoder(
            input_dim=node_input_dim,
            hidden_dims=node_encoder_config.get("hidden_dims", [256, 128]),
            output_dim=node_output_dim,
            dropout=node_encoder_config.get("dropout", 0.1),
            activation=node_encoder_config.get("activation", "relu"),
            batch_norm=node_encoder_config.get("batch_norm", True)
        )
        
        # Structure encoder
        structure_hidden_dim = structure_encoder_config.get("hidden_dim", 128)
        
        self.structure_encoder = StructureEncoder(
            node_dim=node_output_dim,
            hidden_dim=structure_hidden_dim,
            num_layers=structure_encoder_config.get("num_layers", 3),
            num_edge_types=structure_encoder_config.get("num_edge_types", 10),
            gnn_type=structure_encoder_config.get("gnn_type", "gcn"),
            heads=structure_encoder_config.get("heads", 4),
            dropout=structure_encoder_config.get("dropout", 0.1),
            pooling=structure_encoder_config.get("pooling", "mean"),
            residual=structure_encoder_config.get("residual", True)
        )
        
        # Prediction heads
        self.heads = Heads(
            plan_emb_dim=structure_hidden_dim,
            node_emb_dim=structure_hidden_dim,
            tasks=tasks,
            hidden_dims=heads_config.get("hidden_dims", [128, 64]),
            dropout=heads_config.get("dropout", 0.1),
            uncertainty=heads_config.get("uncertainty", False)
        )
        
        # Loss function
        self.loss_fn_module = MultiTaskLoss(
            tasks=tasks,
            loss_weights=heads_config.get("loss_weights"),
            adaptive_weighting=heads_config.get("adaptive_weighting", True)
        )
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized LQOModel with tasks: {tasks}")
        logger.info(f"Feature dimensions: {feature_spec.feature_dims}")
        logger.info(f"Device: {self.device}")
    
    def forward(self, batch: Union[PlanBatch, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            batch: PlanBatch object or dictionary containing:
                - node_features: [N, d_node] node features
                - edge_index: [2, E] edge indices  
                - edge_types: [E] edge type indices
                - batch_idx: [N] batch assignment for nodes
                - plan_sizes: [B] number of nodes per plan
        
        Returns:
            Dictionary containing predictions for all tasks
        """
        # Handle different input formats
        if isinstance(batch, dict):
            # Convert dict to PlanBatch if necessary
            if not isinstance(batch, PlanBatch):
                batch = PlanBatch(
                    node_features=batch["node_features"],
                    edge_index=batch["edge_index"], 
                    edge_types=batch["edge_types"],
                    batch_idx=batch["batch_idx"],
                    plan_sizes=batch["plan_sizes"],
                    metadata=batch.get("metadata", {})
                )
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # 1. Node encoding
        node_embeddings = self.node_encoder(batch.node_features)  # [N, d_node]
        
        # 2. Structure encoding
        structure_output = self.structure_encoder(
            nodes=node_embeddings,
            edges=batch.edge_index,
            edge_types=batch.edge_types,
            batch_idx=batch.batch_idx
        )
        
        node_emb = structure_output["node_emb"]  # [N, d_h]
        plan_emb = structure_output.get("plan_emb")  # [B, d_h]
        
        # 3. Prediction heads
        predictions = self.heads(
            plan_emb=plan_emb,
            node_emb=node_emb,
            extra_ctx=batch.metadata
        )
        
        return predictions
    
    def loss_fn(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions from forward pass
            targets: Ground truth targets
        
        Returns:
            Dictionary with individual task losses and total loss
        """
        return self.loss_fn_module(predictions, targets)
    
    def predict(
        self, 
        plan_json: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        High-level prediction interface
        
        Args:
            plan_json: Single plan dict or list of plan dicts
        
        Returns:
            Dictionary with predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to batch format
            batch = self.feature_spec.tensorize(plan_json)
            
            # Forward pass
            predictions = self.forward(batch)
            
            # Move to CPU for easier handling
            predictions = {k: v.cpu() for k, v in predictions.items()}
            
        return predictions
    
    def get_plan_embedding(
        self, 
        plan_json: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> torch.Tensor:
        """
        Extract plan-level embeddings
        
        Args:
            plan_json: Single plan dict or list of plan dicts
        
        Returns:
            Plan embeddings [B, d_h]
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to batch format
            batch = self.feature_spec.tensorize(plan_json)
            batch = batch.to(self.device)
            
            # Forward pass through encoders only
            node_embeddings = self.node_encoder(batch.node_features)
            
            structure_output = self.structure_encoder(
                nodes=node_embeddings,
                edges=batch.edge_index,
                edge_types=batch.edge_types,
                batch_idx=batch.batch_idx
            )
            
            plan_emb = structure_output.get("plan_emb")
            
            if plan_emb is not None:
                return plan_emb.cpu()
            else:
                # Fallback: global pooling of node embeddings
                from torch_geometric.nn import global_mean_pool
                return global_mean_pool(node_embeddings, batch.batch_idx).cpu()
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "feature_spec_config": {
                "config": self.feature_spec.config.dict(),
                "feature_dims": self.feature_spec.feature_dims
            },
            "tasks": self.tasks
        }
        
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(
        cls, 
        path: str, 
        device: Optional[torch.device] = None
    ) -> "LQOModel":
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        # Reconstruct feature spec
        from .feature_spec import NodeFeatureConfig
        config = NodeFeatureConfig(**checkpoint["feature_spec_config"]["config"])
        feature_spec = FeatureSpec(config)
        
        # Create model
        model = cls(
            feature_spec=feature_spec,
            tasks=checkpoint["tasks"],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}")
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "tasks": self.tasks,
            "feature_dims": self.feature_spec.feature_dims,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assume float32
        }
    
    def freeze_encoder(self, freeze: bool = True):
        """Freeze/unfreeze encoder parameters"""
        for param in self.node_encoder.parameters():
            param.requires_grad = not freeze
        for param in self.structure_encoder.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Encoder parameters {status}")
    
    def freeze_heads(self, freeze: bool = True):
        """Freeze/unfreeze prediction head parameters"""
        for param in self.heads.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Prediction head parameters {status}")
