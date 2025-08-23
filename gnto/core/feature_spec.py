"""
Feature specification and plan processing utilities
"""

from typing import Dict, List, Any, Optional, Union
import torch
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field


class PlanBatch:
    """Batch of tensorized execution plans"""
    
    def __init__(
        self,
        node_features: torch.Tensor,  # [total_nodes, d_node]
        edge_index: torch.Tensor,     # [2, total_edges] 
        edge_types: torch.Tensor,     # [total_edges]
        batch_idx: torch.Tensor,      # [total_nodes] - which plan each node belongs to
        plan_sizes: torch.Tensor,     # [batch_size] - number of nodes per plan
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.batch_idx = batch_idx
        self.plan_sizes = plan_sizes
        self.metadata = metadata or {}
        
    @property
    def batch_size(self) -> int:
        return len(self.plan_sizes)
    
    @property
    def total_nodes(self) -> int:
        return len(self.node_features)
    
    def to(self, device: torch.device) -> "PlanBatch":
        """Move batch to device"""
        return PlanBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_types=self.edge_types.to(device),
            batch_idx=self.batch_idx.to(device),
            plan_sizes=self.plan_sizes.to(device),
            metadata=self.metadata
        )


class NodeFeatureConfig(BaseModel):
    """Configuration for node features"""
    # Continuous features
    continuous_features: List[str] = Field(
        default=["rows", "ndv", "selectivity", "io_cost", "cpu_cost", "parallel_degree"]
    )
    
    # Categorical features
    categorical_features: Dict[str, int] = Field(
        default={
            "operator_type": 50,  # vocab size
            "join_type": 10,
            "index_type": 20,
            "storage_format": 15,
            "hint": 30
        }
    )
    
    # Structure features
    structure_features: List[str] = Field(
        default=["is_blocking", "is_pipeline", "is_probe", "is_build", "stage_id"]
    )


class FeatureSpec:
    """
    Specification for feature extraction and plan tensorization
    Handles validation and conversion of execution plans to tensor format
    """
    
    def __init__(self, config: Optional[NodeFeatureConfig] = None):
        self.config = config or NodeFeatureConfig()
        self.feature_dims = self._compute_feature_dims()
        
    def _compute_feature_dims(self) -> Dict[str, int]:
        """Compute dimensionality of different feature types"""
        dims = {
            "continuous": len(self.config.continuous_features),
            "categorical": sum(self.config.categorical_features.values()),
            "structure": len(self.config.structure_features)
        }
        dims["total"] = sum(dims.values())
        return dims
    
    def validate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a single execution plan
        
        Args:
            plan: Dictionary representation of execution plan
            
        Returns:
            Validated and normalized plan dictionary
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "normalized_plan": None
        }
        
        try:
            # Check required fields
            required_fields = ["nodes", "edges"]
            for field in required_fields:
                if field not in plan:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["is_valid"] = False
            
            if not validation_result["is_valid"]:
                return validation_result
            
            # Validate nodes
            nodes = plan["nodes"]
            if not isinstance(nodes, list) or len(nodes) == 0:
                validation_result["errors"].append("Nodes must be a non-empty list")
                validation_result["is_valid"] = False
                return validation_result
            
            # Validate each node has required features
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    validation_result["errors"].append(f"Node {i} must be a dictionary")
                    validation_result["is_valid"] = False
                    continue
                
                # Check for required continuous features with defaults
                for feat in self.config.continuous_features:
                    if feat not in node:
                        validation_result["warnings"].append(
                            f"Node {i} missing feature {feat}, using default 0.0"
                        )
                        node[feat] = 0.0
                
                # Check for required categorical features
                for feat in self.config.categorical_features:
                    if feat not in node:
                        validation_result["warnings"].append(
                            f"Node {i} missing feature {feat}, using default 'unknown'"
                        )
                        node[feat] = "unknown"
            
            # Validate edges
            edges = plan["edges"]
            if not isinstance(edges, list):
                validation_result["errors"].append("Edges must be a list")
                validation_result["is_valid"] = False
                return validation_result
            
            # Normalize edge format
            normalized_edges = []
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    edge_dict = {
                        "source": edge[0],
                        "target": edge[1],
                        "type": edge[2] if len(edge) > 2 else "default"
                    }
                    normalized_edges.append(edge_dict)
                elif isinstance(edge, dict):
                    if "source" not in edge or "target" not in edge:
                        validation_result["errors"].append("Edge missing source or target")
                        validation_result["is_valid"] = False
                        continue
                    edge.setdefault("type", "default")
                    normalized_edges.append(edge)
            
            # Create normalized plan
            normalized_plan = {
                "nodes": nodes,
                "edges": normalized_edges,
                "metadata": plan.get("metadata", {})
            }
            
            validation_result["normalized_plan"] = normalized_plan
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    def _extract_node_features(self, node: Dict[str, Any]) -> torch.Tensor:
        """Extract features from a single node"""
        features = []
        
        # Continuous features
        continuous_vals = []
        for feat in self.config.continuous_features:
            val = node.get(feat, 0.0)
            continuous_vals.append(float(val))
        features.extend(continuous_vals)
        
        # Categorical features (one-hot encoding)
        for feat, vocab_size in self.config.categorical_features.items():
            val = node.get(feat, "unknown")
            # Simple hash-based encoding for now
            # In production, you'd want proper vocabulary mapping
            encoded_val = hash(str(val)) % vocab_size
            one_hot = [0.0] * vocab_size
            one_hot[encoded_val] = 1.0
            features.extend(one_hot)
        
        # Structure features
        structure_vals = []
        for feat in self.config.structure_features:
            val = node.get(feat, 0)
            structure_vals.append(float(val))
        features.extend(structure_vals)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def tensorize(self, plan: Union[Dict[str, Any], List[Dict[str, Any]]]) -> PlanBatch:
        """
        Convert execution plan(s) to tensor format
        
        Args:
            plan: Single plan dict or list of plan dicts
            
        Returns:
            PlanBatch object containing tensorized plans
        """
        if isinstance(plan, dict):
            plans = [plan]
        else:
            plans = plan
        
        # Validate all plans
        validated_plans = []
        for p in plans:
            validation_result = self.validate(p)
            if not validation_result["is_valid"]:
                raise ValueError(f"Plan validation failed: {validation_result['errors']}")
            validated_plans.append(validation_result["normalized_plan"])
        
        # Extract features
        all_node_features = []
        all_edges = []
        batch_indices = []
        plan_sizes = []
        
        node_offset = 0
        
        for batch_idx, plan_dict in enumerate(validated_plans):
            nodes = plan_dict["nodes"]
            edges = plan_dict["edges"]
            
            # Extract node features
            plan_node_features = []
            for node in nodes:
                node_feat = self._extract_node_features(node)
                plan_node_features.append(node_feat)
                batch_indices.append(batch_idx)
            
            all_node_features.extend(plan_node_features)
            plan_sizes.append(len(nodes))
            
            # Process edges
            edge_list = []
            edge_types = []
            
            for edge in edges:
                source = edge["source"] + node_offset
                target = edge["target"] + node_offset
                edge_type = edge.get("type", "default")
                
                edge_list.append([source, target])
                # Simple edge type encoding
                edge_type_id = hash(edge_type) % 10  # Assume 10 edge types max
                edge_types.append(edge_type_id)
            
            all_edges.extend(edge_list)
            node_offset += len(nodes)
        
        # Convert to tensors
        node_features = torch.stack(all_node_features) if all_node_features else torch.empty(0, self.feature_dims["total"])
        
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_types = torch.empty(0, dtype=torch.long)
        
        batch_idx = torch.tensor(batch_indices, dtype=torch.long)
        plan_sizes = torch.tensor(plan_sizes, dtype=torch.long)
        
        return PlanBatch(
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            batch_idx=batch_idx,
            plan_sizes=plan_sizes,
            metadata={"num_plans": len(validated_plans)}
        )
