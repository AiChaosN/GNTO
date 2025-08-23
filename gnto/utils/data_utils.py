"""
Data loading and processing utilities
"""

import json
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from ..core.feature_spec import FeatureSpec, PlanBatch


logger = logging.getLogger(__name__)


class PlanDataset(Dataset):
    """Dataset for execution plans"""
    
    def __init__(
        self,
        plans: List[Dict[str, Any]],
        targets: Optional[List[Dict[str, Any]]] = None,
        feature_spec: Optional[FeatureSpec] = None,
        transform: Optional[callable] = None
    ):
        self.plans = plans
        self.targets = targets or [{} for _ in plans]
        self.feature_spec = feature_spec
        self.transform = transform
        
        if len(self.plans) != len(self.targets):
            raise ValueError("Number of plans and targets must match")
    
    def __len__(self) -> int:
        return len(self.plans)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        plan = self.plans[idx]
        target = self.targets[idx]
        
        # Apply transform if provided
        if self.transform:
            plan = self.transform(plan)
        
        return {
            "plan": plan,
            "target": target,
            "idx": idx
        }


def collate_plans(batch: List[Dict[str, Any]], feature_spec: FeatureSpec) -> Tuple[PlanBatch, Dict[str, torch.Tensor]]:
    """
    Collate function for plan batches
    
    Args:
        batch: List of samples from PlanDataset
        feature_spec: Feature specification for tensorization
    
    Returns:
        Tuple of (PlanBatch, target_dict)
    """
    plans = [item["plan"] for item in batch]
    targets = [item["target"] for item in batch]
    
    # Tensorize plans
    plan_batch = feature_spec.tensorize(plans)
    
    # Collate targets
    target_dict = {}
    if targets and targets[0]:  # Check if targets are not empty
        # Get all target keys
        all_keys = set()
        for target in targets:
            all_keys.update(target.keys())
        
        # Collate each target type
        for key in all_keys:
            values = []
            for target in targets:
                if key in target:
                    val = target[key]
                    if isinstance(val, (int, float)):
                        values.append(val)
                    elif isinstance(val, (list, tuple)):
                        values.extend(val)
                    else:
                        values.append(val)
            
            if values and isinstance(values[0], (int, float)):
                target_dict[key] = torch.tensor(values, dtype=torch.float32)
            elif values:
                target_dict[key] = values
    
    return plan_batch, target_dict


def load_dataset(
    data_path: Union[str, Path],
    feature_spec: FeatureSpec,
    target_columns: Optional[List[str]] = None,
    format: str = "auto"
) -> PlanDataset:
    """
    Load dataset from file
    
    Args:
        data_path: Path to data file
        feature_spec: Feature specification
        target_columns: List of target column names
        format: File format (auto, json, pickle, csv, parquet)
    
    Returns:
        PlanDataset object
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Auto-detect format
    if format == "auto":
        suffix = data_path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in [".pkl", ".pickle"]:
            format = "pickle"
        elif suffix == ".csv":
            format = "csv"
        elif suffix == ".parquet":
            format = "parquet"
        else:
            raise ValueError(f"Cannot auto-detect format for {data_path}")
    
    logger.info(f"Loading dataset from {data_path} (format: {format})")
    
    # Load data based on format
    if format == "json":
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            plans = data.get("plans", [])
            targets = data.get("targets", [])
        else:
            # Assume list of records
            plans = []
            targets = []
            for record in data:
                plan = record.copy()
                target = {}
                
                # Extract target columns
                if target_columns:
                    for col in target_columns:
                        if col in plan:
                            target[col] = plan.pop(col)
                
                plans.append(plan)
                targets.append(target)
    
    elif format == "pickle":
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            plans = data.get("plans", [])
            targets = data.get("targets", [])
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            plans, targets = data
        else:
            raise ValueError("Pickle file must contain dict with 'plans'/'targets' or tuple of (plans, targets)")
    
    elif format in ["csv", "parquet"]:
        if format == "csv":
            df = pd.read_csv(data_path)
        else:
            df = pd.read_parquet(data_path)
        
        # Extract plan and target data
        plans = []
        targets = []
        
        for _, row in df.iterrows():
            plan = {}
            target = {}
            
            for col, val in row.items():
                if target_columns and col in target_columns:
                    target[col] = val
                else:
                    # Try to parse JSON columns
                    if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                        try:
                            plan[col] = json.loads(val)
                        except:
                            plan[col] = val
                    else:
                        plan[col] = val
            
            plans.append(plan)
            targets.append(target)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded {len(plans)} plans with {len(targets)} targets")
    
    return PlanDataset(plans, targets, feature_spec)


def create_data_loaders(
    dataset: PlanDataset,
    feature_spec: FeatureSpec,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create train/val/test data loaders
    
    Args:
        dataset: PlanDataset object
        feature_spec: Feature specification
        batch_size: Batch size
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        Dictionary with train/val/test data loaders
    """
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Calculate split indices
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val
    
    # Split dataset
    if shuffle:
        indices = torch.randperm(n_samples).tolist()
    else:
        indices = list(range(n_samples))
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subsets
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create collate function
    def collate_fn(batch):
        return collate_plans(batch, feature_spec)
    
    # Create data loaders
    data_loaders = {}
    
    if len(train_dataset) > 0:
        data_loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    if len(val_dataset) > 0:
        data_loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    if len(test_dataset) > 0:
        data_loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return data_loaders


def normalize_features(
    dataset: PlanDataset,
    feature_columns: List[str],
    method: str = "standard"
) -> Tuple[PlanDataset, Dict[str, Any]]:
    """
    Normalize continuous features in dataset
    
    Args:
        dataset: Input dataset
        feature_columns: List of feature columns to normalize
        method: Normalization method (standard, minmax)
    
    Returns:
        Tuple of (normalized_dataset, normalization_stats)
    """
    # Extract feature values
    feature_values = {col: [] for col in feature_columns}
    
    for plan in dataset.plans:
        for node in plan.get("nodes", []):
            for col in feature_columns:
                if col in node:
                    feature_values[col].append(node[col])
    
    # Compute normalization statistics
    stats = {}
    
    for col in feature_columns:
        values = np.array(feature_values[col])
        
        if method == "standard":
            mean = values.mean()
            std = values.std()
            stats[col] = {"mean": mean, "std": std, "method": "standard"}
        elif method == "minmax":
            min_val = values.min()
            max_val = values.max()
            stats[col] = {"min": min_val, "max": max_val, "method": "minmax"}
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    # Apply normalization
    normalized_plans = []
    
    for plan in dataset.plans:
        normalized_plan = plan.copy()
        normalized_nodes = []
        
        for node in plan.get("nodes", []):
            normalized_node = node.copy()
            
            for col in feature_columns:
                if col in normalized_node:
                    val = normalized_node[col]
                    
                    if method == "standard":
                        normalized_val = (val - stats[col]["mean"]) / (stats[col]["std"] + 1e-8)
                    elif method == "minmax":
                        val_range = stats[col]["max"] - stats[col]["min"]
                        normalized_val = (val - stats[col]["min"]) / (val_range + 1e-8)
                    
                    normalized_node[col] = normalized_val
            
            normalized_nodes.append(normalized_node)
        
        normalized_plan["nodes"] = normalized_nodes
        normalized_plans.append(normalized_plan)
    
    # Create normalized dataset
    normalized_dataset = PlanDataset(
        normalized_plans,
        dataset.targets,
        dataset.feature_spec,
        dataset.transform
    )
    
    return normalized_dataset, stats


def apply_normalization(
    plan: Dict[str, Any],
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply normalization to a single plan
    
    Args:
        plan: Input plan
        stats: Normalization statistics
    
    Returns:
        Normalized plan
    """
    normalized_plan = plan.copy()
    normalized_nodes = []
    
    for node in plan.get("nodes", []):
        normalized_node = node.copy()
        
        for col, stat in stats.items():
            if col in normalized_node:
                val = normalized_node[col]
                
                if stat["method"] == "standard":
                    normalized_val = (val - stat["mean"]) / (stat["std"] + 1e-8)
                elif stat["method"] == "minmax":
                    val_range = stat["max"] - stat["min"]
                    normalized_val = (val - stat["min"]) / (val_range + 1e-8)
                else:
                    normalized_val = val
                
                normalized_node[col] = normalized_val
        
        normalized_nodes.append(normalized_node)
    
    normalized_plan["nodes"] = normalized_nodes
    return normalized_plan
