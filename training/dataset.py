"""Dataset classes for loading and preprocessing query execution plans."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

try:
    from ..models import DataPreprocessor, PlanNode
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import DataPreprocessor, PlanNode

logger = logging.getLogger(__name__)


@dataclass
class PlanSample:
    """A single training sample containing a query plan and target values."""
    plan_json: Dict[str, Any]
    plan_tree: PlanNode
    targets: Dict[str, float]
    metadata: Dict[str, Any]


class PlanDataset:
    """Dataset class for query execution plans.
    
    This class handles loading CSV data containing JSON query plans,
    preprocessing them into structured trees, and providing samples
    for training.
    """
    
    def __init__(self, 
                 csv_path: Union[str, Path],
                 target_columns: Optional[List[str]] = None,
                 preprocessor: Optional[DataPreprocessor] = None):
        """Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing plans
            target_columns: List of target column names to extract from plans
            preprocessor: DataPreprocessor instance for parsing plans
        """
        self.csv_path = Path(csv_path)
        self.target_columns = target_columns or ['Actual Total Time']
        self.preprocessor = preprocessor or DataPreprocessor()
        
        self.samples: List[PlanSample] = []
        self.statistics: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}  # Alias for compatibility
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data from CSV file."""
        logger.info(f"Loading data from {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        if 'json' not in df.columns:
            raise ValueError("CSV must contain 'json' column with plan data")
        
        # Process each row
        valid_samples = 0
        for idx, row in df.iterrows():
            try:
                # Parse JSON plan
                plan_json = json.loads(row['json'])
                
                # Extract plan tree (use the root "Plan" node from JSON)
                plan_root = plan_json.get('Plan', plan_json)
                plan_tree = self.preprocessor.preprocess(plan_root)
                
                # Extract target values
                targets = self._extract_targets(plan_json)
                
                # Create metadata
                metadata = {
                    'row_id': idx,
                    'plan_size': self._count_nodes(plan_tree),
                }
                
                sample = PlanSample(
                    plan_json=plan_json,
                    plan_tree=plan_tree, 
                    targets=targets,
                    metadata=metadata
                )
                
                self.samples.append(sample)
                valid_samples += 1
                
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {valid_samples} samples")
        self._compute_statistics()
    
    def _extract_targets(self, plan_json: Dict[str, Any]) -> Dict[str, float]:
        """Extract target values from plan JSON."""
        targets = {}
        
        # Navigate to the root plan node
        plan_root = plan_json.get('Plan', {})
        
        for target_col in self.target_columns:
            if target_col in plan_root:
                targets[target_col] = float(plan_root[target_col])
            else:
                # Try common variations
                variations = [
                    target_col.replace(' ', ''),
                    target_col.replace(' ', '_').lower(),
                    target_col.lower()
                ]
                
                found = False
                for var in variations:
                    if var in plan_root:
                        targets[target_col] = float(plan_root[var])
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Target column '{target_col}' not found in plan")
                    targets[target_col] = 0.0
        
        return targets
    
    def _count_nodes(self, node: PlanNode) -> int:
        """Count total number of nodes in plan tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        if not self.samples:
            return
        
        # Plan size statistics
        plan_sizes = [sample.metadata['plan_size'] for sample in self.samples]
        self.statistics['plan_size'] = {
            'min': min(plan_sizes),
            'max': max(plan_sizes), 
            'mean': np.mean(plan_sizes),
            'std': np.std(plan_sizes)
        }
        
        # Target statistics
        for target_col in self.target_columns:
            values = [sample.targets[target_col] for sample in self.samples 
                     if target_col in sample.targets]
            
            if values:
                self.statistics[target_col] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        # Update stats alias
        self.stats = self.statistics
        logger.info(f"Dataset statistics: {self.statistics}")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> PlanSample:
        """Get sample by index."""
        return self.samples[idx]
    
    def get_targets(self, target_column: str) -> np.ndarray:
        """Get all target values for a specific column."""
        values = []
        for sample in self.samples:
            if target_column in sample.targets:
                values.append(sample.targets[target_column])
        return np.array(values)
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple['PlanDataset', 'PlanDataset', 'PlanDataset']:
        """Split dataset into train/validation/test sets."""
        n_samples = len(self.samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle samples
        indices = np.random.permutation(n_samples)
        
        # Create splits
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create new dataset instances
        train_dataset = self._create_subset(train_indices, "train")
        val_dataset = self._create_subset(val_indices, "validation")
        test_dataset = self._create_subset(test_indices, "test")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, indices: np.ndarray, split_name: str) -> 'PlanDataset':
        """Create a subset dataset with given indices."""
        # Create new instance with same configuration
        subset = PlanDataset.__new__(PlanDataset)
        subset.csv_path = self.csv_path
        subset.target_columns = self.target_columns
        subset.preprocessor = self.preprocessor
        subset.statistics = {}
        subset.stats = {}
        
        # Copy selected samples
        subset.samples = [self.samples[i] for i in indices]
        
        # Recompute statistics for subset
        subset._compute_statistics()
        subset.stats = subset.statistics  # Update alias
        
        logger.info(f"Created {split_name} subset with {len(subset)} samples")
        return subset


def create_plan_dataset(csv_path: Union[str, Path],
                       target_columns: Optional[List[str]] = None,
                       **kwargs) -> PlanDataset:
    """Factory function to create a PlanDataset.
    
    Args:
        csv_path: Path to CSV file
        target_columns: Target columns to extract
        **kwargs: Additional arguments passed to PlanDataset
    
    Returns:
        PlanDataset instance
    """
    return PlanDataset(csv_path=csv_path, target_columns=target_columns, **kwargs)
