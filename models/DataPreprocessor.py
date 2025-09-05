"""Utilities for turning raw query plans into structured objects.

The real project uses database specific JSON plans.  For the purpose of this
skeleton repository we provide a tiny, generic representation so that other
modules can operate on a predictable structure.  The :class:`DataPreprocessor`
turns nested dictionaries that resemble PostgreSQL's ``EXPLAIN`` output into
:class:`PlanNode` trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import numpy as np


@dataclass
class PlanNode:
    node_type: str
    children: List["PlanNode"] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)
    node_vector: Optional[np.ndarray] = None

    def getExtraInfo(self) -> Dict[str, Any]:
        return self.extra_info


class DataPreprocessor:
    """Convert raw JSON style plans into :class:`PlanNode` trees."""

    plan_key: str = "Plans"  # key that contains child nodes in the raw dict

    def preprocess(self, plan: Dict[str, Any]) -> PlanNode:
        """Recursively convert ``plan`` into a :class:`PlanNode` tree.

        Parameters
        ----------
        plan:
            JSON-like dictionary describing a plan node.  Nested plans are
            expected under ``self.plan_key``.
        """

        node = PlanNode(
            node_type=plan.get("Node Type", "Unknown"),
            extra_info={k: v for k, v in plan.items() if k != self.plan_key},
        )

        for child in plan.get(self.plan_key, []):
            node.children.append(self.preprocess(child))

        return node

    # Small convenience wrappers -------------------------------------------------
    def preprocess_all(self, plans: Iterable[Dict[str, Any]]) -> List[PlanNode]:
        """Preprocess a list of raw plans."""

        return [self.preprocess(p) for p in plans]
    
    # 下面是树可视化函数 两个函数 display_tree 和 get_tree_stats ------------------------------------------------
    def display_tree(self, node: PlanNode, show_details: bool = True, max_depth: int = None) -> str:
        lines = []
        self._build_tree_lines(node, lines, "", True, show_details, 0, max_depth)
        return "\n".join(lines)
    
    def print_tree(self, node: PlanNode, show_details: bool = True, max_depth: int = None):
        print(self.display_tree(node, show_details, max_depth))
    
    def get_tree_stats(self, node: PlanNode) -> Dict[str, Any]:
        stats = {
            'total_nodes': 0,
            'max_depth': 0,
            'node_types': {},
            'nodes_by_depth': {},
            'leaf_nodes': 0,
            'internal_nodes': 0
        }
        
        self._collect_tree_stats(node, stats, 0)
        
        return stats
    
    def _build_tree_lines(self, node: PlanNode, lines: List[str], prefix: str, 
                         is_last: bool, show_details: bool, current_depth: int, max_depth: int):
        
        # Check depth limit
        if max_depth is not None and current_depth > max_depth:
            return
        
        # Create the tree structure symbols
        connector = "└── " if is_last else "├── "
        
        # Build the main line
        main_line = f"{prefix}{connector}{node.node_type}"
        
        # Add details if requested
        if show_details and node.extra_info:
            details = []
            
            # Show important cost/performance info
            for key in ['Total Cost', 'Startup Cost', 'Plan Rows', 'Plan Width', 
                       'Actual Total Time', 'Actual Rows']:
                if key in node.extra_info:
                    value = node.extra_info[key]
                    if isinstance(value, float):
                        details.append(f"{key}: {value:.2f}")
                    else:
                        details.append(f"{key}: {value}")
            
            # Show relation/index names
            for key in ['Relation Name', 'Alias', 'Index Name']:
                if key in node.extra_info:
                    details.append(f"{key}: {node.extra_info[key]}")
            
            # Show join type
            if 'Join Type' in node.extra_info:
                details.append(f"Join Type: {node.extra_info['Join Type']}")
            
            if details:
                main_line += f" ({', '.join(details[:])})"
                # main_line += f", node_vector: {node.node_vector}"
                if node.node_vector is not None:
                    main_line += f", node_vector_shape: {node.node_vector.shape}"
                # if len(details) > 3:
                #     main_line += f" [+{len(details)-3} more]"
        
        lines.append(main_line)
        
        # Prepare prefix for children
        if is_last:
            child_prefix = prefix + "    "
        else:
            child_prefix = prefix + "│   "
        
        # Process children
        if node.children:
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._build_tree_lines(child, lines, child_prefix, is_last_child, 
                                     show_details, current_depth + 1, max_depth)
    
    def _collect_tree_stats(self, node: PlanNode, stats: Dict[str, Any], depth: int):
        """Recursively collect statistics about the tree."""
        
        # Update total nodes
        stats['total_nodes'] += 1
        
        # Update max depth
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        # Update node types count
        node_type = node.node_type
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Update nodes by depth
        stats['nodes_by_depth'][depth] = stats['nodes_by_depth'].get(depth, 0) + 1
        
        # Check if leaf or internal node
        if node.children:
            stats['internal_nodes'] += 1
            # Process children
            for child in node.children:
                self._collect_tree_stats(child, stats, depth + 1)
        else:
            stats['leaf_nodes'] += 1