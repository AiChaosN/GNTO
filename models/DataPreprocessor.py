"""Utilities for turning raw query plans into structured objects.

The real project uses database specific JSON plans.  For the purpose of this
skeleton repository we provide a tiny, generic representation so that other
modules can operate on a predictable structure.  The :class:`DataPreprocessor`
turns nested dictionaries that resemble PostgreSQL's ``EXPLAIN`` output into
:class:`PlanNode` trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class PlanNode:
    """A minimal tree node used throughout the modelling pipeline.

    Parameters
    ----------
    node_type:
        Name of the plan node, e.g. ``Seq Scan`` or ``Hash Join``.
    children:
        Child nodes in the execution plan tree.
    extra_info:
        Additional key/value pairs describing the node.  The preprocessor keeps
        them untouched so that downstream components can decide what to use.
    """

    node_type: str
    children: List["PlanNode"] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)


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
