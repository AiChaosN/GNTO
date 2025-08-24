"""Encode :class:`~models.DataPreprocessor.PlanNode` objects into vectors.

The real project uses sophisticated graph neural networks.  Here we simply
create a bag-of-node-types representation so that downstream components have a
numeric vector to operate on.
"""

from __future__ import annotations

from typing import Dict, Iterable, List
import numpy as np


class Encoder:
    """Naive encoder that assigns each node type a one-hot vector."""

    def __init__(self) -> None:
        self.node_index: Dict[str, int] = {}

    # ------------------------------------------------------------------ utilities
    def _ensure_index(self, node_type: str) -> int:
        if node_type not in self.node_index:
            self.node_index[node_type] = len(self.node_index)
        return self.node_index[node_type]

    def _one_hot(self, idx: int) -> np.ndarray:
        vec = np.zeros(len(self.node_index), dtype=float)
        vec[idx] = 1.0
        return vec

    # --------------------------------------------------------------------- encoder
    def encode(self, node) -> np.ndarray:
        """Encode a single :class:`~models.DataPreprocessor.PlanNode`.

        ``node`` is expected to expose ``node_type`` and ``children`` attributes
        like the :class:`PlanNode` dataclass defined in :mod:`DataPreprocessor`.
        The returned array has length equal to the number of unique node types
        encountered so far.
        """

        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        vec = self._one_hot(idx)
        if getattr(node, "children", None):
            child_vecs = [self.encode(ch) for ch in node.children]
            vec = self._pad_and_sum([vec] + child_vecs)
        return vec

    def encode_all(self, nodes: Iterable) -> List[np.ndarray]:
        """Encode a list of nodes into vectors."""

        return [self.encode(n) for n in nodes]

    # ---------------------------------------------------------------- internal ---
    def _pad_and_sum(self, vecs: List[np.ndarray]) -> np.ndarray:
        """Pad vectors to equal length and return their sum."""

        max_len = max(len(v) for v in vecs)
        padded = np.zeros((len(vecs), max_len))
        for i, v in enumerate(vecs):
            padded[i, : len(v)] = v
        return padded.sum(axis=0)
