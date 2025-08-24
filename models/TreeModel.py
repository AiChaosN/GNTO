"""Simple tree aggregation model.

The original project is expected to use complex neural network structures that
operate on trees.  Here we provide a tiny, differentiable-free implementation
that aggregates vectors produced by :class:`Encoder` into a single vector using a
configurable reduction operation.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


class TreeModel:
    """Aggregate encoded node vectors into a single representation."""

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        """Reduce ``vectors`` into a single vector.

        Parameters
        ----------
        vectors:
            Iterable of numpy arrays representing encoded plan nodes.
        """

        stacked = self._pad_and_stack(list(vectors))
        if self.reduction == "mean":
            return stacked.mean(axis=0)
        return stacked.sum(axis=0)

    # ------------------------------------------------------------------ helpers --
    def _pad_and_stack(self, vecs: Iterable[np.ndarray]) -> np.ndarray:
        vecs = list(vecs)
        if not vecs:
            return np.zeros(0)
        max_len = max(len(v) for v in vecs)
        stacked = np.zeros((len(vecs), max_len))
        for i, v in enumerate(vecs):
            stacked[i, : len(v)] = v
        return stacked
