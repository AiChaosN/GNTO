"""Prediction heads for GNTO.

``PredictionHead`` is intentionally lightweight. It applies a linear model to the
vector produced by :class:`TreeEncoder`. The weights can either be provided
explicitly or are initialised to ones. The class stores the weights so it can
be reused for multiple predictions.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


class PredictionHead:
    """A minimal linear prediction head."""

    def __init__(self, weights: Iterable[float] | None = None) -> None:
        self.weights = np.array(list(weights), dtype=float) if weights is not None else None

    def predict(self, features: np.ndarray) -> float:
        """Return a scalar prediction for ``features``."""

        if self.weights is None:
            self.weights = np.ones_like(features, dtype=float)
        if len(self.weights) < len(features):
            self.weights = np.pad(self.weights, (0, len(features) - len(self.weights)))
        return float(np.dot(self.weights[: len(features)], features))
