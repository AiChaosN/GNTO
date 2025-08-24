"""Helper functions used across the GNTO models."""

from __future__ import annotations

import random
from typing import Iterable, List
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for ``random`` and ``numpy``.

    Deterministic behaviour is convenient when experimenting with different
    model components.  This helper mirrors the behaviour of the real project in a
    simplified form.
    """

    random.seed(seed)
    np.random.seed(seed)


def flatten(nested: Iterable[Iterable]) -> List:
    """Flatten a nested iterable into a list."""

    out: List = []
    for inner in nested:
        out.extend(inner)
    return out
