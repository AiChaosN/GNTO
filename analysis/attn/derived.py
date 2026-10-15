"""Root-cost controls for the TCR inputs, duplicated from analysis/dependency_study/run_crossdb_dep.py because the
two experiments import different ``common`` modules (exp2 vs exp3) and cannot share an import."""
from __future__ import annotations

import numpy as np


def _walk(node):
    yield node
    for ch in node.get("Plans", []):
        yield from _walk(ch)


def slog(x: float) -> float:
    """Signed log1p, the scaling PG-stat applies to sizes and counts."""
    return float(np.sign(x) * np.log1p(abs(x)))


def local_cost(node) -> float:
    """The operator's cost residual: its cumulative Total Cost minus the cumulative costs of its inputs."""
    return float(node.get("Total Cost", 0.0)) - sum(float(ch.get("Total Cost", 0.0)) for ch in node.get("Plans", []))


class DerivedFeaturizer:
    """``rootcost``: operator type + Plan Rows + the root's cumulative cost broadcast to every operator.
    ``localcost``: additionally the operator's cost residual; ``resid``: type + rows + the residual alone (TLR). Scales: the root cost uses the Total Cost scaler of the
    training statistics, the residual a median / inter-quartile-range scaler fitted on the training nodes."""

    def __init__(self, fz, features: str, train_plans):
        assert features in ("rootcost", "localcost", "resid", "residslog")
        self.fz, self.features = fz, features
        self.num_types = fz.num_types
        self.keep = [i for i in range(fz.node_length) if i != fz.num_types]          # types + Plan Rows
        if features in ("localcost", "resid", "residslog"):    # resid: type + rows + the residual, no root cost
            vals = [local_cost(n) for p in train_plans for n in _walk(p["Plan"])]
            if features == "residslog":                        # residslog: the residual signed-log scaled, as PG-stat scales sizes
                vals = [slog(v) for v in vals]
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            self.local_center, self.local_scale = float(q50), float(q75 - q25) or 1.0
        self.node_length = len(self.keep) + (2 if features == "localcost" else 1)       # rootcost / resid: one extra channel
        self._root = 0.0

    def bind(self, root) -> None:
        self._root = self.fz.scale("Total Cost", root.get("Total Cost", 0.0))

    def node_vector(self, node):
        base = self.fz.node_vector(node)[self.keep]
        if self.features == "rootcost":
            extra = [self._root]
        elif self.features == "resid":
            extra = [(local_cost(node) - self.local_center) / self.local_scale]
        elif self.features == "residslog":
            extra = [(slog(local_cost(node)) - self.local_center) / self.local_scale]
        else:
            extra = [(local_cost(node) - self.local_center) / self.local_scale, self._root]
        return np.concatenate([base, np.asarray(extra, dtype=np.float32)])
