"""Plan-selection folds (exp/exp3_e2e/run_kfold.py, unchanged) with the PG-stat inputs plus the operator's own cost.

PG-stat already carries PostgreSQL's cumulative startup and total cost of every operator. This wrapper appends one
channel, the operator's local cost (its Total Cost minus the Total Cost of its inputs), signed-log transformed and
scaled by the median / inter-quartile range of the training nodes, by substituting the featurizer class that
run_kfold.make_featurizer instantiates. Everything else (folds, candidates, objective, epochs) is run_kfold's.

    python analysis/tlr/run_kfold_local.py --features pg <the other arguments of run_kfold.py>
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp3_e2e"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapters import pg_features as pf  # noqa: E402


def local_cost(node) -> float:
    return float(node.get("Total Cost", 0.0)) - sum(float(ch.get("Total Cost", 0.0)) for ch in node.get("Plans", []))


def slog(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


class PGFeaturizerLocal(pf.PGFeaturizer):
    """PG-stat + the operator's own cost."""

    def __init__(self, catalog):
        super().__init__(catalog)
        self.local_center, self.local_scale = 0.0, 1.0

    @property
    def node_length(self) -> int:
        return super().node_length + 1

    def fit(self, plans):
        plans = list(plans)
        super().fit(plans)
        vals = np.array([slog(local_cost(n)) for p in plans for n in pf._walk(p["Plan"] if "Plan" in p else p)])
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        self.local_center = float(q50)
        self.local_scale = float(q75 - q25) or (float(vals.std()) or 1.0)
        return self

    def node_vector(self, node):
        base = super().node_vector(node)
        extra = np.array([(slog(local_cost(node)) - self.local_center) / self.local_scale], dtype=np.float32)
        return np.concatenate([base, extra])


def main() -> None:
    pf.PGFeaturizer = PGFeaturizerLocal          # run_kfold imports the class from the module at call time
    import run_kfold as rk
    rk.main()


if __name__ == "__main__":
    main()
