"""Plan-selection folds (exp/exp3_e2e/run_kfold.py, unchanged) with the TCR inputs taken apart as in the root-cost
controls of the dependency study: ``rootcost`` = type + rows + the root's cumulative cost broadcast to every operator;
``localcost`` = rootcost + the operator's cost residual; ``resid`` = type + rows + the residual (TLR, no root cost).
Use with --features dace.

    python analysis/attn/run_kfold_cost.py --cost rootcost <the arguments of run_kfold.py>
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp3_e2e"), os.path.join(ROOT, "analysis", "attn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from derived import DerivedFeaturizer  # noqa: E402
import run_kfold as rk  # noqa: E402


def main() -> None:
    if "--cost" not in sys.argv:
        raise SystemExit("usage: run_kfold_cost.py --cost rootcost|localcost|resid <run_kfold arguments>")
    i = sys.argv.index("--cost")
    cost = sys.argv[i + 1]
    assert cost in ("rootcost", "localcost", "resid", "residslog"), cost
    sys.argv = [sys.argv[0]] + sys.argv[1:i] + sys.argv[i + 2:]
    original = rk.make_featurizer

    def make_featurizer(stats, a, train_plans=None):
        fz = original(stats, a, train_plans)
        if a.features != "dace" or getattr(a, "drop_features", ""):
            raise SystemExit("the cost controls are defined for the plain TCR (dace) inputs")
        plans = [p if "Plan" in p else {"Plan": p} for p in (train_plans or [])]
        return DerivedFeaturizer(fz, cost, plans)

    rk.make_featurizer = make_featurizer
    rk.main()


if __name__ == "__main__":
    main()
