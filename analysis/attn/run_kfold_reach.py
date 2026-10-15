"""Plan-selection folds (exp/exp3_e2e/run_kfold.py, unchanged) with the edges of every candidate plan replaced by the
edge set of the requested reach (analysis/attn/reach.py). Use with --set graph.conv=transformer.

    python analysis/attn/run_kfold_reach.py --reach subplan <the arguments of run_kfold.py>
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp3_e2e"), os.path.join(ROOT, "analysis", "attn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from reach import REACHES, reach_edges  # noqa: E402
import run_kfold as rk  # noqa: E402


def main() -> None:
    if "--reach" not in sys.argv:
        raise SystemExit("usage: run_kfold_reach.py --reach local|subplan|global <run_kfold arguments>")
    i = sys.argv.index("--reach")
    reach = sys.argv[i + 1]
    assert reach in REACHES, reach
    sys.argv = [sys.argv[0]] + sys.argv[1:i] + sys.argv[i + 2:]
    original = rk.to_graph

    def to_graph(plan, fz, y_norm, qid, hint, ms, censored=False, parent_to_child=False):
        if parent_to_child:
            raise SystemExit("the reach control is defined for attention convs, not for tree convolution")
        d = original(plan, fz, y_norm, qid, hint, ms, censored, parent_to_child=False)
        d.edge_index = reach_edges(d.edge_index, d.x.size(0), reach)
        return d

    rk.to_graph = to_graph
    rk.main()


if __name__ == "__main__":
    main()
