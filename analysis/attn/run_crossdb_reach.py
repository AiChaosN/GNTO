"""Cross-database runs of the reach control: run_crossdb_dep.py unchanged, with the edges of every plan replaced by
the edge set of the requested reach (analysis/attn/reach.py). Use with --conv transformer.

    python analysis/attn/run_crossdb_reach.py --reach subplan --conv transformer --features nocost --seed 42 \
        --split paper --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2 \
        --out results/analysis/dependency_20261003/crossdb/nocost_trsubplan_L3_true_s42
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "analysis", "dependency_study"), os.path.join(ROOT, "analysis", "attn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from reach import REACHES, reach_edges  # noqa: E402
import run_crossdb_dep as rcd  # noqa: E402


def main() -> None:
    if "--reach" not in sys.argv:
        raise SystemExit("usage: run_crossdb_reach.py --reach local|subplan|global <run_crossdb_dep arguments>")
    i = sys.argv.index("--reach")
    reach = sys.argv[i + 1]
    assert reach in REACHES, reach
    sys.argv = [sys.argv[0]] + sys.argv[1:i] + sys.argv[i + 2:]
    original = rcd.plan_to_graph

    def plan_to_graph_reach(root, fz):
        x, ei = original(root, fz)
        return x, reach_edges(ei, x.size(0), reach)

    rcd.plan_to_graph = plan_to_graph_reach
    rcd.main()


if __name__ == "__main__":
    main()
