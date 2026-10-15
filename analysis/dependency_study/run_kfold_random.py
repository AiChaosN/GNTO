"""Plan-selection folds with a random tree in place of the plan topology (random-tree control on JOB / TPC-H).

Runs exp/exp3_e2e/run_kfold.py unchanged except for one function: every candidate plan keeps its operators and
features, but its edges are replaced by a random tree over the same operators (both directions, fixed per
query and hint configuration, identical in training and testing).  Use with graph.conv=gcn or gatv2 only.

    python analysis/dependency_study/run_kfold_random.py <the arguments of run_kfold.py>
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))

import torch                                          # noqa: E402

import run_kfold as rk                                # noqa: E402


def random_tree(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Both directions of a uniformly random recursive tree over ``n`` nodes (same construction as
    run_crossdb_dep.random_tree; duplicated because the two experiments have different ``common`` modules)."""
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    perm = rng.permutation(n)
    edges = []
    for k in range(1, n):
        parent, child = int(perm[rng.integers(0, k)]), int(perm[k])
        edges += [[parent, child], [child, parent]]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

_original = rk.to_graph


def to_graph(plan, fz, y_norm, qid, hint, ms, censored=False, parent_to_child=False):
    if parent_to_child:
        raise SystemExit("the random-tree control is defined for gcn / gatv2, not for tree convolution")
    d = _original(plan, fz, y_norm, qid, hint, ms, censored, parent_to_child=False)
    seed = [20261003, int(qid) % (2 ** 31), zlib.crc32(str(hint).encode())]
    d.edge_index = random_tree(d.x.size(0), np.random.default_rng(seed))
    return d


rk.to_graph = to_graph

if __name__ == "__main__":
    rk.main()
