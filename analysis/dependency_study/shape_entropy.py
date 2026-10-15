"""How much of a plan's tree shape is left undetermined by its multiset of operators?

A model without message passing (None) sees a plan as a multiset of operators.  Relating operators can only add
information if plans with the same multiset differ in shape.  For each benchmark this script computes

    H(shape)            entropy of the tree shape (operator types + topology, children unordered), in bits
    H(shape | bag)      what remains once the multiset of operator types is known
    unique              share of plans whose multiset occurs with exactly one shape in the benchmark
    modal               share of plans whose shape is the most frequent one for their multiset

It uses operator types only, the weakest description of the multiset (row estimates would shrink the residual
further), so H(shape | bag) is an upper bound on what the topology adds.

    python analysis/dependency_study/shape_entropy.py     # -> results/analysis/dependency_20261003/tables/shape_entropy.csv
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.dace import WORKLOADS, load_plans, plans_path  # noqa: E402


def shape(node) -> str:
    return "(" + node.get("Node Type", "") + "".join(sorted(shape(ch) for ch in node.get("Plans", []))) + ")"


def bag(node, acc):
    acc.append(node.get("Node Type", ""))
    for ch in node.get("Plans", []):
        bag(ch, acc)
    return acc


def stats(name, roots):
    n = len(roots)
    shapes = [shape(r) for r in roots]
    bags = [tuple(sorted(bag(r, []))) for r in roots]
    by = collections.defaultdict(collections.Counter)
    for b, s in zip(bags, shapes):
        by[b][s] += 1
    h = -sum(c / n * math.log2(c / n) for c in collections.Counter(shapes).values())
    hc = sum(sum(cnt.values()) / n * -sum(c / sum(cnt.values()) * math.log2(c / sum(cnt.values())) for c in cnt.values())
             for cnt in by.values())
    return dict(benchmark=name, plans=n, bags=len(by), shapes=len(set(shapes)), h_shape=h, h_shape_given_bag=hc,
                explained_pct=100 * (1 - hc / h) if h else float("nan"),
                unique_pct=100 * sum(sum(c.values()) for c in by.values() if len(c) == 1) / n,
                modal_pct=100 * sum(max(c.values()) for c in by.values()) / n)


def main() -> None:
    rows = []
    corpus = [json.loads(s)["Plan"] for part in range(20)
              for s in pd.read_csv(os.path.join(ROOT, "data", f"train_plan_part{part}.csv"))["json"]]
    rows.append(stats("IMDB corpus", corpus))
    rows.append(stats("20 databases", [p["Plan"] for db in WORKLOADS for p in load_plans(plans_path(db))]))
    for label, wl in (("TPC-H", "tpch52"), ("JOB", "job112")):
        roots = []
        for path in sorted(glob.glob(os.path.join(ROOT, "data", "e2e", f"hint_plans_v2_{wl}", "q_*.json"))):
            seen = set()
            for r in json.load(open(path))["results"]:
                if r["fingerprint"] not in seen:
                    seen.add(r["fingerprint"])
                    roots.append(r["plan"].get("Plan", r["plan"]))
        rows.append(stats(label, roots))
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "analysis", "dependency_20261003", "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "shape_entropy.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
