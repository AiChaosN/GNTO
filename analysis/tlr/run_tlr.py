"""GNTO and DACE on the 20-database benchmark with decomposed cost inputs (TLR).

TLR gives every operator its type, its own non-cumulative cost (Total Cost minus the Total Cost of its inputs), the
cumulative cost of the plan's root and its Plan Rows: [types | local cost | root cost | rows].  TCR, the default
inputs of exp2, gives [types | cumulative Total Cost | rows].  The root-cost controls of the dependency study
(analysis/dependency_study/run_crossdb_dep.py --features rootcost|localcost) use the same decomposition.

This wrapper imports the frozen exp2 runners unchanged and only swaps the featurizer that load_benchmark returns, so
split, statistics (fitted on the training plans only), training loop, selection rule and outputs are those of exp2.

    python analysis/tlr/run_tlr.py --model gnto --split paper --statistics train --seed 42 --epochs 100 --select last \
        --device cpu --out results/analysis/tlr_20261008/gnto_paper1_s42
    python analysis/tlr/run_tlr.py --model dace --split rand --split-seed 1 ... --out results/analysis/tlr_20261008/dace_rand1_s42
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp2_dace"), os.path.join(ROOT, "exp"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as c  # noqa: E402  (exp2_dace/common.py)


def _walk(node):
    yield node
    for ch in node.get("Plans", []):
        yield from _walk(ch)


def local_cost(node) -> float:
    return float(node.get("Total Cost", 0.0)) - sum(float(ch.get("Total Cost", 0.0)) for ch in node.get("Plans", []))


class TLRFeaturizer:
    """[types | local cost | root cost | Plan Rows]; the root cost of a node is looked up by the identity of the plan
    node object, which the frozen runners traverse unchanged. Scales: root cost and rows use the Total Cost / Plan Rows
    scalers fitted on the training plans; the local cost its own median / inter-quartile-range scaler fitted on the
    training nodes."""

    def __init__(self, fz, plans_by_db, train_plans, with_root: bool = True):
        self.fz = fz
        self.with_root = with_root                      # False = "resid": [types | local cost | rows], no root cost
        self.num_types = fz.num_types
        self.type_index = fz.type_index
        self.node_length = fz.num_types + (3 if with_root else 2)
        self.root_cost = {}
        for plans in plans_by_db.values():
            for p in plans:
                root = p["Plan"]
                rc = float(root.get("Total Cost", 0.0))
                for n in _walk(root):
                    self.root_cost[id(n)] = rc
        vals = np.array([local_cost(n) for p in train_plans for n in _walk(p["Plan"])])
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        self.local_center, self.local_scale = float(q50), float(q75 - q25) or 1.0
        self.stats = dict(fz.stats, **{"Local Cost": {"center": self.local_center, "scale": self.local_scale}})

    def node_vector(self, node):
        vec = np.zeros(self.node_length, dtype=np.float32)
        idx = self.type_index.get(node.get("Node Type", ""), None)
        if idx is not None and idx < self.num_types:
            vec[idx] = 1.0
        vec[self.num_types] = (local_cost(node) - self.local_center) / self.local_scale
        if self.with_root:
            vec[self.num_types + 1] = self.fz.scale("Total Cost", self.root_cost.get(id(node), node.get("Total Cost", 0.0)))
            vec[self.num_types + 2] = self.fz.scale("Plan Rows", node.get("Plan Rows", 0.0))
        else:
            vec[self.num_types + 1] = self.fz.scale("Plan Rows", node.get("Plan Rows", 0.0))
        return vec


def main() -> None:
    if "--model" not in sys.argv:
        raise SystemExit("usage: run_tlr.py --model gnto|dace <exp2 runner arguments>")
    i = sys.argv.index("--model")
    model = sys.argv[i + 1]
    argv = [sys.argv[0]] + sys.argv[1:i] + sys.argv[i + 2:]
    inputs = "tlr"
    if "--inputs" in argv:                                     # tlr (default) | resid = local cost without the root cost
        j = argv.index("--inputs"); inputs = argv[j + 1]; argv = argv[:j] + argv[j + 2:]
    assert inputs in ("tlr", "resid"), inputs
    with_root = inputs == "tlr"
    subset = "--subset-zeroshot" in argv                       # the zero-shot model's training plans (as in analysis/baselines/run_gnto_subset.py)
    if subset:
        argv.remove("--subset-zeroshot")
        split_name = argv[argv.index("--split") + 1] if "--split" in argv else "paper"
        split_seed = int(argv[argv.index("--split-seed") + 1]) if "--split-seed" in argv else 1
        train_ids, _, _ = c.make_split(split_name, split_seed)
        train_dbs = {c.WORKLOADS[k] for k in train_ids}
        trimmed = {db: n for db, n in {"imdb_full": 5000, "tpc_h": 5000}.items() if db in train_dbs}
        original_load_plans = c.load_plans

        def load_plans_subset(path):
            plans = original_load_plans(path)
            db = os.path.basename(path).replace("_filted.json", "")
            return plans[:trimmed[db]] if db in trimmed else plans

        c.load_plans = load_plans_subset
        if "--out" in argv:
            out_dir = argv[argv.index("--out") + 1]
            os.makedirs(out_dir, exist_ok=True)
            json.dump({"training_plans": "zero-shot subset", "trimmed_to_first_n": trimmed},
                      open(os.path.join(out_dir, "subset.json"), "w"), indent=2)
    if "--no-cache" not in argv:
        argv.append("--no-cache")                              # the exp2 cache keys on the TCR features
    original = c.load_benchmark

    def load_benchmark_tlr(a, log):
        bench = original(a, log)
        bench["fz"] = TLRFeaturizer(bench["fz"], bench["plans"], [bench["plans"][db][i] for db, i in bench["train_index"]],
                                    with_root=with_root)
        bench["stats"] = bench["fz"].stats
        bench["cache_id"] = bench["cache_id"] + "_" + inputs
        channels = ["type one-hot", "local cost", "root cost", "rows"] if with_root else ["type one-hot", "local cost", "rows"]
        log(f"{inputs.upper()} inputs: [{' | '.join(channels)}], node_length {bench['fz'].node_length}; "
            f"local-cost scaler center {bench['fz'].local_center:.3f} scale {bench['fz'].local_scale:.3f}")
        out = a.out if getattr(a, "out", None) else None
        if out:
            os.makedirs(out, exist_ok=True)
            json.dump({"inputs": inputs.upper(), "channels": channels,
                       "local_cost_scaler": {"center": bench["fz"].local_center, "scale": bench["fz"].local_scale}},
                      open(os.path.join(out, "tlr.json"), "w"), indent=2)
        return bench

    if model == "gnto":
        import run_gnto as runner
    elif model == "dace":
        import run_dace as runner
    else:
        raise SystemExit(f"unknown model {model}")
    runner.load_benchmark = load_benchmark_tlr
    sys.argv = argv
    runner.main()


if __name__ == "__main__":
    main()
