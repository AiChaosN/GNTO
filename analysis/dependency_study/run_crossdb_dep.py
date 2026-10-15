"""Dependency study on the cross-database benchmark: how much of the accuracy comes from
relating plan operators to each other, and which relations matter.

This script lives outside the source-frozen ``models/adapters/exp/scripts`` trees. It imports
the frozen experiment code unchanged (data split, train-only statistics, training loop, metrics)
and varies only three things:

* ``--features``  which per-node inputs are visible
      full      operator type + Total Cost + Plan Rows   (the protocol of exp2)
      nocost    operator type + Plan Rows                (no optimizer cost: each node sees only its own output estimate)
      norows    operator type + Total Cost
      typeonly  operator type
      costrows  Total Cost + Plan Rows, no operator type
      costonly  Total Cost alone;  rowsonly  Plan Rows alone
      resid     type + Plan Rows + the operator's cost residual (no cumulative and no root cost);  residrows  Plan Rows + residual
* ``--conv`` / ``--layers`` / ``--virtual-node``  how operators are related (none = not at all)
* ``--topology``  which edges the encoder sees
      true      parent <-> child edges of the plan
      random    a random tree over the same operators, in training and in testing

For every model trained on the true topology the test plans are additionally scored with
(a) a random tree and (b) no edges; the drop measures how much the trained model relies on the
plan structure.  Outputs follow the exp2 contract under ``--out``.

    python analysis/dependency_study/run_crossdb_dep.py --features nocost --conv gatv2 --seed 42 \
        --out results/analysis/dependency_20261003/crossdb/nocost_gatv2_L3_true_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp2_dace"), os.path.join(ROOT, "exp"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as c                                             # noqa: E402  (exp2_dace/common.py)
import run_gnto as rg                                          # noqa: E402  (frozen training loop)
from adapters.dace_adapter import plan_to_graph               # noqa: E402
from models.dace import count_nodes                           # noqa: E402
from models.gnto import GNTO, get_preset                      # noqa: E402

FEATURES = ("full", "nocost", "norows", "typeonly", "costrows", "costonly", "rowsonly", "rootcost", "localcost", "rootcum", "localcostshuf", "resid", "residrows", "residonly", "residtype")
TOPOLOGIES = ("true", "random")


def local_cost(node) -> float:
    """The operator's own cost: its cumulative Total Cost minus the cumulative costs of its inputs."""
    return float(node.get("Total Cost", 0.0)) - sum(float(ch.get("Total Cost", 0.0)) for ch in node.get("Plans", []))


class DerivedFeaturizer:
    """Root-cost controls. ``rootcost``: operator type + Plan Rows + the root's cumulative cost, the same
    value broadcast to every operator. ``localcost``: additionally each operator's own non-cumulative cost. Both carry
    the whole-plan cost estimate; neither hands an operator the summary of its sub-plan that the cumulative Total Cost
    of ``full`` is. Scales: the root cost uses the Total Cost scaler of the training statistics, the local cost a
    median / inter-quartile-range scaler fitted on the training nodes. ``rootcum``:
    the TCR inputs of ``full`` (type + cumulative Total Cost + Plan Rows) plus the root's cumulative cost, so that the
    residual of ``localcost`` is compared with the cumulative cost at equal information about the whole plan.
    ``localcostshuf``: the residuals of a plan permuted among its operators, so that
    the distribution of residuals within the plan is kept and their association with the operators is broken."""

    def __init__(self, fz, features: str, train_plans):
        assert features in ("rootcost", "localcost", "rootcum", "localcostshuf", "resid", "residrows", "residonly", "residtype")
        self.fz, self.features = fz, features
        self.num_types = fz.num_types
        if features == "rootcum":
            self.keep = list(range(fz.node_length))                                   # types + Total Cost + Plan Rows
        elif features == "residrows":
            self.keep = [fz.num_types + 1]                                            # Plan Rows only
        elif features == "residonly":
            self.keep = []                                                            # the residual alone
        elif features == "residtype":
            self.keep = list(range(fz.num_types))                                     # types + residual
        else:
            self.keep = [i for i in range(fz.node_length) if i != fz.num_types]      # types + Plan Rows
        if features in ("localcost", "localcostshuf", "resid", "residrows", "residonly", "residtype"):
            vals = [local_cost(n) for p in train_plans for n in _walk(p["Plan"])]
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            self.local_center, self.local_scale = float(q50), float(q75 - q25) or 1.0
        # resid / residrows: the residual replaces the cumulative cost, no root cost at all
        self.node_length = len(self.keep) + (2 if features in ("localcost", "localcostshuf") else 1)
        self._root = 0.0
        self._shuf, self._plan_counter = {}, 0

    def bind(self, root) -> None:
        self._root = self.fz.scale("Total Cost", root.get("Total Cost", 0.0))
        if self.features == "localcostshuf":
            nodes = list(_walk(root))
            vals = [local_cost(n) for n in nodes]
            perm = np.random.default_rng([20261010, self._plan_counter]).permutation(len(nodes))
            self._plan_counter += 1
            self._shuf = {id(n): float(vals[j]) for n, j in zip(nodes, perm)}

    def node_vector(self, node):
        base = self.fz.node_vector(node)[self.keep]
        if self.features in ("rootcost", "rootcum"):
            extra = [self._root]
        elif self.features == "localcost":
            extra = [(local_cost(node) - self.local_center) / self.local_scale, self._root]
        elif self.features in ("resid", "residrows", "residonly", "residtype"):
            extra = [(local_cost(node) - self.local_center) / self.local_scale]
        else:
            extra = [(self._shuf[id(node)] - self.local_center) / self.local_scale, self._root]
        return np.concatenate([base, np.asarray(extra, dtype=np.float32)])


class WithPlanSize:
    """Appends the plan's size, log(1 + number of operators) standardized on the training plans, to every node vector
    (the residuals of a plan sum to the root's cost, so their mean is the root cost over n; giving every configuration n
    controls for the plan size)."""

    def __init__(self, inner, train_plans):
        self.inner, self.num_types, self.keep = inner, inner.num_types, inner.keep
        ns = np.log1p([sum(1 for _ in _walk(p["Plan"])) for p in train_plans])
        self.mu, self.sd = float(ns.mean()), float(ns.std()) or 1.0
        self.node_length = inner.node_length + 1
        self._z = 0.0

    def bind(self, root) -> None:
        if hasattr(self.inner, "bind"):
            self.inner.bind(root)
        self._z = (np.log1p(sum(1 for _ in _walk(root))) - self.mu) / self.sd

    def node_vector(self, node):
        return np.concatenate([self.inner.node_vector(node), np.asarray([self._z], dtype=np.float32)])


class SubsetFeaturizer:
    """``NodeFeaturizer`` restricted to some channels ([types | Total Cost | Plan Rows])."""

    def __init__(self, fz, features: str):
        drop = {"full": [], "nocost": [fz.num_types], "norows": [fz.num_types + 1],
                "typeonly": [fz.num_types, fz.num_types + 1],
                "costrows": list(range(fz.num_types)),                     # cost + rows without the operator type
                "costonly": list(range(fz.num_types)) + [fz.num_types + 1], # Total Cost alone
                "rowsonly": list(range(fz.num_types)) + [fz.num_types]}[features]      # Plan Rows alone
        self.fz = fz
        self.keep = [i for i in range(fz.node_length) if i not in drop]
        self.num_types = fz.num_types
        self.node_length = len(self.keep)

    def node_vector(self, node):
        return self.fz.node_vector(node)[self.keep]


def plan_depth(node) -> int:
    return 1 + max((plan_depth(ch) for ch in node.get("Plans", [])), default=0)


def _walk(node):
    yield node
    for ch in node.get("Plans", []):
        yield from _walk(ch)


def random_tree(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Both directions of a uniformly random recursive tree over ``n`` nodes."""
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    perm = rng.permutation(n)
    edges = []
    for k in range(1, n):
        parent, child = int(perm[rng.integers(0, k)]), int(perm[k])
        edges += [[parent, child], [child, parent]]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build(plans, fz, db_id: int, topology: str, topo_seed: int, parent_to_child: bool = False, edges: str = "both"):
    out = []
    for i, p in enumerate(plans):
        root = p["Plan"]
        if hasattr(fz, "bind"):
            fz.bind(root)
        x, ei = plan_to_graph(root, fz)
        if topology == "random":
            ei = random_tree(x.size(0), np.random.default_rng([topo_seed, db_id, i]))
        if parent_to_child:
            # Tree convolution reads each parent -> child edge once; nodes are in DFS pre-order,
            # so the parent always has the smaller index (same rule as exp3_e2e/run_kfold.py).
            ei = ei[:, ei[0] < ei[1]]
        elif edges != "both":
            # Edge-direction ablation. Messages flow from ei[0] to ei[1]; DFS pre-order puts the parent
            # before the child. "up": child -> parent only, an operator reads its inputs (the tree-encoder direction);
            # "down": parent -> child only, an operator reads its consumer (the single-database configuration).
            assert topology == "true", "the direction ablation needs the plan topology"
            ei = ei[:, ei[0] > ei[1]] if edges == "up" else ei[:, ei[0] < ei[1]]
        d = Data(x=x, edge_index=ei, y=torch.tensor([float(root.get("Actual Total Time", 0.0))], dtype=torch.float))
        d.n_nodes, d.db_id, d.plan_id = count_nodes(root), db_id, i
        out.append(d)
    return out


def retopo(items, mode: str, topo_seed: int):
    """Copies of ``items`` with the edges replaced (test-time intervention)."""
    out = []
    for d in items:
        e = d.clone()
        if mode == "random":
            e.edge_index = random_tree(d.x.size(0), np.random.default_rng([topo_seed + 1, d.db_id, d.plan_id]))
        else:                                                  # "noedges": only the self-loops of the conv remain
            e.edge_index = torch.zeros((2, 0), dtype=torch.long)
        out.append(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="full", choices=FEATURES)
    ap.add_argument("--conv", default="gatv2")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--virtual-node", action="store_true")
    ap.add_argument("--topology", default="true", choices=TOPOLOGIES)
    ap.add_argument("--topo-seed", type=int, default=0)
    ap.add_argument("--edges", default="both", choices=("both", "up", "down"),
                    help="parent-child edges the encoder reads: both (GNTO), up = child -> parent only (an operator reads "
                         "its inputs), down = parent -> child only (it reads its consumer); self-loops: --set graph.self_loops=false")
    ap.add_argument("--extra-n", action="store_true", help="append the plan size log(1+n) to every node vector")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    c.add_common_args(ap, default_bs=128)
    a = ap.parse_args()
    if not a.out:
        ap.error("--out is required (results/analysis/...)")
    a.no_cache = True                                          # never touch the shared exp2 cache
    out_dir = c.finalize_args(a, "gnto")
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        raise SystemExit(f"completed output exists: {out_dir}")
    log = c.Logger(os.path.join(out_dir, "log.txt"))
    device = torch.device(a.device)

    bench = c.load_benchmark(a, log)
    c.write_benchmark_manifest(out_dir, bench)
    plans = bench["plans"]
    train_plans = [plans[db][i] for db, i in bench["train_index"]]
    if a.features in ("rootcost", "localcost", "rootcum", "localcostshuf", "resid", "residrows", "residonly", "residtype"):
        fz = DerivedFeaturizer(bench["fz"], a.features, train_plans)
    else:
        fz = SubsetFeaturizer(bench["fz"], a.features)
    if a.extra_n:
        fz = WithPlanSize(fz, train_plans)
    overrides = [f"graph.conv={a.conv}", f"graph.num_layers={a.layers}", f"graph.heads={a.heads}"]
    if a.virtual_node:
        overrides.append("graph.virtual_node=true")
    cfg = get_preset("dace_zero_shot").override(node__in_dim=fz.node_length).override(*overrides, *a.set)
    c.write_config(out_dir, a, model="GNTO", preset="dace_zero_shot", gnto_config=cfg.to_dict(),
                   study={"features": a.features, "topology": a.topology, "topo_seed": a.topo_seed,
                          "kept_channels": fz.keep, "extra_n": a.extra_n, "edges": a.edges})
    log(f"dependency study: features={a.features} extra_n={a.extra_n} conv={a.conv} L={a.layers} heads={a.heads} "
        f"vnode={a.virtual_node} topology={a.topology} edges={a.edges} -> {out_dir}\n{cfg.describe()}")

    t0 = time.time()
    dbs = bench["train_dbs"] + bench["test_dbs"]
    tree_edges = a.conv.startswith("tcnn")
    if tree_edges and (a.topology != "true" or any(len(n.get("Plans", [])) > 2 for db in dbs for p in plans[db]
                                                    for n in _walk(p["Plan"]))):
        raise SystemExit("tree convolution needs the true topology and binary plans")
    per_db = {db: build(plans[db], fz, i, a.topology, a.topo_seed, tree_edges, a.edges) for i, db in enumerate(dbs)}
    sizes = {db: len(plans[db]) for db in plans}
    pool = [d for db in bench["train_dbs"] for d in per_db[db]]
    train_list = [pool[i] for i in c.index_positions(bench["train_index"], bench["train_dbs"], sizes)]
    val_list = [pool[i] for i in c.index_positions(bench["val_index"], bench["train_dbs"], sizes)]
    timing = {"featurize_s": time.time() - t0}

    c.set_seed(a.seed)
    model = GNTO.from_config(cfg).to(device)
    params = {"total": model.num_parameters(), "trainable": model.num_parameters(trainable_only=True)}
    log(f"params: {params['total']:,}")
    t0 = time.time()
    train_info = rg.train(model, train_list, val_list, device, a, out_dir, log, select=a.select)
    timing["train_s"] = time.time() - t0
    model = GNTO.load(train_info["ckpt"]).to(device)

    results, interventions = {}, {}
    depth = {db: [plan_depth(p["Plan"]) for p in plans[db]] for db in bench["test_dbs"]}
    for db in bench["test_dbs"]:
        preds, targets = rg.predict(model, per_db[db], device)
        results[db] = c.db_metrics(preds, targets)
        path = c.write_predictions(out_dir, db, preds, targets, [d.n_nodes for d in per_db[db]])
        import pandas as pd
        df = pd.read_csv(path)
        df["depth"] = depth[db]
        if a.topology == "true" and a.conv != "none" and not tree_edges:
            for mode in ("random", "noedges"):
                p2, _ = rg.predict(model, retopo(per_db[db], mode, a.topo_seed), device)
                df[f"pred_ms_{mode}"] = p2
                interventions.setdefault(mode, {})[db] = c.db_metrics(p2, targets)
        df.to_csv(path, index=False)
        log(f"{db:16s} n={results[db]['n']:5d} {c.format_levels(results[db])}")
    extra = {"preset": "dace_zero_shot", "overrides": overrides + a.set, "gnto_config": cfg.to_dict(),
             "study": {"features": a.features, "conv": a.conv, "layers": a.layers, "heads": a.heads,
                       "virtual_node": a.virtual_node, "topology": a.topology, "edges": a.edges}}
    if interventions:
        extra["test_time_interventions"] = {m: {"qerror": v, "avg": c.averages(v)} for m, v in interventions.items()}
    s = c.write_summary(out_dir, "GNTO", params, bench, results, train_info, timing, extra=extra)
    log(f"average over test DBs: {c.format_levels(s['avg'])}")
    for m, v in interventions.items():
        log(f"  test-time {m:8s}: {c.format_levels(c.averages(v))}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
