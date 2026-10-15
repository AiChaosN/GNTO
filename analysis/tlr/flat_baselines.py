"""Flat baselines for the cost decomposition: gradient-boosted trees on (a) the root's cost alone, (b) a flat vector
per plan (operator-type counts, size, depth, root cost and rows, aggregated rows and cumulative costs) and (c) the
same vector plus summaries of the per-operator cost residuals.  Same splits and estimand as root_calibration.py:
fitted on all plans of the training databases, per-database Q-error percentiles on the test databases, averaged.

    python analysis/tlr/flat_baselines.py   # -> results/analysis/tlr_20261008/tables/flat_baselines.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.dace import WORKLOADS, load_plans, make_split, plans_path  # noqa: E402


def walk(node, depth=1):
    yield node, depth
    for ch in node.get("Plans", []):
        yield from walk(ch, depth + 1)


def residual(node):
    return float(node.get("Total Cost", 0.0)) - sum(float(c.get("Total Cost", 0.0)) for c in node.get("Plans", []))


def slog(v):
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.log1p(np.abs(v))


def features(plan, types):
    root = plan["Plan"]
    nodes = list(walk(root))
    n = len(nodes)
    counts = np.zeros(len(types))
    rows, costs, res = [], [], []
    for node, _ in nodes:
        t = node.get("Node Type", "")
        if t in types:
            counts[types[t]] += 1
        rows.append(np.log1p(float(node.get("Plan Rows", 0.0))))
        costs.append(np.log1p(float(node.get("Total Cost", 0.0))))
        res.append(residual(node))
    rows, costs, res = np.array(rows), np.array(costs), np.array(res)
    rootcost = np.log1p(float(root.get("Total Cost", 0.0)))
    base = [rootcost, np.log1p(float(root.get("Plan Rows", 0.0))), n, max(d for _, d in nodes),
            rows.sum(), rows.mean(), rows.max(), costs.mean(), costs.max()]
    resid = list(slog([res.sum(), res.max(), res.min(), res.mean(), np.abs(res).max()])) + [float(res.std()), float((res < 0).sum())]
    return rootcost, np.concatenate([base, counts]), np.concatenate([base, counts, resid])


def main() -> None:
    plans = {db: load_plans(plans_path(db)) for db in WORKLOADS}
    rows = []
    for split, args in (("paper1", ("paper", 1)), ("rand1", ("rand", 1)), ("rand2", ("rand", 2))):
        train_ids, test_ids, _ = make_split(*args)
        types = {}
        for i in train_ids:
            for p in plans[WORKLOADS[i]]:
                for node, _ in walk(p["Plan"]):
                    types.setdefault(node.get("Node Type", ""), len(types))
        def xy(ids):
            X0, X1, X2, Y = [], [], [], []
            for i in ids:
                for p in plans[WORKLOADS[i]]:
                    a, b, c = features(p, types)
                    X0.append([a]); X1.append(b); X2.append(c)
                    Y.append(np.log(max(float(p["Plan"].get("Actual Total Time", 0.0)), 1e-3)))
            return np.array(X0), np.array(X1), np.array(X2), np.array(Y)
        Xtr = xy(train_ids)
        models = {"gbt_rootcost": 0, "gbt_flat": 1, "gbt_flat_residual": 2}
        for name, k in models.items():
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
            m.fit(Xtr[k], Xtr[3])
            levels = {50: [], 90: [], 99: []}
            for i in test_ids:
                Xte = xy([i])
                q = np.exp(np.abs(m.predict(Xte[k]) - Xte[3]))
                for lv in levels:
                    levels[lv].append(float(np.percentile(q, lv)))
            rows.append(dict(split=split, model=name, n_features=Xtr[k].shape[1], q50=np.mean(levels[50]), q90=np.mean(levels[90]),
                             q99=np.mean(levels[99]), q50_per_db=";".join(f"{v:.3f}" for v in levels[50])))
            print(split, name, {k2: round(float(np.mean(v)), 3) for k2, v in levels.items()}, flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "analysis", "tlr_20261008", "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "flat_baselines.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["q50_per_db"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
