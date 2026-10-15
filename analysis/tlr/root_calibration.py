"""Root-cost calibration baseline: predict the latency of a plan from the root's estimated Total Cost alone,
log(latency) = a + b log(1 + cost), fitted by least squares on the plans of the training databases of each split;
Q-error percentiles per test database, averaged over the test databases.

    python analysis/tlr/root_calibration.py   # -> results/analysis/tlr_20261008/tables/root_calibration.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.dace import WORKLOADS, load_plans, make_split, plans_path  # noqa: E402


def xy(plans):
    x = np.log1p(np.array([float(p["Plan"].get("Total Cost", 0.0)) for p in plans]))
    y = np.log(np.maximum(np.array([float(p["Plan"].get("Actual Total Time", 0.0)) for p in plans]), 1e-3))
    return x, y


def main() -> None:
    plans = {db: load_plans(plans_path(db)) for db in WORKLOADS}
    rows = []
    for split, args in (("paper1", ("paper", 1)), ("rand1", ("rand", 1)), ("rand2", ("rand", 2))):
        train_ids, test_ids, _ = make_split(*args)
        xs, ys = zip(*(xy(plans[WORKLOADS[i]]) for i in train_ids))
        x, y = np.concatenate(xs), np.concatenate(ys)
        b, a = np.polyfit(x, y, 1)                                   # log latency = a + b log(1 + cost)
        levels = {50: [], 90: [], 99: []}
        for i in test_ids:
            xt, yt = xy(plans[WORKLOADS[i]])
            q = np.exp(np.abs(a + b * xt - yt))                      # Q-error = max(pred/act, act/pred)
            for k in levels:
                levels[k].append(float(np.percentile(q, k)))
        rows.append(dict(split=split, a=a, b=b, q50=np.mean(levels[50]), q90=np.mean(levels[90]), q99=np.mean(levels[99]),
                         q50_per_db=";".join(f"{v:.3f}" for v in levels[50])))
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "analysis", "tlr_20261008", "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "root_calibration.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
