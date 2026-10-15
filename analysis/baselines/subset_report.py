"""GNTO trained on the zero-shot model's plans (analysis/baselines/run_gnto_subset.py) against the released zero-shot
model, on the same ten test databases: macro Q-error (mean over databases of the per-database quantile, seeds
averaged) and the paired difference with a database-bootstrap interval, as in report.py.  Read-only.

    python analysis/baselines/subset_report.py   # -> results/analysis/dependency_20261003/tables/baselines_zeroshot_subset.csv
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEP = os.path.join(ROOT, "results", "analysis", "dependency_20261003")
SUB = os.path.join(ROOT, "results", "analysis", "zs_subset_20261008")
LEVELS = (50, 90, 99)


def qerr(pred, actual):
    p, t = np.maximum(pred, 1e-7), np.maximum(actual, 1e-7)
    return np.maximum(p / t, t / p)


def per_db(run_dirs, dbs):
    """seed-averaged  db x level  quantiles of the Q-error"""
    per = []
    for d in run_dirs:
        per.append([[np.percentile(qerr(*(pd.read_csv(os.path.join(d, "predictions", f"{db}.csv"))[["pred_ms", "actual_ms"]]
                                            .abs().to_numpy().T)), k) for k in LEVELS] for db in dbs])
    return np.mean(per, axis=0)


def main() -> None:
    rows = []
    for split, zname, prefix, inputs in (("paper1", "paper", "gnto", "tcr"), ("rand1", "rand1", "gnto", "tcr"),
                                         ("paper1", "paper", "gnto_resid", "tlr"), ("rand1", "rand1", "gnto_resid", "tlr")):
        z_runs = sorted(glob.glob(os.path.join(DEP, "crossdb_zeroshot", f"zeroshot_{zname}_s*", "summary.json")))
        g_runs = sorted(glob.glob(os.path.join(SUB, f"{prefix}_{split}_s*", "summary.json")))
        if not z_runs or not g_runs:
            continue
        dbs = json.load(open(z_runs[0]))["split"]["test_dbs"]
        z = per_db([os.path.dirname(r) for r in z_runs], dbs)
        g = per_db([os.path.dirname(r) for r in g_runs], dbs)
        n_train = json.load(open(os.path.join(os.path.dirname(g_runs[0]), "data_manifest.json")))
        draws = np.random.default_rng(20261003).integers(0, len(dbs), size=(20000, len(dbs)))
        for j, k in enumerate(LEVELS):
            d = g[:, j] - z[:, j]
            boot = d[draws].mean(axis=1)
            rows.append(dict(split=split, inputs=inputs, metric=f"q{k}", gnto_subset=float(g[:, j].mean()), zeroshot=float(z[:, j].mean()),
                             delta=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                             dbs_lower=int((d < 0).sum()), gnto_seeds=len(g_runs), zeroshot_seeds=len(z_runs),
                             train_plans=len(n_train["train_index"]) + len(n_train["val_index"])))
    df = pd.DataFrame(rows)
    out = os.path.join(DEP, "tables", "baselines_zeroshot_subset.csv")
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
