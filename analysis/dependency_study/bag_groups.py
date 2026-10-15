"""Does the plan topology help where the bag of operator types leaves the shape undetermined?

Splits the test plans of the published split into plans whose multiset of operator types occurs with more than one
tree shape in the 110,000-plan corpus (ambiguous bag) and plans whose bag has a single shape (unique), and repeats the
paired contrasts of analyze_crossdb.py within each group, from the per-plan predictions of the trained models
(no retraining). Output: results/analysis/dependency_20261003/tables/crossdb_bag_groups.csv

    python analysis/dependency_study/bag_groups.py
"""
from __future__ import annotations

import collections
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.dace import PAPER_TEST_IDS, WORKLOADS, load_plans, plans_path  # noqa: E402

DEP = os.path.join(ROOT, "results", "analysis", "dependency_20261003")


def shape(n) -> str:
    return "(" + n.get("Node Type", "") + "".join(sorted(shape(c) for c in n.get("Plans", []))) + ")"


def bag(n, acc):
    acc.append(n.get("Node Type", ""))
    for c in n.get("Plans", []):
        bag(c, acc)
    return acc


def main() -> None:
    by = collections.defaultdict(set)
    plans = {}
    for db in WORKLOADS:
        plans[db] = [p["Plan"] for p in load_plans(plans_path(db))]
        for r in plans[db]:
            by[tuple(sorted(bag(r, [])))].add(shape(r))
    test = [WORKLOADS[i] for i in PAPER_TEST_IDS]
    amb = {db: np.array([len(by[tuple(sorted(bag(r, [])))]) > 1 for r in plans[db]]) for db in test}
    joins = {db: np.array([sum(1 for t in bag(r, []) if "Join" in t or "Nested Loop" in t) for r in plans[db]]) for db in test}
    strata = {"j0": lambda j: j == 0, "j1": lambda j: j == 1, "j2": lambda j: j == 2, "j3+": lambda j: j >= 3}
    n_all = sum(len(m) for m in amb.values())
    n_amb = sum(int(m.sum()) for m in amb.values())

    def load(cfg):
        out = {}
        for d in sorted(glob.glob(os.path.join(DEP, "crossdb", cfg + "_s*"))):
            if os.path.isdir(d):
                out[int(d.rsplit("_s", 1)[1])] = {db: pd.read_csv(os.path.join(d, "predictions", db + ".csv")) for db in test}
        return out

    def medians(cfg, col):
        runs, res = load(cfg), {}
        for s, bydb in runs.items():
            for db in test:
                df, m = bydb[db], amb[db]
                assert len(df) == len(m), (cfg, db, len(df), len(m))
                if col == "qerror":
                    q = df.qerror.values
                else:
                    p, a = df[col].values, df.actual_ms.values
                    q = np.maximum(p / a, a / p)
                res[(s, db, "amb")], res[(s, db, "uni")], res[(s, db, "all")] = np.median(q[m]), np.median(q[~m]), np.median(q)
                for name, sel in strata.items():                     # same join count, ambiguous vs unique bag
                    k = sel(joins[db])
                    res[(s, db, f"amb.{name}")] = np.median(q[m & k]) if (m & k).sum() >= 20 else np.nan
                    res[(s, db, f"uni.{name}")] = np.median(q[~m & k]) if (~m & k).sum() >= 20 else np.nan
        return res, sorted(runs)

    rng = np.random.default_rng(20261007)
    draws = rng.integers(0, len(test), size=(4000, len(test)))
    rows = []
    contrasts = [("gatv2 vs none", "nocost", "nocost_gatv2_L3_true", "qerror", "nocost_none_L3_true", "qerror"),
                 ("gatv2 vs none6", "nocost", "nocost_gatv2_L3_true", "qerror", "nocost_none_L6_true", "qerror"),
                 ("gatv2 true vs random tree", "nocost", "nocost_gatv2_L3_true", "qerror", "nocost_gatv2_L3_random", "qerror"),
                 ("gatv2 testrandom vs gatv2", "nocost", "nocost_gatv2_L3_true", "pred_ms_random", "nocost_gatv2_L3_true", "qerror"),
                 ("gcn vs none", "nocost", "nocost_gcn_L3_true", "qerror", "nocost_none_L3_true", "qerror"),
                 ("gatv2 vs none", "full", "full_gatv2_L3_true", "qerror", "full_none_L3_true", "qerror"),
                 ("gatv2 true vs random tree", "full", "full_gatv2_L3_true", "qerror", "full_gatv2_L3_random", "qerror"),
                 ("gatv2 testrandom vs gatv2", "full", "full_gatv2_L3_true", "pred_ms_random", "full_gatv2_L3_true", "qerror")]
    for label, feat, ca, cola, cb, colb in contrasts:
        A, sa = medians(ca, cola)
        B, sb = medians(cb, colb)
        seeds = sorted(set(sa) & set(sb))
        groups = ["all", "amb", "uni"] + [f"{g}.{st}" for st in strata for g in ("amb", "uni")]
        per_db_rel = {}
        for g in groups:
            a = np.array([[A[(s, db, g)] for db in test] for s in seeds]).mean(axis=0)
            b = np.array([[B[(s, db, g)] for db in test] for s in seeds]).mean(axis=0)
            ok = ~(np.isnan(a) | np.isnan(b))                        # databases that have the stratum
            a, b, d = a[ok], b[ok], draws % max(ok.sum(), 1)
            if ok.sum() < 3:
                continue
            dd = rng.integers(0, ok.sum(), size=draws.shape)
            rb = a[dd].mean(axis=1) / b[dd].mean(axis=1) - 1
            per_db_rel[g] = (a, b, ok)
            n_pl = sum(int(((amb[db] if g.startswith("amb") else ~amb[db]) & (strata[g.split(".")[1]](joins[db]) if "." in g else True)).sum())
                       for db in test) if g != "all" else n_all
            rows.append(dict(contrast=label, feat=feat, group=g, n_seeds=len(seeds), n_dbs=int(ok.sum()), a_mean=a.mean(), b_mean=b.mean(),
                             rel=a.mean() / b.mean() - 1, rel_lo=np.percentile(rb, 2.5), rel_hi=np.percentile(rb, 97.5),
                             dbs_a_lower=int((a < b).sum()), n_plans=n_pl))
        # interaction: ambiguous minus unique effect, per database, bootstrapped over databases
        a1, b1, _ = per_db_rel["amb"]; a2, b2, _ = per_db_rel["uni"]
        diff = (a1 / b1 - 1) - (a2 / b2 - 1)
        ib = diff[draws].mean(axis=1)
        rows.append(dict(contrast=label, feat=feat, group="amb-minus-uni", n_seeds=len(seeds), n_dbs=len(test), a_mean=np.nan, b_mean=np.nan,
                         rel=diff.mean(), rel_lo=np.percentile(ib, 2.5), rel_hi=np.percentile(ib, 97.5), dbs_a_lower=int((diff < 0).sum()), n_plans=n_all))
    df = pd.DataFrame(rows)
    out = os.path.join(DEP, "tables")
    df.to_csv(os.path.join(out, "crossdb_bag_groups.csv"), index=False)
    print(f"test plans {n_all}, ambiguous bag {n_amb} ({100 * n_amb / n_all:.1f}%)")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
