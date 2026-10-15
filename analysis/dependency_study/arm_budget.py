"""Plan selection under a budget of k hint configurations (Bao's regime), re-scored from the stored per-candidate
predictions of the held-out-template runs; no model is retrained.

For every fold the arms are hint 0 (PostgreSQL's default plan) plus the k-1 hints with the best geometric-mean
speed-up over the default on the training queries of that fold (measured latencies of the training folds only).
A selector then picks, among the distinct plans of these arms, the candidate with the lowest stored prediction.
Output: results/analysis/dependency_20261003/tables/steering_budget.csv

    python analysis/dependency_study/arm_budget.py            # job112 and tpch52, k = 5 and 10
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
from protocol import template_key  # noqa: E402

BASE = os.path.join(ROOT, "results", "main_runs")
DATA = os.path.join(ROOT, "data", "e2e")
MODELS = [("gnto_pg_rank", "pg_gatv2_rank1"), ("gnto_pg", "pg_gatv2_rank0"), ("gnto_tcr_rank", "dace_gatv2_rank1"),
          ("bao", "bao_extended_capped"), ("limao", "limao_extended_capped"), ("tcnn_pg_rank", "pg_tcnn_rank1"),
          ("none_pg_rank", "pg_none_rank1"), ("gcn_pg_rank", "pg_gcn_rank1")]


def cluster_ci(values: pd.Series, rng, n_boot=20000):
    tmpl = values.index.map(template_key)
    groups = [values[tmpl == t].to_numpy() for t in sorted(set(tmpl))]
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    idx = rng.integers(len(groups), size=(n_boot, len(groups)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.exp(values.mean())), float(np.exp(np.percentile(boot, 2.5))), float(np.exp(np.percentile(boot, 97.5)))


def workload(wl: str):
    lat, rep = {}, {}
    for p in sorted(glob.glob(os.path.join(DATA, f"hint_plans_v2_{wl}", "q_*.json"))):
        d = json.load(open(p))
        lat[d["query_id"]] = {r["hint_id"]: r["evaluation_ms"] for r in d["results"]}
        rep[d["query_id"]] = {r["hint_id"]: r["representative_hint"] for r in d["results"]}
    return lat, rep


def arms_for(fold_of, lat, queries, f, k):
    train = [q for q in queries if fold_of[q] != f]
    score = {h: np.mean([np.log(lat[q][0] / lat[q][h]) for q in train]) for h in lat[queries[0]] if h != 0}
    return [0] + sorted(score, key=score.get, reverse=True)[:k - 1]


def main() -> None:
    rng = np.random.default_rng(20261007)
    rows = []
    for wl in ("job112", "tpch52"):
        lat, rep = workload(wl)
        queries = sorted(lat)
        for k in (5, 10):
            for label, cfg in MODELS:
                per_seed = []
                for d in sorted(glob.glob(os.path.join(BASE, wl, cfg + "_s*"))):
                    if not os.path.exists(os.path.join(d, "predictions.csv")):
                        continue
                    fold_of = {q: v["fold"] for q, v in json.load(open(os.path.join(d, "folds.json")))["queries"].items()}
                    pr = pd.read_csv(os.path.join(d, "predictions.csv"))
                    pred = {(r.query_id, r.hint_id): r.pred for r in pr.itertuples()}
                    logs = {}
                    for f in sorted(set(fold_of.values())):
                        arms = arms_for(fold_of, lat, queries, f, k)
                        for q in [q for q in queries if fold_of[q] == f]:
                            cands = {rep[q][h] for h in arms}
                            pick = min(cands, key=lambda h: pred.get((q, h), np.inf))
                            logs[q] = np.log(lat[q][0] / lat[q][pick])
                    per_seed.append(pd.Series(logs))
                if not per_seed:
                    continue
                v = pd.concat(per_seed, axis=1).mean(axis=1)
                g, lo, hi = cluster_ci(v, rng)
                rows.append(dict(workload=wl, k=k, model=label, config=cfg, n_seeds=len(per_seed), geo=g, lo=lo, hi=hi))
            # the best plan within the budget (oracle) and the arm choice itself, with the folds of the GNTO run
            d = sorted(glob.glob(os.path.join(BASE, wl, "pg_gatv2_rank1_s*")))[0]
            fold_of = {q: v["fold"] for q, v in json.load(open(os.path.join(d, "folds.json")))["queries"].items()}
            logs = {}
            for f in sorted(set(fold_of.values())):
                arms = arms_for(fold_of, lat, queries, f, k)
                for q in [q for q in queries if fold_of[q] == f]:
                    logs[q] = np.log(lat[q][0] / min(lat[q][h] for h in arms))
            g, lo, hi = cluster_ci(pd.Series(logs), rng)
            rows.append(dict(workload=wl, k=k, model="oracle", config="oracle_within_budget", n_seeds=0, geo=g, lo=lo, hi=hi))
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "analysis", "dependency_20261003", "tables")
    df.to_csv(os.path.join(out, "steering_budget.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
