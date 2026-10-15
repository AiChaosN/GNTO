"""Exp3 tables from ``results/exp3_e2e/<steerer>/`` -> ``results/exp3_e2e/tables/[<name>/]``.

* ``steering.csv``   total runtime, speed-up vs PG default, runtime vs oracle, regret, wins / losses /
                     2x regressions, oracle hits - one row per steerer
* ``by_joins.csv``   the same per join-count bucket of the PG-default plan
* ``quality.csv``    (model steerers) Q-error of the cost model on the hint plans and per-query rank
                     correlation between predicted and measured runtime
* ``per_query.csv``  chosen plan's runtime per query and steerer (wide), with oracle and PG default
* ``summary.md``

Usage::

    python exp/exp3_e2e/analyze.py                                   # every steerer found
    python exp/exp3_e2e/analyze.py --steerers pg_default,pg_cost,bao,gnto,oracle --name main
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JOIN_BUCKETS, RESULTS, md  # noqa: E402

LABELS = {"pg_default": "PG default", "oracle": "Oracle", "pg_cost": "PG cost", "bao": "Bao", "gnto": "GNTO"}
ORDER = ["pg_default", "pg_cost", "bao", "gnto", "oracle"]


def label_of(name: str) -> str:
    base, _, tag = name.partition("_") if name not in LABELS else (name, "", "")
    if name in LABELS:
        return LABELS[name]
    return f"{LABELS.get(base, base)} ({tag})" if tag else LABELS.get(base, base)


def discover(results_dir: str = RESULTS) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        out[os.path.basename(os.path.dirname(p))] = os.path.dirname(p)
    return out


def load(names: List[str], results_dir: str = RESULTS) -> Dict[str, dict]:
    runs = {}
    for n in names:
        d = os.path.join(results_dir, n)
        with open(os.path.join(d, "summary.json")) as f:
            s = json.load(f)
        runs[n] = {"summary": s, "picks": pd.read_csv(os.path.join(d, "picks.csv")),
                   "preds": pd.read_csv(os.path.join(d, "predictions.csv")) if os.path.exists(os.path.join(d, "predictions.csv")) else None}
    return runs


def steering_table(runs: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for n, r in runs.items():
        s = r["summary"]
        rows.append({"steerer": label_of(n), "dir": n, "n_queries": s["n_queries"],
                     # primary: per-query speed-up vs PG default (see common.relative_metrics)
                     "geomean_speedup": s.get("rel_speedup_geomean"), "p25": s.get("rel_speedup_p25"),
                     "p50": s.get("rel_speedup_p50"), "p75": s.get("rel_speedup_p75"),
                     "better_pct": s.get("better_pct"), "worse_pct": s.get("worse_pct"), "tie_pct": s.get("tie_pct"),
                     "sign_p": s.get("sign_test_p"), "wilcoxon_p": s.get("wilcoxon_p_vs_pg"),
                     # secondary: workload totals (dominated by the slowest queries)
                     "total_s": s["total_runtime_ms"] / 1000, "speedup_vs_pg": s["speedup_vs_pg_default"],
                     "median_regret": s["median_rel_regret"], "p90_regret": s["p90_rel_regret"],
                     "regressions_2x": s["regressions_2x"], "oracle_hits": s["oracle_hits"],
                     "params": (s.get("params") or {}).get("total", "")})
    return pd.DataFrame(rows)


def by_joins_table(runs: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for b in JOIN_BUCKETS:
        for n, r in runs.items():
            g = r["summary"]["by_join_bucket"].get(b)
            if g:
                rows.append({"join_bucket": b, "steerer": label_of(n), "n": g["n"], "total_s": g["total_runtime_ms"] / 1000,
                             "pg_default_s": g["pg_default_total_ms"] / 1000, "oracle_s": g["oracle_total_ms"] / 1000,
                             "geomean_speedup": g.get("rel_speedup_geomean"), "p50": g.get("rel_speedup_p50"),
                         "better": g.get("better_n"), "worse": g.get("worse_n"), "tie": g.get("tie_n"),
                         "total_speedup_vs_pg": g["speedup_vs_pg_default"], "mean_regret": g["mean_rel_regret"]})
    return pd.DataFrame(rows)


def quality_table(runs: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for n, r in runs.items():
        q = r["summary"].get("prediction_quality")
        if q:
            rows.append({"steerer": label_of(n), **{k: q[k] for k in ("n_plans", "q50", "q90", "q99", "q_mean", "per_query_spearman_mean",
                                                                     "per_query_spearman_median", "per_query_kendall_mean", "global_spearman")}})
    return pd.DataFrame(rows)


def per_query_table(runs: Dict[str, dict]) -> pd.DataFrame:
    base = None
    for n, r in runs.items():
        p = r["picks"][["query_id", "n_joins", "n_candidates", "pg_default_ms", "oracle_ms", "pick_ms"]].rename(columns={"pick_ms": f"{n}_ms"})
        base = p if base is None else base.merge(p.drop(columns=["n_joins", "n_candidates", "pg_default_ms", "oracle_ms"]), on="query_id")
    return base


def write_tables(runs: Dict[str, dict], out_dir: str, title: str) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    t = {"steering": steering_table(runs), "by_joins": by_joins_table(runs), "quality": quality_table(runs), "per_query": per_query_table(runs)}
    for k, df in t.items():
        df.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"# {title}\n\n## Steering (112 queries x up to 49 hint sets, measured runtimes)\n" + md(t["steering"])
                + "\n## By join count of the PG-default plan\n" + md(t["by_joins"])
                + "\n## Cost-model quality on the hint plans\n" + md(t["quality"]))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steerers", default="", help="comma list of steerer dirs (default: all found, canonical order)")
    ap.add_argument("--name", default="")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()
    found = discover(a.results)
    names = [n.strip() for n in a.steerers.split(",") if n.strip()] or sorted(found, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"not found: {missing}; available: {sorted(found)}")
    runs = load(names, a.results)
    out_dir = os.path.join(a.results, "tables", a.name) if a.name else os.path.join(a.results, "tables")
    t = write_tables(runs, out_dir, f"Exp3 - {a.name or 'plan steering'}")
    fmt = lambda v: f"{v:.3f}"
    print("== steering ==\n" + t["steering"].drop(columns=["dir"]).to_string(index=False, float_format=fmt))
    print("\n== by join bucket ==\n" + t["by_joins"].to_string(index=False, float_format=fmt))
    if not t["quality"].empty:
        print("\n== cost-model quality on hint plans ==\n" + t["quality"].to_string(index=False, float_format=fmt))
    print(f"\ntables -> {out_dir}")


if __name__ == "__main__":
    main()
