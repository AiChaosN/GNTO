"""Candidates planned with a disabled method: PostgreSQL adds a disabling penalty to the estimated
cost of a plan that uses a method its enable_* switch turned off. Counts the hint records and the distinct candidates
whose root Total Cost carries it (>= 1e9), per selection workload.  Read-only.

    python analysis/dependency_study/penalty_counts.py   # -> results/analysis/dependency_20261003/tables/penalty_counts.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
from common import load_packs, PG_DEFAULT_HINT  # noqa: E402
from protocol import plan_fingerprint  # noqa: E402

THRESHOLD = 1e9


def cost(r):
    p = r["plan"]
    return float(p["Plan"]["Total Cost"]) if "Plan" in p else float(p["Total Cost"])


def main() -> None:
    rows = []
    for wl in ("job112", "tpch52"):
        packs = load_packs(os.path.join(ROOT, "data", "e2e", f"hint_plans_v2_{wl}"))
        records = penalized = distinct = distinct_pen = default_pen = twins = 0
        for pack in packs:
            groups = {}
            for r in sorted(pack["results"], key=lambda r: int(r["hint_id"])):
                if "plan" not in r:
                    continue
                records += 1
                penalized += cost(r) >= THRESHOLD
                groups.setdefault(plan_fingerprint(r["plan"]), []).append(r)
            for rs in groups.values():
                distinct += 1
                pen = cost(rs[0]) >= THRESHOLD
                distinct_pen += pen
                twins += pen and any(cost(r) < THRESHOLD for r in rs)
                default_pen += pen and any(int(r["hint_id"]) == PG_DEFAULT_HINT for r in rs)
        rows.append(dict(workload=wl, records=records, records_penalized=penalized, distinct=distinct, distinct_penalized=distinct_pen,
                         penalized_with_unpenalized_twin=twins, default_plan_penalized=default_pen))
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "analysis", "dependency_20261003", "tables", "penalty_counts.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
