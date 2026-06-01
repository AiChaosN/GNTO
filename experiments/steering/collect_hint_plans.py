"""
Collect ``EXPLAIN (FORMAT JSON)`` plans for each (query x hint_set) pair against
a live PostgreSQL instance.

This is the **online** step of the Bao-style steering experiment (review item
R1-D5). The output (one JSON file per query) is consumed by
``run_steering_eval.py`` for the **offline** scoring step.

Usage:
    PGHOST=... PGUSER=... PGDATABASE=imdbload \\
        python experiments/steering/collect_hint_plans.py \\
            --queries-csv path/to/queries.csv \\
            --query-col sql \\
            --query-id-col query_id \\
            --output-dir results/HintPlans \\
            [--analyze]   # also EXPLAIN ANALYZE for real latencies (SLOW)

If ``--analyze`` is given, each query x hint pair is actually executed by PG;
that's the only way to get ground-truth end-to-end latency (the metric R1-D5
ultimately cares about). Without it we only have PG's cost estimates, which
are good enough to validate that the pipeline + the model wiring works.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from pathlib import Path

import pandas as pd

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from experiments.steering.hint_sets import HINT_SETS


def run_explain(cur, sql: str, analyze: bool) -> dict:
    prefix = "EXPLAIN (FORMAT JSON, COSTS true, VERBOSE false"
    if analyze:
        prefix += ", ANALYZE true, TIMING true, BUFFERS true"
    prefix += ") "
    cur.execute(prefix + sql)
    rows = cur.fetchall()
    # PG returns a 1-row list-of-list; the JSON is rows[0][0]
    return rows[0][0][0]


def collect_for_query(cur, sql: str, analyze: bool, timeout_ms: int = 60_000):
    """Return list of dicts: hint_id, total_cost, exec_time_ms (or None), plan_json."""
    results = []
    cur.execute(f"SET statement_timeout TO {timeout_ms};")
    for hs in HINT_SETS:
        for stmt in hs["sql"].split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt + ";")
        try:
            t0 = time.perf_counter()
            plan = run_explain(cur, sql, analyze=analyze)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            results.append({"hint_id": hs["hint_id"], "error": str(e), "elapsed_ms": None})
            continue

        total_cost = plan["Plan"]["Total Cost"]
        actual_ms = plan.get("Execution Time") if analyze else None
        results.append({
            "hint_id": hs["hint_id"],
            "total_cost": float(total_cost),
            "actual_exec_time_ms": float(actual_ms) if actual_ms is not None else None,
            "wall_clock_ms": elapsed_ms,
            "plan": plan,
        })
    # Reset to all-on so subsequent runs see PG defaults
    for t in ("enable_hashjoin", "enable_mergejoin", "enable_nestloop",
              "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"):
        cur.execute(f"SET {t} TO on;")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-csv", required=True)
    ap.add_argument("--query-col", default="sql")
    ap.add_argument("--query-id-col", default="query_id")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--analyze", action="store_true", help="Run EXPLAIN ANALYZE (executes the query)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N queries")
    ap.add_argument("--timeout-ms", type=int, default=60_000)
    args = ap.parse_args()

    import psycopg2  # lazy import: only required when actually collecting

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.queries_csv)
    if args.limit:
        df = df.head(args.limit)

    conn = psycopg2.connect("")  # honors PGHOST/PGUSER/PGDATABASE env vars
    conn.autocommit = True
    cur = conn.cursor()

    for _, row in df.iterrows():
        qid = row[args.query_id_col]
        sql = row[args.query_col]
        out_path = out_dir / f"q_{qid}.json"
        if out_path.exists():
            print(f"[skip] q_{qid} (already collected)")
            continue
        print(f"[run ] q_{qid} ({len(HINT_SETS)} hint sets)...")
        try:
            results = collect_for_query(cur, sql, analyze=args.analyze, timeout_ms=args.timeout_ms)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            continue
        with open(out_path, "w") as f:
            json.dump({"query_id": qid, "sql": sql, "analyze": args.analyze, "results": results}, f)
        print(f"  -> wrote {out_path}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
