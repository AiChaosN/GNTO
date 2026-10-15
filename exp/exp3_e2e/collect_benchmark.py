"""Collect all hint plans before randomized repeated execution, on one DB snapshot.

Credentials use libpq environment variables (PGHOST/PGPORT/PGUSER/PGDATABASE/
PGPASSFILE); they are never written to output. Example:

  PGDATABASE=imdb PGPORT=5432 python exp/exp3_e2e/collect_benchmark.py \
      --queries data/e2e/queries_job112.csv --out data/e2e/job112_v2

Only read-only statements are executed. Existing completed packs are validated and
resumed; a different collection protocol may not reuse the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import file_sha256, object_sha256, provenance
from hint_sets import HINT_SETS
from protocol import FINGERPRINT_VERSION, plan_fingerprint, template_key, validate_complete_pack


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def read_queries(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        qid = row.get("query_id", row.get("id"))
        sql = row.get("sql", row.get("query"))
        if not qid or not sql:
            raise ValueError("query CSV must have query_id (or id) and sql (or query)")
        if not str(qid).replace("_", "").replace("-", "").isalnum():
            raise ValueError("query IDs must be safe filename components")
        out.append({"query_id": str(qid), "sql": sql.rstrip().rstrip(";"),
                    "template": row.get("template") or template_key(qid)})
    if len({r["query_id"] for r in out}) != len(out):
        raise ValueError("duplicate query IDs")
    return out


def catalog_snapshot(conn):
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute("SELECT version() AS version, current_database() AS database")
        meta = dict(c.fetchone())
        c.execute("""SELECT c.relname, c.reltuples, c.relpages, c.relallvisible
                     FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                     WHERE n.nspname='public' AND c.relkind IN ('r','m','p')""")
        tables = {r["relname"]: {k: v for k, v in r.items() if k != "relname"} for r in c.fetchall()}
        c.execute("""SELECT t.relname AS table, i.relname AS name, i.relpages,
                     x.indisunique AS unique, x.indnatts AS natts,
                     pg_get_indexdef(i.oid) AS definition
                     FROM pg_index x JOIN pg_class i ON x.indexrelid=i.oid
                     JOIN pg_class t ON x.indrelid=t.oid
                     JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname='public'""")
        indexes = {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in c.fetchall()}
        c.execute("""SELECT tablename, attname, null_frac, avg_width, n_distinct, correlation,
                     most_common_vals::text AS mcv, most_common_freqs AS mcf,
                     histogram_bounds::text AS hist FROM pg_stats WHERE schemaname='public'""")
        columns = {f"{r['tablename']}.{r['attname']}":
                   {k: v for k, v in r.items() if k not in ('tablename', 'attname')} for r in c.fetchall()}
        c.execute("SELECT name, setting FROM pg_settings WHERE category LIKE 'Query Tuning%' OR name IN "
                  "('shared_buffers','work_mem','effective_cache_size','max_parallel_workers_per_gather','server_version')")
        settings = {r["name"]: r["setting"] for r in c.fetchall()}
    return {"source": meta, "tables": tables, "indexes": indexes, "columns": columns, "settings": settings}


def set_hint(cur, hint):
    cur.execute(HINT_SETS[hint]["sql"])


def explain(cur, sql, execute=False):
    options = "FORMAT JSON, COSTS true"
    if execute:
        options += ", ANALYZE true, TIMING false, BUFFERS true"
    cur.execute(f"EXPLAIN ({options}) " + sql)
    return cur.fetchone()[0][0]


def collect_query(conn, query, out, protocol, protocol_hash):
    path = out / f"q_{query['query_id']}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"protocol mismatch: {path}")
        if saved.get("complete"):
            validate_complete_pack(saved)
            return
    pack = {**query, "protocol_version": 2, "protocol_sha256": protocol_hash,
            "expected_hint_ids": list(range(len(HINT_SETS))), "complete": False,
            "latency_metric": "capped_statement_wall_ms",
            "timeout_ms": protocol["timeout_ms"], "results": []}
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (str(protocol["timeout_ms"]),))
        # Persist all planning-time plans before any execution can time out.
        for hint in HINT_SETS:
            set_hint(cur, hint["hint_id"])
            start = time.perf_counter()
            plan = explain(cur, query["sql"])
            pack["results"].append({"hint_id": hint["hint_id"], "plan": plan,
                                    "fingerprint": plan_fingerprint(plan),
                                    "planning_wall_ms": 1000 * (time.perf_counter() - start)})
        atomic_json(path, pack)
        groups = {}
        for r in pack["results"]:
            groups.setdefault(r["fingerprint"], []).append(r)
        representatives = [rows[0] for rows in groups.values()]
        rng = random.Random(f"{protocol['seed']}:{query['query_id']}")
        samples = {r["fingerprint"]: [] for r in representatives}
        for round_id in range(protocol["warmup"] + protocol["repeats"]):
            order = representatives.copy()
            rng.shuffle(order)
            for row in order:
                set_hint(cur, row["hint_id"])
                start = time.perf_counter()
                try:
                    measured = explain(cur, query["sql"], execute=True)
                    if plan_fingerprint(measured) != row["fingerprint"]:
                        raise RuntimeError(f"plan changed during measurement: {query['query_id']}, hint {row['hint_id']}")
                    sample = {"execution_ms": float(measured["Execution Time"]), "timeout": False,
                              "planning_ms": measured.get("Planning Time"), "round": round_id}
                except psycopg2.errors.QueryCanceled:
                    sample = {"execution_ms": None, "timeout": True,
                              "lower_bound_ms": protocol["timeout_ms"], "round": round_id}
                sample["wall_ms"] = 1000 * (time.perf_counter() - start)
                if round_id >= protocol["warmup"]:
                    samples[row["fingerprint"]].append(sample)
            pack["measurements_in_progress"] = samples
            atomic_json(path, pack)
        for rows in groups.values():
            values = samples[rows[0]["fingerprint"]]
            # statement_timeout covers planning as well as execution. Wall latency
            # is therefore the common budgeted target; it is NOT an execution-only
            # lower bound. Keep server execution counters as separate diagnostics.
            budget = [protocol["timeout_ms"] if s["timeout"] else min(s["wall_ms"], protocol["timeout_ms"])
                      for s in values]
            score = float(np.median(budget))
            execution_median = float(np.median([protocol["timeout_ms"] if s["timeout"] else s["execution_ms"]
                                               for s in values]))
            for row in rows:
                row.update(measurements=values, evaluation_ms=score,
                           actual_exec_time_ms=None if execution_median >= protocol["timeout_ms"] else execution_median,
                           censored=score >= protocol["timeout_ms"],
                           timeout_fraction=float(np.mean([s["timeout"] for s in values])),
                           representative_hint=rows[0]["hint_id"])
        pack.pop("measurements_in_progress", None)
        pack.update(complete=True, unique_plan_count=len(groups),
                    finished=datetime.now(timezone.utc).isoformat())
        atomic_json(path, pack)
        set_hint(cur, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-ms", type=int, default=32000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.repeats < 3 or a.repeats % 2 == 0 or a.warmup < 1 or a.timeout_ms <= 0:
        ap.error("use an odd number of >=3 measurements, >=1 warmup and a positive timeout")
    queries = read_queries(a.queries)
    if a.limit:
        queries = queries[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(application_name="gnto_collection", connect_timeout=5)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("SET lock_timeout = '5s'")
        snapshot = catalog_snapshot(conn)
        protocol = {"timeout_ms": a.timeout_ms, "repeats": a.repeats, "warmup": a.warmup,
                    "seed": a.seed, "fingerprint_version": FINGERPRINT_VERSION,
                    "catalog_sha256": object_sha256(snapshot), "queries_sha256": file_sha256(a.queries),
                    "latency": "median min(EXPLAIN ANALYZE statement wall_ms, timeout_ms); includes planning and instrumentation; timeouts remain candidates",
                    "execution": "EXPLAIN ANALYZE TIMING false; randomized unique plans; warm cache",
                    "provenance": provenance()}
        protocol["collector_sources"] = {name: file_sha256(Path(__file__).with_name(name))
                                         for name in ("collect_benchmark.py", "protocol.py", "hint_sets.py")}
        ph = object_sha256({k: v for k, v in protocol.items() if k != "provenance"})
        manifest = out / "collection_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["protocol_sha256"] != ph:
            raise ValueError("output directory belongs to a different source/data/protocol snapshot")
        atomic_json(manifest, {"protocol_sha256": ph, **protocol})
        atomic_json(out / "pg_catalog_snapshot.json", snapshot)
        for i, query in enumerate(queries):
            collect_query(conn, query, out, protocol, ph)
            print(f"[{i+1}/{len(queries)}] {query['query_id']} complete", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
