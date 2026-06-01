"""
Parse the LIMAO/BAO/GNTO experiment logs into a clean L3 (end-to-end steering)
comparison table.

Sources (from LIMAOLifeLongRLDB/gnto_ex/archive/<run>/logs/):
  - bao_log   : ``x x <q> <t> PG`` (PG default) + ``BAO <q> <t>`` (Bao cost model picks)
  - gnto_log  : same PG lines + ``GNTO <q> <t>`` (GNTO cost model picks)

Each query is run multiple times across LIMAO's training iterations. We aggregate
across runs (mean / median) per (query, steerer) and write a unified CSV.

Note: both ``BAO`` and ``GNTO`` rows are LIMAO-framework runs with different
cost models. There is no separate "pure Bao without LIMAO's lifelong extensions"
log here -- if needed, that requires a fresh experiment with the lifelong
features disabled.
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


PG_PAT = re.compile(r"^x x (\S+\.sql) ([\d\.]+) PG\s*$")
MODEL_PAT = re.compile(r"^(BAO|GNTO)\s+(\S+\.sql)\s+([\d\.]+)\s*$")


def parse_log(path):
    """Return list of (model_tag, query, time_seconds). model_tag in {PG, BAO, GNTO}."""
    rows = []
    with open(path) as f:
        for line in f:
            m = PG_PAT.match(line)
            if m:
                rows.append(("PG", m.group(1), float(m.group(2))))
                continue
            m = MODEL_PAT.match(line)
            if m:
                rows.append((m.group(1), m.group(2), float(m.group(3))))
    return rows


def aggregate(rows):
    """Group by (model_tag, query) -> mean, median, min, count."""
    df = pd.DataFrame(rows, columns=["model", "query", "time"])
    agg = df.groupby(["model", "query"]).agg(
        mean=("time", "mean"),
        median=("time", "median"),
        min=("time", "min"),
        count=("time", "size"),
    ).reset_index()
    return df, agg


def build_table(agg, models=("PG", "BAO", "GNTO")):
    """Pivot to one column per model (using mean time)."""
    pieces = []
    for m in models:
        sub = agg[agg["model"] == m].set_index("query")[["mean", "median", "min", "count"]]
        sub.columns = [f"{m}_{c}" for c in sub.columns]
        pieces.append(sub)
    out = pd.concat(pieces, axis=1, join="outer")
    return out


def steerer_summary(agg, model_tag, baseline_tag="PG"):
    """Aggregate-across-queries metrics for one steerer vs PG baseline."""
    rows = []
    for query in agg["query"].unique():
        m_rows = agg[(agg["model"] == model_tag) & (agg["query"] == query)]
        pg_rows = agg[(agg["model"] == baseline_tag) & (agg["query"] == query)]
        if m_rows.empty or pg_rows.empty:
            continue
        m_t = float(m_rows["mean"].iloc[0])
        pg_t = float(pg_rows["mean"].iloc[0])
        rows.append({"query": query, "model_time": m_t, "pg_time": pg_t, "speedup": pg_t / m_t})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"model": model_tag, "n_queries": 0}
    return {
        "model": model_tag,
        "n_queries": len(df),
        "total_model_time_s": float(df["model_time"].sum()),
        "total_pg_time_s": float(df["pg_time"].sum()),
        "global_speedup_vs_pg": float(df["pg_time"].sum() / df["model_time"].sum()),
        "mean_speedup_vs_pg": float(df["speedup"].mean()),
        "median_speedup_vs_pg": float(df["speedup"].median()),
        "wins_over_pg": int((df["speedup"] > 1.0).sum()),
        "losses_vs_pg": int((df["speedup"] < 1.0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bao-log", default="/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_133515/logs/bao_log")
    ap.add_argument("--gnto-log", default="/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_133515/logs/gnto_log")
    ap.add_argument("--out-dir", default=os.path.join(GNTO_PATH, "results", "Summary_0519"))
    args = ap.parse_args()

    bao_rows = parse_log(args.bao_log)
    gnto_rows = parse_log(args.gnto_log)
    print(f"Parsed: bao_log={len(bao_rows)} rows ({sum(1 for r in bao_rows if r[0]=='BAO')} BAO + "
          f"{sum(1 for r in bao_rows if r[0]=='PG')} PG)")
    print(f"        gnto_log={len(gnto_rows)} rows ({sum(1 for r in gnto_rows if r[0]=='GNTO')} GNTO + "
          f"{sum(1 for r in gnto_rows if r[0]=='PG')} PG)")

    # Combine PG runs from both logs (they're the same baseline executed in both experiments)
    combined = bao_rows + gnto_rows
    df_raw, agg = aggregate(combined)
    pivot = build_table(agg)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "l3_endtoend_pivot.csv")
    pivot.to_csv(csv_path)
    print(f"\nWrote per-query pivot: {csv_path}")

    # Build summary across all queries
    summaries = [steerer_summary(agg, m) for m in ("BAO", "GNTO")]
    summary_path = os.path.join(args.out_dir, "l3_endtoend_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "source_bao_log": args.bao_log,
                   "source_gnto_log": args.gnto_log,
                   "steerers": summaries}, f, indent=2)

    print(f"\n=== L3 steering summary (vs PG default) ===")
    for s in summaries:
        print(f"\n[{s['model']}] over {s.get('n_queries', 0)} queries")
        for k, v in s.items():
            if k == "model" or k == "n_queries":
                continue
            print(f"  {k}: {v}")

    # Per-query head printout
    print("\n=== Per-query head (first 10) ===")
    cols = ["PG_mean", "BAO_mean", "GNTO_mean"]
    head_df = pivot[cols].head(10).copy()
    print(head_df.round(3).to_string())


if __name__ == "__main__":
    main()
