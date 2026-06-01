"""
3-way L3 (end-to-end steering) comparison: PureBAO vs LIMAO+BAO vs LIMAO+GNTO vs PG default.

All three runs went through the same 112 JOB queries from
``LIMAOLifeLongRLDB/imdb_assorted_3/`` over 4 LIMAO iterations.

Inputs (logs of the form ``BAO|GNTO <q>.sql <t>`` and ``x x <q>.sql <t> PG``):
  - LIMAO + BAO model :  LIMAOLifeLongRLDB/gnto_ex/archive/.../logs/bao_log
  - LIMAO + GNTO model:  LIMAOLifeLongRLDB/gnto_ex/archive/.../logs/gnto_log
  - Pure BAO          :  results/PureBao_<ts>/bao_run.log

Output: ``results/Summary_0519/l3_3way_<ts>.json`` + pivot CSV with per-query
mean times per steerer and a summary of total runtime / wins-over-PG.
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PG_PAT = re.compile(r"^x x (\S+\.sql) ([\d\.]+) PG\s*$")
MODEL_PAT = re.compile(r"^(BAO|GNTO)\s+(\S+\.sql)\s+([\d\.]+)\s*$")


def parse(path, model_label=None):
    """Return list of (label, query, time).

    ``model_label`` overrides whatever tag is on the file (so we can rename
    "BAO" -> "LIMAO+BAO" or "PureBAO" depending on which log it came from).
    """
    rows = []
    with open(path) as f:
        for line in f:
            m = PG_PAT.match(line)
            if m:
                rows.append(("PG", m.group(1), float(m.group(2))))
                continue
            m = MODEL_PAT.match(line)
            if m:
                tag = model_label if model_label else m.group(1)
                rows.append((tag, m.group(2), float(m.group(3))))
    return rows


def aggregate(rows):
    df = pd.DataFrame(rows, columns=["model", "query", "time"])
    return df.groupby(["model", "query"]).agg(
        mean=("time", "mean"),
        median=("time", "median"),
        min=("time", "min"),
        count=("time", "size"),
    ).reset_index()


def summarize_vs_pg(agg, steerer, baseline="PG"):
    sub = agg[agg["model"] == steerer].set_index("query")["mean"]
    pg = agg[agg["model"] == baseline].set_index("query")["mean"]
    common = sub.index.intersection(pg.index)
    if len(common) == 0:
        return {"steerer": steerer, "n": 0}
    s = sub.loc[common]; p = pg.loc[common]
    speedup = p / s
    return {
        "steerer": steerer,
        "n_queries": int(len(common)),
        "total_steerer_s": float(s.sum()),
        "total_pg_s": float(p.sum()),
        "global_speedup_vs_pg": float(p.sum() / s.sum()),
        "mean_speedup_vs_pg": float(speedup.mean()),
        "median_speedup_vs_pg": float(speedup.median()),
        "wins_over_pg": int((speedup > 1.0).sum()),
        "losses_vs_pg": int((speedup < 1.0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limao-bao-log",
                    default="/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_133515/logs/bao_log")
    ap.add_argument("--limao-gnto-log",
                    default="/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_133515/logs/gnto_log")
    ap.add_argument("--pure-bao-log",
                    default=os.path.join(GNTO_PATH, "results", "PureBao_0519_1825", "bao_run.log"))
    ap.add_argument("--out-dir",
                    default=os.path.join(GNTO_PATH, "results", "Summary_0519"))
    args = ap.parse_args()

    rows = []
    rows += parse(args.limao_bao_log, model_label="LIMAO+BAO")
    rows += parse(args.limao_gnto_log, model_label="LIMAO+GNTO")
    rows += parse(args.pure_bao_log, model_label="PureBAO")
    # PG rows come from every file; we combine and let pandas mean over them
    print(f"Total rows parsed: {len(rows)}")

    agg = aggregate(rows)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pivot = agg.pivot(index="query", columns="model", values="mean")
    csv_path = os.path.join(args.out_dir, "l3_3way_pivot.csv")
    pivot.to_csv(csv_path)
    print(f"Wrote per-query pivot: {csv_path}")

    summaries = [summarize_vs_pg(agg, m) for m in ("PureBAO", "LIMAO+BAO", "LIMAO+GNTO")]
    out = {
        "logs": {
            "limao_bao_log": args.limao_bao_log,
            "limao_gnto_log": args.limao_gnto_log,
            "pure_bao_log": args.pure_bao_log,
        },
        "steerers": summaries,
        "note": (
            "All three runs used the same 112-query workload (LIMAO's imdb_assorted_3) "
            "and 4 outer iterations. PG default is averaged across all three runs' "
            "warmup phases. Higher speedup = better."
        ),
    }
    json_path = os.path.join(args.out_dir, "l3_3way_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote summary: {json_path}\n")

    # Console table
    print(f"{'Steerer':<14}{'n':>4}{'total(s)':>11}{'PG total':>11}{'gobal x':>9}{'mean x':>8}{'med x':>8}{'wins':>6}{'loss':>6}")
    for s in summaries:
        if s.get("n_queries", 0) == 0:
            print(f"{s['steerer']:<14}{'-':>4}"); continue
        print(f"{s['steerer']:<14}{s['n_queries']:>4d}{s['total_steerer_s']:>11.2f}"
              f"{s['total_pg_s']:>11.2f}{s['global_speedup_vs_pg']:>9.3f}"
              f"{s['mean_speedup_vs_pg']:>8.3f}{s['median_speedup_vs_pg']:>8.3f}"
              f"{s['wins_over_pg']:>6d}{s['losses_vs_pg']:>6d}")


if __name__ == "__main__":
    main()
