"""
Complete 2x2 (framework x model) L3 comparison, all at 20 iterations.

Cells:
  PureBAO + TreeCNN  (BAO framework + Bao's own model)
  PureBAO + GNTO     (BAO framework + GNTO model)
  LIMAO   + TreeCNN  (LIMAO framework + Bao's model)
  LIMAO   + GNTO     (LIMAO framework + GNTO model)

Each cell is parsed from its own log; PG default is averaged across all four.
Output: results/Summary_0519/l3_4way_pivot.csv + l3_4way_summary.json.
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


def parse(path, model_label):
    """Return list of (label, query, time)."""
    rows = []
    with open(path) as f:
        for line in f:
            m = PG_PAT.match(line)
            if m:
                rows.append(("PG", m.group(1), float(m.group(2))))
                continue
            m = MODEL_PAT.match(line)
            if m:
                rows.append((model_label, m.group(2), float(m.group(3))))
    return rows


def aggregate(rows):
    df = pd.DataFrame(rows, columns=["model", "query", "time"])
    return df.groupby(["model", "query"]).agg(
        mean=("time", "mean"),
        median=("time", "median"),
        count=("time", "size"),
    ).reset_index()


def summarize(agg, model_tag, baseline="PG"):
    sub = agg[agg["model"] == model_tag].set_index("query")["mean"]
    pg = agg[agg["model"] == baseline].set_index("query")["mean"]
    common = sub.index.intersection(pg.index)
    if len(common) == 0:
        return {"steerer": model_tag, "n": 0}
    s = sub.loc[common]; p = pg.loc[common]
    speedup = p / s
    return {
        "steerer": model_tag,
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
    ap.add_argument("--purebao-bao-log", required=True)
    ap.add_argument("--purebao-gnto-log", required=True)
    ap.add_argument("--limao-bao-log", required=True)
    ap.add_argument("--limao-gnto-log", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = []
    rows += parse(args.purebao_bao_log, "PureBAO+TreeCNN")
    rows += parse(args.purebao_gnto_log, "PureBAO+GNTO")
    rows += parse(args.limao_bao_log, "LIMAO+TreeCNN")
    rows += parse(args.limao_gnto_log, "LIMAO+GNTO")
    print(f"Total rows: {len(rows)}")

    agg = aggregate(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pivot = agg.pivot(index="query", columns="model", values="mean")
    pivot.to_csv(os.path.join(args.out_dir, "l3_4way_pivot.csv"))

    summaries = [summarize(agg, m) for m in
                 ("PureBAO+TreeCNN", "PureBAO+GNTO", "LIMAO+TreeCNN", "LIMAO+GNTO")]
    out = {
        "design": "2x2: framework x model, all 20 iterations",
        "steerers": summaries,
        "logs": {
            "PureBAO+TreeCNN": args.purebao_bao_log,
            "PureBAO+GNTO": args.purebao_gnto_log,
            "LIMAO+TreeCNN": args.limao_bao_log,
            "LIMAO+GNTO": args.limao_gnto_log,
        },
    }
    with open(os.path.join(args.out_dir, "l3_4way_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'Steerer':<20}{'n':>4}{'total':>10}{'PG':>8}{'global x':>10}{'mean x':>8}{'med x':>8}{'wins':>6}{'loss':>6}")
    for s in summaries:
        if s.get("n_queries", 0) == 0:
            print(f"{s['steerer']:<20}{'-':>4}"); continue
        print(f"{s['steerer']:<20}{s['n_queries']:>4d}{s['total_steerer_s']:>10.1f}"
              f"{s['total_pg_s']:>8.1f}{s['global_speedup_vs_pg']:>10.3f}"
              f"{s['mean_speedup_vs_pg']:>8.3f}{s['median_speedup_vs_pg']:>8.3f}"
              f"{s['wins_over_pg']:>6d}{s['losses_vs_pg']:>6d}")


if __name__ == "__main__":
    main()
