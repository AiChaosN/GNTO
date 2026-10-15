"""Plan selection with PG-stat + local cost against the PG-stat base runs: G over the default plans and the
paired per-query ratio, template bootstrap (20,000 draws), seeds averaged in log space.

    python analysis/tlr/analyze_selection.py      # -> results/analysis/tlr_20261008/selection/tables/selection_local.csv
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
from protocol import template_key  # noqa: E402

NEW = os.path.join(ROOT, "results", "analysis", "tlr_20261008", "selection")
OLD = os.path.join(ROOT, "results", "main_runs")


def cluster_ci(values: pd.Series, rng, n_boot=20000):
    tmpl = values.index.map(template_key)
    groups = [values[tmpl == t].to_numpy() for t in sorted(set(tmpl))]
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    idx = rng.integers(len(groups), size=(n_boot, len(groups)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.exp(values.mean())), float(np.exp(np.percentile(boot, 2.5))), float(np.exp(np.percentile(boot, 97.5)))


def per_query_log_speedup(pattern):
    """Mean over seeds of log(default / pick) per query, from picks.csv of every seed matching the glob."""
    parts = []
    for d in sorted(glob.glob(pattern)):
        p = os.path.join(d, "picks.csv")
        if os.path.exists(p):
            df = pd.read_csv(p).set_index("query_id")
            parts.append(np.log(df.pg_default_ms / df.pick_ms))
    if not parts:
        return None, 0
    return pd.concat(parts, axis=1).mean(axis=1), len(parts)


def main() -> None:
    rng = np.random.default_rng(20261008)
    rows = []
    for wl in ("job112", "tpch52"):
        for conv in ("gatv2", "none", "tcnn", "gcn"):
            for rank in (0, 1):
                new, n_new = per_query_log_speedup(os.path.join(NEW, wl, f"pgl_{conv}_rank{rank}_s*"))
                old, n_old = per_query_log_speedup(os.path.join(OLD, wl, f"pg_{conv}_rank{rank}_s*"))
                if new is None:
                    continue
                g, lo, hi = cluster_ci(new, rng)
                row = dict(workload=wl, conv=conv, rank=rank, n_seeds=n_new, G_local=g, lo=lo, hi=hi)
                if old is not None:
                    g0, lo0, hi0 = cluster_ci(old, rng)
                    r, rlo, rhi = cluster_ci(new - old, rng)          # > 1: local cost selects faster plans
                    row.update(G_pgstat=g0, pg_lo=lo0, pg_hi=hi0, ratio=r, ratio_lo=rlo, ratio_hi=rhi,
                               queries_faster=int((new - old > np.log(1.05)).sum()), queries_slower=int((new - old < -np.log(1.05)).sum()))
                rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(NEW, "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "selection_local.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
