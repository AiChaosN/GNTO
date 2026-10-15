"""Plan selection on held-out templates for the reach, depth and cost variants against the matching main runs
(three-layer GATv2 / None with the same inputs and objective): G over the default plans and the paired per-query
ratio, template bootstrap (20,000 draws), seeds averaged in log space.

    python analysis/attn/analyze_selection_variants.py   # -> results/analysis/attn_20261008/selection/tables/variants.csv
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

NEW = os.path.join(ROOT, "results", "analysis", "attn_20261008", "selection")
OLD = os.path.join(ROOT, "results", "main_runs")


def cluster_ci(values: pd.Series, rng, n_boot=20000):
    tmpl = values.index.map(template_key)
    groups = [values[tmpl == t].to_numpy() for t in sorted(set(tmpl))]
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    idx = rng.integers(len(groups), size=(n_boot, len(groups)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.exp(values.mean())), float(np.exp(np.percentile(boot, 2.5))), float(np.exp(np.percentile(boot, 97.5)))


def per_query(pattern):
    parts = []
    for d in sorted(glob.glob(pattern)):
        p = os.path.join(d, "picks.csv")
        if os.path.exists(p):
            df = pd.read_csv(p).set_index("query_id")
            parts.append(np.log(df.pg_default_ms / df.pick_ms))
    return (pd.concat(parts, axis=1).mean(axis=1), len(parts)) if parts else (None, 0)


def main() -> None:
    rng = np.random.default_rng(20261008)
    rows = []
    for wl in ("job112", "tpch52"):
        for feat in ("dace", "pg"):
            for rank in (0, 1):
                base, nb = per_query(os.path.join(OLD, wl, f"{feat}_gatv2_rank{rank}_s*"))
                variants = [(f"{feat}_tr{r}_rank{rank}", f"transformer {r}") for r in ("local", "subplan", "global")]
                variants.append((f"{feat}_gatv2L1_rank{rank}", "gatv2 1 layer"))
                if feat == "dace":
                    for cost in ("rootcost", "localcost", "resid", "residslog"):
                        for conv in ("gatv2", "none"):
                            variants.append((f"{cost}_{conv}_rank{rank}", f"{conv} {cost}"))
                for tag, label in variants:
                    new, n = per_query(os.path.join(NEW, wl, f"{tag}_s*"))
                    if new is None:
                        continue
                    # cost variants of None are compared with None on TCR
                    b, nb_ = (base, nb)
                    if label.startswith("none"):
                        b, nb_ = per_query(os.path.join(OLD, wl, f"{feat}_none_rank{rank}_s*"))
                    g, lo, hi = cluster_ci(new, rng)
                    row = dict(workload=wl, inputs=feat, rank=rank, variant=label, tag=tag, n_seeds=n, G=g, lo=lo, hi=hi)
                    if b is not None:
                        g0, lo0, hi0 = cluster_ci(b, rng)
                        r, rlo, rhi = cluster_ci(new - b, rng)
                        row.update(G_base=g0, ratio=r, ratio_lo=rlo, ratio_hi=rhi,
                                   faster=int((new - b > np.log(1.05)).sum()), slower=int((new - b < -np.log(1.05)).sum()))
                    rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(NEW, "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "variants.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
