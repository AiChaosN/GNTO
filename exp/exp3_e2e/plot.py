"""Exp3 figures -> ``results/exp3_e2e/figures/[<name>/]``.

* ``total_runtime``   total measured runtime of the chosen plans per steerer, with PG default and oracle lines
* ``regret_cdf``      per-query relative regret (chosen / oracle) CDF per steerer
* ``by_joins``        total runtime per join-count bucket and steerer
* ``rank_quality``    per-query Spearman correlation (predicted vs. measured over a query's variants), model steerers
* ``per_query``       chosen-plan runtime per query (log), steerers vs. oracle, sorted by PG default

Usage::

    python exp/exp3_e2e/plot.py
    python exp/exp3_e2e/plot.py --steerers pg_default,pg_cost,bao,gnto,oracle --name main
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ORDER, discover, label_of, load, write_tables  # noqa: E402
from common import JOIN_BUCKETS, RESULTS  # noqa: E402

COLORS = {"pg_default": "#7f7f7f", "oracle": "#2ca02c", "pg_cost": "#bcbd22", "bao": "#e07b39", "gnto": "#1f5fa8"}
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def color(n):
    return COLORS.get(n, COLORS.get(n.split("_")[0], "#9467bd"))


def _save(fig, out, name, formats):
    for ext in formats:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  {name}.{'/'.join(formats)}")


def plot_total(runs, out, formats):
    names = list(runs)
    tot = [runs[n]["summary"]["total_runtime_ms"] / 1000 for n in names]
    fig, ax = plt.subplots(figsize=(1.8 + 1.3 * len(names), 4.2))
    bars = ax.bar([label_of(n) for n in names], tot, color=[color(n) for n in names], width=0.6)
    for b, v, n in zip(bars, tot, names):
        ax.annotate(f"{v:.1f}s\n(x{runs[n]['summary']['speedup_vs_pg_default']:.2f})", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=9)
    s0 = next(iter(runs.values()))["summary"]
    ax.axhline(s0["pg_default_total_ms"] / 1000, color=COLORS["pg_default"], ls="--", lw=1, label="PG default")
    ax.axhline(s0["oracle_total_ms"] / 1000, color=COLORS["oracle"], ls="--", lw=1, label="oracle")
    ax.set_ylabel("total runtime of chosen plans (s)")
    ax.set_ylim(top=max(tot) * 1.3)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title(f"{s0['n_queries']} queries, up to 49 hint sets each")
    fig.tight_layout()
    _save(fig, out, "total_runtime", formats)


def plot_regret_cdf(runs, out, formats):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for n, r in runs.items():
        if n == "oracle":
            continue
        q = np.sort(r["picks"].rel_regret.to_numpy())
        ax.step(q, np.arange(1, len(q) + 1) / len(q), where="post", label=f"{label_of(n)} (mean {q.mean():.1f})", color=color(n), lw=2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.set_xlabel("relative regret = chosen runtime / oracle runtime (log)")
    ax.set_ylabel("fraction of queries")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out, "regret_cdf", formats)


def plot_by_joins(runs, out, formats):
    names = list(runs)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(JOIN_BUCKETS)); width = 0.8 / len(names)
    for i, n in enumerate(names):
        by = runs[n]["summary"]["by_join_bucket"]
        vals = [by[b]["total_runtime_ms"] / 1000 if b in by else np.nan for b in JOIN_BUCKETS]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=label_of(n), color=color(n))
    ns = [next(iter(runs.values()))["summary"]["by_join_bucket"].get(b, {}).get("n", 0) for b in JOIN_BUCKETS]
    ax.set_xticks(x, [f"{b} joins\n(n={n})" for b, n in zip(JOIN_BUCKETS, ns)])
    ax.set_ylabel("total runtime (s)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "by_joins", formats)


def plot_rank_quality(runs, out, formats):
    data, labels, cols = [], [], []
    for n, r in runs.items():
        if r["preds"] is None:
            continue
        rhos = [spearmanr(g.pred, g.actual_ms).correlation for _, g in r["preds"].groupby("query_id") if len(g) >= 3 and g.pred.nunique() > 1]
        data.append([v for v in rhos if not np.isnan(v)]); labels.append(label_of(n)); cols.append(color(n))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(data), 4.2))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("per-query Spearman (pred vs. measured runtime)")
    ax.set_title("how well each model ranks a query's hint variants")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, "rank_quality", formats)


def plot_per_query(runs, out, formats):
    base = next(iter(runs.values()))["picks"].sort_values("pg_default_ms").reset_index(drop=True)
    order = base.query_id.tolist()
    fig, ax = plt.subplots(figsize=(13, 4.4))
    x = np.arange(len(order))
    ax.plot(x, base.pg_default_ms / 1000, color=COLORS["pg_default"], lw=1.5, label="PG default")
    ax.plot(x, base.oracle_ms / 1000, color=COLORS["oracle"], lw=1.5, label="oracle")
    for n, r in runs.items():
        if n in ("pg_default", "oracle"):
            continue
        p = r["picks"].set_index("query_id").loc[order]
        ax.scatter(x, p.pick_ms / 1000, s=14, color=color(n), label=label_of(n), alpha=0.85)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.set_xlabel("queries sorted by PG-default runtime")
    ax.set_ylabel("runtime of chosen plan (s, log)")
    ax.legend(ncol=5, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "per_query", formats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steerers", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()
    found = discover(a.results)
    names = [n.strip() for n in a.steerers.split(",") if n.strip()] or sorted(found, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
    runs = load(names, a.results)
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    out = os.path.join(a.results, "figures", a.name) if a.name else os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    write_tables(runs, os.path.join(a.results, "tables", a.name) if a.name else os.path.join(a.results, "tables"), f"Exp3 - {a.name or 'plan steering'}")
    print(f"figures -> {out}")
    plot_total(runs, out, formats)
    plot_regret_cdf(runs, out, formats)
    plot_by_joins(runs, out, formats)
    plot_rank_quality(runs, out, formats)
    plot_per_query(runs, out, formats)


if __name__ == "__main__":
    main()
