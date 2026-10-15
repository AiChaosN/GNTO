"""Exp1 figures for a list of model directories under ``results/exp1_qerror/``.

Writes PNG + PDF to ``results/exp1_qerror/figures/[<name>/]``:

* ``qerror_levels``      grouped bars, Q50 / Q75 / Q90 / Q95 / Q99 per workload (log y)
* ``qerror_cdf``         per-plan Q-error CDF per workload, guides at the five levels
* ``qerror_epochs``      validation Q-error at each level vs. training epoch
* ``qerror_by_joins``    Q50 / Q90 / Q99 vs. number of joins
* ``qerror_by_latency``  Q50 / Q90 / Q99 vs. actual-latency tier
* ``model_params``       parameter counts
* ``qerror_holdout``     (``--holdout``) validation plans with / without a held-out join pair

Usage::

    python exp/exp1_qerror/plot.py                                          # qf vs gnto -> figures/
    python exp/exp1_qerror/plot.py --models gnto_joinid,gnto --name ablation_join_encoding
    python exp/exp1_qerror/plot.py --models gnto_joinid_holdout,gnto_holdout,qf --holdout --name ablation_join_encoding_holdout
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (HOLDOUT_LABELS, LATENCY_LABELS, WORKLOAD_LABELS, Model, bucket_table, discover,  # noqa: E402
                     epoch_table, holdout_table, level_table, load_models, params_table, parse_labels, qerror_of,
                     write_tables)
from common import LEVELS, RESULTS  # noqa: E402

PALETTE = {"QueryFormer": "#e07b39", "GNTO": "#1f5fa8", "GNTO (string-id join)": "#9a9a9a"}
FALLBACK = ["#2ca02c", "#9467bd", "#8c564b", "#17becf", "#d62728"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LEVEL_LABELS = [f"Q{k}" for k in LEVELS]

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def colors_for(models: List[Model]) -> Dict[str, str]:
    out, k = {}, 0
    for m in models:
        if m.label in PALETTE:
            out[m.label] = PALETTE[m.label]
        else:
            out[m.label] = FALLBACK[k % len(FALLBACK)]
            k += 1
    return out


def _save(fig, out_dir: str, name: str, formats) -> None:
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  {name}.{'/'.join(formats)}")


def _log_axis(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    decades = np.log10(max(hi, 1e-9)) - np.log10(max(lo, 1e-9))
    subs = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0) if decades <= 1.3 else (1.0, 2.0, 5.0) if decades <= 3.0 else (1.0,)
    a.set_major_locator(ticker.LogLocator(base=10.0, subs=subs, numticks=30))
    a.set_major_formatter(ticker.FormatStrFormatter("%g"))
    a.set_minor_locator(ticker.NullLocator())
    a.set_minor_formatter(ticker.NullFormatter())


def _grouped_bars(ax, groups, series: Dict[str, List[float]], colors, title, annotate=True):
    x = np.arange(len(groups))
    k = max(len(series), 1)
    width = 0.8 / k
    for i, (label, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - (k - 1) / 2) * width, vals, width, label=label, color=colors.get(label), alpha=0.9)
        if annotate:
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                                fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, groups)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0, top=ax.get_ylim()[1] * 1.6)
    _log_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)


def workloads_of(models: List[Model]) -> List[str]:
    seen = []
    for m in models:
        for w in m.workloads:
            if w not in seen:
                seen.append(w)
    return seen


# --------------------------------------------------------------------------- #
def plot_levels(models, out, formats, colors, annotate: bool = True):
    lt = level_table(models)
    ws = workloads_of(models)
    fig, axes = plt.subplots(1, len(ws), figsize=(4.6 * len(ws), 4.2), squeeze=False)
    for ax, w in zip(axes[0], ws):
        g = lt[lt.workload == w]
        series = {m.label: [float(g[g.model == m.label][f"q{k}"].iloc[0]) for k in LEVELS] for m in models if (g.model == m.label).any()}
        _grouped_bars(ax, LEVEL_LABELS, series, colors, f"{WORKLOAD_LABELS.get(w, w)} (n={int(g.n.iloc[0]):,})",
                      annotate=annotate)
    axes[0][0].set_ylabel("Q-error (log)")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Q-error by percentile level", y=1.02)
    fig.tight_layout()
    _save(fig, out, "qerror_levels", formats)


def plot_cdf(models, out, formats, colors):
    ws = workloads_of(models)
    fig, axes = plt.subplots(1, len(ws), figsize=(4.6 * len(ws), 4.0), squeeze=False)
    for ax, w in zip(axes[0], ws):
        for m in models:
            df = m.preds.get(w)
            if df is None:
                continue
            s = np.sort(qerror_of(df))
            ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", label=m.label, color=colors[m.label], lw=2)
        for k in LEVELS:
            ax.axhline(k / 100, color="grey", lw=0.6, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(left=1.0)
        _log_axis(ax, "x")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Q-error (log)")
        ax.set_title(WORKLOAD_LABELS.get(w, w))
    axes[0][0].set_ylabel("fraction of plans")
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Per-plan Q-error distribution (dotted: the 50 / 75 / 90 / 95 / 99 % levels)", y=1.02)
    fig.tight_layout()
    _save(fig, out, "qerror_cdf", formats)


def plot_epochs(models, out, formats, colors):
    et = epoch_table(models)
    if et.empty:
        print("  (no training logs; skipping qerror_epochs)")
        return
    cols = [f"val_q{k}" for k in LEVELS if f"val_q{k}" in et]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.9 * len(cols), 3.8), squeeze=False)
    for ax, c in zip(axes[0], cols):
        for i, m in enumerate(models):
            g = et[et.model == m.label]
            if not g.empty:
                ax.plot(g["epoch"], g[c], label=m.label, color=colors[m.label], marker=MARKERS[i % len(MARKERS)], ms=4, lw=2)
        ax.set_yscale("log")
        _log_axis(ax)
        ax.set_xlabel("epoch")
        ax.set_title(c.replace("val_q", "Q") + " (validation)")
    axes[0][0].set_ylabel("Q-error (log)")
    axes[0][0].legend()
    fig.suptitle("Validation Q-error per level during training", y=1.03)
    fig.tight_layout()
    _save(fig, out, "qerror_epochs", formats)


def plot_buckets(models, bt: pd.DataFrame, out, name, xlabel, formats, colors, order=None, show=(50, 90, 99)):
    if bt.empty:
        print(f"  (no data; skipping {name})")
        return
    ws = [w for w in workloads_of(models) if w in set(bt.workload)]
    fig, axes = plt.subplots(len(ws), len(show), figsize=(4.2 * len(show), 3.4 * len(ws)), squeeze=False)
    for r, w in enumerate(ws):
        g = bt[bt.workload == w]
        buckets = [b for b in (order if order is not None else sorted(g.bucket.unique(), key=lambda v: (str(type(v)), v))) if b in set(g.bucket)]
        ns = [int(g[g.bucket == b]["n"].max()) for b in buckets]
        for c, k in enumerate(show):
            ax = axes[r][c]
            series = {}
            for m in models:
                gm = g[g.model == m.label].set_index("bucket")
                if len(gm):
                    series[m.label] = [float(gm.loc[b, f"q{k}"]) if b in gm.index else np.nan for b in buckets]
            _grouped_bars(ax, [f"{b}\n(n={n})" for b, n in zip(buckets, ns)], series, colors,
                          f"{WORKLOAD_LABELS.get(w, w)} - Q{k}", annotate=False)
            if r == len(ws) - 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel("Q-error (log)")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, name, formats)


def plot_params(models, out, formats, colors):
    p = params_table(models)
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(p), 3.8))
    vals = p["total_params"].to_numpy(float)
    bars = ax.bar(p["model"], vals, color=[colors.get(m) for m in p["model"]], width=0.55)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v / 1e6:.2f} M", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=11)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} K"))
    ax.set_ylabel("parameters (log)")
    ax.set_ylim(top=vals.max() * 3)
    ax.grid(axis="x", visible=False)
    if len(vals) >= 2 and vals.min() > 0:
        ax.set_title(f"model size ({vals.max() / vals.min():.1f}x)")
    fig.tight_layout()
    _save(fig, out, "model_params", formats)


def plot_holdout(models, out, formats, colors):
    ht = holdout_table(models)
    if ht.empty:
        print("  (no holdout_joins in these models; skipping qerror_holdout)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), squeeze=False)
    for ax, plans in zip(axes[0], HOLDOUT_LABELS):
        g = ht[ht.plans == plans]
        series = {m.label: [float(g[g.model == m.label][f"q{k}"].iloc[0]) for k in LEVELS] for m in models if (g.model == m.label).any()}
        _grouped_bars(ax, LEVEL_LABELS, series, colors, f"val: {plans} (n={int(g.n.iloc[0]):,})")
    axes[0][0].set_ylabel("Q-error (log)")
    axes[0][0].legend(loc="upper left")
    hj = sorted(set().union(*(m.holdout_joins for m in models)))
    fig.suptitle("held out of training: " + ", ".join(hj), y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, out, "qerror_holdout", formats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qf,gnto", help="comma list of model dirs under results/exp1_qerror")
    ap.add_argument("--labels", default="", help="display names, e.g. gnto_joinid='GNTO (string id)'")
    ap.add_argument("--holdout", action="store_true", help="also draw the held-out-join figure")
    ap.add_argument("--name", default="", help="sub-directory under figures/ and tables/")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()
    found = discover(a.results)
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    missing = [n for n in names if n not in found and not os.path.isdir(n)]
    if missing:
        raise SystemExit(f"not found under {a.results}: {missing}; available: {sorted(found)}")
    models = load_models(names, a.results, parse_labels(a.labels))
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    out = os.path.join(a.results, "figures", a.name) if a.name else os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    write_tables(models, os.path.join(a.results, "tables", a.name) if a.name else os.path.join(a.results, "tables"),
                 holdout=a.holdout, title=f"Exp1 - {a.name or 'QueryFormer vs GNTO'}")
    colors = colors_for(models)
    print(f"figures -> {out}")
    plot_levels(models, out, formats, colors)
    plot_cdf(models, out, formats, colors)
    plot_epochs(models, out, formats, colors)
    plot_buckets(models, bucket_table(models, "n_joins"), out, "qerror_by_joins", "number of joins", formats, colors)
    plot_buckets(models, bucket_table(models, "actual_ms"), out, "qerror_by_latency", "actual latency", formats, colors, order=LATENCY_LABELS)
    plot_params(models, out, formats, colors)
    if a.holdout:
        plot_holdout(models, out, formats, colors)
    with open(os.path.join(out, "MODELS.txt"), "w") as f:
        f.write("\n".join(f"{m.name}\t{m.label}\t{m.dir}" for m in models) + "\n")


if __name__ == "__main__":
    main()
