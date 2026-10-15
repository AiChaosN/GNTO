"""Exp2 figures (cross-database generalisation) -> ``results/exp2_dace/figures/[<name>/]``.

* ``per_db_median``   per test database: median Q-error of every variant (error bars = std over seeds), one panel per split
* ``levels``          Q50 / Q75 / Q90 / Q95 / Q99 over all test plans pooled (seed-averaged), one panel per split
* ``cdf``             per-plan Q-error CDF over all test plans (first seed of every variant), one panel per split
* ``epochs``          validation median / mean Q-error vs epoch (seed-averaged)
* ``model_params``    parameter counts

Usage::

    python exp/exp2_dace/plot.py                          # dace vs gnto, every split found
    python exp/exp2_dace/plot.py --models dace,gnto --splits paper
    python exp/exp2_dace/plot.py --models gnto,gnto_L5 --name depth
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
from analyze import Run, avg_table, discover, epoch_table, label_of, params_table, per_db_table, select_runs, write_tables  # noqa: E402
from common import LEVELS, RESULTS, WORKLOADS, level_stats  # noqa: E402

PALETTE = {"DACE": "#e07b39", "GNTO": "#1f5fa8"}
FALLBACK = ["#9a9a9a", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def colors_for(variants: List[str]) -> Dict[str, str]:
    out, k = {}, 0
    for v in variants:
        lab = label_of(v)
        if lab in PALETTE:
            out[v] = PALETTE[lab]
        else:
            out[v] = FALLBACK[k % len(FALLBACK)]
            k += 1
    return out


def _save(fig, out, name, formats):
    for ext in formats:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=160)
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


def splits_of(runs: List[Run]) -> List[str]:
    return sorted({r.split for r in runs}, key=lambda s: (s != "paper", s))


def plot_per_db(runs, variants, out, formats, colors):
    pdb = per_db_table(runs)
    splits = splits_of(runs)
    fig, axes = plt.subplots(len(splits), 1, figsize=(11, 3.8 * len(splits)), squeeze=False)
    for ax, split in zip(axes[:, 0], splits):
        g = pdb[pdb.split == split]
        dbs = [d for d in WORKLOADS if d in set(g.db)]
        x = np.arange(len(dbs))
        width = 0.8 / max(len(variants), 1)
        for i, v in enumerate(variants):
            gv = g[g.variant == v].set_index("db")
            if gv.empty:
                continue
            vals = [float(gv.loc[d, "q50"]) if d in gv.index else np.nan for d in dbs]
            errs = [float(gv.loc[d, "q50_std"]) if d in gv.index else 0.0 for d in dbs]
            ax.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width, yerr=errs if any(errs) else None,
                   capsize=2, label=f"{label_of(v)} (avg {np.nanmean(vals):.2f})", color=colors[v], alpha=0.9)
            ax.axhline(np.nanmean(vals), color=colors[v], lw=1, ls="--", alpha=0.7)
        seeds = sorted({r.seed for r in runs if r.split == split})
        ax.set_xticks(x, dbs, rotation=20, ha="right")
        ax.set_ylim(bottom=1.0)
        ax.set_ylabel("median Q-error")
        ax.set_title(f"split {split}: test databases (seeds {', '.join(map(str, seeds))}; dashed = average over databases)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "per_db_median", formats)


def plot_levels(runs, variants, out, formats, colors):
    avg = avg_table(runs)
    splits = splits_of(runs)
    fig, axes = plt.subplots(1, len(splits), figsize=(4.8 * len(splits), 4.2), squeeze=False)
    labels = [f"Q{k}" for k in LEVELS]
    x = np.arange(len(LEVELS))
    for ax, split in zip(axes[0], splits):
        g = avg[avg.split == split]
        width = 0.8 / max(len(variants), 1)
        for i, v in enumerate(variants):
            gv = g[g.variant == v]
            if gv.empty or f"pooled_q{LEVELS[0]}" not in gv:
                continue
            vals = [float(gv[f"pooled_q{k}"].iloc[0]) for k in LEVELS]
            bars = ax.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width, label=label_of(v), color=colors[v], alpha=0.9)
            for b, val in zip(bars, vals):
                ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, val), ha="center", va="bottom", fontsize=8,
                            xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0, top=ax.get_ylim()[1] * 1.6)
        _log_axis(ax)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"split {split}: all test plans pooled")
    axes[0][0].set_ylabel("Q-error (log)")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "levels", formats)


def plot_cdf(runs, variants, out, formats, colors):
    splits = splits_of(runs)
    fig, axes = plt.subplots(1, len(splits), figsize=(4.8 * len(splits), 4.0), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for v in variants:
            rs = [r for r in runs if r.split == split and r.variant == v and r.preds]
            if not rs:
                continue
            r = min(rs, key=lambda r: r.seed)
            q = np.sort(np.concatenate([df["qerror"].to_numpy() for df in r.preds.values()]))
            ax.step(q, np.arange(1, len(q) + 1) / len(q), where="post", label=f"{label_of(v)} (seed {r.seed})", color=colors[v], lw=2)
        for k in LEVELS:
            ax.axhline(k / 100, color="grey", lw=0.6, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(left=1.0)
        _log_axis(ax, "x")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Q-error (log)")
        ax.set_title(f"split {split}: test plans")
    axes[0][0].set_ylabel("fraction of plans")
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out, "cdf", formats)


def plot_epochs(runs, variants, out, formats, colors):
    et = epoch_table(runs)
    if et.empty:
        print("  (no training logs; skipping epochs)")
        return
    splits = splits_of(runs)
    cols = [c for c in ("val_q50", "val_q_mean") if c in et]
    fig, axes = plt.subplots(len(splits), len(cols), figsize=(4.6 * len(cols), 3.6 * len(splits)), squeeze=False)
    for r_, split in enumerate(splits):
        for c_, col in enumerate(cols):
            ax = axes[r_][c_]
            for i, v in enumerate(variants):
                g = et[(et.split == split) & (et.variant == v)]
                if g.empty:
                    continue
                m = g.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
                ax.plot(m["epoch"], m["mean"], label=label_of(v), color=colors[v], marker=MARKERS[i % len(MARKERS)], ms=4, lw=2)
                if m["std"].notna().any():
                    ax.fill_between(m["epoch"], m["mean"] - m["std"].fillna(0), m["mean"] + m["std"].fillna(0), color=colors[v], alpha=0.15)
            ax.set_yscale("log")
            _log_axis(ax)
            ax.set_xlabel("epoch")
            ax.set_title(f"split {split}: validation {col.replace('val_q', 'Q').replace('Q_mean', 'mean Q-error')}")
            if c_ == 0:
                ax.set_ylabel("Q-error (log)")
    axes[0][0].legend()
    fig.tight_layout()
    _save(fig, out, "epochs", formats)


def plot_params(runs, variants, out, formats, colors):
    p = params_table(runs)
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(p), 3.8))
    vals = p["total_params"].to_numpy(float)
    bars = ax.bar(p["model"], vals, color=[colors[v] for v in p["variant"]], width=0.55)
    for b, v, tr in zip(bars, vals, p["trainable_params"]):
        ax.annotate(f"{v / 1e3:.1f} K\n({tr / 1e3:.1f} K trainable)", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} K"))
    ax.set_ylabel("parameters (log)")
    ax.set_ylim(top=vals.max() * 4)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, "model_params", formats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="dace,gnto", help="comma list of variants (first = reference)")
    ap.add_argument("--splits", default="", help="comma list of splits (default: all found)")
    ap.add_argument("--name", default="", help="sub-directory under figures/ and tables/")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--include-smoke", action="store_true", help="also consider *_smoke runs (variant '<model>_smoke')")
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()
    variants = [v.strip() for v in a.models.split(",") if v.strip()]
    splits = [s.strip() for s in a.splits.split(",") if s.strip()] or None
    runs = select_runs(variants, splits, a.results, a.include_smoke)
    if not runs:
        raise SystemExit(f"no runs for {variants} / {splits}; found: {sorted(discover(a.results))}")
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    out = os.path.join(a.results, "figures", a.name) if a.name else os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    write_tables(runs, os.path.join(a.results, "tables", a.name) if a.name else os.path.join(a.results, "tables"),
                 variants[0], title=f"Exp2 - {a.name or 'DACE vs GNTO'}")
    colors = colors_for(variants)
    print(f"figures -> {out}")
    plot_per_db(runs, variants, out, formats, colors)
    plot_levels(runs, variants, out, formats, colors)
    plot_cdf(runs, variants, out, formats, colors)
    plot_epochs(runs, variants, out, formats, colors)
    plot_params(runs, variants, out, formats, colors)
    with open(os.path.join(out, "RUNS.txt"), "w") as f:
        f.write("\n".join(os.path.basename(r.dir) for r in runs) + "\n")


if __name__ == "__main__":
    main()
