"""Exp4 figures: the per-group ablation figures and the depth curves.

Two kinds of output, both under ``results/exp4_ablation/figures/``:

* **groups** - for each ablation group (``encoder``, ``edges``, ``norm``, ``treeattn``, ``depth``,
  ``res``, ``joins``, ``joins_holdout``) the standard Exp1 figure set (Q-error levels, CDF, epochs,
  by joins, by latency, parameters) plus its tables, drawn by Exp1's plotting code on this results
  root so there is no second copy of it here.
* **curves** - ``ablation_res_depth_curves``: three panels (val Q50 / Q99 / mean) of the
  selection-free statistic (mean of the val evaluations at epochs 80 / 90 / 99) against depth, per
  width, with and without the residual.  Needs the tables written by ``analyze.py``.

    python exp/exp4_ablation/plot.py                     # every group + the curves
    python exp/exp4_ablation/plot.py --groups encoder,depth
    python exp/exp4_ablation/plot.py --groups none       # only the curves
    python exp/exp4_ablation/plot.py --models gnto_w64_L16,gnto --name my_ablation   # ad-hoc group
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.join(HERE, "..", "exp1_qerror")
sys.path.insert(0, EXP1)   # the ablations run Exp1's protocol, so its common/analyze come first
from common import GNTO_PATH  # noqa: E402
import analyze as exp1_analyze  # noqa: E402

RESULTS = os.path.join(GNTO_PATH, "results", "exp4_ablation")


def _exp1_plot():
    """Exp1's plot.py, loaded by path (this module is called plot.py too)."""
    spec = importlib.util.spec_from_file_location("exp1_plot", os.path.join(EXP1, "plot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# group -> (model dirs, display labels, held-out-join figure?)
GROUPS: "dict[str, tuple[list, dict, bool]]" = {
    "encoder": (["gnto", "gnto_gcn", "gnto_tcnn"],
                {"gnto": "GAT (GATv2)", "gnto_gcn": "GNN (GCN)", "gnto_tcnn": "TCNN"}, False),
    "edges": (["gnto", "gnto_bidir", "gnto_reverse", "gnto_gcn", "gnto_gcn_bidir", "gnto_tcnn"],
              {"gnto": "GAT parent->child", "gnto_bidir": "GAT bidirectional", "gnto_reverse": "GAT child->parent",
               "gnto_gcn": "GCN parent->child", "gnto_gcn_bidir": "GCN bidirectional", "gnto_tcnn": "TCNN bottom-up"}, False),
    "norm": (["gnto", "gnto_nonorm", "gnto_sumpool", "gnto_nonorm_sum", "gnto_gcn", "gnto_tcnn"],
             {"gnto": "GNTO (softmax attn, mean pool)", "gnto_nonorm": "sigmoid attn", "gnto_sumpool": "sum pool",
              "gnto_nonorm_sum": "sigmoid attn + sum pool"}, False),
    "treeattn": (["gnto", "gnto_gatv2_pos", "gnto_gcn", "gnto_tcnn", "gnto_tcnn_attn", "gnto_tcnn_attn_softmax"],
                 {"gnto": "GNTO", "gnto_gatv2_pos": "GAT + child position", "gnto_tcnn_attn": "TCNN + gated slots",
                  "gnto_tcnn_attn_softmax": "TCNN + softmax slots"}, False),
    # keep every group at <= 6 models: past that the generic figures cycle colours and the bar
    # labels collide.  Depth across many configurations is shown by the curves figure instead.
    "depth": (["gnto", "gnto_L5", "gnto_L7"],
              {"gnto": "GAT L3 (default)", "gnto_L5": "GAT L5", "gnto_L7": "GAT L7"}, False),
    "depth_convs": (["gnto", "gnto_L7", "gnto_gcn", "gnto_gcn_L7", "gnto_tcnn", "gnto_tcnn_L7"],
                    {"gnto": "GAT L3", "gnto_L7": "GAT L7", "gnto_gcn": "GCN L3", "gnto_gcn_L7": "GCN L7",
                     "gnto_tcnn": "TCNN L3", "gnto_tcnn_L7": "TCNN L7"}, False),
    "res": (["gnto", "gnto_nores", "gnto_w64_L5", "gnto_w64_L5_nores", "gnto_w64_L24", "gnto_w64_L24_nores"],
            {"gnto": "default L3", "gnto_nores": "default L3, no res", "gnto_w64_L5": "w64 L5",
             "gnto_w64_L5_nores": "w64 L5, no res", "gnto_w64_L24": "w64 L24", "gnto_w64_L24_nores": "w64 L24, no res"}, False),
    "width": (["gnto", "gnto_w64_L24", "gnto_w32_L24", "gnto_w32_L48", "gnto_w16_L64"],
              {"gnto": "default (256/128/64) L3", "gnto_w64_L24": "w64 L24", "gnto_w32_L24": "w32 L24",
               "gnto_w32_L48": "w32 L48", "gnto_w16_L64": "w16 L64"}, False),
    "joins": (["gnto_joinid", "gnto"], {"gnto_joinid": "GNTO (string-id join)", "gnto": "GNTO"}, False),
    "joins_holdout": (["gnto_joinid_holdout", "gnto_holdout", "qf"],
                      {"gnto_joinid_holdout": "GNTO (string-id join)", "gnto_holdout": "GNTO", "qf": "QueryFormer"}, True),
}
# the sub-directory each group writes to
DIRNAME = {"encoder": "ablation_encoder", "edges": "ablation_edges", "norm": "ablation_norm",
           "treeattn": "ablation_tree_attn", "depth": "ablation_depth", "depth_convs": "ablation_depth_by_conv",
           "res": "ablation_residual", "width": "ablation_width",
           "joins": "ablation_join_encoding", "joins_holdout": "ablation_join_encoding_holdout"}


def plot_group(results: str, models, labels, name: str, formats, holdout: bool = False) -> None:
    """Exp1's figure set + tables for one list of model dirs, written under ``<name>/``."""
    ep = _exp1_plot()
    found = exp1_analyze.discover(results)
    missing = [m for m in models if m not in found]
    if missing:
        print(f"[skip] {name}: missing {missing}")
        return
    loaded = exp1_analyze.load_models(models, results, labels)
    out = os.path.join(results, "figures", name)
    os.makedirs(out, exist_ok=True)
    exp1_analyze.write_tables(loaded, os.path.join(results, "tables", name), holdout=holdout,
                              title=f"Exp4 - {name}")
    colors = {m.label: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(loaded)}
    annotate = len(loaded) <= 3          # past three series the bar labels collide
    ep.plot_levels(loaded, out, formats, colors, annotate=annotate)
    ep.plot_cdf(loaded, out, formats, colors)
    ep.plot_epochs(loaded, out, formats, colors)
    ep.plot_buckets(loaded, exp1_analyze.bucket_table(loaded, "n_joins"), out, "qerror_by_joins",
                    "number of joins", formats, colors)
    ep.plot_buckets(loaded, exp1_analyze.bucket_table(loaded, "actual_ms"), out, "qerror_by_latency",
                    "actual latency", formats, colors, order=exp1_analyze.LATENCY_LABELS)
    ep.plot_params(loaded, out, formats, colors)
    if holdout:
        ep.plot_holdout(loaded, out, formats, colors)
    with open(os.path.join(out, "MODELS.txt"), "w") as f:
        f.write("\n".join(f"{m.name}\t{m.label}\t{m.dir}" for m in loaded) + "\n")
    print(f"figures + tables -> {out}")

TICKS = [3, 5, 7, 9, 12, 16, 24, 32, 48, 64]
# categorical palette, slots 1-3 (colour-blind safe on the light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# categorical palette in slot order - 8 distinct hues, no cycling in a group
CATEGORICAL = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def plot_curves(a) -> None:
    """The depth x width x residual curves (needs analyze.py's all_runs.csv)."""
    table = os.path.join(a.results, "tables", a.curves_name, "all_runs.csv")
    if not os.path.exists(table):
        raise SystemExit(f"{table} missing - run exp/exp4_ablation/analyze.py first")
    OUT = os.path.join(a.results, "figures", a.curves_name)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(table)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})

    s42 = df[df.seed == 42]
    w64r = s42[(s42.width == "64") & (s42.residual == "identity")].sort_values("layers")
    w64n = s42[(s42.width == "64") & (s42.residual == "none")].sort_values("layers")
    w32 = s42[s42.width == "32"].sort_values("layers")
    w16 = s42[s42.width == "16"]
    dflt = df[df.run.isin(["gnto", "gnto_s7", "gnto_s123"])]
    l5r = df[(df.width == "64") & (df.layers == 5) & (df.residual == "identity")]
    l5n = df[(df.width == "64") & (df.layers == 5) & (df.residual == "none")]

    panels = [("last3_val_q50", "val Q50 (median Q-error)"), ("last3_val_q99", "val Q99"), ("last3_val_mean", "val mean Q-error")]
    TICKS = [3, 5, 7, 9, 12, 16, 24, 32, 48, 64]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, (m, title) in zip(axes, panels):
        in_scale = pd.concat([w64r[m], w64n[w64n.layers < 24][m], w32[m], w16[m], dflt[m], l5r[m], l5n[m]])
        lo, hi = in_scale.min(), in_scale.max(); pad = (hi - lo) * 0.12
        ymin, ymax = lo - pad, hi + pad * 1.6

        # default GNTO reference: 3-seed mean +- std
        mu, sd = dflt[m].mean(), dflt[m].std(ddof=1)
        ax.axhspan(mu - sd, mu + sd, color=MUTED, alpha=0.12, lw=0, zorder=1)
        ax.axhline(mu, color=MUTED, lw=1.2, zorder=2)

        ring = dict(markeredgecolor=SURFACE, markeredgewidth=1.6)
        ax.plot(w64r.layers, w64r[m], color=BLUE, lw=2, marker="o", ms=7.5, solid_capstyle="round", zorder=5, **ring)
        w64n_in = w64n[w64n.layers < 24]
        ax.plot(w64n_in.layers, w64n_in[m], color=ORANGE, lw=2, marker="s", ms=7, solid_capstyle="round", zorder=4, **ring)
        ax.plot(w32.layers, w32[m], color=AQUA, lw=2, marker="^", ms=8.5, solid_capstyle="round", zorder=5, **ring)
        ax.plot(w16.layers, w16[m], color=AQUA, lw=0, marker="D", ms=7.5, markerfacecolor=SURFACE, markeredgecolor=AQUA, markeredgewidth=2, zorder=6)

        # 3-seed spread at L5 (mean +- std), drawn just beside the seed-42 points
        for sub, col, dx in [(l5r, BLUE, 0.93), (l5n, ORANGE, 1.075)]:
            ax.errorbar([5 * dx], [sub[m].mean()], yerr=[sub[m].std(ddof=1)], color=col, lw=1.4, capsize=3, capthick=1.2, alpha=0.85, zorder=3)

        # off-scale: 24-layer no-residual run
        off = w64n[w64n.layers == 24][m].iloc[0]
        ax.set_ylim(ymin, ymax)
        y_edge = ymax - (ymax - ymin) * 0.035
        ax.plot([9, 24], [w64n_in[m].iloc[-1], y_edge], color=ORANGE, lw=1.2, alpha=0.45, zorder=3, clip_on=False)
        ax.plot([24], [y_edge], marker="^", color=ORANGE, ms=9, markeredgecolor=SURFACE, markeredgewidth=1.6, clip_on=False, zorder=7)
        ax.annotate(f"L24 no residual: {off:.2f} (off scale)", xy=(24, ymax - (ymax - ymin) * 0.035), xytext=(-8, -2),
                    textcoords="offset points", ha="right", va="center", fontsize=9, color=INK2)

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(TICKS)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in TICKS]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(2.6, 76)
        ax.set_title(title, loc="left", color=INK, pad=8)
        ax.set_xlabel("GAT layers (log scale)")
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
        ax.tick_params(length=0)

    # selective direct labels: series ends on the last panel only; reference label on the first
    ax = axes[2]
    ax.annotate("w64 residual", xy=(16, w64r[w64r.layers == 16].last3_val_mean.iloc[0]), xytext=(7, 5), textcoords="offset points", fontsize=9, color=INK2)
    ax.annotate("w32 residual", xy=(48, w32.last3_val_mean.iloc[-1]), xytext=(-2, -15), textcoords="offset points", fontsize=9, color=INK2, ha="center")
    ax.annotate("w16", xy=(64, w16.last3_val_mean.iloc[0]), xytext=(0, 9), textcoords="offset points", fontsize=9, color=INK2, ha="center")
    ax.annotate("w64 no residual", xy=(9, w64n[w64n.layers == 9].last3_val_mean.iloc[0]), xytext=(0, -15), textcoords="offset points", fontsize=9, color=INK2, ha="center")
    mu0 = dflt.last3_val_q50.mean()
    axes[0].annotate("default GNTO (302K, 3 seeds ± std)", xy=(2.75, mu0), xytext=(0, -13), textcoords="offset points", fontsize=9, color=INK2)

    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, label="width 64, identity residual"),
               Line2D([], [], color=ORANGE, lw=2, marker="s", ms=6.5, markeredgecolor=SURFACE, label="width 64, no residual"),
               Line2D([], [], color=AQUA, lw=2, marker="^", ms=8, markeredgecolor=SURFACE, label="width 32, identity residual"),
               Line2D([], [], color=AQUA, lw=0, marker="D", ms=7, markerfacecolor=SURFACE, markeredgecolor=AQUA, markeredgewidth=2, label="width 16, identity residual"),
               Line2D([], [], color=MUTED, lw=1.2, label="default GNTO (256/128/64, L3), 3-seed mean ± std")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.005), ncol=5, frameon=False, fontsize=10, handlelength=2.2, columnspacing=1.6)
    fig.text(0.005, -0.035, "Val split, 10,000 plans; each value = mean of the evaluations at epochs 80 / 90 / 99 (no checkpoint selection); single seed (42) except the "
             "error bars at L5 = mean ± 1 std over seeds 42 / 7 / 123. Lower is better.", fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in a.formats.split(","):
        fig.savefig(os.path.join(OUT, f"depth_width_residual.{ext}"), bbox_inches="tight", dpi=160)
    print("->", OUT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--groups", default="all", help=f"comma list of {', '.join(GROUPS)}, or 'all' / 'none'")
    ap.add_argument("--models", default="", help="ad-hoc group: comma list of model dirs (needs --name)")
    ap.add_argument("--labels", default="", help="ad-hoc group: display names, e.g. gnto_w64_L16='16 layers'")
    ap.add_argument("--name", default="", help="ad-hoc group: sub-directory under figures/ and tables/")
    ap.add_argument("--holdout", action="store_true", help="ad-hoc group: also draw the held-out-join figure")
    ap.add_argument("--curves", default="auto", choices=["auto", "yes", "no"], help="draw the depth curves")
    ap.add_argument("--curves-name", default="ablation_res_depth_curves")
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]

    if a.models:
        if not a.name:
            raise SystemExit("--models needs --name")
        ep = _exp1_plot()
        plot_group(a.results, [m.strip() for m in a.models.split(",") if m.strip()],
                   ep.parse_labels(a.labels), a.name, formats, a.holdout)
    else:
        names = list(GROUPS) if a.groups == "all" else ([] if a.groups == "none" else
                                                        [g.strip() for g in a.groups.split(",") if g.strip()])
        unknown = [g for g in names if g not in GROUPS]
        if unknown:
            raise SystemExit(f"unknown group(s) {unknown}; known: {', '.join(GROUPS)}")
        for g in names:
            models, labels, holdout = GROUPS[g]
            plot_group(a.results, models, labels, DIRNAME[g], formats, holdout)

    if a.curves == "yes" or (a.curves == "auto" and not a.models):
        plot_curves(a)


if __name__ == "__main__":
    main()
