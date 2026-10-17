"""End-to-end plan selection with the operator's own cost (TLR inputs): the online replay and the offline
held-out-template selection of the own-cost inputs against the inputs they replace.  Read-only on every run directory;
writes three new tables and never the existing ones (online_*.csv of analyze_online.py, variants.csv,
selection_local.csv).

Inputs.  TCR = operator type + cumulative Total Cost + Plan Rows (``dace``).  TLR = type + Plan Rows + the operator's
own cost (its Total Cost minus its inputs' Total Cost, median / IQR scaled, no cumulative cost): offline ``resid``
(analysis/attn/run_kfold_cost.py --cost resid) and ``residg`` (--cost residg, the own cost with the grouping nodes of
ordered-binary-v1 resolved), online ``gnto_tlr`` (resolved).  PG-stat = ``pg`` (adapters/pg_features.py); PG-stat +
own cost: offline ``pgl`` (analysis/tlr/run_kfold_local.py) and ``pglg`` (--resolve-groups), online ``gnto_pgl``
(resolved).  Only TPC-H has operators with more than two inputs (53 of 643 plans): on JOB resid / pgl already are the
resolved inputs, and there are no residg / pglg runs.

Online (analysis/online/run_online.py, synchronous retraining, runs with summary.json only): load / per_query /
cluster_ci and the windows of analysis/online/analyze_online.py, converged = the last ten passes, learning = passes
1..end.  Per selector: G over PostgreSQL's default plans, the range of the per-seed converged G, the first pass from
which a seed stays at or above the default plans (median and max over the seeds that do), and better / worse / tie
of the per-query converged speed-up.  Contrasts per query, ratio > 1: the first selector's plans are faster.

Offline (exp/exp3_e2e/run_kfold.py, five held-out template folds, seeds 42 / 7 / 123, runs with picks.csv and
summary.json): the convention of analysis/attn/analyze_selection_variants.py and analysis/tlr/analyze_selection.py,
per query the seed mean of log(default / pick), G = geometric mean over queries, ratio = paired per-query ratio
against the run with the same encoder, objective and workload and the replaced inputs.  Rows: resid / residg vs TCR
(gatv2, none, gcn, tcnn), pgl / pglg vs PG-stat (gatv2, none, tcnn), and on TPC-H residg vs resid and pglg vs pgl
(what resolving the grouping nodes changes).

Counts use a 5 % band (better / worse / tie against the default plans, faster / slower in a contrast).  Intervals:
95 %, templates resampled (20,000 draws), a fresh numpy default_rng(20261015) per table.  A row whose runs are
missing still consumes the draws it would use, so no interval depends on which other runs have finished.  Missing or
unfinished runs are skipped with a note.

    python analysis/tlr/analyze_e2e_tlr.py     # -> results/analysis/online_20261005/tables/online_tlr_<wl>_{summary,contrasts}.csv
                                               #    results/analysis/tlr_20261008/selection/tables/e2e_tlr_offline.csv
    python analysis/tlr/analyze_e2e_tlr.py --part offline --out-offline /tmp/x     # one part, elsewhere
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "online"))
import analyze_online as ao                                      # noqa: E402
from protocol import template_key                                # noqa: E402

ao.MODELS = ["gnto", "gnto_pgl", "gnto_tcr", "gnto_tlr", "bao"]  # ao.load reads the module-level list
ONLINE = os.path.join(ROOT, "results", "analysis", "online_20261005")
SEL_TLR = os.path.join(ROOT, "results", "analysis", "attn_20261008", "selection")
SEL_PGL = os.path.join(ROOT, "results", "analysis", "tlr_20261008", "selection")
OLD = os.path.join(ROOT, "results", "main_runs")
WORKLOADS = ("job112", "tpch52")
PAIRS = (("gnto_tlr", "gnto_tcr"), ("gnto_pgl", "gnto"), ("gnto_tlr", "bao"), ("gnto_pgl", "bao"),
         ("gnto_tlr", "gnto"), ("gnto_pgl", "gnto_tlr"), ("gnto", "bao"))
TLR_CONVS, PGL_CONVS = ("gatv2", "none", "gcn", "tcnn"), ("gatv2", "none", "tcnn")
# (arm, base, encoders, workloads): offline rows in table order
OFFLINE = (("resid", "dace", TLR_CONVS, WORKLOADS), ("residg", "dace", TLR_CONVS, ("tpch52",)),
           ("residg", "resid", TLR_CONVS, ("tpch52",)), ("pgl", "pg", PGL_CONVS, WORKLOADS),
           ("pglg", "pg", PGL_CONVS, ("tpch52",)), ("pglg", "pgl", PGL_CONVS, ("tpch52",)))
SEED = 20261015
BAND = float(np.log(1.05))


class Boot:
    """The template bootstrap of one table: a fresh generator, rows in a fixed order.  ``skip`` makes the draws of a
    row that cannot be computed (ao.cluster_ci draws n_boot x n_templates integers per call), so the other rows keep
    their intervals when that row's runs finish."""

    def __init__(self, n_boot: int, n_templates: dict):
        self.rng, self.n_boot, self.n_templates = np.random.default_rng(SEED), n_boot, n_templates

    def ci(self, values: pd.Series) -> tuple:
        return ao.cluster_ci(values, self.n_boot, self.rng)

    def skip(self, wl: str, calls: int) -> None:
        k = self.n_templates.get(wl)
        for _ in range(calls if k else 0):
            self.rng.integers(k, size=(self.n_boot, k))


def n_templates(query_ids) -> int:
    return len({template_key(q) for q in query_ids})


def counts(v: pd.Series) -> tuple:
    """Queries above / below / within the 5 % band of log ratios v."""
    up, down = int((v > BAND).sum()), int((v < -BAND).sum())
    return up, down, len(v) - up - down


def ci(g: float, lo: float, hi: float) -> str:
    return f"{g:.3f} [{lo:.3f}, {hi:.3f}]"


def n_lines(path: str) -> int:
    try:
        with open(path) as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


# ----------------------------------------------------------------------------------------------------------- online
def online(a, wl: str) -> None:
    runs = ao.load(a.online_root, wl, "sync")                     # complete runs (summary.json) only
    print(f"\n== online replay, {wl}")
    for m in ao.MODELS:
        left = [f"s{s} ({n_lines(os.path.join(a.online_root, wl, f'{m}_s{s}', 'passes.csv'))} passes, no summary.json)"
                if os.path.isdir(os.path.join(a.online_root, wl, f"{m}_s{s}")) else f"s{s} (no run)"
                for s in a.online_seeds if s not in runs.get(m, {})]
        if left:
            print(f"  note: {m}: skipped {', '.join(left)}")
    if not runs:
        print("  note: no complete run, nothing written")
        return
    lens = sorted({len(ps) for by in runs.values() for ps, _ in by.values()})
    n_pass = lens[0]
    if len(lens) > 1:
        print(f"  note: the runs have {lens} passes, all are cut to {n_pass} (as analyze_online.py does)")
    windows = {"converged": (n_pass - a.converged, n_pass), "learning": (1, n_pass)}
    nt = {wl: n_templates(set().union(*(set(ch.query_id) for by in runs.values() for _, ch in by.values())))}
    print(f"  {n_pass} passes; converged = passes {n_pass - a.converged}-{n_pass - 1}, learning = 1-{n_pass - 1}; "
          f"G over the default plans, 95 % intervals over {nt[wl]} templates")

    boot, summary = Boot(a.n_boot, nt), []
    for m in ao.MODELS:
        if m not in runs:
            boot.skip(wl, len(windows))
            continue
        by = runs[m]
        chs = [ch for _, ch in by.values()]
        g = np.array([ps.geo_speedup.to_numpy()[:n_pass] for ps, _ in by.values()])        # per seed and pass
        rec = dict(workload=wl, model=m, n_seeds=len(by), seeds=" ".join(str(s) for s in sorted(by)), n_passes=n_pass)
        for w, (lo, hi) in windows.items():
            v = ao.per_query(chs, lo, hi, "default_ms") - ao.per_query(chs, lo, hi, "ms")
            geo, l, h = boot.ci(v)
            per_seed = np.exp(np.log(g[:, lo:hi]).mean(axis=1))
            rec.update({f"{w}_passes": f"{lo}-{hi - 1}", f"{w}_G": geo, f"{w}_lo": l, f"{w}_hi": h,
                        f"{w}_seed_min": float(per_seed.min()), f"{w}_seed_max": float(per_seed.max())})
            if w == "converged":
                rec["better"], rec["worse"], rec["tie"] = counts(v)
        # first pass from which every later pass is at least as fast as the default plans, per seed
        stable = np.array([next((p for p in range(1, n_pass) if (row[p:] >= 1.0).all()), np.nan) for row in g], float)
        ok = stable[~np.isnan(stable)]
        rec.update(stable_pass_median=float(np.median(ok)) if len(ok) else np.nan,
                   stable_pass_max=float(ok.max()) if len(ok) else np.nan, stable_seeds=len(ok))
        best = ao.per_query(chs, 0, n_pass, "default_ms") - ao.per_query(chs, 0, n_pass, "best_ms")
        rec["oracle_G"] = float(np.exp(best.mean()))
        summary.append(rec)

    boot, contrasts, skipped = Boot(a.n_boot, nt), [], []
    for x, y in PAIRS:
        if x not in runs or y not in runs:
            boot.skip(wl, len(windows))
            skipped.append(f"{x} vs {y}")
            continue
        cx, cy = [ch for _, ch in runs[x].values()], [ch for _, ch in runs[y].values()]
        for w, (lo, hi) in windows.items():
            v = ao.per_query(cy, lo, hi) - ao.per_query(cx, lo, hi)              # > 0: the plans of x are faster
            r, l, h = boot.ci(v)
            faster, slower, tie = counts(v)
            contrasts.append(dict(workload=wl, contrast=f"{x}_vs_{y}", window=w, passes=f"{lo}-{hi - 1}",
                                  seeds_x=len(runs[x]), seeds_y=len(runs[y]), ratio=r, lo=l, hi=h,
                                  queries_faster=faster, queries_slower=slower, queries_tie=tie))
    if skipped:
        print(f"  note: contrasts skipped, a selector without complete runs: {', '.join(skipped)}")

    out = a.out_online or os.path.join(a.online_root, "tables")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(out, f"online_tlr_{wl}_summary.csv"), index=False)
    pd.DataFrame(contrasts).to_csv(os.path.join(out, f"online_tlr_{wl}_contrasts.csv"), index=False)
    print(pd.DataFrame([{"model": r["model"], "seeds": r["n_seeds"],
                         "converged G": ci(r["converged_G"], r["converged_lo"], r["converged_hi"]),
                         "seed range": f"{r['converged_seed_min']:.3f}-{r['converged_seed_max']:.3f}",
                         "learning G": ci(r["learning_G"], r["learning_lo"], r["learning_hi"]),
                         ">= default from pass (med/max, seeds)":
                             f"{r['stable_pass_median']:g} / {r['stable_pass_max']:g} ({r['stable_seeds']}/{r['n_seeds']})",
                         "better/worse/tie": f"{r['better']}/{r['worse']}/{r['tie']}"} for r in summary]).to_string(index=False))
    if contrasts:
        print(pd.DataFrame([{"contrast": c["contrast"].replace("_vs_", " vs "), "window": c["window"],
                             "seeds": f"{c['seeds_x']}/{c['seeds_y']}", "ratio": ci(c["ratio"], c["lo"], c["hi"]),
                             "faster/slower": f"{c['queries_faster']}/{c['queries_slower']}"} for c in contrasts]
                           ).to_string(index=False))
    print(f"  -> {out}/online_tlr_{wl}_{{summary,contrasts}}.csv")


# ---------------------------------------------------------------------------------------------------------- offline
def offline_series(root: str, wl: str, tag: str, seeds) -> tuple:
    """Seed mean of log(default / pick) per query over the complete runs <root>/<wl>/<tag>_s<seed>, their seeds, and
    the expected seeds that are skipped (unfinished: a run directory without picks.csv + summary.json)."""
    parts, done, unfinished = [], [], []
    for d in sorted(glob.glob(os.path.join(root, wl, f"{tag}_s*"))):
        m = re.fullmatch(re.escape(tag) + r"_s(\d+)", os.path.basename(d))
        if m is None or not os.path.isdir(d):
            continue                                             # the launch logs <tag>_s<seed>.log
        if not (os.path.exists(os.path.join(d, "picks.csv")) and os.path.exists(os.path.join(d, "summary.json"))):
            unfinished.append(int(m.group(1)))
            continue
        df = pd.read_csv(os.path.join(d, "picks.csv")).set_index("query_id")
        parts.append(np.log(df.pg_default_ms / df.pick_ms))
        done.append(int(m.group(1)))
    left = [f"s{s} unfinished" for s in sorted(unfinished)] + [f"s{s} no run" for s in seeds if s not in done + unfinished]
    return (pd.concat(parts, axis=1).mean(axis=1) if parts else None), sorted(done), left


def offline(a) -> None:
    roots = {"resid": a.sel_tlr, "residg": a.sel_tlr, "pgl": a.sel_pgl, "pglg": a.sel_pgl, "dace": a.base, "pg": a.base}
    specs = [(wl, arm, base, conv, rank) for wl in WORKLOADS for arm, base, convs, wls in OFFLINE if wl in wls
             for conv in convs for rank in (0, 1)]
    print("\n== offline selection, held-out template folds (G over the default plans, 95 % template intervals)")
    print("  job112 has no residg / pglg rows: no JOB plan has an operator with more than two inputs, so resid / pgl "
          "already are the resolved inputs there")
    series = {}
    for wl, arm, base, conv, rank in specs:                      # load first: the template counts precede any draw
        for inp in (arm, base):
            if (wl, inp, conv, rank) not in series:
                series[wl, inp, conv, rank] = offline_series(roots[inp], wl, f"{inp}_{conv}_rank{rank}", a.seeds)
    for (wl, inp, conv, rank), (s, done, left) in series.items():
        if s is not None and left:                               # the row is computed from fewer seeds
            print(f"  note: {wl} {inp}_{conv}_rank{rank}: seeds {' '.join(map(str, done))} only, skipped {', '.join(left)}")
    nt = {wl: n_templates(set().union(*(set(s.index) for (w, *_), (s, *_) in series.items() if w == wl and s is not None)))
          for wl in WORKLOADS}
    boot, rows, skipped = Boot(a.n_boot, nt), [], {}
    for wl, arm, base, conv, rank in specs:
        (new, sn, _), (old, so, _) = series[wl, arm, conv, rank], series[wl, base, conv, rank]
        if new is None or old is None:
            boot.skip(wl, 3)
            missing = [inp for inp, s in ((arm, new), (base, old)) if s is None]
            skipped.setdefault((wl, arm, base), []).append((f"{conv}/{rank}", missing,
                                                            [x for inp in missing for x in series[wl, inp, conv, rank][2]]))
            continue
        g, lo, hi = boot.ci(new)
        g0, lo0, hi0 = boot.ci(old)
        r, rlo, rhi = boot.ci(new - old)                         # > 1: the arm selects faster plans than the base
        vs_default, base_vs_default, vs_base = counts(new), counts(old), counts(new - old)
        rows.append(dict(workload=wl, arm=arm, base=base, conv=conv, rank=rank, n_seeds=len(sn),
                         seeds=" ".join(map(str, sn)), n_seeds_base=len(so), seeds_base=" ".join(map(str, so)),
                         G=g, lo=lo, hi=hi, **dict(zip(("better", "worse", "tie"), vs_default)),
                         G_base=g0, base_lo=lo0, base_hi=hi0,
                         **dict(zip(("base_better", "base_worse", "base_tie"), base_vs_default)),
                         ratio=r, ratio_lo=rlo, ratio_hi=rhi,
                         **dict(zip(("queries_faster", "queries_slower", "queries_tie"), vs_base))))
    for (wl, arm, base), what in skipped.items():
        left = [x for _, _, ls in what for x in ls]
        print(f"  note: {wl} {arm} vs {base}: {len(what)} row(s) skipped ({' '.join(cr for cr, _, _ in what)}, conv/rank), "
              f"no complete run of {' / '.join(sorted({i for _, ms, _ in what for i in ms}))}: "
              f"{sum('unfinished' in x for x in left)} seed(s) unfinished, {sum('no run' in x for x in left)} without a run")
    if not rows:
        print("  note: no row could be computed, nothing written")
        return
    os.makedirs(a.out_offline, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_offline, "e2e_tlr_offline.csv"), index=False)
    print(pd.DataFrame([{"workload": r["workload"], "arm vs base": f"{r['arm']} vs {r['base']}", "conv": r["conv"],
                         "rank": r["rank"], "seeds": f"{r['n_seeds']}/{r['n_seeds_base']}", "G": ci(r["G"], r["lo"], r["hi"]),
                         "G base": ci(r["G_base"], r["base_lo"], r["base_hi"]),
                         "ratio": ci(r["ratio"], r["ratio_lo"], r["ratio_hi"]),
                         "faster/slower": f"{r['queries_faster']}/{r['queries_slower']}",
                         "b/w/t": f"{r['better']}/{r['worse']}/{r['tie']}",
                         "b/w/t base": f"{r['base_better']}/{r['base_worse']}/{r['base_tie']}"} for r in rows]).to_string(index=False))
    print(f"  -> {a.out_offline}/e2e_tlr_offline.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["all", "online", "offline"])
    ap.add_argument("--online-root", default=ONLINE, help="runs <workload>/<model>_s<seed> of run_online.py")
    ap.add_argument("--sel-tlr", default=SEL_TLR, help="offline runs <workload>/{resid,residg}_<conv>_rank<r>_s<seed>")
    ap.add_argument("--sel-pgl", default=SEL_PGL, help="offline runs <workload>/{pgl,pglg}_<conv>_rank<r>_s<seed>")
    ap.add_argument("--base", default=OLD, help="main offline runs, <workload>/{dace,pg}_<conv>_rank<r>_s<seed>")
    ap.add_argument("--out-online", help="directory of online_tlr_<wl>_*.csv (default <online-root>/tables)")
    ap.add_argument("--out-offline", help="directory of e2e_tlr_offline.csv (default <sel-pgl>/tables)")
    ap.add_argument("--converged", type=int, default=10, help="number of final passes in the converged window")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--online-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="expected seeds, for the notes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123], help="expected offline seeds, for the notes")
    a = ap.parse_args()
    a.out_offline = a.out_offline or os.path.join(a.sel_pgl, "tables")
    pd.set_option("display.width", 240)
    if a.part in ("all", "online"):
        for wl in WORKLOADS:
            online(a, wl)
    if a.part in ("all", "offline"):
        offline(a)


if __name__ == "__main__":
    main()
