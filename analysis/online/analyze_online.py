"""Summaries of the online runs (analysis/online/run_online.py).  Read-only.

Windows are fixed in advance: *converged* = the last ten passes, *learning* = every pass after the first (which
runs PostgreSQL's default plans for every model).  Intervals resample the templates of the workload (cluster
bootstrap, 20,000 draws); a query enters with the geometric mean of its speed-up over the seeds and the passes of the
window.

    python analysis/online/analyze_online.py      # -> results/analysis/online_20261005/tables/*.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
from protocol import template_key                                # noqa: E402

MODELS = ["gnto", "gnto_none", "gnto_tcr", "bao", "limao"]


def load(root: str, workload: str, schedule: str = "sync"):
    """schedule 'sync': the runs <model>_s<seed>; 'async': the runs <model>_s<seed>_async (background trainer)."""
    runs = {}
    for m in MODELS:
        for d in sorted(glob.glob(os.path.join(root, workload, f"{m}_s*"))):
            base = os.path.basename(d)
            if base.endswith("_greedy") or base.endswith("_async") != (schedule == "async"):
                continue
            if not os.path.exists(os.path.join(d, "summary.json")):
                continue
            ch = pd.read_csv(os.path.join(d, "choices.csv"))
            ps = pd.read_csv(os.path.join(d, "passes.csv"))
            runs.setdefault(m, {})[int(base.replace("_async", "").rsplit("_s", 1)[1])] = (ps, ch)
    return runs


def per_query(chs, lo: int, hi: int, col: str = "ms") -> pd.Series:
    """Mean log latency per query over the seeds and the passes lo..hi-1."""
    parts = []
    for ch in chs:
        w = ch[(ch["pass"] >= lo) & (ch["pass"] < hi)]
        parts.append(np.log(w[col].clip(lower=1e-3)).groupby(w["query_id"]).mean())
    return pd.concat(parts, axis=1).mean(axis=1)


def cluster_ci(values: pd.Series, n_boot: int, rng) -> tuple:
    """Geometric mean of exp(values) over queries, templates resampled."""
    tmpl = values.index.map(template_key)
    groups = [values[tmpl == t].to_numpy() for t in sorted(set(tmpl))]
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    idx = rng.integers(len(groups), size=(n_boot, len(groups)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.exp(values.mean())), float(np.exp(np.percentile(boot, 2.5))), float(np.exp(np.percentile(boot, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "analysis", "online_20261005"))
    ap.add_argument("--workload", default="job112")
    ap.add_argument("--converged", type=int, default=10, help="number of final passes in the converged window")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--long-ms", type=float, default=1000.0, help="steer-only-long-queries policy: default latency threshold in ms")
    ap.add_argument("--schedule", default="sync", choices=["sync", "async"], help="which runs: retrained between passes, or background trainer")
    a = ap.parse_args()
    runs = load(a.root, a.workload, a.schedule)
    out = os.path.join(a.root, "tables")
    overhead = None
    lat_path = os.path.join(ROOT, "results", "main_runs", a.workload, "pg_gatv2_rank1_s42", "latency_cpu1.csv")
    if os.path.exists(lat_path):
        lat = pd.read_csv(lat_path).set_index("query_id")
        overhead = (lat.enumeration_wall_ms - lat.pg_planning_wall_ms + lat.full_model_ms).clip(lower=0)
        print(f"selection overhead per query (ms): median {overhead.median():.0f}, mean {overhead.mean():.0f} ({len(overhead)} queries)")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261005)
    n_pass = min(len(ps) for m in runs.values() for ps, _ in m.values())
    windows = {"converged": (n_pass - a.converged, n_pass), "learning": (1, n_pass), "first10": (1, 11)}

    curve, summary, contrasts = [], [], []
    for m, by_seed in runs.items():
        g = np.array([ps.geo_speedup.to_numpy()[:n_pass] for ps, _ in by_seed.values()])
        tot = np.array([(ps.total_s / ps.default_total_s).to_numpy()[:n_pass] for ps, _ in by_seed.values()])
        s2 = np.array([ps.slower_2x.to_numpy()[:n_pass] for ps, _ in by_seed.values()])
        for p in range(n_pass):
            curve.append(dict(model=m, **{"pass": p}, n_seeds=len(by_seed), geo=float(np.exp(np.log(g[:, p]).mean())),
                              geo_min=float(g[:, p].min()), geo_max=float(g[:, p].max()), total_ratio=float(tot[:, p].mean()),
                              slower_2x=float(s2[:, p].mean())))
        # first pass from which every later pass is at least as fast as the default plans (per seed)
        stable = [next((p for p in range(1, n_pass) if (row[p:] >= 1.0).all()), np.nan) for row in g]
        reach = [next((p for p in range(1, n_pass) if row[p] >= 1.1), np.nan) for row in g]
        med = lambda v: float(np.nanmedian(v)) if not np.isnan(v).all() else np.nan      # noqa: E731
        # cumulative workload time over all passes (exploration included), the first pass from which the cumulative
        # time stays at or below the default plans' (per seed), and the retraining seconds per pass
        cum = [float(ps.total_s.to_numpy()[:n_pass].sum() / ps.default_total_s.to_numpy()[:n_pass].sum()) for ps, _ in by_seed.values()]
        even = []
        for ps, _ in by_seed.values():
            ok = (ps.total_s.cumsum() <= ps.default_total_s.cumsum()).to_numpy()[:n_pass]
            even.append(next((p for p in range(1, n_pass) if ok[p:].all()), np.nan))
        train = [float(ps.train_s.to_numpy()[1:n_pass].mean()) for ps, _ in by_seed.values()]
        # asynchronous schedule: how many retrainings ran, and how stale the serving model was (passes between the
        # last pass in its training data and the pass it served), over the passes served by a model
        sched = {}
        if "model_pass" in next(iter(by_seed.values()))[0].columns:
            stale, nre, first = [], [], []
            for ps, _ in by_seed.values():
                mp = ps.model_pass.to_numpy()[:n_pass]
                served = np.arange(n_pass)[mp >= 0]
                stale.append(float((served - mp[mp >= 0] - 1).mean()) if len(served) else np.nan)
                nre.append(int(ps.retrained.to_numpy()[:n_pass].sum()))
                first.append(int(served[0]) if len(served) else np.nan)
            sched = dict(stale_passes_mean=float(np.nanmean(stale)), retrainings_mean=float(np.mean(nre)),
                         first_model_pass_median=float(np.nanmedian(first)),
                         train_s_total_mean=float(np.mean([ps.train_s.to_numpy()[:n_pass].sum() for ps, _ in by_seed.values()])))
        # the same, with the selection overhead charged per query: planning the 49 configurations (minus the one
        # planning that the executed plan's EXPLAIN ANALYZE already contains) plus featurizing and scoring, taken
        # from the per-query measurements of the offline run (latency_cpu1.csv of the GNTO selector, seed 42); the
        # scoring time of GNTO is charged to every selector
        e2e = {}
        if overhead is not None:
            cum_e2e, even_e2e, conv_e2e = [], [], []
            for ps, ch in by_seed.values():
                c = ch[ch["pass"] < n_pass].copy()
                c["e2e"] = c.ms + c.query_id.map(overhead).fillna(overhead.median())
                c.loc[c["pass"] == 0, "e2e"] = c.loc[c["pass"] == 0, "ms"]            # pass 0 runs the default plans
                pp = c.groupby("pass").agg(model=("e2e", "sum"), default=("default_ms", "sum"))
                cm, cd = pp.model.cumsum(), pp.default.cumsum()
                cum_e2e.append(float(cm.iloc[-1] / cd.iloc[-1]))
                ok = (cm <= cd).to_numpy()
                even_e2e.append(next((p for p in range(1, n_pass) if ok[p:].all()), np.nan))
                w = pp.iloc[n_pass - a.converged:]
                conv_e2e.append(float(w.model.sum() / w.default.sum()))
            e2e = dict(e2e_cum_ratio_median=float(np.median(cum_e2e)), e2e_converged_time_median=float(np.median(conv_e2e)),
                       e2e_breakeven_pass_median=med(even_e2e), e2e_breakeven_seeds=int(np.sum(~np.isnan(even_e2e))))
            # the same under the policy 'steer only the queries whose default plan ran for at least --long-ms' (known from
            # pass 0 of a recurring workload); the other queries keep the default plan and pay no overhead
            cum_l, even_l, conv_l, n_l = [], [], [], []
            for ps, ch in by_seed.values():
                c = ch[ch["pass"] < n_pass].copy()
                steer = c.default_ms >= a.long_ms
                c["pol"] = np.where(steer, c.ms + c.query_id.map(overhead).fillna(overhead.median()), c.default_ms)
                c.loc[c["pass"] == 0, "pol"] = c.loc[c["pass"] == 0, "ms"]
                pp = c.groupby("pass").agg(model=("pol", "sum"), default=("default_ms", "sum"))
                cm, cd = pp.model.cumsum(), pp.default.cumsum()
                cum_l.append(float(cm.iloc[-1] / cd.iloc[-1]))
                ok = (cm <= cd).to_numpy()
                even_l.append(next((p for p in range(1, n_pass) if ok[p:].all()), np.nan))
                w = pp.iloc[n_pass - a.converged:]
                conv_l.append(float(w.model.sum() / w.default.sum()))
                n_l.append(int(steer[c["pass"] == 1].sum()))
            e2e.update(long_ms=float(a.long_ms), long_n_queries=float(np.median(n_l)), long_cum_ratio_median=float(np.median(cum_l)),
                       long_converged_time_median=float(np.median(conv_l)), long_breakeven_pass_median=med(even_l),
                       long_breakeven_pass_max=float(np.nanmax(even_l)) if not np.isnan(even_l).all() else np.nan,
                       long_breakeven_seeds=int(np.sum(~np.isnan(even_l))))
        rec = dict(model=m, n_seeds=len(by_seed), n_passes=n_pass, stable_pass_median=med(stable),
                   cum_ratio_median=float(np.median(cum)), cum_ratio_min=float(np.min(cum)), cum_ratio_max=float(np.max(cum)),
                   breakeven_pass_median=med(even), breakeven_pass_max=float(np.nanmax(even)) if not np.isnan(even).all() else np.nan,
                   breakeven_seeds=int(np.sum(~np.isnan(even))), train_s_mean=float(np.mean(train)), **e2e, **sched,
                   stable_pass_max=float(np.nanmax(stable)) if not np.isnan(stable).all() else np.nan,
                   stable_seeds=int(np.sum(~np.isnan(stable))), reach110_pass_median=med(reach),
                   reach110_seeds=int(np.sum(~np.isnan(reach))))
        chs = [ch for _, ch in by_seed.values()]
        for w, (lo, hi) in windows.items():
            v = per_query(chs, lo, hi, "default_ms") - per_query(chs, lo, hi, "ms")
            geo, l, h = cluster_ci(v, a.n_boot, rng)
            rec.update({f"{w}_geo": geo, f"{w}_lo": l, f"{w}_hi": h,
                        f"{w}_seed_min": float(np.exp(np.log(g[:, lo:hi]).mean(axis=1)).min()),
                        f"{w}_seed_max": float(np.exp(np.log(g[:, lo:hi]).mean(axis=1)).max()),
                        f"{w}_total_ratio": float(1.0 / tot[:, lo:hi].mean()), f"{w}_slower_2x": float(s2[:, lo:hi].mean())})
        best = per_query(chs, 0, n_pass, "default_ms") - per_query(chs, 0, n_pass, "best_ms")
        rec["oracle_geo"] = float(np.exp(best.mean()))
        summary.append(rec)
    for x, y in (("gnto", "bao"), ("gnto", "limao"), ("gnto", "gnto_tcr"), ("gnto_tcr", "bao"), ("gnto", "gnto_none"), ("gnto_none", "bao")):
        if x not in runs or y not in runs:
            continue
        cx, cy = [ch for _, ch in runs[x].values()], [ch for _, ch in runs[y].values()]
        for w, (lo, hi) in windows.items():
            v = per_query(cy, lo, hi) - per_query(cx, lo, hi)                # > 0: the plans of x are faster
            geo, l, h = cluster_ci(v, a.n_boot, rng)
            contrasts.append(dict(contrast=f"{x}_vs_{y}", window=w, passes=f"{lo}-{hi - 1}", ratio=geo, lo=l, hi=h,
                                  queries_faster=int((v > np.log(1.05)).sum()), queries_slower=int((v < -np.log(1.05)).sum())))
    sfx = ("" if a.workload == "job112" else f"_{a.workload}") + ("" if a.schedule == "sync" else "_async")
    pd.DataFrame(curve).to_csv(os.path.join(out, f"online{sfx}_curve.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out, f"online{sfx}_summary.csv"), index=False)
    pd.DataFrame(contrasts).to_csv(os.path.join(out, f"online{sfx}_contrasts.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["model", "n_seeds", "n_passes", "stable_pass_median", "stable_seeds", "reach110_pass_median", "first10_geo",
            "converged_geo", "converged_lo", "converged_hi", "converged_seed_min", "converged_seed_max", "learning_geo",
            "learning_total_ratio", "converged_slower_2x"]
    print(pd.DataFrame(summary)[cols].round(3).to_string(index=False))
    print(pd.DataFrame(contrasts).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
