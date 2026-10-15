"""Plan-selection analysis on the main steering batch plus the dependency-study runs.

Read-only.  Estimands (query-weighted geometric mean of PG_default / picked, averaged over the fitted seeds;
whole-template bootstrap with common draws).
Writes:

    steering_configs.csv     per configuration: G with interval, total-time ratio with interval, faster / equal /
                             slower shares (+-5 %), >2x regressions, within-query Spearman, Q-error of the candidates
    steering_vs_baselines.csv  GNTO configurations against the Bao / LIMAO extensions (paired ratio of picked latencies)
    steering_relation.csv    effect of relating operators (GCN / GATv2 / tree convolution vs None), per cell and pooled
    steering_qerror.csv      paired prediction-accuracy contrast vs None (geometric-mean Q-error ratio)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = (42, 7, 123)
DRAWS, BOOT_SEED = 20000, 20261002


def boot_index(n_clusters: int, draws: int = DRAWS) -> np.ndarray:
    return np.random.default_rng(BOOT_SEED).integers(n_clusters, size=(draws, n_clusters))


def cluster_geo(logv: np.ndarray, inv: np.ndarray, ix: np.ndarray):
    """logv: (seed, query) log ratios -> geometric mean and template-cluster interval."""
    means = logv.mean(axis=0)
    sums, counts = np.bincount(inv, weights=means), np.bincount(inv)
    d = np.exp(sums[ix].sum(axis=1) / counts[ix].sum(axis=1))
    return float(np.exp(means.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cluster_total(num: np.ndarray, den: np.ndarray, inv: np.ndarray, ix: np.ndarray):
    """ratio of sums (workload total), seed-averaged; num/den: (seed, query)."""
    k = inv.max() + 1
    ns = np.stack([np.bincount(inv, weights=num[s], minlength=k) for s in range(num.shape[0])])
    ds = np.stack([np.bincount(inv, weights=den[s], minlength=k) for s in range(den.shape[0])])
    point = float(np.mean(num.sum(axis=1) / den.sum(axis=1)))
    d = np.mean(ns[:, ix].sum(axis=2) / ds[:, ix].sum(axis=2), axis=0)
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load(workload: str, base: str, dep: str):
    runs = {}
    for root in (os.path.join(base, workload), os.path.join(dep, "steering", workload),
                 os.path.join(ROOT, "results", "analysis", "attn_20261008", "selection", workload),      # reach / depth / cost variants
                 os.path.join(ROOT, "results", "analysis", "tlr_20261008", "selection", workload)):      # PG-stat + own cost (pgl)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            m = re.match(r"^(.*)_s(42|7|123)$", name)
            if not m or not os.path.exists(os.path.join(d, "summary.json")):
                continue
            picks = pd.read_csv(os.path.join(d, "picks.csv")).set_index("query_id").sort_index()
            preds = pd.read_csv(os.path.join(d, "predictions.csv"))
            runs.setdefault(m.group(1), {})[int(m.group(2))] = (picks, preds)
    ref = {m: pd.read_csv(os.path.join(base, workload, "references", m, "picks.csv")).set_index("query_id").sort_index()
           for m in ("pg_default", "pg_cost", "oracle")}
    return runs, ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "results", "main_runs"))
    ap.add_argument("--dep", default=os.path.join(ROOT, "results", "analysis", "dependency_20261003"))
    a = ap.parse_args()
    out = os.path.join(a.dep, "tables")
    os.makedirs(out, exist_ok=True)
    cfg_rows, base_rows, rel_rows, q_rows, bias_rows, rnd_rows, inp_rows = [], [], [], [], [], [], []
    for wl in ("job112", "tpch52"):
        runs, ref = load(wl, a.base, a.dep)
        ids = list(ref["pg_default"].index)
        templates = ref["pg_default"]["template"].to_numpy()
        _, inv = np.unique(templates, return_inverse=True)
        ix = boot_index(inv.max() + 1)
        pg = ref["pg_default"]["pg_default_ms"].to_numpy()

        def picked(name):
            seeds = [s for s in SEEDS if s in runs[name]]
            arr = []
            for s in seeds:
                p = runs[name][s][0]
                assert list(p.index) == ids
                arr.append(p["pick_ms"].to_numpy())
            return np.stack(arr), seeds

        def per_query_logq(name, s):
            pr = runs[name][s][1]
            pr = pr[~pr["censored"].astype(bool)]
            lq = np.abs(np.log(pr["pred"].clip(lower=1e-6) / pr["actual_ms"]))
            return lq.groupby(pr["query_id"]).mean().reindex(ids).to_numpy(), pr

        for name in ["pg_cost", "oracle"] + sorted(runs):
            if name in ref:
                pk, seeds = ref[name]["pick_ms"].to_numpy()[None, :], [None]
            else:
                pk, seeds = picked(name)
            g, lo, hi = cluster_geo(np.log(pg[None, :] / pk), inv, ix)
            t, tlo, thi = cluster_total(np.repeat(pg[None, :], len(pk), 0), pk, inv, ix)
            sp = pg[None, :] / pk
            row = dict(workload=wl, config=name, n_seeds=len(seeds), geo=g, geo_lo=lo, geo_hi=hi,
                       total=t, total_lo=tlo, total_hi=thi,
                       faster_pct=float((sp > 1.05).mean() * 100), slower_pct=float((sp < 1 / 1.05).mean() * 100),
                       equal_pct=float(((sp <= 1.05) & (sp >= 1 / 1.05)).mean() * 100),
                       reg2x=float((sp < 0.5).sum(axis=1).mean()),
                       seed_sd=float(np.exp(np.log(sp).mean(axis=1)).std(ddof=1)) if len(seeds) > 1 else np.nan)
            if name not in ref:
                q50, q90, q99, rho, gq = [], [], [], [], []
                for s in seeds:
                    lq, pr = per_query_logq(name, s)
                    q = np.maximum(pr["pred"] / pr["actual_ms"], pr["actual_ms"] / pr["pred"])
                    q50.append(np.median(q)), q90.append(np.percentile(q, 90)), q99.append(np.percentile(q, 99))
                    gq.append(np.exp(np.abs(np.log(q)).mean()))
                    r = [spearmanr(g_["pred"], g_["actual_ms"]).statistic for _, g_ in pr.groupby("query_id")
                         if len(g_) >= 3 and g_["actual_ms"].nunique() > 1 and g_["pred"].nunique() > 1]
                    rho.append(np.nanmean(r))
                row.update(q50=float(np.mean(q50)), q90=float(np.mean(q90)), q99=float(np.mean(q99)),
                           geo_q=float(np.mean(gq)), spearman=float(np.mean(rho)))
            cfg_rows.append(row)

        # GNTO configurations against the extended baselines (same candidates, same folds, same seeds)
        for base in ("bao_extended_capped", "limao_extended_capped", "reqo_tcr", "reqoval_tcr", "reqoval_full"):
            if base not in runs or len(runs[base]) != 3:
                continue
            bp, _ = picked(base)
            for name in sorted(runs):
                if not re.match(r"^(pg|dace|resid|residslog|pgl)_(none|gcn|gatv2|tcnn)_rank[01]$", name):
                    continue
                pk, seeds = picked(name)
                if len(seeds) != 3:
                    continue
                g, lo, hi = cluster_geo(np.log(bp / pk), inv, ix)
                base_rows.append(dict(workload=wl, config=name, baseline=base, ratio=g, lo=lo, hi=hi))

        # relation effect: conv vs none, per cell and pooled over objectives (and features)
        feats = sorted({n.split("_")[0] for n in runs if re.match(r"^[a-z]+_(none)_rank[01]$", n)})
        for conv in ("gcn", "gatv2", "tcnn", "bigg", "gatv2_vs_bigg"):       # bigg: Reqo's encoder (analysis/baselines)
            pooled_all = []
            for feat in feats:
                pooled = []
                for rank in (0, 1):
                    a_, b_ = f"{feat}_{conv}_rank{rank}", f"{feat}_none_rank{rank}"
                    if conv == "gatv2_vs_bigg":                              # GNTO's encoder against BiGG
                        a_, b_ = f"{feat}_gatv2_rank{rank}", f"{feat}_bigg_rank{rank}"
                    if a_ not in runs or b_ not in runs:
                        continue
                    pa, sa = picked(a_)
                    pb, sb = picked(b_)
                    if sa != sb:
                        continue
                    lr = np.log(pb / pa)                                    # >0: conv picks faster plans than None
                    g, lo, hi = cluster_geo(lr, inv, ix)
                    rel_rows.append(dict(workload=wl, features=feat, conv=conv, objective=f"rank{rank}", ratio=g, lo=lo, hi=hi,
                                         n_seeds=len(sa)))
                    pooled.append(lr)
                    # prediction accuracy: geometric-mean Q-error ratio (conv / none), <1: conv more accurate
                    la = np.stack([per_query_logq(a_, s)[0] for s in sa])
                    lb = np.stack([per_query_logq(b_, s)[0] for s in sa])
                    ok = ~np.isnan(la).any(axis=0) & ~np.isnan(lb).any(axis=0)
                    _, inv2 = np.unique(templates[ok], return_inverse=True)
                    g2, lo2, hi2 = cluster_geo((la - lb)[:, ok], inv2, boot_index(inv2.max() + 1))
                    q_rows.append(dict(workload=wl, features=feat, conv=conv, objective=f"rank{rank}",
                                       geoq_ratio=g2, lo=lo2, hi=hi2))
                if pooled:
                    lr = np.concatenate(pooled, axis=0)
                    g, lo, hi = cluster_geo(lr, inv, ix)
                    rel_rows.append(dict(workload=wl, features=feat, conv=conv, objective="pooled", ratio=g, lo=lo, hi=hi,
                                         n_seeds=lr.shape[0]))
                    if feat in ("dace", "pg"):
                        pooled_all.append(lr)
            if pooled_all:
                lr = np.concatenate(pooled_all, axis=0)
                g, lo, hi = cluster_geo(lr, inv, ix)
                rel_rows.append(dict(workload=wl, features="dace+pg", conv=conv, objective="pooled", ratio=g, lo=lo, hi=hi,
                                     n_seeds=lr.shape[0]))
        # random tree instead of the plan topology (pointwise objective): against None and against the true topology
        for conv in ("gcn", "gatv2"):
            pooled = {"random vs none": [], "true vs random": []}
            for feat in ("dace", "pg"):
                names = {"rnd": f"{feat}rnd_{conv}_rank0", "true": f"{feat}_{conv}_rank0", "none": f"{feat}_none_rank0"}
                if not all(n in runs and len(runs[n]) == 3 for n in names.values()):
                    continue
                pk = {k: picked(n)[0] for k, n in names.items()}
                lq = {k: np.stack([per_query_logq(n, s)[0] for s in SEEDS]) for k, n in names.items()}
                for label, a_, b_ in (("random vs none", "rnd", "none"), ("true vs random", "true", "rnd")):
                    sel = np.log(pk[b_] / pk[a_])                          # >0: a selects faster plans than b
                    g, lo, hi = cluster_geo(sel, inv, ix)
                    acc = lq[a_] - lq[b_]                                  # <0: a predicts more accurately than b
                    ok = ~np.isnan(acc).any(axis=0)
                    _, inv2 = np.unique(templates[ok], return_inverse=True)
                    g2, lo2, hi2 = cluster_geo(acc[:, ok], inv2, boot_index(inv2.max() + 1))
                    rnd_rows.append(dict(workload=wl, features=feat, conv=conv, contrast=label, sel_ratio=g, sel_lo=lo, sel_hi=hi,
                                         geoq_ratio=g2, geoq_lo=lo2, geoq_hi=hi2))
                    pooled[label].append((sel, acc))
            for label, items in pooled.items():
                if len(items) == 2:
                    sel = np.concatenate([x[0] for x in items], axis=0)
                    acc = np.concatenate([x[1] for x in items], axis=0)
                    g, lo, hi = cluster_geo(sel, inv, ix)
                    ok = ~np.isnan(acc).any(axis=0)
                    _, inv2 = np.unique(templates[ok], return_inverse=True)
                    g2, lo2, hi2 = cluster_geo(acc[:, ok], inv2, boot_index(inv2.max() + 1))
                    rnd_rows.append(dict(workload=wl, features="dace+pg", conv=conv, contrast=label, sel_ratio=g, sel_lo=lo,
                                         sel_hi=hi, geoq_ratio=g2, geoq_lo=lo2, geoq_hi=hi2))

        # what the optimizer's estimates do to plan selection: inputs without cost (and rows) against TCR, pointwise
        for feat in ("typeonly", "nocost"):
            pooled = []
            for conv in ("none", "gcn", "gatv2", "tcnn"):
                na, nb = f"{feat}_{conv}_rank0", f"dace_{conv}_rank0"
                if na in runs and nb in runs and len(runs[na]) == 3 and len(runs[nb]) == 3:
                    lr = np.log(picked(nb)[0] / picked(na)[0])            # >0: fewer inputs select faster plans
                    g, lo, hi = cluster_geo(lr, inv, ix)
                    inp_rows.append(dict(workload=wl, inputs=feat, conv=conv, ratio=g, lo=lo, hi=hi))
                    pooled.append(lr)
            if len(pooled) == 4:
                g, lo, hi = cluster_geo(np.concatenate(pooled, axis=0), inv, ix)
                inp_rows.append(dict(workload=wl, inputs=feat, conv="pooled", ratio=g, lo=lo, hi=hi))

        # diagnostics: median pred/actual over all candidates vs over the selected candidates, and G split by
        # headroom (oracle speed-up over the default plan < 1.3 vs >= 1.3)
        gain = ref["pg_default"]["pg_default_ms"].to_numpy() / ref["oracle"]["pick_ms"].to_numpy()
        low = gain < 1.3
        for name in sorted(runs):
            seeds = [s for s in SEEDS if s in runs[name]]
            all_r, sel_r, g_low, g_high = [], [], [], []
            for s in seeds:
                p, pr = runs[name][s]
                pr = pr[~pr["censored"].astype(bool)]
                all_r.append(np.median(pr["pred"] / pr["actual_ms"]))
                m = pr.merge(p[["pick_hint"]], left_on="query_id", right_index=True)
                m = m[m["hint_id"] == m["pick_hint"]]
                sel_r.append(np.median(m["pred"] / m["actual_ms"]))
                lg = np.log(pg / p["pick_ms"].to_numpy())
                g_low.append(np.exp(lg[low].mean())), g_high.append(np.exp(lg[~low].mean()))
            bias_rows.append(dict(workload=wl, config=name, n_seeds=len(seeds), pred_over_actual_all=float(np.mean(all_r)),
                                  pred_over_actual_selected=float(np.mean(sel_r)), n_low_headroom=int(low.sum()),
                                  geo_low_headroom=float(np.mean(g_low)), n_headroom=int((~low).sum()),
                                  geo_headroom=float(np.mean(g_high))))
    pd.DataFrame(bias_rows).to_csv(os.path.join(out, "steering_bias.csv"), index=False)
    pd.DataFrame(rnd_rows).to_csv(os.path.join(out, "steering_random.csv"), index=False)
    pd.DataFrame(inp_rows).to_csv(os.path.join(out, "steering_inputs.csv"), index=False)
    pd.DataFrame(cfg_rows).to_csv(os.path.join(out, "steering_configs.csv"), index=False)
    pd.DataFrame(base_rows).to_csv(os.path.join(out, "steering_vs_baselines.csv"), index=False)
    pd.DataFrame(rel_rows).to_csv(os.path.join(out, "steering_relation.csv"), index=False)
    pd.DataFrame(q_rows).to_csv(os.path.join(out, "steering_qerror.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(pd.DataFrame(cfg_rows).round(3).to_string(index=False))
    print(pd.DataFrame(base_rows).round(3).to_string(index=False))
    print(pd.DataFrame(rel_rows).round(3).to_string(index=False))
    print(pd.DataFrame(q_rows).round(3).to_string(index=False))
    print(pd.DataFrame(rnd_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
