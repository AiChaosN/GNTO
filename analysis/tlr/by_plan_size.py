"""Own cost (TLR) against the cumulative cost (TCR) by plan size on the 20-database benchmark.

Plans are split at the median plan of the benchmark: at most ten operators, and more than ten. For every model the
estimand of analysis/dependency_study/analyze_crossdb.py is kept: the percentile is computed per database
and seed, averaged over databases and seeds; the relative change is the ratio of these means minus one; the interval
resamples the test databases (20,000 draws, rng 20261003, the same draws for both models). A database enters a size
group only if it has at least 20 plans of that group in every seed of both models.

Models: GNTO (three splits; TLR = results/analysis/tlr_20261008/gnto_resid_*, TCR = the
results/main_runs/crossdb/gnto_*_trainstats runs) and the encoders that were run
with both inputs (fixed split; bag and GATv2 also on random split 1).

GNTO is also pooled over the three splits at fourteen percentiles (pooled_profile).

    python analysis/tlr/by_plan_size.py      # -> results/analysis/tlr_20261008/tables/by_plan_size.csv
                                             #    + tables/by_percentile_pooled.csv
                                             #    + tables/by_plan_size_macros.tex (\\setres macros sz.*)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEP = os.path.join(ROOT, "results", "analysis", "dependency_20261003")
TLR = os.path.join(ROOT, "results", "analysis", "tlr_20261008")
TCR = os.path.join(ROOT, "results", "main_runs", "crossdb")
MACROS = os.path.join(TLR, "tables", "by_plan_size_macros.tex")
LEVELS = (50, 90, 99)
CUT = 10
MIN_PLANS = 20
GROUPS = {"small": lambda n: n <= CUT, "large": lambda n: n > CUT}
ENCODERS = [("none", "L3"), ("none", "L6"), ("gcn", "L3"), ("gcn", "L6"), ("gatv2", "L3"), ("tcnn", "L3"),
            ("bigg", "L3"), ("trlocal", "L3"), ("trsubplan", "L3"), ("trglobal", "L3")]


def qerr(pred, actual):
    p, t = np.maximum(pred, 1e-7), np.maximum(actual, 1e-7)
    return np.maximum(p / t, t / p)


def load(dirs):
    """{seed: {db: DataFrame}} of finished runs."""
    runs = {}
    for d in sorted(dirs):
        if not os.path.isdir(os.path.join(d, "predictions")):
            continue
        seed = int(os.path.basename(d).split("_s")[-1].split("_")[0])
        runs[seed] = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(d, "predictions", "*.csv"))}
    return runs


def levels(runs, group):
    """{seed: {db: {level: q}}}; a database without enough plans of the group is left out."""
    out = {}
    for seed, dbs in runs.items():
        out[seed] = {}
        for db, df in dbs.items():
            sel = df[GROUPS[group](df["n_nodes"].values)]
            if len(sel) >= MIN_PLANS:
                q = qerr(sel["pred_ms"].values, sel["actual_ms"].values)
                out[seed][db] = {k: float(np.percentile(q, k)) for k in LEVELS}
    return out


def contrast(a_runs, b_runs, group, rng_seed=20261003, draws=20000):
    la, lb = levels(a_runs, group), levels(b_runs, group)
    seeds = sorted(set(la) & set(lb))
    dbs = sorted(set.intersection(*[set(la[s]) & set(lb[s]) for s in seeds]))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(dbs), size=(draws, len(dbs)))
    res = {}
    for k in LEVELS:
        A = np.array([[la[s][db][k] for db in dbs] for s in seeds])
        B = np.array([[lb[s][db][k] for db in dbs] for s in seeds])
        rel = A.mean() / B.mean() - 1
        boot = A.mean(axis=0)[idx].mean(axis=1) / B.mean(axis=0)[idx].mean(axis=1) - 1
        res[k] = dict(a=float(A.mean()), b=float(B.mean()), rel=float(rel),
                      lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
    return res, len(dbs), len(seeds)


def share_large(runs):
    n = [(df["n_nodes"].values > CUT).mean() for dbs in runs.values() for df in dbs.values()]
    return float(np.mean(n))


PROFILE = (50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99)
POOL_GROUPS = {"all": lambda n: n >= 0, **GROUPS}


def pooled_profile(gnto_pairs, rng_seed=20261003, draws=20000):
    """GNTO, TLR against TCR at many percentiles, pooled over the three database splits.

    A cell is a (split, test database); its percentile is the mean over the seeds of the per-seed percentile, as in
    contrast(). The change is the ratio of the means over all cells minus one. The interval resamples the databases,
    each with all its cells across the splits, since one database is a test database of more than one split."""
    rows = []
    for group in POOL_GROUPS:
        A, B, DB = [], [], []
        for split, a_runs, b_runs in gnto_pairs:
            la, lb = {}, {}
            for runs, out in ((a_runs, la), (b_runs, lb)):
                for seed, dbs in runs.items():
                    for db, df in dbs.items():
                        sel = df[POOL_GROUPS[group](df["n_nodes"].values)]
                        if len(sel) >= MIN_PLANS:
                            out.setdefault(seed, {})[db] = np.percentile(qerr(sel["pred_ms"].values,
                                                                              sel["actual_ms"].values), PROFILE)
            seeds = sorted(set(la) & set(lb))
            dbs = sorted(set.intersection(*[set(la[s]) & set(lb[s]) for s in seeds]))
            for db in dbs:
                A.append(np.mean([la[s][db] for s in seeds], axis=0))
                B.append(np.mean([lb[s][db] for s in seeds], axis=0))
                DB.append(db)
        A, B, DB = np.array(A), np.array(B), np.array(DB)
        uniq = np.unique(DB)
        members = [np.where(DB == d)[0] for d in uniq]
        rng = np.random.default_rng(rng_seed)
        boot = np.empty((draws, len(PROFILE)))
        for i in range(draws):
            idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
            boot[i] = A[idx].mean(axis=0) / B[idx].mean(axis=0) - 1
        rel = A.mean(axis=0) / B.mean(axis=0) - 1
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        for j, k in enumerate(PROFILE):
            rows.append(dict(group=group, level=k, n_cells=len(DB), n_dbs=len(uniq), tcr=float(B.mean(axis=0)[j]),
                             tlr=float(A.mean(axis=0)[j]), rel=float(rel[j]), lo=float(lo[j]), hi=float(hi[j])))
    return pd.DataFrame(rows)


def main() -> None:
    rows = []
    pairs = []
    for split in ("paper1", "rand1", "rand2"):
        a = load(glob.glob(os.path.join(TLR, f"gnto_resid_{split}_s*")))
        b = load(glob.glob(os.path.join(TCR, f"gnto_{split}_s*_trainstats")))
        pairs.append(("gnto", split, a, b))
    for sub, split in (("crossdb", "paper1"), ("crossdb_rand1", "rand1")):
        for enc, L in ENCODERS:
            a = load(glob.glob(os.path.join(DEP, sub, f"resid_{enc}_{L}_true_s*")))
            b = load(glob.glob(os.path.join(DEP, sub, f"full_{enc}_{L}_true_s*")))
            if a and b:
                pairs.append((f"{enc}{'' if L == 'L3' else L}", split, a, b))
    for model, split, a, b in pairs:
        for group in GROUPS:
            res, n_dbs, n_seeds = contrast(a, b, group)
            for k, r in res.items():
                rows.append(dict(model=model, split=split, group=group, level=f"q{k}", n_dbs=n_dbs, n_seeds=n_seeds,
                                 share_large=share_large(b), tcr=r["b"], tlr=r["a"], rel=r["rel"], lo=r["lo"], hi=r["hi"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(TLR, "tables"), exist_ok=True)
    df.to_csv(os.path.join(TLR, "tables", "by_plan_size.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.assign(rel=lambda d: (100 * d.rel).round(1), lo=lambda d: (100 * d.lo).round(1),
                    hi=lambda d: (100 * d.hi).round(1)).to_string(index=False))

    # ---- macros ---------------------------------------------------------------------------
    f = lambda x: f"{100 * x:+.1f}".replace("+", "+") if x >= 0 else f"{100 * x:.1f}"
    m = {}
    for r in rows:
        key = f"sz.{r['model']}.{r['split']}.{r['group']}.{r['level']}"
        m[f"{key}.rel"] = f(r["rel"]); m[f"{key}.rello"] = f(r["lo"]); m[f"{key}.relhi"] = f(r["hi"])
        m[f"{key}.relpct"] = f"{abs(100 * r['rel']):.1f}"
        m[f"{key}.tcr"] = f"{r['tcr']:.3f}" if r["level"] == "q50" else f"{r['tcr']:.2f}"
        m[f"{key}.tlr"] = f"{r['tlr']:.3f}" if r["level"] == "q50" else f"{r['tlr']:.2f}"
    grid = df[(df.model != "gnto")]
    for group in GROUPS:
        for lev in ("q50", "q90", "q99"):
            g = grid[(grid.group == group) & (grid.level == lev)]
            m[f"sz.grid.{group}.{lev}.n"] = str(len(g))
            m[f"sz.grid.{group}.{lev}.nlower"] = str(int((g.rel < 0).sum()))
            m[f"sz.grid.{group}.{lev}.nsig"] = str(int((g.hi < 0).sum()))
            neg = g[g.rel < 0]
            m[f"sz.grid.{group}.{lev}.minpct"] = f"{abs(100 * neg.rel.max()):.1f}" if len(neg) else "--"
            m[f"sz.grid.{group}.{lev}.maxpct"] = f"{abs(100 * neg.rel.min()):.1f}" if len(neg) else "--"
            m[f"sz.grid.{group}.{lev}.lopct"] = f"{100 * g.rel.min():+.1f}"
            m[f"sz.grid.{group}.{lev}.hipct"] = f"{100 * g.rel.max():+.1f}"
        gn = df[(df.model == "gnto") & (df.group == group)]
        for lev in ("q50", "q90", "q99"):
            x = gn[gn.level == lev]
            m[f"sz.gnto.{group}.{lev}.minpct"] = f"{abs(100 * x.rel.max()):.1f}"
            m[f"sz.gnto.{group}.{lev}.maxpct"] = f"{abs(100 * x.rel.min()):.1f}"
            m[f"sz.gnto.{group}.{lev}.nlower"] = str(int((x.rel < 0).sum()))
            m[f"sz.gnto.{group}.{lev}.nsig"] = str(int((x.hi < 0).sum()))
            m[f"sz.gnto.{group}.{lev}.absmax"] = f"{100 * x.rel.abs().max():.1f}"
        for lev in ("q50", "q90", "q99"):
            g = grid[(grid.group == group) & (grid.level == lev)]
            m[f"sz.grid.{group}.{lev}.absmax"] = f"{100 * g.rel.abs().max():.1f}"
    m["sz.share.large.pct"] = f"{100 * df[(df.model == 'gnto') & (df.split == 'paper1')].share_large.iloc[0]:.0f}"
    m["sz.cut"] = str(CUT)

    # ---- GNTO pooled over the three splits, by percentile ----------------------------------
    pool = pooled_profile([(split, a, b) for model, split, a, b in pairs if model == "gnto"])
    pool.to_csv(os.path.join(TLR, "tables", "by_percentile_pooled.csv"), index=False)
    print(pool.assign(rel=lambda d: (100 * d.rel).round(1), lo=lambda d: (100 * d.lo).round(1),
                      hi=lambda d: (100 * d.hi).round(1)).to_string(index=False))
    for group, g in pool.groupby("group"):
        m[f"sz.pool.{group}.ndb"] = str(int(g.n_dbs.iloc[0]))
        for r in g.itertuples():
            key = f"sz.pool.{group}.q{r.level}"
            m[f"{key}.rel"] = f(r.rel); m[f"{key}.rello"] = f(r.lo); m[f"{key}.relhi"] = f(r.hi)
            m[f"{key}.relpct"] = f"{abs(100 * r.rel):.1f}"
            m[f"{key}.relpct0"] = f"{abs(100 * r.rel):.0f}"
        tail = g[g.level >= 70]
        m[f"sz.pool.{group}.q70up.minpct"] = f"{abs(100 * tail.rel.max()):.0f}"
        m[f"sz.pool.{group}.q70up.maxpct"] = f"{abs(100 * tail.rel.min()):.0f}"
        above = [k for k in g.level if (g[g.level >= k].hi < 0).all()]          # every interval from k on excludes zero
        m[f"sz.pool.{group}.sigfrom"] = str(int(min(above))) if above else "--"
    with open(MACROS, "w") as fh:
        fh.write("% generated by analysis/tlr/by_plan_size.py - do not edit\n")
        for k in sorted(m):
            fh.write(f"\\setres{{{k}}}{{{m[k]}}}\n")
    print(f"-> {os.path.join(TLR, 'tables', 'by_plan_size.csv')}\n-> {MACROS} ({len(m)} macros)")


if __name__ == "__main__":
    main()
