"""Stability of the main effects by seed, database, fold and template.  Read-only.

Cross-database (published split and split 2): for each paired contrast the relative change of the macro Q-error per
seed (same seed on both sides), its range over seeds, and the number of the 10 test databases whose seed-averaged
quantile moves in the same direction as the mean.  Selection folds (JOB, TPC-H): the per-query ratio of the picked
plans' latencies, seed by seed, per template fold (seed-averaged) and per template.  Online (JOB): GNTO vs Bao-ext per
seed over the converged window.

    python analysis/dependency_study/stability.py   # -> results/analysis/dependency_20261003/tables/stability.csv
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from analyze_crossdb import LEVELS, load_runs, macro, cfg_name  # noqa: E402

DEP = os.path.join(ROOT, "results", "analysis", "dependency_20261003")
BASE = os.path.join(ROOT, "results", "main_runs")
ATTN = os.path.join(ROOT, "results", "analysis", "attn_20261008", "selection")
ONLINE = os.path.join(ROOT, "results", "analysis", "online_20261005")

CROSSDB = [  # label, slug, a, b   (a - b; negative = a lower error)
    ("cost withheld (None): type+rows vs TCR", "nocost", "nocost|none|L3|true", "full|none|L3|true"),
    ("root cost vs cumulative (None)", "rootcost.none", "rootcost|none|L3|true", "full|none|L3|true"),
    ("root cost vs cumulative (GATv2)", "rootcost.gatv2", "rootcost|gatv2|L3|true", "full|gatv2|L3|true"),
    ("root cost + residual vs cumulative (None)", "localcost.none", "localcost|none|L3|true", "full|none|L3|true"),
    ("root cost + residual vs cumulative (GATv2)", "localcost.gatv2", "localcost|gatv2|L3|true", "full|gatv2|L3|true"),
    ("None with residual vs GATv2 with TCR", "localcost.none.gatv2", "localcost|none|L3|true", "full|gatv2|L3|true"),
    ("root cost + cumulative vs cumulative (None)", "rootcum.none", "rootcum|none|L3|true", "full|none|L3|true"),
    ("root cost + cumulative vs cumulative (GATv2)", "rootcum.gatv2", "rootcum|gatv2|L3|true", "full|gatv2|L3|true"),
    ("residual vs cumulative, both with root cost (None)", "localcost.rootcum.none", "localcost|none|L3|true", "rootcum|none|L3|true"),
    ("residual vs cumulative, both with root cost (GATv2)", "localcost.rootcum.gatv2", "localcost|gatv2|L3|true", "rootcum|gatv2|L3|true"),
    ("residual vs shuffled residual (None)", "localcost.shuf.none", "localcost|none|L3|true", "localcostshuf|none|L3|true"),
    ("residual vs cumulative, both with root cost and plan size (None)", "localcostn.rootcumn.none", "localcostn|none|L3|true", "rootcumn|none|L3|true"),
    ("residual vs cumulative, six-layer bag (None6)", "localcost.rootcum.none6", "localcost|none|L6|true", "rootcum|none|L6|true"),
    ("own cost (residual, no root) vs cumulative (None)", "resid.none", "resid|none|L3|true", "full|none|L3|true"),
    ("own cost (residual, no root) vs cumulative (GATv2)", "resid.gatv2", "resid|gatv2|L3|true", "full|gatv2|L3|true"),
    ("type dropped, cumulative: C+R vs TCR (None)", "costrows.none", "costrows|none|L3|true", "full|none|L3|true"),
    ("relations, type+rows: GATv2 vs None", "rel.nocost", "nocost|gatv2|L3|true", "nocost|none|L3|true"),
    ("relations, TCR: GATv2 vs None", "rel.full", "full|gatv2|L3|true", "full|none|L3|true"),
    ("relations, residual inputs: GATv2 vs None", "rel.localcost", "localcost|gatv2|L3|true", "localcost|none|L3|true"),
    ("depth: 3 layers vs 1 (type+rows)", "depth.l3", "nocost|gatv2|L3|true", "nocost|gatv2|L1|true"),
    ("depth: 5 layers vs 1 (type+rows)", "depth.l5", "nocost|gatv2|L5|true", "nocost|gatv2|L1|true"),
    ("virtual node vs 3 layers (type+rows)", "vn", "nocost|gatv2+vn|L3|true", "nocost|gatv2|L3|true"),
    ("reach: whole sub-plan vs parent and children (type+rows)", "reach.subplan.nocost", "nocost|trsubplan|L3|true", "nocost|trlocal|L3|true"),
    ("reach: whole plan vs parent and children (type+rows)", "reach.global.nocost", "nocost|trglobal|L3|true", "nocost|trlocal|L3|true"),
    ("reach: whole sub-plan vs parent and children (TCR)", "reach.subplan.full", "full|trsubplan|L3|true", "full|trlocal|L3|true"),
    ("reach: whole plan vs parent and children (TCR)", "reach.global.full", "full|trglobal|L3|true", "full|trlocal|L3|true"),
    ("random tree vs plan topology (GATv2, TCR)", "random.full", "full|gatv2|L3|random", "full|gatv2|L3|true"),
    ("random tree vs plan topology (GATv2, type+rows)", "random.nocost", "nocost|gatv2|L3|random", "nocost|gatv2|L3|true"),
    ("BiGG vs GATv2 (type+rows)", "bigg.nocost", "nocost|bigg|L3|true", "nocost|gatv2|L3|true"),
    ("BiGG vs GATv2 (TCR)", "bigg.full", "full|bigg|L3|true", "full|gatv2|L3|true"),
]
SELECTION = [  # label, slug, workload, a (dir, root), b (dir, root)   ratio > 1 = a picks faster plans
    ("ranking term (PG-stat, GATv2)", "rank.pg", "pg_gatv2_rank1", BASE, "pg_gatv2_rank0", BASE),
    ("ranking term (TCR, GATv2)", "rank.dace", "dace_gatv2_rank1", BASE, "dace_gatv2_rank0", BASE),
    ("ranking term (PG-stat, None)", "rank.pg.none", "pg_none_rank1", BASE, "pg_none_rank0", BASE),
    ("PG-stat vs TCR (GATv2, ranking)", "pg.dace", "pg_gatv2_rank1", BASE, "dace_gatv2_rank1", BASE),
    ("relations: GATv2 vs None (PG-stat, ranking)", "rel.pg", "pg_gatv2_rank1", BASE, "pg_none_rank1", BASE),
    ("relations: GATv2 vs None (TCR, ranking)", "rel.dace", "dace_gatv2_rank1", BASE, "dace_none_rank1", BASE),
    ("one layer vs three (PG-stat, ranking)", "l1.pg", "pg_gatv2L1_rank1", ATTN, "pg_gatv2_rank1", BASE),
    ("one layer vs three (TCR, ranking)", "l1.dace", "dace_gatv2L1_rank1", ATTN, "dace_gatv2_rank1", BASE),
    ("root cost vs cumulative (GATv2, ranking)", "rootcost.gatv2", "rootcost_gatv2_rank1", ATTN, "dace_gatv2_rank1", BASE),
    ("root cost vs cumulative (None, ranking)", "rootcost.none", "rootcost_none_rank1", ATTN, "dace_none_rank1", BASE),
    ("GATv2 vs tree convolution (PG-stat, ranking)", "tcnn.pg", "pg_gatv2_rank1", BASE, "pg_tcnn_rank1", BASE),
]
SEEDS_SEL = (42, 7, 123)


def rel_change(A, B):
    return float(A.mean() / B.mean() - 1.0)


def crossdb_rows(sub: str, split: str):
    runs = load_runs(DEP, sub)
    table = {}
    for r in runs:
        table.setdefault(cfg_name(r["key"]), {})[r["seed"]] = macro(r["preds"])
    dbs = sorted(runs[0]["preds"]) if runs else []
    rows = []
    for label, slug, a, b in CROSSDB:
        if a not in table or b not in table:
            continue
        seeds = sorted(set(table[a]) & set(table[b]))
        for k in LEVELS:
            A = np.array([[table[a][s][k][db] for db in dbs] for s in seeds])
            B = np.array([[table[b][s][k][db] for db in dbs] for s in seeds])
            per_seed = [rel_change(A[i], B[i]) for i in range(len(seeds))]
            diff_db = (A - B).mean(axis=0)
            mean = rel_change(A, B)
            same = int((np.sign(diff_db) == np.sign(mean)).sum()) if mean != 0 else int((diff_db == 0).sum())
            rows.append(dict(study="crossdb", split=split, label=label, slug=slug, metric=f"q{k}", n_seeds=len(seeds),
                             mean_rel=mean, seed_min=min(per_seed), seed_max=max(per_seed),
                             seeds_same_sign=int(sum(np.sign(v) == np.sign(mean) for v in per_seed)),
                             units=len(dbs), units_same_sign=same, unit="databases",
                             per_seed=" ".join(f"{s}:{v:+.3f}" for s, v in zip(seeds, per_seed))))
    return rows


def picks(root, wl, name, seed):
    p = os.path.join(root, wl, f"{name}_s{seed}", "picks.csv")
    return pd.read_csv(p).set_index("query_id") if os.path.exists(p) else None


def selection_rows():
    rows = []
    for wl in ("job112", "tpch52"):
        lat = pd.read_csv(os.path.join(BASE, wl, "pg_gatv2_rank1_s42", "latency_cpu1.csv")).set_index("query_id")
        fold = lat["fold"]
        for label, slug, a, ra, b, rb in SELECTION:
            per_seed, logs = {}, []
            for s in SEEDS_SEL:
                pa, pb = picks(ra, wl, a, s), picks(rb, wl, b, s)
                if pa is None or pb is None:
                    continue
                v = np.log(pb.pick_ms) - np.log(pa.pick_ms)          # > 0: a faster
                per_seed[s] = float(np.exp(v.mean()))
                logs.append(v)
            if not logs:
                continue
            v = pd.concat(logs, axis=1).mean(axis=1)
            mean = float(np.exp(v.mean()))
            by_fold = v.groupby(fold.reindex(v.index)).mean().apply(np.exp)
            by_tmpl = v.groupby(pa["template"].reindex(v.index)).mean().apply(np.exp)
            up = mean >= 1.0
            rows.append(dict(study="selection", split=wl, label=label, slug=slug, metric="ratio", n_seeds=len(per_seed),
                             mean_rel=mean, seed_min=min(per_seed.values()), seed_max=max(per_seed.values()),
                             seeds_same_sign=int(sum((x >= 1.0) == up for x in per_seed.values())),
                             units=int(len(by_fold)), units_same_sign=int(((by_fold >= 1.0) == up).sum()), unit="folds",
                             templates=int(len(by_tmpl)), templates_same_sign=int(((by_tmpl >= 1.0) == up).sum()),
                             per_seed=" ".join(f"{s}:{x:.3f}" for s, x in per_seed.items()),
                             per_fold=" ".join(f"{int(k)}:{x:.3f}" for k, x in by_fold.items())))
    return rows


def online_rows(n_pass=30, converged=10):
    rows = []
    for x, y, label, slug in (("gnto", "bao", "online: GNTO vs Bao-ext (converged passes)", "online.gnto.bao"),
                              ("gnto", "gnto_none", "online: GATv2 vs None (converged passes)", "online.gnto.none"),
                              ("gnto", "gnto_tcr", "online: PG-stat vs TCR (converged passes)", "online.gnto.tcr")):
        per_seed = {}
        for s in range(1, 6):
            fx, fy = (os.path.join(ONLINE, "job112", f"{m}_s{s}", "choices.csv") for m in (x, y))
            if not (os.path.exists(fx) and os.path.exists(fy)):
                continue
            cx, cy = pd.read_csv(fx), pd.read_csv(fy)
            w = lambda c: np.log(c[(c["pass"] >= n_pass - converged) & (c["pass"] < n_pass)].groupby("query_id").ms.apply(lambda v: np.exp(np.log(v).mean())))  # noqa: E731
            per_seed[s] = float(np.exp((w(cy) - w(cx)).mean()))
        if not per_seed:
            continue
        mean = float(np.exp(np.mean(np.log(list(per_seed.values())))))
        rows.append(dict(study="online", split="job112", label=label, slug=slug, metric="ratio", n_seeds=len(per_seed),
                         mean_rel=mean, seed_min=min(per_seed.values()), seed_max=max(per_seed.values()),
                         seeds_same_sign=int(sum((x >= 1.0) == (mean >= 1.0) for x in per_seed.values())),
                         units=len(per_seed), units_same_sign=int(sum((x >= 1.0) == (mean >= 1.0) for x in per_seed.values())), unit="seeds",
                         per_seed=" ".join(f"{s}:{x:.3f}" for s, x in per_seed.items())))
    return rows


def main() -> None:
    rows = crossdb_rows("crossdb", "published") + crossdb_rows("crossdb_rand1", "split2") + selection_rows() + online_rows()
    df = pd.DataFrame(rows)
    out = os.path.join(DEP, "tables", "stability.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    show = df[(df.metric != "q99")]
    print(show[["study", "split", "label", "metric", "n_seeds", "mean_rel", "seed_min", "seed_max", "seeds_same_sign", "units_same_sign", "units", "per_seed"]].round(3).to_string(index=False))
    print("tail (q99):")
    print(df[df.metric == "q99"][["split", "label", "mean_rel", "seed_min", "seed_max", "seeds_same_sign", "units_same_sign"]].round(3).to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
