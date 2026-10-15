"""Aggregate the cross-database dependency study (run_crossdb_dep.py outputs).

Reads finished runs only; writes CSV tables under ``<root>/tables``:

    crossdb_configs.csv       one row per configuration: macro Q-error (mean over the 10 unseen databases of the
                              per-database quantile), mean and SD over seeds, parameters
    crossdb_contrasts.csv     paired differences between configurations with a database-level bootstrap interval
    crossdb_interventions.csv models trained on the true plan topology, scored with a random tree / without edges
    crossdb_by_joins.csv      pooled Q-error by the number of joins of the plan
    crossdb_per_db.csv        per-database medians

Estimands: quantiles are computed per database and seed, averaged over databases,
then over seeds; intervals resample the 10 test databases (same draw for all seeds and both configurations).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.dace import load_plans, plans_path  # noqa: E402

LEVELS = (50, 90, 99)
NAME = re.compile(r"^(?P<feat>full|nocost|norows|typeonly|costrows|costonly|rowsonly|resid|residrows|residonly|residtype|rootcost|localcost|rootcum|localcostshuf|fulln|nocostn|rootcostn|rootcumn|localcostn)_(?P<conv>[a-z0-9]+?)(?P<vn>vn)?_L(?P<L>\d+)_(?P<topo>true|random)_s(?P<seed>\d+)$")


def qerr(pred, actual):
    p, t = np.maximum(pred, 1e-7), np.maximum(actual, 1e-7)
    return np.maximum(p / t, t / p)


def n_joins(node) -> int:
    own = 1 if ("Join" in node.get("Node Type", "") or node.get("Node Type") == "Nested Loop") else 0
    return own + sum(n_joins(ch) for ch in node.get("Plans", []))


def load_runs(root: str, sub: str = "crossdb"):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, sub, "*"))):
        m = NAME.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        key = dict(feat=m["feat"], conv=m["conv"] + ("+vn" if m["vn"] else ""), L=int(m["L"]), topo=m["topo"])
        preds = {db: pd.read_csv(os.path.join(d, "predictions", f"{db}.csv")) for db in s["split"]["test_dbs"]}
        runs.append(dict(key=key, seed=int(m["seed"]), params=s["params"]["total"], preds=preds, dir=d,
                         train_s=s["timing_s"].get("train_s")))
    return runs


def macro(preds, col="pred_ms"):
    """{level: {db: quantile}}"""
    return {k: {db: float(np.percentile(qerr(df[col].values, df["actual_ms"].values), k)) for db, df in preds.items()}
            for k in LEVELS}


def cfg_name(key):
    return f"{key['feat']}|{key['conv']}|L{key['L']}|{key['topo']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "analysis", "dependency_20261003"))
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--runs", default="crossdb", help="sub-directory of --root that holds the runs (crossdb, crossdb_rand1)")
    ap.add_argument("--tables", default="tables", help="sub-directory of --root for the CSV tables")
    a = ap.parse_args()
    out = os.path.join(a.root, a.tables)
    os.makedirs(out, exist_ok=True)
    runs = load_runs(a.root, a.runs)
    if not runs:
        raise SystemExit("no finished runs")
    dbs = sorted(runs[0]["preds"])
    by_cfg = {}
    for r in runs:
        by_cfg.setdefault(cfg_name(r["key"]), []).append(r)

    # ---- per configuration ------------------------------------------------------------------
    rows, per_db_rows, table = [], [], {}
    for name, rs in by_cfg.items():
        seeds = sorted(r["seed"] for r in rs)
        m = {r["seed"]: macro(r["preds"]) for r in rs}                       # seed -> level -> db -> q
        table[name] = m
        row = dict(config=name, **rs[0]["key"], n_seeds=len(rs), seeds=" ".join(map(str, seeds)), params=rs[0]["params"])
        for k in LEVELS:
            per_seed = [np.mean([m[s][k][db] for db in dbs]) for s in seeds]
            row[f"q{k}"] = float(np.mean(per_seed))
            row[f"q{k}_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
        # pooled geometric mean of the Q-error (mean |log q|), macro over databases
        gm = [np.mean([np.exp(np.mean(np.log(qerr(r["preds"][db]["pred_ms"].values, r["preds"][db]["actual_ms"].values))))
                       for db in dbs]) for r in rs]
        row["geo_q"] = float(np.mean(gm))
        row["train_min"] = float(np.mean([r["train_s"] for r in rs if r["train_s"]])) / 60
        rows.append(row)
        for db in dbs:
            per_db_rows.append(dict(config=name, db=db, **{f"q{k}": float(np.mean([m[s][k][db] for s in seeds])) for k in LEVELS}))
    cfg = pd.DataFrame(rows).sort_values(["feat", "topo", "L", "conv"])
    cfg.to_csv(os.path.join(out, "crossdb_configs.csv"), index=False)
    pd.DataFrame(per_db_rows).to_csv(os.path.join(out, "crossdb_per_db.csv"), index=False)

    # models trained on the true topology, scored with a random tree / without edges (test-time interventions)
    for name, rs in list(by_cfg.items()):
        if "pred_ms_random" in next(iter(rs[0]["preds"].values())).columns:
            for mode in ("random", "noedges"):
                table[f"{name}@{mode}"] = {r["seed"]: macro(r["preds"], f"pred_ms_{mode}") for r in rs}

    # ---- paired contrasts -------------------------------------------------------------------
    rng = np.random.default_rng(20261003)
    draws = rng.integers(0, len(dbs), size=(a.draws, len(dbs)))

    def contrast(name_a, name_b, label):
        if name_a not in table or name_b not in table:
            return []
        seeds = sorted(set(table[name_a]) & set(table[name_b]))
        res = []
        for k in LEVELS:
            A = np.array([[table[name_a][s][k][db] for db in dbs] for s in seeds])   # seeds x dbs
            B = np.array([[table[name_b][s][k][db] for db in dbs] for s in seeds])
            diff = (A - B).mean(axis=0)                                              # per database, seed-averaged
            boot = diff[draws].mean(axis=1)
            rel = A.mean() / B.mean() - 1
            relboot = A.mean(axis=0)[draws].mean(axis=1) / B.mean(axis=0)[draws].mean(axis=1) - 1
            res.append(dict(contrast=label, a=name_a, b=name_b, metric=f"q{k}", n_seeds=len(seeds),
                            a_mean=float(A.mean()), b_mean=float(B.mean()), diff=float(diff.mean()),
                            lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                            rel=float(rel), rel_lo=float(np.percentile(relboot, 2.5)), rel_hi=float(np.percentile(relboot, 97.5)),
                            dbs_a_lower=int((diff < 0).sum())))
        return res

    cons = []
    for feat in ("full", "nocost", "norows", "typeonly", "costrows", "costonly", "rowsonly", "resid", "residrows", "residonly", "residtype", "rootcost", "localcost", "rootcum", "localcostshuf", "fulln", "rootcumn", "localcostn"):
        base = f"{feat}|none|L3|true"
        for conv in ("gcn", "gat", "gatv2", "tcnn", "gatv2+vn"):
            cons += contrast(f"{feat}|{conv}|L3|true", base, f"{conv} vs none ({feat})")
        cons += contrast(f"{feat}|gatv2|L3|true", f"{feat}|gcn|L3|true", f"gatv2 vs gcn ({feat})")
        cons += contrast(f"{feat}|tcnn|L3|true", f"{feat}|gatv2|L3|true", f"tcnn vs gatv2 ({feat})")
        for conv in ("gcn", "gatv2"):
            cons += contrast(f"{feat}|{conv}|L3|random", base, f"{conv} random tree vs none ({feat})")
            cons += contrast(f"{feat}|{conv}|L3|true", f"{feat}|{conv}|L3|random", f"{conv} true vs random tree ({feat})")
        for L in (1, 2, 5):
            cons += contrast(f"{feat}|gatv2|L{L}|true", base, f"gatv2 L{L} vs none ({feat})")
            cons += contrast(f"{feat}|gatv2|L{L}|true", f"{feat}|gatv2|L3|true", f"gatv2 L{L} vs L3 ({feat})")
        cons += contrast(f"{feat}|gatv2+vn|L3|true", f"{feat}|gatv2|L3|true", f"virtual node vs gatv2 ({feat})")
        # reach control (analysis/attn): one attention layer type with three reaches
        for reach in ("trlocal", "trsubplan", "trglobal"):
            cons += contrast(f"{feat}|{reach}|L3|true", base, f"{reach} vs none ({feat})")
            cons += contrast(f"{feat}|{reach}|L3|true", f"{feat}|gatv2|L3|true", f"{reach} vs gatv2 ({feat})")
            cons += contrast(f"{feat}|{reach}|L3|true", f"{feat}|none|L6|true", f"{reach} vs none6 ({feat})")
        cons += contrast(f"{feat}|trsubplan|L3|true", f"{feat}|trlocal|L3|true", f"trsubplan vs trlocal ({feat})")
        cons += contrast(f"{feat}|trglobal|L3|true", f"{feat}|trlocal|L3|true", f"trglobal vs trlocal ({feat})")
        # how much a deeper reach could still gain over one local layer (paired, same databases)
        cons += contrast(f"{feat}|gatv2|L5|true", f"{feat}|gatv2|L1|true", f"gatv2 L5 vs gatv2 L1 ({feat})")
        cons += contrast(f"{feat}|gatv2|L3|true", f"{feat}|gatv2|L1|true", f"gatv2 L3 vs gatv2 L1 ({feat})")
        cons += contrast(f"{feat}|gatv2+vn|L3|true", f"{feat}|gatv2|L1|true", f"virtual node vs gatv2 L1 ({feat})")
        cons += contrast(f"{feat}|gat|L3|true", f"{feat}|gatv2|L3|true", f"gat vs gatv2 ({feat})")
        for conv in ("gcn", "gatv2"):
            for mode in ("random", "noedges"):
                cons += contrast(f"{feat}|{conv}|L3|true@{mode}", base, f"{conv} test{mode} vs none ({feat})")
                cons += contrast(f"{feat}|{conv}|L3|true@{mode}", f"{feat}|{conv}|L3|true", f"{conv} test{mode} vs {conv} ({feat})")
        cons += contrast(f"{feat}|none|L6|true", base, f"none6 vs none ({feat})")
        cons += contrast(f"{feat}|gatv2|L3|true", f"{feat}|none|L6|true", f"gatv2 vs none6 ({feat})")
        cons += contrast(f"{feat}|gatv2|L1|true", f"{feat}|none|L6|true", f"gatv2 L1 vs none6 ({feat})")
        cons += contrast(f"{feat}|gcn|L6|true", f"{feat}|none|L6|true", f"gcn6 vs none6 ({feat})")
        cons += contrast(f"{feat}|gatv2|L3|true", f"{feat}|gcn|L6|true", f"gatv2 vs gcn6 ({feat})")
        cons += contrast(f"{feat}|gatv2|L3|random", f"{feat}|none|L6|true", f"gatv2 random tree vs none6 ({feat})")
        # Reqo's BiGG encoder in the same harness (analysis/baselines/run_crossdb_reqo.py)
        cons += contrast(f"{feat}|bigg|L3|true", base, f"bigg vs none ({feat})")
        cons += contrast(f"{feat}|bigg|L3|true", f"{feat}|none|L6|true", f"bigg vs none6 ({feat})")
        cons += contrast(f"{feat}|gatv2|L3|true", f"{feat}|bigg|L3|true", f"gatv2 vs bigg ({feat})")
        cons += contrast(f"{feat}|reqo|L3|true", f"{feat}|gatv2|L3|true", f"reqo vs gatv2 ({feat})")
    for conv in ("none", "gcn", "gatv2", "tcnn"):
        cons += contrast(f"nocost|{conv}|L3|true", f"full|{conv}|L3|true", f"nocost vs full ({conv})")
        cons += contrast(f"typeonly|{conv}|L3|true", f"nocost|{conv}|L3|true", f"typeonly vs nocost ({conv})")
        # input ladder without the operator type: cost + rows, and type + cost
        cons += contrast(f"costrows|{conv}|L3|true", f"full|{conv}|L3|true", f"costrows vs full ({conv})")
        cons += contrast(f"costrows|{conv}|L3|true", f"nocost|{conv}|L3|true", f"costrows vs nocost ({conv})")
        cons += contrast(f"norows|{conv}|L3|true", f"full|{conv}|L3|true", f"norows vs full ({conv})")
        cons += contrast(f"norows|{conv}|L3|true", f"nocost|{conv}|L3|true", f"norows vs nocost ({conv})")
        cons += contrast(f"costrows|{conv}|L3|true", f"norows|{conv}|L3|true", f"costrows vs norows ({conv})")
        for fa in ("typeonly", "costonly", "rowsonly"):                      # the single inputs against TCR and against each pair
            cons += contrast(f"{fa}|{conv}|L3|true", f"full|{conv}|L3|true", f"{fa} vs full ({conv})")
        cons += contrast(f"costonly|{conv}|L3|true", f"costrows|{conv}|L3|true", f"costonly vs costrows ({conv})")
        cons += contrast(f"rowsonly|{conv}|L3|true", f"costrows|{conv}|L3|true", f"rowsonly vs costrows ({conv})")
        cons += contrast(f"costonly|{conv}|L3|true", f"norows|{conv}|L3|true", f"costonly vs norows ({conv})")
        # the residual in place of the cumulative cost, without the root cost
        cons += contrast(f"resid|{conv}|L3|true", f"full|{conv}|L3|true", f"resid vs full ({conv})")
        cons += contrast(f"resid|{conv}|L3|true", f"nocost|{conv}|L3|true", f"resid vs nocost ({conv})")
        cons += contrast(f"resid|{conv}|L3|true", f"localcost|{conv}|L3|true", f"resid vs localcost ({conv})")
        cons += contrast(f"residrows|{conv}|L3|true", f"costrows|{conv}|L3|true", f"residrows vs costrows ({conv})")
        cons += contrast(f"residrows|{conv}|L3|true", f"resid|{conv}|L3|true", f"residrows vs resid ({conv})")
        cons += contrast(f"residonly|{conv}|L3|true", f"costonly|{conv}|L3|true", f"residonly vs costonly ({conv})")
        cons += contrast(f"residtype|{conv}|L3|true", f"norows|{conv}|L3|true", f"residtype vs norows ({conv})")
        # root-cost controls: the whole-plan cost without the per-operator sub-plan summary
        cons += contrast(f"rootcost|{conv}|L3|true", f"full|{conv}|L3|true", f"rootcost vs full ({conv})")
        cons += contrast(f"localcost|{conv}|L3|true", f"full|{conv}|L3|true", f"localcost vs full ({conv})")
        cons += contrast(f"nocost|{conv}|L3|true", f"rootcost|{conv}|L3|true", f"nocost vs rootcost ({conv})")
        cons += contrast(f"localcost|{conv}|L3|true", f"rootcost|{conv}|L3|true", f"localcost vs rootcost ({conv})")
        # root cost + cumulative cost: is the residual better than the cumulative cost at equal whole-plan information?
        cons += contrast(f"rootcum|{conv}|L3|true", f"full|{conv}|L3|true", f"rootcum vs full ({conv})")
        cons += contrast(f"rootcum|{conv}|L3|true", f"rootcost|{conv}|L3|true", f"rootcum vs rootcost ({conv})")
        cons += contrast(f"localcost|{conv}|L3|true", f"rootcum|{conv}|L3|true", f"localcost vs rootcum ({conv})")
        # controls: shuffled residuals; plan size n given to every configuration
        cons += contrast(f"localcostshuf|{conv}|L3|true", f"rootcum|{conv}|L3|true", f"localcostshuf vs rootcum ({conv})")
        cons += contrast(f"localcost|{conv}|L3|true", f"localcostshuf|{conv}|L3|true", f"localcost vs localcostshuf ({conv})")
        cons += contrast(f"localcostn|{conv}|L3|true", f"rootcumn|{conv}|L3|true", f"localcostn vs rootcumn ({conv})")
        cons += contrast(f"localcostn|{conv}|L3|true", f"fulln|{conv}|L3|true", f"localcostn vs fulln ({conv})")
        cons += contrast(f"rootcumn|{conv}|L3|true", f"fulln|{conv}|L3|true", f"rootcumn vs fulln ({conv})")
        cons += contrast(f"fulln|{conv}|L3|true", f"full|{conv}|L3|true", f"fulln vs full ({conv})")
        cons += contrast(f"localcostn|{conv}|L3|true", f"localcost|{conv}|L3|true", f"localcostn vs localcost ({conv})")
    for conv in ("trlocal", "trsubplan", "trglobal", "bigg"):      # the residual through the Transformer reaches and BiGG
        cons += contrast(f"resid|{conv}|L3|true", f"full|{conv}|L3|true", f"resid vs full ({conv})")
    for feat in ("nocost", "resid"):                                 # edge direction and self-loops of GATv2
        for v in ("gatv2up", "gatv2down", "gatv2noself"):
            cons += contrast(f"{feat}|{v}|L3|true", f"{feat}|gatv2|L3|true", f"{v} vs gatv2 ({feat})")
            cons += contrast(f"{feat}|{v}|L3|true", f"{feat}|none|L3|true", f"{v} vs none ({feat})")
    for conv in ("none", "gcn", "gatv2", "tcnn", "trlocal", "trsubplan", "trglobal", "bigg"):
        cons += contrast(f"full|{conv}|L3|true", f"resid|{conv}|L3|true", f"full vs resid ({conv})")   # TCR as the ablation of TLR
    # the parameter-matched six-layer bag under the decomposed inputs
    for fa, fb in (("rootcost", "full"), ("localcost", "full"), ("rootcum", "full"), ("localcost", "rootcum"), ("localcost", "rootcost")):
        cons += contrast(f"{fa}|none|L6|true", f"{fb}|none|L6|true", f"{fa} vs {fb} (none6)")
    # cross contrasts: GATv2 on the local / root cost vs None on the cumulative cost (full)
    cons += contrast("localcost|gatv2|L3|true", "full|none|L3|true", "gatv2 localcost vs none full (x)")
    cons += contrast("rootcost|gatv2|L3|true", "full|none|L3|true", "gatv2 rootcost vs none full (x)")
    pd.DataFrame(cons).to_csv(os.path.join(out, "crossdb_contrasts.csv"), index=False)

    # ---- test-time interventions ---------------------------------------------------------------
    irows = []
    for name, rs in by_cfg.items():
        if "pred_ms_random" not in next(iter(rs[0]["preds"].values())).columns:
            continue
        row = dict(config=name, n_seeds=len(rs))
        for mode, col in (("true", "pred_ms"), ("random", "pred_ms_random"), ("noedges", "pred_ms_noedges")):
            for k in LEVELS:
                row[f"{mode}_q{k}"] = float(np.mean([np.mean(list(macro(r["preds"], col)[k].values())) for r in rs]))
        irows.append(row)
    pd.DataFrame(irows).to_csv(os.path.join(out, "crossdb_interventions.csv"), index=False)

    # ---- by number of joins ----------------------------------------------------------------------
    joins = {db: np.array([n_joins(p["Plan"]) for p in load_plans(plans_path(db))]) for db in dbs}
    buckets = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 9)]
    jrows = []
    for name, rs in by_cfg.items():
        for label, lo, hi in buckets:
            q50, q90, n = [], [], 0
            for r in rs:
                qs = np.concatenate([qerr(df["pred_ms"].values, df["actual_ms"].values)[(joins[db][df["plan_id"].values] >= lo)
                                                                                         & (joins[db][df["plan_id"].values] <= hi)]
                                     for db, df in r["preds"].items()])
                q50.append(np.median(qs)), q90.append(np.percentile(qs, 90))
                n = len(qs)
            jrows.append(dict(config=name, joins=label, n_plans=n, q50=float(np.mean(q50)), q90=float(np.mean(q90))))
    pd.DataFrame(jrows).to_csv(os.path.join(out, "crossdb_by_joins.csv"), index=False)

    # paired relative change of the pooled median against None, per number of joins; whole databases are resampled
    jdraws = np.random.default_rng(20261004).integers(0, len(dbs), size=(1000, len(dbs)))

    def bucket_arrays(name, lo, hi):
        """seed -> list over databases of the Q-errors of the plans in the bucket"""
        res = {}
        for r in by_cfg[name]:
            res[r["seed"]] = [qerr(r["preds"][db]["pred_ms"].values, r["preds"][db]["actual_ms"].values)[
                (joins[db][r["preds"][db]["plan_id"].values] >= lo) & (joins[db][r["preds"][db]["plan_id"].values] <= hi)]
                for db in dbs]
        return res

    jc = []
    for feat in ("full", "nocost", "typeonly"):
        base = f"{feat}|none|L3|true"
        if base not in by_cfg:
            continue
        for conv in ("gcn", "gatv2", "tcnn"):
            name = f"{feat}|{conv}|L3|true"
            if name not in by_cfg:
                continue
            for label, lo, hi in buckets:
                A, B = bucket_arrays(name, lo, hi), bucket_arrays(base, lo, hi)
                seeds = sorted(set(A) & set(B))
                if not seeds:
                    continue

                def med(arrs, idx):
                    return np.mean([np.median(np.concatenate([arrs[s][i] for i in idx])) for s in seeds])
                full_idx = np.arange(len(dbs))
                a0, b0 = med(A, full_idx), med(B, full_idx)
                boot = np.array([med(A, ix) / med(B, ix) - 1 for ix in jdraws])
                jc.append(dict(feat=feat, conv=conv, joins=label, n_seeds=len(seeds), n_plans=int(sum(len(x) for x in A[seeds[0]])),
                               q50=float(a0), none_q50=float(b0), rel=float(a0 / b0 - 1),
                               lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    pd.DataFrame(jc).to_csv(os.path.join(out, "crossdb_joins_contrasts.csv"), index=False)

    # ---- by join algorithm: a nested loop multiplies the work of its inner side by the rows of its outer side,
    #      i.e. the cost of one operator depends on another operator; hash and merge joins add up ---------------
    def kinds(node, acc):
        acc.add(node.get("Node Type", ""))
        for ch in node.get("Plans", []):
            kinds(ch, acc)
        return acc
    strata = {}
    for db in dbs:
        lab = []
        for p in load_plans(plans_path(db)):
            k = kinds(p["Plan"], set())
            lab.append("nested loop" if "Nested Loop" in k else ("hash/merge only" if ("Hash Join" in k or "Merge Join" in k) else "no join"))
        strata[db] = np.array(lab)
    nrows = []
    for name, rs in by_cfg.items():
        for label in ("no join", "hash/merge only", "nested loop"):
            q50, q90, gq, n = [], [], [], 0
            for r in rs:
                qs = np.concatenate([qerr(df["pred_ms"].values, df["actual_ms"].values)[strata[db][df["plan_id"].values] == label]
                                     for db, df in r["preds"].items()])
                q50.append(np.median(qs)), q90.append(np.percentile(qs, 90)), gq.append(np.exp(np.log(qs).mean()))
                n = len(qs)
            nrows.append(dict(config=name, stratum=label, n_plans=n, q50=float(np.mean(q50)), q90=float(np.mean(q90)),
                              geo_q=float(np.mean(gq))))
    pd.DataFrame(nrows).to_csv(os.path.join(out, "crossdb_by_join_kind.csv"), index=False)

    pd.set_option("display.width", 250)
    print(cfg[["config", "n_seeds", "params", "q50", "q50_sd", "q90", "q90_sd", "q99", "geo_q", "train_min"]].round(3).to_string(index=False))
    print()
    c = pd.DataFrame(cons)
    if len(c):
        print(c[c.metric.isin(["q50", "q90"])][["contrast", "metric", "n_seeds", "a_mean", "b_mean", "diff", "lo", "hi", "rel",
                                                "dbs_a_lower"]].round(3).to_string(index=False))
    print()
    if irows:
        print(pd.DataFrame(irows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
