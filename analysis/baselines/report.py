"""Summary of the Reqo / BiGG and zero-shot baselines next to DACE and GNTO.

Reads finished runs only:
    results/analysis/dependency_20261003/crossdb[_rand1]/{full,nocost}_bigg_L3_true_s*     BiGG encoder in the GNTO harness
    results/analysis/dependency_20261003/crossdb_zeroshot/zeroshot_<split>_s*               released zero-shot model
    results/analysis/dependency_20261003/steering/<workload>/{reqo,reqoval}_*, *_bigg_*    Reqo and BiGG in plan selection
    results/main_runs_stats/crossdb_per_database.csv                                        DACE and GNTO
and the tables written by analysis/dependency_study/analyze_crossdb.py / analyze_steering.py (run those first).

    python analysis/baselines/report.py            # prints markdown, writes results/analysis/dependency_20261003/tables/baselines_*.csv
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEP = os.path.join(ROOT, "results", "analysis", "dependency_20261003")
AUDIT = os.path.join(ROOT, "results", "main_runs_stats")
TLR = os.path.join(ROOT, "results", "analysis", "tlr_20261008")               # GNTO with TLR inputs
LEVELS = (50, 90, 99)


def qerr(pred, actual):
    p, t = np.maximum(pred, 1e-7), np.maximum(actual, 1e-7)
    return np.maximum(p / t, t / p)


def boot(diff, draws):
    b = diff[draws].mean(axis=1)
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def crossdb_zeroshot(rows):
    audit = pd.read_csv(os.path.join(AUDIT, "crossdb_per_database.csv"))
    for split, name in (("paper1", "paper"), ("rand1", "rand1")):
        runs = sorted(glob.glob(os.path.join(DEP, "crossdb_zeroshot", f"zeroshot_{name}_s*", "summary.json")))
        if not runs:
            continue
        per = []                                                    # seed x db x level
        for r in runs:
            s = json.load(open(r))
            dbs = s["split"]["test_dbs"]
            per.append([[np.percentile(qerr(*(pd.read_csv(os.path.join(os.path.dirname(r), "predictions", f"{db}.csv"))
                                              [["pred_ms", "actual_ms"]].to_numpy().T)), k) for k in LEVELS] for db in dbs])
        z = np.mean(per, axis=0)                                    # db x level, mean over seeds
        a = audit[audit.split == split]
        ref = {m: a[a.model == m].groupby("database")[["q50", "q90", "q99"]].mean().reindex(dbs).to_numpy() for m in ("dace", "gnto")}
        tlr_runs = sorted(glob.glob(os.path.join(TLR, f"gnto_resid_{split}_s*", "summary.json")))
        if tlr_runs:                                                    # GNTO's own inputs (TLR): per-database percentiles, mean over seeds
            ref["gntotlr"] = np.mean([[[np.percentile(qerr(*(pd.read_csv(os.path.join(os.path.dirname(r), "predictions", f"{db}.csv"))
                                                              [["pred_ms", "actual_ms"]].to_numpy().T)), k) for k in LEVELS] for db in dbs]
                                      for r in tlr_runs], axis=0)
        draws = np.random.default_rng(20261003).integers(0, len(dbs), size=(20000, len(dbs)))
        params = json.load(open(runs[0]))["params"]["total"]
        for j, k in enumerate(LEVELS):
            row = dict(split=split, metric=f"q{k}", zeroshot=float(z[:, j].mean()), dace=float(ref["dace"][:, j].mean()),
                       gnto=float(ref["gnto"][:, j].mean()), zeroshot_seeds=len(runs), zeroshot_params=params)
            if "gntotlr" in ref:
                row["gntotlr"] = float(ref["gntotlr"][:, j].mean())
                row["gntotlr_seeds"] = len(tlr_runs)
            for m in ref:
                d, lo, hi = boot(ref[m][:, j] - z[:, j], draws)     # < 0: the reference model (DACE / GNTO) is more accurate
                row.update({f"{m}_minus_zeroshot": d, f"{m}_lo": lo, f"{m}_hi": hi,
                            f"{m}_dbs_lower": int((ref[m][:, j] < z[:, j]).sum())})
            rows.append(row)
        pd.DataFrame({"database": dbs, "zeroshot_q50": z[:, 0], "dace_q50": ref["dace"][:, 0], "gnto_q50": ref["gnto"][:, 0]}
                     ).to_csv(os.path.join(DEP, "tables", f"baselines_zeroshot_per_db_{split}.csv"), index=False)


def main() -> None:
    out = os.path.join(DEP, "tables")
    pd.set_option("display.width", 250)
    print("# Baselines\n")

    rows = []
    crossdb_zeroshot(rows)
    if rows:
        z = pd.DataFrame(rows)
        z.to_csv(os.path.join(out, "baselines_zeroshot.csv"), index=False)
        print("## Unseen databases: zero-shot (released code) against DACE and GNTO, macro Q-error over the test databases\n")
        for _, r in z.iterrows():
            print(f"- {r.split} {r.metric}: zero-shot {r.zeroshot:.3f} ({int(r.zeroshot_seeds)} seed(s), {int(r.zeroshot_params):,} params) | "
                  f"DACE {r.dace:.3f} | GNTO {r.gnto:.3f} | GNTO - zero-shot {r.gnto_minus_zeroshot:+.3f} "
                  f"[{r.gnto_lo:+.3f}, {r.gnto_hi:+.3f}] (GNTO lower on {int(r.gnto_dbs_lower)}/10) | DACE - zero-shot "
                  f"{r.dace_minus_zeroshot:+.3f} [{r.dace_lo:+.3f}, {r.dace_hi:+.3f}]")
        print()

    for tables, label in (("tables", "published split"), ("tables_rand1", "random split 1")):
        p = os.path.join(DEP, tables, "crossdb_configs.csv")
        if not os.path.exists(p):
            continue
        cfg = pd.read_csv(p, dtype={"topo": str})
        sel = cfg[cfg.config.str.contains(r"\|(?:bigg|reqo)\|") | cfg.config.isin(
            ["full|none|L3|true", "full|none|L6|true", "full|gatv2|L3|true", "nocost|none|L3|true", "nocost|none|L6|true",
             "nocost|gatv2|L3|true", "full|gcn|L6|true", "nocost|gcn|L6|true"])]
        if not sel.config.str.contains("bigg|reqo").any():
            continue
        print(f"## Unseen databases, {label}: BiGG encoder (Reqo) inside the GNTO harness\n")
        print(sel[["config", "n_seeds", "params", "q50", "q90", "q99"]].round(3).to_string(index=False))
        con = pd.read_csv(os.path.join(DEP, tables, "crossdb_contrasts.csv"))
        con = con[con.contrast.str.contains("bigg|reqo")]
        for _, r in con.iterrows():
            print(f"- {r.contrast} {r.metric}: {100 * r.rel:+.1f}% [{100 * r.rel_lo:+.1f}, {100 * r.rel_hi:+.1f}] ({int(r.n_seeds)} seeds)")
        print()

    p = os.path.join(out, "steering_configs.csv")
    if os.path.exists(p):
        st = pd.read_csv(p)
        keep = st[st.config.str.contains("reqo|bigg|bao_|limao_|pg_cost|oracle") | st.config.isin(
            ["dace_gatv2_rank0", "dace_gatv2_rank1", "pg_gatv2_rank0", "pg_gatv2_rank1", "dace_none_rank0", "pg_none_rank1"])]
        print("## Plan selection (G: geometric-mean speed-up over PostgreSQL's default plan)\n")
        cols = [c for c in ("workload", "config", "n_seeds", "geo", "geo_lo", "geo_hi", "total", "reg2x", "spearman", "q50") if c in keep]
        print(keep[cols].round(2).sort_values(["workload", "geo"], ascending=[True, False]).to_string(index=False))
        b = pd.read_csv(os.path.join(out, "steering_vs_baselines.csv"))
        b = b[b.baseline.str.contains("reqo") & b.config.isin(["pg_gatv2_rank1", "dace_gatv2_rank0", "dace_gatv2_rank1"])]
        print("\nGNTO against Reqo, ratio of the selected plans' latencies (> 1: GNTO faster):")
        for _, r in b.iterrows():
            print(f"- {r.workload} {r.config} vs {r.baseline}: {r.ratio:.2f} [{r.lo:.2f}, {r.hi:.2f}]")
        rel = pd.read_csv(os.path.join(out, "steering_relation.csv"))
        rel = rel[rel.conv.str.contains("bigg") & (rel.objective == "pooled")]
        print("\nBiGG encoder in the factorial, pooled over objectives (> 1: first selects faster plans):")
        for _, r in rel.iterrows():
            what = "GATv2 vs BiGG" if r.conv == "gatv2_vs_bigg" else "BiGG vs None"
            print(f"- {r.workload} {r.features}: {what} {r.ratio:.2f} [{r.lo:.2f}, {r.hi:.2f}]")


if __name__ == "__main__":
    main()
