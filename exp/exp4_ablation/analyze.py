"""Exp4 ablation tables: residual x depth x width of the GNTO graph encoder.

Reads the runs produced by ``run_ablations.sh`` (groups ``res``, ``width``, ``seeds``) and writes
``results/exp4_ablation/tables/ablation_res_depth_curves/``:

    all_runs.csv                  one row per run (selected checkpoint + selection-free metrics)
    seeds3.csv                    the three 3-seed groups, mean +- std
    paired_tests.csv              per-query Wilcoxon tests between those groups
    overview_by_depth_width.csv   depth x width, validation split and both held-out workloads
    summary.md                    all of the above as Markdown

``last3_*`` = mean of the val evaluations at epochs 80 / 90 / 99, i.e. the end of training with no
checkpoint selection; every other column comes from the checkpoint the protocol picked (best val Q90),
which is also the only one the synthetic / JOB-light predictions exist for.

    python exp/exp4_ablation/analyze.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp1_qerror"))   # the ablations run Exp1's protocol
sys.path.insert(0, os.path.join(HERE, ".."))
from common import GNTO_PATH  # noqa: E402
from _shared import md  # noqa: E402

RESULTS = os.path.join(GNTO_PATH, "results", "exp4_ablation")

OUT_NAME = "ablation_res_depth_curves"
SEED_GROUPS = {
    "default (256/128/64, L3, linear res)": ["gnto", "gnto_s7", "gnto_s123"],
    "w64 L5 identity res": ["gnto_w64_L5", "gnto_w64_L5_s7", "gnto_w64_L5_s123"],
    "w64 L5 no res": ["gnto_w64_L5_nores", "gnto_w64_L5_nores_s7", "gnto_w64_L5_nores_s123"],
}
# (run, width, layers, residual, seed) - the runs this analysis covers
RUNS = [("gnto", "256/128/64", 3, "linear", 42), ("gnto_s7", "256/128/64", 3, "linear", 7),
        ("gnto_s123", "256/128/64", 3, "linear", 123), ("gnto_nores", "256/128/64", 3, "none", 42)]
RUNS += [(f"gnto_w64_L{L}", "64", L, "identity", 42) for L in (3, 5, 7, 9, 12, 16, 24)]
RUNS += [(f"gnto_w64_L{L}_nores", "64", L, "none", 42) for L in (3, 5, 7, 9, 24)]
RUNS += [("gnto_w64_L5_s7", "64", 5, "identity", 7), ("gnto_w64_L5_s123", "64", 5, "identity", 123),
         ("gnto_w64_L5_nores_s7", "64", 5, "none", 7), ("gnto_w64_L5_nores_s123", "64", 5, "none", 123),
         ("gnto_w32_L24", "32", 24, "identity", 42), ("gnto_w32_L32", "32", 32, "identity", 42),
         ("gnto_w32_L48", "32", 48, "identity", 42), ("gnto_w16_L64", "16", 64, "identity", 42)]
WORKLOADS = ("val", "synthetic", "job-light")


def geo(root: str, run: str, workload: str) -> float:
    return float(np.exp(np.log(pd.read_csv(f"{root}/{run}/predictions/{workload}.csv").qerror.values).mean()))


def metrics(root: str, run: str) -> dict:
    s = json.load(open(f"{root}/{run}/summary.json"))
    q = s["qerror"]
    t = pd.read_csv(f"{root}/{run}/training_log.csv")
    last3 = t[t.val_q50.notna()].tail(3)
    jl = pd.read_csv(f"{root}/{run}/predictions/job-light.csv")
    j3 = jl[jl.n_joins >= 3].qerror.values
    return dict(
        params=s["params"]["total"], best_epoch=s["train"]["best_epoch"], final_train_loss=t.loss.iloc[-1],
        last3_val_q50=last3.val_q50.mean(), last3_val_q90=last3.val_q90.mean(),
        last3_val_q99=last3.val_q99.mean(), last3_val_mean=last3.val_q_mean.mean(),
        val_geo=geo(root, run, "val"), val_q50=q["val"]["q50"], val_q90=q["val"]["q90"], val_q99=q["val"]["q99"],
        syn_geo=geo(root, run, "synthetic"), syn_q50=q["synthetic"]["q50"], syn_q90=q["synthetic"]["q90"],
        syn_q99=q["synthetic"]["q99"], jl_geo=geo(root, run, "job-light"), jl_q50=q["job-light"]["q50"],
        jl_q90=q["job-light"]["q90"], jl_q99=q["job-light"]["q99"],
        jl3_geo=float(np.exp(np.log(j3).mean())), jl3_q90=float(np.quantile(j3, .9)))


def paired(root: str, a: list, b: list, workload: str, mask=None) -> dict:
    """Per-query log Q-error averaged over the seeds of each group, then Wilcoxon signed-rank."""
    la = np.mean([np.log(pd.read_csv(f"{root}/{r}/predictions/{workload}.csv").qerror.values) for r in a], axis=0)
    lb = np.mean([np.log(pd.read_csv(f"{root}/{r}/predictions/{workload}.csv").qerror.values) for r in b], axis=0)
    if mask is not None:
        la, lb = la[mask], lb[mask]
    d = lb - la
    return dict(geo_A=float(np.exp(la.mean())), geo_B=float(np.exp(lb.mean())), ratio_B_over_A=float(np.exp(d.mean())),
                B_worse=int((d > 0).sum()), n=len(d), wilcoxon_p=float(wilcoxon(d).pvalue))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--name", default=OUT_NAME, help="sub-directory under tables/")
    a = ap.parse_args()
    out = os.path.join(a.results, "tables", a.name)
    os.makedirs(out, exist_ok=True)

    missing = [r for r, *_ in RUNS if not os.path.exists(f"{a.results}/{r}/summary.json")]
    if missing:
        raise SystemExit(f"missing runs: {', '.join(missing)}\nrun: bash exp/exp4_ablation/run_ablations.sh res width seeds")

    df = pd.DataFrame([dict(run=r, width=w, layers=L, residual=res, seed=s, **metrics(a.results, r))
                       for r, w, L, res, s in RUNS])
    df.to_csv(f"{out}/all_runs.csv", index=False)

    keys = ["last3_val_q50", "last3_val_q99", "last3_val_mean", "val_geo", "val_q90", "syn_geo", "jl_geo",
            "jl_q90", "jl3_q90", "jl3_geo"]
    seeds3 = pd.DataFrame([{"model": g, "params": int(df.set_index("run").loc[rs].params.iloc[0]),
                            **{k: f"{df.set_index('run').loc[rs][k].mean():.4f} ± {df.set_index('run').loc[rs][k].std(ddof=1):.4f}"
                               for k in keys}} for g, rs in SEED_GROUPS.items()])
    seeds3.to_csv(f"{out}/seeds3.csv", index=False)

    nj = pd.read_csv(f"{a.results}/gnto/predictions/job-light.csv").n_joins.values
    subsets = [("val", "val", None), ("synthetic", "synthetic", None), ("job-light all", "job-light", None),
               ("job-light <=2 join", "job-light", nj <= 2), ("job-light 3+ join", "job-light", nj >= 3)]
    res_grp, nores_grp, dflt = (SEED_GROUPS["w64 L5 identity res"], SEED_GROUPS["w64 L5 no res"],
                                SEED_GROUPS["default (256/128/64, L3, linear res)"])
    tests = pd.DataFrame([{"A": na, "B": nb, "subset": lab, **paired(a.results, A, B, wl, m)}
                          for (na, A), (nb, B) in [(("w64 L5 identity res", res_grp), ("w64 L5 no res", nores_grp)),
                                                   (("w64 L5 identity res", res_grp), ("default", dflt))]
                          for lab, wl, m in subsets])
    tests.to_csv(f"{out}/paired_tests.csv", index=False)

    # depth x width overview: seeds of the same configuration averaged
    grouped = df.groupby(["width", "layers", "residual"], as_index=False).mean(numeric_only=True)
    grouped["seeds"] = df.groupby(["width", "layers", "residual"]).size().values
    overview = grouped[["width", "layers", "residual", "seeds", "params", "val_q50", "val_q90", "val_q99", "val_geo",
                        "last3_val_q50", "last3_val_q99", "last3_val_mean", "syn_q50", "syn_q90", "syn_q99", "syn_geo",
                        "jl_q50", "jl_q90", "jl_q99", "jl_geo", "jl3_geo", "jl3_q90"]].sort_values(["residual", "width", "layers"])
    overview.to_csv(f"{out}/overview_by_depth_width.csv", index=False)

    with open(f"{out}/summary.md", "w") as f:
        f.write("# Exp1 ablation - residual x depth x width (GNTO graph encoder)\n\n")
        f.write("`last3_*` = mean of the val evaluations at epochs 80/90/99 (end of training, no checkpoint "
                "selection). Every other column comes from the protocol's checkpoint (best val Q90), the only one "
                "the synthetic / JOB-light predictions exist for. geo = per-query geometric mean Q-error.\n\n")
        f.write("## Depth x width (seeds of the same configuration averaged)\n\n" + md(overview, "{:.4f}"))
        f.write("\n## Three seeds (42 / 7 / 123)\n\n" + md(seeds3))
        f.write("\n## Paired per-query tests (Wilcoxon signed-rank)\n\n" + md(tests, "{:.4g}"))
        f.write("\n## All runs\n\n" + md(df, "{:.4f}"))
    pd.set_option("display.width", 250)
    print(overview.round(4).to_string(index=False))
    print()
    print(seeds3.to_string(index=False))
    print()
    print(tests.round(4).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
