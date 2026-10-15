"""Paired comparison of GNTO and DACE with TLR inputs (residual + root cost) and with the residual alone (resid), and of each
model's TLR/resid against its TCR runs.

Per split and percentile: the value of every test database is the mean over seeds of the per-database percentile;
differences are paired per database and bootstrapped over the ten test databases (20,000 draws).

    python analysis/tlr/analyze_tlr.py        # -> results/analysis/tlr_20261008/tables/tlr_paired.csv
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TLR = os.path.join(ROOT, "results", "analysis", "tlr_20261008")
TCR = os.path.join(ROOT, "results", "main_runs", "crossdb")
LEVELS = (50, 90, 99)


def per_db_levels(run_dir):
    """{db: {q50, q90, q99}} from the stored predictions of one run."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "predictions", "*.csv"))):
        df = pd.read_csv(p)
        q = df.qerror.values if "qerror" in df.columns else np.maximum(df.pred_ms / df.actual_ms, df.actual_ms / df.pred_ms)
        out[os.path.basename(p)[:-4]] = {k: float(np.percentile(q, k)) for k in LEVELS}
    return out


def collect(root, model, split):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, f"{model}_{split}_s*"))):
        if os.path.exists(os.path.join(d, "summary.json")):
            seed = int(os.path.basename(d).split("_s")[1].split("_")[0])
            runs[seed] = per_db_levels(d)
    return runs


def table(runs):
    """{db: {level: mean over seeds}}"""
    seeds = sorted(runs)
    dbs = sorted(runs[seeds[0]])
    return {db: {k: float(np.mean([runs[s][db][k] for s in seeds])) for k in LEVELS} for db in dbs}, len(seeds)


def paired(a, b, rng, draws=20000):
    dbs = sorted(set(a) & set(b))
    res = {}
    for k in LEVELS:
        x = np.array([a[db][k] - b[db][k] for db in dbs])
        idx = rng.integers(0, len(dbs), size=(draws, len(dbs)))
        boot = x[idx].mean(axis=1)
        res[k] = dict(a_mean=float(np.mean([a[db][k] for db in dbs])), b_mean=float(np.mean([b[db][k] for db in dbs])),
                      diff=float(x.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                      a_wins=int((x < 0).sum()), n_dbs=len(dbs))
    return res


def main() -> None:
    rng = np.random.default_rng(20261008)
    rows = []
    for split in ("paper1", "rand1", "rand2"):
        sets = {}
        for model in ("gnto", "dace"):
            for inputs, root in (("tlr", TLR), ("resid", TLR), ("tcr", TCR)):
                runs = collect(root, f"{model}_resid" if inputs == "resid" else model, split)
                if inputs == "tcr":
                    runs = {}
                    for d in sorted(glob.glob(os.path.join(TCR, f"{model}_{split}_s*_trainstats"))):
                        seed = int(os.path.basename(d).split("_s")[1].split("_")[0])
                        runs[seed] = per_db_levels(d)
                if runs:
                    sets[(model, inputs)], n = table(runs)
                    sets[(model, inputs, "n")] = n
        for (ma, ia), (mb, ib), label in ((("gnto", "tlr"), ("dace", "tlr"), "GNTO-TLR vs DACE-TLR"),
                                          (("gnto", "tlr"), ("gnto", "tcr"), "GNTO-TLR vs GNTO-TCR"),
                                          (("dace", "tlr"), ("dace", "tcr"), "DACE-TLR vs DACE-TCR"),
                                          (("gnto", "tlr"), ("dace", "tcr"), "GNTO-TLR vs DACE-TCR"),
                                          (("gnto", "tcr"), ("dace", "tcr"), "GNTO-TCR vs DACE-TCR"),
                                          (("gnto", "resid"), ("gnto", "tcr"), "GNTO-resid vs GNTO-TCR"),
                                          (("dace", "resid"), ("dace", "tcr"), "DACE-resid vs DACE-TCR"),
                                          (("gnto", "resid"), ("gnto", "tlr"), "GNTO-resid vs GNTO-TLR"),
                                          (("dace", "resid"), ("dace", "tlr"), "DACE-resid vs DACE-TLR"),
                                          (("gnto", "resid"), ("dace", "resid"), "GNTO-resid vs DACE-resid"),
                                          (("gnto", "resid"), ("dace", "tcr"), "GNTO-resid vs DACE-TCR")):
            if (ma, ia) in sets and (mb, ib) in sets:
                for k, r in paired(sets[(ma, ia)], sets[(mb, ib)], rng).items():
                    rows.append(dict(split=split, contrast=label, metric=f"q{k}", n_seeds_a=sets[(ma, ia, "n")],
                                     n_seeds_b=sets[(mb, ib, "n")], **r))
    df = pd.DataFrame(rows)
    out = os.path.join(TLR, "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "tlr_paired.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
