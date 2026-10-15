"""Exp2 tables for the common-budget comparison (100 epochs, final-epoch model, 3 seeds).

Compares, on the paper split, the runs written by ``run_e100.sh`` against the 10-epoch runs of the
default protocol, and tests every pair over the 10 zero-shot test databases (Wilcoxon signed-rank on
the seed-averaged per-database metric).  Writes ``results/exp2_dace/tables/e100/``:

    summary.csv / paired_tests.csv / per_db.csv / summary.md

    python exp/exp2_dace/analyze_e100.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import RESULTS  # noqa: E402
from _shared import md  # noqa: E402

SEEDS = (42, 7, 123)
# label -> (model, tag);  tag "" = the default 10-epoch protocol
VARIANTS = [("DACE (10 ep, upstream protocol)", "dace", ""), ("DACE (100 ep, last epoch)", "dace", "e100"),
            ("GNTO default (10 ep)", "gnto", ""), ("GNTO default (100 ep, last epoch)", "gnto", "e100"),
            ("GNTO w64_L16 (10 ep)", "gnto", "w64_L16"), ("GNTO w64_L16 (100 ep, last epoch)", "gnto", "w64_L16_e100")]
PAIRS = [("GNTO default (100 ep, last epoch)", "DACE (100 ep, last epoch)"),
         ("GNTO w64_L16 (100 ep, last epoch)", "DACE (100 ep, last epoch)"),
         ("GNTO w64_L16 (100 ep, last epoch)", "GNTO default (100 ep, last epoch)")]
METRICS = ("q50", "q90", "q99", "mean", "geo")


def per_db(root: str, run: str) -> pd.DataFrame:
    """One row per test database: the usual Q-error levels over that database's plans."""
    out = {}
    for f in sorted(glob.glob(f"{root}/{run}/predictions/*.csv")):
        q = pd.read_csv(f).qerror.values
        out[os.path.basename(f)[:-4]] = dict(q50=np.median(q), q90=np.quantile(q, .9), q99=np.quantile(q, .99),
                                             mean=q.mean(), geo=float(np.exp(np.log(q).mean())))
    return pd.DataFrame(out).T


def load(root: str, model: str, tag: str):
    dirs = [f"{model}_paper_s{s}" + (f"_{tag}" if tag else "") for s in SEEDS]
    dirs = [d for d in dirs if os.path.exists(f"{root}/{d}/summary.json")]
    if not dirs:
        return None
    seed_dbs = [per_db(root, d) for d in dirs]
    s = json.load(open(f"{root}/{dirs[0]}/summary.json"))
    return dict(dirs=dirs, seed_dbs=seed_dbs, per_db=sum(seed_dbs) / len(seed_dbs),
                params=s["params"]["total"], epochs=s["train"].get("epochs"), select=s["train"].get("select"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--name", default="e100", help="sub-directory under tables/")
    a = ap.parse_args()
    out = os.path.join(a.results, "tables", a.name)
    os.makedirs(out, exist_ok=True)

    V = {label: load(a.results, model, tag) for label, model, tag in VARIANTS}
    missing = [k for k, v in V.items() if v is None]
    if missing:
        raise SystemExit(f"missing runs for: {', '.join(missing)}\nrun: bash exp/exp2_dace/run_e100.sh")

    summary = pd.DataFrame([dict(model=k, params=v["params"], epochs=v["epochs"], select=v["select"],
                                 seeds=len(v["seed_dbs"]),
                                 avg_q50=np.mean([d.q50.mean() for d in v["seed_dbs"]]),
                                 avg_q50_std=np.std([d.q50.mean() for d in v["seed_dbs"]], ddof=1),
                                 avg_q90=v["per_db"].q90.mean(), avg_q99=v["per_db"].q99.mean(),
                                 avg_mean=v["per_db"]["mean"].mean(), avg_geo=v["per_db"].geo.mean())
                            for k, v in V.items()])
    tests = pd.DataFrame([{"model_A": A, "model_B (reference)": B, "metric": m,
                           "A_avg": V[A]["per_db"][m].mean(), "B_avg": V[B]["per_db"][m].mean(),
                           "A_better_dbs": int(((V[A]["per_db"][m] - V[B]["per_db"][m]) < 0).sum()),
                           "n_dbs": len(V[A]["per_db"]),
                           "wilcoxon_p": float(wilcoxon(V[A]["per_db"][m] - V[B]["per_db"][m]).pvalue)}
                          for A, B in PAIRS for m in METRICS])
    e100 = {k: v for k, v in V.items() if "100 ep" in k}
    cols = {f"{k} q50": v["per_db"].q50 for k, v in e100.items()}
    cols.update({f"{k} q99": v["per_db"].q99 for k, v in e100.items()})
    per_database = pd.DataFrame(cols).reset_index(names="database")

    summary.to_csv(f"{out}/summary.csv", index=False)
    tests.to_csv(f"{out}/paired_tests.csv", index=False)
    per_database.to_csv(f"{out}/per_db.csv", index=False)
    with open(f"{out}/summary.md", "w") as f:
        f.write("# Exp2 at a common budget - 100 epochs, final-epoch model (paper split, 3 seeds)\n\n")
        f.write("avg_q50 = mean of the 10 test databases' median Q-errors (± std over seeds); the other columns are "
                "averaged over seeds first, then over databases.\n\n")
        f.write("## Summary\n\n" + md(summary, "{:.4f}"))
        f.write("\n## Paired over the 10 test databases (Wilcoxon signed-rank)\n\n" + md(tests, "{:.4g}"))
        f.write("\n## Per test database (100 epochs)\n\n" + md(per_database, "{:.3f}"))
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print()
    print(tests.round(4).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
