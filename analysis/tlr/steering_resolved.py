"""Plan-selection rows of the own-cost inputs on TPC-H with the grouping nodes resolved.

The existing steering_configs.csv (analysis/dependency_study/analyze_steering.py) holds TPC-H rows for resid_* and
pgl_* whose own cost counts a grouping node of ordered-binary-v1 as an input of cost 0 (53 of the 643 TPC-H plans;
none on JOB or on the 20-database benchmark; see analysis/attn/derived.own_cost).  The resolved runs residg_* and
pglg_* lie in the directories that analyze_steering reads.  This script reruns analyze_steering unchanged into a
temporary directory, checks that every existing row of steering_configs.csv is reproduced exactly (one fixed bootstrap
draw per workload), and writes the TPC-H residg_* / pglg_* rows to

    results/analysis/tlr_20261008/selection/tables/steering_configs_resolved.csv

which replace the TPC-H resid_* / pgl_* rows.  No existing table is overwritten.

    python analysis/tlr/steering_resolved.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEP = os.path.join(ROOT, "results", "analysis", "dependency_20261003")
OUT = os.path.join(ROOT, "results", "analysis", "tlr_20261008", "selection", "tables", "steering_configs_resolved.csv")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.symlink(os.path.join(DEP, "steering"), os.path.join(tmp, "steering"))
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "dependency_study", "analyze_steering.py"),
                        "--dep", tmp], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        new = pd.read_csv(os.path.join(tmp, "tables", "steering_configs.csv"))
    pub = pd.read_csv(os.path.join(DEP, "tables", "steering_configs.csv"))
    both = pub.merge(new, on=["workload", "config"], how="left", suffixes=("", "_new"))
    num = [c for c in pub.columns if c not in ("workload", "config") and pd.api.types.is_numeric_dtype(pub[c])]
    diff = max(float(np.nanmax(np.abs(both[c] - both[c + "_new"]))) if both[c].notna().any() else 0.0 for c in num)
    assert both[num[0] + "_new"].notna().all() and diff == 0.0, f"existing rows not reproduced (max diff {diff})"
    res = new[(new.workload == "tpch52") & new.config.str.match(r"^(residg|pglg)_")]
    res.to_csv(OUT, index=False)
    print(f"existing rows reproduced ({len(pub)}); wrote {len(res)} resolved TPC-H rows to {OUT}")


if __name__ == "__main__":
    main()
