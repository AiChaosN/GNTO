"""Exp2 tables: cross-database Q-error of every run under ``results/exp2_dace/``.

Run directories are named ``<model>_<split>_s<seed>[_<tag>]``; ``<model>[_<tag>]`` is a
*variant* (e.g. ``dace``, ``gnto``, ``gnto_L5``) and seeds of the same variant + split
are aggregated (mean +- std).  Written to ``results/exp2_dace/tables/[<name>/]``:

* ``params.csv``    parameter counts
* ``per_db.csv``    per test database: Q50 / Q90 / Q99 / mean (mean +- std over seeds)
* ``avg.csv``       averages over the test databases (the "average median") and the
                    levels of all test plans pooled
* ``wins.csv``      per split: databases won by each variant against the reference (first in
                    ``--models``) on seed-averaged medians, paired Wilcoxon and sign tests
* ``epochs.csv``    validation Q-error vs epoch
* ``summary.md``    all of the above

Usage::

    python exp/exp2_dace/analyze.py                            # dace vs gnto, every split found
    python exp/exp2_dace/analyze.py --models dace,gnto --splits paper
    python exp/exp2_dace/analyze.py --models gnto,gnto_L5 --name depth
    python exp/exp2_dace/analyze.py --list
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LEVELS, RESULTS, WORKLOADS, level_stats, md  # noqa: E402

MODEL_LABELS = {"dace": "DACE", "gnto": "GNTO"}
RUN_RE = re.compile(r"^(?P<model>[a-z]+)_(?P<split>paper|rand\d+)_s(?P<seed>\d+)(?:_(?P<tag>.+))?$")


@dataclass
class Run:
    dir: str
    model: str
    split: str
    seed: int
    tag: str
    summary: dict
    preds: Dict[str, pd.DataFrame] = field(default_factory=dict)
    log: Optional[pd.DataFrame] = None

    @property
    def variant(self) -> str:
        return f"{self.model}_{self.tag}" if self.tag else self.model

    @property
    def label(self) -> str:
        base = MODEL_LABELS.get(self.model, self.model)
        return f"{base} ({self.tag})" if self.tag else base

    @property
    def test_dbs(self) -> List[str]:
        return list(self.summary["split"]["test_dbs"])


def discover(results_dir: str = RESULTS, include_smoke: bool = False) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        d = os.path.dirname(p)
        name = os.path.basename(d)
        if RUN_RE.match(name) and (include_smoke or not name.endswith("_smoke")):
            out[name] = d
    return out


def load_run(d: str) -> Run:
    m = RUN_RE.match(os.path.basename(d))
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    r = Run(dir=d, model=m["model"], split=m["split"], seed=int(m["seed"]), tag=m["tag"] or "", summary=summary)
    for p in sorted(glob.glob(os.path.join(d, "predictions", "*.csv"))):
        r.preds[os.path.basename(p)[:-4]] = pd.read_csv(p)
    if os.path.exists(os.path.join(d, "training_log.csv")):
        r.log = pd.read_csv(os.path.join(d, "training_log.csv"))
    return r


def select_runs(variants: Sequence[str], splits: Optional[Sequence[str]] = None, results_dir: str = RESULTS,
                include_smoke: bool = False) -> List[Run]:
    runs = [load_run(d) for d in discover(results_dir, include_smoke).values()]
    runs = [r for r in runs if r.variant in variants and (not splits or r.split in splits)]
    order = {v: i for i, v in enumerate(variants)}
    return sorted(runs, key=lambda r: (r.split, order[r.variant], r.seed))


def label_of(variant: str) -> str:
    model, _, tag = variant.partition("_")
    base = MODEL_LABELS.get(model, model)
    return f"{base} ({tag})" if tag else base


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def params_table(runs: List[Run]) -> pd.DataFrame:
    seen, rows = set(), []
    for r in runs:
        if r.variant in seen:
            continue
        seen.add(r.variant)
        rows.append({"model": r.label, "variant": r.variant, "total_params": r.summary["params"]["total"],
                     "trainable_params": r.summary["params"]["trainable"]})
    return pd.DataFrame(rows)


def _agg(values: List[float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=0)) if len(v) > 1 else 0.0


def per_db_table(runs: List[Run]) -> pd.DataFrame:
    rows = []
    groups: Dict[Tuple[str, str], List[Run]] = {}
    for r in runs:
        groups.setdefault((r.split, r.variant), []).append(r)
    for (split, variant), rs in groups.items():
        dbs = [db for db in WORKLOADS if db in rs[0].summary["qerror"]]
        for db in dbs:
            row = {"split": split, "model": rs[0].label, "variant": variant, "db": db,
                   "seeds": len(rs), "n": rs[0].summary["qerror"][db]["n"]}
            for k in [f"q{p}" for p in LEVELS] + ["q_mean"]:
                m, s = _agg([r.summary["qerror"][db][k] for r in rs])
                row[k], row[f"{k}_std"] = m, s
            rows.append(row)
    return pd.DataFrame(rows)


def avg_table(runs: List[Run]) -> pd.DataFrame:
    rows = []
    groups: Dict[Tuple[str, str], List[Run]] = {}
    for r in runs:
        groups.setdefault((r.split, r.variant), []).append(r)
    for (split, variant), rs in groups.items():
        row = {"split": split, "model": rs[0].label, "variant": variant, "seeds": len(rs),
               "seed_list": ",".join(str(r.seed) for r in rs)}
        for k in [f"q{p}" for p in LEVELS] + ["q_mean"]:
            m, s = _agg([r.summary["avg"][k] for r in rs])
            row[f"avg_{k}"], row[f"avg_{k}_std"] = m, s
        pooled = [level_stats(np.concatenate([df["qerror"].to_numpy() for df in r.preds.values()])) for r in rs if r.preds]
        if pooled:
            for k in [f"q{p}" for p in LEVELS] + ["q_mean"]:
                row[f"pooled_{k}"] = float(np.mean([p[k] for p in pooled]))
        rows.append(row)
    return pd.DataFrame(rows)


def wins_table(per_db: pd.DataFrame, reference: str, metric: str = "q50") -> pd.DataFrame:
    """Databases where each variant beats the reference on the seed-averaged ``metric``,
    plus paired Wilcoxon (two-sided and 'less') and exact sign test over the databases."""
    from scipy import stats as sps
    rows = []
    for split, g in per_db.groupby("split", sort=False):
        ref = g[g.variant == reference].set_index("db")[metric]
        for variant, gv in g[g.variant != reference].groupby("variant", sort=False):
            other = gv.set_index("db")[metric]
            dbs = [d for d in ref.index if d in other.index]
            a, b = other.loc[dbs].to_numpy(), ref.loc[dbs].to_numpy()
            wins, losses = int((a < b).sum()), int((a > b).sum())
            row = {"split": split, "variant": variant, "reference": reference, "metric": metric, "n_dbs": len(dbs),
                   "wins": wins, "losses": losses, "ties": len(dbs) - wins - losses,
                   "mean_ratio": float(np.mean(a / b))}
            if len(dbs) >= 5 and np.any(a != b):
                row["wilcoxon_p_two_sided"] = float(sps.wilcoxon(a, b).pvalue)
                row["wilcoxon_p_less"] = float(sps.wilcoxon(a, b, alternative="less").pvalue)
                row["sign_test_p_two_sided"] = float(sps.binomtest(wins, wins + losses).pvalue) if wins + losses else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def epoch_table(runs: List[Run]) -> pd.DataFrame:
    frames = []
    for r in runs:
        if r.log is None:
            continue
        f = r.log[[c for c in ("epoch", "loss", "time", "val_loss", "val_q50", "val_q90", "val_q99", "val_q_mean") if c in r.log]].copy()
        f.insert(0, "seed", r.seed)
        f.insert(0, "variant", r.variant)
        f.insert(0, "model", r.label)
        f.insert(0, "split", r.split)
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_tables(runs: List[Run], out_dir: str, reference: str, title: str = "Exp2") -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    pdb = per_db_table(runs)
    tables = {"params": params_table(runs), "per_db": pdb, "avg": avg_table(runs),
              "wins": wins_table(pdb, reference) if len({r.variant for r in runs}) > 1 else pd.DataFrame(),
              "epochs": epoch_table(runs)}
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    show = ["split", "model", "db", "seeds", "n", "q50", "q50_std", "q90", "q99", "q_mean"]
    avg_show = ["split", "model", "seeds"] + [f"avg_q{p}" for p in LEVELS] + ["avg_q_mean"] + [c for c in tables["avg"] if c.startswith("pooled_")]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"# {title}\n\nruns: " + ", ".join(sorted({os.path.basename(r.dir) for r in runs})) + "\n"
                "\n## Parameter count\n" + md(tables["params"])
                + "\n## Average over test databases (mean over seeds; avg_qNN = mean of per-DB QNN, pooled_qNN = QNN over all test plans)\n"
                + md(tables["avg"][[c for c in avg_show if c in tables["avg"]]])
                + f"\n## Databases won against {label_of(reference)} (seed-averaged per-DB medians)\n" + md(tables["wins"])
                + "\n## Per test database\n" + md(pdb[[c for c in show if c in pdb]])
                + "\n## Validation Q-error vs epoch\n" + md(tables["epochs"]))
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="dace,gnto", help="comma list of variants (first = reference)")
    ap.add_argument("--splits", default="", help="comma list of splits (default: all found)")
    ap.add_argument("--name", default="", help="sub-directory under tables/")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--include-smoke", action="store_true", help="also consider *_smoke runs (variant '<model>_smoke')")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, d in discover(a.results, a.include_smoke).items():
            print(f"{n:28s} {d}")
        return
    variants = [v.strip() for v in a.models.split(",") if v.strip()]
    splits = [s.strip() for s in a.splits.split(",") if s.strip()] or None
    runs = select_runs(variants, splits, a.results, a.include_smoke)
    if not runs:
        raise SystemExit(f"no runs for {variants} / {splits} under {a.results}; found: {sorted(discover(a.results))}")
    out_dir = os.path.join(a.results, "tables", a.name) if a.name else os.path.join(a.results, "tables")
    t = write_tables(runs, out_dir, variants[0], title=f"Exp2 - {a.name or 'DACE vs GNTO'}")
    fmt = lambda v: f"{v:.3f}"
    print("== parameters ==\n" + t["params"].to_string(index=False))
    print("\n== averages over test DBs ==\n" + t["avg"][["split", "model", "seeds", "avg_q50", "avg_q50_std", "avg_q90", "avg_q99", "avg_q_mean"]].to_string(index=False, float_format=fmt))
    if not t["wins"].empty:
        print("\n== wins vs reference ==\n" + t["wins"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n== per DB (median) ==\n" + t["per_db"].pivot_table(index=["split", "db"], columns="model", values="q50").to_string(float_format=fmt))
    print(f"\ntables -> {out_dir}")


if __name__ == "__main__":
    main()
