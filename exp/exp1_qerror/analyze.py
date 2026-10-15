"""Exp1 tables: Q-error *levels* of every model directory under ``results/exp1_qerror/``.

Reads the per-model contract (``common.py``) and writes CSV + Markdown tables to
``results/exp1_qerror/tables/[<name>/]``:

* ``params.csv``            parameter counts
* ``qerror_levels.csv``     Q50 / Q75 / Q90 / Q95 / Q99 / mean (+ Spearman, Kendall) per model x workload
* ``qerror_ratio.csv``      every model / the first model, per level (< 1: better than the reference)
* ``qerror_by_joins.csv``   the levels per number of joins (query-complexity tiers)
* ``qerror_by_latency.csv`` the levels per actual-latency tier (<10 ms, 10-100 ms, 100 ms-1 s, >1 s)
* ``qerror_holdout.csv``    (``--holdout``) validation plans split by "contains a join pair that was
                            held out of training"; the flag is recomputed from the ``joins`` column, so
                            reference models trained on everything are split the same way
* ``qerror_epochs.csv``     validation levels vs. epoch
* ``summary.md``            all of the above

Usage::

    python exp/exp1_qerror/analyze.py                                   # qf vs gnto -> tables/
    python exp/exp1_qerror/analyze.py --models gnto_joinid,gnto --name ablation_join_encoding
    python exp/exp1_qerror/analyze.py --models gnto_joinid_holdout,gnto_holdout,qf --holdout --name ablation_join_encoding_holdout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LEVELS, RESULTS, WORKLOADS, level_stats, md  # noqa: E402

MODEL_LABELS = {
    "qf": "QueryFormer", "gnto": "GNTO", "gnto_joinid": "GNTO (string-id join)",
    "gnto_holdout": "GNTO", "gnto_joinid_holdout": "GNTO (string-id join)",
}
WORKLOAD_LABELS = {"val": "Held-out (parts 18-19)", "job-light": "JOB-light", "synthetic": "Synthetic"}
LATENCY_BINS = [0, 10, 100, 1000, np.inf]
LATENCY_LABELS = ["<10 ms", "10-100 ms", "100 ms-1 s", ">1 s"]
HOLDOUT_LABELS = ["seen joins", "held-out join"]


@dataclass
class Model:
    name: str                       # directory name
    label: str
    dir: str
    summary: dict
    config: dict
    preds: Dict[str, pd.DataFrame] = field(default_factory=dict)
    log: Optional[pd.DataFrame] = None

    @property
    def workloads(self) -> List[str]:
        return [w for w in WORKLOADS if w in self.preds] + sorted(set(self.preds) - set(WORKLOADS))

    @property
    def holdout_joins(self) -> Set[str]:
        return set(self.summary.get("holdout_joins") or [])


def discover(results_dir: str = RESULTS) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        d = os.path.dirname(p)
        if not d.endswith("_smoke"):
            out[os.path.basename(d)] = d
    return out


def load_model(name: str, results_dir: str = RESULTS, label: Optional[str] = None) -> Model:
    d = name if os.path.isdir(name) else os.path.join(results_dir, name)
    name = os.path.basename(os.path.normpath(d))
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    config = {}
    if os.path.exists(os.path.join(d, "config.json")):
        with open(os.path.join(d, "config.json")) as f:
            config = json.load(f)
    m = Model(name=name, label=label or MODEL_LABELS.get(name, name), dir=d, summary=summary, config=config)
    for p in sorted(glob.glob(os.path.join(d, "predictions", "*.csv"))):
        m.preds[os.path.basename(p)[:-4]] = pd.read_csv(p)
    if os.path.exists(os.path.join(d, "training_log.csv")):
        m.log = pd.read_csv(os.path.join(d, "training_log.csv"))
    return m


def load_models(names: Sequence[str], results_dir: str = RESULTS, labels: Optional[Dict[str, str]] = None) -> List[Model]:
    labels = labels or {}
    return [load_model(n, results_dir, labels.get(n)) for n in names]


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def qerror_of(df: pd.DataFrame) -> np.ndarray:
    return df["qerror"].to_numpy(dtype=np.float64)


def params_table(models: List[Model]) -> pd.DataFrame:
    return pd.DataFrame([{"model": m.label, "dir": m.name, "total_params": m.summary["params"]["total"],
                          "trainable_params": m.summary["params"]["trainable"]} for m in models])


def level_table(models: List[Model]) -> pd.DataFrame:
    rows = []
    for w in WORKLOADS:
        for m in models:
            df = m.preds.get(w)
            if df is None:
                continue
            row = {"model": m.label, "workload": w, "n": len(df), **level_stats(qerror_of(df))}
            q = m.summary.get("qerror", {}).get(w, {})
            row.update({k: q.get(k, np.nan) for k in ("spearman", "kendall", "pairwise_acc")})
            rows.append(row)
    return pd.DataFrame(rows)


def ratio_table(lt: pd.DataFrame, reference: str) -> pd.DataFrame:
    """``model / reference`` per level and workload (< 1: lower Q-error than the reference)."""
    cols = [f"q{k}" for k in LEVELS] + ["q_mean"]
    rows = []
    for w, g in lt.groupby("workload", sort=False):
        ref = g[g.model == reference]
        if ref.empty:
            continue
        for _, r in g[g.model != reference].iterrows():
            rows.append({"model": r["model"], "reference": reference, "workload": w,
                         **{c: float(r[c] / ref[c].iloc[0]) for c in cols}})
    return pd.DataFrame(rows)


def bucket_table(models: List[Model], by: str, bins=None, labels=None, min_n: int = 5) -> pd.DataFrame:
    rows = []
    for m in models:
        for w, df in m.preds.items():
            if by == "actual_ms" or bins is not None:
                key = pd.cut(df[by], bins=bins if bins is not None else LATENCY_BINS,
                             labels=labels if labels is not None else LATENCY_LABELS, right=False)
            else:
                key = df[by].astype(int)
            q = qerror_of(df)
            for bucket, idx in df.groupby(key, observed=True).indices.items():
                if len(idx) >= min_n:
                    rows.append({"model": m.label, "workload": w, "bucket": bucket, "n": int(len(idx)), **level_stats(q[idx])})
    out = pd.DataFrame(rows)
    if len(out):
        out["_w"] = out["workload"].map({w: i for i, w in enumerate(WORKLOADS)}).fillna(99)
        out["_m"] = out["model"].map({m.label: i for i, m in enumerate(models)})
        out = out.sort_values(["_w", "bucket", "_m"]).drop(columns=["_w", "_m"]).reset_index(drop=True)
    return out


def holdout_flags(df: pd.DataFrame, holdout: Set[str]) -> np.ndarray:
    joins = df["joins"].fillna("").astype(str)
    return joins.map(lambda s: any(j in holdout for j in s.split("|") if j)).to_numpy()


def holdout_table(models: List[Model], holdout: Optional[Set[str]] = None, workload: str = "val") -> pd.DataFrame:
    """Levels on ``workload`` plans with / without a held-out join pair.  ``holdout``
    defaults to the union of the models' own ``holdout_joins``; every model
    (including references trained on everything) is split with the same set."""
    holdout = holdout or set().union(*(m.holdout_joins for m in models))
    if not holdout:
        return pd.DataFrame()
    rows = []
    for m in models:
        df = m.preds.get(workload)
        if df is None or "joins" not in df:
            continue
        flag = holdout_flags(df, holdout)
        q = qerror_of(df)
        for f, name in ((False, HOLDOUT_LABELS[0]), (True, HOLDOUT_LABELS[1])):
            idx = np.where(flag == f)[0]
            if len(idx):
                rows.append({"model": m.label, "trained_without": bool(m.holdout_joins), "plans": name,
                             "n": int(len(idx)), **level_stats(q[idx])})
    return pd.DataFrame(rows)


def epoch_table(models: List[Model]) -> pd.DataFrame:
    frames = []
    for m in models:
        if m.log is None:
            continue
        cols = ["epoch", "loss", "time", "val_loss"] + [c for c in m.log if c.startswith("val_q")]
        f = m.log[[c for c in cols if c in m.log]].copy()
        f.insert(0, "model", m.label)
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def write_tables(models: List[Model], out_dir: str, holdout: bool = False, title: str = "Exp1") -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    lt = level_table(models)
    tables = {
        "params": params_table(models),
        "qerror_levels": lt,
        "qerror_ratio": ratio_table(lt, models[0].label) if len(models) > 1 else pd.DataFrame(),
        "qerror_by_joins": bucket_table(models, "n_joins"),
        "qerror_by_latency": bucket_table(models, "actual_ms"),
        "qerror_epochs": epoch_table(models),
    }
    if holdout:
        tables["qerror_holdout"] = holdout_table(models)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    parts = [f"# {title}\n", "models: " + ", ".join(f"`{m.name}` = {m.label}" for m in models) + "\n",
             "\n## Parameter count\n", md(tables["params"]),
             "\n## Q-error per level and workload\n", md(lt),
             f"\n## Ratio to {models[0].label} (< 1: better)\n", md(tables["qerror_ratio"]),
             "\n## By number of joins\n", md(tables["qerror_by_joins"]),
             "\n## By actual latency tier\n", md(tables["qerror_by_latency"])]
    if holdout:
        hj = sorted(set().union(*(m.holdout_joins for m in models)))
        parts += [f"\n## Validation plans by held-out join pair ({', '.join(hj)})\n", md(tables["qerror_holdout"])]
    parts += ["\n## Validation Q-error vs epoch\n", md(tables["qerror_epochs"])]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("".join(parts))
    return tables


def parse_labels(spec: str) -> Dict[str, str]:
    return dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qf,gnto", help="comma list of model dirs under results/exp1_qerror (first = reference)")
    ap.add_argument("--labels", default="", help="display names, e.g. gnto_joinid='GNTO (string id)',gnto=GNTO")
    ap.add_argument("--holdout", action="store_true", help="add the held-out-join split (models must carry holdout_joins)")
    ap.add_argument("--name", default="", help="sub-directory under tables/ (default: none = the main comparison)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--list", action="store_true", help="list the model dirs found and exit")
    a = ap.parse_args()
    found = discover(a.results)
    if a.list:
        for n, d in found.items():
            print(f"{n:24s} {d}")
        return
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    missing = [n for n in names if n not in found and not os.path.isdir(n)]
    if missing:
        raise SystemExit(f"not found under {a.results}: {missing}; available: {sorted(found)}")
    models = load_models(names, a.results, parse_labels(a.labels))
    out_dir = os.path.join(a.results, "tables", a.name) if a.name else os.path.join(a.results, "tables")
    tables = write_tables(models, out_dir, holdout=a.holdout, title=f"Exp1 - {a.name or 'QueryFormer vs GNTO'}")
    fmt = lambda v: f"{v:.3f}"
    print("== parameters ==\n" + tables["params"].to_string(index=False))
    print("\n== Q-error levels ==\n" + tables["qerror_levels"][["model", "workload", "n"] + [f"q{k}" for k in LEVELS] + ["q_mean"]]
          .to_string(index=False, float_format=fmt))
    if not tables["qerror_ratio"].empty:
        print(f"\n== ratio to {models[0].label} ==\n" + tables["qerror_ratio"].drop(columns="reference").to_string(index=False, float_format=fmt))
    if a.holdout:
        print("\n== held-out join pairs (val) ==\n" + tables["qerror_holdout"].to_string(index=False, float_format=fmt))
    print(f"\ntables -> {out_dir}")


if __name__ == "__main__":
    main()
