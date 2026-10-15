"""Shared pieces of Exp1 (cost-model accuracy vs. QueryFormer): data split, workloads,
metrics, the per-model output contract, and the CLI arguments every runner accepts.

Output contract - every runner writes one directory ``results/exp1_qerror/<model>/``::

    config.json                 run arguments, preset / hyper-parameters, split, git rev, timing
    training_log.csv            one row per evaluated epoch (train / val Q-error percentiles)
    best_model.pt               checkpoint selected on the validation ``--select`` metric
    predictions/<workload>.csv  query_id, pred_ms, actual_ms, qerror, n_nodes, n_joins, joins, holdout_join
    summary.json                model name, parameter count, per-workload Q-error levels + rank metrics
    log.txt

``analyze.py`` / ``plot.py`` only read this contract, so a new model is a new runner and
an ablation is the same runner writing to another directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

_EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_EXP_ROOT, os.path.dirname(_EXP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _shared import GNTO_PATH, Logger, git_rev, md, set_seed, provenance  # noqa: E402,F401
from models.gnto.metrics import evaluation_summary  # noqa: E402
from models.qf import formatJoin, get_job_table_sample, plan_csv_path, workload_plan_csv, workload_sample_prefix  # noqa: E402

EXP = "exp1_qerror"
RESULTS = os.path.join(GNTO_PATH, "results", EXP)
CACHE_DIR = os.path.join(GNTO_PATH, "data", "process", "cache")
LEVELS: Tuple[int, ...] = (50, 75, 90, 95, 99)
JOIN_TYPES = ("Nested Loop", "Hash Join", "Merge Join")
WORKLOADS = ("val", "job-light", "synthetic")
PRED_COLUMNS = ["query_id", "pred_ms", "actual_ms", "qerror", "n_nodes", "n_joins", "joins", "holdout_join"]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def parse_range(spec: str) -> List[int]:
    """'0-14' -> [0..14]; '0,3,5' -> [0,3,5]."""
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# --------------------------------------------------------------------------- #
# plans: split, workloads, per-plan statistics
# --------------------------------------------------------------------------- #
def load_parts(parts: Iterable[int], limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(plan_csv_path(i)) for i in parts], ignore_index=True)
    return df.iloc[:limit].reset_index(drop=True) if limit else df


TEMPLATE_SPLIT = os.path.join(GNTO_PATH, "data", "qf", "template_split.json")


def load_template_split(limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Template-disjoint 80 / 10 / 10 split (``build_template_split.py``): no query template
    (relations + join conditions + filter columns) has plans in two of the three sets."""
    if not os.path.exists(TEMPLATE_SPLIT):
        raise FileNotFoundError(f"{TEMPLATE_SPLIT} missing - run exp/exp1_qerror/build_template_split.py")
    ids = json.load(open(TEMPLATE_SPLIT))
    df = load_parts(range(20)).set_index("id", drop=False)
    out = []
    for name, lim in (("train", limit), ("val", limit // 2 if limit else None), ("test", limit // 2 if limit else None)):
        part = df.loc[ids[name]].reset_index(drop=True)
        out.append(part.iloc[:lim].reset_index(drop=True) if lim else part)
    return tuple(out)


def load_split(a: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if getattr(a, "split", "parts") == "template":
        train, val, a.test_df = load_template_split(a.limit)
        return train, val
    train = load_parts(parse_range(a.train_parts), a.limit)
    val = load_parts(parse_range(a.val_parts), a.limit // 2 if a.limit else None)
    return train, val


def load_workload(name: str):
    """``(df, table_sample)`` for ``job-light`` / ``synthetic`` (their own sample bitmaps)."""
    return pd.read_csv(workload_plan_csv(name)).reset_index(drop=True), get_job_table_sample(workload_sample_prefix(name))


def plan_joins(root: Dict[str, Any]) -> List[str]:
    """Normalised join strings (as in ``encoding.idx2join``) found anywhere in the plan."""
    out: List[str] = []
    stack = [root]
    while stack:
        n = stack.pop()
        try:
            j = formatJoin(n)
        except Exception:
            j = None
        if j:
            out.append(j)
        stack.extend(n.get("Plans", []))
    return out


def plan_stats(df: pd.DataFrame, holdout: Optional[Set[str]] = None) -> pd.DataFrame:
    """Row-aligned ``query_id, n_nodes, n_joins, joins, holdout_join`` for a plan DataFrame."""
    n_nodes, n_joins, joins, flag = [], [], [], []
    for js in df["json"]:
        root = json.loads(js)["Plan"]
        nn_, nj, stack = 0, 0, [root]
        while stack:
            n = stack.pop()
            nn_ += 1
            nj += n["Node Type"] in JOIN_TYPES
            stack.extend(n.get("Plans", []))
        pj = plan_joins(root)
        n_nodes.append(nn_)
        n_joins.append(nj)
        joins.append("|".join(pj))
        flag.append(bool(holdout) and any(j in holdout for j in pj))
    return pd.DataFrame({"query_id": df["id"].values, "n_nodes": n_nodes, "n_joins": n_joins,
                         "joins": joins, "holdout_join": flag})


def parse_holdout(spec: str, encoding) -> Set[str]:
    holdout = {j.strip() for j in spec.split(",") if j.strip()}
    unknown = holdout - set(encoding.join2idx)
    if unknown:
        raise SystemExit(f"--exclude-joins not in vocabulary: {sorted(unknown)}; "
                         f"known: {sorted(j for j in encoding.join2idx if j)}")
    return holdout


def apply_holdout(train_df: pd.DataFrame, holdout: Set[str]) -> Tuple[pd.DataFrame, int]:
    """Drop every training plan that contains one of the held-out join pairs."""
    if not holdout:
        return train_df, 0
    keep = ~plan_stats(train_df, holdout)["holdout_join"].values
    return train_df[keep].reset_index(drop=True), int((~keep).sum())


def prepare_data(a: argparse.Namespace, encoding, log) -> Dict[str, Any]:
    """Split + workloads + per-plan stats, identical for every runner.

    Returns ``{'train_df', 'val_df', 'holdout', 'eval': {workload: (df, table_sample_or_None)},
    'stats': {workload: DataFrame}}``; ``table_sample`` is None for ``val`` (use the
    training sample file)."""
    train_df, val_df = load_split(a)
    if getattr(a, "split", "parts") == "template":
        log(f"split: template-disjoint -> train {len(train_df):,} | val {len(val_df):,} | test {len(a.test_df):,} plans")
    else:
        log(f"split: train parts {a.train_parts} -> {len(train_df):,} plans | val parts {a.val_parts} -> {len(val_df):,} plans")
    holdout = parse_holdout(a.exclude_joins, encoding) if a.exclude_joins else set()
    if holdout:
        train_df, dropped = apply_holdout(train_df, holdout)
        log(f"holding out joins {sorted(holdout)}: dropped {dropped:,} train plans -> {len(train_df):,}")
    workloads = [w.strip() for w in a.workloads.split(",") if w.strip()]
    if getattr(a, "split", "parts") == "template":
        workloads = ["test"] + [w for w in workloads if w != "test"]      # held-out templates, used once
    evals: Dict[str, Any] = {}
    for w in workloads:
        evals[w] = (val_df, None) if w == "val" else (a.test_df, None) if w == "test" else load_workload(w)
        log(f"workload {w}: {len(evals[w][0])} plans")
    stats = {w: plan_stats(df, holdout) for w, (df, _) in evals.items()}
    if holdout:
        for w, st in stats.items():
            log(f"workload {w}: {int(st['holdout_join'].sum())} of {len(st)} plans contain a held-out join")
    return {"train_df": train_df, "val_df": val_df, "holdout": sorted(holdout), "eval": evals, "stats": stats}


# --------------------------------------------------------------------------- #
# metrics + output contract
# --------------------------------------------------------------------------- #
def qerror(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(preds, dtype=np.float64), 1e-10)
    t = np.maximum(np.asarray(targets, dtype=np.float64), 1e-10)
    return np.maximum(p / t, t / p)


def level_stats(q: np.ndarray, levels: Tuple[int, ...] = LEVELS) -> Dict[str, float]:
    out = {f"q{k}": float(np.percentile(q, k)) for k in levels}
    out["q_mean"] = float(np.mean(q))
    return out


def workload_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    s = evaluation_summary(preds, targets, percentiles=LEVELS)
    row: Dict[str, Any] = {"n": int(len(preds))}
    row.update(level_stats(qerror(preds, targets)))
    row.update({k: float(s[k]) for k in ("spearman", "kendall", "pairwise_acc")})
    return row


def write_predictions(out_dir: str, workload: str, preds: np.ndarray, targets: np.ndarray, stats: pd.DataFrame) -> str:
    """Row-aligned with ``stats`` (the workload DataFrame order)."""
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    df = pd.DataFrame({"query_id": stats["query_id"].values, "pred_ms": preds, "actual_ms": targets,
                       "qerror": qerror(preds, targets), "n_nodes": stats["n_nodes"].values,
                       "n_joins": stats["n_joins"].values, "joins": stats["joins"].values,
                       "holdout_join": stats["holdout_join"].values})[PRED_COLUMNS]
    path = os.path.join(out_dir, "predictions", f"{workload}.csv")
    df.to_csv(path, index=False)
    return path


def write_summary(out_dir: str, model: str, params: Dict[str, int], per_workload: Dict[str, Dict[str, Any]],
                  train: Optional[Dict[str, Any]] = None, timing: Optional[Dict[str, float]] = None,
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = {"model": model, "params": params, "qerror": per_workload, "train": train or {},
               "timing_s": timing or {}, "finished": datetime.now().isoformat(timespec="seconds")}
    if extra:
        summary.update(extra)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def write_config(out_dir: str, a: argparse.Namespace, **more: Any) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"experiment": EXP, "out_dir": out_dir, "args": {k: v for k, v in vars(a).items() if not k.startswith("_")},
           "git_rev": git_rev(), "provenance": provenance(), "torch": torch.__version__, "started": datetime.now().isoformat(timespec="seconds")}
    cfg.update(more)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def format_levels(row: Dict[str, Any]) -> str:
    return " ".join(f"Q{k}={row[f'q{k}']:.3f}" for k in LEVELS)


# --------------------------------------------------------------------------- #
# CLI shared by the runners
# --------------------------------------------------------------------------- #
def add_common_args(ap: argparse.ArgumentParser, default_out: str) -> None:
    g = ap.add_argument_group("data")
    g.add_argument("--split", default="parts", choices=["parts", "template"],
                   help="parts: --train-parts / --val-parts (90 / 10 by query id); template: the template-disjoint "
                        "80 / 10 / 10 split of data/qf/template_split.json, which adds the held-out workload 'test'")
    g.add_argument("--train-parts", default="0-17")
    g.add_argument("--val-parts", default="18-19")
    g.add_argument("--workloads", default=",".join(WORKLOADS), help="comma list of: val, job-light, synthetic")
    g.add_argument("--limit", type=int, help="first N train plans (and N/2 val plans)")
    g.add_argument("--exclude-joins", default="", metavar="J1,J2,...",
                   help="drop every TRAIN plan containing one of these normalised join strings "
                        "(e.g. 'mc.movie_id = mi.movie_id'); predictions carry a holdout_join flag")
    g.add_argument("--num-workers", type=int, default=16, help="featurisation processes")
    g.add_argument("--no-cache", action="store_true", help="do not read/write data/process/cache")
    t = ap.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--bs", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--step-size", type=int, default=20)
    t.add_argument("--gamma", type=float, default=0.7)
    t.add_argument("--clip", type=float, default=50.0)
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--select", default="q90", choices=[f"q{k}" for k in LEVELS], help="validation metric that picks the checkpoint")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ckpt", help="skip training: evaluate this checkpoint")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    o = ap.add_argument_group("output")
    o.add_argument("--out", help=f"output dir (default {default_out})")
    o.add_argument("--tag", default="", help="suffix appended to the default output dir name")
    o.add_argument("--smoke", action="store_true", help="2 epochs on 1024 / 512 plans, no cache, dir suffix _smoke")


def finalize_args(a: argparse.Namespace, default_name: str) -> str:
    """Apply --smoke, compute the output dir, return it."""
    if a.smoke:
        a.limit, a.epochs, a.eval_every, a.no_cache = 1024, 2, 1, True
    a.use_cache = not (a.no_cache or a.limit)
    name = (default_name + ("_tmpl" if a.split == "template" else "") + ("_holdout" if a.exclude_joins else "")
            + (f"_{a.tag}" if a.tag else "") + ("_smoke" if a.smoke else ""))
    return a.out or os.path.join(RESULTS, name)
