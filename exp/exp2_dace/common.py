"""Shared pieces of Exp2 (cross-database generalisation vs. DACE): the 20-database
benchmark, the 10 / 10 database split, the 90 / 10 train / validation split of the
training plans, per-database metrics, the per-model output contract and the shared CLI.

Output contract - every runner writes ``results/exp2_dace/<model>_<split>_s<seed>/``::

    config.json               arguments, hyper-parameters, split (train / test databases), git rev
    training_log.csv          one row per epoch (train loss, validation Q-error levels)
    best_model.pt             checkpoint with the lowest validation mean Q-error
    predictions/<test_db>.csv plan_id, pred_ms, actual_ms, qerror, n_nodes
    summary.json              model, params, split, per-database Q-error levels, averages, timing
    log.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

_EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_EXP_ROOT, os.path.dirname(_EXP_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _shared import GNTO_PATH, Logger, git_rev, md, set_seed, provenance, object_sha256, file_sha256  # noqa: E402,F401
from models.dace.plan_utils import fit_node_statistics
from models.dace import (DACE_DATA_DIR, MAX_RUNTIME, WORKLOADS, NodeFeaturizer, count_nodes, load_plans,  # noqa: E402
                         load_statistics, make_split, plans_path)

EXP = "exp2_dace"
RESULTS = os.path.join(GNTO_PATH, "results", EXP)
CACHE_DIR = os.path.join(GNTO_PATH, "data", "process", "cache", "dace")
LEVELS: Tuple[int, ...] = (50, 75, 90, 95, 99)
PRED_COLUMNS = ["plan_id", "pred_ms", "actual_ms", "qerror", "n_nodes"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_benchmark(a: argparse.Namespace, log) -> Dict[str, Any]:
    """Statistics + featurizer + the plans of every database in the split.

    Returns ``{'stats', 'fz', 'train_ids', 'test_ids', 'split_name', 'train_dbs', 'test_dbs',
    'plans': {db: [plan...]}, 'train_index': [(db, i)...], 'val_index': [(db, i)...]}``.
    The training plans of the training databases are split 90 / 10 with
    ``sklearn.train_test_split(random_state=seed)`` (DACE's own protocol), on the
    same concatenated (database order, file order) list for every runner."""
    train_ids, test_ids, split_name = make_split(a.split, a.split_seed)
    train_dbs, test_dbs = [WORKLOADS[i] for i in train_ids], [WORKLOADS[i] for i in test_ids]
    plans: Dict[str, list] = {}
    for db in train_dbs + test_dbs:
        p = load_plans(plans_path(db, a.data_dir))
        plans[db] = p[:a.limit] if a.limit else p
    pool = [(db, i) for db in train_dbs for i in range(len(plans[db]))]
    tr, va = train_test_split(pool, test_size=a.val_frac, random_state=a.seed)
    scope = getattr(a, "statistics", "train")
    stats = (fit_node_statistics(plans[db][i] for db, i in tr) if scope == "train"
             else load_statistics(a.data_dir))
    fz = NodeFeaturizer(stats)
    source_hashes = {db: file_sha256(plans_path(db, a.data_dir)) for db in plans}
    cache_id = object_sha256({"statistics": stats, "sources": source_hashes,
                             "order": train_dbs + test_dbs})[:20]
    log(f"split {split_name}: train DBs {train_dbs}\n           test DBs {test_dbs}")
    log(f"train plans {len(tr):,} | val plans {len(va):,} (val_frac {a.val_frac}, seed {a.seed}) | "
        f"test plans {sum(len(plans[d]) for d in test_dbs):,}")
    return {"stats": stats, "fz": fz, "train_ids": train_ids, "test_ids": test_ids, "split_name": split_name,
            "train_dbs": train_dbs, "test_dbs": test_dbs, "plans": plans, "train_index": tr, "val_index": va,
            "statistics_scope": scope, "cache_id": cache_id, "source_hashes": source_hashes}


def write_benchmark_manifest(out_dir, bench):
    """Persist preprocessing and exact train/validation membership beside the model."""
    os.makedirs(out_dir, exist_ok=True)
    for name, value in (("feature_statistics.json", bench["stats"]),
                        ("data_manifest.json", {k: bench[k] for k in
                         ("statistics_scope", "source_hashes", "cache_id", "train_index", "val_index",
                          "train_dbs", "test_dbs")})):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(value, f, indent=2)


def index_positions(index: Sequence[Tuple[str, int]], dbs: Sequence[str], sizes: Dict[str, int]) -> np.ndarray:
    """Positions of ``(db, i)`` pairs inside the concatenation of ``dbs`` (in that order)."""
    offset, pos = {}, 0
    for db in dbs:
        offset[db] = pos
        pos += sizes[db]
    return np.asarray([offset[db] + i for db, i in index], dtype=np.int64)


# --------------------------------------------------------------------------- #
# metrics + output contract
# --------------------------------------------------------------------------- #
def qerror(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(preds, dtype=np.float64), 1e-7)
    t = np.maximum(np.asarray(targets, dtype=np.float64), 1e-7)
    return np.maximum(p / t, t / p)


def level_stats(q: np.ndarray) -> Dict[str, float]:
    out = {f"q{k}": float(np.percentile(q, k)) for k in LEVELS}
    out["q_mean"] = float(q.mean())
    out["q_max"] = float(q.max())
    return out


def db_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    return {"n": int(len(preds)), **level_stats(qerror(preds, targets))}


def write_predictions(out_dir: str, db: str, preds: np.ndarray, targets: np.ndarray, n_nodes: Sequence[int]) -> str:
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    df = pd.DataFrame({"plan_id": np.arange(len(preds)), "pred_ms": preds, "actual_ms": targets,
                       "qerror": qerror(preds, targets), "n_nodes": np.asarray(n_nodes)})[PRED_COLUMNS]
    path = os.path.join(out_dir, "predictions", f"{db}.csv")
    df.to_csv(path, index=False)
    return path


def averages(per_db: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Mean over test databases of each level (the "average median")."""
    keys = [f"q{k}" for k in LEVELS] + ["q_mean"]
    return {k: float(np.mean([m[k] for m in per_db.values()])) for k in keys}


def write_summary(out_dir: str, model: str, params: Dict[str, int], bench: Dict[str, Any],
                  per_db: Dict[str, Dict[str, Any]], train: Dict[str, Any], timing: Dict[str, float],
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = {"model": model, "params": params,
               "split": {"name": bench["split_name"], "train_ids": bench["train_ids"], "test_ids": bench["test_ids"],
                         "train_dbs": bench["train_dbs"], "test_dbs": bench["test_dbs"],
                         "n_train": len(bench["train_index"]), "n_val": len(bench["val_index"])},
               "statistics_scope": bench["statistics_scope"], "cache_id": bench["cache_id"],
               "qerror": per_db, "avg": averages(per_db), "train": train, "timing_s": timing,
               "finished": datetime.now().isoformat(timespec="seconds")}
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
    return " ".join(f"Q{k}={row[f'q{k}']:.3f}" for k in LEVELS) + f" mean={row['q_mean']:.3f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common_args(ap: argparse.ArgumentParser, default_bs: int) -> None:
    g = ap.add_argument_group("data")
    g.add_argument("--data-dir", default=DACE_DATA_DIR)
    g.add_argument("--split", default="paper", choices=["paper", "rand"], help="paper = the fixed 10/10 split; rand = seeded random 10/10")
    g.add_argument("--split-seed", type=int, default=1, help="seed of the random split (rand only)")
    g.add_argument("--statistics", choices=["train", "published"], default="train",
                   help="train: fit on training plans only; published: statistics.json in --data-dir (DACE's get_statistic over all 20 databases)")
    g.add_argument("--val-frac", type=float, default=0.1, help="fraction of training plans held out for checkpoint selection")
    g.add_argument("--limit", type=int, help="use only the first N plans of every database")
    g.add_argument("--no-cache", action="store_true", help="do not read/write data/process/cache/dace")
    t = ap.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--bs", type=int, default=default_bs)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ckpt", help="skip training: evaluate this checkpoint")
    t.add_argument("--select", default="val_q_mean", choices=["val_q_mean", "last"],
                   help="which epoch to evaluate: the best validation mean Q-error (default) or the last epoch")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    o = ap.add_argument_group("output")
    o.add_argument("--out", help="output dir (default results/exp2_dace/<model>_<split>_s<seed>)")
    o.add_argument("--tag", default="")
    o.add_argument("--smoke", action="store_true", help="2 epochs on 300 plans per database, no cache")


def finalize_args(a: argparse.Namespace, model_key: str) -> str:
    torch.set_num_threads(a.threads)
    if a.smoke:
        a.limit, a.epochs, a.no_cache = 300, 2, True
    a.use_cache = not (a.no_cache or a.limit)
    _, _, split_name = make_split(a.split, a.split_seed)
    name = f"{model_key}_{split_name}_s{a.seed}" + (f"_{a.tag}" if a.tag else "") + ("_smoke" if a.smoke else "")
    if a.statistics == "train":
        name += "_trainstats"
    return a.out or os.path.join(RESULTS, name)
