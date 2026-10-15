"""Zero-shot cost model (Hilprecht & Binnig, PVLDB 2022) on the cross-database splits of exp2.

The released code (https://github.com/DataManagementLab/zero-shot-cost-estimation, cloned to
``../baselines_src/zero-shot-cost-estimation``) is run unchanged in its own environment (python 3.8, torch 1.9.1,
dgl 0.9.1; its interpreter is given by ``ZS_PYTHON``, default ``python``) on the released parsed plans
(``runs/parsed_plans.zip`` of https://osf.io/ga2xj).
For 18 of the 20 databases these are the plans of the DACE benchmark in the same order with the same labels; for
``imdb_full`` and ``tpc_h`` the benchmark holds 10,000 plans of which the first 5,000 are the released ones, so
zero-shot trains on 50,000 of the 60,000 training plans.  Every *test* database of the published split and of random
split 1 is identical plan by plan; random split 2 tests on ``imdb_full`` and is therefore not run.

Configuration: the released tuned hyper-parameters for PostgreSQL-estimated cardinalities
(``setup/tuned_hyperparameters/tune_est_best_config.json``), Q-loss, at most ``--epochs`` epochs with the released
early stopping (best validation median Q-error on 15 % of the training plans, patience 20).

Protocol difference to the release, as for every model of this benchmark: the scaling statistics of the numeric
features are fitted on the *training* databases.  The vocabularies of the categorical features (operator names, data
types, ...) are taken over all databases because the released encoder fails on a value it has not seen.

    python analysis/baselines/run_zeroshot.py --split paper --seed 42 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.dace import DACE_DATA_DIR, WORKLOADS, count_nodes, load_plans, make_split, plans_path   # noqa: E402

BASE = os.path.join(os.path.dirname(ROOT), "baselines_src")
REPO = os.path.join(BASE, "zero-shot-cost-estimation")
PLANS = os.path.join(BASE, "zs_runs", "plans")
PY = os.environ.get("ZS_PYTHON", "python")
LEVELS = (50, 75, 90, 95, 99)


def zs_env(threads: int):
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    return env


def statistics(train_dbs, work, env):
    """Numeric scalers from the training databases, categorical vocabularies from all databases."""
    target = os.path.join(work, "statistics.json")
    if os.path.exists(target):
        return target
    parts = {}
    for name, dbs in (("train", train_dbs), ("all", WORKLOADS)):
        path = os.path.join(work, f"statistics_{name}.json")
        if not os.path.exists(path):
            files = [os.path.join(PLANS, f"{db}.json") for db in dbs]
            code = ("from models.preprocessing.feature_statistics import gather_feature_statistics;"
                    f"gather_feature_statistics({files!r}, {path!r})")
            subprocess.run([PY, "-c", code], cwd=REPO, env=env, check=True)
        parts[name] = json.load(open(path))
    merged = dict(parts["all"])
    for k, v in parts["train"].items():
        if v["type"] == "numeric":
            merged[k] = v
    json.dump(merged, open(target, "w"))
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="paper", choices=["paper", "rand"])
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results", "analysis", "dependency_20261003", "crossdb_zeroshot"))
    ap.add_argument("--convert-only", action="store_true", help="rebuild predictions/ and summary.json from the released test csv files")
    a = ap.parse_args()
    train_ids, test_ids, split_name = make_split(a.split, a.split_seed)
    train_dbs, test_dbs = [WORKLOADS[i] for i in train_ids], [WORKLOADS[i] for i in test_ids]
    if "imdb_full" in test_dbs or "tpc_h" in test_dbs:
        raise SystemExit("this split tests on a database whose benchmark file is not the released zero-shot workload")
    out = os.path.join(a.out_root, f"zeroshot_{split_name}_s{a.seed}")
    if os.path.exists(os.path.join(out, "summary.json")) and not a.convert_only:
        raise SystemExit(f"completed output exists: {out}")
    work = os.path.join(BASE, "zs_runs", split_name)
    target = os.path.join(work, f"s{a.seed}")
    os.makedirs(target, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    env = zs_env(a.threads)
    previous = json.load(open(os.path.join(out, "summary.json"))) if a.convert_only else None
    stats = statistics(train_dbs, work, env)
    cmd = [PY, "train.py", "--train_model",
           "--workload_runs", *[os.path.join(PLANS, f"{db}.json") for db in train_dbs],
           "--test_workload_runs", *[os.path.join(PLANS, f"{db}.json") for db in test_dbs],
           "--statistics_file", stats, "--target", target, "--filename_model", "zs",
           "--hyperparameter_path", "setup/tuned_hyperparameters/tune_est_best_config.json",
           "--max_no_epochs", str(a.epochs), "--loss_class_name", "QLoss", "--device", "cpu",
           "--num_workers", str(a.workers), "--seed", str(a.seed)]
    if not a.convert_only:
      json.dump({"command": cmd, "split": split_name, "train_dbs": train_dbs, "test_dbs": test_dbs, "seed": a.seed,
               "epochs": a.epochs, "statistics": "numeric: training databases; categorical vocabularies: all databases",
               "source": "https://github.com/DataManagementLab/zero-shot-cost-estimation",
               "repo_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(out, "config.json"), "w"), indent=2)
    t0 = time.time()
    if a.convert_only:
        train_s = previous["timing_s"]["train_s"]
    else:
        with open(os.path.join(out, "train.log"), "w") as log:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        train_s = time.time() - t0

    # released test csv (one row: labels, predictions and sample ids in seconds) -> the output contract of exp2
    csv.field_size_limit(10 ** 9)
    os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
    per_db = {}
    for db in test_dbs:
        row = next(csv.DictReader(open(os.path.join(target, f"test_zs_{db}.csv"))))
        idx = np.asarray(json.loads(row["val_sample_idxs"]))
        pred = np.empty(len(idx))
        pred[idx] = np.abs(np.asarray(json.loads(row["val_preds"]))) * 1000.0     # |prediction|, as the released Q-error metric
        bench = load_plans(plans_path(db, DACE_DATA_DIR))
        assert len(bench) == len(pred), (db, len(bench), len(pred))
        actual = np.asarray([p["Plan"]["Actual Total Time"] for p in bench], dtype=np.float64)   # label of the benchmark
        label = np.empty(len(idx))
        label[idx] = np.asarray(json.loads(row["val_labels"])) * 1000.0
        assert np.allclose(label, actual, rtol=0.35, atol=1.0), db                               # same plans, same order
        q = np.maximum(np.maximum(pred, 1e-7) / np.maximum(actual, 1e-7), np.maximum(actual, 1e-7) / np.maximum(pred, 1e-7))
        pd.DataFrame({"plan_id": np.arange(len(pred)), "pred_ms": pred, "actual_ms": actual, "qerror": q,
                      "n_nodes": [count_nodes(p["Plan"]) for p in bench]}).to_csv(
            os.path.join(out, "predictions", f"{db}.csv"), index=False)
        per_db[db] = {"n": int(len(q)), **{f"q{k}": float(np.percentile(q, k)) for k in LEVELS}, "q_mean": float(q.mean())}
    hist = list(csv.DictReader(open(os.path.join(target, "zs.csv"))))
    import torch
    state = torch.load(os.path.join(target, "zs.pt"), map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else {}
    n_params = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))) if isinstance(sd, dict) else None
    summary = {"model": "zeroshot", "params": {"total": n_params},
               "split": {"name": split_name, "train_dbs": train_dbs, "test_dbs": test_dbs, "n_train_plans": 5000 * len(train_dbs)},
               "statistics_scope": "train (numeric) / all (categorical vocabularies)", "qerror": per_db,
               "avg": {k: float(np.mean([m[k] for m in per_db.values()])) for k in [f"q{l}" for l in LEVELS] + ["q_mean"]},
               "train": {"epochs_run": len(hist), "max_epochs": a.epochs, "select": "best validation median Q-error (released)"},
               "timing_s": {"train_s": train_s}, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary["avg"], indent=1))


if __name__ == "__main__":
    main()
