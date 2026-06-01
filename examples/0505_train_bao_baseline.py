"""Bao baseline (SIGMOD 2021) on the QueryFormer-format IMDB workload.

Train splits / val splits mirror examples/1216_compGntoWithQF_addPlanrows.py
(parts 0-17 train, parts 18-19 val) so Q-Error metrics are directly comparable
with GNTO's SOTA run.

Outputs a CSV log + final summary under results/Bao_<timestamp>/.

Addresses VLDB R1-D5 / R3-D2: adds Bao native tree-CNN as cost-estimation
baseline alongside QueryFormer.
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from adapters.bao_adapter import (
    BaoCostPredictor, plans_from_qf_df, calc_q_error,
)
from adapters.qf_adapter import load_qf_resources


def main():
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(GNTO_PATH, "results", f"Bao_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Outputs → {save_dir}")

    # Reuse the same data root as QF runs (so train/val splits match GNTO).
    print("Loading QueryFormer resources (for data path only)...")
    res = load_qf_resources()
    data_path = res["data_path"]

    print("Loading train plans (parts 0-17)...")
    train_df = pd.concat([
        pd.read_csv(data_path + f"plan_and_cost/train_plan_part{i}.csv")
        for i in range(18)
    ])
    print(f"  train: {len(train_df)}")

    print("Loading val plans (parts 18-19)...")
    val_df = pd.concat([
        pd.read_csv(data_path + f"plan_and_cost/train_plan_part{i}.csv")
        for i in range(18, 20)
    ])
    print(f"  val:   {len(val_df)}")

    train_plans, train_y = plans_from_qf_df(train_df, target="execution_time")
    val_plans, val_y = plans_from_qf_df(val_df, target="execution_time")

    print("\n=== Training Bao (BaoNet, 100 epochs hardcoded upstream) ===")
    t0 = time.time()
    bao = BaoCostPredictor(verbose=True)
    bao.fit(train_plans, train_y)
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s")

    bao.save(os.path.join(save_dir, "bao_model"))

    print("\n=== Evaluating ===")
    t0 = time.time()
    val_preds = bao.predict(val_plans)
    val_infer_time = time.time() - t0
    train_preds = bao.predict(train_plans)

    train_q = calc_q_error(train_preds, train_y)
    val_q = calc_q_error(val_preds, val_y)

    summary = {
        "n_train": len(train_plans),
        "n_val": len(val_plans),
        "train_time_sec": train_time,
        "val_inference_time_sec": val_infer_time,
        "val_inference_per_plan_ms": 1000.0 * val_infer_time / max(1, len(val_plans)),
        "train_q50": float(train_q[0]),
        "train_q75": float(train_q[1]),
        "train_q90": float(train_q[2]),
        "train_q95": float(train_q[3]),
        "train_q99": float(train_q[4]),
        "val_q50": float(val_q[0]),
        "val_q75": float(val_q[1]),
        "val_q90": float(val_q[2]),
        "val_q95": float(val_q[3]),
        "val_q99": float(val_q[4]),
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({
        "split": ["train"] * len(train_plans) + ["val"] * len(val_plans),
        "pred": np.concatenate([train_preds, val_preds]),
        "target": np.concatenate([train_y, val_y]),
    }).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    print("\n=== Bao Q-Error Summary ===")
    print(f"Train: Q50={summary['train_q50']:.3f}  Q90={summary['train_q90']:.3f}  Q99={summary['train_q99']:.3f}")
    print(f"Val:   Q50={summary['val_q50']:.3f}  Q90={summary['val_q90']:.3f}  Q99={summary['val_q99']:.3f}")
    print(f"Inference: {summary['val_inference_per_plan_ms']:.2f} ms/plan")
    print(f"Saved → {save_dir}")


if __name__ == "__main__":
    main()
