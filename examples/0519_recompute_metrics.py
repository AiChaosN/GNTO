"""
Re-evaluate existing GNTO / QueryFormer / Bao checkpoints with the unified
metric suite (utils/metrics.py + utils/benchmark.py + utils/plan_stats.py).

Reads existing checkpoints / saved predictions, produces a single
``results/Recompute_<timestamp>/summary.json`` with:

- Q-Error percentiles (Q50/Q75/Q90/Q95/Q99)
- Ranking metrics (Spearman rho, Kendall tau, pairwise accuracy)
- Inference latency (single-plan CPU & GPU, batch throughput)
- Model size + peak GPU memory
- Per-join-bucket Q-Error (1-2 / 3-4 / 5+)

This addresses review items R3-D1 (ranking + latency) and R2-W2 (join-stratified).
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from adapters.qf_adapter import (
    load_qf_resources, QueryFormerToPyGDataset, GNTO_QF_Model,
    evaluate_full,
)
from utils.metrics import evaluation_summary
from utils.benchmark import benchmark_pyg
from utils.plan_stats import join_counts_from_df, stratify_by_join_count


def _to_serializable(obj):
    """Recursively convert numpy / torch values for json.dump."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def eval_pyg_model(model_cls_factory, ckpt_path, val_df, qf_res, device,
                   add_plan_rows: bool, cache_name: str, label: str):
    """Load a PyG checkpoint and produce the full metric package."""
    print(f"\n[{label}] Building val dataset (add_plan_rows={add_plan_rows})...")
    val_ds = QueryFormerToPyGDataset(
        val_df, qf_res["encoding"], qf_res["hist_file"], qf_res["table_sample"],
        qf_res["cost_norm"], cache_name=cache_name, add_plan_rows=add_plan_rows,
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    print(f"[{label}] Loading {ckpt_path}...")
    model = model_cls_factory(qf_res["encoding"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    summary = evaluate_full(model, val_loader, qf_res["cost_norm"], device)
    preds = summary.pop("preds")
    targets = summary.pop("targets")

    # Benchmark on a sample batch
    sample_batch = next(iter(val_loader))
    bench_gpu = benchmark_pyg(model, sample_batch, device,
                              n_warmup=5, n_single=50, n_throughput=20)
    bench_cpu = benchmark_pyg(model.cpu(), sample_batch.cpu(), torch.device("cpu"),
                              n_warmup=5, n_single=50, n_throughput=20)

    # Stratify
    join_counts = join_counts_from_df(val_df)
    n = min(len(preds), len(join_counts))
    strat = stratify_by_join_count(preds[:n], targets[:n], join_counts[:n])

    return {
        "label": label,
        "ckpt": ckpt_path,
        "n_val": int(n),
        "metrics": summary,
        "benchmark_gpu": bench_gpu,
        "benchmark_cpu": bench_cpu,
        "stratified_by_joins": strat,
    }


def eval_bao_from_predictions(pred_csv, val_df, label="Bao"):
    """Re-score Bao's saved predictions.csv with the unified metrics."""
    print(f"\n[{label}] Loading {pred_csv}...")
    df = pd.read_csv(pred_csv)
    sub = df[df["split"] == "val"].reset_index(drop=True)
    preds = sub["pred"].to_numpy()
    targets = sub["target"].to_numpy()

    summary = evaluation_summary(preds, targets)

    # Stratify - assume val_df rows align with val predictions (they should,
    # since the Bao baseline loaded the same val_df via plans_from_qf_df)
    join_counts = join_counts_from_df(val_df)
    n = min(len(preds), len(join_counts))
    strat = stratify_by_join_count(preds[:n], targets[:n], join_counts[:n])

    return {
        "label": label,
        "predictions_csv": pred_csv,
        "n_val": int(n),
        "metrics": summary,
        "benchmark_note": "see existing Bao summary.json for val_inference_per_plan_ms",
        "stratified_by_joins": strat,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(GNTO_PATH, "results", f"Recompute_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print("Loading QF resources...")
    qf_res = load_qf_resources()

    print("Loading val_df (parts 18-19)...")
    val_dfs = [pd.read_csv(qf_res["data_path"] + f"plan_and_cost/train_plan_part{i}.csv")
               for i in (18, 19)]
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    print(f"val_df rows: {len(val_df)}")

    results = []

    # --- 1) Bao: existing predictions.csv ---
    bao_pred_csv = os.path.join(GNTO_PATH, "results", "Bao_0506_1435", "predictions.csv")
    if os.path.exists(bao_pred_csv):
        try:
            r = eval_bao_from_predictions(bao_pred_csv, val_df, label="Bao")
            results.append(r)
        except Exception as e:
            print(f"[Bao] FAILED: {e}")

    # Note: real QueryFormer (transformer, 4.48M params) is evaluated by a
    # separate script (examples/0519_eval_real_qf.py) because it uses QF's
    # native PlanTreeDataset + collator, not our PyG path.

    # --- 2) GNTO SOTA (with PlanRows + GATv2) ---
    gnto_ckpt = os.path.join(GNTO_PATH, "results", "GNTO_QF_0519_1607", "best_model.pth")
    if os.path.exists(gnto_ckpt):
        try:
            r = eval_pyg_model(
                model_cls_factory=lambda enc: GNTO_QF_Model(enc, add_plan_rows=True),
                ckpt_path=gnto_ckpt, val_df=val_df, qf_res=qf_res,
                device=device, add_plan_rows=True, cache_name="val_full",
                label="GNTO",
            )
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[GNTO] FAILED: {e}")

    # --- write summary ---
    out = {
        "timestamp": timestamp,
        "device": str(device),
        "val_n": len(val_df),
        "models": results,
    }
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(_to_serializable(out), f, indent=2, default=str)
    print(f"\n=== DONE ===\nWrote {out_path}")

    # Quick comparison print
    print("\n=== Comparison table ===")
    cols = ("q50", "q90", "q99", "spearman", "kendall", "pairwise_acc")
    print(f"{'Model':<12}" + "".join(f"{c:>12}" for c in cols))
    for r in results:
        m = r["metrics"]
        vals = [m.get(c, float("nan")) for c in cols]
        print(f"{r['label']:<12}" + "".join(f"{v:>12.4f}" for v in vals))


if __name__ == "__main__":
    main()
