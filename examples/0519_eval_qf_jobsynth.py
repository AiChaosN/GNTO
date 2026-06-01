"""
Evaluate GNTO (current best ckpt) and the real QueryFormer (transformer, 4.48M)
on the two QF datasets that weren't yet covered: JOB-light (70 queries) and
synthetic (500 queries).

These datasets carry the more complex queries the paper needs for the per-join
breakdown (R2-W2), and they're what QF's own `Training V1_Runned.ipynb`
reports against.

Output: ``results/QFvsGNTO_jobsynth_<timestamp>/summary.json`` with all four
runs (GNTO/JOB-light, GNTO/synthetic, RealQF/JOB-light, RealQF/synthetic),
each carrying Q-Error + ranking metrics + benchmark + join-count breakdown.
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

QF_PATH = os.path.abspath(os.path.join(GNTO_PATH, "..", "QueryFormer_VLDB2022"))
if QF_PATH not in sys.path:
    sys.path.insert(0, QF_PATH)

from adapters.qf_adapter import (
    load_qf_resources, QueryFormerToPyGDataset, GNTO_QF_Model, evaluate_full,
)
from model.model import QueryFormer
from model.dataset import PlanTreeDataset
from model.database_util import collator, get_hist_file, get_job_table_sample
from model.util import Normalizer

from utils.metrics import evaluation_summary
from utils.benchmark import benchmark_pyg, benchmark_callable, peak_gpu_memory_mb, model_size
from utils.plan_stats import join_counts_from_df, stratify_by_join_count


DATASETS = {
    # name: (plan_csv_filename, workload_sample_path_relative)
    "JOB-light": ("job-light_plan.csv", "workloads/job-light"),
    "synthetic": ("synthetic_plan.csv", "workloads/synthetic"),
}


def _serializable(obj):
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _unnormalize_qf_pred(pred_norm, cost_norm):
    val = pred_norm * (cost_norm.maxi - cost_norm.mini) + cost_norm.mini
    return np.exp(val) - 0.001


def eval_gnto(ds_name, df, gnto_ckpt, qf_res, table_sample, device):
    print(f"\n[GNTO] {ds_name} (n={len(df)}) building PyG dataset...")
    ds = QueryFormerToPyGDataset(
        df, qf_res["encoding"], qf_res["hist_file"], table_sample,
        qf_res["cost_norm"], cache_name=None, add_plan_rows=True,
    )
    loader = DataLoader(ds, batch_size=min(64, len(df)), shuffle=False, num_workers=0)

    model = GNTO_QF_Model(qf_res["encoding"], add_plan_rows=True).to(device)
    model.load_state_dict(torch.load(gnto_ckpt, map_location=device, weights_only=False))
    model.eval()

    summary = evaluate_full(model, loader, qf_res["cost_norm"], device)
    preds = summary.pop("preds"); targets = summary.pop("targets")

    sample_batch = next(iter(loader))
    bench_gpu = benchmark_pyg(model, sample_batch, device, n_warmup=5, n_single=30, n_throughput=10)
    bench_cpu = benchmark_pyg(model.cpu(), sample_batch.cpu(), torch.device("cpu"),
                              n_warmup=2, n_single=20, n_throughput=10)
    jc = join_counts_from_df(df)
    n = min(len(preds), len(jc))
    strat = stratify_by_join_count(preds[:n], targets[:n], jc[:n])

    return {
        "model": "GNTO",
        "dataset": ds_name,
        "n": int(n),
        "metrics": summary,
        "benchmark_gpu": bench_gpu,
        "benchmark_cpu": bench_cpu,
        "stratified_by_joins": strat,
    }


def _iter_qf_batches(ds, bs):
    n = len(ds)
    for i in range(0, n, bs):
        j = min(i + bs, n)
        items = [ds[k] for k in range(i, j)]
        yield collator(list(zip(*items)))


def eval_real_qf(ds_name, df, qf_ckpt, qf_res, table_sample, device):
    print(f"\n[RealQF] {ds_name} (n={len(df)}) building PlanTreeDataset...")
    card_norm = Normalizer(1, 100)
    ds = PlanTreeDataset(df, None, qf_res["encoding"], qf_res["hist_file"],
                         card_norm, qf_res["cost_norm"], "cost", table_sample)

    model = QueryFormer(
        emb_size=64, ffn_dim=128, head_size=12, n_layers=8,
        dropout=0.1, pred_hid=128, use_sample=True, use_hist=True,
    ).to(device)
    sd = torch.load(qf_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    preds = []
    BATCH = min(64, len(ds))
    with torch.no_grad():
        for batch, _labels in _iter_qf_batches(ds, BATCH):
            class _B: pass
            b = _B()
            b.attn_bias = batch.attn_bias.to(device)
            b.rel_pos = batch.rel_pos.to(device)
            b.heights = batch.heights.to(device)
            b.x = batch.x.to(device)
            cost_pred, _ = model(b)
            preds.extend(_unnormalize_qf_pred(cost_pred.view(-1).cpu().numpy(), qf_res["cost_norm"]))
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(ds.costs, dtype=np.float64)
    summary = evaluation_summary(preds, targets)

    # benchmark
    first_batch, _ = next(iter(_iter_qf_batches(ds, BATCH)))
    is_cuda = device.type == "cuda"
    bgpu = type("B", (), {})()
    bgpu.attn_bias = first_batch.attn_bias.to(device)
    bgpu.rel_pos = first_batch.rel_pos.to(device); bgpu.heights = first_batch.heights.to(device)
    bgpu.x = first_batch.x.to(device)
    bgpu_1 = type("B", (), {})()
    bgpu_1.attn_bias = bgpu.attn_bias[:1]; bgpu_1.rel_pos = bgpu.rel_pos[:1]
    bgpu_1.heights = bgpu.heights[:1]; bgpu_1.x = bgpu.x[:1]
    bench_b = benchmark_callable(lambda: model(bgpu), n_warmup=3, n_repeat=10, cuda_sync=is_cuda)
    bench_s = benchmark_callable(lambda: model(bgpu_1), n_warmup=3, n_repeat=20, cuda_sync=is_cuda)
    peak_mb = peak_gpu_memory_mb(lambda: model(bgpu), device) if is_cuda else None
    bench_gpu = {
        "device": str(device), "batch_size": int(bgpu.x.size(0)),
        "single_plan_ms_mean": bench_s["mean_ms"], "single_plan_ms_median": bench_s["median_ms"],
        "single_plan_ms_p99": bench_s["p99_ms"],
        "batch_ms_mean": bench_b["mean_ms"],
        "throughput_plans_per_sec": int(bgpu.x.size(0)) / (bench_b["mean_ms"] / 1000.0),
        "peak_gpu_mb": peak_mb,
        **model_size(model),
    }

    jc = join_counts_from_df(df)
    strat = stratify_by_join_count(preds, targets, jc)

    return {
        "model": "RealQF",
        "dataset": ds_name,
        "n": int(len(preds)),
        "metrics": summary,
        "benchmark_gpu": bench_gpu,
        "stratified_by_joins": strat,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(GNTO_PATH, "results", f"QFvsGNTO_jobsynth_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    qf_res = load_qf_resources()
    gnto_ckpt = os.path.join(GNTO_PATH, "results", "GNTO_QF_0519_1607", "best_model.pth")
    qf_ckpt = os.path.join(QF_PATH, "results", "full", "cost", "best_model.pt")

    print(f"GNTO ckpt: {gnto_ckpt}")
    print(f"Real QF ckpt: {qf_ckpt}")

    all_runs = []
    for ds_name, (csv_name, workload_rel) in DATASETS.items():
        df = pd.read_csv(os.path.join(qf_res["data_path"], csv_name)).reset_index(drop=True)
        print(f"\n=========  {ds_name}: {len(df)} queries  =========")
        sample_path = os.path.join(qf_res["data_path"], workload_rel)
        table_sample = get_job_table_sample(sample_path)
        print(f"  table_sample size: {len(table_sample)}")

        try:
            all_runs.append(eval_gnto(ds_name, df, gnto_ckpt, qf_res, table_sample, device))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[GNTO/{ds_name}] FAILED: {e}")

        try:
            all_runs.append(eval_real_qf(ds_name, df, qf_ckpt, qf_res, table_sample, device))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[RealQF/{ds_name}] FAILED: {e}")

    out = {"timestamp": timestamp, "device": str(device), "runs": all_runs}
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(_serializable(out), f, indent=2, default=str)

    print(f"\n\n=== DONE === wrote {out_path}\n")
    cols = ("q50", "q90", "q99", "spearman", "kendall", "pairwise_acc")
    print(f"{'model':<10}{'dataset':<14}{'n':>6}" + "".join(f"{c:>12}" for c in cols))
    for r in all_runs:
        m = r["metrics"]; vals = [m.get(c, float("nan")) for c in cols]
        print(f"{r['model']:<10}{r['dataset']:<14}{r['n']:>6}" + "".join(f"{v:>12.4f}" for v in vals))
    print()
    for r in all_runs:
        print(f"--- {r['model']} on {r['dataset']} ---")
        for b_, s in r["stratified_by_joins"].items():
            if s["n"] > 0:
                print(f"  joins={b_:<4} n={s['n']:4d}  Q50={s.get('q50', 0):.3f}  Q90={s.get('q90', 0):.3f}  Q99={s.get('q99', 0):.3f}")


if __name__ == "__main__":
    main()
