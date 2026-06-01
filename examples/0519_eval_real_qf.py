"""
Evaluate the **real** QueryFormer (Zhao et al., VLDB 2022) baseline against
the same val split (partitions 18-19) used for GNTO recompute.

Distinguished from the existing ``QueryFormer`` entry in
``0519_recompute_metrics.py``: that one was a GNTO-family replica
(FeatureEmbed + a single GATConv). The *real* QF has 8 Transformer encoder
layers and 4.48M params; QF itself contains no GAT.

Output: ``results/RealQF_<timestamp>/summary.json`` with the same metric
schema as ``Recompute_*/summary.json`` so it slots directly into Table A.
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

# Need QueryFormer_VLDB2022 on path BEFORE GNTO so model.* resolves there
QF_PATH = os.path.abspath(os.path.join(GNTO_PATH, "..", "QueryFormer_VLDB2022"))
if QF_PATH not in sys.path:
    sys.path.insert(0, QF_PATH)

from model.model import QueryFormer
from model.dataset import PlanTreeDataset
from model.database_util import collator, get_hist_file, get_job_table_sample, Encoding
from model.util import Normalizer

from utils.metrics import evaluation_summary
from utils.benchmark import benchmark_callable, peak_gpu_memory_mb, model_size
from utils.plan_stats import join_counts_from_df, stratify_by_join_count


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _qf_predict_to_seconds(pred_norm, cost_norm):
    """QF's cost head outputs normalized; invert via cost_norm to raw exec time (ms)."""
    val = pred_norm * (cost_norm.maxi - cost_norm.mini) + cost_norm.mini
    # QF target = log(exec_time + 0.001)  (per TrainingV1.py / its Normalizer)
    return np.exp(val) - 0.001


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(GNTO_PATH, "results", f"RealQF_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # === QF resources ===
    data_path = os.path.join(QF_PATH, "data", "imdb") + "/"
    hist_file = get_hist_file(data_path + "histogram_string.csv")
    cost_norm = Normalizer(-3.61192, 12.290855)
    card_norm = Normalizer(1, 100)
    encoding = torch.load(os.path.join(QF_PATH, "checkpoints", "encoding.pt"),
                          weights_only=False)["encoding"]
    table_sample = get_job_table_sample(data_path + "train")

    # === val split: parts 18-19 ===
    print("Loading val partitions 18-19...")
    val_df = pd.concat([
        pd.read_csv(data_path + f"plan_and_cost/train_plan_part{i}.csv") for i in (18, 19)
    ]).reset_index(drop=True)
    print(f"  rows: {len(val_df)}")

    # === Dataset & loader (QF format) ===
    print("Building PlanTreeDataset (this preprocesses every plan, takes a while)...")
    t0 = time.perf_counter()
    val_ds = PlanTreeDataset(val_df, None, encoding, hist_file,
                              card_norm, cost_norm, "cost", table_sample)
    print(f"  built in {time.perf_counter() - t0:.1f}s")

    BATCH_SIZE = 128

    def iter_batches(ds, bs):
        """Mirror QF training-time mini-batching: collator wants transposed input."""
        n = len(ds)
        for i in range(0, n, bs):
            j = min(i + bs, n)
            items = [ds[k] for k in range(i, j)]
            yield collator(list(zip(*items)))

    # === Model + checkpoint ===
    model = QueryFormer(
        emb_size=64, ffn_dim=128, head_size=12, n_layers=8,
        dropout=0.1, pred_hid=128, use_sample=True, use_hist=True,
    ).to(device)
    ckpt_path = os.path.join(QF_PATH, "results", "full", "cost", "best_model.pt")
    print(f"Loading {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # === Inference ===
    print("Running inference...")
    preds_unnorm = []
    with torch.no_grad():
        for batch, _labels in iter_batches(val_ds, BATCH_SIZE):
            class _B: pass
            b = _B()
            b.attn_bias = batch.attn_bias.to(device)
            b.rel_pos = batch.rel_pos.to(device)
            b.heights = batch.heights.to(device)
            b.x = batch.x.to(device)
            cost_pred, _ = model(b)
            cost_pred = cost_pred.view(-1).cpu().numpy()
            preds_unnorm.extend(_qf_predict_to_seconds(cost_pred, cost_norm))

    preds_unnorm = np.asarray(preds_unnorm, dtype=np.float64)
    targets_raw = np.asarray(val_ds.costs, dtype=np.float64)

    # === Metrics ===
    summary = evaluation_summary(preds_unnorm, targets_raw)
    print("\n=== Real QueryFormer (4.48M params) ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    # === Benchmark ===
    print("\nRunning latency benchmark...")
    # pick one batch via the same iter_batches helper
    first_batch, _first_labels = next(iter(iter_batches(val_ds, BATCH_SIZE)))
    b = type("B", (), {})()
    b.attn_bias = first_batch.attn_bias.to(device)
    b.rel_pos = first_batch.rel_pos.to(device)
    b.heights = first_batch.heights.to(device)
    b.x = first_batch.x.to(device)
    is_cuda = device.type == "cuda"

    # GPU batch + single-plan (build a 1-plan batch by slicing the first sample)
    def _batch_forward():
        with torch.no_grad():
            model(b)

    bench_batch_gpu = benchmark_callable(_batch_forward, n_warmup=5, n_repeat=30, cuda_sync=is_cuda)

    # 1-plan batch
    b1 = type("B", (), {})()
    b1.attn_bias = b.attn_bias[:1]; b1.rel_pos = b.rel_pos[:1]
    b1.heights = b.heights[:1]; b1.x = b.x[:1]
    def _single_forward():
        with torch.no_grad():
            model(b1)
    bench_single_gpu = benchmark_callable(_single_forward, n_warmup=5, n_repeat=50, cuda_sync=is_cuda)

    peak_mb = peak_gpu_memory_mb(_batch_forward, device) if is_cuda else None
    ms = model_size(model)
    bench_gpu = {
        "device": str(device),
        "batch_size": int(b.x.size(0)),
        "single_plan_ms_mean": bench_single_gpu["mean_ms"],
        "single_plan_ms_median": bench_single_gpu["median_ms"],
        "single_plan_ms_p99": bench_single_gpu["p99_ms"],
        "batch_ms_mean": bench_batch_gpu["mean_ms"],
        "throughput_plans_per_sec": int(b.x.size(0)) / (bench_batch_gpu["mean_ms"] / 1000.0),
        "peak_gpu_mb": peak_mb,
        **ms,
    }

    # CPU
    model_cpu = QueryFormer(
        emb_size=64, ffn_dim=128, head_size=12, n_layers=8,
        dropout=0.1, pred_hid=128, use_sample=True, use_hist=True,
    )
    model_cpu.load_state_dict(sd)
    model_cpu.eval()
    bc = type("B", (), {})()
    bc.attn_bias = first_batch.attn_bias; bc.rel_pos = first_batch.rel_pos
    bc.heights = first_batch.heights; bc.x = first_batch.x
    bc1 = type("B", (), {})()
    bc1.attn_bias = bc.attn_bias[:1]; bc1.rel_pos = bc.rel_pos[:1]
    bc1.heights = bc.heights[:1]; bc1.x = bc.x[:1]
    def _cpu_batch():
        with torch.no_grad(): model_cpu(bc)
    def _cpu_single():
        with torch.no_grad(): model_cpu(bc1)
    bench_batch_cpu = benchmark_callable(_cpu_batch, n_warmup=2, n_repeat=10, cuda_sync=False)
    bench_single_cpu = benchmark_callable(_cpu_single, n_warmup=2, n_repeat=20, cuda_sync=False)
    bench_cpu = {
        "device": "cpu",
        "batch_size": int(bc.x.size(0)),
        "single_plan_ms_mean": bench_single_cpu["mean_ms"],
        "batch_ms_mean": bench_batch_cpu["mean_ms"],
        "throughput_plans_per_sec": int(bc.x.size(0)) / (bench_batch_cpu["mean_ms"] / 1000.0),
    }

    # Stratify
    join_counts = join_counts_from_df(val_df)
    n = min(len(preds_unnorm), len(join_counts))
    strat = stratify_by_join_count(preds_unnorm[:n], targets_raw[:n], join_counts[:n])

    out = {
        "label": "QueryFormer (real, transformer)",
        "ckpt": ckpt_path,
        "n_val": int(len(preds_unnorm)),
        "metrics": summary,
        "benchmark_gpu": bench_gpu,
        "benchmark_cpu": bench_cpu,
        "stratified_by_joins": strat,
    }
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(_to_serializable(out), f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(f"\nGPU: single={bench_gpu['single_plan_ms_mean']:.2f}ms  batch={bench_gpu['batch_ms_mean']:.2f}ms  "
          f"throughput={bench_gpu['throughput_plans_per_sec']:.0f}/s  peak_mb={bench_gpu['peak_gpu_mb']:.1f}  "
          f"params={bench_gpu['n_params']:,}  size_mb={bench_gpu['size_mb']:.2f}")
    print(f"CPU: single={bench_cpu['single_plan_ms_mean']:.2f}ms  batch={bench_cpu['batch_ms_mean']:.2f}ms  "
          f"throughput={bench_cpu['throughput_plans_per_sec']:.0f}/s")
    for b_, s in strat.items():
        if s["n"] > 0:
            print(f"  by-joins {b_}: n={s['n']:5d}  Q50={s.get('q50',0):.3f}  Q90={s.get('q90',0):.3f}  Q99={s.get('q99',0):.3f}")


if __name__ == "__main__":
    main()
