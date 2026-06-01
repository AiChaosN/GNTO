"""
Inference latency / throughput / memory benchmarks for cost-prediction baselines.

Addresses review item R3-D1 (prediction latency as a scalability metric) and
R2-W1 (memory footprint analysis). Two backends are supported:

- ``benchmark_pyg(...)`` for PyG-style models (GNTO / QueryFormer)
- ``benchmark_callable(...)`` for arbitrary forward callables (Bao, zero-shot)
"""

from __future__ import annotations

import time
import gc
from typing import Callable, Optional

import numpy as np


def model_size(model) -> dict:
    """Parameter count and dense-storage footprint (params + buffers)."""
    import torch

    if not hasattr(model, "parameters"):
        return {"n_params": None, "size_mb": None}
    n_params = sum(p.numel() for p in model.parameters())
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_total += sum(b.numel() * b.element_size() for b in model.buffers())
    return {"n_params": int(n_params), "size_mb": bytes_total / 1e6}


def benchmark_callable(
    forward_fn: Callable[[], object],
    *,
    n_warmup: int = 10,
    n_repeat: int = 100,
    cuda_sync: bool = False,
) -> dict:
    """Measure per-call latency of a no-arg forward callable.

    Args:
        forward_fn: any 0-arg callable that runs one inference step
        n_warmup: discarded warm-up calls (JIT/caches)
        n_repeat: timed calls
        cuda_sync: if True, ``torch.cuda.synchronize()`` after each call
            (required for accurate GPU timing). Set to False for CPU/no-GPU.

    Returns: dict with mean/median/p50/p90/p99 latency in milliseconds.
    """
    import torch

    for _ in range(n_warmup):
        forward_fn()
    if cuda_sync:
        torch.cuda.synchronize()

    times = np.empty(n_repeat, dtype=np.float64)
    for i in range(n_repeat):
        t0 = time.perf_counter()
        forward_fn()
        if cuda_sync:
            torch.cuda.synchronize()
        times[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
        "p99_ms": float(np.percentile(times, 99)),
        "n_repeat": int(n_repeat),
    }


def peak_gpu_memory_mb(forward_fn: Callable[[], object], device) -> Optional[float]:
    """Peak GPU memory consumed by a single ``forward_fn()`` call (MB).

    Returns None if ``device`` is not a CUDA device.
    """
    import torch

    if getattr(device, "type", None) != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    forward_fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1e6


def benchmark_pyg(
    model,
    sample_batch,
    device,
    *,
    n_warmup: int = 10,
    n_single: int = 100,
    n_throughput: int = 50,
) -> dict:
    """End-to-end benchmark for a PyG model on a single ready-made batch.

    Reports:
        - single-plan latency (mean/median/p90/p99) on whatever device the
          model is currently on
        - per-batch latency for the supplied ``sample_batch``
        - throughput in plans/sec (= batch size / mean batch latency)
        - peak GPU memory (None if CPU)
        - model_size (params + bytes)

    The caller is responsible for moving model + sample to the desired device
    before calling. To measure CPU vs GPU separately, call this twice.
    """
    import torch

    is_cuda = getattr(device, "type", None) == "cuda"
    model.eval()
    sample_batch = sample_batch.to(device)
    batch_size = int(getattr(sample_batch, "num_graphs", 1))

    with torch.no_grad():
        # Build a 1-plan batch from sample_batch[0] if possible; otherwise reuse
        try:
            from torch_geometric.data import Batch
            single = Batch.from_data_list([sample_batch.get_example(0).to(device)])
        except Exception:
            single = sample_batch

        single_lat = benchmark_callable(
            lambda: model(single),
            n_warmup=n_warmup,
            n_repeat=n_single,
            cuda_sync=is_cuda,
        )

        batch_lat = benchmark_callable(
            lambda: model(sample_batch),
            n_warmup=n_warmup,
            n_repeat=n_throughput,
            cuda_sync=is_cuda,
        )

        peak_mb = peak_gpu_memory_mb(lambda: model(sample_batch), device) if is_cuda else None

    throughput = batch_size / (batch_lat["mean_ms"] / 1000.0)
    return {
        "device": str(device),
        "batch_size": batch_size,
        "single_plan_ms_mean": single_lat["mean_ms"],
        "single_plan_ms_median": single_lat["median_ms"],
        "single_plan_ms_p99": single_lat["p99_ms"],
        "batch_ms_mean": batch_lat["mean_ms"],
        "throughput_plans_per_sec": float(throughput),
        "peak_gpu_mb": peak_mb,
        **model_size(model),
    }
