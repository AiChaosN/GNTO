"""
Unified evaluation metrics for cost-prediction baselines (GNTO / QueryFormer / Bao / Zero-shot).

Single source of truth for:
- Q-Error percentiles (matches the existing qf_adapter / bao_adapter formula)
- Global ranking metrics (Spearman rho, Kendall tau, pairwise accuracy)
- Per-query Top-K regret (requires group_ids; used for Bao-style steering eval)

These metrics address review items R3-D1 (ranking + latency) and feed Table 4.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, kendalltau


def qerror_percentiles(preds, targets, percentiles=(50, 75, 90, 95, 99)):
    """Per-sample Q-Error = max(p/t, t/p); returns dict {qNN: value}."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    qerrors = np.empty(len(preds), dtype=np.float64)

    both_zero = (preds == 0) & (targets == 0)
    either_zero = ((preds == 0) | (targets == 0)) & ~both_zero
    nz = ~(both_zero | either_zero)

    qerrors[both_zero] = 1.0
    qerrors[either_zero] = np.inf
    p, t = preds[nz], targets[nz]
    qerrors[nz] = np.maximum(p / t, t / p)

    pcts = np.percentile(qerrors, list(percentiles))
    return {f"q{int(pc)}": float(v) for pc, v in zip(percentiles, pcts)}


def ranking_metrics(preds, targets):
    """Global Spearman rho + Kendall tau between predicted and actual."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(preds) < 2:
        return {"spearman": float("nan"), "kendall": float("nan")}
    rho, _ = spearmanr(preds, targets)
    tau, _ = kendalltau(preds, targets)
    return {"spearman": float(rho), "kendall": float(tau)}


def pairwise_accuracy(preds, targets, n_samples=100_000, seed=0):
    """Fraction of pairs (i, j) where the model orders them like ground truth.

    Equivalent to (Kendall tau + 1) / 2 when there are no ties. Sampled for
    large N to keep cost O(n_samples) instead of O(N^2).
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    N = len(preds)
    if N < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    total_pairs = N * (N - 1) // 2
    if total_pairs <= n_samples:
        i, j = np.triu_indices(N, k=1)
    else:
        i = rng.integers(0, N, size=n_samples)
        j = rng.integers(0, N, size=n_samples)
        keep = i != j
        i, j = i[keep], j[keep]

    correct = (preds[i] < preds[j]) == (targets[i] < targets[j])
    return float(correct.mean())


def topk_regret(preds, targets, group_ids, k=1):
    """Per-group regret from trusting the model's top-k pick.

    For each group, takes the k smallest-predicted plans and reports the best
    actual runtime among them. Lower predicted = better (cost/latency).
    Groups with fewer than 2 plans are skipped.
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    group_ids = np.asarray(group_ids)

    abs_regrets, rel_regrets = [], []
    for g in np.unique(group_ids):
        m = group_ids == g
        if m.sum() < 2:
            continue
        pg, tg = preds[m], targets[m]
        topk_idx = np.argsort(pg)[:k]
        model_pick = tg[topk_idx].min()
        oracle = tg.min()
        abs_regrets.append(model_pick - oracle)
        if oracle > 0:
            rel_regrets.append(model_pick / oracle)

    if not abs_regrets:
        return {
            f"top{k}_regret_mean": float("nan"),
            f"top{k}_regret_median": float("nan"),
            f"top{k}_relative_regret_mean": float("nan"),
        }
    return {
        f"top{k}_regret_mean": float(np.mean(abs_regrets)),
        f"top{k}_regret_median": float(np.median(abs_regrets)),
        f"top{k}_relative_regret_mean": float(np.mean(rel_regrets)) if rel_regrets else float("nan"),
    }


def evaluation_summary(
    preds,
    targets,
    group_ids=None,
    percentiles=(50, 75, 90, 95, 99),
    include_pairwise=True,
    pairwise_samples=100_000,
):
    """One-stop summary used by all adapters. Flat dict for easy CSV/JSON logging."""
    summary = {}
    summary.update(qerror_percentiles(preds, targets, percentiles))
    summary.update(ranking_metrics(preds, targets))
    if include_pairwise:
        summary["pairwise_acc"] = pairwise_accuracy(preds, targets, n_samples=pairwise_samples)
    if group_ids is not None:
        summary.update(topk_regret(preds, targets, group_ids, k=1))
    return summary
