
from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple
from scipy.stats import spearmanr

def q_error(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    q = np.maximum(pred / np.maximum(label, 1e-9), label / np.maximum(pred, 1e-9))
    return float(np.median(q)), float(np.percentile(q, 90)), float(np.percentile(q, 99))

def selected_runtime(pred: Sequence[float], actual: Sequence[float]) -> float:
    """Runtime of the plan the model would pick (min pred)."""
    idx = int(np.argmin(pred))
    return float(actual[idx])

def surpassed_plans(pred: Sequence[float], actual: Sequence[float]) -> float:
    idx = int(np.argmin(pred))
    a = float(actual[idx])
    return float((np.array(actual) > a).mean() * 100.0)

def rank_corr(pred: Sequence[float], actual: Sequence[float]) -> float:
    rho, _ = spearmanr(pred, actual)
    return float(rho if rho == rho else 0.0)  # handle nan

def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = ((y_true==1) & (y_pred==1)).sum()
    tn = ((y_true==0) & (y_pred==0)).sum()
    fn = ((y_true==1) & (y_pred==0)).sum()
    fp = ((y_true==0) & (y_pred==1)).sum()
    tpr = tp / (tp + fn + 1e-9); tnr = tn / (tn + fp + 1e-9)
    return float(0.5*(tpr+tnr))

def pick_rate(optimal_idx: Sequence[int], picked_idx: Sequence[int]) -> float:
    opt = np.array(optimal_idx); pick = np.array(picked_idx)
    return float((opt == pick).mean() * 100.0)
