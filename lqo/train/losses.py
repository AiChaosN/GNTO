
from __future__ import annotations
import torch
import torch.nn.functional as F

def mse_log(pred_log: torch.Tensor, target_ms: torch.Tensor) -> torch.Tensor:
    """MSE in log-space to tame heavy tails."""
    eps = 1e-6
    t = torch.log(target_ms + eps)
    return F.mse_loss(pred_log, t)

def pairwise_margin_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Pairwise ranking: scores[i] should be < scores[j] if label[i] > label[j] (lower runtime better).
    labels: actual runtimes; smaller is better.
    """
    # Build pairs i,j with label[i] < label[j]
    idx = torch.arange(scores.shape[0], device=scores.device)
    i = idx.unsqueeze(1).expand(-1, scores.shape[0]).reshape(-1)
    j = idx.unsqueeze(0).expand(scores.shape[0], -1).reshape(-1)
    mask = labels[i] < labels[j]
    if mask.sum() == 0:
        return scores.new_zeros((), requires_grad=True)
    s_i = scores[i[mask]]
    s_j = scores[j[mask]]
    # We want s_i < s_j (i better than j), use hinge
    return torch.relu(margin - (s_j - s_i)).mean()
