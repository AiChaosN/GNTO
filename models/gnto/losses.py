"""Training objectives.

* point-wise: ``mse``, ``l1``, ``huber`` (on log-latency)
* pair-wise:  :class:`PairwiseRankingLoss` - orders candidate plans *within a
  group* (e.g. the hint-set candidates of one query).  Only relative order
  matters for plan steering.
* :class:`CombinedLoss` = point-wise + ``rank_weight`` * pair-wise.

All losses take ``(pred, target)`` of shape ``[B]`` (or ``[B, 1]``); the
ranking losses additionally take ``groups`` (``LongTensor [B]``).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _flat(t: Tensor) -> Tensor:
    return t.reshape(-1)


class PairwiseRankingLoss(nn.Module):
    """For every pair ``(i, j)`` in the same group with ``target_i < target_j``
    (plan *i* is faster) penalise ``pred_i >= pred_j``.

    ``kind='logistic'``: ``softplus(-(pred_j - pred_i))`` (RankNet).
    ``kind='hinge'``:    ``relu(margin - (pred_j - pred_i))``.
    Pairs whose targets differ by less than ``min_gap`` (in the target's units,
    e.g. log-seconds) are ignored - they are too close to order reliably.
    Returns 0 when the batch contains no comparable pair.
    """

    def __init__(self, kind: str = "logistic", margin: float = 0.1, min_gap: float = 0.0):
        super().__init__()
        if kind not in ("logistic", "hinge"):
            raise ValueError("kind must be 'logistic' or 'hinge'")
        self.kind, self.margin, self.min_gap = kind, margin, min_gap

    def forward(self, pred: Tensor, target: Tensor, groups: Optional[Tensor] = None) -> Tensor:
        pred, target = _flat(pred), _flat(target).to(pred.dtype)
        n = pred.numel()
        if n < 2:
            return pred.sum() * 0.0
        if groups is None:
            groups = torch.zeros(n, dtype=torch.long, device=pred.device)
        groups = _flat(groups)
        same = groups.unsqueeze(1) == groups.unsqueeze(0)
        gap = target.unsqueeze(0) - target.unsqueeze(1)          # gap[i, j] = t_j - t_i
        valid = same & (gap > self.min_gap)                      # i faster than j
        if not valid.any():
            return pred.sum() * 0.0
        diff = pred.unsqueeze(0) - pred.unsqueeze(1)             # diff[i, j] = p_j - p_i  (want > 0)
        diff = diff[valid]
        if self.kind == "logistic":
            return F.softplus(-diff).mean()
        return F.relu(self.margin - diff).mean()


class PointwiseLoss(nn.Module):
    def __init__(self, kind: str = "huber", delta: float = 1.0):
        super().__init__()
        if kind not in ("mse", "l1", "huber"):
            raise ValueError("kind must be 'mse', 'l1' or 'huber'")
        self.kind, self.delta = kind, delta

    def forward(self, pred: Tensor, target: Tensor, groups: Optional[Tensor] = None) -> Tensor:
        pred, target = _flat(pred), _flat(target).to(pred.dtype)
        if self.kind == "mse":
            return F.mse_loss(pred, target)
        if self.kind == "l1":
            return F.l1_loss(pred, target)
        return F.huber_loss(pred, target, delta=self.delta)


class CombinedLoss(nn.Module):
    """``point(pred, target) + rank_weight * rank(pred, target, groups)``."""

    def __init__(self, point: nn.Module, rank: nn.Module, rank_weight: float = 1.0):
        super().__init__()
        self.point, self.rank, self.rank_weight = point, rank, rank_weight

    def forward(self, pred: Tensor, target: Tensor, groups: Optional[Tensor] = None) -> Tensor:
        return self.point(pred, target) + self.rank_weight * self.rank(pred, target, groups)


def build_loss(name: str = "huber", delta: float = 1.0, rank_weight: float = 1.0,
               rank_kind: str = "logistic", margin: float = 0.1, min_gap: float = 0.0) -> nn.Module:
    """``name`` in ``mse | l1 | huber | pairwise | <point>+pairwise`` (e.g. ``huber+pairwise``)."""
    name = name.lower().replace(" ", "")
    if name == "pairwise":
        return PairwiseRankingLoss(rank_kind, margin, min_gap)
    if "+" in name:
        point_name, rank_name = name.split("+", 1)
        if rank_name != "pairwise":
            raise ValueError(f"unknown combined loss {name!r}")
        return CombinedLoss(PointwiseLoss(point_name, delta),
                            PairwiseRankingLoss(rank_kind, margin, min_gap), rank_weight)
    return PointwiseLoss(name, delta)
