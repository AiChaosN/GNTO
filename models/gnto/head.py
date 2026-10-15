"""Prediction head: plan embedding -> scalar (log-latency) or vector output."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import HeadConfig


class ResBlock(nn.Module):
    """``x + fc2(drop(gelu(fc1(norm(x)))))`` (identical to the legacy ResBlock)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        return x + self.fc2(h)


class PredictionHead(nn.Module):
    """``kind='resmlp'``: ``Linear -> GELU -> ResBlock x num_blocks -> LayerNorm -> Linear``
    (parameter names ``inp / blocks / norm / out`` match the legacy
    ``PredictionHead_V2``).  ``kind='mlp'``: plain ``Linear -> ReLU`` stack.

    ``out_activation``: ``none`` (log-space regression), ``sigmoid`` (targets
    min-max normalised to [0, 1], as in the QueryFormer pipeline) or
    ``softplus`` (strictly positive output).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_blocks: int = 2, dropout: float = 0.0,
                 out_dim: int = 1, out_activation: str = "none", kind: str = "resmlp"):
        super().__init__()
        self.in_dim, self.out_dim, self.kind = in_dim, out_dim, kind
        self.out_activation = out_activation
        if kind == "resmlp":
            self.inp = nn.Linear(in_dim, hidden_dim)
            self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
            self.norm = nn.LayerNorm(hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)
        elif kind == "mlp":
            layers = []
            d = in_dim
            for _ in range(max(num_blocks, 1)):
                layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown head kind {kind!r}")

    @classmethod
    def from_config(cls, cfg: HeadConfig, in_dim: int) -> "PredictionHead":
        return cls(in_dim=in_dim, hidden_dim=cfg.hidden_dim, num_blocks=cfg.num_blocks,
                   dropout=cfg.dropout, out_dim=cfg.out_dim, out_activation=cfg.out_activation,
                   kind=cfg.kind)

    def forward(self, z: Tensor) -> Tensor:
        if self.kind == "resmlp":
            h = F.gelu(self.inp(z))
            for block in self.blocks:
                h = block(h)
            y = self.out(self.norm(h))
        else:
            y = self.net(z)
        if self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation == "softplus":
            y = F.softplus(y)
        return y
