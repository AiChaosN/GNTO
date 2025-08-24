
from __future__ import annotations
import torch
from torch import nn

class RankHead(nn.Module):
    """Scores a plan for learning-to-rank tasks (higher is better by default)."""
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z: torch.Tensor, extra_feats: torch.Tensor | None = None) -> torch.Tensor:
        if extra_feats is not None:
            z = torch.cat([z, extra_feats], dim=-1)
        return self.mlp(z).squeeze(-1)
