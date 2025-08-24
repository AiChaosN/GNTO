
from __future__ import annotations
import torch
from torch import nn

class CostHead(nn.Module):
    """Regression head predicting runtime (ms) or cost proxy in log-space."""
    def __init__(self, in_dim: int, hidden: int = 128, log_target: bool = True):
        super().__init__()
        self.log_target = log_target
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z: torch.Tensor, extra_feats: torch.Tensor | None = None) -> torch.Tensor:
        if extra_feats is not None:
            z = torch.cat([z, extra_feats], dim=-1)
        out = self.mlp(z).squeeze(-1)
        if self.log_target:
            out = torch.relu(out)  # keep non-negative before exp if you later invert
        return out
