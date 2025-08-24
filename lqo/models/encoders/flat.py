
from __future__ import annotations
import torch
from torch import nn

class FlatPlanEncoder(nn.Module):
    """Simple MLP encoder that consumes a flat numeric feature vector.
    Input: tensor[B, F]
    Output: tensor[B, D]
    """
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
