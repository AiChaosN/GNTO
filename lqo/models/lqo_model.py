
from __future__ import annotations
import torch
from torch import nn

class LQOModel(nn.Module):
    """Composable LQO model = Encoder + Head (+ optional extra features).
    - encoder: maps features->embedding
    - head: cost or rank head
    - use_pg_cost: if True, expects extra_feats to include PG cost signal
    """
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, extra_feats: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z, extra_feats)
