"""Plan-level readout: node embeddings -> one vector per plan."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter, softmax

from .config import ReadoutConfig
from .graph_utils import first_node_index


class Readout(nn.Module):
    """``pooling``:

    * ``mean`` / ``max`` / ``sum``  - permutation-invariant pooling
    * ``attention``                 - ``z = sum_i softmax_i(gate(h_i)) h_i``; with
      ``gate_hidden=0`` the gate is ``Linear(d,1)`` (legacy ``GlobalAttention``),
      otherwise ``q^T tanh(W h_i)``
    * ``first``                     - embedding of each graph's first node (the
      plan root for DFS pre-order builders)
    * ``virtual``                   - embedding of the graph's virtual node
    """

    def __init__(self, in_dim: int, pooling: str = "mean", gate_hidden: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.pooling = pooling
        self.gate_nn: Optional[nn.Module] = None
        if pooling == "attention":
            if gate_hidden > 0:
                self.gate_nn = nn.Sequential(nn.Linear(in_dim, gate_hidden), nn.Tanh(),
                                             nn.Linear(gate_hidden, 1, bias=False))
            else:
                self.gate_nn = nn.Linear(in_dim, 1)

    @classmethod
    def from_config(cls, cfg: ReadoutConfig, in_dim: int) -> "Readout":
        return cls(in_dim=in_dim, pooling=cfg.pooling, gate_hidden=cfg.gate_hidden)

    def attention_weights(self, h: Tensor, batch: Optional[Tensor]) -> Tensor:
        """Per-node importance ``alpha_i`` (sums to 1 within each graph)."""
        if self.gate_nn is None:
            raise RuntimeError("attention_weights() requires pooling='attention'")
        gate = self.gate_nn(h).view(-1)
        if batch is None:
            return torch.softmax(gate, dim=0)
        return softmax(gate, batch)

    def forward(self, h: Tensor, batch: Optional[Tensor] = None,
                aux: Optional[Dict[str, Any]] = None, return_weights: bool = False
                ) -> Tensor | Tuple[Tensor, Tensor]:
        weights = None
        if self.pooling == "attention":
            alpha = self.attention_weights(h, batch)
            weights = alpha
            if batch is None:
                z = (alpha.unsqueeze(-1) * h).sum(dim=0, keepdim=True)
            else:
                z = scatter(alpha.unsqueeze(-1) * h, batch, dim=0, reduce="sum")
        elif self.pooling == "virtual":
            if aux is None or "virtual" not in aux:
                raise RuntimeError("pooling='virtual' needs the graph encoder's virtual-node output")
            z = aux["virtual"]
        elif self.pooling == "first":
            z = h[:1] if batch is None else h[first_node_index(batch)]
        elif batch is None:
            if self.pooling == "mean":
                z = h.mean(dim=0, keepdim=True)
            elif self.pooling == "max":
                z = h.max(dim=0, keepdim=True).values
            else:
                z = h.sum(dim=0, keepdim=True)
        else:
            pool = {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}[self.pooling]
            z = pool(h, batch)
        if return_weights:
            return z, weights
        return z
