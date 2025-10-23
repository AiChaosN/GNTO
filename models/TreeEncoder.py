from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

import numpy as np

# TreeEncoder_GATMini
class TreeEncoder_GATMini(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads1=8, drop=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden_dim,
                            heads=heads1, dropout=drop, concat=True)
        self.gat2 = GATConv(in_channels=hidden_dim * heads1, out_channels=out_dim,
                            heads=1, dropout=drop, concat=False)
        self.drop = drop

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.gat2(x, edge_index)

        g = global_mean_pool(x, batch)
        return g

class GATTreeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads1=8, drop=0.5, pooling: str = "mean"):
        super().__init__()
        assert pooling in {"mean", "max", "sum"}
        self.drop = drop
        self.pooling = pooling

        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads1, concat=True, dropout=drop)   # [N, hidden*heads]
        self.norm1 = nn.LayerNorm(hidden_dim * heads1)

        self.gat2 = GATConv(hidden_dim * heads1, out_dim, heads=1, concat=False, dropout=drop)  # [N, out_dim]
        # 维度不等，做投影残差（有时能稳提升）
        self.proj_res = nn.Linear(hidden_dim * heads1, out_dim)

    def _pool(self, x, batch):
        if batch is None:
            if self.pooling == "mean": return x.mean(dim=0, keepdim=True)
            if self.pooling == "max":  return x.max(dim=0, keepdim=True).values
            return x.sum(dim=0, keepdim=True)
        else:
            if self.pooling == "mean": return global_mean_pool(x, batch)
            if self.pooling == "max":  return global_max_pool(x, batch)
            return global_add_pool(x, batch)

    def forward(self, x, edge_index, batch=None):
        # layer 1
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        # layer 2 + 残差到 out_dim
        out = self.gat2(x, edge_index)
        out = out + self.proj_res(x)   # residual
        g = self._pool(out, batch)
        return g