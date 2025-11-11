from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import (
    GATConv, 
    GATv2Conv, 
    global_mean_pool, 
    global_max_pool, 
    global_add_pool
)

try:
    from torch_geometric.utils import dropout_edge
except Exception:
    dropout_edge = None

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

class GATTreeEncoder_V2(nn.Module):
    """
    更深的三层GAT，支持：
    - 可选边失活 edge_drop（训练期随机丢边）
    - 每层 LayerNorm + 残差（维度不等时用线性投影）
    - JK 聚合（'last' | 'sum' | 'max'），默认 'last'
    - 池化：'mean' | 'max' | 'sum'
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads1: int = 8,
        heads2: int = 4,
        drop: float = 0.5,
        pooling: str = "mean",
        jk: str = "last",
        edge_drop: float = 0.0,   # 丢边概率（仅训练期且安装了PyG utils时生效）
    ):
        super().__init__()
        assert pooling in {"mean", "max", "sum"}
        assert jk in {"last", "sum", "max"}

        self.drop = drop
        self.pooling = pooling
        self.jk = jk
        self.edge_drop = edge_drop

        # ---- Layer 1 ----
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads1, concat=True, dropout=drop)
        dim1 = hidden_dim * heads1
        self.norm1 = nn.LayerNorm(dim1)
        # 输入到第1层输出的维度可能不等，做残差投影
        self.proj_res1 = nn.Linear(in_dim, dim1) if in_dim != dim1 else nn.Identity()

        # ---- Layer 2 ----
        self.gat2 = GATConv(dim1, hidden_dim, heads=heads2, concat=True, dropout=drop)
        dim2 = hidden_dim * heads2
        self.norm2 = nn.LayerNorm(dim2)
        self.proj_res2 = nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity()

        # ---- Layer 3 (to out_dim) ----
        self.gat3 = GATConv(dim2, out_dim, heads=1, concat=False, dropout=drop)
        self.norm3 = nn.LayerNorm(out_dim)
        self.proj_res3 = nn.Linear(dim2, out_dim)  # 到 out_dim 的残差投影

    # 与你V1一致的池化接口
    def _pool(self, x, batch):
        if batch is None:
            if self.pooling == "mean": return x.mean(dim=0, keepdim=True)
            if self.pooling == "max":  return x.max(dim=0, keepdim=True).values
            return x.sum(dim=0, keepdim=True)
        else:
            if self.pooling == "mean": return global_mean_pool(x, batch)
            if self.pooling == "max":  return global_max_pool(x, batch)
            return global_add_pool(x, batch)

    def _maybe_edge_drop(self, edge_index):
        if self.training and self.edge_drop > 0.0 and dropout_edge is not None:
            ei, _ = dropout_edge(edge_index, p=self.edge_drop, force_undirected=False)
            return ei
        return edge_index

    def forward(self, x, edge_index, batch=None):
        # 可选：训练时按概率随机丢边
        ei = self._maybe_edge_drop(edge_index)

        # ----- Layer 1 -----
        h1 = self.gat1(x, ei)                 # [N, dim1]
        h1 = self.norm1(h1)
        h1 = F.elu(h1 + self.proj_res1(x))    # 残差
        h1 = F.dropout(h1, p=self.drop, training=self.training)

        # ----- Layer 2 -----
        h2 = self.gat2(h1, ei)                # [N, dim2]
        h2 = self.norm2(h2)
        h2 = F.elu(h2 + self.proj_res2(h1))   # 残差
        h2 = F.dropout(h2, p=self.drop, training=self.training)

        # ----- Layer 3 -----
        h3 = self.gat3(h2, ei)                # [N, out_dim]
        h3 = self.norm3(h3)
        h3 = F.elu(h3 + self.proj_res3(h2))   # 残差
        h3 = F.dropout(h3, p=self.drop, training=self.training)

        # ----- JK 聚合（节点级）-----
        if self.jk == "last":
            h = h3
        elif self.jk == "sum":
            # 将不同层的表示投影到相同维度再相加（h1/h2需要投到 out_dim）
            if not hasattr(self, "_jk_p1"):
                self._jk_p1 = nn.Linear(h1.size(-1), h3.size(-1)).to(h1.device)
                self._jk_p2 = nn.Linear(h2.size(-1), h3.size(-1)).to(h2.device)
            h = self._jk_p1(h1) + self._jk_p2(h2) + h3
        else:  # "max"
            if not hasattr(self, "_jk_p1"):
                self._jk_p1 = nn.Linear(h1.size(-1), h3.size(-1)).to(h1.device)
                self._jk_p2 = nn.Linear(h2.size(-1), h3.size(-1)).to(h2.device)
            h = torch.maximum(torch.maximum(self._jk_p1(h1), self._jk_p2(h2)), h3)

        # 图级池化
        g = self._pool(h, batch)  # [B, out_dim]
        return g

class GATv2TreeEncoder_V3(nn.Module):
    """
    三层 GATv2:
    - 每层：GATv2Conv + LayerNorm + 残差（维度不等时线性投影）
    - 支持 edge_drop（训练期随机丢边）
    - JK 聚合: 'last' | 'sum' | 'max'
    - 图池化: 'mean' | 'max' | 'sum'
    兼容你的 V2 用法，便于一行替换。
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads1: int = 8,
        heads2: int = 4,
        drop: float = 0.5,
        pooling: str = "mean",
        jk: str = "last",
        edge_drop: float = 0.0,
    ):
        super().__init__()
        assert pooling in {"mean", "max", "sum"}
        assert jk in {"last", "sum", "max"}

        self.drop = drop
        self.pooling = pooling
        self.jk = jk
        self.edge_drop = edge_drop

        # ---- Layer 1 ----
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads1, dropout=drop, concat=True)
        dim1 = hidden_dim * heads1
        self.norm1 = nn.LayerNorm(dim1)
        self.proj_res1 = nn.Linear(in_dim, dim1) if in_dim != dim1 else nn.Identity()

        # ---- Layer 2 ----
        self.gat2 = GATv2Conv(dim1, hidden_dim, heads=heads2, dropout=drop, concat=True)
        dim2 = hidden_dim * heads2
        self.norm2 = nn.LayerNorm(dim2)
        self.proj_res2 = nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity()

        # ---- Layer 3 (to out_dim) ----
        self.gat3 = GATv2Conv(dim2, out_dim, heads=1, dropout=drop, concat=False)
        self.norm3 = nn.LayerNorm(out_dim)
        self.proj_res3 = nn.Linear(dim2, out_dim)  # 到 out_dim 的残差投影

        # ---- JK 需要的投影（将 h1/h2 投到 out_dim）----
        if jk in {"sum", "max"}:
            self.jk_p1 = nn.Linear(dim1, out_dim)
            self.jk_p2 = nn.Linear(dim2, out_dim)
        else:
            self.jk_p1 = None
            self.jk_p2 = None

    def _pool(self, x, batch):
        if batch is None:
            if self.pooling == "mean": return x.mean(dim=0, keepdim=True)
            if self.pooling == "max":  return x.max(dim=0, keepdim=True).values
            return x.sum(dim=0, keepdim=True)
        if self.pooling == "mean": return global_mean_pool(x, batch)
        if self.pooling == "max":  return global_max_pool(x, batch)
        return global_add_pool(x, batch)

    def _maybe_edge_drop(self, edge_index):
        if self.training and self.edge_drop > 0.0 and dropout_edge is not None:
            ei, _ = dropout_edge(edge_index, p=self.edge_drop, force_undirected=False)
            return ei
        return edge_index

    def forward(self, x, edge_index, batch=None):
        ei = self._maybe_edge_drop(edge_index)

        # ----- Layer 1 -----
        h1 = self.gat1(x, ei)                 # [N, dim1]
        h1 = self.norm1(h1)
        h1 = F.elu(h1 + self.proj_res1(x))
        h1 = F.dropout(h1, p=self.drop, training=self.training)

        # ----- Layer 2 -----
        h2 = self.gat2(h1, ei)                # [N, dim2]
        h2 = self.norm2(h2)
        h2 = F.elu(h2 + self.proj_res2(h1))
        h2 = F.dropout(h2, p=self.drop, training=self.training)

        # ----- Layer 3 -----
        h3 = self.gat3(h2, ei)                # [N, out_dim]
        h3 = self.norm3(h3)
        h3 = F.elu(h3 + self.proj_res3(h2))
        h3 = F.dropout(h3, p=self.drop, training=self.training)

        # ----- JK 聚合 -----
        if self.jk == "last":
            h = h3
        elif self.jk == "sum":
            h = self.jk_p1(h1) + self.jk_p2(h2) + h3
        else:  # "max"
            h = torch.maximum(torch.maximum(self.jk_p1(h1), self.jk_p2(h2)), h3)

        # 图级池化
        g = self._pool(h, batch)  # [B, out_dim]
        return g

