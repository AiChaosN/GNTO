from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

import numpy as np

# 将PlanNode转换为Graph
class TreeToGraphConverter:
    def __init__(self, encoder=None, bidirectional=True):
        self.encoder = encoder
        self.bidirectional = bidirectional

    def tree_to_graph(self, root):
        nodes, edges = [], []

        def dfs(node, parent_idx):
            idx = len(nodes)
            nodes.append(node)
            if parent_idx is not None:
                edges.append((parent_idx, idx))
                if self.bidirectional:
                    edges.append((idx, parent_idx))
            for ch in node.children:
                dfs(ch, idx)

        dfs(root, None)

        # 1) 正确地先取 vec
        feats_list = []
        for n in nodes:
            if self.encoder is not None:
                vec = self.encoder.encode_node(n)
            else:
                if n.node_vector is None:
                    raise ValueError("某些节点缺少 node_vector；请先填充或提供 encoder。")
                vec = n.node_vector

            # 2) 统一为 numpy.float32（你的 _process_tree 里按 numpy 处理）
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().cpu().numpy()
            else:
                vec = np.asarray(vec)
            feats_list.append(vec.astype(np.float32))

        # 3) edge_index
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)


        feats_list = torch.stack(
            [torch.as_tensor(f, dtype=torch.float32) for f in feats_list],
            dim=0
        )
        return edge_index, feats_list



# 使用GAT模型进行编码
class GATTreeEncoder(nn.Module):
    """
    图级表示的 GAT 编码器：
    - 输入: x [N, F_in], edge_index [2, E]
    - 输出: graph embedding [output_dim]
    - 支持多层、多头、dropout、三种池化: mean/max/sum
    """
    def __init__(self,
                 input_dim:   int,
                 hidden_dim:  int,
                 output_dim:  int,
                 num_layers:  int = 4,
                 num_heads:   int = 4,
                 dropout:     float = 0.1,
                 pooling:     str  = "mean"):
        super().__init__()
        assert num_layers >= 1, "num_layers 至少为 1"
        assert pooling in {"mean", "max", "sum"}, "pooling 需为 mean/max/sum"

        self.pooling = pooling
        self.dropout = dropout

        layers = nn.ModuleList()
        in_dim = input_dim
        out_dim_each_head = hidden_dim

        # 第1层
        layers.append(GATConv(in_dim, out_dim_each_head, heads=num_heads, concat=True, dropout=dropout))
        in_dim = out_dim_each_head * num_heads  # concat 后维度

        # 中间层
        for _ in range(num_layers - 2):
            layers.append(GATConv(in_dim, out_dim_each_head, heads=num_heads, concat=True, dropout=dropout))
            in_dim = out_dim_each_head * num_heads

        # 最后一层
        if num_layers >= 2:
            layers.append(GATConv(in_dim, out_dim_each_head, heads=num_heads, concat=True, dropout=dropout))
            in_dim = out_dim_each_head * num_heads

        self.convs = layers
        self.norms = nn.ModuleList([nn.LayerNorm(in_dim if i == len(layers)-1 else layers[i+1].out_channels * num_heads
                                                 if hasattr(layers[i+1], 'out_channels') else in_dim)
                                    for i in range(len(layers))]) if len(layers) > 0 else nn.ModuleList()

        # 图级输出投影
        self.output_proj = nn.Linear(in_dim, output_dim)

    def _pool_batch(self, h, batch):
        if self.pooling == "mean":
            return global_mean_pool(h, batch)  # [B, H*heads]
        if self.pooling == "max":
            return global_max_pool(h, batch)
        return global_add_pool(h, batch)       # sum

    def _pool_single(self, h):
        # 单图时对 N 维做聚合；返回 [1, H*heads]，保持与 batch 版对齐
        if self.pooling == "mean":
            return h.mean(dim=0, keepdim=True)
        if self.pooling == "max":
            return h.max(dim=0, keepdim=True).values
        return h.sum(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [N, F_in]
        edge_index: [2, E]
        batch: [N] 或 None
        return: [B, output_dim]    （若单图，B=1）
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)      # [N, hidden*heads]
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            g = self._pool_single(h)     # [1, hidden*heads]
        else:
            g = self._pool_batch(h, batch)  # [B, hidden*heads]

        out = self.output_proj(g)         # [B, output_dim]
        return out