from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool


##########
# NodeEncoder #
##########
class NodeEncoder_Mini(nn.Module):
    """
    输入: data.x 形状 [N, F_in]
    输出: node_embs [N, d_node]
    """
    def __init__(self, in_dim: int, d_node: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_node),
            nn.ReLU(),
            nn.LayerNorm(d_node),
        )
    def forward(self, x):
        return self.proj(x)

class NodeEncoder(nn.Module):
    def __init__(
        self,
        num_cols: List[str],
        cat_cols: List[str],
        cat_cardinalities: Dict[str, int],   # 每个类别列的词表大小（含UNK）
        num_mean: Dict[str, float],          # 数值列均值（训练集统计）
        num_std: Dict[str, float],           # 数值列标准差（训练集统计，避免0，用>=1e-6）
        emb_dims: Optional[Dict[str, int]] = None,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cols = num_cols
        self.cat_cols = cat_cols

        # 数值列标准化参数注册为 buffer（不参与训练，但会跟随 .to(device)/保存）
        mu = [float(num_mean[c]) for c in num_cols]
        sd = [max(float(num_std[c]), 1e-6) for c in num_cols]
        self.register_buffer("num_mean", torch.tensor(mu).view(1, -1))
        self.register_buffer("num_std",  torch.tensor(sd).view(1, -1))

        self.use_num = len(num_cols) > 0
        self.use_cat = len(cat_cols) > 0

        # 类别 embedding
        if self.use_cat:
            self.embs = nn.ModuleDict()
            self.emb_out_dims = {}
            for c in cat_cols:
                card = int(cat_cardinalities[c])
                d = emb_dims[c] if (emb_dims and c in emb_dims) else default_emb_dim(card)
                self.embs[c] = nn.Embedding(num_embeddings=card, embedding_dim=d, padding_idx=0)
                self.emb_out_dims[c] = d

        # 数值通道的线性升维（可选）
        self.num_proj: Optional[nn.Linear] = None
        if self.use_num:
            # 不升维就直接拼接；也可以把数值过一层线性/BN后再拼
            self.num_in_dim = len(num_cols)
            self.num_proj = nn.Identity()

        # 拼接后的总维度
        total_dim = 0
        if self.use_num:
            total_dim += self.num_in_dim
        if self.use_cat:
            total_dim += sum(self.emb_out_dims.values())

        # 可选 BN/Dropout
        self.bn = nn.BatchNorm1d(total_dim) if (use_batchnorm and total_dim > 1) else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_dim = total_dim

    def forward(self, batch_num: Optional[torch.Tensor], batch_cat: Optional[Dict[str, torch.Tensor]]):
        """
        batch_num: [B, len(num_cols)] 的 float 张量（可为 None）
        batch_cat: dict[col] -> [B] 的 Long 张量（每列是类别 id，0 作为 UNK/PAD）
        """
        feats = []
        if self.use_num and batch_num is not None:
            x_num = (batch_num - self.num_mean) / self.num_std  # 标准化
            x_num = self.num_proj(x_num)                        # Identity or Linear
            feats.append(x_num)

        if self.use_cat and batch_cat is not None:
            emb_list = []
            for c in self.cat_cols:
                ids = batch_cat[c]  # [B]
                emb = self.embs[c](ids)  # [B, d_c]
                emb_list.append(emb)
            x_cat = torch.cat(emb_list, dim=-1) if len(emb_list) > 1 else emb_list[0]
            feats.append(x_cat)

        x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=-1)  # [B, out_dim]
        # BN 要求 B>1；单样本推理时可自动跳过或切 eval()
        x = self.bn(x) if x.shape[0] > 1 else x
        x = self.dropout(x)
        return x


##########
# TreeEncoder #
##########
# TreeEncoder_GATMini
class TreeEncoder_GATMini(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads1=8, drop=0.6):
        super().__init__()
        # 第一层:
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden_dim,
                            heads=heads1, dropout=drop, concat=True)
        # 第二层:
        self.gat2 = GATConv(in_channels=hidden_dim*heads1, out_channels=out_dim,
                            heads=1, dropout=drop, concat=False)
        self.drop = drop

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.gat2(x, edge_index)
        return x

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

##########
# PredictionHead #
##########
# FNN 模型 64 -> 128 -> 64 -> 1
class PredictionHead_FNNMini(nn.Module):
    def __init__(self, in_dim=64, hidden_dims=(128, 64), out_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, plan_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(plan_emb)

# FNN 模型 64 -> 128 -> 64 -> 1
class PredictionHead(nn.Module):
    def __init__(self, in_dim=64, hidden_dims=(128, 64), out_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, plan_emb: torch.Tensor) -> torch.Tensor:
        if plan_emb.dim() == 1:
            plan_emb = plan_emb.unsqueeze(0)
        out = self.mlp(plan_emb)       # [B, 1]
        return out.squeeze(-1)         # [B] 或标量

