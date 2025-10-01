from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .DataPreprocessor import PlanNod

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