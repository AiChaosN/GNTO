from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default_emb_dim(cardinality: int, max_dim: int = 64) -> int:
    """
    根据类别基数自动计算embedding维度
    常见启发式：min(max(8, round(card**0.25)*8), max_dim)
    """
    if cardinality <= 2:
        return 4
    d = int(round(cardinality ** 0.25)) * 8
    d = max(8, d)
    d = min(max_dim, d)
    return d

class PredicateEncoder1(nn.Module):
    def __init__(self, num_cols=32, col_dim=8, num_ops=6, op_dim=3):
        super().__init__()
        self.col_emb = nn.Embedding(num_cols, col_dim)
        self.op_emb  = nn.Embedding(num_ops,  op_dim)

    def forward(self, data):
        # data: (col1_ids, op_ids, col2_or_num, is_join)  全部展平的一维 [N*3]
        col1, op, col2_or_num, is_join = data
        col1     = col1.long()
        op       = op.long()
        col2_ids = col2_or_num.long()
        num_val  = col2_or_num.float()
        is_join  = is_join.long()

        gate = is_join.unsqueeze(-1).float()  # [T,1]

        col1_emb = self.col_emb(col1)                   # [T, 8]
        op_emb   = self.op_emb(op)                      # [T, 3]
        col2_emb = self.col_emb(col2_ids) * gate        # [T, 8]
        num_feat = num_val.unsqueeze(-1) * (1.0 - gate) # [T, 1]
        is_join_f = gate                                # [T, 1]

        return torch.cat([col1_emb, op_emb, col2_emb, num_feat, is_join_f], dim=-1)
        # 维度: 8 + 3 + 8 + 1 + 1 = 21

class NodeEncoder_V1(nn.Module):
    def __init__(self, num_node_types, num_cols, num_ops,
                 type_dim=16, num_dim=2, out_dim=39):
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, type_dim)
        self.pred_enc = PredicateEncoder1(num_cols=num_cols, num_ops=num_ops,
                                          col_dim=8, op_dim=3)

        # 可选：数值 rows/width 的小投影；不想投影可直接拼接
        self.num_proj = nn.Identity()  # 如需非线性，换成 Linear/MLP

        d_pred = 8 + 3 + 8 + 1 + 1  # 21
        in_dim = type_dim + num_dim + d_pred
        self.proj = nn.Linear(in_dim, out_dim)  # 把拼接后的节点向量映射到 d_node

    def forward(self, x: torch.Tensor):
        """
        x: [N, 15]  ->  [node_type | rows,width | (col1,op,c2n,ij)*3]
        """
        N = x.size(0)

        # 1) 基本字段
        node_type  = x[:, 0].long()      # [N]
        rows_width = x[:, 1:3].float()   # [N,2]

        t_emb = self.type_emb(node_type)         # [N, type_dim]
        n_feat = self.num_proj(rows_width)       # [N, 2]

        # 2) 谓词拆解 -> [N,3,4]
        preds_raw = x[:, 3:].view(N, 3, 4)

        # 展平成一维（按批次所有谓词）→ [N*3]
        col1 = preds_raw[..., 0].round().long().reshape(-1)
        op   = preds_raw[..., 1].round().long().reshape(-1)
        c2n  = preds_raw[..., 2].float().reshape(-1)      # 既当 col2_id 也当 num（已归一化）
        ij   = preds_raw[..., 3].round().long().reshape(-1)

        # 3) 编码所有谓词 → [N*3, 21] → 回到 [N, 3, 21]
        p_all = self.pred_enc((col1, op, c2n, ij))   # [N*3, 21]
        p_all = p_all.view(N, 3, -1)                 # [N, 3, 21]

        # 4) 掩码 + 平均（空槽位全0时不参与）
        presence = (preds_raw.abs().sum(dim=-1) > 0).float()   # [N,3]
        denom = presence.sum(dim=1, keepdim=True).clamp_min(1.0)
        p_vec = (p_all * presence.unsqueeze(-1)).sum(dim=1) / denom   # [N, 21]

        # 5) 拼接并线性映射到 d_node
        node_vec = torch.cat([t_emb, n_feat, p_vec], dim=-1)   # [N, 16+2+21=39]
        out = self.proj(node_vec)                              # [N, out_dim]
        return out

class NodeEncoder_V2(nn.Module):
    def __init__(self, num_node_types, num_cols, num_ops,
                 type_dim=16, num_dim=2, out_dim=39, hidden_dim=64, drop=0.2):
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, type_dim)

        self.pred_enc = PredicateEncoder1(num_cols=num_cols, num_ops=num_ops,
                                          col_dim=8, op_dim=3)

        # 数值特征 MLP：2 → 8 → 2
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, 8),
            nn.ReLU(),
            nn.Linear(8, num_dim)
        )

        d_pred = 8 + 3 + 8 + 1 + 1  # 与V1保持一致
        in_dim = type_dim + num_dim + d_pred  # 16 + 2 + 21 = 39

        # 更深的投影层 (V2 核心变化)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [N, 15]  ->  [node_type | rows,width | (col1,op,c2n,ij)*3]
        """
        N = x.size(0)

        # === (1) 基本字段 ===
        node_type  = x[:, 0].long()
        rows_width = x[:, 1:3].float()

        t_emb = self.type_emb(node_type)          # [N, type_dim]
        n_feat = self.num_proj(rows_width)        # [N, 2]

        # === (2) 谓词拆解 ===
        preds_raw = x[:, 3:].view(N, 3, 4)
        col1 = preds_raw[..., 0].round().long().reshape(-1)
        op   = preds_raw[..., 1].round().long().reshape(-1)
        c2n  = preds_raw[..., 2].float().reshape(-1)
        ij   = preds_raw[..., 3].round().long().reshape(-1)

        # === (3) 谓词编码 ===
        p_all = self.pred_enc((col1, op, c2n, ij))
        p_all = p_all.view(N, 3, -1)

        # === (4) 掩码 + 平均池化 ===
        presence = (preds_raw.abs().sum(dim=-1) > 0).float()
        denom = presence.sum(dim=1, keepdim=True).clamp_min(1.0)
        p_vec = (p_all * presence.unsqueeze(-1)).sum(dim=1) / denom   # [N, 21]

        # === (5) 拼接并映射 ===
        node_vec = torch.cat([t_emb, n_feat, p_vec], dim=-1)   # [N, 39]
        out = self.proj(node_vec)                              # [N, out_dim]
        return out


class NodeEncoder_Mini(nn.Module):
    """
    简单的节点编码器
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
    
    def forward(self, batch, data):
        return
    """
    混合编码器，结合数值特征和类别特征
    参考archive中的MixedNodeEncoder实现
    """
    def __init__(
        self,
        num_in_dim: int,
        cat_cardinalities: List[int],
        d_node: int,
        emb_dims: Optional[List[int]] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.num_in_dim = num_in_dim
        self.cat_cardinalities = cat_cardinalities
        
        # 构建每个类别特征的embedding
        if emb_dims is None:
            emb_dims = [default_emb_dim(card, max_dim=32) for card in cat_cardinalities]
        
        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=dim, padding_idx=0)
            for card, dim in zip(cat_cardinalities, emb_dims)
        ])
        
        # 计算拼接后的总维度
        cat_total_dim = sum(emb_dims)
        total_dim = num_in_dim + cat_total_dim
        
        # MLP投影
        hidden_dim = hidden_dim or max(d_node * 2, 64)
        layers = [nn.Linear(total_dim, hidden_dim), nn.ReLU()]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.extend([
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_node),
            nn.ReLU(),
            nn.LayerNorm(d_node)
        ])
        
        self.proj = nn.Sequential(*layers)
    
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_num: [N, num_in_dim] 数值特征
        x_cat: [N, len(cat_cardinalities)] 类别特征ID
        """
        N = x_cat.size(0) if x_cat is not None else x_num.size(0)
        
        # 处理类别特征
        cat_vecs = []
        if x_cat is not None:
            for j, emb in enumerate(self.embs):
                cat_vecs.append(emb(x_cat[:, j]))
        
        x_cat_emb = torch.cat(cat_vecs, dim=-1) if cat_vecs else torch.zeros((N, 0), device=x_num.device)
        
        # 拼接数值和类别特征
        if x_num.numel() == 0:
            x = x_cat_emb
        elif x_cat_emb.numel() == 0:
            x = x_num
        else:
            x = torch.cat([x_num, x_cat_emb], dim=-1)
        
        return self.proj(x)