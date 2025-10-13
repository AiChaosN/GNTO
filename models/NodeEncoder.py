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

# 谓词编码器 操作符(>, <, =)每个操作符3维,2个列每个8维,1个常量1维,1个flag1维,输出16维
class PredicateEncoder(nn.Module):
    def __init__(self, op_num=6, col_dim=16, const_dim=1, out_dim=16):
        super().__init__()

        op_dir = {
            ">": 0,
            "<": 1,
            "=": 2,
        }

        col_dir = {
            "v_col1": 0,
            "v_col2": 1,
            "const": 2,
        }

        self.op_emb = nn.Embedding(op_num, 3)
        self.col_emb = nn.Embedding(col_dim, 8)

    def forward(self, x):
        for item in x:
            op_vec = self.op_emb(self.op_dir[item[0]])
            col_vec = self.col_emb(self.col_dir[item[1]])
            if item[3]: # is_join
                sec_vec = self.col_emb(self.col_dir[item[2]])
            else:
                sec_vec = item[2]
            x = torch.cat([op_vec, col_vec, sec_vec, item[3]], dim=-1)
            break
        return x

class NodeEncoder_V0(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
        num_node_types: int = 13, node_type_dim: int = 16,
        num_cols: int = 16, col_dim: int = 8,
        num_ops: int = 6, op_dim: int = 3):
        super().__init__()
        
        # Node Type Embedding
        self.node_type_emb = nn.Embedding(num_node_types, node_type_dim)

        # Predicate Encoder
        self.predicate_encoder = PredicateEncoder()
    
    def forward(self, x):
        # Node Type Embedding [16]
        node_type_emb = self.node_type_emb(x["node_type_id"])

        # Num Encoder: Plan Rows, Plan Width [2]
        num_vec = torch.cat([x["plan_rows"], x["plan_width"]], dim=-1)

        # Predicate Encoder [16]
        predicate_emb = self.predicate_encoder(x["predicatge_list"])

        x = torch.cat([node_type_emb, predicate_emb, num_vec], dim=-1)
        assert x.shape[-1] == self.out_dim, f"Output dimension mismatch: {x.shape[-1]} != {self.out_dim}"

        # Output [16 + 16 + 2 = 34]
        return x


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

class NodeEncoder_Enhanced(nn.Module):
    """
    增强版节点编码器，支持更多特征类型和编码方式
    - 支持数值特征标准化
    - 支持类别特征embedding
    - 支持注意力机制
    - 支持残差连接
    """
    def __init__(
        self,
        in_dim: int,
        d_node: int,
        num_node_types: int = 13,  # PostgreSQL查询计划节点类型数量
        use_attention: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_node = d_node
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 隐藏层维度
        hidden_dim = hidden_dim or max(d_node * 2, 64)
        
        # 主要的特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_node)
        )
        
        # 节点类型embedding（如果需要）
        self.node_type_emb = nn.Embedding(num_node_types, d_node // 4)
        
        # 注意力机制（用于特征选择）
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_node,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(d_node)
        
        # 残差连接的投影（如果输入输出维度不匹配）
        if use_residual and in_dim != d_node:
            self.residual_proj = nn.Linear(in_dim, d_node)
        else:
            self.residual_proj = None
            
        self.final_norm = nn.LayerNorm(d_node)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, node_type_ids=None):
        """
        x: [N, in_dim] 节点特征
        node_type_ids: [N] 节点类型ID（可选）
        """
        # 主要特征投影
        h = self.feature_proj(x)  # [N, d_node]
        
        # 添加节点类型embedding
        if node_type_ids is not None:
            type_emb = self.node_type_emb(node_type_ids)  # [N, d_node//4]
            # 扩展到d_node维度
            type_emb = F.pad(type_emb, (0, h.size(-1) - type_emb.size(-1)))
            h = h + type_emb
        
        # 自注意力机制
        if self.use_attention:
            h_att, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            h_att = h_att.squeeze(0)
            h = self.attention_norm(h + h_att)
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            h = h + residual
        
        # 最终标准化和dropout
        h = self.final_norm(h)
        h = self.dropout(h)
        
        return h

class NodeEncoder_Vectorized(nn.Module):
    """
    基于手工特征工程的节点编码器
    参考example中的NodeVectorizer实现
    """
    def __init__(
        self,
        node_types: List[str],
        d_node: int,
        plan_rows_max: float = 2e8,
        use_parallel_feature: bool = True,
        use_cost_features: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_types = node_types
        self.node_type_mapping = {k: i for i, k in enumerate(node_types)}
        self.plan_rows_max = plan_rows_max
        self.use_parallel_feature = use_parallel_feature
        self.use_cost_features = use_cost_features
        
        # 计算输入特征维度
        feature_dim = len(node_types)  # one-hot node type
        if use_parallel_feature:
            feature_dim += 2  # parallel aware (True/False)
        feature_dim += 1  # plan rows (normalized)
        if use_cost_features:
            feature_dim += 2  # startup cost, total cost
        
        # 特征投影层
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_node * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_node * 2),
            nn.Linear(d_node * 2, d_node),
            nn.ReLU(),
            nn.LayerNorm(d_node)
        )
    
    def vectorize_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        """
        将节点字典列表转换为特征向量
        """
        vectors = []
        for node in nodes:
            vector = []
            
            # 1. Node Type (one-hot)
            node_type_vec = [0.0] * len(self.node_types)
            if node["Node Type"] in self.node_type_mapping:
                node_type_vec[self.node_type_mapping[node["Node Type"]]] = 1.0
            vector.extend(node_type_vec)
            
            # 2. Parallel Aware
            if self.use_parallel_feature:
                parallel_vec = [0.0, 0.0]
                parallel_vec[int(node.get("Parallel Aware", False))] = 1.0
                vector.extend(parallel_vec)
            
            # 3. Plan Rows (normalized)
            plan_rows = float(node.get("Plan Rows", 0)) / self.plan_rows_max
            vector.append(plan_rows)
            
            # 4. Cost features (optional)
            if self.use_cost_features:
                startup_cost = float(node.get("Startup Cost", 0)) / 1000.0  # 简单归一化
                total_cost = float(node.get("Total Cost", 0)) / 1000.0
                vector.extend([startup_cost, total_cost])
            
            vectors.append(vector)
        
        return torch.tensor(vectors, dtype=torch.float32)
    
    def forward(self, x):
        """
        x: [N, feature_dim] 或者节点字典列表
        """
        if isinstance(x, list):
            # 如果输入是节点字典列表，先向量化
            x = self.vectorize_nodes(x)
        
        return self.proj(x)

class NodeEncoder_Mixed(nn.Module):
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