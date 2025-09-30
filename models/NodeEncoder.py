from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import re
from .DataPreprocessor import PlanNode
from .Utils import _safe_float


# class NodeEncoder(nn.Module):
    
#     def __init__(self, 
#                  operator_embedding_dim: int = 32,
#                  stats_hidden_dim: int = 16,
#                  predicate_dim: int = 8,
#                  output_dim: int = 64) -> None:

#         super().__init__()
        
#         # 配置参数
#         self.operator_embedding_dim = operator_embedding_dim
#         self.stats_hidden_dim = stats_hidden_dim
#         self.predicate_dim = predicate_dim
#         self.output_dim = output_dim
        
#         # 算子类型词汇表
#         self.node_type_vocab: Dict[str, int] = {}
        
#         # 核心统计特征键
#         self.stats_keys = ['Plan Rows', 'Plan Width', 'Startup Cost', 'Total Cost']
        
#         # 谓词特征键
#         self.predicate_keys = ['Filter', 'Index Cond', 'Hash Cond', 'Merge Cond', 'Join Filter']
        
#         # 延迟初始化的组件 (在第一次forward时初始化)
#         self.operator_embedding: Optional[nn.Embedding] = None
#         self.stats_mlp: Optional[nn.Sequential] = None
#         self.output_projection: Optional[nn.Linear] = None
        
#         self._initialized = False
    
#     def _ensure_initialized(self, node):

#         if self._initialized:
#             # 检查是否需要扩展embedding层
#             self._update_operator_vocab(node)
#             current_vocab_size = len(self.node_type_vocab)
#             if current_vocab_size > self.operator_embedding.num_embeddings:
#                 # 需要扩展embedding层
#                 old_embedding = self.operator_embedding
#                 self.operator_embedding = nn.Embedding(current_vocab_size, self.operator_embedding_dim)
#                 # 复制旧的权重
#                 with torch.no_grad():
#                     self.operator_embedding.weight[:old_embedding.num_embeddings] = old_embedding.weight
#             return
            
#         # 1. 初始化算子embedding
#         self._update_operator_vocab(node)
#         vocab_size = len(self.node_type_vocab)
#         self.operator_embedding = nn.Embedding(vocab_size, self.operator_embedding_dim)
        
#         # 2. 初始化统计特征MLP
#         stats_input_dim = len(self.stats_keys)
#         self.stats_mlp = nn.Sequential(
#             nn.Linear(stats_input_dim, self.stats_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.stats_hidden_dim, self.stats_hidden_dim)
#         )
        
#         # 3. 计算concat后的总维度
#         total_dim = self.operator_embedding_dim + self.stats_hidden_dim + self.predicate_dim
        
#         # 4. 初始化输出投影层
#         self.output_projection = nn.Linear(total_dim, self.output_dim)
        
#         self._initialized = True
    
#     def _update_operator_vocab(self, node):

#         node_type = getattr(node, "node_type", "Unknown")
#         if node_type not in self.node_type_vocab:
#             self.node_type_vocab[node_type] = len(self.node_type_vocab)
    
#     def _encode_operator(self, node) -> torch.Tensor:
#         """编码算子类型 → Embedding"""
#         self._update_operator_vocab(node)
#         node_type = getattr(node, "node_type", "Unknown")
#         idx = self.node_type_vocab[node_type]
#         idx_tensor = torch.tensor([idx], dtype=torch.long)
#         return self.operator_embedding(idx_tensor).squeeze(0)  # [embedding_dim]
    
#     def _encode_stats(self, node) -> torch.Tensor:
#         """编码数据统计 → MLP"""
#         extra_info = getattr(node, 'extra_info', {})
#         stats_values = []
        
#         for key in self.stats_keys:
#             value = extra_info.get(key, 0.0)
#             if isinstance(value, (int, float)):
#                 stats_values.append(float(value))
#             else:
#                 try:
#                     stats_values.append(float(str(value).replace(',', '')))
#                 except:
#                     stats_values.append(0.0)
        
#         # 简单的log标准化
#         stats_tensor = torch.tensor(stats_values, dtype=torch.float32)
#         stats_tensor = torch.log1p(stats_tensor)  # log(1 + x)
        
#         # 通过MLP
#         return self.stats_mlp(stats_tensor)  # [stats_hidden_dim]
    
#     def _encode_predicate(self, node) -> torch.Tensor:
#         """编码谓词信息 → 简单特征"""
#         extra_info = getattr(node, 'extra_info', {})
        
#         # 收集所有谓词信息
#         predicates = []
#         for key in self.predicate_keys:
#             if key in extra_info and extra_info[key]:
#                 predicates.append(str(extra_info[key]))
        
#         if not predicates:
#             return torch.zeros(self.predicate_dim, dtype=torch.float32)
        
#         # 简单的复杂度特征
#         all_predicates = ' '.join(predicates).lower()
        
#         features = []
        
#         # 1. 谓词数量 (归一化)
#         features.append(min(len(predicates) / 5.0, 1.0))
        
#         # 2. 是否有范围过滤
#         range_patterns = ['>', '<', '>=', '<=', 'between']
#         features.append(float(any(pattern in all_predicates for pattern in range_patterns)))
        
#         # 3. 是否包含子查询
#         subquery_patterns = ['exists', 'in (select', 'subplan']
#         features.append(float(any(pattern in all_predicates for pattern in subquery_patterns)))
        
#         # 4. 是否有函数调用
#         features.append(float('(' in all_predicates))
        
#         # 5. 是否有LIKE模式匹配
#         features.append(float('like' in all_predicates or '%' in all_predicates))
        
#         # 6. 连接条件数量 (归一化)
#         join_count = all_predicates.count('=')
#         features.append(min(join_count / 3.0, 1.0))
        
#         # 7-8. 填充到predicate_dim维度
#         while len(features) < self.predicate_dim:
#             features.append(0.0)
        
#         return torch.tensor(features[:self.predicate_dim], dtype=torch.float32)
    
#     # 编码函数
#     def forward(self, node) -> torch.Tensor:
#         """分块编码 + Concat
        
#         Parameters
#         ----------
#         node: PlanNode
#             查询计划节点
            
#         Returns
#         -------
#         torch.Tensor
#             节点编码向量 [output_dim]
#         """
#         # 确保初始化
#         self._ensure_initialized(node)
        
#         # 1. 算子类型编码
#         operator_vec = self._encode_operator(node)  # [operator_embedding_dim]
        
#         # 2. 统计特征编码
#         stats_vec = self._encode_stats(node)  # [stats_hidden_dim]
        
#         # 3. 谓词特征编码
#         predicate_vec = self._encode_predicate(node)  # [predicate_dim]
        
#         # 4. Concat所有特征
#         combined = torch.cat([operator_vec, stats_vec, predicate_vec], dim=0)
        
#         # 5. 输出投影
#         output = self.output_projection(combined)  # [output_dim]
        
#         return output
    
#     def encode_node(self, node) -> torch.Tensor:
#         """编码单个节点并存储到node.node_vector
        
#         Parameters
#         ----------
#         node: PlanNode
#             查询计划节点
            
#         Returns
#         -------
#         torch.Tensor
#             节点编码向量
#         """
#         vector = self.forward(node)
#         node.node_vector = vector
#         return vector
    
#     def encode_nodes(self, nodes: Iterable) -> List[torch.Tensor]:
#         """编码多个节点
        
#         Parameters
#         ----------
#         nodes: Iterable
#             查询计划节点列表
            
#         Returns
#         -------
#         List[torch.Tensor]
#             节点编码向量列表
#         """
#         return [self.encode_node(node) for node in nodes]
    
#     @staticmethod
#     def collect_nodes(root: PlanNode, method: str = "dfs") -> List[PlanNode]:
#         """
#         遍历 PlanNode 树，收集所有节点为列表
        
#         Parameters
#         ----------
#         root : PlanNode
#             根节点
#         method : str
#             遍历方式，可选 "dfs" (深度优先) 或 "bfs" (广度优先)
        
#         Returns
#         -------
#         List[PlanNode]
#             树中所有节点的列表
#         """
#         nodes = []
        
#         if method == "dfs":
#             # 递归深度优先
#             def dfs(node: PlanNode):
#                 nodes.append(node)
#                 for child in node.children:
#                     dfs(child)
#             dfs(root)
        
#         elif method == "bfs":
#             # 队列广度优先
#             queue = [root]
#             while queue:
#                 node = queue.pop(0)
#                 nodes.append(node)
#                 queue.extend(node.children)
        
#         else:
#             raise ValueError("method 必须是 'dfs' 或 'bfs'")
        
#         return nodes

class NodeEncoder(nn.Module):
    """
    固定词表版本 + 列统计增强：
    - node_type_vocab: 训练前离线构建
    - stats_scalers:   训练前离线构建（针对 Plan Rows / Cost 等）
    - col_vocab / col_scalers / col_stats_map: 来自 CSV（name/min/max/cardinality/num_unique_values）
    """

    def __init__(self, 
                 # 算子类型词汇表
                 node_type_vocab: Dict[str, int],
                 # 统计特征缩放器
                 stats_scalers: Dict[str, Tuple[float, float]],
                 # 列统计信息
                 col_vocab: Dict[str, int],
                 col_scalers: Dict[str, Tuple[float, float]],
                 col_stats_map: Dict[str, Dict[str, float]],
                 # embedding维度
                 operator_embedding_dim: int = 32,
                 stats_hidden_dim: int = 16,
                 predicate_dim: int = 8,
                 column_embedding_dim: int = 16,
                 column_stat_dim: int = 8,
                 output_dim: int = 64) -> None:
        super().__init__()

        # 固定词表/缩放器
        self.node_type_vocab = dict(node_type_vocab)
        self.unk_idx = self.node_type_vocab.get("<UNK>", 0)
        self.stats_keys = ['Plan Rows', 'Plan Width', 'Startup Cost', 'Total Cost']
        self.stats_scalers = stats_scalers

        # 列统计（CSV）
        self.col_vocab = dict(col_vocab)  # {"<UNK_COL>":0, "t.id":1, ...}
        self.col_unk = self.col_vocab.get("<UNK_COL>", 0)
        self.col_scalers = col_scalers    # {"min":(mu,std), "max":..., "cardinality":..., "num_unique_values":...}
        self.col_stats_map = col_stats_map# {"t.id": {"min":..., "max":..., ...}, ...}

        # 尺寸
        self.operator_embedding_dim = operator_embedding_dim
        self.stats_hidden_dim = stats_hidden_dim
        self.predicate_dim = predicate_dim
        self.column_embedding_dim = column_embedding_dim
        self.column_stat_dim = column_stat_dim
        self.output_dim = output_dim

        # 模块
        vocab_size = max(self.node_type_vocab.values()) + 1 if self.node_type_vocab else 1
        self.operator_embedding = nn.Embedding(vocab_size, self.operator_embedding_dim)

        self.stats_mlp = nn.Sequential(
            nn.Linear(len(self.stats_keys), self.stats_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.stats_hidden_dim, self.stats_hidden_dim)
        )

        # 列：embedding + 统计聚合映射
        self.column_embedding = nn.Embedding(max(self.col_vocab.values()) + 1, self.column_embedding_dim)
        self.column_stat_proj = nn.Linear(4, self.column_stat_dim)  # 4 个字段: min/max/cardinality/num_unique_values

        # 输出投影
        total_dim = (self.operator_embedding_dim 
                     + self.stats_hidden_dim 
                     + self.predicate_dim
                     + self.column_embedding_dim 
                     + self.column_stat_dim)
        self.output_projection = nn.Linear(total_dim, self.output_dim)

    # --------- 子编码：算子 ---------
    def _encode_operator(self, node: PlanNode) -> torch.Tensor:
        t = getattr(node, "node_type", "Unknown")
        if not isinstance(t, str): t = str(t)
        idx = self.node_type_vocab.get(t, self.unk_idx)
        idx = torch.tensor([idx], dtype=torch.long, device=self.operator_embedding.weight.device)
        return self.operator_embedding(idx).squeeze(0)

    # --------- 子编码：计划数值（log1p 标准化后 MLP）---------
    def _encode_stats(self, node: PlanNode) -> torch.Tensor:
        extra = getattr(node, 'extra_info', {}) or getattr(node, 'info', {}) or {}
        vals = []
        for k in self.stats_keys:
            raw = _safe_float(extra.get(k, 0.0))
            x = np.log1p(max(0.0, raw))
            mu, std = self.stats_scalers.get(k, (0.0, 1.0))
            vals.append((x - mu) / std)
        t = torch.tensor(vals, dtype=torch.float32, device=self.output_projection.weight.device)
        return self.stats_mlp(t)

    # --------- 子编码：谓词粗特征（沿用你之前的 8 维规则特征）---------
    def _encode_predicate_flags(self, node: PlanNode) -> torch.Tensor:
        extra = getattr(node, 'extra_info', {}) or getattr(node, 'info', {}) or {}
        predicate_keys = ['Filter', 'Index Cond', 'Hash Cond', 'Merge Cond', 'Join Filter']
        preds = []
        for k in predicate_keys:
            v = extra.get(k)
            if v: preds.append(str(v))
        if not preds:
            return torch.zeros(self.predicate_dim, dtype=torch.float32, device=self.output_projection.weight.device)

        s = ' '.join(preds).lower()
        feats = []
        feats.append(min(len(preds)/5.0, 1.0))                                           # 条数
        feats.append(float(any(tok in s for tok in ['>', '<', '>=', '<=', 'between']))) # 范围
        feats.append(float(any(tok in s for tok in ['exists', 'in (select', 'subplan'])))# 子查询
        feats.append(float('(' in s))                                                    # 函数
        feats.append(float('like' in s or '%' in s))                                     # like
        feats.append(min(s.count('=')/3.0, 1.0))                                         # 等值计数
        while len(feats) < self.predicate_dim: feats.append(0.0)
        return torch.tensor(feats[:self.predicate_dim], dtype=torch.float32, device=self.output_projection.weight.device)

    # --------- 子编码：列嵌入 + 列统计聚合（来自 CSV）---------
    def _encode_columns(self, node: PlanNode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          col_emb_vec:   [column_embedding_dim]  —— 多列 embedding 平均
          col_stats_vec: [column_stat_dim]      —— 多列统计(min/max/card/uniq)经缩放后平均，再线性映射
        """
        cols = extract_columns_from_node(node)  # ["t.id", "ci.person_id", ...]
        device = self.output_projection.weight.device

        if not cols:
            return (torch.zeros(self.column_embedding_dim, dtype=torch.float32, device=device),
                    torch.zeros(self.column_stat_dim,   dtype=torch.float32, device=device))

        # 1) 列嵌入平均
        ids = [self.col_vocab.get(c, self.col_unk) for c in cols]
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        col_emb = self.column_embedding(ids_t).mean(dim=0)  # [d_col]

        # 2) 列统计：对每列取 [min,max,cardinality,num_unique] -> log1p 标准化 -> 平均
        stats_mat = []
        for c in cols:
            st = self.col_stats_map.get(c)
            if st is None:
                # 用 0 作为缺失的原值 -> log1p(0)=0 -> 标准化后也是常数
                v = [0.0, 0.0, 0.0, 0.0]
            else:
                v = [
                    _safe_float(st.get("min", 0.0)),
                    _safe_float(st.get("max", 0.0)),
                    _safe_float(st.get("cardinality", 0.0)),
                    _safe_float(st.get("num_unique_values", 0.0)),
                ]
            # log1p & z-score
            v = [np.log1p(max(0.0, x)) for x in v]
            mu_min, std_min = self.col_scalers["min"]
            mu_max, std_max = self.col_scalers["max"]
            mu_c, std_c = self.col_scalers["cardinality"]
            mu_u, std_u = self.col_scalers["num_unique_values"]
            v = [
                (v[0] - mu_min)/std_min,
                (v[1] - mu_max)/std_max,
                (v[2] - mu_c)/std_c,
                (v[3] - mu_u)/std_u,
            ]
            stats_mat.append(v)

        stats_mat = torch.tensor(stats_mat, dtype=torch.float32, device=device)  # [K,4]
        stats_avg = stats_mat.mean(dim=0, keepdim=False)                          # [4]
        col_stats_vec = self.column_stat_proj(stats_avg)                          # [column_stat_dim]
        return col_emb, col_stats_vec

    # --------- 前向 ---------
    def forward(self, node: PlanNode) -> torch.Tensor:
        op_vec   = self._encode_operator(node)
        stats    = self._encode_stats(node)
        pred_f   = self._encode_predicate_flags(node)
        col_emb, col_stats = self._encode_columns(node)

        z = torch.cat([op_vec, stats, pred_f, col_emb, col_stats], dim=0)
        return self.output_projection(z)

    # 兼容方法
    def encode_node(self, node: PlanNode) -> torch.Tensor:
        vec = self.forward(node)
        node.node_vector = vec
        return vec

    def encode_nodes(self, nodes: List[PlanNode]) -> List[torch.Tensor]:
        return [self.encode_node(n) for n in nodes]


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