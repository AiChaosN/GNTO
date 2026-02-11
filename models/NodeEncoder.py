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

# 新版本,适配新数据集
class PredicateEncoderV3(nn.Module):
    """
    将单个节点的一组谓词字段编码为一个向量：
      Inputs (from data, dtype=long/float):
        - op_type_id        [N]
        - lhs_col_id        [N]
        - rhs_is_col        [N] {0/1}
        - rhs_col_id        [N]
        - rhs_lit_is_num    [N] {0/1}
        - rhs_lit_val       [N] (float, 已做log/标准化)
        - rhs_lit_bucket    [N] (哈希桶id；无字符串时可为0/UNK)
    输出一个 [N, d_pred] 的向量
    """
    def __init__(
        self,
        num_cols: int,
        num_ops: int,
        num_str_buckets: int,
        col_dim: int = 16,
        op_dim: int = 8,
        str_dim: int = 8,
        num_val_dim: int = 8,
        out_dim: int = 32,
        drop: float = 0.0,
    ):
        super().__init__()
        self.col_emb = nn.Embedding(num_cols, col_dim)
        self.op_emb  = nn.Embedding(num_ops,  op_dim)
        self.str_emb = nn.Embedding(num_str_buckets, str_dim)

        # 把数值常量 (rhs_lit_val) 提升到向量
        self.num_proj = nn.Sequential(
            nn.Linear(1, num_val_dim),
            nn.ReLU(),
            nn.Linear(num_val_dim, num_val_dim)
        )

        # 融合 op/lhs/rhs 三部分后再压缩
        in_dim = op_dim + col_dim + max(col_dim, num_val_dim + str_dim)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim)
        )

    def forward(
        self,
        op_type_id: torch.LongTensor,
        lhs_col_id: torch.LongTensor,
        rhs_is_col: torch.LongTensor,
        rhs_col_id: torch.LongTensor,
        rhs_lit_is_num: torch.LongTensor,
        rhs_lit_val: torch.FloatTensor,
        rhs_lit_bucket: torch.LongTensor,
    ) -> torch.FloatTensor:
        # Embeddings
        e_op  = self.op_emb(op_type_id.clamp_min(0))       # [N, op_dim]
        e_lhs = self.col_emb(lhs_col_id.clamp_min(0))      # [N, col_dim]
        e_rhs_col = self.col_emb(rhs_col_id.clamp_min(0))  # [N, col_dim]
        e_rhs_str = self.str_emb(rhs_lit_bucket.clamp_min(0))  # [N, str_dim]

        # 数值常量向量
        v_num = self.num_proj(rhs_lit_val.view(-1, 1))     # [N, num_val_dim]

        # 门控：优先列；否则常量 -> 数值 or 字符串
        g_join = rhs_is_col.float().unsqueeze(-1)          # [N,1]
        g_num  = rhs_lit_is_num.float().unsqueeze(-1)      # [N,1]

        # 把“常量通道”拼一起再线性齐次到 col_dim，以便与列向量等维融合
        const_vec = torch.cat([v_num, e_rhs_str], dim=-1)  # [N, num_val_dim + str_dim]
        # 占位线性把常量通道映射到与列 embedding 相同的维度
        const_to_col = nn.functional.linear(
            const_vec,
            weight=torch.zeros(const_vec.size(-1), e_rhs_col.size(-1), device=const_vec.device)
        )
        # 为了可学习，改用一个参数层（缓存一次）
        if not hasattr(self, "_const_lin"):
            self._const_lin = nn.Linear(const_vec.size(-1), e_rhs_col.size(-1), bias=False).to(const_vec.device)
        const_proj = self._const_lin(const_vec)            # [N, col_dim]

        rhs_vec = g_join * e_rhs_col + (1.0 - g_join) * (g_num * const_proj + (1.0 - g_num) * const_proj)

        z = torch.cat([e_op, e_lhs, rhs_vec], dim=-1)
        out = self.fuse(z)                                  # [N, out_dim]
        return out

class NodeEncoder_V3(nn.Module):
    """
    读取 data 上的多路输入并输出节点向量：
      - 类型嵌入：op_name_id  -> type_emb
      - 谓词嵌入：PredicateEncoderV3(...)
      - 数值通道：x_num -> 小 MLP
      - 拼接 -> 深层投影到 out_dim
    期望 data 包含：
      data.op_name_id (long)
      data.op_type_id (long)
      data.lhs_col_id (long)
      data.rhs_col_id (long)
      data.rhs_is_col (float/long -> 0/1)
      data.rhs_lit_is_num (float/long -> 0/1)
      data.rhs_lit_val (float)
      data.rhs_lit_bucket (long)
      data.x_num (float): 建议包含 [est_card, est_width, est_cost, has_predicate, rhs_is_col, rhs_lit_is_num, rhs_lit_val, literal_feature]
    """
    def __init__(
        self,
        num_node_types: int,
        num_cols: int,
        num_ops: int,
        num_str_buckets: int = 1000,
        type_dim: int = 16,
        num_dim_in: int = 8,    # 你的 x_num 列数（与构图时一致）
        num_dim_out: int = 8,   # 数值通道映射后的维度
        pred_out: int = 32,     # 谓词编码输出维度
        out_dim: int = 39,
        hidden: int = 64,
        drop: float = 0.2,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, type_dim)

        self.pred_enc = PredicateEncoderV3(
            num_cols=num_cols,
            num_ops=num_ops,
            num_str_buckets=num_str_buckets,
            col_dim=16, op_dim=8, str_dim=8, num_val_dim=8,
            out_dim=pred_out, drop=0.0
        )

        self.num_proj = nn.Sequential(
            nn.Linear(num_dim_in, 16),
            nn.ReLU(),
            nn.Linear(16, num_dim_out)
        )

        in_dim = type_dim + pred_out + num_dim_out

        # 两层残差块 + LN
        self.proj_in = nn.Linear(in_dim, hidden)
        self.norm1   = nn.LayerNorm(hidden)
        self.fc1     = nn.Linear(hidden, hidden)
        self.drop    = nn.Dropout(drop)
        self.fc2     = nn.Linear(hidden, out_dim)
        self.norm2   = nn.LayerNorm(out_dim)

    def forward(self, data):
        # 类型嵌入
        t = self.type_emb(data.op_name_id)  # [N, type_dim]

        # 谓词嵌入
        p = self.pred_enc(
            op_type_id=data.op_type_id,
            lhs_col_id=data.lhs_col_id,
            rhs_is_col=data.rhs_is_col.long(),
            rhs_col_id=data.rhs_col_id,
            rhs_lit_is_num=data.rhs_lit_is_num.long(),
            rhs_lit_val=data.rhs_lit_val,
            rhs_lit_bucket=data.rhs_lit_bucket
        )  # [N, pred_out]

        # 数值通道
        x_num = self.num_proj(data.x_num)    # [N, num_dim_out]

        # 融合 & 残差
        h = torch.cat([t, p, x_num], dim=-1)  # [N, in_dim]
        h = self.proj_in(h)                   # [N, hidden]
        h = self.norm1(h + F.relu(h))         # pre-act small residual
        h = self.drop(self.fc1(h))
        h = F.relu(h)
        h = self.fc2(h)                       # [N, out_dim]
        h = self.norm2(h)
        return h

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

# 移植自 QueryFormer 的 FeatureEmbed，提供丰富的统计信息编码能力
# 注意：使用此 Encoder 需要输入特征 x 具有特定的格式和维度 (通常为 1165 维)
# 输入 x 应包含: [Type(1), Join(1), Filters(9), FilterMask(3), Hists(150), TableSample(1001)]
class NodeEncoder_QF(nn.Module):
    def __init__(self, embed_size=64, tables=None, types=None, joins=None, columns=None, 
                 ops=None, use_sample=True, use_hist=True, bin_number=50):
        super(NodeEncoder_QF, self).__init__()
        
        # 默认词表大小 (如果未提供)
        self.tables = tables if tables is not None else 20
        self.types = types if types is not None else 20
        self.joins = joins if joins is not None else 10
        self.columns = columns if columns is not None else 30
        self.ops = ops if ops is not None else 10
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(self.types, embed_size)
        self.tableEmbed = nn.Embedding(self.tables, embed_size)
        
        self.columnEmbed = nn.Embedding(self.columns, embed_size)
        self.opEmbed = nn.Embedding(self.ops, embed_size//8)

        self.linearFilter2 = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearFilter = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)

        self.linearType = nn.Linear(embed_size, embed_size)
        
        self.linearJoin = nn.Linear(embed_size, embed_size)
        
        self.linearSample = nn.Linear(1000, embed_size)
        
        self.linearHist = nn.Linear(3, embed_size) # QueryFormer 原版是 3 -> embed_size (batch*50*3 -> transpose -> batch*50*embed)
        # 注意: QueryFormer 原版代码中 self.linearHist = nn.Linear(bin_number, embed_size) 是错误的? 
        # 所以原版代码确实是 nn.Linear(bin_number, embed_size)
        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(self.joins, embed_size)
        
        if use_hist:
            # 5个部分: Type, Filter, Join, Table, Hist
            self.project = nn.Linear(embed_size*5 + embed_size//8+1, embed_size) # 最终映射到 embed_size 以适配 GNTO 接口
        else:
            self.project = nn.Linear(embed_size*4 + embed_size//8+1, embed_size)
    
    # input: B by 1165 (type, join, f1, f2, f3, mask1, mask2, mask3, hists, samples)
    def forward(self, feature):
        # 确保 feature 是 float (部分 Embedding 需要 long，会在内部转换)
        
        # 切分特征
        # 1 + 1 + 9 + 3 + 150 + 1001 = 1165
        typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(
            feature, 
            (1, 1, 9, 3, self.bin_number*3, 1001), 
            dim=-1
        )
        
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        
        histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)
    
        if self.use_hist:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb), dim=1)
        else:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim=1)
            
        # 投影到单一向量，作为 Node Embedding
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(1)
    
    def getTable(self, table_sample):
        table, sample = torch.split(table_sample, (1, 1000), dim=-1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        
        if self.use_sample:
            emb = emb + self.linearSample(sample)
        return emb
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())
        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # hists: [B, 150] -> view -> [B, 50, 3] -> transpose -> [B, 3, 50]
        histExpand = hists.view(-1, self.bin_number, 3).transpose(1, 2)
        
        # linearHist: [50] -> [embed_size]
        emb = self.linearHist(histExpand) # [B, 3, embed_size]
        
        # Mask: filtersMask [B, 3]
        emb[~filtersMask.bool()] = 0.  # mask out unused filters
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(emb, dim=1) # [B, embed_size]
        
        # 防止除以0
        num_filters = num_filters.view(-1, 1).clamp_min(1.0)
        avg = total / num_filters
        
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        ## filtersId: [B, 9] -> view -> [B, 3, 3] -> transpose -> [B, 3, 3] (col, op, val)
        filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2) # 这里的 transpose 看起来是为了变成 (N, 3, 3) where dim 2 is (col, op, val)?
        # QueryFormer 代码: filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        # 假设 filtersId 是 [col1, col2, col3, op1, op2, op3, val1, val2, val3] ? 
        # 不，通常是 [c1, o1, v1, c2, o2, v2, c3, o3, v3]。如果是这样，view(-1, 3, 3) 变成 [[c1, o1, v1], [c2, o2, v2]...]
        # 此时 transpose(1, 2) 会变成 [[c1, c2, c3], [o1, o2, o3], [v1, v2, v3]] ??
        # 等等，QueryFormer 取值逻辑：
        # colsId = filterExpand[:,:,0]
        # 如果 view 之后是 [[c1,o1,v1], [c2,o2,v2], [c3,o3,v3]]
        # 那么 [:,:,0] 得到 [c1, c2, c3]。这是对的。
        # 那么 transpose(1,2) 这一步是为什么？
        # 如果原始数据是按照 (3个filter x 3个属性) 排列，view 出来就是 (filter_idx, attr_idx)。
        # 此时 [:,:,0] 取的是第0个属性（即 col）。
        # 所以 transpose 也许是不需要的，或者原始数据排列方式很奇怪（比如 [c1, c2, c3, o1, o2, o3...]）。
        # 无论如何，我们照搬 QueryFormer 的逻辑，假设输入数据格式与之一致。
        
        # 修正: 如果直接 copy QueryFormer 代码:
        # filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        # colsId = filterExpand[:,:,0]
        # 假设 filtersId 排列是: [c1, c2, c3, o1, o2, o3, v1, v2, v3]
        # view(-1, 3, 3) -> 
        # row0: c1 c2 c3
        # row1: o1 o2 o3
        # row2: v1 v2 v3
        # transpose(1, 2) ->
        # row0: c1 o1 v1
        # row1: c2 o2 v2
        # row2: c3 o3 v3
        # 这样 [:,:,0] 取到 c1, c2, c3。
        # 所以这里的假设是输入数据按“属性优先”排列的。
        
        filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2)
        colsId = filterExpand[:, :, 0].long()
        opsId = filterExpand[:, :, 1].long()
        vals = filterExpand[:, :, 2].unsqueeze(-1) # b by 3 by 1
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        
        concat = torch.cat((col, op, vals), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        
        ## apply mask
        concat[~filtersMask.bool()] = 0.
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(concat, dim=1)
        
        num_filters = num_filters.view(-1, 1).clamp_min(1.0)
        avg = total / num_filters
                
        return avg


class NodeEncoder_QF_AddPlanrows(nn.Module):
    def __init__(self, embed_size=64, tables=None, types=None, joins=None, columns=None, 
                 ops=None, use_sample=True, use_hist=True, bin_number=50):
        super(NodeEncoder_QF_AddPlanrows, self).__init__()
        
        # 默认词表大小 (如果未提供)
        self.tables = tables if tables is not None else 20
        self.types = types if types is not None else 20
        self.joins = joins if joins is not None else 10
        self.columns = columns if columns is not None else 30
        self.ops = ops if ops is not None else 10
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(self.types, embed_size)
        self.tableEmbed = nn.Embedding(self.tables, embed_size)
        
        self.columnEmbed = nn.Embedding(self.columns, embed_size)
        self.opEmbed = nn.Embedding(self.ops, embed_size//8)

        self.linearFilter2 = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearFilter = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)

        self.linearType = nn.Linear(embed_size, embed_size)
        
        self.linearJoin = nn.Linear(embed_size, embed_size)
        
        self.linearSample = nn.Linear(1000, embed_size)
        
        self.linearHist = nn.Linear(3, embed_size) # QueryFormer 原版是 3 -> embed_size (batch*50*3 -> transpose -> batch*50*embed)
        # 注意: QueryFormer 原版代码中 self.linearHist = nn.Linear(bin_number, embed_size) 是错误的? 
        # 所以原版代码确实是 nn.Linear(bin_number, embed_size)
        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(self.joins, embed_size)
        
        if use_hist:
            # 5个部分: Type, Filter, Join, Table, Hist
            self.project = nn.Linear(embed_size*5 + embed_size//8+1, embed_size) # 最终映射到 embed_size 以适配 GNTO 接口
        else:
            self.project = nn.Linear(embed_size*4 + embed_size//8+1, embed_size)
            
        # === 新增: Plan Rows 映射层 ===
        self.rows_proj = nn.Linear(1, self.embed_size)
    
    # input: B by 1165 (type, join, f1, f2, f3, mask1, mask2, mask3, hists, samples)
    def forward(self, feature):
        # 确保 feature 是 float (部分 Embedding 需要 long，会在内部转换)
        
        # 1. 尝试切分出 plan_rows (假设 feature 可能多了一维)
        # 1165 + 1 = 1166
        if feature.size(-1) == 1166:
             # 前 1165 维是 QF 特征，最后 1 维是 plan_rows
            qf_feature, plan_rows = torch.split(feature, (1165, 1), dim=-1)
            rows_emb = F.relu(self.rows_proj(plan_rows))
        else:
            qf_feature = feature
            rows_emb = None
        if feature.size(-1) == 1166:
             # 前 1165 维是 QF 特征，最后 1 维是 plan_rows
            qf_feature, plan_rows = torch.split(feature, (1165, 1), dim=-1)
            # 对 plan_rows 做简单的 Log + Linear 处理 (假设已经 log1p 了，这里直接过线性层)
            # 需要在 init 中定义 self.rows_proj = nn.Linear(1, embed_size)
            if not hasattr(self, 'rows_proj'):
                # 动态添加层 (hacky but works for hotfix)，建议正式用时写在 init 里
                self.rows_proj = nn.Linear(1, self.embed_size).to(feature.device)
            
            rows_emb = F.relu(self.rows_proj(plan_rows))
        else:
            qf_feature = feature
            rows_emb = None

        # 切分特征
        # 1 + 1 + 9 + 3 + 150 + 1001 = 1165
        typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(
            qf_feature, 
            (1, 1, 9, 3, self.bin_number*3, 1001), 
            dim=-1
        )
        
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        
        histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)
    
        if self.use_hist:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb), dim=1)
        else:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim=1)
            
        # 投影到单一向量，作为 Node Embedding
        final = F.leaky_relu(self.project(final))
        
        # 如果有 plan_rows，加到 final 上 (ResNet style) 或者 concat 后再投影
        if rows_emb is not None:
             final = final + rows_emb # 简单相加，前提是维度都是 embed_size
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(1)
    
    def getTable(self, table_sample):
        table, sample = torch.split(table_sample, (1, 1000), dim=-1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        
        if self.use_sample:
            emb = emb + self.linearSample(sample)
        return emb
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())
        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # hists: [B, 150] -> view -> [B, 50, 3] -> transpose -> [B, 3, 50]
        histExpand = hists.view(-1, self.bin_number, 3).transpose(1, 2)
        
        # linearHist: [50] -> [embed_size]
        emb = self.linearHist(histExpand) # [B, 3, embed_size]
        
        # Mask: filtersMask [B, 3]
        emb[~filtersMask.bool()] = 0.  # mask out unused filters
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(emb, dim=1) # [B, embed_size]
        
        # 防止除以0
        num_filters = num_filters.view(-1, 1).clamp_min(1.0)
        avg = total / num_filters
        
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        ## filtersId: [B, 9] -> view -> [B, 3, 3] -> transpose -> [B, 3, 3] (col, op, val)
        filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2) # 这里的 transpose 看起来是为了变成 (N, 3, 3) where dim 2 is (col, op, val)?
        # QueryFormer 代码: filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        # 假设 filtersId 是 [col1, col2, col3, op1, op2, op3, val1, val2, val3] ? 
        # 不，通常是 [c1, o1, v1, c2, o2, v2, c3, o3, v3]。如果是这样，view(-1, 3, 3) 变成 [[c1, o1, v1], [c2, o2, v2]...]
        # 此时 transpose(1, 2) 会变成 [[c1, c2, c3], [o1, o2, o3], [v1, v2, v3]] ??
        # 等等，QueryFormer 取值逻辑：
        # colsId = filterExpand[:,:,0]
        # 如果 view 之后是 [[c1,o1,v1], [c2,o2,v2], [c3,o3,v3]]
        # 那么 [:,:,0] 得到 [c1, c2, c3]。这是对的。
        # 那么 transpose(1,2) 这一步是为什么？
        # 如果原始数据是按照 (3个filter x 3个属性) 排列，view 出来就是 (filter_idx, attr_idx)。
        # 此时 [:,:,0] 取的是第0个属性（即 col）。
        # 所以 transpose 也许是不需要的，或者原始数据排列方式很奇怪（比如 [c1, c2, c3, o1, o2, o3...]）。
        # 无论如何，我们照搬 QueryFormer 的逻辑，假设输入数据格式与之一致。
        
        # 修正: 如果直接 copy QueryFormer 代码:
        # filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        # colsId = filterExpand[:,:,0]
        # 假设 filtersId 排列是: [c1, c2, c3, o1, o2, o3, v1, v2, v3]
        # view(-1, 3, 3) -> 
        # row0: c1 c2 c3
        # row1: o1 o2 o3
        # row2: v1 v2 v3
        # transpose(1, 2) ->
        # row0: c1 o1 v1
        # row1: c2 o2 v2
        # row2: c3 o3 v3
        # 这样 [:,:,0] 取到 c1, c2, c3。
        # 所以这里的假设是输入数据按“属性优先”排列的。
        
        filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2)
        colsId = filterExpand[:, :, 0].long()
        opsId = filterExpand[:, :, 1].long()
        vals = filterExpand[:, :, 2].unsqueeze(-1) # b by 3 by 1
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        
        concat = torch.cat((col, op, vals), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        
        ## apply mask
        concat[~filtersMask.bool()] = 0.
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(concat, dim=1)
        
        num_filters = num_filters.view(-1, 1).clamp_min(1.0)
        avg = total / num_filters
                
        return avg
        
class NodeEncoder_V4(nn.Module):
    """
    [V4 Update]
    基于 V2 修改，严格对齐 GNTO 论文的 DeepSets 设计：
    1. 谓词聚合方式改为 Sum (求和)，保留 Total Filtering Mass。
       (V1/V2 使用了 Mean Pooling)
    2. 输入 x 结构假设与 V1/V2 一致: [node_type | rows,width | (col1,op,c2n,ij)*3]
    """
    def __init__(self, num_node_types, num_cols, num_ops,
                 type_dim=16, num_dim=2, out_dim=39, hidden_dim=64, drop=0.2):
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, type_dim)

        # 谓词编码器 (复用 V1/V2 的逻辑)
        self.pred_enc = PredicateEncoder1(num_cols=num_cols, num_ops=num_ops,
                                          col_dim=8, op_dim=3)

        # 数值特征 MLP
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, 8),
            nn.ReLU(),
            nn.Linear(8, num_dim)
        )

        d_pred = 8 + 3 + 8 + 1 + 1  # 21
        in_dim = type_dim + num_dim + d_pred

        # 投影层
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
        x: [N, 15] -> [node_type | rows,width | (col1,op,c2n,ij)*3]
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

        # === (4) DeepSets Sum Aggregation (关键修改) ===
        # 论文强调 Sum 以保留 "total filtering mass"
        presence = (preds_raw.abs().sum(dim=-1) > 0).float()
        
        # V2: sum / denom (Mean)
        # V4: sum (Sum)
        p_vec = (p_all * presence.unsqueeze(-1)).sum(dim=1)  # [N, 21]

        # === (5) 拼接并映射 ===
        node_vec = torch.cat([t_emb, n_feat, p_vec], dim=-1)   # [N, 39]
        out = self.proj(node_vec)                              # [N, out_dim]
        return out
