"""
查询计划表示学习的神经网络编码器
包含节点编码器和结构编码器两个核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from typing import Dict, Optional, Tuple, List
import math


class NodeEncoder(nn.Module):
    """
    节点编码器 - 将执行计划中每个节点的特征编码为向量
    
    功能：
    - 将原始节点特征（如行数、算子类型等）转换为密集的向量表示
    - 支持连续特征和类别特征的混合编码
    - 使用多层感知机(MLP)进行特征变换
    """
    
    def __init__(
        self,
        input_dim: int,          # 输入特征维度
        hidden_dims: List[int] = [256, 128],  # 隐藏层维度列表
        output_dim: int = 128,   # 输出向量维度
        dropout: float = 0.1,    # Dropout比例
        activation: str = "relu", # 激活函数类型
        batch_norm: bool = True  # 是否使用批量归一化
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # 线性变换
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))   # 批量归一化
            
            # 添加激活函数
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            # 添加Dropout防止过拟合
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias)         # 偏置初始化为0
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 将节点特征编码为向量
        
        参数:
            node_features: [N, input_dim] 节点特征张量
            
        返回:
            [N, output_dim] 节点嵌入向量
        """
        return self.encoder(node_features)


class StructureEncoder(nn.Module):
    """
    结构编码器 - 使用图神经网络(GNN)编码执行计划结构
    
    功能：
    - 将执行计划的树/DAG结构编码为单个向量表示
    - 支持多种GNN架构：GCN、GAT等
    - 通过消息传递机制聚合节点信息
    - 最终输出整个计划的向量表示
    """
    
    def __init__(
        self,
        node_dim: int,           # 节点特征维度（来自NodeEncoder的输出）
        hidden_dim: int = 128,   # GNN隐藏层维度
        num_layers: int = 3,     # GNN层数
        num_edge_types: int = 10, # 边类型数量
        gnn_type: str = "gcn",   # GNN类型：gcn或gat
        heads: int = 4,          # GAT的注意力头数
        dropout: float = 0.1,    # Dropout比例
        pooling: str = "mean",   # 池化方式：mean, max, sum, attention
        residual: bool = True    # 是否使用残差连接
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.gnn_type = gnn_type.lower()
        self.pooling = pooling
        self.residual = residual
        
        # 边类型嵌入层
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        
        # 输入投影层：将节点特征投影到GNN隐藏维度
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GNN层列表和层归一化列表
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # 构建GNN层
        for i in range(num_layers):
            if self.gnn_type == "gcn":
                # 图卷积网络层
                layer = GCNConv(hidden_dim, hidden_dim)
            elif self.gnn_type == "gat":
                # 图注意力网络层
                layer = GATConv(
                    hidden_dim, 
                    hidden_dim // heads, 
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            else:
                raise ValueError(f"不支持的GNN类型: {gnn_type}")
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # 池化层用于生成计划级表示
        if pooling == "attention":
            self.attention_pool = AttentionPooling(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        nodes: torch.Tensor,  # [N, node_dim] 节点特征
        edges: torch.Tensor,  # [2, E] 边索引
        edge_types: torch.Tensor,  # [E] 边类型
        batch_idx: Optional[torch.Tensor] = None  # [N] 节点所属批次索引
    ) -> Dict[str, torch.Tensor]:
        """
        GNN前向传播 - 将计划结构编码为向量
        
        参数:
            nodes: [N, node_dim] 节点特征张量
            edges: [2, E] 边索引张量
            edge_types: [E] 边类型索引
            batch_idx: [N] 节点批次分配（用于批处理）
            
        返回:
            包含以下内容的字典:
            - node_emb: [N, hidden_dim] 节点嵌入向量
            - plan_emb: [B, hidden_dim] 计划级嵌入向量（如果提供了batch_idx）
        """
        # 1. 投影输入特征到GNN隐藏维度
        x = self.input_proj(nodes)  # [N, hidden_dim]
        
        # 2. 获取边类型嵌入
        edge_attr = self.edge_type_embedding(edge_types)  # [E, hidden_dim]
        
        # 3. 通过GNN层进行消息传递
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual_x = x  # 保存残差连接的输入
            
            # GNN消息传递
            if self.gnn_type == "gcn":
                x = gnn_layer(x, edges)
            elif self.gnn_type == "gat":
                x = gnn_layer(x, edges)
            
            # 层归一化和激活
            x = layer_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # 残差连接（跳跃连接）
            if self.residual and i > 0:
                x = x + residual_x
        
        result = {"node_emb": x}  # 保存节点级嵌入
        
        # 4. 如果提供了批次索引，进行计划级池化
        if batch_idx is not None:
            if self.pooling == "mean":
                # 平均池化：计算每个计划的节点平均值
                plan_emb = global_mean_pool(x, batch_idx)
            elif self.pooling == "max":
                # 最大池化：计算每个计划的节点最大值
                plan_emb = global_max_pool(x, batch_idx)
            elif self.pooling == "sum":
                # 求和池化：计算每个计划的节点和
                plan_emb = torch.zeros(batch_idx.max() + 1, x.size(1), device=x.device)
                plan_emb.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, x.size(1)), x)
            elif self.pooling == "attention":
                # 注意力池化：使用注意力机制加权聚合
                plan_emb = self.attention_pool(x, batch_idx)
            else:
                raise ValueError(f"不支持的池化方式: {self.pooling}")
            
            result["plan_emb"] = plan_emb  # 保存计划级嵌入
        
        return result


class AttentionPooling(nn.Module):
    """Attention-based pooling for graph-level representation"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] node embeddings
            batch_idx: [N] batch assignment
            
        Returns:
            [B, hidden_dim] pooled embeddings
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [N, 1]
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Pool by batch
        batch_size = batch_idx.max() + 1
        pooled = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for i in range(batch_size):
            mask = (batch_idx == i)
            if mask.sum() > 0:
                batch_x = x[mask]  # [N_i, hidden_dim]
                batch_weights = attn_weights[mask]  # [N_i, 1]
                batch_weights = F.softmax(batch_weights, dim=0)
                pooled[i] = (batch_x * batch_weights).sum(dim=0)
        
        return pooled


class TreeLSTMEncoder(nn.Module):
    """
    Tree-LSTM encoder for hierarchical execution plans
    Alternative to GNN-based structure encoding
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM cell components
        self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_f = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_u = nn.Parameter(torch.zeros(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Tree-LSTM
        
        Args:
            x: [N, input_dim] node features
            adjacency: [N, N] adjacency matrix representing tree structure
            
        Returns:
            [N, hidden_dim] node embeddings
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Process nodes in topological order (simplified for now)
        for t in range(seq_len):
            # Get children states (simplified - assumes sequential processing)
            child_h = h  # In practice, you'd aggregate children states
            child_c = c
            
            # Input gate
            i_t = torch.sigmoid(self.W_i(x[:, t]) + self.U_i(child_h) + self.b_i)
            
            # Forget gate
            f_t = torch.sigmoid(self.W_f(x[:, t]) + self.U_f(child_h) + self.b_f)
            
            # Output gate
            o_t = torch.sigmoid(self.W_o(x[:, t]) + self.U_o(child_h) + self.b_o)
            
            # Update gate
            u_t = torch.tanh(self.W_u(x[:, t]) + self.U_u(child_h) + self.b_u)
            
            # Update cell state
            c = f_t * child_c + i_t * u_t
            
            # Update hidden state
            h = o_t * torch.tanh(c)
            
            h = self.dropout(h)
        
        return h
