"""Pure Node-level Encoder for PlanNode objects.

This is the correct NODE ENCODER layer in the architecture:
📊 Architecture Position: Step 2 (Node-level Encoding)
- Input: Individual TreeNode with attributes (node_type, extra_info)
- Output: Node-level embedding vector
- Scope: ONLY single node feature extraction

⚠️  IMPORTANT: This encoder handles ONLY node-level features.
NO tree structure processing, NO recursive aggregation.
Structure-level encoding is handled by TreeModel.

🛠️ 编码方式：分块编码 (Multi-View Encoding)
- 算子类型 → Embedding
- 数据统计 → MLP
- 谓词信息 → Encoder
- 最后 Concat 所有特征
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import re
from .DataPreprocessor import PlanNode


class NodeEncoder(nn.Module):
    """Pure node-level encoder using multi-view encoding strategy.
    分块编码策略：
    1. 算子类型 → Embedding Layer
    2. 数据统计 → MLP (标准化 + 全连接)
    3. 谓词信息 → Simple Encoder (复杂度特征)
    4. 最后 Concat 所有特征

    """
    
    def __init__(self, 
                 operator_embedding_dim: int = 32,
                 stats_hidden_dim: int = 16,
                 predicate_dim: int = 8,
                 output_dim: int = 64) -> None:
        """Initialize the multi-view node encoder.
        
        Parameters
        ----------
        operator_embedding_dim: int
            算子类型embedding维度
        stats_hidden_dim: int
            统计特征MLP隐层维度
        predicate_dim: int
            谓词特征维度
        output_dim: int
            最终输出维度
        """
        super().__init__()
        
        # 配置参数
        self.operator_embedding_dim = operator_embedding_dim
        self.stats_hidden_dim = stats_hidden_dim
        self.predicate_dim = predicate_dim
        self.output_dim = output_dim
        
        # 算子类型词汇表
        self.node_type_vocab: Dict[str, int] = {}
        
        # 核心统计特征键
        self.stats_keys = ['Plan Rows', 'Plan Width', 'Startup Cost', 'Total Cost']
        
        # 谓词特征键
        self.predicate_keys = ['Filter', 'Index Cond', 'Hash Cond', 'Merge Cond', 'Join Filter']
        
        # 延迟初始化的组件 (在第一次forward时初始化)
        self.operator_embedding: Optional[nn.Embedding] = None
        self.stats_mlp: Optional[nn.Sequential] = None
        self.output_projection: Optional[nn.Linear] = None
        
        self._initialized = False
    
    def _ensure_initialized(self, node):
        """确保所有组件都已初始化"""
        if self._initialized:
            # 检查是否需要扩展embedding层
            self._update_operator_vocab(node)
            current_vocab_size = len(self.node_type_vocab)
            if current_vocab_size > self.operator_embedding.num_embeddings:
                # 需要扩展embedding层
                old_embedding = self.operator_embedding
                self.operator_embedding = nn.Embedding(current_vocab_size, self.operator_embedding_dim)
                # 复制旧的权重
                with torch.no_grad():
                    self.operator_embedding.weight[:old_embedding.num_embeddings] = old_embedding.weight
            return
            
        # 1. 初始化算子embedding
        self._update_operator_vocab(node)
        vocab_size = len(self.node_type_vocab)
        self.operator_embedding = nn.Embedding(vocab_size, self.operator_embedding_dim)
        
        # 2. 初始化统计特征MLP
        stats_input_dim = len(self.stats_keys)
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_input_dim, self.stats_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.stats_hidden_dim, self.stats_hidden_dim)
        )
        
        # 3. 计算concat后的总维度
        total_dim = self.operator_embedding_dim + self.stats_hidden_dim + self.predicate_dim
        
        # 4. 初始化输出投影层
        self.output_projection = nn.Linear(total_dim, self.output_dim)
        
        self._initialized = True
    
    def _update_operator_vocab(self, node):
        """更新算子类型词汇表"""
        node_type = getattr(node, "node_type", "Unknown")
        if node_type not in self.node_type_vocab:
            self.node_type_vocab[node_type] = len(self.node_type_vocab)
    
    def _encode_operator(self, node) -> torch.Tensor:
        """编码算子类型 → Embedding"""
        self._update_operator_vocab(node)
        node_type = getattr(node, "node_type", "Unknown")
        idx = self.node_type_vocab[node_type]
        idx_tensor = torch.tensor([idx], dtype=torch.long)
        return self.operator_embedding(idx_tensor).squeeze(0)  # [embedding_dim]
    
    def _encode_stats(self, node) -> torch.Tensor:
        """编码数据统计 → MLP"""
        extra_info = getattr(node, 'extra_info', {})
        stats_values = []
        
        for key in self.stats_keys:
            value = extra_info.get(key, 0.0)
            if isinstance(value, (int, float)):
                stats_values.append(float(value))
            else:
                try:
                    stats_values.append(float(str(value).replace(',', '')))
                except:
                    stats_values.append(0.0)
        
        # 简单的log标准化
        stats_tensor = torch.tensor(stats_values, dtype=torch.float32)
        stats_tensor = torch.log1p(stats_tensor)  # log(1 + x)
        
        # 通过MLP
        return self.stats_mlp(stats_tensor)  # [stats_hidden_dim]
    
    def _encode_predicate(self, node) -> torch.Tensor:
        """编码谓词信息 → 简单特征"""
        extra_info = getattr(node, 'extra_info', {})
        
        # 收集所有谓词信息
        predicates = []
        for key in self.predicate_keys:
            if key in extra_info and extra_info[key]:
                predicates.append(str(extra_info[key]))
        
        if not predicates:
            return torch.zeros(self.predicate_dim, dtype=torch.float32)
        
        # 简单的复杂度特征
        all_predicates = ' '.join(predicates).lower()
        
        features = []
        
        # 1. 谓词数量 (归一化)
        features.append(min(len(predicates) / 5.0, 1.0))
        
        # 2. 是否有范围过滤
        range_patterns = ['>', '<', '>=', '<=', 'between']
        features.append(float(any(pattern in all_predicates for pattern in range_patterns)))
        
        # 3. 是否包含子查询
        subquery_patterns = ['exists', 'in (select', 'subplan']
        features.append(float(any(pattern in all_predicates for pattern in subquery_patterns)))
        
        # 4. 是否有函数调用
        features.append(float('(' in all_predicates))
        
        # 5. 是否有LIKE模式匹配
        features.append(float('like' in all_predicates or '%' in all_predicates))
        
        # 6. 连接条件数量 (归一化)
        join_count = all_predicates.count('=')
        features.append(min(join_count / 3.0, 1.0))
        
        # 7-8. 填充到predicate_dim维度
        while len(features) < self.predicate_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.predicate_dim], dtype=torch.float32)
    
    # 编码函数
    def forward(self, node) -> torch.Tensor:
        """分块编码 + Concat
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
            
        Returns
        -------
        torch.Tensor
            节点编码向量 [output_dim]
        """
        # 确保初始化
        self._ensure_initialized(node)
        
        # 1. 算子类型编码
        operator_vec = self._encode_operator(node)  # [operator_embedding_dim]
        
        # 2. 统计特征编码
        stats_vec = self._encode_stats(node)  # [stats_hidden_dim]
        
        # 3. 谓词特征编码
        predicate_vec = self._encode_predicate(node)  # [predicate_dim]
        
        # 4. Concat所有特征
        combined = torch.cat([operator_vec, stats_vec, predicate_vec], dim=0)
        
        # 5. 输出投影
        output = self.output_projection(combined)  # [output_dim]
        
        return output
    
    def encode_node(self, node) -> torch.Tensor:
        """编码单个节点并存储到node.node_vector
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
            
        Returns
        -------
        torch.Tensor
            节点编码向量
        """
        vector = self.forward(node)
        node.node_vector = vector
        return vector
    
    def encode_nodes(self, nodes: Iterable) -> List[torch.Tensor]:
        """编码多个节点
        
        Parameters
        ----------
        nodes: Iterable
            查询计划节点列表
            
        Returns
        -------
        List[torch.Tensor]
            节点编码向量列表
        """
        return [self.encode_node(node) for node in nodes]
    
    @staticmethod
    def collect_nodes(root: PlanNode, method: str = "dfs") -> List[PlanNode]:
        """
        遍历 PlanNode 树，收集所有节点为列表
        
        Parameters
        ----------
        root : PlanNode
            根节点
        method : str
            遍历方式，可选 "dfs" (深度优先) 或 "bfs" (广度优先)
        
        Returns
        -------
        List[PlanNode]
            树中所有节点的列表
        """
        nodes = []
        
        if method == "dfs":
            # 递归深度优先
            def dfs(node: PlanNode):
                nodes.append(node)
                for child in node.children:
                    dfs(child)
            dfs(root)
        
        elif method == "bfs":
            # 队列广度优先
            queue = [root]
            while queue:
                node = queue.pop(0)
                nodes.append(node)
                queue.extend(node.children)
        
        else:
            raise ValueError("method 必须是 'dfs' 或 'bfs'")
        
        return nodes

    # 获取信息
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim
    
    def get_vocab_size(self) -> int:
        """获取算子词汇表大小"""
        return len(self.node_type_vocab)
    
    def get_config(self) -> Dict[str, Any]:
        """获取编码器配置信息"""
        return {
            'operator_embedding_dim': self.operator_embedding_dim,
            'stats_hidden_dim': self.stats_hidden_dim,
            'predicate_dim': self.predicate_dim,
            'output_dim': self.output_dim,
            'vocab_size': len(self.node_type_vocab),
            'initialized': self._initialized
        }