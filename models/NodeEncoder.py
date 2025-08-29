"""Pure Node-level Encoder for PlanNode objects.

This is the correct NODE ENCODER layer in the architecture:
📊 Architecture Position: Step 2 (Node-level Encoding)
- Input: Individual TreeNode with attributes (node_type, extra_info)
- Output: Node-level embedding vector
- Scope: ONLY single node feature extraction

⚠️  IMPORTANT: This encoder handles ONLY node-level features.
NO tree structure processing, NO recursive aggregation.
Structure-level encoding is handled by TreeModel.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Any, Union, Tuple
import numpy as np
from dataclasses import dataclass
import re
from collections import Counter

import torch
import torch.nn as nn


@dataclass
class NodeFeatures:
    """Container for node features extracted from query plan nodes."""
    
    node_type_vector: np.ndarray  # One-hot encoding of node type
    numerical_features: np.ndarray  # Numerical features like cost, rows, etc.
    categorical_features: np.ndarray  # Encoded categorical features
    predicate_features: np.ndarray  # Encoded predicate/condition features
    relation_features: np.ndarray  # Relation/index information features
    execution_context_features: np.ndarray  # Execution context features
    historical_features: np.ndarray  # Historical/actual feedback features
    combined_features: np.ndarray  # Combined feature vector


class NodeEncoder(nn.Module):
    """Pure node-level encoder for converting individual PlanNodes to feature vectors.
    
    This is the NODE ENCODER layer in the correct architecture:
    📊 Architecture Position: Step 2 (Node-level Encoding)
    - Input: Individual TreeNode with attributes (node_type, extra_info)
    - Output: Node-level embedding vector
    - Scope: Single node feature extraction ONLY
    
    ⚠️  CRITICAL: This encoder handles ONLY node-level features.
    - NO recursive processing of children
    - NO tree structure aggregation
    - NO graph construction
    
    Structure-level encoding (tree/graph aggregation) is handled by TreeModel.
    
    Two encoding strategies:
    1. Simple mode: Only node type (one-hot)
    2. Rich mode: Node type + numerical + categorical features
    """
    
    def __init__(self,
                 rich_features: bool = False,
                 feature_dim: Optional[int] = None,
                 include_numerical: bool = True,
                 include_categorical: bool = True,
                 normalize_features: bool = True) -> None:
        """Initialize the node encoder.
        
        Parameters
        ----------
        rich_features:
            If True, extracts comprehensive features beyond just node type
        feature_dim:
            Target dimension for output vectors (if None, uses dynamic size)
        include_numerical:
            Whether to include numerical features from extra_info
        include_categorical:
            Whether to include categorical features from extra_info
        normalize_features:
            Whether to normalize numerical features
        """
        super().__init__()

        # Core vocabulary for node types
        self.node_index: Dict[str, int] = {}
        
        # Feature extraction configuration
        self.rich_features = rich_features
        self.feature_dim = feature_dim
        self.include_numerical = include_numerical
        self.include_categorical = include_categorical
        self.normalize_features = normalize_features
        
        # Additional vocabularies for rich features
        self.categorical_vocabs: Dict[str, Dict[str, int]] = {}
        
        # Predefined feature keys for query plans
        self.numerical_keys = [
            'Total Cost', 'Startup Cost', 'Plan Rows', 'Plan Width',
            'Actual Total Time', 'Actual Rows', 'Actual Loops'
        ]
        
        self.categorical_keys = [
            'Join Type', 'Scan Direction', 'Strategy', 'Parent Relationship',
            'Relation Name', 'Index Name', 'Sort Method'
        ]
        
        # Additional feature keys for enhanced encoding
        self.predicate_keys = [
            'Filter', 'Index Cond', 'Hash Cond', 'Merge Cond', 'Join Filter'
        ]
        
        self.relation_keys = [
            'Relation Name', 'Alias', 'Index Name', 'Schema'
        ]
        
        self.execution_context_keys = [
            'Workers Planned', 'Workers Launched', 'Parallel Aware'
        ]
        
        self.historical_keys = [
            'Actual Total Time', 'Actual Startup Time', 'Actual Rows', 
            'Actual Loops', 'Shared Hit Blocks', 'Shared Read Blocks'
        ]
        
        # Embedding dimensions for different components
        self.operator_embedding_dim = 32
        self.relation_embedding_dim = 16
        self.predicate_embedding_dim = 24
    
    # ------------------------------------------------------------------ utilities
    def _ensure_index(self, node_type: str) -> int:
        """Ensure a node type exists in vocabulary and return its index."""
        if node_type not in self.node_index:
            self.node_index[node_type] = len(self.node_index)
        return self.node_index[node_type]
    
    def _one_hot(self, idx: int) -> np.ndarray:
        """Create one-hot vector for given index."""
        vec = np.zeros(len(self.node_index), dtype=float)
        vec[idx] = 1.0
        return vec
    
    def _ensure_categorical_vocab(self, vocab_dict: Dict[str, int], key: str) -> int:
        """Ensure a key exists in categorical vocabulary and return its index."""
        if key not in vocab_dict:
            vocab_dict[key] = len(vocab_dict)
        return vocab_dict[key]
    
    # --------------------------------------------------------------------- encoder
    def encode_node(self, node) -> 'torch.Tensor':
        """Encode a SINGLE PlanNode to feature vector and store it in node.node_vector.
        
        ⚠️  IMPORTANT: This method processes ONLY the given node.
        It does NOT process children or any tree structure.
        
        Parameters
        ----------
        node:
            PlanNode object to encode (children are IGNORED)
            
        Returns
        -------
        np.ndarray:
            Node-level feature vector (also stored in node.node_vector)
        """
        if not self.rich_features:
            # Simple mode: only node type
            vector = self._encode_simple(node)
        else:
            # Rich mode: comprehensive features
            vector = self._encode_rich(node)

        # Convert to torch tensor for downstream models
        tensor = torch.as_tensor(vector, dtype=torch.float32)

        # Store the vector in the node
        node.node_vector = tensor
        return tensor
    
    def _encode_simple(self, node) -> np.ndarray:
        """Simple encoding: only node type (one-hot)."""
        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        return self._one_hot(idx)
    
    def _encode_rich(self, node) -> np.ndarray:
        """Rich encoding: node type + numerical + categorical features."""
        # Start with node type encoding
        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        node_type_vec = self._one_hot(idx)
        
        # Collect all feature components
        feature_components = [node_type_vec]
        
        # Add numerical features
        if self.include_numerical:
            numerical_features = self._extract_numerical_features(node)
            if len(numerical_features) > 0:
                feature_components.append(numerical_features)
        
        # Add categorical features
        if self.include_categorical:
            categorical_features = self._extract_categorical_features(node)
            if len(categorical_features) > 0:
                feature_components.append(categorical_features)
        
        # Combine all features
        combined_features = np.concatenate(feature_components)
        
        # Apply fixed dimension if specified
        if self.feature_dim is not None:
            combined_features = self._resize_vector(combined_features, self.feature_dim)
        
        return combined_features
    
    def encode_nodes(self, nodes: Iterable) -> List['torch.Tensor']:
        """Encode multiple nodes into vectors and store them in each node.node_vector.
        
        Each node is processed independently and its vector is stored in node.node_vector.
        
        Parameters
        ----------
        nodes:
            Iterable of PlanNode objects to encode
            
        Returns
        -------
        List[np.ndarray]:
            List of node-level feature vectors (also stored in each node.node_vector)
        """
        return [self.encode_node(node) for node in nodes]

    # --------------------------------------------------------------------- nn.Module
    def forward(self, node):
        """Alias for encode_node to comply with nn.Module interface."""
        return self.encode_node(node)
    
    # --------------------------------------------------------- feature extraction ---
    def _extract_numerical_features(self, node) -> np.ndarray:
        """Extract numerical features from node's extra_info."""
        extra_info = getattr(node, 'extra_info', {})
        features = []
        
        for key in self.numerical_keys:
            value = extra_info.get(key, 0.0)
            
            # Handle different value types
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                try:
                    features.append(float(value))
                except ValueError:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        if not features:
            return np.array([])
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize_features:
            features = self._normalize_numerical(features)
        
        return features
    
    def _extract_categorical_features(self, node) -> np.ndarray:
        """Extract categorical features from node's extra_info."""
        extra_info = getattr(node, 'extra_info', {})
        features = []
        
        for key in self.categorical_keys:
            value = extra_info.get(key, 'Unknown')
            
            # Ensure vocabulary exists for this key
            if key not in self.categorical_vocabs:
                self.categorical_vocabs[key] = {}
            
            # Convert value to string and get index
            str_value = str(value) if value is not None else 'Unknown'
            idx = self._ensure_categorical_vocab(self.categorical_vocabs[key], str_value)
            features.append(idx)
        
        return np.array(features, dtype=np.float32) if features else np.array([])
    
    # ===================================================================
    # 输入维度编码方法 (Input Dimension Encoding Methods)
    # ===================================================================
    
    def encode_operator_type(self, node, method: str = 'one_hot') -> np.ndarray:
        """算子类型编码 (Operator Type Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        method: str
            编码方法: 'one_hot', 'embedding', 'learned_embedding'
            
        Returns
        -------
        np.ndarray
            算子类型特征向量
        """
        node_type = getattr(node, "node_type", "Unknown")
        
        if method == 'one_hot':
            idx = self._ensure_index(node_type)
            return self._one_hot(idx)
        
        elif method == 'embedding':
            # 简单的固定embedding (可以用预训练的算子embedding替换)
            idx = self._ensure_index(node_type)
            # 创建简单的embedding向量
            embedding = np.zeros(self.operator_embedding_dim, dtype=np.float32)
            # 使用hash来创建伪随机但确定性的embedding
            hash_val = hash(node_type) % self.operator_embedding_dim
            embedding[hash_val] = 1.0
            # 添加一些变化
            for i in range(min(3, self.operator_embedding_dim)):
                embedding[(hash_val + i + 1) % self.operator_embedding_dim] = 0.5
            return embedding / np.linalg.norm(embedding)
        
        elif method == 'learned_embedding':
            # 可学习的embedding (需要训练)
            # 这里返回索引，实际训练时会用embedding层
            idx = self._ensure_index(node_type)
            return np.array([idx], dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def encode_estimated_stats(self, node, normalize: bool = True) -> np.ndarray:
        """数据统计编码 (Estimated Stats Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        normalize: bool
            是否标准化数值
            
        Returns
        -------
        np.ndarray
            数据统计特征向量
        """
        extra_info = getattr(node, 'extra_info', {})
        stats_features = []
        
        # 核心统计信息
        core_stats = ['Plan Rows', 'Plan Width', 'Startup Cost', 'Total Cost']
        
        for stat in core_stats:
            value = extra_info.get(stat, 0.0)
            if isinstance(value, (int, float)):
                stats_features.append(float(value))
            else:
                try:
                    stats_features.append(float(str(value).replace(',', '')))
                except:
                    stats_features.append(0.0)
        
        if not stats_features:
            return np.array([])
        
        features = np.array(stats_features, dtype=np.float32)
        
        if normalize:
            # 使用log normalization for cost and rows
            features[:2] = np.log1p(features[:2])  # rows, width
            features[2:] = np.log1p(features[2:])  # costs
            
            # 简单的min-max normalization
            features = np.clip(features, 0, np.percentile(features, 95) if np.any(features > 0) else 1)
            max_val = np.maximum(features.max(), 1.0)
            features = features / max_val
        
        return features
    
    def encode_predicate_info(self, node, method: str = 'complexity') -> np.ndarray:
        """谓词/条件信息编码 (Predicate Info Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        method: str
            编码方法: 'complexity', 'bow', 'token_embedding'
            
        Returns
        -------
        np.ndarray
            谓词特征向量
        """
        extra_info = getattr(node, 'extra_info', {})
        
        # 收集所有谓词信息
        predicates = []
        for key in self.predicate_keys:
            if key in extra_info and extra_info[key]:
                predicates.append(str(extra_info[key]))
        
        if not predicates:
            if method == 'complexity':
                return np.zeros(6, dtype=np.float32)  # 6个复杂度特征
            else:
                return np.zeros(self.predicate_embedding_dim, dtype=np.float32)
        
        if method == 'complexity':
            return self._encode_predicate_complexity(predicates)
        elif method == 'bow':
            return self._encode_predicate_bow(predicates)
        elif method == 'token_embedding':
            return self._encode_predicate_tokens(predicates)
        else:
            raise ValueError(f"Unknown predicate encoding method: {method}")
    
    def encode_relation_info(self, node, method: str = 'embedding') -> np.ndarray:
        """关系/索引信息编码 (Relation/Index Info Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        method: str
            编码方法: 'embedding', 'one_hot', 'features'
            
        Returns
        -------
        np.ndarray
            关系信息特征向量
        """
        extra_info = getattr(node, 'extra_info', {})
        
        if method == 'features':
            # 布尔特征组合
            features = []
            
            # 是否使用索引
            has_index = bool(extra_info.get('Index Name') or extra_info.get('Index Cond'))
            features.append(float(has_index))
            
            # 是否有表名
            has_relation = bool(extra_info.get('Relation Name'))
            features.append(float(has_relation))
            
            # 是否有别名
            has_alias = bool(extra_info.get('Alias'))
            features.append(float(has_alias))
            
            # 扫描类型
            scan_types = ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']
            node_type = getattr(node, "node_type", "Unknown")
            for scan_type in scan_types:
                features.append(float(scan_type in node_type))
            
            return np.array(features, dtype=np.float32)
        
        elif method == 'embedding':
            # 关系名embedding
            relation_name = extra_info.get('Relation Name', 'Unknown')
            return self._create_relation_embedding(relation_name)
        
        elif method == 'one_hot':
            # 关系名one-hot
            relation_name = extra_info.get('Relation Name', 'Unknown')
            if 'relation_vocab' not in self.categorical_vocabs:
                self.categorical_vocabs['relation_vocab'] = {}
            idx = self._ensure_categorical_vocab(self.categorical_vocabs['relation_vocab'], relation_name)
            vec = np.zeros(len(self.categorical_vocabs['relation_vocab']), dtype=float)
            vec[idx] = 1.0
            return vec
        
        else:
            raise ValueError(f"Unknown relation encoding method: {method}")
    
    def encode_execution_context(self, node) -> np.ndarray:
        """执行上下文编码 (Execution Context Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
            
        Returns
        -------
        np.ndarray
            执行上下文特征向量
        """
        extra_info = getattr(node, 'extra_info', {})
        context_features = []
        
        # 是否并行
        workers_planned = extra_info.get('Workers Planned', 0)
        is_parallel = float(workers_planned > 0)
        context_features.append(is_parallel)
        
        # 并行度
        parallel_degree = float(workers_planned) if workers_planned else 0.0
        context_features.append(parallel_degree)
        
        # 是否是阻塞算子
        node_type = getattr(node, "node_type", "Unknown")
        blocking_ops = ['Sort', 'Hash', 'Aggregate', 'WindowAgg', 'Materialize']
        is_blocking = float(any(op in node_type for op in blocking_ops))
        context_features.append(is_blocking)
        
        # 是否是连接算子
        join_ops = ['Hash Join', 'Merge Join', 'Nested Loop']
        is_join = float(any(op in node_type for op in join_ops))
        context_features.append(is_join)
        
        # Join基数比 (如果是join算子)
        if is_join:
            # 这里可以根据实际情况计算左右输入基数比
            # 简化处理，使用计划行数作为代理
            plan_rows = extra_info.get('Plan Rows', 1.0)
            cardinality_ratio = np.log1p(float(plan_rows))
            context_features.append(cardinality_ratio)
        else:
            context_features.append(0.0)
        
        # Pipeline特征
        pipeline_ops = ['Seq Scan', 'Index Scan', 'Filter']
        is_pipeline = float(any(op in node_type for op in pipeline_ops))
        context_features.append(is_pipeline)
        
        return np.array(context_features, dtype=np.float32)
    
    def encode_historical_feedback(self, node, normalize: bool = True) -> np.ndarray:
        """历史/真实反馈编码 (Historical/Actual Feedback Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        normalize: bool
            是否标准化
            
        Returns
        -------
        np.ndarray
            历史反馈特征向量
        """
        extra_info = getattr(node, 'extra_info', {})
        historical_features = []
        
        # 实际执行统计
        actual_stats = ['Actual Total Time', 'Actual Startup Time', 'Actual Rows', 'Actual Loops']
        
        for stat in actual_stats:
            value = extra_info.get(stat, 0.0)
            if isinstance(value, (int, float)):
                historical_features.append(float(value))
            else:
                try:
                    historical_features.append(float(str(value)))
                except:
                    historical_features.append(0.0)
        
        # I/O统计
        io_stats = ['Shared Hit Blocks', 'Shared Read Blocks', 'Shared Dirtied Blocks', 'Shared Written Blocks']
        for stat in io_stats:
            value = extra_info.get(stat, 0.0)
            historical_features.append(float(value) if value else 0.0)
        
        # 计算估计vs实际的比率特征
        if len(historical_features) >= 4:  # 确保有实际统计
            # 时间比率
            estimated_cost = extra_info.get('Total Cost', 1.0)
            actual_time = historical_features[0]  # Actual Total Time
            if estimated_cost > 0 and actual_time > 0:
                time_ratio = actual_time / estimated_cost
                historical_features.append(time_ratio)
            else:
                historical_features.append(1.0)
            
            # 行数比率
            estimated_rows = extra_info.get('Plan Rows', 1.0)
            actual_rows = historical_features[2]  # Actual Rows
            if estimated_rows > 0 and actual_rows > 0:
                rows_ratio = actual_rows / estimated_rows
                historical_features.append(rows_ratio)
            else:
                historical_features.append(1.0)
        
        if not historical_features:
            return np.array([])
        
        features = np.array(historical_features, dtype=np.float32)
        
        if normalize:
            # 对时间和I/O统计使用log normalization
            features[:4] = np.log1p(features[:4])  # actual stats
            features[4:8] = np.log1p(features[4:8])  # io stats
            
            # 比率特征使用clip
            if len(features) > 8:
                features[8:] = np.clip(features[8:], 0.01, 100)  # ratio features
                features[8:] = np.log(features[8:])
        
        return features
    
    # ===================================================================
    # 编码方式实现 (Encoding Implementation Methods)
    # ===================================================================
    
    def encode_concatenation_mlp(self, node, components: List[str] = None) -> np.ndarray:
        """简单拼接 + MLP 编码方式
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        components: List[str]
            要包含的特征组件: ['operator', 'stats', 'predicate', 'relation', 'context', 'historical']
            
        Returns
        -------
        np.ndarray
            拼接后的特征向量
        """
        if components is None:
            components = ['operator', 'stats', 'context']
        
        feature_vectors = []
        
        for component in components:
            if component == 'operator':
                vec = self.encode_operator_type(node, method='one_hot')
            elif component == 'stats':
                vec = self.encode_estimated_stats(node)
            elif component == 'predicate':
                vec = self.encode_predicate_info(node, method='complexity')
            elif component == 'relation':
                vec = self.encode_relation_info(node, method='features')
            elif component == 'context':
                vec = self.encode_execution_context(node)
            elif component == 'historical':
                vec = self.encode_historical_feedback(node)
            else:
                continue
            
            if len(vec) > 0:
                feature_vectors.append(vec)
        
        if not feature_vectors:
            return np.array([])
        
        # 拼接所有特征
        combined = np.concatenate(feature_vectors)
        
        # 如果指定了目标维度，调整大小
        if self.feature_dim is not None:
            combined = self._resize_vector(combined, self.feature_dim)
        
        return combined
    
    def encode_multi_view(self, node, view_methods: Dict[str, str] = None) -> Dict[str, np.ndarray]:
        """分块编码 (Multi-View Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        view_methods: Dict[str, str]
            各个视图的编码方法
            
        Returns
        -------
        Dict[str, np.ndarray]
            各个视图的特征向量
        """
        if view_methods is None:
            view_methods = {
                'operator': 'embedding',
                'stats': 'normalize',
                'predicate': 'complexity',
                'relation': 'features'
            }
        
        views = {}
        
        if 'operator' in view_methods:
            views['operator'] = self.encode_operator_type(node, method=view_methods['operator'])
        
        if 'stats' in view_methods:
            views['stats'] = self.encode_estimated_stats(node, normalize=True)
        
        if 'predicate' in view_methods:
            views['predicate'] = self.encode_predicate_info(node, method=view_methods['predicate'])
        
        if 'relation' in view_methods:
            views['relation'] = self.encode_relation_info(node, method=view_methods['relation'])
        
        if 'context' in view_methods:
            views['context'] = self.encode_execution_context(node)
        
        if 'historical' in view_methods:
            views['historical'] = self.encode_historical_feedback(node)
        
        return views
    
    def encode_knowledge_enhanced(self, node, include_cost_model: bool = True) -> np.ndarray:
        """知识增强编码 (Knowledge Enhanced Encoding)
        
        Parameters
        ----------
        node: PlanNode
            查询计划节点
        include_cost_model: bool
            是否包含传统代价模型的输出
            
        Returns
        -------
        np.ndarray
            知识增强的特征向量
        """
        # 基础特征
        base_features = self.encode_concatenation_mlp(node, ['operator', 'stats', 'context'])
        
        enhanced_features = [base_features]
        
        if include_cost_model:
            # 传统代价模型特征
            extra_info = getattr(node, 'extra_info', {})
            cost_features = []
            
            # 代价模型输出
            startup_cost = extra_info.get('Startup Cost', 0.0)
            total_cost = extra_info.get('Total Cost', 0.0)
            
            cost_features.extend([float(startup_cost), float(total_cost)])
            
            # 选择性估计 (简化版)
            plan_rows = extra_info.get('Plan Rows', 1.0)
            if plan_rows > 0:
                selectivity = min(1.0, float(plan_rows) / 1000000)  # 假设基准表大小
            else:
                selectivity = 0.0
            cost_features.append(selectivity)
            
            # 代价比率
            if total_cost > 0 and startup_cost >= 0:
                cost_ratio = startup_cost / total_cost
                cost_features.append(cost_ratio)
            else:
                cost_features.append(0.0)
            
            cost_vec = np.array(cost_features, dtype=np.float32)
            enhanced_features.append(cost_vec)
        
        # 拼接所有增强特征
        combined = np.concatenate(enhanced_features)
        
        if self.feature_dim is not None:
            combined = self._resize_vector(combined, self.feature_dim)
        
        return combined
    
    # ===================================================================
    # 谓词编码辅助方法 (Predicate Encoding Helper Methods)
    # ===================================================================
    
    def _encode_predicate_complexity(self, predicates: List[str]) -> np.ndarray:
        """编码谓词复杂度特征"""
        complexity_features = []
        
        all_predicates = ' '.join(predicates).lower()
        
        # 谓词数量
        complexity_features.append(float(len(predicates)))
        
        # 是否有范围过滤
        range_patterns = ['>', '<', '>=', '<=', 'between', 'range']
        has_range = float(any(pattern in all_predicates for pattern in range_patterns))
        complexity_features.append(has_range)
        
        # 是否包含子查询
        subquery_patterns = ['exists', 'in (select', 'any', 'all', 'subplan']
        has_subquery = float(any(pattern in all_predicates for pattern in subquery_patterns))
        complexity_features.append(has_subquery)
        
        # 是否有函数调用
        function_patterns = ['(', 'upper', 'lower', 'substr', 'date', 'cast']
        has_function = float(any(pattern in all_predicates for pattern in function_patterns))
        complexity_features.append(has_function)
        
        # 是否有LIKE模式匹配
        has_like = float('like' in all_predicates or '%' in all_predicates)
        complexity_features.append(has_like)
        
        # 连接条件数量 (简化估计)
        join_patterns = ['=', 'join', 'on']
        join_count = sum(all_predicates.count(pattern) for pattern in join_patterns[:1])  # 只计算=
        complexity_features.append(float(join_count))
        
        return np.array(complexity_features, dtype=np.float32)
    
    def _encode_predicate_bow(self, predicates: List[str]) -> np.ndarray:
        """Bag-of-Words编码谓词"""
        # 简化的BoW实现
        all_text = ' '.join(predicates).lower()
        
        # 预定义的重要词汇
        vocab = [
            '=', '>', '<', '>=', '<=', 'like', 'in', 'exists', 'and', 'or', 'not',
            'between', 'is', 'null', 'true', 'false', 'cast', 'date', 'time',
            'upper', 'lower', 'substr', 'length', 'count', 'sum', 'avg', 'max', 'min'
        ]
        
        bow_features = []
        for word in vocab:
            count = all_text.count(word)
            bow_features.append(float(count))
        
        # 标准化
        bow_array = np.array(bow_features, dtype=np.float32)
        if bow_array.sum() > 0:
            bow_array = bow_array / bow_array.sum()
        
        # 调整到目标维度
        if len(bow_array) < self.predicate_embedding_dim:
            padded = np.zeros(self.predicate_embedding_dim, dtype=np.float32)
            padded[:len(bow_array)] = bow_array
            return padded
        else:
            return bow_array[:self.predicate_embedding_dim]
    
    def _encode_predicate_tokens(self, predicates: List[str]) -> np.ndarray:
        """Token embedding编码谓词"""
        # 简化的token embedding
        all_text = ' '.join(predicates).lower()
        
        # 提取tokens
        tokens = re.findall(r'\w+', all_text)
        
        if not tokens:
            return np.zeros(self.predicate_embedding_dim, dtype=np.float32)
        
        # 简单的token embedding (使用hash)
        token_embeddings = []
        for token in tokens[:10]:  # 限制token数量
            hash_val = hash(token) % self.predicate_embedding_dim
            embedding = np.zeros(self.predicate_embedding_dim, dtype=np.float32)
            embedding[hash_val] = 1.0
            token_embeddings.append(embedding)
        
        if token_embeddings:
            # 平均池化
            avg_embedding = np.mean(token_embeddings, axis=0)
            return avg_embedding / np.linalg.norm(avg_embedding) if np.linalg.norm(avg_embedding) > 0 else avg_embedding
        else:
            return np.zeros(self.predicate_embedding_dim, dtype=np.float32)
    
    def _create_relation_embedding(self, relation_name: str) -> np.ndarray:
        """创建关系名embedding"""
        # 使用hash创建确定性embedding
        embedding = np.zeros(self.relation_embedding_dim, dtype=np.float32)
        
        hash_val = hash(relation_name) % self.relation_embedding_dim
        embedding[hash_val] = 1.0
        
        # 添加一些变化
        for i in range(min(2, self.relation_embedding_dim)):
            pos = (hash_val + i + 1) % self.relation_embedding_dim
            embedding[pos] = 0.5
        
        return embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
    
    def extract_node_features(self, node) -> NodeFeatures:
        """Extract comprehensive features from a single node.
        
        Parameters
        ----------
        node:
            PlanNode object to extract features from
            
        Returns
        -------
        NodeFeatures:
            Container with different types of extracted features
        """
        # Extract different types of features
        idx = self._ensure_index(getattr(node, "node_type", "Unknown"))
        node_type_vec = self._one_hot(idx)
        numerical_vec = self._extract_numerical_features(node)
        categorical_vec = self._extract_categorical_features(node)
        
        # Extract new feature types
        predicate_vec = self.encode_predicate_info(node, method='complexity')
        relation_vec = self.encode_relation_info(node, method='features')
        context_vec = self.encode_execution_context(node)
        historical_vec = self.encode_historical_feedback(node)
        
        # Combine features
        feature_components = [node_type_vec]
        if len(numerical_vec) > 0:
            feature_components.append(numerical_vec)
        if len(categorical_vec) > 0:
            feature_components.append(categorical_vec)
        if len(predicate_vec) > 0:
            feature_components.append(predicate_vec)
        if len(relation_vec) > 0:
            feature_components.append(relation_vec)
        if len(context_vec) > 0:
            feature_components.append(context_vec)
        if len(historical_vec) > 0:
            feature_components.append(historical_vec)
        
        combined_vec = np.concatenate(feature_components)
        if self.feature_dim is not None:
            combined_vec = self._resize_vector(combined_vec, self.feature_dim)
        
        return NodeFeatures(
            node_type_vector=node_type_vec,
            numerical_features=numerical_vec,
            categorical_features=categorical_vec,
            predicate_features=predicate_vec,
            relation_features=relation_vec,
            execution_context_features=context_vec,
            historical_features=historical_vec,
            combined_features=combined_vec
        )
    
    # ---------------------------------------------------------------- internal ---
    def _normalize_numerical(self, features: np.ndarray) -> np.ndarray:
        """Normalize numerical features using min-max scaling."""
        if len(features) == 0:
            return features
        
        # Simple normalization - clip outliers and scale
        features = np.clip(features, 0, np.percentile(features, 95) if np.any(features > 0) else 1)
        max_val = np.maximum(features.max(), 1.0)
        return features / max_val
    
    def _resize_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize vector to target dimension."""
        if len(vector) == target_dim:
            return vector
        elif len(vector) < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:len(vector)] = vector
            return padded
        else:
            # Truncate
            return vector[:target_dim]
    
    # -------------------------------------------------------------- configuration ---
    def enable_rich_features(self, feature_dim: Optional[int] = None):
        """Enable rich feature extraction."""
        self.rich_features = True
        if feature_dim is not None:
            self.feature_dim = feature_dim
    
    def disable_rich_features(self):
        """Disable rich features (only node type)."""
        self.rich_features = False
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about current feature configuration."""
        return {
            'rich_features': self.rich_features,
            'feature_dim': self.feature_dim,
            'include_numerical': self.include_numerical,
            'include_categorical': self.include_categorical,
            'node_types_count': len(self.node_index),
            'categorical_vocabs': {k: len(v) for k, v in self.categorical_vocabs.items()}
        }
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get the sizes of different vocabularies."""
        vocab_sizes = {
            'node_type': len(self.node_index)
        }
        
        for key, vocab in self.categorical_vocabs.items():
            vocab_sizes[f'categorical_{key}'] = len(vocab)
        
        return vocab_sizes


# ===================================================================
# 便利工厂函数 (Convenience Factory Functions)
# ===================================================================

def create_simple_node_encoder() -> NodeEncoder:
    """Create a simple node encoder (only node type)."""
    return NodeEncoder(rich_features=False)


def create_rich_node_encoder(feature_dim: int = 64, 
                           include_numerical: bool = True,
                           include_categorical: bool = True,
                           normalize_features: bool = True) -> NodeEncoder:
    """Create a rich node encoder with comprehensive features."""
    return NodeEncoder(
        rich_features=True,
        feature_dim=feature_dim,
        include_numerical=include_numerical,
        include_categorical=include_categorical,
        normalize_features=normalize_features
    )


def create_concatenation_encoder(components: List[str] = None, 
                               feature_dim: int = 128) -> NodeEncoder:
    """创建简单拼接编码器
    
    Parameters
    ----------
    components: List[str]
        要包含的特征组件: ['operator', 'stats', 'predicate', 'relation', 'context', 'historical']
    feature_dim: int
        目标特征维度
        
    Returns
    -------
    NodeEncoder
        配置好的拼接编码器
    """
    encoder = NodeEncoder(rich_features=True, feature_dim=feature_dim)
    encoder._encoding_method = 'concatenation'
    encoder._encoding_components = components or ['operator', 'stats', 'context']
    return encoder


def create_multi_view_encoder(view_methods: Dict[str, str] = None,
                            feature_dim: int = 128) -> NodeEncoder:
    """创建分块编码器
    
    Parameters
    ----------
    view_methods: Dict[str, str]
        各个视图的编码方法
    feature_dim: int
        目标特征维度
        
    Returns
    -------
    NodeEncoder
        配置好的分块编码器
    """
    encoder = NodeEncoder(rich_features=True, feature_dim=feature_dim)
    encoder._encoding_method = 'multi_view'
    encoder._view_methods = view_methods or {
        'operator': 'embedding',
        'stats': 'normalize',
        'predicate': 'complexity',
        'relation': 'features'
    }
    return encoder


def create_knowledge_enhanced_encoder(include_cost_model: bool = True,
                                   feature_dim: int = 128) -> NodeEncoder:
    """创建知识增强编码器
    
    Parameters
    ----------
    include_cost_model: bool
        是否包含传统代价模型特征
    feature_dim: int
        目标特征维度
        
    Returns
    -------
    NodeEncoder
        配置好的知识增强编码器
    """
    encoder = NodeEncoder(rich_features=True, feature_dim=feature_dim)
    encoder._encoding_method = 'knowledge_enhanced'
    encoder._include_cost_model = include_cost_model
    return encoder


def create_predicate_focused_encoder(predicate_method: str = 'complexity',
                                   feature_dim: int = 96) -> NodeEncoder:
    """创建谓词重点编码器
    
    Parameters
    ----------
    predicate_method: str
        谓词编码方法: 'complexity', 'bow', 'token_embedding'
    feature_dim: int
        目标特征维度
        
    Returns
    -------
    NodeEncoder
        配置好的谓词重点编码器
    """
    encoder = NodeEncoder(rich_features=True, feature_dim=feature_dim)
    encoder._encoding_method = 'predicate_focused'
    encoder._predicate_method = predicate_method
    return encoder


def create_historical_aware_encoder(normalize_historical: bool = True,
                                  feature_dim: int = 128) -> NodeEncoder:
    """创建历史感知编码器
    
    Parameters
    ----------
    normalize_historical: bool
        是否标准化历史特征
    feature_dim: int
        目标特征维度
        
    Returns
    -------
    NodeEncoder
        配置好的历史感知编码器
    """
    encoder = NodeEncoder(rich_features=True, feature_dim=feature_dim)
    encoder._encoding_method = 'historical_aware'
    encoder._normalize_historical = normalize_historical
    return encoder


def create_custom_encoder(operator_method: str = 'one_hot',
                        stats_normalize: bool = True,
                        predicate_method: str = 'complexity',
                        relation_method: str = 'features',
                        include_context: bool = True,
                        include_historical: bool = False,
                        feature_dim: int = 128) -> NodeEncoder:
    """创建自定义编码器
    
    Parameters
    ----------
    operator_method: str
        算子编码方法: 'one_hot', 'embedding', 'learned_embedding'
    stats_normalize: bool
        是否标准化统计特征
    predicate_method: str
        谓词编码方法: 'complexity', 'bow', 'token_embedding'
    relation_method: str
        关系编码方法: 'embedding', 'one_hot', 'features'
    include_context: bool
        是否包含执行上下文特征
    include_historical: bool
        是否包含历史特征
    feature_dim: int
        目标特征维度
        
    Returns
    -------
    NodeEncoder
        配置好的自定义编码器
    """
    encoder = NodeEncoder(rich_features=True, feature_dim=feature_dim)
    encoder._encoding_method = 'custom'
    encoder._custom_config = {
        'operator_method': operator_method,
        'stats_normalize': stats_normalize,
        'predicate_method': predicate_method,
        'relation_method': relation_method,
        'include_context': include_context,
        'include_historical': include_historical
    }
    return encoder
