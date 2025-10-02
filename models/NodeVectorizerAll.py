"""
NodeVectorizerAll - 完整的节点向量化器
支持DataFrame中所有39个特征的向量化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
import torch
import warnings

class NodeVectorizerAll:
    """
    完整的节点向量化器，支持所有DataFrame特征
    """
    
    def __init__(self):
        # PostgreSQL查询计划中的所有节点类型
        self.node_types = [
            'Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 
            'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 
            'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort'
        ]
        
        # 父子关系类型
        self.parent_relationships = ['Outer', 'Inner', 'SubPlan']
        
        # Join类型
        self.join_types = ['Inner', 'Left', 'Right', 'Full', 'Semi', 'Anti']
        
        # 扫描方向
        self.scan_directions = ['Forward', 'Backward', 'NoMovement']
        
        # 归一化参数（基于经验设置，可以根据实际数据调整）
        self.normalization_params = {
            'Plan Rows': 2e8,
            'Plan Width': 1000,
            'Startup Cost': 1e6,
            'Total Cost': 1e6,
            'Actual Startup Time': 1e4,
            'Actual Total Time': 1e4,
            'Actual Rows': 2e8,
            'Actual Loops': 1000,
            'Rows Removed by Index Recheck': 1e6,
            'Exact Heap Blocks': 1e6,
            'Lossy Heap Blocks': 1e6,
            'Workers Planned': 20,
            'Workers Launched': 20,
            'Workers': 20,
            'Hash Buckets': 1e6,
            'Original Hash Buckets': 1e6,
            'Hash Batches': 100,
            'Original Hash Batches': 100,
            'Peak Memory Usage': 1e6,  # KB
            'Rows Removed by Filter': 1e6,
            'Rows Removed by Join Filter': 1e6,
        }
        
        # 特征维度计算
        self._calculate_feature_dims()
    
    def _calculate_feature_dims(self):
        """计算各类特征的维度"""
        self.dims = {
            'node_type': len(self.node_types),  # 13
            'parallel_aware': 1,  # 0/1
            'relation_name': 1,  # 有无关系名 0/1
            'alias': 1,  # 有无别名 0/1
            'numerical': len(self.normalization_params),  # 22个数值特征
            'parent_relationship': len(self.parent_relationships),  # 3
            'index_name': 1,  # 有无索引名 0/1
            'single_copy': 1,  # 0/1
            'join_type': len(self.join_types),  # 6
            'inner_unique': 1,  # 0/1
            'scan_direction': len(self.scan_directions),  # 3
            'conditions': 6,  # 6种条件类型，每种用0/1表示是否存在
        }
        
        # 总维度
        self.total_dim = sum(self.dims.values())
        print(f"NodeVectorizerAll总特征维度: {self.total_dim}")
        print(f"特征维度分布: {self.dims}")
    
    def _safe_normalize(self, value: Any, max_val: float) -> float:
        """安全的归一化函数"""
        if value is None:
            return 0.0
        
        # 处理数组或Series类型
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            return 0.0
            
        # 处理pandas的NA值
        try:
            if pd.isna(value):
                return 0.0
        except (ValueError, TypeError):
            # 如果pd.isna失败，继续尝试转换
            pass
            
        try:
            val = float(value)
            return min(val / max_val, 1.0)  # 限制最大值为1
        except (ValueError, TypeError):
            return 0.0
    
    def _encode_categorical(self, value: Any, categories: List[str]) -> List[float]:
        """编码类别特征为one-hot"""
        encoding = [0.0] * len(categories)
        try:
            if pd.notna(value) and value in categories:
                encoding[categories.index(value)] = 1.0
        except (ValueError, TypeError):
            # 如果pd.notna失败，跳过编码
            pass
        return encoding
    
    def _encode_binary(self, value: Any) -> float:
        """编码二进制特征"""
        if value is None:
            return 0.0
        
        try:
            if pd.isna(value):
                return 0.0
        except (ValueError, TypeError):
            pass
            
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            return 1.0 if value.strip() else 0.0
        return 1.0 if value else 0.0
    
    def _encode_existence(self, value: Any) -> float:
        """编码存在性特征（有值为1，无值为0）"""
        if value is None:
            return 0.0
        
        try:
            if pd.notna(value) and str(value).strip():
                return 1.0
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def vectorize_node(self, node_row: pd.Series) -> List[float]:
        """
        将单个节点（DataFrame的一行）向量化
        """
        vector = []
        
        # 1. Node Type (one-hot encoding) - 13维
        node_type = node_row.get('Node Type', '')
        vector.extend(self._encode_categorical(node_type, self.node_types))
        
        # 2. Parallel Aware - 1维
        vector.append(self._encode_binary(node_row.get('Parallel Aware', False)))
        
        # 3. Relation Name存在性 - 1维
        vector.append(self._encode_existence(node_row.get('Relation Name', '')))
        
        # 4. Alias存在性 - 1维
        vector.append(self._encode_existence(node_row.get('Alias', '')))
        
        # 5. 数值特征 - 22维
        for feature, max_val in self.normalization_params.items():
            vector.append(self._safe_normalize(node_row.get(feature, 0), max_val))
        
        # 6. Parent Relationship - 3维
        parent_rel = node_row.get('Parent Relationship', '')
        vector.extend(self._encode_categorical(parent_rel, self.parent_relationships))
        
        # 7. Index Name存在性 - 1维
        vector.append(self._encode_existence(node_row.get('Index Name', '')))
        
        # 8. Single Copy - 1维
        vector.append(self._encode_binary(node_row.get('Single Copy', False)))
        
        # 9. Join Type - 6维
        join_type = node_row.get('Join Type', '')
        vector.extend(self._encode_categorical(join_type, self.join_types))
        
        # 10. Inner Unique - 1维
        vector.append(self._encode_binary(node_row.get('Inner Unique', False)))
        
        # 11. Scan Direction - 3维
        scan_dir = node_row.get('Scan Direction', '')
        vector.extend(self._encode_categorical(scan_dir, self.scan_directions))
        
        # 12. 条件存在性 - 6维
        conditions = [
            'Recheck Cond', 'Index Cond', 'Hash Cond', 
            'Filter', 'Join Filter', 'Merge Cond'
        ]
        for cond in conditions:
            vector.append(self._encode_existence(node_row.get(cond, '')))
        
        # 之后可以做:
        # 相关表存在: 根据各个表进行Embedding
        # 相关筛选attribute: 根据各个attribute进行Embedding
        
        return vector
    
    def vectorize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        向量化整个DataFrame
        
        Returns:
            dict: 包含向量化结果和元信息
        """
        # 按plan_id分组处理
        vectorized_plans = []
        plan_ids = []
        
        for plan_id, plan_group in df.groupby('plan_id'):
            plan_vectors = []
            
            # 按node_idx排序确保节点顺序正确
            plan_group = plan_group.sort_values('node_idx')
            
            for _, node_row in plan_group.iterrows():
                node_vector = self.vectorize_node(node_row)
                plan_vectors.append(node_vector)
            
            vectorized_plans.append(plan_vectors)
            plan_ids.append(plan_id)
        
        return {
            'vectors': vectorized_plans,
            'plan_ids': plan_ids,
            'feature_dim': self.total_dim,
            'feature_names': self._get_feature_names(),
            'normalization_params': self.normalization_params
        }
    
    def _get_feature_names(self) -> List[str]:
        """获取所有特征的名称"""
        names = []
        
        # Node Type
        names.extend([f"NodeType_{nt}" for nt in self.node_types])
        
        # Binary features
        names.extend(['ParallelAware', 'HasRelationName', 'HasAlias'])
        
        # Numerical features
        names.extend(list(self.normalization_params.keys()))
        
        # Parent Relationship
        names.extend([f"ParentRel_{pr}" for pr in self.parent_relationships])
        
        # Other categorical/binary features
        names.append('HasIndexName')
        names.append('SingleCopy')
        names.extend([f"JoinType_{jt}" for jt in self.join_types])
        names.append('InnerUnique')
        names.extend([f"ScanDir_{sd}" for sd in self.scan_directions])
        
        # Conditions
        conditions = ['Recheck Cond', 'Index Cond', 'Hash Cond', 
                     'Filter', 'Join Filter', 'Merge Cond']
        names.extend([f"Has{cond.replace(' ', '')}" for cond in conditions])
        
        return names
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'total_nodes': len(df),
            'total_plans': df['plan_id'].nunique(),
            'avg_nodes_per_plan': len(df) / df['plan_id'].nunique(),
            'node_type_distribution': df['Node Type'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # 数值特征统计
        numerical_stats = {}
        for feature in self.normalization_params.keys():
            if feature in df.columns:
                col_data = df[feature].dropna()
                if len(col_data) > 0:
                    try:
                        # 尝试转换为数值类型，过滤掉非数值数据
                        numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            numerical_stats[feature] = {
                                'min': float(numeric_data.min()),
                                'max': float(numeric_data.max()),
                                'mean': float(numeric_data.mean()),
                                'std': float(numeric_data.std())
                            }
                    except (ValueError, TypeError):
                        # 如果转换失败，跳过这个特征
                        continue
        
        stats['numerical_stats'] = numerical_stats
        return stats

def create_enhanced_vectorizer_all(df: pd.DataFrame) -> tuple:
    """
    创建增强版的完整向量化器
    
    Args:
        df: 包含所有特征的DataFrame
    
    Returns:
        tuple: (vectorized_data, vectorizer)
    """
    vectorizer = NodeVectorizerAll()
    
    print("开始向量化DataFrame...")
    print(f"DataFrame形状: {df.shape}")
    print(f"包含的计划数: {df['plan_id'].nunique()}")
    
    # 向量化
    result = vectorizer.vectorize_dataframe(df)
    
    print(f"向量化完成!")
    print(f"特征维度: {result['feature_dim']}")
    print(f"向量化的计划数: {len(result['vectors'])}")
    
    # 统计信息
    stats = vectorizer.get_statistics(df)
    print(f"平均每个计划的节点数: {stats['avg_nodes_per_plan']:.2f}")
    
    return result, vectorizer

# 便捷函数：兼容原有接口
def NodeVectorizerAll_compat(matrix_plans: List[List[dict]], node_type_mapping: Dict[str, int]) -> List[List[List]]:
    """
    兼容原有接口的向量化函数
    将matrix_plans转换为DataFrame后进行完整向量化
    """
    # 转换为DataFrame格式
    rows = []
    for pid, plan in enumerate(matrix_plans):
        for nid, node in enumerate(plan):
            rows.append({"plan_id": pid, "node_idx": nid, **node})
    
    df = pd.DataFrame(rows)
    
    # 使用完整向量化器
    vectorizer = NodeVectorizerAll()
    result = vectorizer.vectorize_dataframe(df)
    
    return result['vectors']

if __name__ == "__main__":
    # 测试代码
    print("NodeVectorizerAll测试")
    
    # 创建示例数据
    sample_data = {
        'plan_id': [0, 0, 1, 1],
        'node_idx': [0, 1, 0, 1],
        'Node Type': ['Hash Join', 'Seq Scan', 'Index Scan', 'Sort'],
        'Parallel Aware': [False, True, False, False],
        'Relation Name': ['table1', 'table2', '', 'table3'],
        'Alias': ['t1', 't2', '', 't3'],
        'Plan Rows': [1000, 5000, 100, 2000],
        'Startup Cost': [10.5, 0.0, 0.5, 15.2],
        'Total Cost': [100.5, 50.0, 5.5, 25.2],
        'Actual Rows': [950, 4800, 98, 1950],
        'Join Type': ['Inner', '', '', ''],
        'Filter': ['', 'id > 100', '', ''],
    }
    
    df = pd.DataFrame(sample_data)
    
    # 测试向量化
    result, vectorizer = create_enhanced_vectorizer_all(df)
    
    print(f"\n测试结果:")
    print(f"计划0的第一个节点向量长度: {len(result['vectors'][0][0])}")
    print(f"计划0的第一个节点向量前10维: {result['vectors'][0][0][:10]}")
