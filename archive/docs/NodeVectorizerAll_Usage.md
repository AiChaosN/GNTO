# NodeVectorizerAll 使用指南

## 概述

`NodeVectorizerAll` 是一个完整的节点向量化器，能够处理PostgreSQL查询计划DataFrame中的所有39个特征，将它们转换为58维的向量表示。相比原始的`NodeVectorizer`只使用3个特征（16维），`NodeVectorizerAll`提供了更全面的特征表示。

## 特征对比

### 原始 NodeVectorizer (16维)
- Node Type (13维): one-hot编码
- Parallel Aware (2维): True/False one-hot编码  
- Plan Rows (1维): 归一化的行数

### NodeVectorizerAll (58维)
包含以下特征类别：

1. **Node Type (13维)**: PostgreSQL节点类型的one-hot编码
2. **Parallel Aware (1维)**: 是否并行感知 (0/1)
3. **Relation Name (1维)**: 是否有关系名 (0/1)
4. **Alias (1维)**: 是否有别名 (0/1)
5. **数值特征 (21维)**: 包括各种成本、时间、行数等数值特征，经过归一化处理
6. **Parent Relationship (3维)**: 父子关系类型 (Outer/Inner/SubPlan)
7. **Index Name (1维)**: 是否有索引名 (0/1)
8. **Single Copy (1维)**: 是否单副本 (0/1)
9. **Join Type (6维)**: 连接类型 (Inner/Left/Right/Full/Semi/Anti)
10. **Inner Unique (1维)**: 内连接是否唯一 (0/1)
11. **Scan Direction (3维)**: 扫描方向 (Forward/Backward/NoMovement)
12. **条件存在性 (6维)**: 各种条件是否存在 (Recheck Cond/Index Cond/Hash Cond/Filter/Join Filter/Merge Cond)

## 使用方法

### 基本使用

```python
from models.NodeVectorizerAll import NodeVectorizerAll, create_enhanced_vectorizer_all
import pandas as pd

# 假设你已经有了plans_df (从graphs_to_df得到的DataFrame)
# 使用便捷函数
result, vectorizer = create_enhanced_vectorizer_all(plans_df)

print(f"特征维度: {result['feature_dim']}")  # 58
print(f"向量化的计划数: {len(result['vectors'])}")
print(f"特征名称: {result['feature_names'][:10]}...")

# 获取向量化结果
vectorized_plans = result['vectors']  # List[List[List[float]]]
```

### 详细使用

```python
# 直接使用NodeVectorizerAll类
vectorizer = NodeVectorizerAll()

# 向量化DataFrame
result = vectorizer.vectorize_dataframe(plans_df)

# 获取统计信息
stats = vectorizer.get_statistics(plans_df)
print(f"总节点数: {stats['total_nodes']}")
print(f"总计划数: {stats['total_plans']}")
print(f"平均每计划节点数: {stats['avg_nodes_per_plan']:.2f}")
print(f"节点类型分布: {stats['node_type_distribution']}")
```

### 在训练流程中使用

```python
# 在enhanced_training_example.py中的使用方式
def main():
    # ... 数据预处理 ...
    
    # 选择向量化方式
    use_full_vectorizer = True  # True: 使用58维完整向量化，False: 使用16维原始向量化
    
    if use_full_vectorizer:
        print("使用NodeVectorizerAll进行完整向量化...")
        vectorizer_result, vectorizer = create_enhanced_vectorizer_all(plans_df)
        res = vectorizer_result['vectors']
        F_num = vectorizer_result['feature_dim']  # 58
    else:
        print("使用原始NodeVectorizer...")
        res = NodeVectorizer(new_matrix_plans, node_type_mapping)
        F_num = 16
    
    # 创建模型
    node_encoder = NodeEncoderFactory.create_enhanced_encoder(
        in_dim=F_num,  # 16 或 58
        d_node=32,
        use_attention=True,
        use_residual=True,
        dropout=0.1
    )
    
    # ... 继续训练流程 ...
```

## 特征详细说明

### 数值特征归一化

以下数值特征会被归一化到[0,1]范围：

```python
normalization_params = {
    'Plan Rows': 2e8,              # 计划行数
    'Plan Width': 1000,            # 计划宽度
    'Startup Cost': 1e6,           # 启动成本
    'Total Cost': 1e6,             # 总成本
    'Actual Startup Time': 1e4,    # 实际启动时间
    'Actual Total Time': 1e4,      # 实际总时间
    'Actual Rows': 2e8,            # 实际行数
    'Actual Loops': 1000,          # 实际循环次数
    'Workers Planned': 20,         # 计划工作者数
    'Workers Launched': 20,        # 启动工作者数
    'Hash Buckets': 1e6,           # 哈希桶数
    'Peak Memory Usage': 1e6,      # 峰值内存使用 (KB)
    # ... 等等
}
```

### 类别特征编码

类别特征使用one-hot编码：

```python
# 节点类型 (13维)
node_types = [
    'Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 
    'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 
    'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort'
]

# 连接类型 (6维)
join_types = ['Inner', 'Left', 'Right', 'Full', 'Semi', 'Anti']

# 扫描方向 (3维)
scan_directions = ['Forward', 'Backward', 'NoMovement']
```

## 性能对比

### 特征维度对比

| 向量化方法 | 特征维度 | 包含特征 | 信息丰富度 |
|------------|----------|----------|------------|
| 原始NodeVectorizer | 16 | 节点类型、并行、行数 | 基础 |
| NodeVectorizerAll | 58 | 所有DataFrame特征 | 完整 |

### 模型参数对比

使用相同的NodeEncoder设置（d_node=32）：

| 编码器类型 | 输入维度 | 参数数量 | 说明 |
|------------|----------|----------|------|
| Mini_16 | 16 | 608 | 最少参数 |
| Mini_58 | 58 | 1,952 | 参数增加3倍 |
| Enhanced_16 | 16 | 8,296 | 带注意力机制 |
| Enhanced_58 | 58 | 12,328 | 完整特征+注意力 |

### 预期效果

使用NodeVectorizerAll的优势：

1. **信息更丰富**: 包含所有查询计划特征，不丢失信息
2. **更好的表示能力**: 58维特征能够捕获更多查询计划的细节
3. **更高的预测精度**: 理论上应该能够提高模型的预测准确性
4. **更好的泛化能力**: 完整的特征表示有助于模型泛化

代价：
1. **计算成本增加**: 特征维度增加带来的计算开销
2. **内存使用增加**: 更多的参数和特征存储需求
3. **可能的过拟合风险**: 特征过多可能导致过拟合

## 最佳实践

### 1. 特征选择建议

```python
# 对于小数据集，可以选择性使用特征
use_full_vectorizer = len(dataset) > 10000  # 大数据集使用完整特征

# 对于计算资源受限的环境
if limited_resources:
    use_full_vectorizer = False
    F_num = 16
else:
    use_full_vectorizer = True
    F_num = 58
```

### 2. 模型配置建议

```python
# 根据特征维度调整模型配置
if F_num == 58:
    # 完整特征时，可以使用更大的模型
    d_node = 64
    use_attention = True
    dropout = 0.2  # 增加dropout防止过拟合
else:
    # 简单特征时，使用较小的模型
    d_node = 32
    use_attention = False
    dropout = 0.1
```

### 3. 训练策略

```python
# 完整特征时的训练配置
if F_num == 58:
    learning_rate = 0.0005  # 降低学习率
    batch_size = 16         # 减少批次大小
    patience = 20           # 增加早停耐心
else:
    learning_rate = 0.001
    batch_size = 32
    patience = 15
```

## 实验建议

建议进行以下对比实验：

1. **特征维度对比**: 16维 vs 58维特征的效果对比
2. **模型复杂度对比**: 不同NodeEncoder在不同特征维度下的表现
3. **训练效率对比**: 训练时间、收敛速度、最终精度的权衡
4. **泛化能力对比**: 在不同测试集上的表现

## 总结

`NodeVectorizerAll`提供了完整的查询计划特征向量化能力，是对原始`NodeVectorizer`的重要扩展。在使用时需要根据数据规模、计算资源和精度要求来选择合适的向量化方法。
