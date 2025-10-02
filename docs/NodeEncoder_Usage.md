# NodeEncoder 使用指南

本文档介绍了如何在GNTO项目中使用增强版的NodeEncoder进行查询计划节点编码。

## 概述

NodeEncoder是查询计划成本预测模型中的重要组件，负责将查询计划中的节点信息编码为向量表示。我们提供了多种不同的NodeEncoder实现，适应不同的使用场景和性能要求。

## 可用的NodeEncoder类型

### 1. NodeEncoder_Mini
最简单的编码器，使用线性投影将输入特征映射到目标维度。

```python
from models import NodeEncoder_Mini

encoder = NodeEncoder_Mini(in_dim=16, d_node=32)
x = torch.randn(batch_size, 16)
encoded = encoder(x)  # 输出形状: [batch_size, 32]
```

**适用场景**：
- 快速原型开发
- 计算资源受限的环境
- 简单的特征编码需求

### 2. NodeEncoder_Enhanced
增强版编码器，支持注意力机制、残差连接等高级特性。

```python
from models import NodeEncoder_Enhanced

encoder = NodeEncoder_Enhanced(
    in_dim=16,
    d_node=64,
    use_attention=True,
    use_residual=True,
    dropout=0.1
)
x = torch.randn(batch_size, 16)
encoded = encoder(x)  # 输出形状: [batch_size, 64]
```

**特性**：
- 自注意力机制用于特征选择
- 残差连接提高训练稳定性
- 支持节点类型embedding
- 可配置的dropout和层归一化

**适用场景**：
- 需要高精度的预测任务
- 复杂的特征交互建模
- 有充足计算资源的环境

### 3. NodeEncoder_Vectorized
基于手工特征工程的编码器，直接处理查询计划节点字典。

```python
from models import NodeEncoder_Vectorized

node_types = ['Seq Scan', 'Index Scan', 'Hash Join', ...]
encoder = NodeEncoder_Vectorized(
    node_types=node_types,
    d_node=32,
    use_cost_features=True,
    use_parallel_feature=True
)

# 可以直接处理节点字典列表
sample_nodes = [
    {
        "Node Type": "Hash Join",
        "Parallel Aware": False,
        "Plan Rows": 1000000,
        "Startup Cost": 1000.0,
        "Total Cost": 5000.0
    },
    # ... 更多节点
]

encoded = encoder(sample_nodes)  # 输出形状: [len(sample_nodes), 32]
```

**特性**：
- 自动特征工程（one-hot编码、归一化等）
- 直接处理查询计划节点字典
- 可配置的特征选择
- 与现有数据预处理管道兼容

**适用场景**：
- 基于传统特征工程的方法
- 需要可解释性的模型
- 与现有代码库集成

### 4. NodeEncoder_Mixed
混合编码器，同时处理数值特征和类别特征。

```python
from models import NodeEncoder_Mixed

encoder = NodeEncoder_Mixed(
    num_in_dim=3,           # 数值特征维度
    cat_cardinalities=[10, 5],  # 类别特征的基数
    d_node=32
)

x_num = torch.randn(batch_size, 3)        # 数值特征
x_cat = torch.randint(0, 10, (batch_size, 2))  # 类别特征ID

encoded = encoder(x_num, x_cat)  # 输出形状: [batch_size, 32]
```

**特性**：
- 分别处理数值和类别特征
- 自动计算embedding维度
- 支持批标准化
- 灵活的特征组合

**适用场景**：
- 混合特征类型的数据
- 需要精细控制特征处理的场景
- 大规模类别特征的处理

## 使用工厂类创建编码器

为了简化编码器的创建和配置，我们提供了`NodeEncoderFactory`工厂类：

```python
from models import NodeEncoderFactory

# 方法1：使用预定义配置
config = NodeEncoderFactory.get_recommended_config(
    use_case="query_plan",
    performance_level="balanced"  # "fast", "balanced", "accurate"
)
encoder = NodeEncoderFactory.create_encoder_from_config(config)

# 方法2：直接创建特定类型
encoder = NodeEncoderFactory.create_enhanced_encoder(
    in_dim=16,
    d_node=64,
    use_attention=True
)

# 方法3：使用便捷函数
from models import create_query_plan_encoder

encoder_wrapper = create_query_plan_encoder(
    d_node=32,
    performance_level="accurate"
)
```

## 集成到训练流程

### 替换现有的NodeEncoder

如果你已经有使用`NodeEncoder_Mini`的代码：

```python
# 原来的代码
from models.NodeEncoder import NodeEncoder_Mini
encoder = NodeEncoder_Mini(in_dim=16, d_node=32)

# 升级到增强版
from models import NodeEncoderFactory
encoder = NodeEncoderFactory.create_enhanced_encoder(
    in_dim=16,
    d_node=32,
    use_attention=True,
    use_residual=True
)
```

### 完整的模型定义

```python
import torch.nn as nn
from models import NodeEncoderFactory, GATTreeEncoder, PredictionHead

class EnhancedPlanCostModel(nn.Module):
    def __init__(self, feature_dim, d_node=64, d_graph=64):
        super().__init__()
        
        # 创建增强版节点编码器
        self.node_encoder = NodeEncoderFactory.create_enhanced_encoder(
            in_dim=feature_dim,
            d_node=d_node,
            use_attention=True,
            use_residual=True,
            dropout=0.1
        )
        
        # 图编码器
        self.tree_encoder = GATTreeEncoder(
            input_dim=d_node,
            hidden_dim=128,
            output_dim=d_graph,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            pooling="mean"
        )
        
        # 预测头
        self.prediction_head = PredictionHead(d_graph, out_dim=1)
    
    def forward(self, data):
        x = self.node_encoder(data.x)
        g = self.tree_encoder(x, data.edge_index, data.batch)
        y = self.prediction_head(g)
        return y
```

## 性能对比

| 编码器类型 | 参数数量 | 推理速度 | 精度 | 内存使用 |
|-----------|----------|----------|------|----------|
| Mini      | 最少     | 最快     | 基础 | 最少     |
| Enhanced  | 中等     | 中等     | 高   | 中等     |
| Vectorized| 少       | 快       | 中等 | 少       |
| Mixed     | 中等     | 中等     | 高   | 中等     |

## 最佳实践

### 1. 选择合适的编码器类型

- **快速原型**：使用`NodeEncoder_Mini`
- **生产环境**：使用`NodeEncoder_Enhanced`或`NodeEncoder_Vectorized`
- **混合特征**：使用`NodeEncoder_Mixed`

### 2. 超参数调优

```python
# 对于大型数据集
encoder = NodeEncoderFactory.create_enhanced_encoder(
    in_dim=feature_dim,
    d_node=128,        # 增加输出维度
    use_attention=True,
    dropout=0.2        # 增加dropout防止过拟合
)

# 对于小型数据集
encoder = NodeEncoderFactory.create_enhanced_encoder(
    in_dim=feature_dim,
    d_node=32,         # 减少输出维度
    use_attention=False,
    dropout=0.1        # 减少dropout
)
```

### 3. 特征工程

对于`NodeEncoder_Vectorized`，确保包含关键特征：

```python
encoder = NodeEncoder_Vectorized(
    node_types=your_node_types,
    d_node=32,
    use_cost_features=True,      # 包含成本特征
    use_parallel_feature=True,   # 包含并行特征
    plan_rows_max=1e8           # 根据数据设置合适的最大值
)
```

## 故障排除

### 常见问题

1. **维度不匹配错误**
   ```python
   # 确保输入维度与编码器期望的维度匹配
   print(f"输入形状: {x.shape}")
   print(f"编码器期望输入维度: {encoder.in_dim}")
   ```

2. **内存不足**
   ```python
   # 减少批次大小或模型维度
   encoder = NodeEncoderFactory.create_enhanced_encoder(
       in_dim=feature_dim,
       d_node=32,  # 减少输出维度
       use_attention=False  # 关闭注意力机制
   )
   ```

3. **训练不稳定**
   ```python
   # 调整学习率和dropout
   encoder = NodeEncoderFactory.create_enhanced_encoder(
       in_dim=feature_dim,
       d_node=64,
       dropout=0.3,  # 增加dropout
       use_residual=True  # 启用残差连接
   )
   ```

## 示例代码

完整的使用示例请参考：
- `examples/node_encoder_usage.py` - 基础使用示例
- `examples/enhanced_training_example.py` - 完整训练流程示例

## 扩展开发

如果需要自定义NodeEncoder，可以继承基类并实现自己的逻辑：

```python
import torch.nn as nn

class CustomNodeEncoder(nn.Module):
    def __init__(self, in_dim, d_node):
        super().__init__()
        self.in_dim = in_dim
        self.d_node = d_node
        
        # 自定义网络结构
        self.layers = nn.Sequential(
            nn.Linear(in_dim, d_node * 2),
            nn.GELU(),  # 使用GELU激活函数
            nn.LayerNorm(d_node * 2),
            nn.Linear(d_node * 2, d_node)
        )
    
    def forward(self, x):
        return self.layers(x)
```

## 总结

新的NodeEncoder实现提供了更多的灵活性和更好的性能。根据你的具体需求选择合适的编码器类型，并使用工厂类简化创建过程。通过适当的超参数调优和特征工程，可以显著提升模型的预测精度。
