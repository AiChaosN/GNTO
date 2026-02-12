# 简化版 NodeEncoder 设计

## 🎯 设计理念

**分块编码 (Multi-View Encoding)**: 每类特征单独编码，最后 concat。

- ✅ **专门化**: 每种特征用最适合的编码方式
- ✅ **简洁**: 代码清晰，易于维护  
- ✅ **高效**: 避免复杂的特征交互计算
- ✅ **可扩展**: 容易添加新的特征块

## 🏗️ 架构设计

### 输入
```python
PlanNode:
  ├── node_type: "Hash Join"
  └── extra_info: {统计信息, 谓词信息, ...}
```

### 分块编码
```python
1. 🎯 算子类型 → Embedding Layer → [32维]
2. 📈 数据统计 → MLP (log标准化+全连接) → [16维]  
3. 🔍 谓词信息 → Simple Encoder (复杂度特征) → [8维]
```

### 特征融合
```python
Concat([32, 16, 8]) → Linear Projection → [64维]
```

## 📝 核心实现

### 1. 算子类型编码
```python
def _encode_operator(self, node) -> torch.Tensor:
    # 动态扩展词汇表 + Embedding层
    node_type = getattr(node, "node_type", "Unknown")
    idx = self.node_type_vocab[node_type] 
    return self.operator_embedding(torch.tensor([idx])).squeeze(0)
```

### 2. 数据统计编码  
```python
def _encode_stats(self, node) -> torch.Tensor:
    # 提取: Plan Rows, Plan Width, Startup Cost, Total Cost
    # 处理: log1p标准化 → MLP
    stats_tensor = torch.log1p(torch.tensor(stats_values))
    return self.stats_mlp(stats_tensor)
```

### 3. 谓词信息编码
```python
def _encode_predicate(self, node) -> torch.Tensor:
    # 复杂度特征: 谓词数量、范围过滤、子查询、函数调用等
    # 返回固定维度的特征向量
    return torch.tensor(complexity_features)
```

## 🚀 使用方式

### 基本使用
```python
# 创建编码器
encoder = create_node_encoder(
    operator_dim=32,
    stats_dim=16, 
    predicate_dim=8,
    output_dim=64
)

# 编码节点
vector = encoder.encode_node(node)  # torch.Tensor [64]
```

### 工厂函数
```python
# 简单版 (小维度)
encoder = create_simple_node_encoder()  # 输出32维

# 标准版
encoder = create_node_encoder()         # 输出64维

# 大容量版 (大维度)  
encoder = create_large_node_encoder()   # 输出128维
```

## 💡 关键特性

### 1. 动态词汇表扩展
- 自动处理新的算子类型
- Embedding层动态扩展，保持已学习权重

### 2. 简洁的特征处理
- 统计特征: log标准化 + MLP
- 谓词特征: 6个复杂度指标
- 无冗余编码方法

### 3. PyTorch原生支持
- 继承自 `nn.Module`
- 支持梯度传播和训练
- 返回 `torch.Tensor`

## 📊 性能对比

| 特性 | 简化版 | 原复杂版 |
|------|--------|----------|
| 代码行数 | ~280行 | ~1150行 |
| 编码方法数 | 3个核心方法 | 15+个方法 |
| 工厂函数 | 3个 | 8个 |
| 维护复杂度 | 低 | 高 |
| 功能完整性 | ✅ 核心功能完整 | ❌ 功能重复冗余 |

## 🔧 扩展指南

如需添加新特征块:

```python
def _encode_new_feature(self, node) -> torch.Tensor:
    # 实现新特征的编码逻辑
    return feature_vector

def forward(self, node) -> torch.Tensor:
    # 在concat中添加新特征
    new_vec = self._encode_new_feature(node)
    combined = torch.cat([operator_vec, stats_vec, predicate_vec, new_vec])
    return self.output_projection(combined)
```

## ✅ 总结

简化版 NodeEncoder 实现了**分块编码**的核心思想:
- 🎯 **算子embedding** + 📈 **统计MLP** + 🔍 **谓词encoder** + 🔗 **concat**
- 代码简洁，功能完整，易于维护和扩展
- 完全符合你的需求: "只保留分块编码，每类特征单独编码，最后concat"
