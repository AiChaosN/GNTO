# 架构设计

## 设计原理

GNTO框架采用模块化的pipeline设计，将查询计划性能预测任务分解为多个独立的、可组合的组件。这种设计允许每个组件专注于特定的功能，同时保持整体架构的灵活性和可扩展性。

## 核心架构

```
Input Plan (JSON)
       ↓
DataPreprocessor ──→ PlanNode Tree
       ↓
Encoder ──→ Numerical Vectors
       ↓  
TreeModel ──→ Aggregated Vector
       ↓
Predictioner ──→ Performance Score
```

## 技术特点

### 1. 模块化设计

**设计目标**：每个组件都可以独立开发、测试和替换

**实现方式**：
- 清晰的接口定义
- 最小化组件间依赖
- 支持依赖注入

**优势**：
- 便于单元测试
- 支持增量开发
- 易于维护和扩展

### 2. 自适应处理

#### 维度对齐
- **问题**：不同查询计划产生的向量维度可能不同
- **解决方案**：TreeModel自动padding到最大维度
- **实现**：使用numpy的零填充策略

#### 动态词汇表
- **问题**：事先无法确定所有可能的节点类型
- **解决方案**：Encoder动态构建节点类型索引
- **实现**：使用字典维护类型到索引的映射

#### 权重适配
- **问题**：预测权重维度可能与特征维度不匹配
- **解决方案**：Predictioner自动调整权重维度
- **策略**：不足时padding零，过多时截断

### 3. 树结构优化

#### 递归编码
```python
def encode(self, node):
    # 编码当前节点
    vec = self._one_hot(node.node_type)
    
    # 递归编码子节点
    if node.children:
        child_vecs = [self.encode(child) for child in node.children]
        vec = self._pad_and_sum([vec] + child_vecs)
    
    return vec
```

#### 结构保持
- 保留原始计划的层次结构
- 维护父子节点关系
- 支持任意深度的嵌套

#### 信息完整性
- 保存所有节点的额外信息
- 支持自定义属性扩展
- 不丢失原始数据

### 4. 性能优化

#### 向量化操作
- 大量使用NumPy的向量化计算
- 避免Python循环的性能开销
- 利用底层BLAS优化

#### 内存效率
- 适当的向量padding策略
- 及时释放中间结果
- 避免不必要的数据复制

#### 可重现性
- 支持随机种子设置
- 确保实验结果可重现
- 便于调试和验证

## 扩展性设计

### 接口抽象

每个组件都定义了清晰的接口，支持自定义实现：

```python
# 抽象接口示例
class BaseEncoder:
    def encode(self, node) -> np.ndarray:
        raise NotImplementedError
    
    def encode_all(self, nodes) -> List[np.ndarray]:
        return [self.encode(node) for node in nodes]
```

### 组件替换

支持在运行时替换任何组件：

```python
# 使用自定义组件
custom_encoder = MyCustomEncoder()
gnto = GNTO(encoder=custom_encoder)
```

### 功能扩展

#### 编码器扩展
- 添加数值特征编码
- 支持更复杂的图神经网络
- 集成预训练的嵌入

#### 聚合器扩展  
- 实现注意力机制
- 支持层次化聚合
- 添加图卷积操作

#### 预测器扩展
- 使用深度神经网络
- 支持多任务学习
- 集成不确定性估计

## 数据流分析

### 数据转换过程

1. **JSON → PlanNode**
   - 输入：嵌套字典结构
   - 输出：树形对象结构
   - 转换：递归解析和对象化

2. **PlanNode → Vector**
   - 输入：树形对象结构
   - 输出：数值向量
   - 转换：one-hot编码和递归聚合

3. **Vectors → Aggregated Vector**
   - 输入：向量列表
   - 输出：单一向量
   - 转换：padding和统计聚合

4. **Vector → Prediction**
   - 输入：特征向量
   - 输出：标量预测值
   - 转换：线性变换

### 维度变化追踪

```
JSON Plan
    ↓ (preprocessing)
PlanNode Tree (structured)
    ↓ (encoding)
Vector [d1] (d1 = number of unique node types)
    ↓ (aggregation)  
Vector [d2] (d2 = max dimension across all vectors)
    ↓ (prediction)
Scalar (performance prediction)
```

## 设计权衡

### 简单性 vs 功能性
- **选择**：优先保持简单性
- **理由**：便于理解和扩展
- **实现**：最小可行实现，支持后续扩展

### 性能 vs 灵活性
- **选择**：在关键路径优化性能，保持接口灵活
- **实现**：使用NumPy优化计算，保持模块化接口

### 通用性 vs 专用性
- **选择**：针对查询计划优化，保持一定通用性
- **实现**：专门的树结构处理，通用的向量操作

## 未来扩展方向

1. **深度学习集成**：支持PyTorch/TensorFlow后端
2. **图神经网络**：原生支持GNN架构
3. **多任务学习**：同时预测多个性能指标
4. **在线学习**：支持增量更新和适应
5. **分布式计算**：支持大规模数据处理
