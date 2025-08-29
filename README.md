# GNTO - Graph Neural Tree Optimizer

GNTO是一个基于机器学习的查询优化框架，专注于查询计划的节点级编码和结构级建模。

## 项目概述

GNTO采用分层架构设计，将查询优化分解为多个独立的组件，每个组件都有清晰的职责分工。项目基于简化的NodeEncoder设计，提供高效、可维护的查询计划处理能力。

## 架构设计

### 核心组件

项目采用4层清晰的处理流程：

```
DataPreprocessor → NodeEncoder → TreeEncoder → PredictionHead
     (JSON)        (分块编码)      (统计聚合)      (线性预测)
```

#### 1. DataPreprocessor - 数据预处理
- **输入**: 原始JSON风格的查询计划
- **输出**: TreeNode结构化数据
- **功能**: 将原始查询计划转换为树形结构，提供树可视化功能

#### 2. NodeEncoder - 节点级编码
- **输入**: TreeNode（每个节点的vector为空）
- **输出**: TreeNode（每个节点都有node_vector）
- **功能**: 使用分块编码策略，将每个查询计划节点编码为数值向量

**分块编码策略**:
- 算子类型 → Embedding Layer → [32维]
- 数据统计 → MLP (log标准化+全连接) → [16维]
- 谓词信息 → Simple Encoder (复杂度特征) → [8维]
- 特征融合: Concat([32, 16, 8]) → Linear Projection → [64维]

#### 3. TreeEncoder - 结构级编码
- **输入**: 带有节点向量的树/DAG
- **输出**: 全局plan embedding
- **功能**: 将所有节点向量聚合成单一的计划表示
- **支持方法**: 统计聚合(mean/sum)、GNN模型(GCN/GAT)

#### 4. PredictionHead - 预测输出
- **输入**: plan embedding向量
- **输出**: 最终预测结果
- **功能**: 线性预测头，支持torch.Tensor和numpy数组输入

### 数据类型

```python
class TreeNode:
    node_type: str                # 节点类型，如 "Seq Scan" / "Hash Join"
    children: List["TreeNode"]    # 子节点列表 (有向树/有向无环图结构)
    extra_info: Dict[str, Any]    # 原始属性（来自JSON，例如表名、代价估计、基数估计等）
    node_vector: Optional[torch.Tensor] = None  # 节点向量 (编码后得到)
```

### NodeEncoder输入特征

NodeEncoder处理的特征类型包括：

#### 基本算子信息
- **node_type** (str) - 算子类型（Hash Join, Merge Join, Seq Scan等）
- **relation_name** (str, 可选) - 表/索引名称
- **alias** (str, 可选) - 表别名

#### 代价/基数信息
- **plan_rows** (float) - 估计行数
- **plan_width** (int) - 每行字节数
- **startup_cost** (float) - 启动代价
- **total_cost** (float) - 总代价

#### 谓词与约束
- **filter** (str) - 过滤条件（表达式）
- **index_cond** (str) - 索引条件
- **join_filter** (str) - 连接条件

#### 执行上下文
- **parallel_aware** (bool) - 是否并行感知
- **actual_rows** (float, 可选) - 真实执行行数
- **actual_time** (float, 可选) - 真实执行时间

## 安装和依赖

### 基础依赖
```
numpy>=1.21.0
torch>=1.9.0
```

### 可选依赖 (GNN功能)
```
torch-geometric>=2.0.0
torch-scatter>=2.0.0
torch-sparse>=0.6.0
```

安装命令：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用流程

```python
from models import (
    DataPreprocessor, 
    create_node_encoder, 
    create_tree_encoder, 
    PredictionHead
)

# 1. 预处理
preprocessor = DataPreprocessor()
tree_node = preprocessor.preprocess(json_plan)

# 2. 节点编码 (自动处理所有节点)
encoder = create_node_encoder()  # 64维标准编码
vectors = encoder.encode_nodes(collect_all_nodes(tree_node))

# 3. 树编码
tree_encoder = create_tree_encoder()
plan_embedding = tree_encoder.forward(vectors)

# 4. 预测
predictor = PredictionHead()
result = predictor.predict(plan_embedding)
```

### 编码器选择

项目提供多种预配置的编码器：

- `create_simple_node_encoder()` - 32维输出，适合快速原型开发
- `create_node_encoder()` - 64维输出，标准分块编码
- `create_large_node_encoder()` - 128维输出，高容量编码

### 树编码选择

```python
# 统计聚合方法
tree_encoder = create_tree_encoder(use_gnn=False, reduction="mean")

# GNN方法 (需要torch-geometric)
tree_encoder = create_tree_encoder(
    use_gnn=True, 
    model_type='gcn',
    input_dim=64,
    hidden_dim=128,
    output_dim=64
)
```

## 项目结构

```
GNTO/
├── docs/
│   ├── NodeEncoder.md           # NodeEncoder设计文档
│   └── 项目清理总结.md         # 项目清理总结
├── examples/
│   └── Demo.py                  # 完整演示代码
├── models/
│   ├── __init__.py              # 模块导入
│   ├── DataPreprocessor.py      # JSON预处理
│   ├── NodeEncoder.py           # 分块编码实现
│   ├── TreeEncoder.py           # 树结构编码
│   └── PredictionHead.py        # 预测头
├── data/                        # 数据文件
├── tmp/                         # 临时文件
├── README.md                    # 项目说明
└── requirements.txt             # 依赖列表
```

## 演示和示例

运行完整演示：
```bash
python examples/Demo.py
```

演示内容包括：
- 逐步处理流程展示
- 多种编码器对比
- 树结构可视化
- 向量信息详细展示

### 示例输出

树结构可视化：
```
└── Gather (Total Cost: 154548.95, Startup Cost: 23540.58, Plan Rows: 567655)
    └── Hash Join (Total Cost: 96783.45, Startup Cost: 22540.58, Plan Rows: 236523)
        ├── Seq Scan (Total Cost: 49166.46, Startup Cost: 0.00, Plan Rows: 649574)
        └── Hash (Total Cost: 15122.68, Startup Cost: 15122.68, Plan Rows: 383592)
            └── Seq Scan (Total Cost: 15122.68, Startup Cost: 0.00, Plan Rows: 383592)
```

节点向量信息：
```
节点 1: Hash Join
  向量维度: torch.Size([64])
  向量值 (前8个): [0.233, 0.296, -1.283, 0.932, -0.670, -0.483, -0.033, -0.388]
  向量范围: [-1.687, 1.124]
```

## 扩展和定制

### 添加新的编码方法

在NodeEncoder基础上扩展新的特征编码：

```python
def encode_custom_feature(self, node) -> torch.Tensor:
    # 实现自定义特征编码
    return feature_vector

# 在forward方法中集成
def forward(self, node) -> torch.Tensor:
    operator_vec = self._encode_operator(node)
    stats_vec = self._encode_stats(node)
    custom_vec = self.encode_custom_feature(node)
    
    combined = torch.cat([operator_vec, stats_vec, custom_vec])
    return self.output_projection(combined)
```

### 自定义树编码

```python
# 继承TreeEncoder实现自定义聚合方法
class CustomTreeEncoder(TreeEncoder):
    def forward(self, vectors):
        # 实现自定义聚合逻辑
        return aggregated_vector
```

## 设计特点

### 优势
- **功能专一**: 每个组件职责清晰，便于理解和维护
- **代码简洁**: 无冗余代码和依赖，架构清晰
- **易于扩展**: 模块化设计，容易添加新功能
- **即插即用**: 所有核心功能都可独立使用
- **PyTorch原生**: 支持梯度传播和端到端训练

### 适用场景
- 查询计划性能预测
- 查询优化器改进
- 数据库系统研究
- 机器学习与数据库交叉研究

## 技术细节

### 分块编码原理

NodeEncoder采用Multi-View编码策略，将不同类型的特征分别处理：

1. **算子类型**: 使用Embedding层将离散的算子类型映射为密集向量
2. **数据统计**: 使用MLP处理数值特征，包含log标准化预处理
3. **谓词信息**: 提取谓词复杂度特征，如范围过滤、子查询、函数调用等
4. **特征融合**: 将所有特征拼接后通过线性投影层输出固定维度向量

### GNN支持

项目支持多种图神经网络模型：
- **GCN**: 图卷积网络，适合处理同构图结构
- **GAT**: 图注意力网络，自动学习节点间的重要性权重

GNN功能需要安装torch-geometric，如果依赖不可用会自动回退到统计聚合方法。