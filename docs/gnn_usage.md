# GNN TreeModel 使用指南

本文档介绍如何在GNTO项目中使用基于图神经网络(GNN)的TreeModel组件。

## 概述

GNN TreeModel是对原始TreeModel的增强版本，它将查询计划的树结构转换为图结构，并使用图神经网络进行更复杂的特征学习和聚合。

### 主要优势

1. **更好的结构理解**: GNN能够更好地捕捉树结构中的复杂关系
2. **丰富的特征提取**: 支持数值特征、分类特征等多种特征类型
3. **可学习的聚合**: 使用神经网络替代简单的统计聚合
4. **注意力机制**: GAT模型支持注意力机制，能够关注重要的节点关系

## 安装依赖

首先安装必要的依赖包：

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install torch torch-geometric torch-scatter torch-sparse
```

## 基本使用

### 1. 使用GNN TreeModel替换原始TreeModel

```python
from models.GNNTreeModel import GNNTreeModel
from models.GNNEncoder import GNNEncoder
from models.DataPreprocessor import DataPreprocessor
from models.Predictioner import Predictioner

# 创建组件
preprocessor = DataPreprocessor()
encoder = GNNEncoder(feature_dim=64, include_numerical=True)
tree_model = GNNTreeModel(
    model_type="gcn",
    input_dim=64,
    hidden_dim=128,
    output_dim=64,
    num_layers=3
)
predictioner = Predictioner()

# 处理查询计划
plan = {
    "Node Type": "Hash Join",
    "Total Cost": 1234.56,
    "Plan Rows": 5000,
    "Plans": [...]
}

structured = preprocessor.preprocess(plan)
encoded = encoder.encode(structured)
vector = tree_model.forward([structured])  # 注意：传入树结构而非向量
prediction = predictioner.predict(vector)
```

### 2. 使用增强的GNTO Pipeline

```python
from models.GNNGnto import GNNGnto

# 创建GNN增强的GNTO实例
gnto = GNNGnto(use_gnn=True)

# 直接处理查询计划
plan = {...}
prediction = gnto.run(plan)

print(f"使用的组件: {gnto.get_component_info()}")
print(f"GNN状态: {gnto.is_using_gnn()}")
```

## 高级配置

### 1. 自定义GNN配置

```python
gnn_config = {
    'encoder': {
        'feature_dim': 128,
        'include_numerical': True,
        'include_categorical': True,
        'normalize_features': True
    },
    'tree_model': {
        'model_type': 'gat',  # 使用GAT而非GCN
        'input_dim': 128,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 4,
        'num_heads': 8,  # GAT的注意力头数
        'dropout': 0.2,
        'pooling': 'max',
        'device': 'cuda'  # 使用GPU
    }
}

gnto = GNNGnto(use_gnn=True, gnn_config=gnn_config)
```

### 2. 不同类型的GNN模型

#### GCN (图卷积网络)
```python
gcn_model = GNNTreeModel(
    model_type="gcn",
    input_dim=64,
    hidden_dim=128,
    num_layers=3,
    dropout=0.1
)
```

#### GAT (图注意力网络)
```python
gat_model = GNNTreeModel(
    model_type="gat",
    input_dim=64,
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

### 3. 特征工程配置

```python
# 基础特征（仅节点类型）
basic_encoder = GNNEncoder(
    feature_dim=32,
    include_numerical=False,
    include_categorical=False
)

# 包含数值特征
numerical_encoder = GNNEncoder(
    feature_dim=64,
    include_numerical=True,
    include_categorical=False
)

# 完整特征
full_encoder = GNNEncoder(
    feature_dim=128,
    include_numerical=True,
    include_categorical=True,
    normalize_features=True
)
```

## 模型训练

### 1. 设置训练模式

```python
# 创建可训练的GNTO实例
gnto = GNNGnto(use_gnn=True)

# 设置为训练模式
gnto.set_training_mode(True)

# 获取模型参数
parameters = gnto.get_model_parameters()

if parameters:
    import torch.optim as optim
    optimizer = optim.Adam(parameters, lr=0.001)
```

### 2. 训练循环示例

```python
import torch
import torch.nn as nn

# 准备训练数据
training_plans = [...]  # 查询计划列表
training_labels = [...]  # 对应的性能标签

# 损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(gnto.get_model_parameters(), lr=0.001)

gnto.set_training_mode(True)

for epoch in range(100):
    total_loss = 0
    for plan, label in zip(training_plans, training_labels):
        optimizer.zero_grad()
        
        prediction = gnto.run(plan)
        loss = criterion(torch.tensor(prediction), torch.tensor(label))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss / len(training_plans)}")
```

### 3. 模型保存和加载

```python
# 保存模型
gnto.save_model("my_gnn_model")

# 加载模型
gnto.load_model("my_gnn_model")
```

## 性能分析

### 1. 组件性能对比

```python
plans = [...]  # 测试计划列表

# 基准测试
timing_results = gnto.benchmark_components(plans, iterations=10)
print("组件性能 (秒/计划):")
for component, time_per_plan in timing_results.items():
    print(f"  {component}: {time_per_plan:.6f}")
```

### 2. GNN vs 原始方法对比

```python
from models.Gnto import GNTO

# 原始方法
original_gnto = GNTO()

# GNN方法
gnn_gnto = GNNGnto(use_gnn=True)

# 对比预测结果
for plan in test_plans:
    orig_pred = original_gnto.run(plan)
    gnn_pred = gnn_gnto.run(plan)
    
    print(f"原始预测: {orig_pred:.4f}, GNN预测: {gnn_pred:.4f}")
```

## 故障排除

### 1. 依赖问题

如果遇到GNN组件导入错误：

```python
from models import is_gnn_available

if not is_gnn_available():
    print("GNN组件不可用，请安装PyTorch和PyTorch Geometric")
    # 自动回退到原始组件
    gnto = GNNGnto(use_gnn=False)
```

### 2. 内存问题

对于大型查询计划，可能需要调整配置：

```python
# 减少模型复杂度
lightweight_config = {
    'tree_model': {
        'hidden_dim': 64,  # 减少隐藏层维度
        'num_layers': 2,   # 减少层数
        'dropout': 0.3     # 增加dropout
    }
}

gnto = GNNGnto(gnn_config=lightweight_config)
```

### 3. GPU使用

```python
import torch

# 检查GPU可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gnn_config = {
    'tree_model': {
        'device': device
    }
}

gnto = GNNGnto(gnn_config=gnn_config)
```

## 示例代码

完整的使用示例请参考 `models/gnn_demo.py` 文件。

运行演示：

```bash
cd models
python gnn_demo.py
```

## 扩展开发

### 1. 自定义GNN模型

可以继承现有的GNN模型类来实现自定义的图神经网络：

```python
from models.GNNTreeModel import GCNTreeModel
import torch.nn as nn

class CustomGNNModel(GCNTreeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加自定义层
        self.custom_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(self, x, edge_index, batch=None):
        # 调用父类方法
        x = super().forward(x, edge_index, batch)
        # 添加自定义处理
        x = self.custom_layer(x)
        return x
```

### 2. 自定义特征提取

```python
from models.GNNEncoder import GNNEncoder

class CustomGNNEncoder(GNNEncoder):
    def _extract_numerical_features(self, node):
        # 自定义数值特征提取逻辑
        features = super()._extract_numerical_features(node)
        # 添加自定义特征
        custom_features = [...]
        return np.concatenate([features, custom_features])
```

这样就可以根据具体需求来扩展和定制GNN组件了。
