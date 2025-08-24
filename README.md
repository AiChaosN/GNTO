# GNTO: A Learned Query Optimizer Framework

**GNTO (GNTO Not Traditional Optimizer)** 是一个用于SQL执行计划优化的学习型查询优化器框架。该框架采用图神经网络和多任务学习，支持代价估计、计划排序、基数估计等多种查询优化任务。

## 项目概述

GNTO框架提供了一个端到端的查询计划性能预测pipeline，将原始的JSON格式查询计划转换为性能预测结果。框架采用模块化设计，每个组件都可以独立使用或替换为自定义实现。

### 核心Pipeline
```
Raw Plan (JSON) → DataPreprocessor → Encoder → TreeModel → Predictioner → Performance Prediction
```

### 主要特性
- **模块化架构**：每个组件都可独立使用和替换
- **树结构处理**：专门针对查询计划的树形结构设计
- **向量化编码**：将计划节点转换为数值向量表示
- **灵活预测**：支持多种聚合方式和自定义权重
- **完整演示**：提供详细的Demo notebook展示所有功能

## 核心组件

- **DataPreprocessor**：将JSON查询计划转换为结构化树
- **Encoder**：将计划节点编码为数值向量
- **TreeModel**：聚合多个向量为单一表示
- **Predictioner**：线性预测头，输出性能预测
- **GNTO**：整合所有组件的主要Pipeline
- **Utils**：工具函数（随机种子、列表扁平化等）



## 项目结构
```
GNTO/
├── README.md                    # 项目说明文档
├── configs/                     # 配置文件目录
├── data/                        # 数据处理模块
│   ├── __init__.py
│   ├── collectors/              # 数据收集器
│   │   ├── __init__.py
│   │   └── postgres_collect.py  # PostgreSQL数据收集
│   ├── enums.py                 # 枚举定义
│   └── featurizers/             # 特征提取器
│       ├── __init__.py
│       └── pg_plan_json.py      # PostgreSQL计划JSON特征化
├── docs/                        # 详细技术文档
│   ├── api.md                   # API文档
│   ├── architecture.md          # 架构设计
│   └── extension.md             # 扩展指南
├── models/                      # 核心模型组件
│   ├── __init__.py
│   ├── DataPreprocessor.py      # 数据预处理器
│   ├── Encoder.py               # 节点编码器
│   ├── Gnto.py                  # 主要Pipeline类
│   ├── Predictioner.py          # 预测头
│   ├── TreeModel.py             # 树模型聚合器
│   ├── Utils.py                 # 工具函数
│   └── Demo.ipynb               # 完整功能演示
├── result/                      # 结果输出
│   └── result.py
└── tmp/                         # 临时文件
    └── lqo_skeleton.zip
```

## 快速开始

### 环境要求
- Python 3.8+
- NumPy
- Jupyter Notebook (用于运行演示)

### 安装
```bash
git clone <repository-url>
cd GNTO
# 确保已安装numpy
pip install numpy
```

### 基本使用

#### 1. 简单示例
```python
from models.Gnto import GNTO

# 创建GNTO实例
gnto = GNTO()

# 示例查询计划
plan = {
    "Node Type": "Hash Join",
    "Join Type": "Inner",
    "Cost": 500.0,
    "Rows": 2500,
    "Plans": [
        {
            "Node Type": "Seq Scan",
            "Relation Name": "users",
            "Cost": 100.0
        },
        {
            "Node Type": "Index Scan",
            "Index Name": "orders_idx",
            "Cost": 150.0
        }
    ]
}

# 获取性能预测
prediction = gnto.run(plan)
print(f"预测性能: {prediction}")
```

#### 2. 自定义组件
```python
from models.DataPreprocessor import DataPreprocessor
from models.Encoder import Encoder
from models.TreeModel import TreeModel
from models.Predictioner import Predictioner
from models.Gnto import GNTO

# 创建自定义组件
custom_tree_model = TreeModel(reduction="sum")  # 使用sum聚合
custom_predictor = Predictioner(weights=[0.1, 0.5, 0.8, 0.3])

# 创建自定义GNTO实例
gnto_custom = GNTO(
    tree_model=custom_tree_model,
    predictioner=custom_predictor
)

# 运行预测
prediction = gnto_custom.run(plan)
```

### 完整演示

运行 `models/Demo.ipynb` 查看所有组件的详细演示：

```bash
cd models
jupyter notebook Demo.ipynb
```

Demo包含：DataPreprocessor处理、Encoder编码、TreeModel聚合、Predictioner预测、Utils工具、完整Pipeline演示等。

## 文档

详细的技术文档请参考：
- [API文档](docs/api.md) - 详细的类和方法说明
- [架构设计](docs/architecture.md) - 技术特点和设计原理  
- [扩展指南](docs/extension.md) - 自定义组件开发指南

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [LCM-Eval](https://github.com/lcm-eval/lcm-eval) 提供的数据集
- PostgreSQL社区提供的查询计划格式参考
