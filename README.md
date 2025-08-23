# GNTO: A Learned Query Optimizer Framework

## 0. 项目结构
```
GNTO/
├── gnto/                    # 核心代码（带中文注释）
│   ├── core/               # 三个核心模块
│   │   ├── encoders.py     # NodeEncoder + StructureEncoder
│   │   ├── heads.py        # 预测头
│   │   ├── feature_spec.py # 特征处理
│   │   └── model.py        # 主模型
│   ├── inference/          # 推理服务
│   └── utils/              # 工具函数
├── examples/               # 使用示例
│   ├── basic_usage.py      # 基本用法
│   └── three_modules_demo.py # 三模块演示
├── docs/                   # 中文文档
│   └── 核心模块说明.md      # 详细说明文档
├── config/                 # 配置文件
├── tests/                  # 测试代码
└── requirements.txt        # 依赖包
```

**GNTO (GNTO Not Traditional Optimizer)** 是一个用于SQL执行计划优化的学习型查询优化器框架。该框架采用图神经网络和多任务学习，支持代价估计、计划排序、基数估计等多种查询优化任务。

## 主要特性

- **模块化架构**: 低耦合设计，便于组件替换和扩展
- **多任务学习**: 支持代价/延迟回归、计划排序、基数估计
- **图神经网络**: 基于GCN/GAT的执行计划结构编码
- **生产就绪**: 包含推理服务、不确定性估计、回退机制
- **灵活配置**: 支持YAML/JSON配置文件
- **完整测试**: 包含单元测试和集成测试

## 架构概览

### 核心组件

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FeatureSpec   │───▶│   NodeEncoder    │───▶│ StructureEncoder│
│ (特征规范)      │    │  (节点编码器)    │    │  (结构编码器)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│InferenceService │◀───│    LQOModel      │◀───│     Heads       │
│  (推理服务)     │    │   (主模型)       │    │   (预测头)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 支持任务

- **代价/延迟回归**: 执行计划的成本和延迟预测
- **计划排序**: 多个候选计划的排序和选择
- **基数估计**: 节点级别的数据量估计
- **不确定性估计**: 预测结果的置信度评估

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd GNTO

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 基本使用

```python
from gnto import LQOModel, FeatureSpec, InferenceService
from gnto.core.feature_spec import NodeFeatureConfig

# 1. 创建特征规范
feature_config = NodeFeatureConfig()
feature_spec = FeatureSpec(feature_config)

# 2. 创建模型
model = LQOModel(
    feature_spec=feature_spec,
    tasks=["cost", "latency", "ranking"]
)

# 3. 创建执行计划
plan = {
    "nodes": [
        {
            "operator_type": "SeqScan",
            "rows": 10000.0,
            "selectivity": 1.0,
            # ... 其他特征
        }
    ],
    "edges": []
}

# 4. 进行预测
predictions = model.predict(plan)
print(f"预测成本: {predictions['cost']}")
print(f"预测延迟: {predictions['latency']}")
```

### 推理服务

```python
# 创建推理服务
service = InferenceService(
    model=model,
    enable_monitoring=True
)

# 批量预测
plans = [plan1, plan2, plan3]
results = service.batch_predict(plans)

# 检查是否需要回退到传统优化器
for result in results:
    if service.should_fallback(result):
        print("建议使用传统优化器")
```

## 项目结构

```
GNTO/
├── gnto/                          # 主要代码
│   ├── core/                      # 核心组件
│   │   ├── feature_spec.py        # 特征规范和计划处理
│   │   ├── encoders.py            # 节点和结构编码器
│   │   ├── heads.py               # 预测头
│   │   └── model.py               # 主模型
│   ├── inference/                 # 推理服务
│   │   └── service.py             # 生产推理服务
│   └── utils/                     # 工具函数
│       ├── config.py              # 配置管理
│       ├── data_utils.py          # 数据处理
│       └── metrics.py             # 评估指标
├── examples/                      # 示例代码
│   └── basic_usage.py             # 基本使用示例
├── tests/                         # 测试代码
│   └── test_basic.py              # 基础测试
├── config/                        # 配置文件
│   └── default_config.yaml        # 默认配置
├── requirements.txt               # 依赖包
├── setup.py                      # 安装脚本
└── README.md                     # 项目文档
```

## 配置

使用YAML配置文件自定义模型和训练参数：

```yaml
# config/my_config.yaml
model:
  tasks: ["cost", "latency", "ranking"]
  structure_encoder:
    gnn_type: "gat"
    num_layers: 4
    hidden_dim: 256

training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 100

inference:
  batch_size: 32
  enable_monitoring: true
  fallback_threshold:
    uncertainty_threshold: 0.5
```

## 支持的特征

### 节点特征
- **连续特征**: 行数、NDV、选择率、I/O成本、CPU成本、并行度
- **类别特征**: 算子类型、连接类型、索引类型、存储格式、提示
- **结构特征**: 阻塞/流水线标志、探测/构建角色、阶段ID

### 边特征
- **边类型**: 流水线、阻塞、探测、构建等
- **数据流**: 父子节点间的数据传递关系

## 模型架构详解

### 1. 表示学习层

- **节点编码器**: MLP网络编码节点特征
- **结构编码器**: GCN/GAT编码计划结构，支持边类型
- **池化策略**: 支持mean、max、attention等池化方法

### 2. 预测头层

- **回归头**: 成本/延迟/内存预测
- **排序头**: 计划排序和选择
- **基数头**: 节点级基数估计
- **不确定性头**: 认知和随机不确定性估计

### 3. 多任务学习

- **自适应权重**: 基于不确定性的任务权重自动调整
- **联合训练**: 多任务共享表示学习
- **损失函数**: 支持MSE、排序损失、NDCG等

## 运行示例

```bash
# 运行基本示例
python examples/basic_usage.py

# 运行测试
python -m pytest tests/ -v

# 使用自定义配置
python examples/basic_usage.py --config config/my_config.yaml
```

## 评估指标

- **回归任务**: MSE, RMSE, MAE, R², MAPE, Q-error
- **排序任务**: Spearman相关性, Kendall's tau, Top-k准确率, NDCG
- **基数估计**: Q-error, 2x/5x/10x准确率
- **不确定性**: 负对数似然, 校准误差

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 致谢

本项目基于最新的学习型查询优化研究，参考了多个开源项目的设计思想。
