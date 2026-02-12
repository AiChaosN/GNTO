# GNTO - Graph Neural Tree Optimizer

GNTO 是一个基于深度学习的数据库查询优化与成本预测框架。它专注于查询计划（Query Plan）的节点级编码和结构级建模，利用图神经网络（GNN）来捕捉查询计划的复杂结构特征，从而进行准确的执行时间或成本预测。

## 🌟 项目亮点

*   **分层架构设计**: 清晰分离数据预处理、节点编码、树/图编码和预测头。
*   **增强的节点编码**: 支持多种特征提取策略，包括基础的 16 维特征和增强的 58 维全特征（包含谓词、统计信息等）。
*   **多种模型支持**: 内置支持 GAT (Graph Attention Network)、GCN (Graph Convolutional Network) 以及基础统计模型。
*   **灵活的配置系统**: 提供开箱即用的训练配置预设（如快速测试、统计模型、GAT 模型等）。
*   **完善的评估体系**: 集成 Q-Error (Median, 90th percentile, Mean) 和 MSE 等评估指标。

## 📂 项目结构

```
GNTO/
├── config/                  # 训练配置管理
│   └── training_config.py   # 预设配置 (quick_test, gat, gcn 等)
├── data/                    # 数据集文件 (CSV/JSON 格式的查询计划)
├── docs/                    # 项目文档
├── examples/                # 示例代码与实验脚本
│   ├── 1002_enhanced_training_example.py  # 核心：使用增强编码器的完整训练示例
│   ├── 0120_test_dace_workload1.py        # DACE 对比测试
│   └── ...                  # 其他对比实验 (QueryFormer 等) 和消融实验脚本
├── models/                  # 核心模型实现
│   ├── DataPreprocessor.py  # 数据预处理与图结构转换
│   ├── NodeEncoder.py       # 节点特征编码器
│   ├── NodeVectorizerAll.py # 全量特征向量化实现
│   ├── TreeEncoder.py       # 树/图编码器 (GAT, GCN)
│   ├── PredictionHead.py    # 预测头
│   └── TrainAndEval.py      # 训练与评估流程封装
├── archive/                 # 归档的旧版本代码
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```

## 🏗️ 架构设计

GNTO 采用模块化的流水线设计：

1.  **DataPreprocessor (数据预处理)**:
    *   解析原始查询计划（CSV/JSON）。
    *   提取 PlanNode 结构。
    *   将树状查询计划转换为图结构 (`edge_index`, 节点特征矩阵)。

2.  **NodeEncoder (节点编码)**:
    *   **NodeVectorizerAll**: 提取丰富的节点特征（最高支持 58 维），包括：
        *   算子类型 (One-hot/Embedding)
        *   代价估算 (Startup Cost, Total Cost, Plan Rows 等)
        *   并行执行信息 (Parallel Aware)
        *   谓词与过滤条件特征
    *   支持多种编码策略：MLP, Attention 增强等。

3.  **TreeEncoder (结构编码)**:
    *   使用图神经网络聚合节点信息。
    *   **GATTreeEncoder**: 利用注意力机制捕捉算子间的依赖关系。
    *   支持配置为 GCN 或其他图模型。

4.  **PredictionHead (预测输出)**:
    *   将图嵌入向量映射为最终的预测值（Execution Time 或 Cost）。

## 🚀 快速开始

### 1. 环境准备

确保安装必要的依赖库（建议使用 Python 3.8+）：

```bash
pip install -r requirements.txt
```

主要依赖包括 `torch`, `torch_geometric`, `pandas`, `numpy`, `scikit-learn` 等。

### 2. 运行示例

使用增强版编码器进行训练的完整示例：

```bash
python examples/1002_enhanced_training_example.py
```

该脚本将演示：
*   不同 NodeEncoder 的对比。
*   数据加载与预处理流程。
*   构建 GAT 模型并进行训练。
*   输出训练过程中的 Loss 和验证集的 Q-Error。

## ⚙️ 配置说明

在 `config/training_config.py` 中定义了多种训练模式，可通过 `get_config` 调用：

*   `quick_test`: 快速调试配置（小 Batch，少 Epoch）。
*   `statistical`: 统计基线模型配置。
*   `gat`: 标准 GAT 模型训练配置（推荐）。
*   `gcn`: GCN 模型训练配置。
*   `multi_target`: 同时预测时间和成本的多目标配置。

示例用法：

```python
from config.training_config import get_config
config = get_config('gat')
```

## 📊 实验与对比

`examples/` 目录下包含多个用于对比实验的脚本：
*   **DACE 对比**: `0120_test_dace_workload1.py`
*   **QueryFormer 对比**: 相关脚本如 `1216_compGntoWithQF.py`
*   **消融实验**: `0204_run_ablation_gnto.py`

这些脚本用于验证 GNTO 在不同 Workload（如 JOB-Light, TPC-H）下相对于其他 SOTA 方法的性能优势。
