# GNTO - Graph Neural Tree Optimizer

GNTO 是一个基于深度学习的数据库查询优化与成本预测框架。它专注于查询计划（Query Plan）的节点级编码和结构级建模，利用图神经网络（GNN）来捕捉查询计划的复杂结构特征，从而进行准确的执行时间或成本预测。

## 项目亮点 (Key Highlights)

*   **SOTA 混合编码架构**: 融合了基于数据分布的统计特征 (QueryFormer-style Histograms) 和基于优化器的先验知识 (Optimizer Cost Estimates)，实现了比纯数据驱动方法更优的性能。
*   **DeepSets 理论对齐**: 节点编码器 (V4) 严格遵循 DeepSets 理论，使用 Sum Pooling 聚合谓词特征，保留了集合的完整信息量 (Total Filtering Mass)。
*   **动态图注意力机制**: 引入 GATv2 和 Global Attention Pooling，解决了传统 GAT 的静态注意力瓶颈，能够动态捕捉查询计划中关键路径和算子的影响。
*   **模块化演进**: 拥有完整的模型演进历史 (V1 -> V4 -> QF+)，支持灵活切换不同的编码器组合进行消融实验。

## 项目结构

```
GNTO/
├── config/                  # 训练配置管理
├── data/                    # 数据集文件
├── docs/                    # 项目文档 (新增架构演进与实验说明)
│   ├── Model_Evolution.md          # 模型各模块详细演进历史
│   └── Experiment_Results_Source.md # 实验结果与脚本对应表
├── examples/                # 示例代码与实验脚本
│   ├── 1216_compGntoWithQF_addPlanrows.py # 【核心】当前 SOTA 模型训练脚本 (QF+ & GATv2)
│   ├── 1216_compGntoWithQF.py             # 对比实验：不带 PlanRows 的版本
│   ├── 1203_train_qf_standard.py          # 基线实验：QueryFormer 复现
│   ├── 0204_run_ablation_gnto.py          # 消融实验自动化脚本
│   └── ...
├── models/                  # 核心模型实现
│   ├── NodeEncoder.py       # 节点编码器 (含 V4, QF, QF_AddPlanrows)
│   ├── TreeEncoder.py       # 树编码器 (含 GATv2, GlobalAttention)
│   ├── PredictionHead.py    # 预测头 (含 ResNet-style V2)
│   └── ...
├── archive/                 # 归档代码
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```

## 核心架构 (Current SOTA)

目前表现最佳的模型配置 (Implemented in `examples/1216_compGntoWithQF_addPlanrows.py`)：

1.  **Node Encoder: `NodeEncoder_QF_AddPlanrows` (Hybrid)**
    *   **基础**: 继承自 QueryFormer，使用 150维直方图 (Histograms) 和 1000维采样 (Table Samples) 捕捉数据分布。
    *   **增强**: 显式注入优化器估算的 `Plan Rows` 作为额外特征通道。
    *   **优势**: 结合了数据驱动的细粒度统计信息和优化器的全局代价估算能力。

2.  **Tree Encoder: `GATv2TreeEncoder_V3`**
    *   **机制**: 使用 **GATv2** (Dynamic Graph Attention) 替代标准 GAT。
    *   **结构**: 3层 GATv2 + LayerNorm + Residual Connections。
    *   **聚合**: 支持 Global Attention Pooling (GAP)，自动学习节点权重进行图级聚合。

3.  **Prediction Head: `PredictionHead_V2`**
    *   **结构**: ResNet-style 的深层预测网络。
    *   **特点**: 包含残差连接和 LayerNorm，相比简单 MLP 具有更强的非线性拟合能力和训练稳定性。

## 快速复现 (Quick Start)

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 训练 SOTA 模型

要复现论文中的最佳结果 (Best Performance)，请运行：

```bash
python examples/1216_compGntoWithQF_addPlanrows.py
```

该脚本会自动：
1.  加载 QueryFormer 格式的数据集（含直方图和采样）。
2.  注入 `Plan Rows` 特征。
3.  训练 GNTO (QF+ / GATv2) 模型。
4.  输出验证集 Q-Error (Median, 90th, 95th, etc.)。

### 3. 查看对比与消融实验

*   **查看模型演进细节**: 请阅读 `docs/Model_Evolution.md`。
*   **查找实验对应关系**: 请阅读 `docs/Experiment_Results_Source.md`。
*   **运行消融实验**:
    ```bash
    python examples/0204_run_ablation_gnto.py
    ```

## 实验结论摘要

*   **GNTO vs QueryFormer**: 引入 `Plan Rows` 和 GATv2 后，GNTO 在复杂查询上的 Q-Error (95th/99th) 显著优于原始 QueryFormer。
*   **GAT vs GATv2**: 动态注意力机制 (GATv2) 在处理长路径依赖时表现更佳。
*   **Hybrid Encoding**: 混合编码 (QF+PlanRows) 证明了优化器的估算值虽然不完美，但包含重要的高阶逻辑信息，能有效辅助神经网络。
