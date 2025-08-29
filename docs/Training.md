# GNTO 训练模块使用指南

GNTO 训练模块提供了完整的查询计划性能预测模型训练功能，支持传统统计方法和图神经网络方法。

## 快速开始

### 1. 基本训练

最简单的训练方式：

```bash
# 使用快速测试配置
python train.py --config quick_test --data data/demo_plan_01.csv

# 使用统计基线模型
python train.py --config statistical --data data/demo_plan_01.csv
```

### 2. 自定义参数

```bash
python train.py --data data/demo_plan_01.csv \
    --model-type statistical \
    --epochs 50 \
    --learning-rate 0.001 \
    --hidden-dim 128 \
    --batch-size 32
```

### 3. GNN 模型训练

如果安装了 PyTorch Geometric：

```bash
# GCN 模型
python train.py --config gcn --data data/demo_plan_01.csv

# GAT 模型  
python train.py --config gat --data data/demo_plan_01.csv
```

## 训练配置

### 预定义配置

| 配置名称 | 模型类型 | 训练轮数 | 描述 |
|---------|---------|---------|------|
| `quick_test` | statistical | 10 | 快速测试配置 |
| `statistical` | statistical | 100 | 统计基线模型 |
| `gcn` | gcn | 200 | 图卷积网络 |
| `gat` | gat | 200 | 图注意力网络 |
| `cost_prediction` | statistical | 100 | 预测查询成本 |

### 查看可用配置

```bash
python train.py --list-configs
```

## 数据格式

训练数据应为 CSV 格式，包含以下列：

- `id`: 样本ID
- `json`: PostgreSQL EXPLAIN 输出的 JSON 字符串

JSON 应包含完整的查询执行计划，例如：

```json
{
  "Plan": {
    "Node Type": "Seq Scan",
    "Relation Name": "table_name",
    "Total Cost": 1000.0,
    "Actual Total Time": 500.0,
    "Plans": [...]
  },
  "Planning Time": 1.5,
  "Execution Time": 502.0
}
```

## 编程接口

### 基本使用

```python
from training import PlanDataset, GNTOTrainer
from config import get_config

# 加载数据集
dataset = PlanDataset("data/demo_plan_01.csv")

# 获取配置
config = get_config("statistical")

# 创建训练器并训练
trainer = GNTOTrainer(config)
results = trainer.train(dataset)

print(f"最佳验证 R² 分数: {results['best_metrics']['val_r2']:.4f}")
```

### 自定义配置

```python
from training.trainer import TrainingConfig, GNTOTrainer

# 创建自定义配置
config = TrainingConfig(
    model_type="statistical",
    hidden_dim=128,
    learning_rate=0.001,
    num_epochs=100,
    target_column="Actual Total Time",
    output_dir="my_training_results"
)

# 训练
trainer = GNTOTrainer(config)
results = trainer.train(dataset)
```

### 数据集操作

```python
from training import PlanDataset

# 加载数据集
dataset = PlanDataset("data/demo_plan_01.csv")

# 查看统计信息
print(f"样本数量: {len(dataset)}")
print(f"统计信息: {dataset.statistics}")

# 数据集分割
train_set, val_set, test_set = dataset.split(
    train_ratio=0.8, 
    val_ratio=0.1
)

# 获取目标值
targets = dataset.get_targets("Actual Total Time")
print(f"目标值范围: {targets.min():.2f} - {targets.max():.2f}")
```

## 模型类型

### 1. Statistical 模型
- 使用传统的统计方法聚合节点特征
- 训练速度快，资源占用少
- 适合作为基线模型

### 2. GCN 模型 (需要 PyTorch Geometric)
- 图卷积网络，考虑查询计划的图结构
- 能够学习节点间的复杂关系
- 适合结构化数据

### 3. GAT 模型 (需要 PyTorch Geometric)
- 图注意力网络，自动学习节点重要性
- 性能通常优于 GCN
- 计算复杂度较高

## 输出结果

训练完成后，会在输出目录生成以下文件：

```
result/training/
├── model_name_training.log      # 训练日志
├── model_name_history.json      # 训练历史
├── model_name_results.json      # 训练结果摘要
└── model_name_best.pkl         # 最佳模型检查点
```

### 结果解读

主要评估指标：

- **R² Score**: 决定系数，越接近1越好
- **MAE**: 平均绝对误差，越小越好
- **RMSE**: 均方根误差，越小越好
- **MAPE**: 平均绝对百分比误差
- **Q-Error**: 查询优化中常用的几何平均误差

## 示例脚本

运行完整的训练示例：

```bash
python examples/train_example.py
```

这个脚本演示了：
- 基本训练流程
- 多模型对比
- 自定义配置使用

## 故障排除

### 常见问题

1. **ImportError: No module named 'torch_geometric'**
   - GNN 模型需要 PyTorch Geometric
   - 使用 statistical 模型或安装相关依赖

2. **FileNotFoundError: Dataset file not found**
   - 检查数据文件路径是否正确
   - 确保 CSV 文件存在且格式正确

3. **JSON 解析错误**
   - 检查 CSV 中的 JSON 字符串格式
   - 确保 JSON 包含必要的 "Plan" 字段

4. **内存不足**
   - 减少 batch_size
   - 使用更简单的模型配置
   - 减少数据集大小

### 调试技巧

```bash
# 启用详细日志
python train.py --data data/demo_plan_01.csv --verbose

# 干运行（不实际训练）
python train.py --data data/demo_plan_01.csv --dry-run

# 快速测试
python train.py --config quick_test --data data/demo_plan_01.csv
```

## 性能优化建议

1. **数据预处理**
   - 确保数据质量，移除异常值
   - 考虑数据归一化

2. **模型选择**
   - 从 statistical 模型开始
   - 数据量大时考虑 GNN 模型

3. **超参数调优**
   - 使用验证集选择最佳参数
   - 考虑学习率调度

4. **硬件加速**
   - 使用 GPU 训练 GNN 模型
   - 增加内存以支持更大的批次大小

## 扩展开发

### 添加新的模型类型

1. 在 `models/` 目录下实现新的编码器
2. 在 `TreeEncoder` 中添加对应的工厂方法
3. 在 `TrainingConfig` 中添加新的模型类型

### 添加新的评估指标

1. 在 `training/metrics.py` 中实现新指标
2. 在 `MetricsTracker` 中添加跟踪逻辑
3. 在训练脚本中显示新指标

### 自定义数据加载

```python
from training.dataset import PlanDataset

class CustomPlanDataset(PlanDataset):
    def _extract_targets(self, plan_json):
        # 自定义目标值提取逻辑
        return custom_targets
```
