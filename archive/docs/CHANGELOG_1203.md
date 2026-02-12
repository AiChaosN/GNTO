# 更新记录 - 2025年12月3日

## 1. 实验目的
本次更新的主要目的是对比和分析三种不同模型设置下的训练表现，并探究模型规模与训练效率之间的关系。

### 对比对象
1.  **QueryFormer (Baseline)**: 
    - 路径: `QueryFormer_VLDB2022/results/full/cost/training_log.csv`
    - 参数量: ~4.48M
2.  **GNTO QF 1203 (Current)**: 
    - 路径: `GNTO/results/GNTO_QF_1203_1912/training_log.csv`
    - 参数量: ~0.29M (相比 QueryFormer 减少约 15 倍)
3.  **GNTO 1101 (Previous)**: 
    - 路径: `GNTO/results/gnto_1101_training_log.csv`

## 2. 新增工具脚本

### `compare_training_logs.py` (位于 GNTO 根目录)
- **功能**: 自动读取上述三个路径的训练日志，生成对比图表。
- **输出**: `training_comparison.png`
- **包含指标**:
  - Loss (对数坐标)
  - Training Time per Epoch
  - Gradient Norm
  - Error Quantiles (Val Q50, Q90, Q95, Q99, etc.)

### `profile_models.py` (位于 GNTO 根目录)
- **功能**: 对 GNTO_QF 和 QueryFormer 进行纯 GPU 计算性能的 Profile 测试。
- **目的**: 验证为何参数量相差 15 倍但训练时间接近。
- **结论**: GNTO 的纯计算速度快约 4 倍，但实际训练中 **CPU 数据加载 (Data Loading)** 是主要瓶颈，掩盖了 GPU 计算速度的差异。

## 3. 代码改动

### `GNTO/examples/train_qf_standard.py`
- **新增**: 在模型初始化后自动打印模型的 **总参数量** 和 **可训练参数量**。
- **改动位置**: `main()` 函数中 `model = GNTO_QF(...)` 之后。

### `QueryFormer_VLDB2022/TrainingV1.py`
- **新增**: 同样添加了模型参数量的打印代码，便于直观对比。

## 4. 重要发现与分析
1.  **性能对比**:
    - GNTO_QF 尽管参数量仅为 QueryFormer 的 **6.4%** (0.29M vs 4.48M)，但在训练集和验证集上的收敛趋势和最终误差表现具有可比性。
2.  **效率分析**:
    - 现象: GNTO_QF 训练一个 Epoch 的时间与 QueryFormer 相差无几（约 8-10s）。
    - 原因: `profile_models.py` 测试表明 GNTO_QF 的 GPU 推理速度确实更快 (11ms vs 42ms per batch)，但 PyTorch Geometric 的 `DataLoader` 在 CPU 上的图构建和数据传输占据了大部分时间（可能是 50ms+），导致 GPU 大部分时间处于等待状态。
    - 建议: 后续若需进一步加速，应重点优化数据加载流水线 (Data Pipeline) 而非模型结构。

