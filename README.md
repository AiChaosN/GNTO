# GNTO Examples

本目录包含GNTO项目的示例代码和演示脚本，帮助用户快速理解和使用GNTO框架。

## 文件说明

### 综合演示
- **`Demo.py`** - 正确架构理解

### 数据类型
```python
class TreeNode:
    node_type: str                # 节点类型，如 "Seq Scan" / "Hash Join"
    children: List["TreeNode"]    # 子节点列表 (有向树/有向无环图结构)
    extra_info: Dict[str, Any]    # 原始属性（来自JSON，例如表名、代价估计、基数估计等）
    node_vector: Optional[np.ndarray] = None  # 节点向量 (编码后得到)
```

### Node Encoder 输入特征

"Node Encoder"的输入特征池，常见字段可分为 **离散** 和 **连续** 两类：

#### (1) 基本算子信息
- **node_type** (str) — 算子类型（Hash Join, Merge Join, Seq Scan…）
- **relation_name** (str, 可选) — 表/索引名称
- **alias** (str, 可选) — 表别名

#### (2) 代价/基数信息
- **plan_rows** (float) — 估计行数
- **plan_width** (int) — 每行字节数
- **startup_cost** (float) — 启动代价
- **total_cost** (float) — 总代价

#### (3) 谓词与约束
- **filter** (str) — 过滤条件（表达式）
- **index_cond** (str) — 索引条件
- **join_filter** (str) — 连接条件

#### (4) 执行上下文（如果有）
- **parallel_aware** (bool) — 是否并行感知
- **actual_rows** (float, 可选) — 真实执行行数（仅在收集 ground truth 时有）
- **actual_time** (float, 可选) — 真实执行时间

## 架构演示

### 处理流程

#### 1. 预处理 (DataPreprocessor)
- **输入**：原始的JSON风格的计划
- **输出**：TreeNode结构（每个Node的vector还为空）

#### 2. 节点级编码 (Node Encoder)
- **输入**：TreeNode（此时每个Node都还为空）
- **输出**：TreeNode（此时每个Node都有自己的node_vector）

#### 3. 结构级编码 (Tree Encoder / Structure Model)
- **输入**：TreeNode（带有节点向量）
- **输出**：把带有节点向量的树/DAG整体编码成一个plan embedding

#### 4. 预测头 (Prediction Head)
- **输入**：全局向量（plan embedding）
- **输出**：最终预测结果，也可以同时保留节点级向量

## 依赖要求

### 基础依赖
- Python 3.8+
- pandas
- numpy

## 示例输出

### 树结构可视化
```
└── Gather (Total Cost: 154548.95, Startup Cost: 23540.58, Plan Rows: 567655)
    └── Hash Join (Total Cost: 96783.45, Startup Cost: 22540.58, Plan Rows: 236523)
        ├── Seq Scan (Total Cost: 49166.46, Startup Cost: 0.00, Plan Rows: 649574)
        └── Hash (Total Cost: 15122.68, Startup Cost: 15122.68, Plan Rows: 383592)
            └── Seq Scan (Total Cost: 15122.68, Startup Cost: 0.00, Plan Rows: 383592)
```

## 自定义使用

### 使用自己的数据
修改数据路径：
```python
data_path = "/path/to/your/query_plans.csv"
```