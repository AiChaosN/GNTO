# GNTO - Graph Neural Tree Optimizer

GNTO是一个基于机器学习的查询优化框架，专注于查询计划的节点级编码和结构级建模。

## 项目概述

GNTO采用分层架构设计，将查询优化分解为多个独立的组件，每个组件都有清晰的职责分工。目前我将真个项目分为四大模块.

## 项目结构

```
GNTO/
├── docs/
│   ├── NodeEncoder.md           # NodeEncoder设计文档
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

## 架构设计

### 核心组件

项目采用4层清晰的处理流程：

```
DataPreprocessor → NodeEncoder → TreeEncoder → PredictionHead
     (JSON)        (分块编码)      (统计聚合)      (线性预测)
```

#### 1. DataPreprocessor - 数据预处理
数据预处理一共分为两大部分:处理Plan数据和获取全局信息.
- 首先是将CSV文件读取为json文件.获取其中的 x 和 y, 也就是目前的 Plan 和 Execution Time
- 然后将json文件中的Plan转换为PlanNode结构.
```python
PlanNode(
    node_type : str 节点的类型,比如Seq Scan, Hash Join等
    children=[] : List["PlanNode"] 这里面的信息就是节点的所有子节点
    extra_info : Dict[str, Any] 这里面的信息就是节点的所有信息
    node_vector : Optional[np.ndarray] 这个就是将上面的extra_info编码后的结果作为vector存放在这里.
)
```

#### 2. NodeEncoder - 节点级编码
该模块主要就是将PlanNode的extra_info编码为node_vector,这个编码的结果将会作为TreeEncoder的输入.
而这也是整个项目中最重要的部分,因为后续的预测结果的好坏,很大程度上取决于这个编码的结果.

**分块编码策略**:
- 算子类型 → Embedding Layer → [32维]
- 数据统计 → MLP (log标准化+全连接) → [16维]
- 谓词信息 → Simple Encoder (复杂度特征) → [8维]
- 特征融合: Concat([32, 16, 8]) → Linear Projection → [64维]
- 未来如果还有其他特征, 也可以继续添加

#### 3. TreeEncoder - 结构级编码
该模块主要有两个Class
- TreeToGraphConverter: 将TreeNode转换为Graph
由于之前的结构为PlanNode结构,这个结构是我自定义的结构体,不适合用于GNN模型,所以需要将PlanNode转换为Graph,
而转换为Graph时的输入就是我的结构体PlanNode,输出则是Graph的 x 和 edge_index,其中x为Graph的节点特征,edge_index为Graph的边索引,代表了节点之间的连接关系.(因为它是有向图,所以需要两个方向的边索引)

- GATTreeEncoder: 使用GAT模型进行编码
这个模型目前使用的就是GAT模型,如果之后想使用其他模型,并且传入的向量则是PlanNode的node_vector,如果后续需要在这部分进行添加维度可以在后续进行修改.

#### 4. PredictionHead - 预测输出
这部分主要就是线性预测头,传入的向量则是TreeEncoder的输出,输出则是最终的预测结果.目前使用的就是最传统的线性预测头.

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


### 示例输出

最开始在SCV文件中的一条Plan的结构为:
```python
plans_json:
 {"Plan": {"Node Type": "Gather", "Parallel Aware": false, "Startup Cost": 23540.58, "Total Cost": 154548.95, "Plan Rows": 567655, "Plan Width": 119, "Actual Startup Time": 386.847, "Actual Total Time": 646.972, "Actual Rows": 283812, "Actual Loops": 1, "Workers Planned": 2, "Workers Launched": 2, "Single Copy": false, "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": true, "Join Type": "Inner", "Startup Cost": 22540.58, "Total Cost": 96783.45, "Plan Rows": 236523, "Plan Width": 119, "Actual Startup Time": 369.985, "Actual Total Time": 518.487, "Actual Rows": 94604, "Actual Loops": 3, "Inner Unique": false, "Hash Cond": "(t.id = mi_idx.movie_id)", "Workers": [], "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": true, "Relation Name": "title", "Alias": "t", "Startup Cost": 0.0, "Total Cost": 49166.46, "Plan Rows": 649574, "Plan Width": 94, "Actual Startup Time": 0.366, "Actual Total Time": 147.047, "Actual Rows": 514421, "Actual Loops": 3, "Filter": "(kind_id = 7)", "Rows Removed by Filter": 328349, "Workers": []}, {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": true, "Startup Cost": 15122.68, "Total Cost": 15122.68, "Plan Rows": 383592, "Plan Width": 25, "Actual Startup Time": 103.547, "Actual Total Time": 103.547, "Actual Rows": 306703, "Actual Loops": 3, "Hash Buckets": 65536, "Original Hash Buckets": 65536, "Hash Batches": 32, "Original Hash Batches": 32, "Peak Memory Usage": 1920, "Workers": [], "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": true, "Relation Name": "movie_info_idx", "Alias": "mi_idx", "Startup Cost": 0.0, "Total Cost": 15122.68, "Plan Rows": 383592, "Plan Width": 25, "Actual Startup Time": 0.28, "Actual Total Time": 54.382, "Actual Rows": 306703, "Actual Loops": 3, "Filter": "(info_type_id > 99)", "Rows Removed by Filter": 153308, "Workers": []}]}]}]}, "Planning Time": 2.382, "Triggers": [], "Execution Time": 654.241}

```

经过DataPreprocessor处理后,最终的结构为:
```
└── Gather (Total Cost: 154548.95, Startup Cost: 23540.58, Plan Rows: 567655, Plan Width: 119, Actual Total Time: 646.97, Actual Rows: 283812)
    └── Hash Join (Total Cost: 96783.45, Startup Cost: 22540.58, Plan Rows: 236523, Plan Width: 119, Actual Total Time: 518.49, Actual Rows: 94604, Join Type: Inner)
        ├── Seq Scan (Total Cost: 49166.46, Startup Cost: 0.00, Plan Rows: 649574, Plan Width: 94, Actual Total Time: 147.05, Actual Rows: 514421, Relation Name: title, Alias: t)
        └── Hash (Total Cost: 15122.68, Startup Cost: 15122.68, Plan Rows: 383592, Plan Width: 25, Actual Total Time: 103.55, Actual Rows: 306703)
            └── Seq Scan (Total Cost: 15122.68, Startup Cost: 0.00, Plan Rows: 383592, Plan Width: 25, Actual Total Time: 54.38, Actual Rows: 306703, Relation Name: movie_info_idx, Alias: mi_idx)
```

经过NodeEncoder编码后,节点向量信息为:
```
└── Gather (Total Cost: 154548.95, Startup Cost: 23540.58, Plan Rows: 567655, Plan Width: 119, Actual Total Time: 646.97, Actual Rows: 283812), node_vector_shape: torch.Size([64])
    └── Hash Join (Total Cost: 96783.45, Startup Cost: 22540.58, Plan Rows: 236523, Plan Width: 119, Actual Total Time: 518.49, Actual Rows: 94604, Join Type: Inner), node_vector_shape: torch.Size([64])
        ├── Seq Scan (Total Cost: 49166.46, Startup Cost: 0.00, Plan Rows: 649574, Plan Width: 94, Actual Total Time: 147.05, Actual Rows: 514421, Relation Name: title, Alias: t), node_vector_shape: torch.Size([64])
        └── Hash (Total Cost: 15122.68, Startup Cost: 15122.68, Plan Rows: 383592, Plan Width: 25, Actual Total Time: 103.55, Actual Rows: 306703), node_vector_shape: torch.Size([64])
            └── Seq Scan (Total Cost: 15122.68, Startup Cost: 0.00, Plan Rows: 383592, Plan Width: 25, Actual Total Time: 54.38, Actual Rows: 306703, Relation Name: movie_info_idx, Alias: mi_idx), node_vector_shape: torch.Size([64])
```

经过TreeToGraphConverter后结构为:
```Python
x.shape, edge_index.shape: torch.Size([4, 64]) torch.Size([2, 6])
```

经过GATTreeEncoder编码后结构为:
```Python
torch.Size([64])
```

经过PredictionHead编码后结构为:
```Python
654.241
```


