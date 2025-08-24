# GNTO: A Learned Query Optimizer Framework
**GNTO (GNTO Not Traditional Optimizer)** 是一个用于SQL执行计划优化的学习型查询优化器框架。该框架采用图神经网络和多任务学习，支持代价估计、计划排序、基数估计等多种查询优化任务。

## 0. 项目流程
1. 数据收集 : Data
数据来源于[LCM-Eval](https://github.com/lcm-eval/lcm-eval)的一个[OSF](https://osf.io/rb5tn/files/osfstorage?view_only=)仓库, 
其中包含了多个数据库的查询计划和执行成本, 以及多个查询优化任务的标签.

2. 数据预处理 DataPreprocess
数据预处理阶段, 将数据转换为模型可以接受的格式.

3. 数据转换 Encoder 阶段 : Plan -> Plan With Vector
将每个点上的数据转换为向量表示, 其中包含节点特征和结构特征.

4. Tree Model 阶段 : Plan With Vector -> Vector
使用TreeModel将前面得到的向量进行树结构编码, 得到一个向量表示.

5. 预测头阶段 : Vector -> Prediction
使用预测头将前面的向量进行预测, 得到一个预测结果.

6. 评估阶段 : Prediction -> Evaluation
使用评估指标将前面的预测结果进行评估, 得到一个评估结果.


### 核心模块
```
DataPreprocer -> Encoder -> TreeModel -> PredictionHead -> Evaluation
```



## 1. 项目结构
```
gnto
├── README.md # 项目说明
├── configs # 配置文件
├── data # 数据
│   ├── __init__.py
│   ├── collectors # 数据收集
│   │   ├── __init__.py
│   │   └── postgres_collect.py # 收集postgres数据
│   ├── enums.py
│   └── featurizers
│       ├── __init__.py
│       └── pg_plan_json.py
├── models # 模型
│   ├── DataPreprocessor.py
│   ├── Encoder.py
│   ├── Gnto.py
│   ├── Predictioner.py
│   ├── TreeModel.py
│   └── Utils.py
├── result
│   └── result.py # 结果
└── tmp
    └── lqo_skeleton.zip # lqo_skeleton 的临时文件

```


## 快速开始
