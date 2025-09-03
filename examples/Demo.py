print("########################\n\
# 读取数据\n\
########################")

import pandas as pd
import json
df = pd.read_csv('../data/demo_plan_01.csv')
print("df:\n", df.head())

#获取json字符串
plans_json = df['json']
print("plans_json:\n", plans_json.iloc[0])




#字符串转json
plans_dict = []
ExecutionTimes = []
for json_str in plans_json:
    plan_dict = json.loads(json_str)
    plans_dict.append(plan_dict['Plan'])
    ExecutionTimes.append(plan_dict['Execution Time'])
print("plans_dict:\n", plans_dict[0])

print("########################\n\
# 预处理数据\n\
########################")

import sys
import os
sys.path.append(os.path.join(os.path.dirname('.'), '..'))

# 预处理数据
from models.DataPreprocessor import PlanNode, DataPreprocessor
preprocessor = DataPreprocessor()
plans_tree = preprocessor.preprocess_all(plans_dict)

# 展示
for i in range(5):
    print(plans_tree[i])
print("--------------------------------")
preprocessor.print_tree(plans_tree[0])


print("########################\n\
# NodeEncoder编码\n\
########################")

from models.NodeEncoder import NodeEncoder
nodeEncoder = NodeEncoder()
from typing import Any, Dict, Iterable, List, Optional

nodeEncodedVectorsBox = []
for plan_tree in plans_tree:
    all_nodes = nodeEncoder.collect_nodes(plan_tree, method="dfs")
    nodeEncodedVectors = nodeEncoder.encode_nodes(all_nodes)
    nodeEncodedVectorsBox.append(nodeEncodedVectors)


print("nodeEncodedVectorsBox:", len(nodeEncodedVectorsBox))
# 查看第一个plan的编码向量
print(len(nodeEncodedVectorsBox[0]))
preprocessor.print_tree(plans_tree[0])

# 检查每个vector是否为空
for i in range(len(nodeEncodedVectorsBox)):
    for j in range(len(nodeEncodedVectorsBox[i])):
        if nodeEncodedVectorsBox[i][j] is None:
            print(f"nodeEncodedVectorsBox[{i}][{j}] is None")


print("########################\n\
# TreeEncoder编码\n\
########################")
from models.TreeEncoder import GATTreeEncoder, TreeToGraphConverter
import torch

treeToGraphConverter = TreeToGraphConverter()
gatTreeEncoder = GATTreeEncoder(
    input_dim=64,      # 一定用实际特征维度
    hidden_dim=64,
    output_dim=64,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    pooling="mean"
)

planEmbeddingBox = []
for plan_tree in plans_tree:
    edge_index, x = treeToGraphConverter.tree_to_graph(plan_tree)
    x = torch.stack(
        [torch.as_tensor(f, dtype=torch.float32) for f in x],
        dim=0
    )
    planEmbeddingBox.append(gatTreeEncoder(x, edge_index))

print("x.shape, edge_index.shape:", x.shape, edge_index.shape)
for i in range(5):
    print(planEmbeddingBox[i].shape)

print("########################\n\
# PredictionHead编码\n\
########################")

# PredictionHead
from models.PredictionHead import PredictionHead
predictionHead = PredictionHead()

predictionBox = []
for planEmbedding in planEmbeddingBox:
    prediction = predictionHead.predict(planEmbedding)
    predictionBox.append(prediction)


print("predictionBox:", len(predictionBox))
for i in range(5):
    print(predictionBox[i], ExecutionTimes[i])

