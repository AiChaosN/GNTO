#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1.读取数据
import glob
import pandas as pd
import json

# 匹配所有 train_plan_0*.csv
files = glob.glob("../data/train_plan_*.csv")
print("找到的文件:", files)

# 读入并合并
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("总数据行数:", len(df))
print("df:\n", df.head())

#获取json字符串
plans_json = df['json']
print("plans_json:\n", plans_json.iloc[0])

#字符串转json
plans_dict = []
ExecutionTimes = []
idx = 0
for json_str in plans_json:
    idx += 1
    plan_dict = json.loads(json_str)
    plans_dict.append(plan_dict['Plan'])
    try:
        ExecutionTimes.append(plan_dict['Execution Time'])
    except:
        print(f"idx: {idx} 不存在Execution Time")
        print(plan_dict)
print("plans_dict:\n", plans_dict[0])



# In[2]:


# 2.数据格式转换 json -> PlanNode
import sys, os
sys.path.append(os.path.abspath(".."))  # 确保当前目录加入路径

# json -> PlanNode
from models.DataPreprocessor import PlanNode, DataPreprocessor
preprocessor = DataPreprocessor()
plans_tree = preprocessor.preprocess_all(plans_dict)


# In[ ]:


# 3.数据格式转换 planNode -> graph 格式
# PlanNode -> edges_list, extra_info_list
def tree_to_graph(root):
    edges_list, extra_info_list = [], []

    def dfs(node, parent_idx):
        idx = len(extra_info_list)
        extra_info_list.append(node.extra_info)
        edges_list.append((idx, idx))
        if parent_idx is not None:
            edges_list.append((parent_idx, idx))
        for ch in node.children:
            dfs(ch, idx)

    dfs(root, None)
    return edges_list, extra_info_list

edges_list, matrix_plans = [], []
for i in plans_tree:
    edges_matrix, extra_info_matrix = tree_to_graph(i)
    # if len(edges_matrix) == 0:
    #     print(i)
    #     assert False
    edges_list.append(edges_matrix)
    matrix_plans.append(extra_info_matrix)

print(matrix_plans[0][0])
print(matrix_plans[0][1])
print(edges_list[0])
print(edges_list[99])



# In[4]:


import pandas as pd

def plans_to_df(data: list[list[dict]]) -> pd.DataFrame:
    rows = []
    for pid, plan in enumerate(data):
        for nid, node in enumerate(plan):
            rows.append({"plan_id": pid, "node_idx": nid, **node})
    df = pd.json_normalize(rows, sep='.')

    df = df.sort_values(["plan_id", "node_idx"], kind="stable").reset_index(drop=True)
    return df

plans_df = plans_to_df(matrix_plans)
plans_df.to_csv("../data/process/plans_df.csv", index=False)
plans_df.head()


# In[5]:


from models.DataPreprocessor import safe_cond_parse

NEED_PARSE_COND_SCAN = [
    "Filter",
    "Index Cond",
    "Recheck Cond"
]

NEED_PARSE_COND_JOIN = [
    "Hash Cond",
    "Join Filter",
    "Merge Cond",
]

NEED_PARSE_COND_COLS = NEED_PARSE_COND_SCAN + NEED_PARSE_COND_JOIN

for col in NEED_PARSE_COND_COLS:
    plans_df[f"{col}_Split"] = plans_df[col].apply(safe_cond_parse)

plans_df.to_csv("../data/process/plans_df_parsed.csv", index=False)



# In[6]:


for col in NEED_PARSE_COND_COLS:
    max_and = 0
    for j in plans_df[f"{col}_Split"]:
        if len(j) > max_and:
            max_and = len(j)
    print(col, max_and)

for col in NEED_PARSE_COND_COLS:
    c = 0
    for j in plans_df[f"{col}_Split"]:
        if j != [] and c < 3:
            print(col, j)
            c += 1
       


# In[7]:


db_info = pd.read_csv("../data/column_min_max_vals.csv")
db_info.head()
db_info["table_name"], db_info["column_name"] = db_info["name"].str.split(".").str[0], db_info["name"].str.split(".").str[1]
db_info.head()


# In[8]:


for idx, row in plans_df.iterrows():
    if (row["Relation Name"] != "") != (row["Alias"] != ""):
        print(row["Relation Name"], row["Alias"])
        assert False


# In[13]:


# 将node_type转换为id
node_type = plans_df["Node Type"].unique()
print(node_type, len(node_type))
node_type_mapping = {k : i for i, k in enumerate(node_type)}
plans_df["NodeType_id"] = plans_df["Node Type"].map(node_type_mapping)
print(plans_df[["Node Type", "NodeType_id"]].head())


# In[14]:


def df_to_plans(df: pd.DataFrame, keep_extra_cols=False) -> list[list[dict]]:
    orig_cols = [c for c in df.columns if c not in {"plan_id", "node_idx"}]
    cols = orig_cols if not keep_extra_cols else [c for c in df.columns if c not in {"plan_id", "node_idx"}]

    out = []
    for pid, g in df.sort_values(["plan_id","node_idx"], kind="stable").groupby("plan_id", sort=True):
        plan_nodes = []
        for _, row in g.sort_values("node_idx", kind="stable").iterrows():
            d = {}
            for k in cols:
                v = row[k]
                if v is None or []:
                    continue
                d[k] = v
            plan_nodes.append(d)
        out.append(plan_nodes)
    return out

new_matrix_plans = df_to_plans(plans_df)
new_matrix_plans[0][0]


# In[25]:


# NodeVectorizer
import re, math
from collections import defaultdict
import numpy as np
import torch
from typing import List

from models.Utils import process_join_cond_field, process_index_cond_field, load_column_stats


node_type_list = ['Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort']
parallel_list = [True, False]
plan_rows_max = 2*10**8

def NodeVectorizer(matrix_plans : List[List[dict]]) -> List[List[List[List]]]:
    res = []
    for mp in matrix_plans:
        plan_matrix = []
        for node in mp:
            node_vector = [0] * (len(node_type_list) + 2 + 1)
            offset = 0
            # 1. node_type
            node_vector[node_type_mapping[node["Node Type"]] + offset] = 1
            offset += len(node_type_list)
            # 2. parallel
            node_vector[parallel_list.index(node["Parallel Aware"]) + offset] = 1
            
            offset += len(parallel_list)
            # 3. rows
            node_vector[offset] = node["Plan Rows"] / plan_rows_max
            plan_matrix.append(node_vector)
        res.append(plan_matrix)
    return res

res = NodeVectorizer(new_matrix_plans)
print("NodeType[13] : parallel[2] : rows[1]")
print(len(res[0]))
print(len(res[0][0]))
print(res[0][0])



# In[ ]:


# 模型搭建

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

class NodeEncoder(nn.Module):
    """
    输入: data.x 形状 [N, F_in]
    输出: node_embs [N, d_node]
    """
    def __init__(self, in_dim: int, d_node: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_node),
            nn.ReLU(),
            nn.LayerNorm(d_node),
        )
    def forward(self, x):
        return self.proj(x)

# ---- 组合总模型 ----
class PlanCostModel(nn.Module):
    """
    NodeEncoder → GATTreeEncoder → PredictionHead
    """
    def __init__(self, nodecoder: nn.Module, treeencoder: nn.Module, predict_head: nn.Module):
        super().__init__()
        self.nodecoder = nodecoder
        self.treeencoder = treeencoder
        self.predict_head = predict_head

    def forward(self, data: Data | Batch):
        """
        期望 data 里至少有:
        - x: [N, F_num] (numerical features)
        - x_cat: [N, F_cat] (categorical features)
        - edge_index: [2, E]
        - batch: [N]  指示每个节点属于哪张图
        """
        x = self.nodecoder(data.x)                                   # [N, d_node]
        g = self.treeencoder(x, data.edge_index, data.batch)         # [B, d_graph]
        y = self.predict_head(g)                                     # [B, out_dim]
        return y


from models.TreeEncoder import GATTreeEncoder
from models.PredictionHead import PredictionHead
# ---- 使用示例 ----
# 使用正确的数值特征维度
F_num = 16
d_node, d_graph = 32, 64
nodecoder = NodeEncoder(
    in_dim=F_num,
    d_node=d_node
)
gatTreeEncoder = GATTreeEncoder(
    input_dim=d_node,      # 一定用实际特征维度
    hidden_dim=64,
    output_dim=d_graph,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    pooling="mean"
)
predict_head = PredictionHead(d_graph, out_dim=1)
model = PlanCostModel(nodecoder, gatTreeEncoder, predict_head)


# In[37]:


print(type(ExecutionTimes))
print(type(res))
print(type(edges_list))
print(model)


# In[38]:


# 4.构建数据集class
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def coerce_x_to_tensor(x_plan, in_dim: int):
    """
    x_plan: 很深的 list（最终行向量长度= in_dim）
    变成 [N, in_dim] 的 float32 Tensor
    """
    x = torch.tensor(x_plan, dtype=torch.float32)
    assert x.numel() % in_dim == 0, f"最后一维应为 {in_dim}，拿到形状 {tuple(x.shape)}"
    x = x.view(-1, in_dim)   # 拉平成 [N, in_dim]
    return x

def coerce_edge_index(ei_like):
    """
    ei_like: list/ndarray/tensor, 形状 [2,E] 或 [E,2]
    返回规范 [2,E] 的 long Tensor
    """
    ei = torch.as_tensor(ei_like, dtype=torch.long)
    if ei.ndim != 2:
        raise ValueError(f"edge_index 需要二维，拿到 {tuple(ei.shape)}")
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.t().contiguous()
    elif ei.shape[0] != 2 and ei.shape[1] != 2:
        raise ValueError(f"edge_index 需为 [2,E] 或 [E,2]，拿到 {tuple(ei.shape)}")
    return ei.contiguous()

def build_dataset(res, edges_list, execution_times, in_dim=16, bidirectional=False):
    assert len(res) == len(edges_list) == len(execution_times), "长度必须一致"
    data_list = []
    for i, (x_plan, ei_like, y) in enumerate(zip(res, edges_list, execution_times)):
        x = coerce_x_to_tensor(x_plan, in_dim)      # [N, in_dim]
        edge_index = coerce_edge_index(ei_like)     # [2,E]
        N = x.size(0)

        # 边索引有效性检查
        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= N:
                raise ValueError(f"plan[{i}] 的 edge_index 越界：节点数 N={N}，但 edge_index.max={int(edge_index.max())}")

        # 可选：做成双向图（若你的 edges 只有父->子）
        if bidirectional:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        y = torch.tensor([float(y)], dtype=torch.float32)  # 图级回归标签
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list


# In[41]:


# 构建数据集
dataset = build_dataset(res, edges_list, ExecutionTimes, in_dim=F_num, bidirectional=True)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: x.shape={dataset[0].x.shape}, edge_index.shape={dataset[0].edge_index.shape}, y={dataset[0].y}")



# In[42]:


# 5. 训练准备
from sklearn.model_selection import train_test_split
import time
import os


# 数据集划分
train_indices, temp_indices = train_test_split(
    range(len(dataset)), test_size=0.3, random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, random_state=42
)

train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")



# In[43]:


# 6. 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
criterion = torch.nn.MSELoss()

# 早停机制
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

early_stopping = EarlyStopping(patience=15, min_delta=0.001)


# In[48]:


# 7. 训练函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(batch)
        loss = criterion(pred, batch.y)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    q_errors_all = []  # 收集整份验证集的 q-error
    eps = 1e-8
    
    Q50_list = []
    Q95_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            total_loss += loss.item()
            num_batches += 1

            
            # 防 0 防负；Q-Error 定义是正数比例
            p = torch.clamp(pred, min=eps)
            t = torch.clamp(batch.y,    min=eps)
            q_error = torch.maximum(p / t, t / p)              # [B]
            q_errors_all.append(q_error.cpu().numpy())

    if q_errors_all:
        q_all = np.concatenate(q_errors_all, axis=0)
        Q50 = float(np.quantile(q_all, 0.5))
        Q95 = float(np.quantile(q_all, 0.95))
    else:
        Q50 = float("nan")
        Q95 = float("nan")

    avg_loss = total_loss / max(1, num_batches)
    print(f"val_loss: {avg_loss:.6f} | Q50: {Q50:.6f} | Q95: {Q95:.6f}")


    return total_loss / num_batches


# In[ ]:


from datetime import datetime

#  8. 训练循环
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                early_stopping, device, weight_path, num_epochs=100):
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("开始训练...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 计算时间
        epoch_time = time.time() - start_time
        
        # 打印进度
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.2f}s")
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"\n早停触发在第 {epoch+1} 轮")
            break
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            date = datetime.now().strftime("%m%d")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'../results/weight_{date}.pth')
    
    print("-" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses

# 开始训练
date = datetime.now().strftime("%m%d")
weight_path = f'../results/{date}.pth'
train_losses, val_losses = train_model(
    model, train_loader, val_loader, optimizer, scheduler, 
    criterion, early_stopping, device, weight_path, num_epochs=100
)


# In[54]:


# 9. 测试评估
def evaluate_model(model, test_loader, device):
    model.eval()
    preds_all, targs_all = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).view(-1).float()   # [B]，先拍平
            y    = batch.y.view(-1).float()        # [B]
            preds_all.append(pred.cpu())
            targs_all.append(y.cpu())

    preds = torch.cat(preds_all)   # [N]
    targs = torch.cat(targs_all)   # [N]

    # MSE（Torch实现）
    mse = torch.mean((preds - targs) ** 2).item()

    # Q-Error（Torch实现）
    eps = 1e-8
    p = torch.clamp(preds, min=eps)
    t = torch.clamp(targs, min=eps)
    q = torch.maximum(p / t, t / p)             # [N]
    Q50 = torch.quantile(q, 0.5).item()
    Q95 = torch.quantile(q, 0.95).item()

    # 如果你需要返回 numpy
    predictions = preds.numpy()
    targets = targs.numpy()

    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    print(f"MSE:  {mse:.6f}")
    print(f"Q50: {Q50:.6f}, Q95: {Q95:.6f}")
    print("="*50)

    return predictions, targets, {'mse': mse, 'Q50': Q50, 'Q95': Q95}


# 加载最佳模型进行测试
try:
    # date = datetime.now().strftime("%m%d")
    checkpoint = torch.load(f'../results/0921.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("已加载最佳模型进行测试")
except FileNotFoundError:
    print("未找到保存的模型，使用当前模型进行测试")

predictions, targets, metrics = evaluate_model(model, test_loader, device)


# In[57]:


# 10. 可视化训练过程和结果
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(12, 4))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 预测 vs 真实值
    plt.subplot(1, 2, 2)
    plt.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.title('Predicted vs True Execution Time')
    plt.grid(True)
    
    plt.tight_layout()
    date = datetime.now().strftime("%m%d")
    plt.savefig(f'../results/training_results_{date}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 创建结果目录
os.makedirs('../results', exist_ok=True)
# 绘制结果
plot_training_history(train_losses, val_losses)


# In[ ]:




