# -*- coding: utf-8 -*-
import os, sys, json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ----------------------------
# 路径
# ----------------------------
sys.path.append(os.path.join(os.path.dirname('.'), '..'))

from models.DataPreprocessor import DataPreprocessor
from models.NodeEncoder import NodeEncoder
from models.TreeEncoder import GATTreeEncoder, TreeToGraphConverter
from models.PredictionHead import PredictionHead

# ----------------------------
# 1) 读取与预处理
# ----------------------------
print("########################\n# 读取数据\n########################")
df = pd.read_csv('../data/demo_plan_01.csv')
print("df.head():\n", df.head())

plans_json = df['json']
plans_dict, exec_times = [], []
for s in plans_json:
    d = json.loads(s)
    plans_dict.append(d['Plan'])
    exec_times.append(float(d['Execution Time']))  # 确保是float

print("样例 Plan JSON → Plan dict:\n", plans_dict[0])

print("########################\n# 预处理数据: 变为树\n########################")
preprocessor = DataPreprocessor()
plans_tree = preprocessor.preprocess_all(plans_dict)

for i in range(min(3, len(plans_tree))):
    print(plans_tree[i])
print("--------------------------------")
if len(plans_tree) > 0:
    preprocessor.print_tree(plans_tree[0])

# ----------------------------
# 2) 组装端到端模型
# ----------------------------
class PlanModel(nn.Module):
    """
    端到端：NodeEncoder -> TreeToGraphConverter -> GATTreeEncoder -> PredictionHead
    优先尝试使用 NodeEncoder 的节点向量，并对齐到 tree_to_graph 的节点顺序（若支持返回 perm）。
    回退方案：若 converter 不支持 perm，则使用 x_list 以保证可运行。
    """
    def __init__(self, node_encoder: NodeEncoder,
                 tree_encoder: GATTreeEncoder,
                 head: PredictionHead):
        super().__init__()
        self.node_encoder = node_encoder
        self.tree_encoder = tree_encoder
        self.head = head
        self.converter = TreeToGraphConverter()

    def forward(self, plan_tree):
        # A) 先拿 NodeEncoder 的节点列表与编码（NodeEncoder 顺序）
        nodes = self.node_encoder.collect_nodes(plan_tree, method="dfs")
        X_node = self.node_encoder.encode_nodes(nodes)   # list/np -> (N, D)
        X_node = torch.as_tensor(X_node, dtype=torch.float32)

        # B) 图结构 + 对齐
        # 优先尝试 return_perm 模式（如果你的 converter 支持）
        try:
            edge_index, x_list_or_perm = self.converter.tree_to_graph(
                plan_tree, return_perm=True  # 期望返回 (edge_index, perm)
            )
            # x_list_or_perm 应该是 perm
            perm = x_list_or_perm
            perm = torch.as_tensor(perm, dtype=torch.long)
            x = X_node.index_select(0, perm)
        except TypeError:
            # 不支持 return_perm=True，则退化为使用 x_list（仍然可跑）
            edge_index, x_list = self.converter.tree_to_graph(plan_tree)
            x = torch.stack(
                [torch.as_tensor(f, dtype=torch.float32) for f in x_list],
                dim=0
            )
            # 警示：此分支不会用到 NodeEncoder 的输出（X_node）

        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        # 自检：索引必须在范围内
        assert edge_index.dtype == torch.long
        assert edge_index.numel() >= 2, "edge_index 为空"
        assert edge_index.max().item() < x.size(0), \
            f"edge_index 索引越界：max={edge_index.max().item()} >= N={x.size(0)}"
        assert x.dim() == 2, f"x 维度应为 [N, D]，实际 {tuple(x.shape)}"

        # C) 图编码 -> 计划向量
        plan_emb = self.tree_encoder(x, edge_index)  # [D_plan] 或 [1, D_plan]

        # D) 预测头
        pred = self.head(plan_emb)  # 标量预测
        return pred


# ----------------------------
# 3) 划分数据集
# ----------------------------
N = len(plans_tree)
indices = list(range(N))
random.seed(42)
random.shuffle(indices)

ratio_train, ratio_valid = 0.8, 0.1
n_train = int(N * ratio_train)
n_valid = int(N * ratio_valid)
train_idx = indices[:n_train]
valid_idx = indices[n_train:n_train + n_valid]
test_idx  = indices[n_train + n_valid:]

def subset(arr, idxs):
    return [arr[i] for i in idxs]

train_trees = subset(plans_tree, train_idx)
valid_trees = subset(plans_tree, valid_idx)
test_trees  = subset(plans_tree,  test_idx)

train_y = subset(exec_times, train_idx)
valid_y = subset(exec_times, valid_idx)
test_y  = subset(exec_times,  test_idx)

# 目标做 log1p
def to_log_tensor(y_list):
    y = torch.as_tensor(y_list, dtype=torch.float32)
    return torch.log1p(y).view(-1, 1)

train_y_log = to_log_tensor(train_y)
valid_y_log = to_log_tensor(valid_y)
test_y_log  = to_log_tensor(test_y)

# ----------------------------
# 4) 简单 DataLoader（单图训练）
# ----------------------------
class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, trees, y_log):
        super().__init__()
        self.trees = trees
        self.y_log = y_log
    def __len__(self):
        return len(self.trees)
    def __getitem__(self, i):
        return self.trees[i], self.y_log[i]

batch_size = 1  # 你的 GATTreeEncoder 接口是单图；后续如需批处理可改 PyG Batch
train_ds = PlanDataset(train_trees, train_y_log)
valid_ds = PlanDataset(valid_trees, valid_y_log)
test_ds  = PlanDataset(test_trees,  test_y_log)

from torch.utils.data import DataLoader
def collate_keep_objects(batch):
    # batch: List[(plan_tree, y_log_tensor)]
    trees = [b[0] for b in batch]                      # 保留为 Python 列表（含多个 PlanNode）
    y     = torch.stack([b[1] for b in batch], dim=0)  # [B, 1]
    return trees, y

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                          collate_fn=collate_keep_objects, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False,
                          collate_fn=collate_keep_objects, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False,
                          collate_fn=collate_keep_objects, num_workers=0)

# ----------------------------
# 5) 初始化模型与优化器
# ----------------------------
nodeEncoder = NodeEncoder()
gatTreeEncoder = GATTreeEncoder(
    input_dim=64,
    hidden_dim=64,
    output_dim=64,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    pooling="mean",
)
predictionHead = PredictionHead()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PlanModel(nodeEncoder, gatTreeEncoder, predictionHead).to(device)

loss_fn = nn.MSELoss()

opt = torch.optim.AdamW([
    {"params": model.node_encoder.parameters(), "lr": 1e-3},
    {"params": model.tree_encoder.parameters(), "lr": 1e-3},
    {"params": model.head.parameters(),          "lr": 1e-3},
])

# ----------------------------
# 6) 训练/验证循环
# ----------------------------
def run_epoch(loader, train_mode=True, max_grad_norm=1.0):
    model.train() if train_mode else model.eval()
    total_loss, n = 0.0, 0

    for trees, y_log in loader:              # trees: List[PlanNode], y_log: [B,1]
        y_log = y_log.to(device)

        # 逐个图前向，累计 loss；也可以改你的 TreeEncoder 支持 PyG Batch 后一次性前向
        batch_losses = []
        preds = []
        for i, t in enumerate(trees):
            pred_i = model(t)                # 标量或 [1]
            if pred_i.dim() == 0:
                pred_i = pred_i.view(1, 1)
            elif pred_i.dim() == 1:
                pred_i = pred_i.view(-1, 1)
            preds.append(pred_i)

        pred_log = torch.cat(preds, dim=0)   # [B,1]
        loss = loss_fn(pred_log, y_log)

        if train_mode:
            opt.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(1, n)

def evaluate(loader):
    model.eval()
    mse_log, mae_time, n = 0.0, 0.0, 0
    with torch.no_grad():
        for plan_tree, y_log in loader:
            y_log = y_log.to(device)                # log1p(time)
            pred_log = model(plan_tree[0])
            if pred_log.dim() == 0:
                pred_log = pred_log.view(1, 1)
            elif pred_log.dim() == 1:
                pred_log = pred_log.view(-1, 1)

            # log 域 MSE
            mse = loss_fn(pred_log, y_log).item()
            mse_log += mse

            # 还原到 time 域做 MAE
            y_time = torch.expm1(y_log)
            p_time = torch.expm1(pred_log)
            mae = torch.mean(torch.abs(p_time - y_time)).item()
            mae_time += mae
            n += 1
    return mse_log / max(1, n), mae_time / max(1, n)

print("########################\n# 开始训练\n########################")
epochs = 20
best_val, best_state = float('inf'), None

for ep in range(1, epochs + 1):
    tr_loss = run_epoch(train_loader, train_mode=True)
    val_mse_log, val_mae_time = evaluate(valid_loader)

    if val_mse_log < best_val:
        best_val = val_mse_log
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[Epoch {ep:02d}] "
          f"train_loss(log-MSE)={tr_loss:.4f}  "
          f"valid_loss(log-MSE)={val_mse_log:.4f}  "
          f"valid_MAE(time)={val_mae_time:.4f}")

# ----------------------------
# 7) 测试评估
# ----------------------------
if best_state is not None:
    model.load_state_dict(best_state)
model.to(device)

test_mse_log, test_mae_time = evaluate(test_loader)
print("########################\n# 测试集指标\n########################")
print(f"test_loss(log-MSE)={test_mse_log:.4f}  test_MAE(time)={test_mae_time:.4f}")

# ----------------------------
# 8) 示例：打印前5条预测(反变换后)
# ----------------------------
print("########################\n# 示例预测 vs. 真实\n########################")
model.eval()
with torch.no_grad():
    for i in range(min(5, len(test_trees))):
        pred_log = model(test_trees[i])
        pred_time = torch.expm1(pred_log).item()
        true_time = test_y[i] if i < len(test_y) else float('nan')
        print(f"[{i}] pred_time={pred_time:.4f}  true_time={true_time:.4f}")
