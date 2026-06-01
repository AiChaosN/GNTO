import sys
import os
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datetime import datetime

# === 路径配置 ===
GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from adapters.qf_adapter import (
    load_qf_resources, QueryFormerToPyGDataset,
    unnormalize, calc_q_error, evaluate
)
from models.TreeEncoder import GATv2TreeEncoder_V3
from models.PredictionHead import PredictionHead_V2
from models.NodeEncoder import NodeEncoder_QF


# === 支持消融实验的模型定义 ===
class GNTO_Ablation(nn.Module):
    def __init__(self, encoding, hidden_dim=64, use_hist=True, use_sample=True, gnn_type="GAT", head_type="Complex"):
        super().__init__()
        self.gnn_type = gnn_type

        num_types = len(encoding.idx2type)
        num_tables = len(encoding.idx2table)
        num_joins = len(encoding.idx2join)
        num_ops = len(encoding.idx2op)
        num_columns = len(encoding.idx2col)

        # 实例化 NodeEncoder_QF 并传入控制开关
        self.node_encoder = NodeEncoder_QF(
            embed_size=64,
            tables=num_tables,
            types=num_types,
            joins=num_joins,
            columns=num_columns,
            ops=num_ops,
            use_sample=use_sample,  # Ablation Control
            use_hist=use_hist,      # Ablation Control
            bin_number=50
        )

        self.input_dim = 64

        # GNN 模块消融
        if gnn_type == "GAT":
            self.gnn = GATv2TreeEncoder_V3(
                in_dim=self.input_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                heads1=4,
                heads2=2,
                drop=0.0
            )
        elif gnn_type == "GCN":
            # 使用3层GCN作为对比
            self.conv1 = GCNConv(self.input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == "No_GNN":
            # 不使用GNN，直接池化
            pass

        # Head 模块消融
        if head_type == "Complex":
            self.head = PredictionHead_V2(
                in_dim=hidden_dim,
                out_dim=1,
                hidden_dims=(64, 64),
                dropout=0.0
            )
        elif head_type == "Simple":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, data):
        x = self.node_encoder(data.x)

        # GNN 处理分支
        if self.gnn_type == "GAT":
            x = self.gnn(x, data.edge_index, data.batch)
        elif self.gnn_type == "GCN":
            x = F.relu(self.conv1(x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = self.conv3(x, data.edge_index)
            x = global_mean_pool(x, data.batch)
        elif self.gnn_type == "No_GNN":
            x = global_mean_pool(x, data.batch)

        return torch.sigmoid(self.head(x))


# === 训练流程封装 ===
def run_experiment(config_name, use_hist, use_sample, gnn_type, head_type,
                   train_loader, val_loader, encoding, cost_norm, device,
                   epochs=50, save_root="../results/ablation"):
    print(f"\n>>> Running Experiment: {config_name} (Hist={use_hist}, Sample={use_sample}, GNN={gnn_type}, Head={head_type})")

    model = GNTO_Ablation(encoding, use_hist=use_hist, use_sample=use_sample,
                          gnn_type=gnn_type, head_type=head_type).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    save_dir = os.path.join(save_root, config_name)
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "log.csv")
    with open(log_file_path, "w") as f:
        f.write("epoch,loss,val_loss,val_q50,val_q75,val_q90,val_q95,val_q99,time\n")

    best_q90 = float('inf')
    results = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step()
        epoch_time = time.time() - start_time

        # 验证
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_loss, q50, q75, q90, q95, q99 = evaluate(model, val_loader, cost_norm, device)

            print(f"[{config_name}] Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Val Q90: {q90:.2f} | Time: {epoch_time:.1f}s")

            with open(log_file_path, "a") as f:
                f.write(f"{epoch},{avg_loss},{val_loss},{q50},{q75},{q90},{q95},{q99},{epoch_time}\n")

            if q90 < best_q90:
                best_q90 = q90
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    final_loss, f_q50, f_q75, f_q90, f_q95, f_q99 = evaluate(model, val_loader, cost_norm, device)

    print(f"Completed {config_name}. Best Val Q90: {best_q90:.4f}")
    return {
        "config": config_name,
        "best_q90": best_q90,
        "final_q50": f_q50,
        "final_q75": f_q75,
        "final_q95": f_q95,
        "final_q99": f_q99
    }


# === 主程序 ===
def main():
    BATCH_SIZE = 128
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime("%m%d_%H%M")
    SAVE_ROOT = f"../results/Ablation_GNTO_{timestamp}"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("Loading QueryFormer resources...")
    res = load_qf_resources()
    encoding = res["encoding"]
    cost_norm = res["cost_norm"]
    data_path = res["data_path"]

    # 加载数据
    print("Loading Data...")
    dfs = [pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv') for i in range(15)]
    train_df = pd.concat(dfs)
    val_dfs = [pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv') for i in range(15, 20)]
    val_df = pd.concat(val_dfs)

    # 构造Dataset (复用缓存)
    train_ds = QueryFormerToPyGDataset(
        train_df, encoding, res["hist_file"], res["table_sample"],
        cost_norm, cache_name="train_full", add_plan_rows=False
    )
    val_ds = QueryFormerToPyGDataset(
        val_df, encoding, res["hist_file"], res["table_sample"],
        cost_norm, cache_name="val_full", add_plan_rows=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 定义实验配置
    experiments = [
        {"name": "Full_Model",    "hist": True,  "sample": True,  "gnn": "GAT",    "head": "Complex"}, # 完整版
        {"name": "No_Hist",       "hist": False, "sample": True,  "gnn": "GAT",    "head": "Complex"}, # 验证特征1
        {"name": "No_Sample",     "hist": True,  "sample": False, "gnn": "GAT",    "head": "Complex"}, # 验证特征2
        {"name": "Replace_GCN",   "hist": True,  "sample": True,  "gnn": "GCN",    "head": "Complex"}, # 验证结构1 (替换)
        {"name": "No_GNN",        "hist": True,  "sample": True,  "gnn": "No_GNN", "head": "Complex"}, # 验证结构2 (删除)
        {"name": "Simple_Head",   "hist": True,  "sample": True,  "gnn": "GAT",    "head": "Simple"},  # 验证Head模块
    ]

    all_results = []

    print(f"\nStarting GNTO Ablation Study with {len(experiments)} configurations...")
    print(f"Results will be saved to {SAVE_ROOT}")

    for exp in experiments:
        res = run_experiment(
            config_name=exp["name"],
            use_hist=exp["hist"],
            use_sample=exp["sample"],
            gnn_type=exp["gnn"],
            head_type=exp["head"],
            train_loader=train_loader,
            val_loader=val_loader,
            encoding=encoding,
            cost_norm=cost_norm,
            device=DEVICE,
            epochs=EPOCHS,
            save_root=SAVE_ROOT
        )
        all_results.append(res)

    # 汇总结果
    print("\n=== Ablation Study Summary ===")
    summary_df = pd.DataFrame(all_results)
    print(summary_df)
    summary_df.to_csv(os.path.join(SAVE_ROOT, "summary.csv"), index=False)
    print(f"\nFull results saved to {os.path.join(SAVE_ROOT, 'summary.csv')}")

if __name__ == "__main__":
    main()
