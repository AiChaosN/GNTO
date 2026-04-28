import sys
import os
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from datetime import datetime

# === 路径配置 ===
GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from adapters.qf_adapter import (
    load_qf_resources, QueryFormerToPyGDataset,
    unnormalize, calc_q_error, evaluate
)

# QF 原版 FeatureEmbed 需要从 QueryFormer 仓库导入
res = load_qf_resources()
QF_PATH = res["qf_path"]
if QF_PATH not in sys.path:
    sys.path.insert(0, QF_PATH)
from model.model import FeatureEmbed

# === 模型定义: QueryFormer baseline (FeatureEmbed + GATConv) ===
class GNTO_QF(nn.Module):
    """QueryFormer baseline 复现: 使用 QF 原版 FeatureEmbed，不用 GNTO 的 NodeEncoder。"""
    def __init__(self, encoding, hidden_dim=64):
        super().__init__()

        num_types = len(encoding.idx2type)
        num_tables = len(encoding.idx2table)
        num_joins = len(encoding.idx2join)
        num_ops = len(encoding.idx2op)
        num_columns = len(encoding.idx2col)

        self.node_encoder = FeatureEmbed(
            embed_size=64,
            tables=num_tables,
            types=num_types,
            joins=num_joins,
            columns=num_columns,
            ops=num_ops,
            use_sample=True,
            use_hist=True,
            bin_number=50
        )

        self.input_dim = 64 * 5 + 64 // 8 + 1

        self.gnn = GATConv(self.input_dim, hidden_dim, heads=4, concat=False)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        x = self.gnn(x, data.edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

# === 主流程 ===
def main():
    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime("%m%d_%H%M")
    SAVE_DIR = f"../results/GNTO_QF_{timestamp}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoding = res["encoding"]
    cost_norm = res["cost_norm"]
    data_path = res["data_path"]

    print("Loading Datasets...")
    dfs = []
    for i in range(18):
        df = pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv')
        dfs.append(df)
    train_df = pd.concat(dfs)

    val_dfs = []
    for i in range(18, 20):
        df = pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv')
        val_dfs.append(df)
    val_df = pd.concat(val_dfs)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

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

    model = GNTO_QF(encoding).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Model Parameters ===")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"========================\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    print("Start Training...")
    best_q90 = float('inf')
    log_file = open(os.path.join(SAVE_DIR, "training_log.csv"), "w")
    log_file.write("epoch,loss,time,lr,grad_norm,train_q_50,train_q_75,train_q_90,train_q_95,train_q_99,val_q_50,val_q_75,val_q_90,val_q_95,val_q_99\n")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0

        collect_train_stats = (epoch % 10 == 0 or epoch == EPOCHS - 1)
        train_preds_all, train_targets_all = [], []
        grad_norm_val = 0.0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            grad_norm_val = grad_norm.item()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

            if collect_train_stats:
                pred_raw = unnormalize(out.detach().cpu().numpy(), cost_norm)
                target_raw = batch.raw_y.cpu().numpy()
                train_preds_all.extend(pred_raw)
                train_targets_all.extend(target_raw)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_ds)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            t_q50, t_q75, t_q90, t_q95, t_q99 = calc_q_error(train_preds_all, train_targets_all)

            val_loss, v_q50, v_q75, v_q90, v_q95, v_q99 = evaluate(model, val_loader, cost_norm, DEVICE)

            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | "
                  f"Val Q50: {v_q50:.2f} | Val Q90: {v_q90:.2f}")

            log_file.write(f"{epoch},{avg_loss},{epoch_time},{current_lr},{grad_norm_val},"
                           f"{t_q50},{t_q75},{t_q90},{t_q95},{t_q99},"
                           f"{v_q50},{v_q75},{v_q90},{v_q95},{v_q99}\n")
            log_file.flush()

            if v_q90 < best_q90:
                best_q90 = v_q90
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"  >>> New best model saved! (Val Q90: {v_q90:.2f})")

    log_file.close()
    print(f"Training finished. Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
