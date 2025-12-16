import sys
import os
import json
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GATConv, global_mean_pool
from datetime import datetime
from tqdm import tqdm  # 引入 tqdm 显示进度条

# === 1. 路径配置 (请根据实际情况微调) ===
QF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../QueryFormer_VLDB2022"))
if QF_PATH not in sys.path:
    sys.path.append(QF_PATH)

# Add GNTO root to path to import models
GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if GNTO_PATH not in sys.path:
    sys.path.append(GNTO_PATH)

try:
    from model.database_util import get_hist_file, get_job_table_sample, Encoding
    from model.dataset import PlanTreeDataset
    from model.model import FeatureEmbed
    from model.util import Normalizer
    
    # Import new models
    from models.TreeEncoder import GATv2TreeEncoder_V3
    from models.PredictionHead import PredictionHead_V2
    from models.NodeEncoder import NodeEncoder_QF # 导入新的 Encoder
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# === 2. 数据集定义 (优化版: 预处理缓存) ===
class QueryFormerToPyGDataset(Dataset):
    def __init__(self, df, encoding, hist_file, table_sample, cost_norm, cache_name=None):
        self.df = df.reset_index(drop=True)
        self.cost_norm = cost_norm
        
        # 尝试加载缓存
        self.data_list = []
        cache_file = None
        if cache_name:
             cache_dir = "../data/process/cache"
             os.makedirs(cache_dir, exist_ok=True)
             cache_file = os.path.join(cache_dir, f"{cache_name}_{len(df)}.pt")
             
             if os.path.exists(cache_file):
                 print(f"Loading cached data from {cache_file}...")
                 self.data_list = torch.load(cache_file)
                 return

        print(f"Preprocessing {len(df)} samples (this may take a while)...")
        
        # 假的 Normalizer，只为了初始化
        dummy_norm = Normalizer(1, 100)
        
        qf_dataset = PlanTreeDataset(
            self.df, None, encoding, hist_file, dummy_norm, dummy_norm, 'cost', table_sample
        )
        
        for idx in tqdm(range(len(self.df)), desc="Processing"):
            json_str = self.df.iloc[idx]['json']
            plan = json.loads(json_str)['Plan']
            query_id = self.df.iloc[idx]['id']
            
            # 生成树特征
            root = qf_dataset.traversePlan(plan, query_id, encoding)
            node_dict = qf_dataset.node2dict(root)
            qf_dataset.treeNodes.clear() # 清理内存

            x = node_dict['features'] 
            adj = node_dict['adjacency_list']
            if len(adj) > 0:
                edge_index = adj.t().long()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                
            # 获取真实标签 (Execution Time)
            exec_time = json.loads(json_str)['Execution Time']
            
            # 归一化标签
            y_val = np.log(float(exec_time) + 0.001)
            y_norm = (y_val - self.cost_norm.mini) / (self.cost_norm.maxi - self.cost_norm.mini)
            y_norm = np.minimum(y_norm, 1)
            y_norm = np.maximum(y_norm, 0.001)
            
            # 存储为 PyG Data 对象
            # 注意: 这里我们把 raw_y 也存进去，方便评估
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y_norm], dtype=torch.float))
            data.raw_y = torch.tensor([exec_time], dtype=torch.float)
            self.data_list.append(data)
        
        # 保存缓存
        if cache_file:
            print(f"Saving cache to {cache_file}...")
            torch.save(self.data_list, cache_file)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

# === 3. 模型定义 (保持不变) ===
class GNTO_QF(nn.Module):
    def __init__(self, encoding, hidden_dim=64):
        super().__init__()
        
        num_types = len(encoding.idx2type)
        num_tables = len(encoding.idx2table)
        num_joins = len(encoding.idx2join)
        num_ops = len(encoding.idx2op)
        num_columns = len(encoding.idx2col)

        self.node_encoder = NodeEncoder_QF(
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
        
        self.input_dim = 64 # NodeEncoder_QF 的输出已经是投影后的 embed_size (64)
        
        self.gnn = GATv2TreeEncoder_V3(
            in_dim=self.input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim, 
            heads1=4,
            heads2=2,
            drop=0.0
        )
        
        self.head = PredictionHead_V2(
            in_dim=hidden_dim,
            out_dim=1,
            hidden_dims=(64, 64),
            dropout=0.0
        )

    def forward(self, data):
        x = self.node_encoder(data.x) 
        x = self.gnn(x, data.edge_index, data.batch)
        return torch.sigmoid(self.head(x))

# === 4. 工具函数 (保持不变) ===
def unnormalize(y_norm, cost_norm):
    val = y_norm * (cost_norm.maxi - cost_norm.mini) + cost_norm.mini
    return np.exp(val) - 0.001

def calc_q_error(preds, targets):
    qerrors = []
    for p, t in zip(preds, targets):
        if p == 0 and t == 0: qerrors.append(1.0)
        elif p == 0: qerrors.append(float('inf'))
        elif t == 0: qerrors.append(float('inf'))
        else:
            qerrors.append(max(p/t, t/p))
    
    return np.percentile(qerrors, [50, 75, 90, 95, 99])

def evaluate(model, loader, cost_norm, DEVICE):
    model.eval()
    preds_all, targets_all = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            pred_raw = unnormalize(out.cpu().numpy(), cost_norm)
            target_raw = batch.raw_y.cpu().numpy()
            
            preds_all.extend(pred_raw)
            targets_all.extend(target_raw)
            
    avg_loss = total_loss / len(loader.dataset)
    q50, q75, q90, q95, q99 = calc_q_error(preds_all, targets_all)
    return avg_loss, q50, q75, q90, q95, q99

# === 5. 主流程 (略微调整 DataLoader 参数) ===
def main():
    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 100 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    SAVE_DIR = f"../results/GNTO_QF_{timestamp}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("Loading QueryFormer resources...")
    data_path = os.path.join(QF_PATH, 'data/imdb/')
    checkpoint_path = os.path.join(QF_PATH, 'checkpoints/encoding.pt')
    
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    table_sample = get_job_table_sample(data_path + 'train') 
    encoding_ckpt = torch.load(checkpoint_path)
    encoding = encoding_ckpt['encoding']
    
    cost_norm = Normalizer(-3.61192, 12.290855)
    
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
    
    # 使用缓存名称 train_full 和 val_full
    train_ds = QueryFormerToPyGDataset(train_df, encoding, hist_file, table_sample, cost_norm, cache_name="train_full")
    val_ds = QueryFormerToPyGDataset(val_df, encoding, hist_file, table_sample, cost_norm, cache_name="val_full")
    
    # 预处理已经完成，DataLoader 可以设 num_workers=0 或较小值
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = GNTO_QF(encoding).to(DEVICE)
    
    # 打印模型参数大小
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Model Parameters ===")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"========================\n")
    # === Model Parameters ===
    # Total Parameters: 598,639
    # Trainable Parameters: 598,639
    # ========================
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