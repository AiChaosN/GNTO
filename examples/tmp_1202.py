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

# === 1. 路径配置 (请根据实际情况微调) ===
QF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../QueryFormer_VLDB2022"))
if QF_PATH not in sys.path:
    sys.path.append(QF_PATH)

try:
    from model.database_util import get_hist_file, get_job_table_sample, Encoding
    from model.dataset import PlanTreeDataset
    from model.model import FeatureEmbed
    from model.util import Normalizer
except ImportError as e:
    print(f"Error importing QueryFormer modules: {e}")
    sys.exit(1)

# === 2. 数据集定义 ===
class QueryFormerToPyGDataset(Dataset):
    def __init__(self, df, encoding, hist_file, table_sample, cost_norm):
        self.df = df.reset_index(drop=True) # 确保索引连续
        self.encoding = encoding
        self.hist_file = hist_file
        self.table_sample = table_sample
        self.cost_norm = cost_norm
        
        # 假的 Normalizer，只为了初始化
        dummy_norm = Normalizer(1, 100)
        
        self.qf_dataset = PlanTreeDataset(
            self.df, None, encoding, hist_file, dummy_norm, dummy_norm, 'cost', table_sample
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        json_str = self.df.iloc[idx]['json']
        plan = json.loads(json_str)['Plan']
        query_id = self.df.iloc[idx]['id']
        
        # 生成树特征
        root = self.qf_dataset.traversePlan(plan, query_id, self.encoding)
        node_dict = self.qf_dataset.node2dict(root)
        self.qf_dataset.treeNodes.clear() # 清理内存

        x = node_dict['features'] 
        adj = node_dict['adjacency_list']
        if len(adj) > 0:
            edge_index = adj.t().long()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        # 获取真实标签 (Execution Time)
        exec_time = json.loads(json_str)['Execution Time']
        
        # 归一化标签 (和 QueryFormer 保持一致)
        # QueryFormer 使用 log(val + 0.001) 然后 min-max 归一化
        # 这里我们为了简单对比，直接返回 log 值作为 target，或者使用 cost_norm
        # 为了严谨，我们使用 cost_norm 的逻辑手动处理:
        # labels = np.log(float(l) + 0.001)
        # labels_norm = (labels - min) / (max - min)
        
        y_val = np.log(float(exec_time) + 0.001)
        y_norm = (y_val - self.cost_norm.mini) / (self.cost_norm.maxi - self.cost_norm.mini)
        y_norm = np.minimum(y_norm, 1)
        y_norm = np.maximum(y_norm, 0.001)
        
        return Data(x=x, edge_index=edge_index, y=torch.tensor([y_norm], dtype=torch.float), raw_y=torch.tensor([exec_time], dtype=torch.float))

# === 3. 模型定义 ===
class GNTO_QF(nn.Module):
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
            nn.Sigmoid() # 因为标签归一化到了 [0, 1]
        )

    def forward(self, data):
        x = self.node_encoder(data.x) 
        x = self.gnn(x, data.edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

# === 4. 工具函数 ===
def unnormalize(y_norm, cost_norm):
    # 反归一化: norm -> log -> exp
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
    
    return np.percentile(qerrors, [50, 90, 95, 99])

def evaluate(model, loader, cost_norm, device):
    model.eval()
    preds_all, targets_all = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            # 反归一化计算 Q-Error
            pred_raw = unnormalize(out.cpu().numpy(), cost_norm)
            target_raw = batch.raw_y.cpu().numpy() # 使用原始标签
            
            preds_all.extend(pred_raw)
            targets_all.extend(target_raw)
            
    avg_loss = total_loss / len(loader.dataset)
    q50, q90, q95, q99 = calc_q_error(preds_all, targets_all)
    return avg_loss, q50, q90, q95, q99

# === 5. 主流程 ===
def main():
    # 配置
    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 50 # 根据需要调整
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 结果保存目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    SAVE_DIR = f"../results/GNTO_QF_{timestamp}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 加载资源
    print("Loading QueryFormer resources...")
    data_path = os.path.join(QF_PATH, 'data/imdb/')
    checkpoint_path = os.path.join(QF_PATH, 'checkpoints/encoding.pt')
    
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    table_sample = get_job_table_sample(data_path + 'train') 
    encoding_ckpt = torch.load(checkpoint_path)
    encoding = encoding_ckpt['encoding']
    
    # 定义 Normalizer (参考 QueryFormer TrainingV1.py 的参数)
    cost_norm = Normalizer(-3.61192, 12.290855)
    
    # 2. 加载全量数据
    print("Loading Datasets...")
    dfs = []
    # 加载前18个part作为训练集
    for i in range(18):
        df = pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv')
        dfs.append(df)
    train_df = pd.concat(dfs)
    
    # 加载后2个part作为验证集
    val_dfs = []
    for i in range(18, 20):
        df = pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv')
        val_dfs.append(df)
    val_df = pd.concat(val_dfs)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    train_ds = QueryFormerToPyGDataset(train_df, encoding, hist_file, table_sample, cost_norm)
    val_ds = QueryFormerToPyGDataset(val_df, encoding, hist_file, table_sample, cost_norm)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. 初始化模型
    model = GNTO_QF(encoding).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 4. 训练循环
    print("Start Training...")
    best_q90 = float('inf')
    log_file = open(os.path.join(SAVE_DIR, "training_log.csv"), "w")
    log_file.write("epoch,loss,val_loss,val_q50,val_q90,val_q95,time\n")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            
        avg_loss = total_loss / len(train_ds)
        
        # 验证
        val_loss, q50, q90, q95, q99 = evaluate(model, val_loader, cost_norm, DEVICE)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Q50: {q50:.2f} | Q90: {q90:.2f} | Q95: {q95:.2f} | Time: {epoch_time:.1f}s")
        
        log_file.write(f"{epoch},{avg_loss},{val_loss},{q50},{q90},{q95},{epoch_time}\n")
        log_file.flush()
        
        # 保存最佳模型
        if q90 < best_q90:
            best_q90 = q90
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  >>> New best model saved! (Q90: {q90:.2f})")
            
    log_file.close()
    print(f"Training finished. Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()