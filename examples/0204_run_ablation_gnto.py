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
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from datetime import datetime
from tqdm import tqdm

# === 1. 路径配置 ===
QF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../QueryFormer_VLDB2022"))
if QF_PATH not in sys.path:
    sys.path.append(QF_PATH)

# Add GNTO root to path
GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if GNTO_PATH not in sys.path:
    sys.path.append(GNTO_PATH)

try:
    from model.database_util import get_hist_file, get_job_table_sample, Encoding
    from model.dataset import PlanTreeDataset
    from model.util import Normalizer
    
    # Import GNTO models
    from models.TreeEncoder import GATv2TreeEncoder_V3
    from models.PredictionHead import PredictionHead_V2
    from models.NodeEncoder import NodeEncoder_QF
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# === 2. 数据集定义 (复用现有逻辑) ===
class QueryFormerToPyGDataset(Dataset):
    def __init__(self, df, encoding, hist_file, table_sample, cost_norm, cache_name=None):
        self.df = df.reset_index(drop=True)
        self.cost_norm = cost_norm
        
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

        print(f"Preprocessing {len(df)} samples...")
        dummy_norm = Normalizer(1, 100)
        qf_dataset = PlanTreeDataset(
            self.df, None, encoding, hist_file, dummy_norm, dummy_norm, 'cost', table_sample
        )
        
        for idx in tqdm(range(len(self.df)), desc="Processing"):
            json_str = self.df.iloc[idx]['json']
            plan = json.loads(json_str)['Plan']
            query_id = self.df.iloc[idx]['id']
            
            root = qf_dataset.traversePlan(plan, query_id, encoding)
            node_dict = qf_dataset.node2dict(root)
            qf_dataset.treeNodes.clear()

            x = node_dict['features'] 
            adj = node_dict['adjacency_list']
            if len(adj) > 0:
                edge_index = adj.t().long()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                
            exec_time = json.loads(json_str)['Execution Time']
            y_val = np.log(float(exec_time) + 0.001)
            y_norm = (y_val - self.cost_norm.mini) / (self.cost_norm.maxi - self.cost_norm.mini)
            y_norm = np.minimum(y_norm, 1)
            y_norm = np.maximum(y_norm, 0.001)
            
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y_norm], dtype=torch.float))
            data.raw_y = torch.tensor([exec_time], dtype=torch.float)
            self.data_list.append(data)
        
        if cache_file:
            print(f"Saving cache to {cache_file}...")
            torch.save(self.data_list, cache_file)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

# === 3. 支持消融实验的模型定义 ===
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
            # 不使用GNN，仅使用MLP进行特征转换（可选）或直接池化
            # 这里为了保持参数量有一定可比性，加一个线性层，或者直接Identity
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
            # 使用简单的线性层作为对比
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, data):
        x = self.node_encoder(data.x) 
        
        # GNN 处理分支
        if self.gnn_type == "GAT":
            # GATv2TreeEncoder_V3 内部包含了 pooling
            x = self.gnn(x, data.edge_index, data.batch)
            
        elif self.gnn_type == "GCN":
            # 标准 GCN 流程
            x = F.relu(self.conv1(x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = self.conv3(x, data.edge_index) # 最后一层通常不加激活或在pooling前加
            x = global_mean_pool(x, data.batch)
            
        elif self.gnn_type == "No_GNN":
            # 直接池化节点特征
            x = global_mean_pool(x, data.batch)
            
        return torch.sigmoid(self.head(x))

# === 4. 工具函数 ===
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

# === 5. 训练流程封装 ===
def run_experiment(config_name, use_hist, use_sample, gnn_type, head_type, train_loader, val_loader, encoding, cost_norm, device, epochs=50, save_root="../results/ablation"):
    print(f"\n>>> Running Experiment: {config_name} (Hist={use_hist}, Sample={use_sample}, GNN={gnn_type}, Head={head_type})")
    
    # 初始化模型
    model = GNTO_Ablation(encoding, use_hist=use_hist, use_sample=use_sample, gnn_type=gnn_type, head_type=head_type).to(device)
    
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

# === 6. 主程序 ===
def main():
    BATCH_SIZE = 128
    EPOCHS = 50  # 每个实验跑50轮，可根据需要调整
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    SAVE_ROOT = f"../results/Ablation_GNTO_{timestamp}"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    print("Loading QueryFormer resources...")
    data_path = os.path.join(QF_PATH, 'data/imdb/')
    checkpoint_path = os.path.join(QF_PATH, 'checkpoints/encoding.pt')
    
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    table_sample = get_job_table_sample(data_path + 'train') 
    encoding_ckpt = torch.load(checkpoint_path)
    encoding = encoding_ckpt['encoding']
    cost_norm = Normalizer(-3.61192, 12.290855)
    
    # 加载数据 (一次性加载)
    print("Loading Data...")
    dfs = [pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv') for i in range(15)]
    train_df = pd.concat(dfs)
    val_dfs = [pd.read_csv(data_path + f'plan_and_cost/train_plan_part{i}.csv') for i in range(15, 20)]
    val_df = pd.concat(val_dfs)
    
    # 构造Dataset (复用缓存)
    train_ds = QueryFormerToPyGDataset(train_df, encoding, hist_file, table_sample, cost_norm, cache_name="train_full")
    val_ds = QueryFormerToPyGDataset(val_df, encoding, hist_file, table_sample, cost_norm, cache_name="val_full")
    
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
