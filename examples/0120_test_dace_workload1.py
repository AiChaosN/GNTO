import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# Add GNTO root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models.TreeEncoder import GATv2TreeEncoder_V3
    from models.PredictionHead import PredictionHead_V2
except ImportError as e:
    print(f"Error importing GNTO models: {e}")
    sys.exit(1)

# DACE Paths
DACE_ROOT = "/home/AiChaosN/Project/Workspace/01_Research/DACE"
WORKLOAD1_DIR = os.path.join(DACE_ROOT, "data/workload1")
STATS_PATH = os.path.join(WORKLOAD1_DIR, "statistics.json")

WORKLOADS = [
    "accidents", "airline", "baseball", "basketball", "carcinogenesis",
    "consumer", "credit", "employee", "fhnk", "financial",
    "geneea", "genome", "hepatitis", "imdb_full", "movielens",
    "seznam", "ssb", "tournament", "tpc_h", "walmart"
]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_statistics():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"Statistics file not found at {STATS_PATH}. Please run DACE setup first.")
    return load_json(STATS_PATH)

def scale_value(val, stats, key):
    center = stats[key]["center"]
    scale = stats[key]["scale"]
    if scale == 0: return 0.0
    return (val - center) / scale

class DaceWorkloadDataset(InMemoryDataset):
    def __init__(self, root, db_names, stats, transform=None, pre_transform=None):
        self.db_names = db_names
        self.stats = stats
        self.node_type_dict = stats["node_types"]["value_dict"]
        self.num_node_types = len(self.node_type_dict)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []
        
        for db_name in self.db_names:
            file_path = os.path.join(self.root, f"{db_name}_filted.json")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            print(f"Processing {db_name}...")
            # DACE saves filtered plans as a list in a JSON file, or line-by-line?
            # Based on DACE setup.py: json.dump(filted_plans, f) -> It's a single JSON list
            try:
                plans = load_json(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            for plan_wrapper in tqdm(plans, desc=db_name, leave=False):
                # DACE structure: [{"Plan": ...}] or {"Plan": ...} ?
                # DACE setup.py: filtered_plans.append(plan[0][0][0]) -> looks deep
                # Let's handle generic case: verify "Plan" key
                
                # Check if it's a list (DACE sometimes wraps plans in lists)
                plan_content = plan_wrapper
                while isinstance(plan_content, list):
                    plan_content = plan_content[0]
                
                if "Plan" not in plan_content:
                    continue
                
                root_node = plan_content["Plan"]
                
                # Build Graph
                x, edge_index = self.plan_to_graph(root_node)
                
                # Target: Actual Total Time
                # DACE normalizes time by dividing by max_runtime (30000) or using log
                # DACE code: run_times = np.array(run_times).astype(np.float32) / configs["max_runtime"] + 1e-7
                # Here we will use raw time for now, and normalize in dataset or model if needed.
                # But to compare with DACE Q-Error, we should predict raw time eventually.
                # Let's store raw time in y.
                exec_time = root_node.get("Actual Total Time", 0.0)
                y = torch.tensor([exec_time], dtype=torch.float)
                
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
        
        return self.collate(data_list)

    def plan_to_graph(self, root):
        nodes = []
        edges = []
        
        def dfs(node, parent_idx):
            current_idx = len(nodes)
            
            # Feature Extraction
            # 1. Node Type (One-Hot)
            node_type = node.get("Node Type", "Unknown")
            type_idx = self.node_type_dict.get(node_type, 0) # Default to 0 or unknown?
            # Creating one-hot vector
            type_vec = [0] * self.num_node_types
            if type_idx < self.num_node_types:
                type_vec[type_idx] = 1
            
            # 2. Numerical Features (Cost, Rows)
            cost = node.get("Total Cost", 0.0)
            rows = node.get("Plan Rows", 0.0)
            
            # Robust Scaling using stats
            cost_scaled = scale_value(cost, self.stats, "Total Cost")
            rows_scaled = scale_value(rows, self.stats, "Plan Rows")
            
            # Combine features
            # Dim = num_node_types + 2
            feat_vec = type_vec + [cost_scaled, rows_scaled]
            nodes.append(feat_vec)
            
            # Add Edge
            if parent_idx is not None:
                # Undirected or directed? DACE uses Parent->Child in adjacency, then specific attention mask.
                # GNTO typically uses undirected or bidirectional GAT. 
                # Let's use bidirectional edges for GAT
                edges.append([parent_idx, current_idx])
                edges.append([current_idx, parent_idx])
            
            # Recurse
            if "Plans" in node:
                for child in node["Plans"]:
                    dfs(child, current_idx)
        
        dfs(root, None)
        
        x = torch.tensor(nodes, dtype=torch.float)
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        return x, edge_index

class GNTO_Workload1_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Simple Node Encoder: Linear projection
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # Extra layer
            nn.ReLU()
        )
        
        # Tree Encoder
        self.tree_encoder = GATv2TreeEncoder_V3(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            heads1=4,
            heads2=4,
            drop=0.1
        )
        
        # Prediction Head
        self.prediction_head = PredictionHead_V2(
            in_dim=hidden_dim,
            out_dim=1, # Predicting scalar time
            hidden_dims=(hidden_dim, hidden_dim) # Must be same dim for ResBlock chain in current V2 impl
        )
        
    def forward(self, data):
        x = self.node_encoder(data.x)
        g = self.tree_encoder(x, data.edge_index, data.batch)
        out = self.prediction_head(g)
        return torch.sigmoid(out) # DACE uses Sigmoid because it normalizes time to [0,1] or similar?
        # Wait, DACE normalizes time by dividing by 30000.
        # If we use Sigmoid, we are restricted to [0, 1].
        # We should match DACE's target normalization.

def q_error_np(preds, targets):
    qerrors = []
    for p, t in zip(preds, targets):
        if p == 0 and t == 0: qerrors.append(1.0)
        elif p == 0: qerrors.append(float('inf'))
        elif t == 0: qerrors.append(float('inf'))
        else:
            qerrors.append(max(p/t, t/p))
    return np.array(qerrors)

def train_and_eval():
    print("Loading Statistics...")
    stats = get_statistics()
    
    # 0-9 Train, 10-19 Test
    train_dbs = WORKLOADS[:10]
    test_dbs = WORKLOADS[10:]
    
    print(f"Train DBs: {train_dbs}")
    print(f"Test DBs: {test_dbs}")
    
    print("Preparing Datasets...")
    # We pass WORKLOAD1_DIR as root
    train_dataset = DaceWorkloadDataset(WORKLOAD1_DIR, train_dbs, stats)
    test_dataset = DaceWorkloadDataset(WORKLOAD1_DIR, test_dbs, stats)
    
    print(f"Train Size: {len(train_dataset)}")
    print(f"Test Size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Setup Model
    num_node_types = len(stats["node_types"]["value_dict"])
    input_dim = num_node_types + 2
    hidden_dim = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNTO_Workload1_Model(input_dim, hidden_dim, 1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # DACE-like Q-Error Loss
    # DACE: log(max(est/run, run/est))
    # Inputs are normalized to [0, 1] (plus epsilon)
    def q_error_loss(preds, targets):
        # Add epsilon to avoid division by zero
        preds = torch.clamp(preds, min=1e-7)
        targets = torch.clamp(targets, min=1e-7)
        
        q = torch.max(preds / targets, targets / preds)
        return torch.mean(torch.log(q))

    criterion = q_error_loss
    
    MAX_RUNTIME = 30000.0
    
    print("Starting Training...")
    epochs = 15 # Optimized for speed/convergence trade-off
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Normalize target to [0, 1] for Sigmoid
            target_norm = batch.y / MAX_RUNTIME
            target_norm = torch.clamp(target_norm, 0, 1)
            
            pred = model(batch).view(-1)
            
            loss = criterion(pred, target_norm.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)
        
    # Evaluation per database
    print("\n=== Evaluating on Individual Test Databases ===")
    results_per_db = {}
    
    for db_name in test_dbs:
        print(f"Evaluating {db_name}...")
        
        # Create a temporary dataset for this specific DB
        db_dataset = DaceWorkloadDataset(WORKLOAD1_DIR, [db_name], stats)
        
        if len(db_dataset) == 0:
            print(f"Skipping {db_name} (empty)")
            continue
            
        db_loader = DataLoader(db_dataset, batch_size=1024, shuffle=False) # Increased batch size for faster eval
        
        db_preds = []
        db_targets = []
        
        model.eval()
        with torch.no_grad():
            for batch in db_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                
                # Un-normalize
                pred_time = pred * MAX_RUNTIME
                target_time = batch.y
                
                db_preds.append(pred_time.cpu().numpy())
                db_targets.append(target_time.cpu().numpy())
        
        db_preds = np.concatenate(db_preds)
        db_targets = np.concatenate(db_targets)
        
        # Ensure no zero
        db_preds = np.maximum(db_preds, 1e-7)
        db_targets = np.maximum(db_targets, 1e-7)
        
        q_errs = q_error_np(db_preds, db_targets)
        
        metrics = {
            "mean": float(np.mean(q_errs)),
            "50th": float(np.quantile(q_errs, 0.5)),
            "90th": float(np.quantile(q_errs, 0.9)),
            "95th": float(np.quantile(q_errs, 0.95)),
            "99th": float(np.quantile(q_errs, 0.99)),
            "max": float(np.max(q_errs))
        }
        
        results_per_db[db_name] = metrics
        print(f"Results for {db_name}: Median={metrics['50th']:.4f}, Mean={metrics['mean']:.4f}")

    print("\n=== Final Summary (Average across test DBs) ===")
    # Calculate average median and mean across all test DBs
    avg_median = np.mean([m['50th'] for m in results_per_db.values()])
    avg_mean = np.mean([m['mean'] for m in results_per_db.values()])
    print(f"Average Median Q-Error: {avg_median:.4f}")
    print(f"Average Mean Q-Error: {avg_mean:.4f}")
    
    # Save results to match DACE output format
    res_path = "gnto_workload1_results.json"
    with open(res_path, 'w') as f:
        json.dump(results_per_db, f, indent=4)
    print(f"Detailed results saved to {res_path}")



if __name__ == "__main__":
    train_and_eval()
