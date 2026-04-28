import sys
import os
import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# === 路径配置 ===
GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from adapters.dace_adapter import (
    load_dace_statistics, DaceWorkloadDataset, GNTO_DACE_Model,
    q_error_np, q_error_loss, WORKLOADS
)

def train_and_eval():
    print("Loading Statistics...")
    stats, workload_dir = load_dace_statistics()

    # 0-9 Train, 10-19 Test
    train_dbs = WORKLOADS[:10]
    test_dbs = WORKLOADS[10:]

    print(f"Train DBs: {train_dbs}")
    print(f"Test DBs: {test_dbs}")

    print("Preparing Datasets...")
    train_dataset = DaceWorkloadDataset(workload_dir, train_dbs, stats)
    test_dataset = DaceWorkloadDataset(workload_dir, test_dbs, stats)

    print(f"Train Size: {len(train_dataset)}")
    print(f"Test Size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Setup Model
    num_node_types = len(stats["node_types"]["value_dict"])
    input_dim = num_node_types + 2
    hidden_dim = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNTO_DACE_Model(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"Model parameters: {param_count:,}")
    print(f"Param size: {param_size_mb:.3f} MB")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    MAX_RUNTIME = 30000.0

    print("Starting Training...")
    epochs = 15

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            target_norm = batch.y / MAX_RUNTIME
            target_norm = torch.clamp(target_norm, 0, 1)

            pred = model(batch).view(-1)

            loss = q_error_loss(pred, target_norm.view(-1))
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

        db_dataset = DaceWorkloadDataset(workload_dir, [db_name], stats)

        if len(db_dataset) == 0:
            print(f"Skipping {db_name} (empty)")
            continue

        db_loader = DataLoader(db_dataset, batch_size=1024, shuffle=False)

        db_preds = []
        db_targets = []

        model.eval()
        with torch.no_grad():
            for batch in db_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)

                pred_time = pred * MAX_RUNTIME
                target_time = batch.y

                db_preds.append(pred_time.cpu().numpy())
                db_targets.append(target_time.cpu().numpy())

        db_preds = np.concatenate(db_preds)
        db_targets = np.concatenate(db_targets)

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
    avg_median = np.mean([m['50th'] for m in results_per_db.values()])
    avg_mean = np.mean([m['mean'] for m in results_per_db.values()])
    print(f"Average Median Q-Error: {avg_median:.4f}")
    print(f"Average Mean Q-Error: {avg_mean:.4f}")

    res_path = "0227_gnto_workload1_results.json"
    with open(res_path, 'w') as f:
        json.dump(results_per_db, f, indent=4)
    print(f"Detailed results saved to {res_path}")


if __name__ == "__main__":
    train_and_eval()
