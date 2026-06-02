"""
Train + evaluate a GNTO (w/o Stats) variant aligned to the 0519 architecture.

Migration to the 0519 source-of-truth (see results/Summary_0519/MAIN_COMPARISON.md)
requires a clean GNTO (w/o Stats) number on the QF-100k validation split using the
SAME architecture/protocol as the 0519 GNTO (w/ Stats) model (GNTO_QF_0519_1607):

  - identical backbone: NodeEncoder_QF_AddPlanrows + GATv2TreeEncoder_V3 + PredictionHead_V2
  - identical protocol: parts 0-17 train, parts 18-19 val, batch 128, Adam lr 1e-3,
    StepLR(step=20, gamma=0.7), grad-clip 50, 100 epochs, best val_q90 checkpoint
  - ONLY difference vs. w/ Stats: histogram + sample-bitmap channels disabled
    (use_hist=False, use_sample=False); structural features + planner Plan Rows kept.

Param count is reported BOTH raw and "effective" (excluding the instantiated-but-
unused linearSample / linearHist layers, which receive no gradient when their flag
is off). The effective count is the honest size of a w/o-Stats deployment and is the
number to put in the paper's Table (tab:params).

This does NOT modify the shared adapter / NodeEncoder, so other scripts (ablation,
0519 eval) are unaffected.
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from adapters.qf_adapter import (
    load_qf_resources, QueryFormerToPyGDataset,
    unnormalize, calc_q_error, evaluate, evaluate_full,
)
from models.NodeEncoder import NodeEncoder_QF_AddPlanrows
from models.TreeEncoder import GATv2TreeEncoder_V3
from models.PredictionHead import PredictionHead_V2
from utils.benchmark import benchmark_pyg
from utils.plan_stats import join_counts_from_df, stratify_by_join_count


class GNTO_QF_Model_WoStats(nn.Module):
    """Exact mirror of adapters.qf_adapter.GNTO_QF_Model(add_plan_rows=True),
    but with the statistical channels disabled (use_sample=False, use_hist=False)."""

    def __init__(self, encoding, hidden_dim=64):
        super().__init__()
        num_types = len(encoding.idx2type)
        num_tables = len(encoding.idx2table)
        num_joins = len(encoding.idx2join)
        num_ops = len(encoding.idx2op)
        num_columns = len(encoding.idx2col)

        self.node_encoder = NodeEncoder_QF_AddPlanrows(
            embed_size=64, tables=num_tables, types=num_types, joins=num_joins,
            columns=num_columns, ops=num_ops,
            use_sample=False, use_hist=False, bin_number=50,
        )
        self.gnn = GATv2TreeEncoder_V3(
            in_dim=64, hidden_dim=hidden_dim, out_dim=hidden_dim,
            heads1=4, heads2=2, drop=0.0,
        )
        self.head = PredictionHead_V2(
            in_dim=hidden_dim, out_dim=1, hidden_dims=(64, 64), dropout=0.0,
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        x = self.gnn(x, data.edge_index, data.batch)
        return torch.sigmoid(self.head(x))


def param_counts(model):
    """Return (raw, effective) param counts; effective excludes the unused
    linearSample / linearHist stats layers (no gradient when flags are off)."""
    ne = model.node_encoder
    dead_ids = set()
    for sub in (getattr(ne, "linearSample", None), getattr(ne, "linearHist", None)):
        if sub is not None:
            for p in sub.parameters():
                dead_ids.add(id(p))
    raw = sum(p.numel() for p in model.parameters())
    eff = sum(p.numel() for p in model.parameters() if id(p) not in dead_ids)
    return int(raw), int(eff)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def main():
    SMOKE = os.environ.get("SMOKE", "0") == "1"
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 2 if SMOKE else 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    SAVE_DIR = os.path.join(GNTO_PATH, "results", f"GNTO_QF_wostats_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading QueryFormer resources...")
    res = load_qf_resources()
    encoding = res["encoding"]
    cost_norm = res["cost_norm"]
    data_path = res["data_path"]

    print("Loading datasets (parts 0-17 train, 18-19 val; using planrows cache)...")
    train_dfs = [pd.read_csv(data_path + f"plan_and_cost/train_plan_part{i}.csv") for i in range(18)]
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_dfs = [pd.read_csv(data_path + f"plan_and_cost/train_plan_part{i}.csv") for i in (18, 19)]
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    train_ds = QueryFormerToPyGDataset(
        train_df, encoding, res["hist_file"], res["table_sample"],
        cost_norm, cache_name="train_full", add_plan_rows=True,
    )
    val_ds = QueryFormerToPyGDataset(
        val_df, encoding, res["hist_file"], res["table_sample"],
        cost_norm, cache_name="val_full", add_plan_rows=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = GNTO_QF_Model_WoStats(encoding).to(DEVICE)
    raw, eff = param_counts(model)
    print(f"\n=== GNTO (w/o Stats) params ===\n  raw (incl. unused stats layers): {raw:,}\n  effective (reported in paper):   {eff:,}\n")

    if SMOKE:
        print("[SMOKE] one forward pass on a val batch...")
        b = next(iter(val_loader)).to(DEVICE)
        with torch.no_grad():
            out = model(b)
        print(f"[SMOKE] forward OK, out shape={tuple(out.shape)}; epochs set to {EPOCHS}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    print("Start training...")
    best_q90 = float("inf")
    best_path = os.path.join(SAVE_DIR, "best_model.pth")
    log_file = open(os.path.join(SAVE_DIR, "training_log.csv"), "w")
    log_file.write("epoch,loss,time,lr,grad_norm,train_q_50,train_q_75,train_q_90,train_q_95,train_q_99,val_q_50,val_q_75,val_q_90,val_q_95,val_q_99\n")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        collect = (epoch % 10 == 0 or epoch == EPOCHS - 1)
        tp, tt = [], []
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
            if collect:
                tp.extend(unnormalize(out.detach().cpu().numpy(), cost_norm))
                tt.extend(batch.raw_y.cpu().numpy())

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_ds)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if collect:
            t_q = calc_q_error(tp, tt)
            val_loss, v50, v75, v90, v95, v99 = evaluate(model, val_loader, cost_norm, DEVICE)
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.6f} | ValLoss {val_loss:.6f} | "
                  f"{epoch_time:.1f}s | LR {current_lr:.2e} | Val Q50 {v50:.3f} Q90 {v90:.3f} Q99 {v99:.2f}")
            log_file.write(f"{epoch},{avg_loss},{epoch_time},{current_lr},{grad_norm_val},"
                           f"{t_q[0]},{t_q[1]},{t_q[2]},{t_q[3]},{t_q[4]},{v50},{v75},{v90},{v95},{v99}\n")
            log_file.flush()
            if v90 < best_q90:
                best_q90 = v90
                torch.save(model.state_dict(), best_path)
                print(f"  >>> new best (Val Q90 {v90:.3f}) saved")
    log_file.close()

    # ---- Final evaluation with the full metric suite (mirrors 0519 recompute) ----
    print("\nReloading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False))
    model.eval()

    full = evaluate_full(model, val_loader, cost_norm, DEVICE)
    preds = full.pop("preds")
    targets = full.pop("targets")

    sample_batch = next(iter(val_loader))
    bench_gpu = benchmark_pyg(model, sample_batch, DEVICE, n_warmup=5, n_single=50, n_throughput=20)
    bench_cpu = benchmark_pyg(model.cpu(), sample_batch.cpu(), torch.device("cpu"),
                              n_warmup=5, n_single=50, n_throughput=20)
    model.to(DEVICE)

    join_counts = join_counts_from_df(val_df)
    n = min(len(preds), len(join_counts))
    strat = stratify_by_join_count(preds[:n], targets[:n], join_counts[:n])

    out = {
        "timestamp": timestamp,
        "label": "GNTO (w/o Stats)",
        "ckpt": best_path,
        "seed": SEED,
        "device": str(DEVICE),
        "n_val": int(n),
        "config": {"use_hist": False, "use_sample": False, "add_plan_rows": True,
                   "hidden_dim": 64, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "n_params_raw": raw,
        "n_params_effective": eff,
        "metrics": full,
        "benchmark_gpu": bench_gpu,
        "benchmark_cpu": bench_cpu,
        "stratified_by_joins": strat,
    }
    out_path = os.path.join(SAVE_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(_to_serializable(out), f, indent=2, default=str)
    print(f"\n=== DONE ===\nWrote {out_path}")
    print(f"params: raw={raw:,} effective={eff:,}")
    print("metrics:", {k: round(full[k], 4) for k in ("q50", "q75", "q90", "q95", "q99",
                                                       "spearman", "kendall", "pairwise_acc") if k in full})


if __name__ == "__main__":
    main()
