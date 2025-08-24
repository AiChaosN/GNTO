
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from lqo.train.losses import mse_log, pairwise_margin_loss

class PlanDataset(Dataset):
    def __init__(self, X: np.ndarray, y_ms: np.ndarray, extra: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_ms, dtype=torch.float32)
        self.extra = torch.tensor(extra, dtype=torch.float32) if extra is not None else None
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        if self.extra is None:
            return self.X[idx], self.y[idx], None
        return self.X[idx], self.y[idx], self.extra[idx]

@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    pairwise_weight: float = 0.0  # set >0 to enable ranking auxiliary loss
    log_interval: int = 50

def train(model: nn.Module, train_set: PlanDataset, valid_set: PlanDataset | None, cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb, eb in loader:
            xb, yb = xb.to(device), yb.to(device)
            eb = eb.to(device) if eb is not None else None
            pred_log = model(xb, eb)
            loss = mse_log(pred_log, yb)
            if cfg.pairwise_weight > 0.0:
                # randomly form small groups to compute pairwise ranking loss
                idx = torch.randperm(len(xb))[:min(64, len(xb))]
                loss += cfg.pairwise_weight * pairwise_margin_loss(pred_log[idx], yb[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % cfg.log_interval == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1
        # optional: evaluate on valid_set
        if valid_set is not None:
            model.eval()
            with torch.no_grad():
                Xv, yv, ev = next(iter(DataLoader(valid_set, batch_size=len(valid_set))))
                Xv, yv = Xv.to(device), yv.to(device)
                ev = ev.to(device) if ev is not None else None
                lv = mse_log(model(Xv, ev), yv).item()
                print(f"epoch {epoch} valid_logMSE {lv:.4f}")
