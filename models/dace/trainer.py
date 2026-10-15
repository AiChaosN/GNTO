"""Plain-PyTorch training loop for DACE reproducing the upstream Lightning protocol
(``DACE/model.py::PL_DACE`` and its workload1 training script): Adam 1e-3, no scheduler,
loss = mean over the batch of the height-weighted per-node log-Q-error, validation =
mean root Q-error on a 10 % split of the training plans, checkpoint = lowest validation
value (``ModelCheckpoint(monitor="val_loss", mode="min")``)."""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import dace_loss, q_error
from .plan_utils import MAX_RUNTIME, DACEDataset

PERCENTILES = (50, 75, 90, 95, 99)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {"total": int(sum(p.numel() for p in model.parameters())),
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def load_checkpoint(model: nn.Module, path: str, device) -> nn.Module:
    """Load a state_dict saved by :func:`train` (unmerged LoRA weights) safely: the model
    must be in train mode (``merged == False``) when the weights are assigned."""
    model.train()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


@torch.no_grad()
def predict(model: nn.Module, ds: DACEDataset, device, bs: int = 512) -> np.ndarray:
    """Root-node predictions in ms."""
    model.eval()
    out: List[np.ndarray] = []
    for seqs, masks, _, _ in DataLoader(ds, batch_size=bs, shuffle=False):
        est = model(seqs.to(device), masks.to(device))
        out.append(est[:, 0].cpu().numpy())
    return np.concatenate(out).astype(np.float64) * MAX_RUNTIME if out else np.empty(0)


def qerror_stats(preds_ms, targets_ms, percentiles=PERCENTILES) -> Dict[str, float]:
    p = np.maximum(np.asarray(preds_ms, dtype=np.float64), 1e-7)
    t = np.maximum(np.asarray(targets_ms, dtype=np.float64), 1e-7)
    q = np.maximum(p / t, t / p)
    out = {f"q{k}": float(np.percentile(q, k)) for k in percentiles}
    out["q_mean"] = float(q.mean())
    out["q_max"] = float(q.max())
    return out


def train(model: nn.Module, train_ds: DACEDataset, val_ds: DACEDataset, device, *, epochs: int = 10,
          bs: int = 512, lr: float = 1e-3, out_dir: Optional[str] = None, seed: int = 42,
          log: Callable[[str], None] = print, select: str = "val_q_mean") -> Dict[str, object]:
    """``select="last"`` keeps the final epoch instead of the best validation epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)
    ckpt = os.path.join(out_dir, "best_model.pt") if out_dir else None
    log_path = os.path.join(out_dir, "training_log.csv") if out_dir else None
    cols = ["epoch", "loss", "time", "lr", "val_loss"] + [f"val_q{k}" for k in PERCENTILES] + ["val_q_mean"]
    if log_path:
        with open(log_path, "w") as f:
            f.write(",".join(cols) + "\n")
    history, best, best_epoch = [], float("inf"), -1
    val_targets = val_ds.root_ms
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for seqs, masks, loss_mask, run_times in loader:
            seqs, masks, loss_mask, run_times = (x.to(device) for x in (seqs, masks, loss_mask, run_times))
            optimizer.zero_grad()
            est = model(seqs, masks)
            loss = torch.mean(dace_loss(est, run_times, loss_mask))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(seqs)
            n += len(seqs)
        preds = predict(model, val_ds, device, bs)
        vq = qerror_stats(preds, val_targets)
        # upstream "val_loss" = mean root q-error on the normalised scale (identical ratio)
        row = {"epoch": epoch, "loss": loss_sum / max(n, 1), "time": time.time() - t0, "lr": lr,
               "val_loss": vq["q_mean"], **{f"val_{k}": v for k, v in vq.items() if k != "q_max"}}
        history.append(row)
        if log_path:
            with open(log_path, "a") as f:
                f.write(",".join(str(row[c]) for c in cols) + "\n")
        log(f"[DACE] epoch {epoch:2d} | loss {row['loss']:.4f} | {row['time']:.1f}s | "
            f"val Q50 {vq['q50']:.3f} Q90 {vq['q90']:.3f} mean {vq['q_mean']:.3f}")
        metric = row["val_q_mean"] if select == "last" else row[select]
        if select == "last" or metric < best:
            best, best_epoch = metric, epoch
            if ckpt:
                model.train()   # loralib merges the LoRA delta into the base weight in eval(); save it unmerged
                torch.save(model.state_dict(), ckpt)
            if select != "last":
                log(f"  >>> [DACE] new best {select}={best:.4f} (epoch {epoch})")
    return {"best_epoch": best_epoch, "best_metric": best, "select": select, "epochs": epochs,
            "history": pd.DataFrame(history), "ckpt": ckpt}


__all__ = ["count_parameters", "predict", "qerror_stats", "train", "load_checkpoint", "PERCENTILES"]
