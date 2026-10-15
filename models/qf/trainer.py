"""Training / evaluation loops for QueryFormer (adapted from upstream ``model/trainer.py``).

Training protocol of the QueryFormer baseline:
bs 128, Adam 1e-3, StepLR(20, 0.7), 100 epochs, MSE on min-max normalised
log latency, grad-clip 50.  ``train`` logs every ``eval_every`` epochs (and the
last one) to ``<out_dir>/training_log.csv`` and keeps the checkpoint with the
best validation ``select`` metric at ``<out_dir>/best_model.pt`` (a plain
``state_dict``).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .database_util import Batch, collator

PERCENTILES = (50, 75, 90, 95, 99)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def qerror_stats(preds_unnorm, labels_unnorm, percentiles: Sequence[int] = PERCENTILES) -> Dict[str, float]:
    """Q-error = max(pred/actual, actual/pred); returns ``{'q50': .., 'q75': .., ..., 'q_mean': ..}``."""
    p = np.maximum(np.asarray(preds_unnorm, dtype=np.float64), 1e-10)
    t = np.maximum(np.asarray(labels_unnorm, dtype=np.float64), 1e-10)
    q = np.maximum(p / t, t / p)
    out = {f"q{k}": float(np.percentile(q, k)) for k in percentiles}
    out["q_mean"] = float(q.mean())
    return out


def iter_batches(ds, bs: int, idxs: Optional[Sequence[int]] = None) -> Iterator[Tuple[Batch, Tuple]]:
    """Yield ``(Batch, (cost_labels, card_labels))`` over ``ds`` in chunks of ``bs``."""
    if idxs is None:
        idxs = range(len(ds))
    idxs = list(idxs)
    for s in range(0, len(idxs), bs):
        chunk = idxs[s:s + bs]
        batch, labels = collator(list(zip(*[ds[j] for j in chunk])))
        yield batch, labels


@torch.no_grad()
def predict(model: nn.Module, ds, device, bs: int = 512) -> np.ndarray:
    """Normalised cost predictions (``sigmoid`` space) for every plan in ``ds``."""
    model.eval()
    out: List[np.ndarray] = []
    for batch, _ in iter_batches(ds, bs):
        cost_preds, _ = model(batch.to(device))
        out.append(cost_preds.view(-1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def evaluate(model: nn.Module, ds, cost_norm, device, bs: int = 512,
             percentiles: Sequence[int] = PERCENTILES) -> Tuple[Dict[str, float], np.ndarray]:
    """``(qerror_stats, preds_ms)`` on the raw execution-time scale."""
    preds = cost_norm.unnormalize_labels(predict(model, ds, device, bs))
    return qerror_stats(preds, ds.costs, percentiles), preds


def train(model: nn.Module, train_ds, val_ds, cost_norm, device, *, epochs: int = 100, bs: int = 128,
          lr: float = 1e-3, clip: float = 50.0, step_size: int = 20, gamma: float = 0.7,
          eval_every: int = 10, select: str = "q90", out_dir: Optional[str] = None, seed: int = 0,
          log: Callable[[str], None] = print) -> Dict[str, object]:
    """Train ``model``; returns ``{'best_epoch', 'best_metric', 'history' (DataFrame), 'ckpt'}``."""
    crit = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    rng = np.random.default_rng(seed)

    ckpt = os.path.join(out_dir, "best_model.pt") if out_dir else None
    log_path = os.path.join(out_dir, "training_log.csv") if out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("epoch,loss,time,lr,grad_norm,"
                    + ",".join(f"train_q{k}" for k in PERCENTILES) + ",train_q_mean,"
                    + "val_loss," + ",".join(f"val_q{k}" for k in PERCENTILES) + ",val_q_mean\n")

    history: List[Dict[str, float]] = []
    best, best_epoch = float("inf"), -1
    train_costs = np.asarray(train_ds.costs, dtype=np.float64)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, grad_norm_sum, n_batches = 0.0, 0.0, 0
        collect = (epoch % eval_every == 0) or (epoch == epochs - 1)
        idxs = rng.permutation(len(train_ds))
        preds_norm: List[np.ndarray] = []

        for batch, labels in iter_batches(train_ds, bs, idxs):
            optimizer.zero_grad()
            y = torch.as_tensor(np.asarray([float(l[0]) for l in labels]), dtype=torch.float32, device=device)
            cost_preds, _ = model(batch.to(device))
            cost_preds = cost_preds.view(-1)
            loss = crit(cost_preds, y)
            loss.backward()
            grad_norm_sum += torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            optimizer.step()
            n_batches += 1
            loss_sum += loss.item() * len(y)
            if collect:
                preds_norm.append(cost_preds.detach().cpu().numpy())

        epoch_time = time.time() - t0
        avg_loss = loss_sum / len(train_ds)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if not collect:
            continue
        tq = qerror_stats(cost_norm.unnormalize_labels(np.concatenate(preds_norm)), train_costs[idxs])
        val_pred_norm = predict(model, val_ds, device)
        val_loss = float(np.mean((val_pred_norm - val_ds.cost_labels.numpy()) ** 2))
        vq = qerror_stats(cost_norm.unnormalize_labels(val_pred_norm), val_ds.costs)
        row = {"epoch": epoch, "loss": avg_loss, "time": epoch_time, "lr": cur_lr,
               "grad_norm": grad_norm_sum / max(n_batches, 1),
               **{f"train_{k}": v for k, v in tq.items()}, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in vq.items()}}
        history.append(row)
        log(f"[QF] epoch {epoch:3d} | loss {avg_loss:.6f} | {epoch_time:.1f}s | lr {cur_lr:.1e} | "
            f"val Q50 {vq['q50']:.3f} Q90 {vq['q90']:.3f} Q99 {vq['q99']:.2f}")
        if log_path:
            with open(log_path, "a") as f:
                f.write(",".join(str(row[c]) for c in
                                 ["epoch", "loss", "time", "lr", "grad_norm"]
                                 + [f"train_q{k}" for k in PERCENTILES] + ["train_q_mean", "val_loss"]
                                 + [f"val_q{k}" for k in PERCENTILES] + ["val_q_mean"]) + "\n")
        metric = vq[select] if select in vq else val_loss
        if metric < best:
            best, best_epoch = metric, epoch
            if ckpt:
                torch.save(model.state_dict(), ckpt)
            log(f"  >>> [QF] new best {select}={best:.4f} (epoch {epoch})")

    return {"best_epoch": best_epoch, "best_metric": best, "history": pd.DataFrame(history), "ckpt": ckpt}


__all__ = ["train", "evaluate", "predict", "qerror_stats", "iter_batches", "count_parameters", "PERCENTILES"]
