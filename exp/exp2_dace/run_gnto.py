"""Exp2 runner: GNTO trained on 10 databases, tested zero-shot on the other 10.

Same databases, node features (22-d, via ``models.dace.NodeFeaturizer``), train /
validation plans and selection rule as ``run_dace.py``.  GNTO consumes the plan
as a graph (parent<->child edges); preset ``dace_zero_shot``: MLP node encoder ->
3-layer GATv2 (heads 4/4/1, dropout 0.1) -> mean readout -> residual head ->
sigmoid; loss = mean log-Q-error of the root prediction (target = runtime / 30 s),
Adam 1e-3, batch 128, 10 epochs, ReduceLROnPlateau(0.5, patience 3) on the train loss.

Writes ``results/exp2_dace/gnto_<split>_s<seed>/`` (contract in ``common.py``).

Usage::

    python exp/exp2_dace/run_gnto.py                              # paper split, seed 42
    python exp/exp2_dace/run_gnto.py --seed 7 --set graph.num_layers=5 --tag L5
    python exp/exp2_dace/run_gnto.py --split rand --split-seed 2
    python exp/exp2_dace/run_gnto.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE_DIR, LEVELS, Logger, add_common_args, db_metrics, finalize_args, format_levels,  # noqa: E402
                    index_positions, level_stats, load_benchmark, qerror, set_seed, write_config,
                    write_predictions, write_summary, write_benchmark_manifest)
from adapters.dace_adapter import MAX_RUNTIME, build_pyg_dataset, q_error_loss  # noqa: E402
from models.gnto import GNTO, get_preset, list_presets  # noqa: E402

MODEL = "GNTO"


@torch.no_grad()
def predict(model: GNTO, data_list, device, bs: int = 1024):
    preds, targets = [], []
    for batch in DataLoader(data_list, batch_size=bs, shuffle=False):
        batch = batch.to(device)
        preds.append(model.predict(batch).cpu().numpy() * MAX_RUNTIME)
        targets.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float64), np.concatenate(targets).astype(np.float64)


def train(model: GNTO, train_list, val_list, device, a, out_dir: str, log, select: str = "val_q_mean") -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loader = DataLoader(train_list, batch_size=a.bs, shuffle=True)
    cols = ["epoch", "loss", "time", "lr", "val_loss"] + [f"val_q{k}" for k in LEVELS] + ["val_q_mean"]
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write(",".join(cols) + "\n")
    best, best_epoch, ckpt = float("inf"), -1, os.path.join(out_dir, "best_model.pt")
    for epoch in range(a.epochs):
        t0 = time.time()
        model.train()
        loss_sum, nb = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            target = torch.clamp(batch.y.view(-1) / MAX_RUNTIME, 0, 1)
            loss = q_error_loss(model(batch).view(-1), target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            nb += 1
        avg_loss = loss_sum / max(nb, 1)
        scheduler.step(avg_loss)
        vp, vt = predict(model, val_list, device)
        vq = level_stats(qerror(vp, vt))
        row = {"epoch": epoch, "loss": avg_loss, "time": time.time() - t0, "lr": optimizer.param_groups[0]["lr"],
               "val_loss": vq["q_mean"], **{f"val_{k}": v for k, v in vq.items() if k != "q_max"}}
        with open(log_path, "a") as f:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
        log(f"[GNTO] epoch {epoch:2d} | loss {avg_loss:.4f} | {row['time']:.1f}s | lr {row['lr']:.1e} | "
            f"val Q50 {vq['q50']:.3f} Q90 {vq['q90']:.3f} mean {vq['q_mean']:.3f}")
        if select == "last" or row[select] < best:
            best, best_epoch = (row["val_q_mean"] if select == "last" else row[select]), epoch
            model.save(ckpt, extra={"epoch": epoch, "val": vq, "select": select})
            if select != "last":
                log(f"  >>> [GNTO] new best {select}={best:.4f} (epoch {epoch})")
    return {"best_epoch": best_epoch, "best_metric": best, "select": select, "epochs": a.epochs, "ckpt": ckpt,
            "trained_here": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="dace_zero_shot", choices=list_presets())
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    add_common_args(ap, default_bs=128)
    a = ap.parse_args()
    out_dir = finalize_args(a, "gnto")
    log = Logger(os.path.join(out_dir, "log.txt"))
    device = torch.device(a.device)

    bench = load_benchmark(a, log)
    write_benchmark_manifest(out_dir, bench)
    fz, plans = bench["fz"], bench["plans"]
    cfg = get_preset(a.preset).override(node__in_dim=fz.node_length)
    if a.set:
        cfg = cfg.override(*a.set)
    write_config(out_dir, a, model=MODEL, preset=a.preset, gnto_config=cfg.to_dict())
    log(f"{MODEL} ({cfg.name}) -> {out_dir}  device {device}\n{cfg.describe()}")

    timing = {}
    t0 = time.time()
    per_db = {db: build_pyg_dataset(plans[db], fz, db_id=i,
                                    cache_path=os.path.join(CACHE_DIR, bench["cache_id"], f"pyg_{db}.pt") if a.use_cache else None)
              for i, db in enumerate(bench["train_dbs"] + bench["test_dbs"])}
    sizes = {db: len(plans[db]) for db in plans}
    pool = [d for db in bench["train_dbs"] for d in per_db[db]]
    train_list = [pool[i] for i in index_positions(bench["train_index"], bench["train_dbs"], sizes)]
    val_list = [pool[i] for i in index_positions(bench["val_index"], bench["train_dbs"], sizes)]
    timing["featurize_s"] = time.time() - t0

    set_seed(a.seed)
    model = GNTO.from_config(cfg).to(device)
    params = {"total": model.num_parameters(), "trainable": model.num_parameters(trainable_only=True)}
    log(f"params: {params['total']:,}\n{model.summary()}")

    if a.ckpt:
        model = GNTO.load(a.ckpt).to(device)
        train_info = {"ckpt": a.ckpt, "trained_here": False}
        log(f"loaded {a.ckpt}")
    else:
        t0 = time.time()
        train_info = train(model, train_list, val_list, device, a, out_dir, log, select=a.select)
        timing["train_s"] = time.time() - t0
        model = GNTO.load(train_info["ckpt"]).to(device)
        log(f"best {train_info['select']}={train_info['best_metric']:.4f} at epoch {train_info['best_epoch']}")

    results = {}
    t0 = time.time()
    for db in bench["test_dbs"]:
        preds, targets = predict(model, per_db[db], device)
        results[db] = db_metrics(preds, targets)
        write_predictions(out_dir, db, preds, targets, [d.n_nodes for d in per_db[db]])
        log(f"{db:16s} n={results[db]['n']:5d} {format_levels(results[db])}")
    timing["infer_s"] = time.time() - t0
    s = write_summary(out_dir, MODEL, params, bench, results, train_info, timing,
                      extra={"preset": a.preset, "overrides": a.set, "gnto_config": cfg.to_dict()})
    log(f"average over test DBs: {format_levels(s['avg'])}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
