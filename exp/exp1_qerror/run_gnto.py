"""Exp1 runner: GNTO Q-error on the shared split and workloads.

Same protocol as ``run_qf.py`` (bs 128, Adam 1e-3, StepLR(20, 0.7), 100 epochs,
MSE, clip 50, checkpoint on validation Q90).  The architecture comes from a
preset plus ``--set`` overrides; the default ``gnto`` preset is the QueryFormer
node features + Plan Rows channel + compositional join encoding, 3-layer GATv2.

Output dir defaults to ``results/exp1_qerror/<preset>`` (``_holdout`` appended
when ``--exclude-joins`` is used), so ablations never overwrite the main run::

    python exp/exp1_qerror/run_gnto.py                              # -> results/exp1_qerror/gnto
    python exp/exp1_qerror/run_gnto.py --preset gnto_joinid         # -> results/exp1_qerror/gnto_joinid   (string-id join ablation)
    python exp/exp1_qerror/run_gnto.py --exclude-joins "mi.movie_id = mi_idx.movie_id,ci.movie_id = mc.movie_id,mi_idx.movie_id = mk.movie_id"
                                                                    # -> results/exp1_qerror/gnto_holdout  (held-out join pairs)
    python exp/exp1_qerror/run_gnto.py --preset gnto --set graph.num_layers=5 --tag L5
    python exp/exp1_qerror/run_gnto.py --smoke
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
from common import (LEVELS, Logger, add_common_args, finalize_args, format_levels, prepare_data,  # noqa: E402
                    set_seed, workload_metrics, write_config, write_predictions, write_summary)
from adapters.qf_adapter import QueryFormerToPyGDataset, unnormalize  # noqa: E402
from models.gnto import GNTO, get_preset, list_presets  # noqa: E402
from models.gnto.node_encoder import qf_vocab_from_encoding  # noqa: E402
from models.qf import load_qf_resources  # noqa: E402
from models.qf.trainer import qerror_stats  # noqa: E402

MODEL = "GNTO"


@torch.no_grad()
def predict(model: GNTO, data_list, cost_norm, device, bs: int = 512):
    """``(pred_ms, actual_ms, val_mse)`` in ``data_list`` order."""
    preds, targets, loss_sum = [], [], 0.0
    for batch in DataLoader(data_list, batch_size=bs, shuffle=False):
        batch = batch.to(device)
        out = model.predict(batch)
        loss_sum += torch.nn.functional.mse_loss(out, batch.y.view(-1)).item() * batch.num_graphs
        preds.append(unnormalize(out.cpu().numpy(), cost_norm))
        targets.append(batch.raw_y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), loss_sum / max(len(data_list), 1)


def train(model: GNTO, train_list, val_list, cost_norm, device, a, out_dir: str, log) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=a.step_size, gamma=a.gamma)
    loader = DataLoader(train_list, batch_size=a.bs, shuffle=True)
    cols = (["epoch", "loss", "time", "lr", "grad_norm"] + [f"train_q{k}" for k in LEVELS] + ["train_q_mean", "val_loss"]
            + [f"val_q{k}" for k in LEVELS] + ["val_q_mean"])
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write(",".join(cols) + "\n")
    best, best_epoch, ckpt = float("inf"), -1, os.path.join(out_dir, "best_model.pt")
    for epoch in range(a.epochs):
        t0 = time.time()
        model.train()
        loss_sum, gn_sum, nb, tp, tt = 0.0, 0.0, 0, [], []
        collect = (epoch % a.eval_every == 0) or (epoch == a.epochs - 1)
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y.view(-1))
            loss.backward()
            gn_sum += torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
            optimizer.step()
            nb += 1
            loss_sum += loss.item() * batch.num_graphs
            if collect:
                tp.append(unnormalize(out.detach().cpu().numpy(), cost_norm))
                tt.append(batch.raw_y.view(-1).cpu().numpy())
        scheduler.step()
        if not collect:
            continue
        tq = qerror_stats(np.concatenate(tp), np.concatenate(tt))
        vp, vt, val_loss = predict(model, val_list, cost_norm, device)
        vq = qerror_stats(vp, vt)
        row = {"epoch": epoch, "loss": loss_sum / len(train_list), "time": time.time() - t0,
               "lr": optimizer.param_groups[0]["lr"], "grad_norm": gn_sum / max(nb, 1),
               **{f"train_{k}": v for k, v in tq.items()}, "val_loss": val_loss, **{f"val_{k}": v for k, v in vq.items()}}
        with open(log_path, "a") as f:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
        log(f"epoch {epoch:3d} | loss {row['loss']:.6f} | {row['time']:.1f}s | lr {row['lr']:.1e} | "
            f"val Q50 {vq['q50']:.3f} Q90 {vq['q90']:.3f} Q99 {vq['q99']:.2f}")
        if vq[a.select] < best:
            best, best_epoch = vq[a.select], epoch
            model.save(ckpt, extra={"epoch": epoch, "val": vq, "select": a.select})
            log(f"  >>> new best {a.select}={best:.4f} (epoch {epoch})")
    return {"best_epoch": best_epoch, "best_metric": best, "select": a.select, "epochs": a.epochs, "ckpt": ckpt,
            "trained_here": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="gnto", choices=list_presets(),
                    help="gnto (default: compositional joins) | gnto_joinid (string-id joins) | qf_bidir | qf_nostats | ...")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, e.g. graph.num_layers=5")
    add_common_args(ap, "results/exp1_qerror/<preset>")
    a = ap.parse_args()
    out_dir = finalize_args(a, a.preset)
    log = Logger(os.path.join(out_dir, "log.txt"))
    device = torch.device(a.device)

    res = load_qf_resources()
    enc, hist, cost_norm = res["encoding"], res["hist_file"], res["cost_norm"]
    cfg = get_preset(a.preset).override(node__vocab=qf_vocab_from_encoding(enc))
    if a.set:
        cfg = cfg.override(*a.set)
    write_config(out_dir, a, model=MODEL, preset=a.preset, gnto_config=cfg.to_dict())
    log(f"{MODEL} ({cfg.name}) -> {out_dir}  device {device}\n{cfg.describe()}")

    data = prepare_data(a, enc, log)
    feat = dict(add_plan_rows=cfg.node.plan_rows, add_join_cols=cfg.node.qf_join_cols, num_workers=a.num_workers)
    std_split = a.use_cache and not a.exclude_joins and a.train_parts == "0-17" and a.val_parts == "18-19"
    cache_tag = None
    if a.use_cache and not a.exclude_joins:
        cache_tag = "tmpl" if a.split == "template" else "full" if std_split else None
    timing = {}
    t0 = time.time()
    if not a.ckpt:
        train_list = QueryFormerToPyGDataset(data["train_df"], enc, hist, res["table_sample"], cost_norm,
                                             cache_name=f"train_{cache_tag}" if cache_tag else None, **feat).data_list
        val_list = QueryFormerToPyGDataset(data["val_df"], enc, hist, res["table_sample"], cost_norm,
                                           cache_name=f"val_{cache_tag}" if cache_tag else None, **feat).data_list
    eval_lists = {w: QueryFormerToPyGDataset(df, enc, hist, ts if ts is not None else res["table_sample"], cost_norm,
                                             cache_name=None, **feat).data_list
                  for w, (df, ts) in data["eval"].items()}
    timing["featurize_s"] = time.time() - t0

    set_seed(a.seed)
    model = GNTO.from_config(cfg).to(device)
    params = {"total": model.num_parameters(), "trainable": model.num_parameters(trainable_only=True)}
    log(f"params: {params['total']:,}\n{model.summary()}")

    if a.ckpt:
        model = GNTO.load(a.ckpt).to(device)
        if model.config.node.qf_input_dim != cfg.node.qf_input_dim:
            raise SystemExit("--ckpt feature layout differs from --preset; pass the matching preset")
        train_info = {"ckpt": a.ckpt, "trained_here": False}
        log(f"loaded {a.ckpt}")
    else:
        t0 = time.time()
        train_info = train(model, train_list, val_list, cost_norm, device, a, out_dir, log)
        timing["train_s"] = time.time() - t0
        model = GNTO.load(train_info["ckpt"]).to(device)
        log(f"best {a.select}={train_info['best_metric']:.4f} at epoch {train_info['best_epoch']}")

    per_workload = {}
    for w, data_list in eval_lists.items():
        t0 = time.time()
        preds, targets, _ = predict(model, data_list, cost_norm, device)
        timing[f"infer_{w}_s"] = time.time() - t0
        per_workload[w] = workload_metrics(preds, targets)
        write_predictions(out_dir, w, preds, targets, data["stats"][w])
        log(f"{w:10s} n={per_workload[w]['n']:5d} {format_levels(per_workload[w])}")

    write_summary(out_dir, MODEL, params, per_workload, train_info, timing,
                  extra={"preset": a.preset, "overrides": a.set, "holdout_joins": data["holdout"], "gnto_config": cfg.to_dict()})
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
