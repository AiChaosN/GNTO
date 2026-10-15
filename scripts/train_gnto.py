"""Config-driven GNTO training on the QueryFormer plan corpus.

Every architectural choice comes from a preset plus ``--set`` overrides, so an
ablation is one command line and the exact config is saved next to the
checkpoint (``config.json`` + ``best_model.pt``, loadable with ``GNTO.load``)::

    # default model (3-layer GATv2, heads 4/2/1, compositional join encoding)
    python scripts/train_gnto.py --preset gnto

    # string-id join ablation (QueryFormer-style per-string join vocabulary)
    python scripts/train_gnto.py --preset gnto_joinid

    # depth / edge-direction ablation
    python scripts/train_gnto.py --preset gnto \
        --set graph.num_layers=5 --set graph.heads=4 --set graph.edge_mode=bidirectional

    # virtual node (1-hop global context) + attention readout, Huber loss
    python scripts/train_gnto.py --preset gnto \
        --set graph.virtual_node=true --set readout.pooling=attention --loss huber

    # pairwise ranking objective on top of MSE (groups = ``template_id`` attribute
    # of the cached Data objects if present, otherwise the whole batch is one group)
    python scripts/train_gnto.py --preset gnto --loss mse+pairwise

    # plumbing test on a cached split (no QueryFormer repo needed)
    python scripts/train_gnto.py --preset gnto_joinid --smoke \
        --train-cache data/process/cache/tmpl_val_planrows_9999.pt \
        --val-cache   data/process/cache/tmpl_test_planrows_9999.pt

Data: by default ``data/train_plan_part*.csv`` are featurised through
``adapters.qf_adapter`` with the QueryFormer resources in ``data/qf`` (cached
under ``data/process/cache``; the cache name is ``<cache-name>[_planrows]_<n>.pt``).  ``--train-cache`` / ``--val-cache`` load
an existing cache file directly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from models.gnto import GNTO, GNTOConfig, get_preset, build_loss, list_presets, qerror_percentiles  # noqa: E402
from models.gnto.node_encoder import qf_vocab_from_encoding  # noqa: E402
from models.qf import load_qf_resources  # noqa: E402

# QueryFormer's cost Normalizer(-3.61192, 12.290855): targets are min-max
# normalised log(exec_time + 0.001); the sigmoid head predicts in that space.
COST_MIN, COST_MAX = -3.61192, 12.290855


def unnormalize(y_norm: np.ndarray) -> np.ndarray:
    return np.exp(y_norm * (COST_MAX - COST_MIN) + COST_MIN) - 0.001


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_range(spec: str) -> list[int]:
    """'0-14' -> [0..14]; '0,3,5' -> [0,3,5]."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_split(args, res):
    """Returns (train_list, val_list) of PyG Data objects."""
    if args.train_cache or args.val_cache:
        if not (args.train_cache and args.val_cache):
            raise SystemExit("--train-cache and --val-cache must be given together")
        train = torch.load(args.train_cache, weights_only=False)
        val = torch.load(args.val_cache, weights_only=False)
        return list(train), list(val)

    import pandas as pd
    from adapters.qf_adapter import QueryFormerToPyGDataset
    from models.qf import plan_csv_path
    plan_rows, join_cols = args.cfg.node.plan_rows, args.cfg.node.qf_join_cols

    def load_parts(parts, cache_name):
        df = pd.concat([pd.read_csv(plan_csv_path(i)) for i in parts])
        ds = QueryFormerToPyGDataset(df, res["encoding"], res["hist_file"], res["table_sample"],
                                     res["cost_norm"], cache_name=cache_name, add_plan_rows=plan_rows,
                                     add_join_cols=join_cols)
        return list(ds.data_list)

    return (load_parts(parse_range(args.train_parts), args.train_cache_name),
            load_parts(parse_range(args.val_parts), args.val_cache_name))


@torch.no_grad()
def evaluate(model: GNTO, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, targets, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        loss_sum += torch.nn.functional.mse_loss(out, batch.y.view(-1)).item() * batch.num_graphs
        n += batch.num_graphs
        preds.append(unnormalize(out.cpu().numpy()))
        targets.append(batch.raw_y.view(-1).cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    q = qerror_percentiles(preds, targets, percentiles=(50, 75, 90, 95, 99))
    q["val_loss"] = loss_sum / max(n, 1)
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="gnto", choices=list_presets())
    ap.add_argument("--config", help="JSON config file (overrides --preset)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, e.g. graph.num_layers=5 (repeatable)")
    ap.add_argument("--loss", default="mse", help="mse | huber | l1 | pairwise | mse+pairwise | huber+pairwise")
    ap.add_argument("--huber-delta", type=float, default=1.0 / (COST_MAX - COST_MIN),
                    help="Huber delta in normalised units (default: 1.0 in log-space)")
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-size", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--clip", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--select", default="q90", choices=["q50", "q75", "q90", "q95", "q99", "val_loss"],
                    help="validation metric used to keep the best checkpoint")
    ap.add_argument("--train-parts", default="0-17")
    ap.add_argument("--val-parts", default="18-19")
    ap.add_argument("--train-cache-name", default="train_full")
    ap.add_argument("--val-cache-name", default="val_full")
    ap.add_argument("--train-cache", help="load a cached list of PyG Data directly")
    ap.add_argument("--val-cache")
    ap.add_argument("--out", help="output dir (default results/<preset>_<timestamp>)")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on <= 1024 graphs, just to test the plumbing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ---- config -----------------------------------------------------------
    cfg = GNTOConfig.load(args.config) if args.config else get_preset(args.preset)
    res = None
    if cfg.node.kind == "queryformer" and not (args.train_cache and args.val_cache):
        res = load_qf_resources()
        cfg = cfg.override(node__vocab=qf_vocab_from_encoding(res["encoding"]))
    if args.set:
        cfg = cfg.override(*args.set)
    args.cfg = cfg

    out_dir = args.out or os.path.join(GNTO_PATH, "results", f"{cfg.name}_{datetime.now():%m%d_%H%M}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.save(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "cfg"}, f, indent=2)

    # ---- data -------------------------------------------------------------
    train_list, val_list = load_split(args, res)
    if args.smoke:
        train_list, val_list, args.epochs, args.eval_every = train_list[:1024], val_list[:512], 2, 1
    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=args.batch_size, shuffle=False)
    print(f"train graphs: {len(train_list):,}  val graphs: {len(val_list):,}  device: {device}")

    # ---- model ------------------------------------------------------------
    model = GNTO.from_config(cfg).to(device)
    print(model.summary())
    criterion = build_loss(args.loss, delta=args.huber_delta, rank_weight=args.rank_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w") as log:
        log.write("epoch,loss,time,lr,val_loss,val_q50,val_q75,val_q90,val_q95,val_q99\n")
    best, best_epoch = float("inf"), -1

    for epoch in range(args.epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            groups = getattr(batch, "template_id", None)
            loss = criterion(out, batch.y.view(-1), groups) if "pairwise" in args.loss else criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss_sum += loss.item() * batch.num_graphs
        scheduler.step()
        train_loss = loss_sum / len(train_list)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            q = evaluate(model, val_loader, device)
            with open(log_path, "a") as log:
                log.write(f"{epoch},{train_loss:.6f},{time.time() - t0:.1f},{optimizer.param_groups[0]['lr']:.2e},"
                          f"{q['val_loss']:.6f},{q['q50']:.4f},{q['q75']:.4f},{q['q90']:.4f},{q['q95']:.4f},{q['q99']:.4f}\n")
            print(f"epoch {epoch:3d} | loss {train_loss:.5f} | val Q50 {q['q50']:.3f} Q90 {q['q90']:.3f} "
                  f"Q99 {q['q99']:.2f} | {time.time() - t0:.1f}s")
            if q[args.select] < best:
                best, best_epoch = q[args.select], epoch
                model.save(os.path.join(out_dir, "best_model.pt"),
                           extra={"epoch": epoch, "val": q, "select": args.select, "loss": args.loss})
                print(f"  >>> saved best ({args.select}={best:.4f})")

    print(f"done. best {args.select}={best:.4f} at epoch {best_epoch}; outputs in {out_dir}")


if __name__ == "__main__":
    main()
