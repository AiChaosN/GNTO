"""Exp2 runner: DACE (ICDE'24) trained on 10 databases, tested zero-shot on the other 10.

Protocol: DACELora (https://github.com/liang-zibo/DACE) with hidden 128,
dropout 0.3 / 0.2, per-node height-weighted log-Q-error loss, Adam 1e-3,
batch 512, 10 epochs, 90 / 10 train / validation split of the training
databases' plans, checkpoint with the lowest validation mean Q-error; test =
root prediction on every plan of each test database, reported per database.

Writes ``results/exp2_dace/dace_<split>_s<seed>/`` (contract in ``common.py``).

Usage::

    python exp/exp2_dace/run_dace.py                              # paper split, seed 42
    python exp/exp2_dace/run_dace.py --seed 7
    python exp/exp2_dace/run_dace.py --split rand --split-seed 1
    python exp/exp2_dace/run_dace.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE_DIR, Logger, add_common_args, db_metrics, finalize_args, format_levels,  # noqa: E402
                    index_positions, load_benchmark, set_seed, write_config, write_predictions, write_summary, write_benchmark_manifest)
from models.dace import (DACE_DEFAULT_HPARAMS, DACEDataset, DACELora, build_dace_dataset, count_parameters,  # noqa: E402
                         load_checkpoint, predict)
from models.dace import trainer as dace_trainer  # noqa: E402

MODEL = "DACE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-only", action="store_true",
                    help="ablation: supervise only the root node (GNTO-style) instead of every sub-plan")
    add_common_args(ap, default_bs=512)
    a = ap.parse_args()
    if a.root_only and not a.tag:
        a.tag = "rootonly"
    out_dir = finalize_args(a, "dace")
    log = Logger(os.path.join(out_dir, "log.txt"))
    device = torch.device(a.device)
    write_config(out_dir, a, model=MODEL, hparams=dict(DACE_DEFAULT_HPARAMS))
    log(f"{MODEL} -> {out_dir}  device {device}")

    bench = load_benchmark(a, log)
    write_benchmark_manifest(out_dir, bench)
    fz, plans = bench["fz"], bench["plans"]
    timing = {}
    t0 = time.time()
    per_db = {db: build_dace_dataset(plans[db], fz, db_id=i,
                                     cache_path=os.path.join(CACHE_DIR, bench["cache_id"], f"dace_{db}.pt") if a.use_cache else None)
              for i, db in enumerate(bench["train_dbs"] + bench["test_dbs"])}
    sizes = {db: len(plans[db]) for db in plans}
    pool = DACEDataset.concat([per_db[db] for db in bench["train_dbs"]])
    train_ds = pool.subset(index_positions(bench["train_index"], bench["train_dbs"], sizes))
    val_ds = pool.subset(index_positions(bench["val_index"], bench["train_dbs"], sizes))
    if a.root_only:
        train_ds.loss_masks[:, 1:] = 0.0      # keep weight 1 on the root, drop every sub-plan term
        log("ablation: root-only supervision (loss mask zeroed for non-root nodes)")
    timing["featurize_s"] = time.time() - t0

    set_seed(a.seed)
    model = DACELora(fz.node_length, output_dim=1, **DACE_DEFAULT_HPARAMS).to(device)
    params = count_parameters(model)
    log(f"params: {params['total']:,} (trainable {params['trainable']:,}; LoRA base weights are frozen upstream)")

    if a.ckpt:
        load_checkpoint(model, a.ckpt, device)
        train_info = {"ckpt": a.ckpt, "trained_here": False}
        log(f"loaded {a.ckpt}")
    else:
        t0 = time.time()
        info = dace_trainer.train(model, train_ds, val_ds, device, epochs=a.epochs, bs=a.bs, lr=a.lr,
                                  out_dir=out_dir, seed=a.seed, log=log, select=a.select)
        timing["train_s"] = time.time() - t0
        load_checkpoint(model, info["ckpt"], device)
        train_info = {k: info[k] for k in ("best_epoch", "best_metric", "select", "epochs", "ckpt")}
        train_info["trained_here"] = True
        log(f"best {info['select']}={info['best_metric']:.4f} at epoch {info['best_epoch']}")

    results = {}
    t0 = time.time()
    for db in bench["test_dbs"]:
        ds = per_db[db]
        preds, targets = predict(model, ds, device, a.bs), ds.root_ms
        results[db] = db_metrics(preds, targets)
        write_predictions(out_dir, db, preds, targets, ds.n_nodes.numpy())
        log(f"{db:16s} n={results[db]['n']:5d} {format_levels(results[db])}")
    timing["infer_s"] = time.time() - t0
    s = write_summary(out_dir, MODEL, params, bench, results, train_info, timing, extra={"hparams": dict(DACE_DEFAULT_HPARAMS)})
    log(f"average over test DBs: {format_levels(s['avg'])}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
