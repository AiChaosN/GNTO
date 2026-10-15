"""Exp1 runner: QueryFormer (VLDB'22) Q-error on the shared split and workloads.

Protocol: published IMDB configuration (emb 64, ffn 128, 12 heads, 8 layers,
dropout 0.1, histograms + sample bitmaps), bs 128, Adam 1e-3, StepLR(20, 0.7),
100 epochs, MSE on min-max normalised log latency, grad clip 50, checkpoint
selected on validation ``--select`` (Q90).  Train = parts 0-17, val = 18-19,
workloads = val / JOB-light / synthetic (their own sample bitmaps).

Writes ``results/exp1_qerror/qf/`` in the contract described in ``common.py``.

Usage::

    python exp/exp1_qerror/run_qf.py                 # ~1 h on one GPU
    python exp/exp1_qerror/run_qf.py --smoke         # 1 min plumbing test -> results/exp1_qerror/qf_smoke
    python exp/exp1_qerror/run_qf.py --ckpt results/exp1_qerror/qf/best_model.pt --out /tmp/qf_eval
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE_DIR, Logger, add_common_args, finalize_args, format_levels, prepare_data,  # noqa: E402
                    set_seed, workload_metrics, write_config, write_predictions, write_summary)
from models.qf import (QF_DEFAULT_HPARAMS, QueryFormer, build_plan_dataset, count_parameters,  # noqa: E402
                       load_qf_resources, seed_everything)
from models.qf import trainer as qf_trainer  # noqa: E402

MODEL = "QueryFormer"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-size", type=int, default=QF_DEFAULT_HPARAMS["emb_size"])
    ap.add_argument("--layers", type=int, default=QF_DEFAULT_HPARAMS["n_layers"])
    ap.add_argument("--heads", type=int, default=QF_DEFAULT_HPARAMS["head_size"])
    ap.add_argument("--ffn-dim", type=int, default=QF_DEFAULT_HPARAMS["ffn_dim"])
    ap.add_argument("--pred-hid", type=int, default=QF_DEFAULT_HPARAMS["pred_hid"])
    ap.add_argument("--threads", type=int, default=4)
    add_common_args(ap, "results/exp1_qerror/qf")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    hparams = dict(QF_DEFAULT_HPARAMS, emb_size=a.emb_size, n_layers=a.layers,
                   head_size=a.heads, ffn_dim=a.ffn_dim, pred_hid=a.pred_hid)
    out_dir = finalize_args(a, "qf")
    log = Logger(os.path.join(out_dir, "log.txt"))
    device = torch.device(a.device)
    write_config(out_dir, a, model=MODEL, hparams=hparams)
    log(f"{MODEL} -> {out_dir}  device {device}")

    res = load_qf_resources()
    enc, hist, cost_norm = res["encoding"], res["hist_file"], res["cost_norm"]
    data = prepare_data(a, enc, log)
    timing = {}

    # featurise before the model touches CUDA (fork-safe)
    pre = "qf_tmpl" if a.split == "template" else "qf"
    cache = (lambda name, df: os.path.join(CACHE_DIR, f"{pre}_{name}_{len(df)}.pt")) if (a.use_cache and not a.exclude_joins) else (lambda name, df: None)
    t0 = time.time()
    if not a.ckpt:
        train_ds = build_plan_dataset(data["train_df"], enc, hist, res["table_sample"], cost_norm,
                                      cache_path=cache("train", data["train_df"]), num_workers=a.num_workers, desc="qf-train")
        val_ds = build_plan_dataset(data["val_df"], enc, hist, res["table_sample"], cost_norm,
                                    cache_path=cache("val", data["val_df"]), num_workers=a.num_workers, desc="qf-val")
    eval_ds = {w: build_plan_dataset(df, enc, hist, ts if ts is not None else res["table_sample"], cost_norm,
                                     num_workers=a.num_workers, desc=f"qf-{w}")
               for w, (df, ts) in data["eval"].items()}
    timing["featurize_s"] = time.time() - t0

    set_seed(a.seed)
    seed_everything()          # upstream seeds the model init with 0
    torch.manual_seed(a.seed)
    model = QueryFormer(**hparams).to(device)
    params = count_parameters(model)
    log(f"params: {params['total']:,}")

    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=False))
        train_info = {"ckpt": a.ckpt, "trained_here": False}
        log(f"loaded {a.ckpt}")
    else:
        t0 = time.time()
        info = qf_trainer.train(model, train_ds, val_ds, cost_norm, device, epochs=a.epochs, bs=a.bs, lr=a.lr,
                                clip=a.clip, step_size=a.step_size, gamma=a.gamma, eval_every=a.eval_every,
                                select=a.select, out_dir=out_dir, seed=a.seed, log=log)
        timing["train_s"] = time.time() - t0
        model.load_state_dict(torch.load(info["ckpt"], map_location=device, weights_only=True))
        train_info = {"best_epoch": info["best_epoch"], "best_metric": info["best_metric"], "select": a.select,
                      "epochs": a.epochs, "ckpt": info["ckpt"], "trained_here": True}
        log(f"best {a.select}={info['best_metric']:.4f} at epoch {info['best_epoch']}")

    per_workload = {}
    for w, ds in eval_ds.items():
        t0 = time.time()
        _, preds = qf_trainer.evaluate(model, ds, cost_norm, device)
        timing[f"infer_{w}_s"] = time.time() - t0
        targets = np.asarray(ds.costs, dtype=np.float64)
        per_workload[w] = workload_metrics(preds, targets)
        write_predictions(out_dir, w, preds, targets, data["stats"][w])
        log(f"{w:10s} n={per_workload[w]['n']:5d} {format_levels(per_workload[w])}")

    write_summary(out_dir, MODEL, params, per_workload, train_info, timing,
                  extra={"holdout_joins": data["holdout"], "hparams": hparams})
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
