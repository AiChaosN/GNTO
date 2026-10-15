"""Evaluate a GNTO checkpoint on a cached plan split.

Works for both checkpoint formats:

* ``GNTO.save`` files (``best_model.pt`` from ``scripts/train_gnto.py``): the
  config is inside the file, nothing else is needed.
* bare ``state_dict`` files written by the pre-rewrite classes: pass
  ``--preset`` / ``--config`` describing the architecture (``gnto_joinid`` for the
  string-id join model); keys are converted automatically.

Examples::

    python scripts/eval_gnto.py --ckpt results/gnto_run/best_model.pt \
        --cache data/process/cache/tmpl_test_planrows_9999.pt

    python scripts/eval_gnto.py --preset gnto_joinid \
        --ckpt path/to/legacy_state_dict.pth \
        --cache data/process/cache/tmpl_test_planrows_9999.pt --save-preds preds.csv

Reports Q-error percentiles, Spearman / Kendall and pairwise accuracy on the raw
(execution-time) scale.  Targets are assumed to be QueryFormer-normalised
(sigmoid head + ``raw_y`` attribute on the cached Data objects).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from models.gnto import GNTO, GNTOConfig, get_preset, list_presets, evaluation_summary  # noqa: E402

COST_MIN, COST_MAX = -3.61192, 12.290855


def unnormalize(y_norm: np.ndarray) -> np.ndarray:
    return np.exp(y_norm * (COST_MAX - COST_MIN) + COST_MIN) - 0.001


def load_model(args) -> GNTO:
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format"):
        model = GNTO.load(args.ckpt)
        if args.preset or args.config:
            print("note: checkpoint carries its own config; --preset/--config ignored")
        return model
    if not (args.preset or args.config):
        raise SystemExit("bare state_dict checkpoint: pass --preset or --config for the architecture")
    cfg = GNTOConfig.load(args.config) if args.config else get_preset(args.preset)
    if args.set:
        cfg = cfg.override(*args.set)
    model = GNTO.from_config(cfg)
    report = model.load_legacy_state_dict(payload, strict=True)
    print(f"loaded pre-rewrite state_dict: {report['renamed']} keys renamed, "
          f"{len(report['dropped'])} dead keys dropped")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="cached list of PyG Data (data/process/cache/*.pt)")
    ap.add_argument("--preset", choices=list_presets())
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, help="evaluate only the first N graphs")
    ap.add_argument("--save-preds", help="write per-plan predictions / targets to this CSV")
    ap.add_argument("--json", help="write the metric summary to this JSON file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args).to(device)
    print(model.summary())

    data = torch.load(args.cache, weights_only=False)
    if args.limit:
        data = data[:args.limit]
    preds, targets = [], []
    for batch in DataLoader(data, batch_size=args.batch_size):
        batch = batch.to(device)
        preds.append(unnormalize(model.predict(batch).cpu().numpy()))
        targets.append(batch.raw_y.view(-1).cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)

    summary = evaluation_summary(preds, targets)
    summary["n"] = int(len(preds))
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
    if args.save_preds:
        np.savetxt(args.save_preds, np.stack([preds, targets], 1), delimiter=",", header="pred_ms,actual_ms", comments="")


if __name__ == "__main__":
    main()
