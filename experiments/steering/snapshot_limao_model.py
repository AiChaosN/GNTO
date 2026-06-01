"""
Watch LIMAO's bao_server model dir and snapshot it after every retrain.

LIMAO's training overwrites ``bao_default_model_gnto/`` in place via
``train_and_swap``. To keep per-iteration history (model parameters + history
db) for later analysis, this script polls the model dir's mtime and copies
the entire dir whenever it changes.

Output layout:
  results/LIMAO_GNTO_20iter_0519/model_snapshots/snap_<iter>_<HHMMSS>/
    nn_weights      (torch state_dict)
    x_transform, y_transform, n  (sklearn preprocessor state)
    history.db      (copy of bao_server/history_GNTO.db at snap time)
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path

DEFAULT_MODEL_DIR = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/bao_server/bao_default_model_gnto"
DEFAULT_HISTORY_DB = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/bao_server/history_GNTO.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--history-db", default=DEFAULT_HISTORY_DB)
    ap.add_argument("--out-root", required=True,
                    help="root directory to dump snapshots into (e.g. results/LIMAO_GNTO_20iter_0519/model_snapshots/)")
    ap.add_argument("--poll-sec", type=float, default=2.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    weight_file = os.path.join(args.model_dir, "nn_weights")
    if not os.path.exists(weight_file):
        print(f"WARN: {weight_file} does not exist yet -- waiting for first retrain", flush=True)

    last_mtime = 0.0
    snap_idx = 0
    while True:
        try:
            mtime = os.path.getmtime(weight_file) if os.path.exists(weight_file) else 0.0
        except OSError:
            mtime = 0.0
        if mtime > last_mtime and mtime > 0:
            snap_idx += 1
            ts = time.strftime("%H%M%S")
            dest = out_root / f"snap_{snap_idx:03d}_{ts}"
            try:
                shutil.copytree(args.model_dir, dest)
                if os.path.exists(args.history_db):
                    shutil.copy2(args.history_db, dest / "history.db")
                print(f"[{ts}] snapshot #{snap_idx} -> {dest}", flush=True)
            except Exception as e:
                print(f"[{ts}] snapshot #{snap_idx} FAILED: {e}", flush=True)
            last_mtime = mtime
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
