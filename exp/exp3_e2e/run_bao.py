"""Exp3 steerer: Bao's tree-convolution cost model (SIGMOD'21, ``BaoForPostgreSQL/bao_server``)
trained on the same QueryFormer corpus split as Exp1 (parts 0-17), then picking the hint set
with the smallest predicted runtime for every query.

Bao's featurizer only knows joins and Seq / Index / Index-Only / Bitmap-Index scans, so
``Bitmap Heap Scan`` subtrees are collapsed into a single ``Bitmap Index Scan`` leaf (the
same rewrite at training and steering time).  ``BaoRegression`` hard-codes 100 epochs.

Usage::

    python exp/exp3_e2e/run_bao.py                        # train (~4 min) -> results/exp3_e2e/bao/
    python exp/exp3_e2e/run_bao.py --ckpt results/exp3_e2e/bao/bao_model   # reuse
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GNTO_PATH, PG_DEFAULT_HINT, Logger, add_common_args, candidates, load_packs, pack_info,  # noqa: E402
                    steerer_dir, write_config, write_steerer)
from adapters.bao_adapter import BaoCostPredictor, _collapse_bitmap_subtree, calc_q_error, plans_from_qf_df  # noqa: E402
from models.qf import plan_csv_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="reuse a saved Bao model directory instead of training")
    ap.add_argument("--train-parts", default="0-17")
    ap.add_argument("--val-parts", default="18-19")
    ap.add_argument("--seed", type=int, default=42)
    add_common_args(ap)
    a = ap.parse_args()
    out = steerer_dir(a, "bao")
    log = Logger(os.path.join(out, "log.txt"))
    write_config(out, a, steerer="bao")

    def parts(spec):
        lo, hi = spec.split("-"); return range(int(lo), int(hi) + 1)

    pred = BaoCostPredictor(verbose=False)
    timing, train_info = {}, {}
    if a.ckpt:
        pred.load(a.ckpt)
        train_info = {"ckpt": a.ckpt, "trained_here": False}
        log(f"loaded Bao model {a.ckpt}")
    else:
        import torch, random
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        train_df = pd.concat([pd.read_csv(plan_csv_path(i)) for i in parts(a.train_parts)], ignore_index=True)
        val_df = pd.concat([pd.read_csv(plan_csv_path(i)) for i in parts(a.val_parts)], ignore_index=True)
        tp, ty = plans_from_qf_df(train_df, target="execution_time")
        vp, vy = plans_from_qf_df(val_df, target="execution_time")
        log(f"training Bao on {len(tp):,} plans (parts {a.train_parts}); val {len(vp):,} (parts {a.val_parts})")
        t0 = time.time()
        pred.fit(tp, ty)
        timing["train_s"] = time.time() - t0
        vq = calc_q_error(pred.predict(vp), vy)
        train_info = {"trained_here": True, "epochs": 100, "n_train": len(tp), "val_qerror": dict(zip(("q50", "q75", "q90", "q95", "q99"), map(float, vq)))}
        log(f"trained in {timing['train_s']:.0f}s | val Q50 {vq[0]:.3f} Q90 {vq[2]:.3f} Q99 {vq[4]:.2f}")
        pred.save(os.path.join(out, "bao_model"))
        train_info["ckpt"] = os.path.join(out, "bao_model")

    packs = load_packs(a.hint_plans)
    t0 = time.time()
    picks_rows, pred_rows = [], []
    for pack in packs:
        info = pack_info(pack); truths = info.pop("truths")
        cands = candidates(pack)
        plans = []
        for _, plan, _ in cands:
            p = copy.deepcopy(plan); _collapse_bitmap_subtree(p["Plan"]); plans.append(p)
        scores = dict(zip([h for h, _, _ in cands], map(float, pred.predict(plans)))) if plans else {}
        for h, s in scores.items():
            pred_rows.append({"query_id": pack["query_id"], "hint_id": h, "pred": s, "actual_ms": truths[h]})
        pick = min(scores, key=scores.get) if scores else PG_DEFAULT_HINT
        info.update(pick_hint=pick, pick_ms=truths[pick], rel_regret=truths[pick] / info["oracle_ms"])
        picks_rows.append(info)
    timing["score_s"] = time.time() - t0
    log(f"scored {len(pred_rows):,} candidate plans in {timing['score_s']:.1f}s")
    n_params = int(sum(p.numel() for p in pred._reg._BaoRegression__net.parameters())) if hasattr(pred._reg, "_BaoRegression__net") else None
    write_steerer(out, "bao", pd.DataFrame(picks_rows), pd.DataFrame(pred_rows), log,
                  params={"total": n_params} if n_params else {}, extra={"train": train_info, "timing_s": timing})


if __name__ == "__main__":
    main()
