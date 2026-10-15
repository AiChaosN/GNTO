"""Exp3 steerer: GNTO cost model (trained in Exp1 on the QueryFormer corpus) picks, for
every query, the hint set whose plan it predicts to be fastest.

The live JOB plans are featurised exactly like the training corpus (QueryFormer node
features + Plan Rows + join columns) except that the per-query sample bitmaps do not
exist for unseen queries and are set to zero (``zero_table_sample=True``); node types /
joins outside the training vocabulary map to id 0.  Both are constant across a query's
49 variants, so the within-query ranking is unaffected.

Usage::

    python exp/exp3_e2e/run_gnto.py                                             # results/exp1_qerror/gnto/best_model.pt -> results/exp3_e2e/gnto/
    python exp/exp3_e2e/run_gnto.py --ckpt results/exp1_qerror/gnto_joinid/best_model.pt --tag joinid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GNTO_PATH, PG_DEFAULT_HINT, Logger, add_common_args, candidates, load_packs, pack_info,  # noqa: E402
                    steerer_dir, write_config, write_steerer)
from adapters.qf_adapter import QueryFormerToPyGDataset, unnormalize  # noqa: E402
from models.gnto import GNTO  # noqa: E402
from models.qf import load_qf_resources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(GNTO_PATH, "results", "exp1_qerror", "gnto", "best_model.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common_args(ap)
    a = ap.parse_args()
    out = steerer_dir(a, "gnto")
    log = Logger(os.path.join(out, "log.txt"))
    device = torch.device(a.device)
    model = GNTO.load(a.ckpt).to(device)
    cfg = model.config
    write_config(out, a, steerer="gnto", ckpt=a.ckpt, gnto_config=cfg.to_dict())
    log(f"GNTO ({cfg.name}) {a.ckpt} -> {out}")

    res = load_qf_resources(zero_table_sample=True)
    packs = load_packs(a.hint_plans)
    # score every candidate plan of every query in one pass (cache: plan id -> prediction)
    rows, keys = [], []
    for pack in packs:
        for h, plan, _ in candidates(pack):
            rows.append({"id": len(rows), "json": json.dumps(plan)})
            keys.append((pack["query_id"], h))
    t0 = time.time()
    ds = QueryFormerToPyGDataset(pd.DataFrame(rows), res["encoding"], res["hist_file"], res["table_sample"], res["cost_norm"],
                                 cache_name=None, add_plan_rows=cfg.node.plan_rows, add_join_cols=cfg.node.qf_join_cols, num_workers=8)
    preds = np.concatenate([unnormalize(model.predict(b.to(device)).cpu().numpy(), res["cost_norm"])
                            for b in DataLoader(ds.data_list, batch_size=512, shuffle=False)])
    log(f"scored {len(preds):,} candidate plans in {time.time() - t0:.1f}s")
    lookup = {k: float(p) for k, p in zip(keys, preds)}

    # steer() scores plans one by one; with a batch-scored lookup we assemble the picks directly
    picks_rows, pred_rows = [], []
    for pack in packs:
        info = pack_info(pack); truths = info.pop("truths")
        scores = {h: lookup[(pack["query_id"], h)] for h, _, _ in candidates(pack)}
        for h, s in scores.items():
            pred_rows.append({"query_id": pack["query_id"], "hint_id": h, "pred": s, "actual_ms": truths[h]})
        pick = min(scores, key=scores.get) if scores else PG_DEFAULT_HINT
        info.update(pick_hint=pick, pick_ms=truths[pick], rel_regret=truths[pick] / info["oracle_ms"])
        picks_rows.append(info)
    picks, pred_df = pd.DataFrame(picks_rows), pd.DataFrame(pred_rows)
    params = {"total": model.num_parameters(), "trainable": model.num_parameters(trainable_only=True)}
    write_steerer(out, "gnto", picks, pred_df, log, params=params, extra={"ckpt": a.ckpt, "preset": cfg.name})


if __name__ == "__main__":
    main()
