"""What do the attention weights of a trained GNTO look like on unseen databases?

For every GATv2 layer of a finished run of ``run_crossdb_dep.py`` this script scores the test plans once and
summarises the attention coefficients ``alpha_ij`` (they sum to one over the incoming edges of node ``i``:
its own self-loop, its parent and its children):

* normalised entropy  H_i / log(deg_i)          1 = uniform weights (what a graph convolution would use)
* attention mass on self / parent / children, next to the uniform share 1 / deg_i
* for operators with two children: share of the children's mass that goes to the child with the larger
  row estimate, and to the first (outer) child

Nothing is trained; the checkpoint and its fitted feature statistics are read from the run directory.

    python analysis/dependency_study/probe_attention.py --run results/analysis/dependency_20261003/crossdb/nocost_gatv2_L3_true_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp2_dace"), os.path.join(ROOT, "exp"), ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.dace import NodeFeaturizer, load_plans, plans_path   # noqa: E402
from models.gnto import GNTO                                      # noqa: E402
from run_crossdb_dep import SubsetFeaturizer, build               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=1000, help="test plans per database")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    summary = json.load(open(os.path.join(a.run, "summary.json")))
    stats = json.load(open(os.path.join(a.run, "feature_statistics.json")))
    features = cfg["study"]["features"]
    fz = SubsetFeaturizer(NodeFeaturizer(stats), features)
    rows_col = fz.keep.index(fz.fz.num_types + 1) if (fz.fz.num_types + 1) in fz.keep else None
    model = GNTO.load(os.path.join(a.run, "best_model.pt"))
    model.eval()

    acc = {}                                                    # layer -> lists

    def add(layer, key, values):
        acc.setdefault(layer, {}).setdefault(key, []).append(np.asarray(values, dtype=np.float64))

    for db_id, db in enumerate(summary["split"]["test_dbs"]):
        items = build(load_plans(plans_path(db))[:a.limit], fz, db_id, "true", 0)
        for batch in DataLoader(items, batch_size=256, shuffle=False):
            with torch.no_grad():
                _, aux = model(batch, return_aux=True)
            n = batch.x.size(0)
            raw_rows = batch.x[:, rows_col].numpy() if rows_col is not None else None
            for layer, att in enumerate(aux["attention"]):
                if att is None:
                    continue
                ei, alpha = att[0].numpy(), att[1].mean(dim=1).numpy()      # mean over heads
                src, dst = ei[0], ei[1]
                deg = np.bincount(dst, minlength=n)
                ent = np.zeros(n)
                np.add.at(ent, dst, -alpha * np.log(np.clip(alpha, 1e-12, None)))
                multi = deg >= 2
                add(layer, "entropy", ent[multi] / np.log(deg[multi]))
                # nodes are in DFS pre-order: the parent of a node has a smaller index
                self_e, parent_e, child_e = src == dst, src < dst, src > dst
                for name, mask in (("self", self_e), ("parent", parent_e), ("children", child_e)):
                    mass = np.zeros(n)
                    np.add.at(mass, dst[mask], alpha[mask])
                    cnt = np.bincount(dst[mask], minlength=n)
                    has = cnt > 0
                    add(layer, f"mass_{name}", mass[has])
                    add(layer, f"uniform_{name}", cnt[has] / deg[has])
                # two-child operators
                order = np.lexsort((src[child_e], dst[child_e]))
                cs, cd, ca = src[child_e][order], dst[child_e][order], alpha[child_e][order]
                two = np.flatnonzero((np.bincount(cd, minlength=n) == 2)[cd])
                if len(two):
                    first, second = two[0::2], two[1::2]
                    tot = ca[first] + ca[second]
                    add(layer, "outer_share", ca[first] / tot)
                    if raw_rows is not None:
                        bigger_first = raw_rows[cs[first]] >= raw_rows[cs[second]]
                        add(layer, "larger_rows_share", np.where(bigger_first, ca[first], ca[second]) / tot)
    out = []
    for layer in sorted(acc):
        row = {"run": os.path.basename(a.run), "layer": layer + 1}
        for key, chunks in acc[layer].items():
            v = np.concatenate(chunks)
            row[key] = float(v.mean())
            if key in ("entropy", "outer_share", "larger_rows_share"):
                row[key + "_p10"], row[key + "_p90"] = float(np.percentile(v, 10)), float(np.percentile(v, 90))
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(a.run, "attention_probe.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
