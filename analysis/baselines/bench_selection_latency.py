"""CPU time of one forward pass over the distinct candidate plans of a JOB query, plan-selection configurations.

GNTO (PG-stat inputs, GATv2, width 64), the BiGG encoder in the same harness and the released Reqo network with its
full inputs are built with the input sizes of the JOB benchmark (untrained weights: only the time matters) and timed
on the real candidates of the 112 queries, one query per batch, one thread.

    python analysis/baselines/bench_selection_latency.py      # -> results/analysis/dependency_20261003/tables/selection_latency.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0]]
import run_kfold_reqo as rq                                    # noqa: E402  (also puts exp3_e2e on the path)
import reqo_db                                                 # noqa: E402

rk = rq.rk
_reqo_fz, _reqo_graph = rk.make_featurizer, rk.to_graph        # the Reqo versions installed by run_kfold_reqo
import importlib                                               # noqa: E402
rk = importlib.reload(rk)                                      # the unpatched harness for GNTO
import run_kfold_bigg as rb                                    # noqa: E402  (patches rk.GNTO / rk.to_graph for BiGG)
from protocol import plan_fingerprint                          # noqa: E402
from adapters.plan_representation import represent_plan       # noqa: E402
from models.gnto import GNTO as RealGNTO, get_preset          # noqa: E402

D = os.path.join(ROOT, "data", "e2e", "hint_plans_v2_job112")
OUT = os.path.join(ROOT, "results", "analysis", "dependency_20261003", "tables", "selection_latency.csv")


class A:
    features, catalog, drop_features = "pg", os.path.join(D, "pg_catalog_snapshot.json"), ""


def main() -> None:
    torch.set_num_threads(1)
    packs = rk.load_packs(D)
    rows = []
    for i, pack in enumerate(packs):
        uniq = {}
        for r in sorted(pack["results"], key=lambda r: int(r["hint_id"])):
            if "plan" in r:
                uniq.setdefault(plan_fingerprint(r["plan"]), r)
        pack["results"] = list(uniq.values())
        for h, plan, ms in rk.candidates(pack):
            rows.append((i, h, represent_plan(plan, "ordered-binary-v1"), ms))
    plans = [r[2] for r in rows]
    stats = rk.build_stats(plans)

    import run_kfold as plain                                  # same module object as rk
    from adapters.pg_features import CatalogStats, PGFeaturizer
    pg_fz = PGFeaturizer(CatalogStats(A.catalog)).fit(plans)
    pre = rb._to_graph                                         # pre-order, both directions (GNTO)
    gnto_items = [pre(pl, pg_fz, 0.0, q, h, ms) for q, h, pl, ms in rows]
    bigg_items = [rb.to_graph(pl, pg_fz, 0.0, q, h, ms) for q, h, pl, ms in rows]
    rq.STATE["stats"] = reqo_db.load_stats(A.catalog)
    reqo_fz = _reqo_fz(stats, A, plans)
    reqo_items = [_reqo_graph(pl, reqo_fz, 0.0, q, h, ms) for q, h, pl, ms in rows]

    cfg = get_preset("dace_zero_shot", f"node.in_dim={pg_fz.node_length}", "graph.conv=gatv2", "graph.heads=1")
    models = {"GNTO (PG-stat, GATv2)": (RealGNTO.from_config(cfg), gnto_items),
              "BiGG encoder in GNTO (PG-stat)": (rb.BiGGInGNTO(cfg), bigg_items),
              "Reqo (released network, Reqo inputs)": (rq.ReqoNet(reqo_fz), reqo_items)}
    by_q = {}
    for k, r in enumerate(rows):
        by_q.setdefault(r[0], []).append(k)
    out = []
    for name, (model, items) in models.items():
        model.eval()
        batches = [Batch.from_data_list([items[k] for k in idx]) for idx in by_q.values()]
        with torch.no_grad():
            for b in batches[:10]:
                model(b)
            times = []
            for _ in range(5):
                for b in batches:
                    t0 = time.perf_counter()
                    model(b)
                    times.append((time.perf_counter() - t0) * 1e3)
        params = sum(p.numel() for p in model.parameters())
        out.append(dict(model=name, params=params, plans_per_query=float(np.median([len(v) for v in by_q.values()])),
                        forward_ms_median=float(np.median(times)), forward_ms_p95=float(np.percentile(times, 95))))
        print(out[-1])
    pd.DataFrame(out).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
