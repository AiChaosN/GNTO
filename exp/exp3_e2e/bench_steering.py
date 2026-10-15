"""CPU cost of the actual saved steering models, with featurization and batching.

Enumeration times are replayed from the collection manifest and are explicitly
separate from the measured local model overhead. This is not an online system run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import candidates, load_packs
from protocol import cluster_interval, plan_fingerprint, template_key
from run_kfold import to_graph, SubsetFeaturizer
from _shared import file_sha256, provenance
from models.gnto import GNTO
from models.dace import NodeFeaturizer
from adapters.pg_features import CatalogStats, PGFeaturizer
from adapters.plan_representation import represent_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.loads((a.run / "config.json").read_text())["args"]
    membership = json.loads((a.run / "folds.json").read_text())["queries"]
    packs = {p["query_id"]: p for p in load_packs(cfg["hint_plans"])}
    models = {}
    records = []
    picks = pd.read_csv(a.run / "picks.csv").set_index("query_id")
    for qid, fold in membership.items():
        f = fold["fold"]
        if f not in models:
            path = a.run / f"fold{f}" / "model.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            extra = payload["extra"]
            model = GNTO.load(path).eval()
            if extra["features"] == "pg":
                state = extra["pg_state"]
                if file_sha256(cfg["catalog"]) != state["catalog_sha256"]:
                    raise ValueError("catalog differs from the training snapshot")
                fz = PGFeaturizer(CatalogStats(cfg["catalog"]))
                fz.type_index = state["type_index"]
                fz.center, fz.scale = np.array(state["center"]), np.array(state["scale"])
            else:
                fz = NodeFeaturizer(extra["stats"])
                if extra.get("drop_features"):
                    fz = SubsetFeaturizer(fz, extra["drop_features"].split(","))
            models[f] = (model, fz, extra)
        model, fz, extra = models[f]
        representation = extra.get("plan_representation", "native")
        if representation != cfg.get("plan_representation", "native"):
            raise ValueError("checkpoint/config plan representation mismatch")
        pack = packs[qid]
        rows = candidates(pack)
        if cfg.get("deduplicate"):
            unique = {}
            for row in rows:
                unique.setdefault(plan_fingerprint(row[1]), row)
            rows = list(unique.values())
        def featurize():
            return [to_graph(represent_plan(p, representation), fz, 0, 0, h, 0,
                             parent_to_child=extra.get("parent_to_child", False)) for h, p, _ in rows]
        graph = Batch.from_data_list(featurize())
        with torch.inference_mode():
            for _ in range(5):
                model(graph)
            forward, full = [], []
            for _ in range(a.repeats):
                start = time.perf_counter()
                model(graph)
                forward.append(1000 * (time.perf_counter() - start))
                start = time.perf_counter()
                model(Batch.from_data_list(featurize()))
                full.append(1000 * (time.perf_counter() - start))
        plan_times = [r.get("planning_wall_ms") for r in pack["results"]]
        enumeration = sum(plan_times) if all(t is not None for t in plan_times) else float("nan")
        default = next(r for r in pack["results"] if int(r["hint_id"]) == 0)
        pg_planning = default.get("planning_wall_ms", float("nan"))
        overhead = float(np.median(full))
        pick, pg = float(picks.loc[qid, "pick_ms"]), float(picks.loc[qid, "pg_default_ms"])
        records.append({"query_id": qid, "template": fold["template"], "fold": f, "n_candidates": len(rows),
                        "forward_ms": float(np.median(forward)), "full_model_ms": overhead,
                        "full_model_p95_ms": float(np.percentile(full, 95)), "enumeration_wall_ms": enumeration,
                        "pg_planning_wall_ms": pg_planning, "pick_ms": pick, "pg_ms": pg,
                        "net_speedup_replay": (pg if pack.get("latency_metric") == "capped_statement_wall_ms"
                                               else pg + pg_planning) / (pick + enumeration + overhead)})
    d = pd.DataFrame(records)
    d.to_csv(a.run / f"latency_cpu{a.threads}.csv", index=False)
    summary = {"provenance": provenance(), "threads": a.threads, "repeats": a.repeats,
               "plan_representation": cfg.get("plan_representation", "native"),
               "forward_ms_median": float(d.forward_ms.median()), "full_model_ms_median": float(d.full_model_ms.median()),
               "scope": "local CPU model timing plus replayed serial enumeration and execution; not an online system measurement"}
    if np.isfinite(d.net_speedup_replay).all():
        summary["net_speedup_replay"] = cluster_interval(np.log(d.net_speedup_replay), d.template)
    (a.run / f"latency_cpu{a.threads}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
