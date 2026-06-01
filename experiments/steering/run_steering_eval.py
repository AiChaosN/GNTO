"""
Offline steering evaluation: given pre-collected hint plans (one JSON per query)
and a trained cost model, compute Top-K regret and end-to-end runtime metrics.

This is the **scoring** half of the Bao-style steering experiment (R1-D5).
Pair with ``collect_hint_plans.py`` (which produces the JSON inputs).

Two ground-truth modes:
  - ``--ground-truth actual`` : uses the ``actual_exec_time_ms`` field
    populated by EXPLAIN ANALYZE. Required to make claims about end-to-end
    latency improvements vs PG default.
  - ``--ground-truth cost``   : uses PG's ``Total Cost`` from EXPLAIN (no
    execution). Useful for sanity-checking the pipeline before running long
    analyze sweeps.

Three steerers are compared:
  1. PG default           : always picks hint_id == 0 (all toggles on).
  2. Oracle               : picks the truly best hint per query.
  3. <model>              : picks the hint with the smallest model prediction.

Output: ``<out>/steering_summary.json`` with per-steerer aggregates plus a
per-query breakdown for debugging.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import numpy as np

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from utils.metrics import topk_regret


def load_query_pack(path):
    with open(path) as f:
        return json.load(f)


def per_query_truth(qp, mode: str):
    """Extract (hint_id, ground_truth) for each variant of a query.

    Returns dict hint_id -> truth (float), skipping erroring variants.
    """
    out = {}
    for r in qp["results"]:
        if "error" in r and r.get("total_cost") is None:
            continue
        if mode == "actual":
            v = r.get("actual_exec_time_ms")
            if v is None:
                continue
        else:
            v = r.get("total_cost")
            if v is None:
                continue
        out[r["hint_id"]] = float(v)
    return out


def per_query_predictions(qp, predict_fn):
    """Use the user-supplied predictor to score each variant.

    ``predict_fn`` takes a single plan dict (PG EXPLAIN JSON) and returns
    a float (lower = better). We feed it ``r["plan"]`` for each variant.
    Variants that errored upstream are skipped.
    """
    out = {}
    for r in qp["results"]:
        if "plan" not in r or "error" in r:
            continue
        try:
            score = float(predict_fn(r["plan"]))
        except Exception:
            continue
        out[r["hint_id"]] = score
    return out


def steerer_pick(predictions: dict, fallback: int = 0) -> int:
    """Return hint_id with the smallest predicted score."""
    if not predictions:
        return fallback
    return min(predictions, key=predictions.get)


def summarize_steerer(name: str, picks_per_query, truths_per_query):
    """Aggregate per-query (pick_truth, oracle_truth) into headline metrics."""
    pick_truths = []
    oracle_truths = []
    rel_regrets = []
    abs_regrets = []
    timeouts_avoided = 0  # picks that beat PG default

    for qid, pick in picks_per_query.items():
        truths = truths_per_query.get(qid, {})
        if not truths or pick not in truths:
            continue
        oracle = min(truths.values())
        pick_t = truths[pick]
        pick_truths.append(pick_t)
        oracle_truths.append(oracle)
        abs_regrets.append(pick_t - oracle)
        if oracle > 0:
            rel_regrets.append(pick_t / oracle)
        pg_default = truths.get(0)
        if pg_default is not None and pick_t < pg_default:
            timeouts_avoided += 1

    if not pick_truths:
        return {"name": name, "n_queries": 0}

    return {
        "name": name,
        "n_queries": len(pick_truths),
        "total_runtime_ms": float(np.sum(pick_truths)),
        "mean_runtime_ms": float(np.mean(pick_truths)),
        "median_runtime_ms": float(np.median(pick_truths)),
        "mean_abs_regret_ms": float(np.mean(abs_regrets)),
        "median_abs_regret_ms": float(np.median(abs_regrets)),
        "mean_relative_regret": float(np.mean(rel_regrets)) if rel_regrets else float("nan"),
        "wins_over_pg_default": int(timeouts_avoided),
    }


def build_predict_fn(model_name: str, checkpoint_path: str = None):
    """Return a ``predict(plan_dict) -> float`` callable for the named model.

    Lower returned values must mean "better" (faster) for steerer_pick to work.
    """
    if model_name == "pg_cost":
        return lambda plan: float(plan["Plan"]["Total Cost"])

    if model_name == "bao":
        # Bao predictor consumes raw PG EXPLAIN JSON directly via its
        # TreeFeaturizer — no preprocessing layer needed.
        from adapters.bao_adapter import BaoCostPredictor, _collapse_bitmap_subtree
        pred = BaoCostPredictor(verbose=False)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                GNTO_PATH, "results", "Bao_0506_1435", "bao_model")
        pred.load(checkpoint_path)

        def _bao_predict(plan):
            # Bao's featurizer chokes on BitmapAnd/Or — collapse first (same
            # rewrite as the training pipeline).
            if "Plan" in plan:
                _collapse_bitmap_subtree(plan["Plan"])
            return float(pred.predict([plan])[0])
        return _bao_predict

    if model_name == "gnto":
        # Wiring is in place but requires a fresh GNTO_QF checkpoint matching
        # current code (see results/Recompute_0519_*/summary.json:note).
        import torch
        import json as _json
        from adapters.qf_adapter import (
            load_qf_resources, GNTO_QF_Model, QueryFormerToPyGDataset, unnormalize,
        )
        res = load_qf_resources()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GNTO_QF_Model(res["encoding"], add_plan_rows=True).to(device)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                GNTO_PATH, "results", "GNTO_QF_1216_1737", "best_model.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        model.eval()

        # Re-run plan->PyG conversion for a single plan dict by funnelling it
        # through QueryFormerToPyGDataset via a 1-row DataFrame.
        import pandas as pd

        def _gnto_predict(plan):
            # The PG EXPLAIN JSON we got from collect_hint_plans is already the
            # top-level dict ({"Plan": ..., "Execution Time": ...}); QF's dataset
            # expects a "json" column with the same shape. Synthesize a 1-row DF.
            row = {"id": 0, "json": _json.dumps(plan)}
            df1 = pd.DataFrame([row])
            ds = QueryFormerToPyGDataset(
                df1, res["encoding"], res["hist_file"], res["table_sample"],
                res["cost_norm"], cache_name=None, add_plan_rows=True,
            )
            from torch_geometric.loader import DataLoader as _DL
            batch = next(iter(_DL(ds, batch_size=1)))
            batch = batch.to(device)
            with torch.no_grad():
                y_norm = model(batch).view(-1).cpu().numpy()
            return float(unnormalize(y_norm, res["cost_norm"])[0])
        return _gnto_predict

    raise ValueError(f"Unknown model_name: {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint-plans-dir", required=True,
                    help="dir containing q_<id>.json files from collect_hint_plans.py")
    ap.add_argument("--ground-truth", choices=("actual", "cost"), default="cost",
                    help="actual = use EXPLAIN ANALYZE timings; cost = use PG cost estimates")
    ap.add_argument("--model", default="pg_cost",
                    help="predictor identifier: pg_cost | gnto | bao")
    ap.add_argument("--checkpoint", default=None,
                    help="override default checkpoint path for the chosen model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qpacks = []
    for p in sorted(glob.glob(os.path.join(args.hint_plans_dir, "q_*.json"))):
        qpacks.append(load_query_pack(p))
    print(f"Loaded {len(qpacks)} query packs from {args.hint_plans_dir}")

    truths = {qp["query_id"]: per_query_truth(qp, args.ground_truth) for qp in qpacks}

    # PG default: always hint_id 0
    pg_picks = {qid: 0 for qid in truths}

    # Oracle: per-query best
    oracle_picks = {qid: min(t, key=t.get) for qid, t in truths.items() if t}

    # Model steerer
    predict_fn = build_predict_fn(args.model, checkpoint_path=args.checkpoint)
    model_picks = {}
    for qp in qpacks:
        preds = per_query_predictions(qp, predict_fn)
        model_picks[qp["query_id"]] = steerer_pick(preds, fallback=0)

    out = {
        "ground_truth": args.ground_truth,
        "model": args.model,
        "n_query_packs": len(qpacks),
        "steerers": [
            summarize_steerer("pg_default", pg_picks, truths),
            summarize_steerer("oracle", oracle_picks, truths),
            summarize_steerer(args.model, model_picks, truths),
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    for s in out["steerers"]:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
