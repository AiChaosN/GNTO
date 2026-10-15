"""Shared pieces of Exp3 (end-to-end plan steering): the pre-collected hint-plan packs,
the steerer metrics and the per-steerer output contract.

Benchmark (``data/e2e/hint_plans_job112``): 112 IMDB queries (110 JOB + 2 CEB), each run
under all 49 Bao hint sets (``hint_sets.py``) with ``EXPLAIN ANALYZE`` on PostgreSQL 16 +
IMDB.  Variants that hit the statement timeout carry no plan and cannot be candidates.
A *steerer* picks one hint set per query; its cost is the measured runtime of that plan.

Output contract - every runner writes ``results/exp3_e2e/<steerer>/``::

    picks.csv        query_id, n_candidates, n_joins, join_bucket, pick_hint, pick_ms,
                     pg_default_ms, oracle_hint, oracle_ms, rel_regret
    predictions.csv  (model steerers) query_id, hint_id, pred, actual_ms
    summary.json     totals, speed-up vs PG default, regret, wins, per join bucket,
                     cost accuracy on the hint plans (model steerers), params
    config.json, log.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

_EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_EXP_ROOT, os.path.dirname(_EXP_ROOT), os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _shared import GNTO_PATH, Logger, git_rev, md, set_seed, provenance  # noqa: E402,F401
from hint_sets import HINT_SETS  # noqa: E402
from protocol import cluster_interval, template_key

EXP = "exp3_e2e"
RESULTS = os.path.join(GNTO_PATH, "results", EXP)
HINT_PLANS = os.path.join(GNTO_PATH, "data", "e2e", "hint_plans_job112")
JOIN_TYPES = ("Nested Loop", "Hash Join", "Merge Join")
JOIN_BUCKETS = ("1-2", "3-4", "5+")
PG_DEFAULT_HINT = 0


# --------------------------------------------------------------------------- #
# hint-plan packs
# --------------------------------------------------------------------------- #
def load_packs(hint_dir: str = HINT_PLANS) -> List[dict]:
    packs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(hint_dir, "q_*.json")))]
    if not packs:
        raise FileNotFoundError(f"no q_*.json under {hint_dir}")
    return packs


def candidates(pack: dict) -> List[Tuple[int, dict, float]]:
    """Plans with observed cost, including budget-capped timeout candidates in v2.

    v1 packs carry no plan for hint sets that timed out, so an evaluation on them is
    conditional on successful execution; ``protocol.validate_complete_pack`` rejects
    incomplete packs.
    """
    out = []
    for r in pack["results"]:
        cost = r.get("evaluation_ms", r.get("actual_exec_time_ms"))
        if "plan" in r and "error" not in r and cost is not None:
            out.append((int(r["hint_id"]), r["plan"], float(cost)))
    return out


def count_joins(plan: dict) -> int:
    n, stack = 0, [plan["Plan"] if "Plan" in plan else plan]
    while stack:
        node = stack.pop()
        n += node.get("Node Type") in JOIN_TYPES
        stack.extend(node.get("Plans", []))
    return n


def join_bucket(n_joins: int) -> str:
    return "1-2" if n_joins <= 2 else "3-4" if n_joins <= 4 else "5+"


def pack_info(pack: dict) -> Dict[str, Any]:
    """Per-query facts that do not depend on the steerer."""
    cands = candidates(pack)
    truths = {h: t for h, _, t in cands}
    default_plan = next((p for h, p, _ in cands if h == PG_DEFAULT_HINT), cands[0][1])
    nj = count_joins(default_plan)
    oracle_hint = min(truths, key=truths.get)
    return {"query_id": pack["query_id"], "template": pack.get("template", template_key(pack["query_id"])),
            "n_candidates": len(cands), "n_joins": nj, "join_bucket": join_bucket(nj),
            "pg_default_ms": truths.get(PG_DEFAULT_HINT, float("nan")), "oracle_hint": oracle_hint,
            "oracle_ms": truths[oracle_hint], "truths": truths}


# --------------------------------------------------------------------------- #
# steering
# --------------------------------------------------------------------------- #
def steer(packs: List[dict], score_fn: Optional[Callable[[dict], float]], name: str, log=print
          ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Apply ``score_fn(plan) -> score`` (lower = better) to every candidate of every query
    and pick the argmin.  ``score_fn=None`` means "always PG default"; ``score_fn="oracle"``
    picks the truly fastest plan.  Returns ``(picks, predictions)``."""
    picks, preds = [], []
    for i, pack in enumerate(packs):
        info = pack_info(pack)
        truths = info.pop("truths")
        if score_fn is None:
            pick = PG_DEFAULT_HINT if PG_DEFAULT_HINT in truths else info["oracle_hint"]
        elif score_fn == "oracle":
            pick = info["oracle_hint"]
        else:
            scores = {}
            for h, plan, actual in candidates(pack):
                s = float(score_fn(plan))
                scores[h] = s
                preds.append({"query_id": info["query_id"], "hint_id": h, "pred": s, "actual_ms": actual})
            pick = min(scores, key=scores.get) if scores else PG_DEFAULT_HINT
        info.update(pick_hint=pick, pick_ms=truths[pick], rel_regret=truths[pick] / info["oracle_ms"])
        picks.append(info)
        if (i + 1) % 25 == 0:
            log(f"[{name}] {i + 1}/{len(packs)} queries")
    return pd.DataFrame(picks), (pd.DataFrame(preds) if preds else None)


SPEED_TOL = 0.05        # a pick within +-5% of PG default counts as a tie (measurement noise)


def relative_metrics(picks: pd.DataFrame, tol: float = SPEED_TOL) -> Dict[str, Any]:
    """Per-query speed-up ``PG default / pick`` - the primary Exp3 metric.

    Totals weight every query by its runtime; ratios are per query and scale-free, and their
    average is the *geometric* mean (the arithmetic mean of ratios is biased upward).
    ``better`` / ``worse`` / ``tie`` are counts over queries with a +-``tol`` dead band.
    """
    ratio = (picks.pg_default_ms / picks.pick_ms.clip(lower=1e-9)).to_numpy()
    lr = np.log(ratio)
    better, worse = ratio > 1 + tol, ratio < 1 - tol
    tie = ~better & ~worse
    moved = (picks.pick_hint != PG_DEFAULT_HINT).to_numpy()
    n_bw = int(better.sum() + worse.sum())
    trimmed = np.sort(lr)[3:-3] if len(lr) > 6 else lr
    out = {"rel_speedup_geomean": float(np.exp(lr.mean())),
           "rel_speedup_geomean_trimmed": float(np.exp(trimmed.mean())),
           "rel_speedup_mean": float(ratio.mean()),
           **{f"rel_speedup_p{q}": float(np.quantile(ratio, q / 100)) for q in (10, 25, 50, 75, 90)},
           "rel_speedup_min": float(ratio.min()),
           "better_n": int(better.sum()), "worse_n": int(worse.sum()), "tie_n": int(tie.sum()),
           "better_pct": float(100 * better.mean()), "worse_pct": float(100 * worse.mean()),
           "tie_pct": float(100 * tie.mean()), "tol": tol,
           "moved_n": int(moved.sum()),
           "moved_better_pct": float(100 * (better & moved).sum() / max(moved.sum(), 1)),
           "moved_worse_pct": float(100 * (worse & moved).sum() / max(moved.sum(), 1))}
    out["wilcoxon_p_vs_pg"] = float(wilcoxon(lr).pvalue) if np.any(lr != 0) else float("nan")
    out["sign_test_p"] = float(binomtest(int(better.sum()), n_bw, 0.5).pvalue) if n_bw else float("nan")
    groups = picks["template"] if "template" in picks else picks.query_id.map(template_key)
    out["template_cluster_interval"] = cluster_interval(lr, groups)
    return out


def summarize(picks: pd.DataFrame) -> Dict[str, Any]:
    total, pg_total, oracle_total = picks.pick_ms.sum(), picks.pg_default_ms.sum(), picks.oracle_ms.sum()
    out = {"n_queries": int(len(picks)), "total_runtime_ms": float(total), "pg_default_total_ms": float(pg_total),
           "oracle_total_ms": float(oracle_total), "speedup_vs_pg_default": float(pg_total / total),
           "runtime_vs_oracle": float(total / oracle_total),
           "mean_rel_regret": float(picks.rel_regret.mean()), "median_rel_regret": float(picks.rel_regret.median()),
           "p90_rel_regret": float(picks.rel_regret.quantile(0.9)), "max_rel_regret": float(picks.rel_regret.max()),
           "wins_vs_pg_default": int((picks.pick_ms < picks.pg_default_ms).sum()),
           "losses_vs_pg_default": int((picks.pick_ms > picks.pg_default_ms).sum()),
           "regressions_2x": int((picks.pick_ms > 2 * picks.pg_default_ms).sum()),
           "oracle_hits": int((picks.pick_hint == picks.oracle_hint).sum()),
           **relative_metrics(picks)}
    by = {}
    for b in JOIN_BUCKETS:
        g = picks[picks.join_bucket == b]
        if len(g):
            rel = relative_metrics(g)
            by[b] = {"n": int(len(g)), "total_runtime_ms": float(g.pick_ms.sum()), "pg_default_total_ms": float(g.pg_default_ms.sum()),
                     "oracle_total_ms": float(g.oracle_ms.sum()), "speedup_vs_pg_default": float(g.pg_default_ms.sum() / g.pick_ms.sum()),
                     "mean_rel_regret": float(g.rel_regret.mean()),
                     **{k: rel[k] for k in ("rel_speedup_geomean", "rel_speedup_p50", "better_n", "worse_n", "tie_n")}}
    out["by_join_bucket"] = by
    if "pick_censored" in picks:
        out["timeout_picks"] = int(picks.pick_censored.sum())
        out["timeout_pick_pct"] = float(100 * picks.pick_censored.mean())
    return out


def prediction_quality(preds: pd.DataFrame) -> Dict[str, Any]:
    """Cost accuracy of a model on the hint plans (Q-error, needs predictions in ms) and
    per-query rank correlation between predicted and measured runtime."""
    from scipy.stats import spearmanr, kendalltau
    censored_n = int(preds.censored.sum()) if "censored" in preds else 0
    preds = preds[~preds.censored.astype(bool)] if "censored" in preds else preds
    if preds.empty:
        return {"n_plans": 0, "censored_plans_excluded_from_accuracy": censored_n}
    p = np.maximum(preds.pred.to_numpy(float), 1e-7); t = np.maximum(preds.actual_ms.to_numpy(float), 1e-7)
    q = np.maximum(p / t, t / p)
    out = {"n_plans": int(len(preds)), "censored_plans_excluded_from_accuracy": censored_n,
           "q50": float(np.percentile(q, 50)), "q90": float(np.percentile(q, 90)),
           "q99": float(np.percentile(q, 99)), "q_mean": float(q.mean())}
    rhos, taus = [], []
    for _, g in preds.groupby("query_id"):
        if len(g) >= 3 and g.pred.nunique() > 1:
            rhos.append(spearmanr(g.pred, g.actual_ms).correlation)
            taus.append(kendalltau(g.pred, g.actual_ms).correlation)
    out.update({"per_query_spearman_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
                "per_query_spearman_median": float(np.nanmedian(rhos)) if rhos else float("nan"),
                "per_query_kendall_mean": float(np.nanmean(taus)) if taus else float("nan"),
                "global_spearman": float(spearmanr(preds.pred, preds.actual_ms).correlation)})
    return out


def write_steerer(out_dir: str, name: str, picks: pd.DataFrame, preds: Optional[pd.DataFrame], log,
                  params: Optional[Dict[str, int]] = None, extra: Optional[Dict[str, Any]] = None,
                  preds_are_ms: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    picks.to_csv(os.path.join(out_dir, "picks.csv"), index=False)
    summary = {"steerer": name, **summarize(picks), "params": params or {}, "finished": datetime.now().isoformat(timespec="seconds")}
    if preds is not None:
        preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        pq = prediction_quality(preds)
        if not preds_are_ms:          # e.g. PG cost units: only rank metrics are meaningful
            for k in ("q50", "q90", "q99", "q_mean"):
                pq[k] = float("nan")
        summary["prediction_quality"] = pq
    if extra:
        summary.update(extra)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    s = summary
    log(f"[{name}] total {s['total_runtime_ms'] / 1000:.1f}s | PG default {s['pg_default_total_ms'] / 1000:.1f}s | "
        f"oracle {s['oracle_total_ms'] / 1000:.1f}s | speed-up x{s['speedup_vs_pg_default']:.2f} | "
        f"regret mean {s['mean_rel_regret']:.2f} median {s['median_rel_regret']:.2f} | wins {s['wins_vs_pg_default']}/{s['n_queries']}")
    return summary


def write_config(out_dir: str, a: argparse.Namespace, **more: Any) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"experiment": EXP, "out_dir": out_dir, "args": {k: v for k, v in vars(a).items() if not k.startswith("_")},
           "git_rev": git_rev(), "provenance": provenance(), "started": datetime.now().isoformat(timespec="seconds")}
    cfg.update(more)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--hint-plans", default=HINT_PLANS, help="directory of q_<id>.json packs")
    ap.add_argument("--out", help="output dir (default results/exp3_e2e/<steerer>)")
    ap.add_argument("--tag", default="", help="suffix for the steerer dir name")


def steerer_dir(a: argparse.Namespace, name: str) -> str:
    return a.out or os.path.join(RESULTS, name + (f"_{a.tag}" if a.tag else ""))
