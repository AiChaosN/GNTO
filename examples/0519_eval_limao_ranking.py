"""
Re-score the LIMAO online-steering trace with ranking metrics.

Background: LIMAO logs ``query, actual_latency, predicted_cost, logged_reward``
for each plan it picked at runtime. The original LIMAO analysis only reports
end-to-end wall-clock latency vs PG default (see
``LIMAOLifeLongRLDB/gnto_ex/compare_models.py``). The reviewer (R3-D1) asks
for ranking-style metrics; Q-Error doesn't apply here because predicted_cost
(PG cost units, ~18-18500) and actual_latency (seconds, 0.15-32) live on
different scales -- so we report Spearman/Kendall/pairwise-accuracy instead.

Output: ``results/LIMAO_ranking_<timestamp>/summary.json`` plus a console
summary aligned with Tables A/B in the paper.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GNTO_PATH not in sys.path:
    sys.path.insert(0, GNTO_PATH)

from utils.metrics import ranking_metrics, pairwise_accuracy


def main():
    src = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/gnto_predictions.csv"
    df = pd.read_csv(src)
    print(f"Loaded {len(df)} predictions from {src}")
    print(f"  actual_latency: {df['actual_latency'].min():.3f}s — {df['actual_latency'].max():.3f}s")
    print(f"  predicted_cost: {df['predicted_cost'].min():.1f} — {df['predicted_cost'].max():.1f}")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(GNTO_PATH, "results", f"LIMAO_ranking_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # ------ Whole-trace ranking: does predicted_cost rank actual_latency? ------
    preds = df["predicted_cost"].to_numpy()
    truths = df["actual_latency"].to_numpy()
    rank = ranking_metrics(preds, truths)
    pair = pairwise_accuracy(preds, truths)
    print("\n=== Whole-trace ranking (n={}) ===".format(len(df)))
    print(f"  Spearman rho:  {rank['spearman']:.4f}")
    print(f"  Kendall  tau:  {rank['kendall']:.4f}")
    print(f"  Pairwise acc:  {pair:.4f}")

    # ------ Per-query ranking: collapse repeated query rows ------
    # Same query appears multiple times (LIMAO retries with different hint
    # variants while learning). For each query, compute Spearman across its
    # own runs -- this tells us whether GNTO can rank hint variants WITHIN
    # a single query, which is the steering-relevant question.
    per_query = []
    for qname, sub in df.groupby("query"):
        if len(sub) < 3:
            continue
        rm = ranking_metrics(sub["predicted_cost"].to_numpy(), sub["actual_latency"].to_numpy())
        per_query.append({
            "query": qname,
            "n_runs": int(len(sub)),
            "spearman": rm["spearman"],
            "kendall": rm["kendall"],
        })

    pq_df = pd.DataFrame(per_query)
    print(f"\n=== Per-query intra-trace ranking ({len(pq_df)} queries with >= 3 runs) ===")
    if not pq_df.empty:
        print(f"  Mean Spearman across queries:    {pq_df['spearman'].mean():.4f}")
        print(f"  Median Spearman across queries:  {pq_df['spearman'].median():.4f}")
        print(f"  Spearman > 0 (model picks correctly more often than random):"
              f"  {(pq_df['spearman'] > 0).sum()} / {len(pq_df)}")
        print(f"  Spearman == 1.0 (perfect intra-query ranking):"
              f"  {(pq_df['spearman'] >= 0.999).sum()} / {len(pq_df)}")

    # ------ Compare predicted_cost vs logged_reward as ranker ------
    # logged_reward is LIMAO's internal reward signal — sanity check that
    # GNTO's cost prediction is at least as good a ranker as LIMAO's own
    # reward function.
    rew_rank = ranking_metrics(df["logged_reward"].to_numpy(), df["actual_latency"].to_numpy())
    print("\n=== LIMAO logged_reward ranker (for comparison) ===")
    print(f"  Spearman:  {rew_rank['spearman']:.4f}  Kendall: {rew_rank['kendall']:.4f}")

    out = {
        "timestamp": timestamp,
        "source": src,
        "n_total": int(len(df)),
        "whole_trace": {
            **rank,
            "pairwise_acc": pair,
        },
        "per_query": per_query,
        "logged_reward_ranking": rew_rank,
        "note": (
            "Q-Error is intentionally NOT reported: predicted_cost (PG cost units, ~18-18500) "
            "and actual_latency (seconds, 0.15-32) live on different scales -- direct |p/t| or "
            "|t/p| is dominated by the unit gap. Ranking metrics directly answer the steering-"
            "relevant question 'does GNTO's cost prediction order plans by their actual runtime?'"
        ),
    }
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    pq_df.to_csv(os.path.join(save_dir, "per_query.csv"), index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
