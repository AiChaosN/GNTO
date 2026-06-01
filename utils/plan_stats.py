"""
Plan-level statistics for stratified evaluation (review item R2-W2).

The reviewer asked for Table 3 results broken down by join count (1-2 / 3-4 / 5+),
since GNTO's claim of better performance on complex queries is more credible
when shown per-bucket. This module provides:

- ``count_joins(plan_dict)``: counts PG join operators in a raw EXPLAIN JSON tree
- ``join_counts_from_df(df)``: vectorized helper for QueryFormer CSV format
- ``stratify_by_join_count(...)``: bucketize a (preds, targets) pair and report
  Q-Error percentiles per bucket
"""

from __future__ import annotations

import json
import numpy as np

PG_JOIN_TYPES = frozenset({"Hash Join", "Merge Join", "Nested Loop"})


def count_joins(plan) -> int:
    """Count join operators in a PG EXPLAIN JSON plan tree.

    Accepts either the top-level dict (with a "Plan" key) or the inner "Plan"
    dict directly. Iterative DFS to avoid Python recursion limits on deep trees.
    """
    if isinstance(plan, str):
        plan = json.loads(plan)
    root = plan["Plan"] if isinstance(plan, dict) and "Plan" in plan else plan

    n = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("Node Type") in PG_JOIN_TYPES:
            n += 1
        for child in node.get("Plans") or []:
            stack.append(child)
    return n


def join_counts_from_df(df, json_col: str = "json") -> np.ndarray:
    """Vectorized count_joins for a pandas DataFrame with a JSON column."""
    return np.fromiter(
        (count_joins(s) for s in df[json_col]),
        dtype=np.int32,
        count=len(df),
    )


def join_bucket(n: int) -> str:
    """Map join count to canonical bucket name."""
    if n <= 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    return "5+"


# Standard buckets used in Table 4 stratified report
STANDARD_BUCKETS = ("1-2", "3-4", "5+")


def stratify_by_join_count(
    preds,
    targets,
    join_counts,
    buckets=STANDARD_BUCKETS,
    percentiles=(50, 90, 99),
):
    """Group (preds, targets) by join-count bucket; return per-bucket Q-Error.

    Returns:
        dict bucket -> {"n": int, "qNN": float, ...}; bucket "0" is added only
        if non-empty (PG plans with no joins, e.g. single-table scans).
    """
    from utils.metrics import qerror_percentiles

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    join_counts = np.asarray(join_counts)

    out = {}
    masks = {
        "0": join_counts == 0,
        "1-2": (join_counts >= 1) & (join_counts <= 2),
        "3-4": (join_counts >= 3) & (join_counts <= 4),
        "5+": join_counts >= 5,
    }
    bucket_list = list(buckets)
    if masks["0"].any():
        bucket_list = ["0"] + bucket_list

    for b in bucket_list:
        m = masks[b]
        n = int(m.sum())
        entry = {"n": n}
        if n > 0:
            entry.update(qerror_percentiles(preds[m], targets[m], percentiles=percentiles))
        out[b] = entry
    return out
