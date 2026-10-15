"""Shared, versioned definitions for collection, replay and clustered inference."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from scipy.stats import wilcoxon

PLAN_FIELDS = {
    "Node Type", "Parent Relationship", "Parallel Aware", "Async Capable", "Join Type",
    "Inner Unique", "Strategy", "Partial Mode", "Scan Direction", "Relation Name", "Alias",
    "Index Name", "Subplan Name", "Filter", "Index Cond", "Recheck Cond", "Hash Cond",
    "Merge Cond", "Join Filter", "Sort Key", "Group Key", "Presorted Key", "Cache Key",
    "Cache Mode", "Workers Planned", "Single Copy", "CTE Name", "Function Name", "Hash Key",
    "Output", "Startup Cost", "Total Cost", "Plan Rows", "Plan Width",
}
FINGERPRINT_VERSION = "operator-predicate-estimates-v1"


def plan_fingerprint(plan):
    """Ignore execution measurements; retain operator order, predicates and estimates."""
    def visit(node):
        out = {k: v for k, v in node.items() if k in PLAN_FIELDS}
        if "Plans" in node:
            out["Plans"] = [visit(c) for c in node["Plans"]]
        return out
    raw = json.dumps(visit(plan.get("Plan", plan)), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def template_key(query_id):
    m = re.fullmatch(r"(\d+)[a-z]*_(\w+)", str(query_id))
    return f"{m.group(1)}_{m.group(2)}" if m else str(query_id)


def assign_folds(packs, group="template", folds=5, split_seed=42):
    keys = [p.get("template", template_key(p["query_id"])) if group == "template" else p["query_id"]
            for p in packs]
    unique = sorted(set(keys))
    if not 2 <= folds <= len(unique):
        raise ValueError("fold count must be between 2 and the number of independent groups")
    order = np.random.default_rng(split_seed).permutation(len(unique))
    mapping = {unique[int(i)]: j % folds for j, i in enumerate(order)}
    return {i: int(mapping[key]) for i, key in enumerate(keys)}


def cluster_interval(log_ratios, groups, *, samples=20000, seed=0):
    """Query-weighted geometric mean; resample whole template clusters.

    This is conditional on the fitted models. Repeated training seeds and folds are
    separate sources of variation, not additional independent queries.
    """
    values = np.asarray(log_ratios, dtype=float)
    groups = np.asarray(groups)
    if len(values) == 0 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("finite, aligned nonempty ratios and groups required")
    names, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    # Work in bounded chunks for larger workloads.
    draws = []
    for start in range(0, samples, 1000):
        ix = rng.integers(len(names), size=(min(1000, samples - start), len(names)))
        draws.extend(np.exp(sums[ix].sum(1) / counts[ix].sum(1)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    means = sums / counts
    return {"geomean": float(np.exp(values.mean())), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_queries": len(values), "n_templates": len(names), "bootstrap_samples": samples,
            "bootstrap_seed": seed, "resampling_unit": "template", "weighting": "query",
            "conditional_on_fitted_models": True,
            "template_equal_weight_geomean": float(np.exp(means.mean())),
            "template_wilcoxon_p": float(wilcoxon(means).pvalue) if np.any(means != 0) else 1.0}


def validate_complete_pack(pack):
    """Reject incomplete candidate sets instead of silently removing timeout arms."""
    expected = pack.get("expected_hint_ids", list(range(49)))
    rows = pack["results"]
    ids = [int(r["hint_id"]) for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"{pack['query_id']}: missing or duplicate hint IDs")
    for r in rows:
        if "plan" not in r:
            raise ValueError(f"{pack['query_id']}, hint {r['hint_id']}: no planning-time plan")
        if r.get("evaluation_ms", r.get("actual_exec_time_ms")) is None:
            raise ValueError(f"{pack['query_id']}, hint {r['hint_id']}: no measured/censored cost")
    if pack.get("protocol_version") == 2 and not pack.get("complete"):
        raise ValueError(f"{pack['query_id']}: collection has not completed")
