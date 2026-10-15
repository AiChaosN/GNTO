"""Opt-in extended Bao/LIMAO input adapter, NOT the released native featurizer.

Uses their existing seven operator slots and their training-only numeric scaler.
Unlisted PG operators have zero operator bits but retain numeric features and all
children. Unary nodes are retained, with an explicit zero-vector padding leaf;
ordered grouping nodes also have zero vectors. No bitmap or SubPlan is dropped.
The network, loss, optimizer and (for LIMAO) module assigner are not modified.
Runs that use it are extended-tree adapter runs, not native reproductions of Bao / LIMAO.
"""
from __future__ import annotations

import numpy as np

from adapters.plan_representation import GROUP_MARKER, GROUP_TYPE

NATIVE_TYPES = ("Nested Loop", "Hash Join", "Merge Join", "Seq Scan", "Index Scan",
                "Index Only Scan", "Bitmap Index Scan")
PADDING_LABEL = "__GNTO_MISSING_CHILD__"


class ExtendedTreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.stats = stats_extractor
        self.width = len(NATIVE_TYPES) + len(stats_extractor({}))

    def plan_to_feature_tree(self, plan):
        children = plan.get("Plans", [])
        if len(children) > 2:
            raise ValueError("extended native adapter requires ordered-binary-v1 input")
        vec = np.zeros(self.width, dtype=np.float64)
        if plan.get(GROUP_MARKER):
            if plan.get("Node Type") != GROUP_TYPE or len(children) != 2:
                raise ValueError("invalid grouping node")
        else:
            kind = plan.get("Node Type")
            if kind in NATIVE_TYPES:
                vec[NATIVE_TYPES.index(kind)] = 1.0
            vec[len(NATIVE_TYPES):] = self.stats(plan)
        if not np.isfinite(vec).all():
            raise ValueError("non-finite native node features")
        if not children:
            return (vec, plan.get("Relation Name", plan.get("Node Type", "Unknown")))
        left = self.plan_to_feature_tree(children[0])
        right = (self.plan_to_feature_tree(children[1]) if len(children) == 2
                 else (np.zeros(self.width, dtype=np.float64), PADDING_LABEL))
        return (vec, left, right)


def install_extended_tree_builder(featurize_module, enabled=True):
    """Change this process only; external source files remain untouched.

    TreeFeaturizer saves an ExtendedTreeBuilder instance, whose stable module
    path makes checkpoints reloadable without a dynamic/local class.
    """
    if tuple(featurize_module.ALL_TYPES) != NATIVE_TYPES:
        raise ValueError("unexpected native feature layout; inspect the adapter")
    if not hasattr(featurize_module, "_gnto_released_tree_builder"):
        featurize_module._gnto_released_tree_builder = featurize_module.TreeBuilder
    featurize_module.TreeBuilder = (ExtendedTreeBuilder if enabled
                                    else featurize_module._gnto_released_tree_builder)
