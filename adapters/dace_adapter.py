"""DACE adapter: DACE's schema-agnostic node features -> PyG graphs for GNTO.

Both models featurise nodes through :class:`models.dace.NodeFeaturizer`
(one-hot node type + RobustScaler-scaled Total Cost / Plan Rows = 22-d), so
they see bit-identical inputs; GNTO additionally gets the tree as a graph
(parent<->child edges) instead of DACE's padded DFS sequence + reachability mask.

Usage::

    from adapters.dace_adapter import plans_to_pyg, build_pyg_dataset, GNTO_DACE_Model
    from models.dace import load_statistics, load_plans, plans_path, NodeFeaturizer
    fz = NodeFeaturizer(load_statistics())
    data_list = build_pyg_dataset(load_plans(plans_path("walmart")), fz, cache_path=...)
    model = GNTO_DACE_Model(fz.node_length)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gnto_root not in sys.path:
    sys.path.insert(0, _gnto_root)

from models.dace import MAX_RUNTIME, NodeFeaturizer, count_nodes  # noqa: E402
from models.gnto import GNTO, legacy_dace_config  # noqa: E402


def plan_to_graph(root: Dict[str, Any], fz: NodeFeaturizer):
    """``(x [N,22], edge_index [2,2E] parent<->child)`` in DFS pre-order (same order as DACE)."""
    nodes, edges = [], []

    def dfs(node, parent):
        cur = len(nodes)
        nodes.append(fz.node_vector(node))
        if parent is not None:
            edges.append([parent, cur])
            edges.append([cur, parent])
        for child in node.get("Plans", []):
            dfs(child, cur)

    dfs(root, None)
    x = torch.from_numpy(np.stack(nodes)).float()
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index


def plans_to_pyg(plans: Sequence[Dict[str, Any]], fz: NodeFeaturizer, db_id: int = 0) -> List[Data]:
    """One ``Data`` per plan: ``x``, ``edge_index``, ``y`` = root Actual Total Time (ms),
    ``n_nodes``, ``db_id``, ``plan_id`` (position in the file)."""
    out = []
    for i, p in enumerate(plans):
        root = p["Plan"]
        x, ei = plan_to_graph(root, fz)
        d = Data(x=x, edge_index=ei, y=torch.tensor([float(root.get("Actual Total Time", 0.0))], dtype=torch.float))
        d.n_nodes, d.db_id, d.plan_id = count_nodes(root), db_id, i
        out.append(d)
    return out


def build_pyg_dataset(plans: Sequence[Dict[str, Any]], fz: NodeFeaturizer, db_id: int = 0,
                      cache_path: Optional[str] = None) -> List[Data]:
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    data_list = plans_to_pyg(plans, fz, db_id)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(data_list, cache_path)
    return data_list


def GNTO_DACE_Model(input_dim: int, hidden_dim: int = 64) -> GNTO:
    """GNTO on DACE features (preset ``dace_zero_shot`` / ``legacy_dace_config``): MLP node
    encoder -> 3-layer GATv2 (heads 4/4/1, dropout 0.1) -> mean readout -> residual head ->
    sigmoid (target = runtime / 30 s)."""
    return GNTO.from_config(legacy_dace_config(input_dim, hidden_dim))


def q_error_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean log Q-error on the root prediction (GNTO's training loss on this benchmark)."""
    preds = torch.clamp(preds, min=1e-7)
    targets = torch.clamp(targets, min=1e-7)
    return torch.mean(torch.log(torch.max(preds / targets, targets / preds)))


def q_error_np(preds, targets) -> np.ndarray:
    p = np.maximum(np.asarray(preds, dtype=np.float64), 1e-7)
    t = np.maximum(np.asarray(targets, dtype=np.float64), 1e-7)
    return np.maximum(p / t, t / p)


__all__ = ["plan_to_graph", "plans_to_pyg", "build_pyg_dataset", "GNTO_DACE_Model", "q_error_loss", "q_error_np",
           "MAX_RUNTIME"]
