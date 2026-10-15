"""Plan trees -> DACE inputs (port of ``DACE/plan_utils.py`` without the global ``ROOT_DIR``).

:class:`NodeFeaturizer` turns one plan node into the 22-d vector
``[one-hot node type (20) | (Total Cost - center) / scale | (Plan Rows - center) / scale]``
(RobustScaler statistics from ``statistics.json``).  It is shared with
``adapters.dace_adapter`` so GNTO and DACE see bit-identical node features.

:func:`plan_encoding` produces what ``DACELora`` consumes: the node sequence in
DFS pre-order padded to ``pad_length`` nodes, per-node runtimes normalised by
``max_runtime``, the tree-reachability attention mask and the ``0.5^height``
loss mask.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

PLAN_PARAMETERS = ("Node Type", "Total Cost", "Plan Rows")
PAD_LENGTH = 50          # upstream configs["pad_length"]
MAX_RUNTIME = 30000.0    # upstream configs["max_runtime"] (ms)
LOSS_WEIGHT = 0.5        # upstream configs["loss_weight"]


def load_plans(path: str) -> List[Dict[str, Any]]:
    """``<db>_filted.json`` -> list of ``{"Plan": tree}`` (nested-list wrappers unwrapped)."""
    with open(path) as f:
        plans = json.load(f)
    out = []
    for p in plans:
        while isinstance(p, list):
            p = p[0]
        if "Plan" in p:
            out.append(p)
    return out


def count_nodes(node: Dict[str, Any]) -> int:
    return 1 + sum(count_nodes(c) for c in node.get("Plans", []))


class NodeFeaturizer:
    """One plan node -> 22-d feature row (one-hot type + RobustScaler-scaled cost / rows)."""

    def __init__(self, stats: Dict[str, Any]):
        self.stats = stats
        self.type_index: Dict[str, int] = dict(stats["node_types"]["value_dict"])
        self.num_types = len(self.type_index)
        self.node_length = self.num_types + len(PLAN_PARAMETERS) - 1
        self.scalers = {k: (float(stats[k]["center"]), float(stats[k]["scale"])) for k in PLAN_PARAMETERS[1:]}

    def scale(self, key: str, value: float) -> float:
        center, scale = self.scalers[key]
        return 0.0 if scale == 0 else (float(value) - center) / scale

    def node_vector(self, node: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.node_length, dtype=np.float32)
        idx = self.type_index.get(node.get("Node Type", ""), None)
        if idx is not None and idx < self.num_types:
            vec[idx] = 1.0
        vec[self.num_types] = self.scale("Total Cost", node.get("Total Cost", 0.0))
        vec[self.num_types + 1] = self.scale("Plan Rows", node.get("Plan Rows", 0.0))
        return vec


def fit_node_statistics(plans) -> Dict[str, Any]:
    """Fit vocabulary and robust scales using only the supplied training plans.

    Runtime fields are deliberately never read. Unknown test operator types map to
    an all-zero type vector through NodeFeaturizer.
    """
    types, values = set(), {k: [] for k in PLAN_PARAMETERS[1:]}
    for plan in plans:
        stack = [plan.get("Plan", plan)]
        while stack:
            node = stack.pop()
            types.add(node.get("Node Type", ""))
            for key in values:
                values[key].append(float(node.get(key, 0.0)))
            stack.extend(node.get("Plans", []))
    if not types:
        raise ValueError("cannot fit feature statistics without training nodes")
    result = {"node_types": {"value_dict": {t: i for i, t in enumerate(sorted(types))}}}
    for key, v in values.items():
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        result[key] = {"center": float(q50), "scale": float(q75 - q25) or 1.0}
    return result


def _dfs(node, seq, adjs, parent, run_times, heights, height):
    cur = len(seq)
    seq.append(node)
    heights.append(height)
    run_times.append(node["Actual Total Time"])
    if parent != -1:
        adjs.append((parent, cur))
    for child in node.get("Plans", []):
        _dfs(child, seq, adjs, cur, run_times, heights, height + 1)


def attention_mask(adjs: Sequence[Tuple[int, int]], n: int, pad_length: int) -> torch.Tensor:
    """True = masked.  A node attends to itself and its descendants (upstream ``get_attention_mask``)."""
    m = np.ones((pad_length, pad_length), dtype=bool)
    for a, b in adjs:
        m[a, b] = False
    for i in range(n):
        for j in range(n):
            if not m[i, j]:
                for k in range(n):
                    if not m[j, k]:
                        m[i, k] = False
    np.fill_diagonal(m, False)
    return torch.from_numpy(m)


def plan_encoding(plan: Dict[str, Any], fz: NodeFeaturizer, pad_length: int = PAD_LENGTH,
                  loss_weight: float = LOSS_WEIGHT, max_runtime: float = MAX_RUNTIME):
    """-> ``(seq [pad*node_length] float32, run_times [pad], attn_mask [pad,pad] bool, loss_mask [pad])``."""
    seq, run_times, adjs, heights = [], [], [], []
    _dfs(plan["Plan"], seq, adjs, -1, run_times, heights, 0)
    n = len(seq)
    if n > pad_length:
        raise ValueError(f"plan has {n} nodes > pad_length {pad_length}")
    rt = np.asarray(run_times + [1.0] * (pad_length - n), dtype=np.float32) / max_runtime + 1e-7
    enc = np.concatenate([fz.node_vector(node) for node in seq]).astype(np.float32)
    enc = pad(torch.from_numpy(enc), (0, pad_length * fz.node_length - enc.shape[0]), value=0.0)
    lm = np.zeros(pad_length, dtype=np.float32)
    lm[:n] = np.power(loss_weight, np.asarray(heights, dtype=np.float32))
    return enc, torch.from_numpy(rt), attention_mask(adjs, n, pad_length), torch.from_numpy(lm)


class DACEDataset(Dataset):
    """Stacked tensors for one or more databases; ``__getitem__`` -> (seq, attn_mask, loss_mask, run_times)."""

    def __init__(self, seqs, attn_masks, loss_masks, run_times, n_nodes=None, db_ids=None):
        self.seqs, self.attn_masks, self.loss_masks, self.run_times = seqs, attn_masks, loss_masks, run_times
        self.n_nodes = n_nodes if n_nodes is not None else torch.zeros(len(seqs), dtype=torch.long)
        self.db_ids = db_ids if db_ids is not None else torch.zeros(len(seqs), dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.attn_masks[i], self.loss_masks[i], self.run_times[i]

    @property
    def root_ms(self) -> np.ndarray:
        """Root actual runtime in ms (undo the normalisation)."""
        return ((self.run_times[:, 0] - 1e-7) * MAX_RUNTIME).numpy().astype(np.float64)

    def subset(self, idx) -> "DACEDataset":
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return DACEDataset(self.seqs[idx], self.attn_masks[idx], self.loss_masks[idx], self.run_times[idx],
                           self.n_nodes[idx], self.db_ids[idx])

    @staticmethod
    def concat(parts: Sequence["DACEDataset"]) -> "DACEDataset":
        return DACEDataset(*[torch.cat([getattr(p, a) for p in parts]) for a in
                             ("seqs", "attn_masks", "loss_masks", "run_times", "n_nodes", "db_ids")])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({a: getattr(self, a) for a in ("seqs", "attn_masks", "loss_masks", "run_times", "n_nodes", "db_ids")}, path)

    @classmethod
    def load(cls, path: str) -> "DACEDataset":
        return cls(**torch.load(path, weights_only=False))


def encode_plans(plans: Sequence[Dict[str, Any]], fz: NodeFeaturizer, db_id: int = 0,
                 pad_length: int = PAD_LENGTH) -> DACEDataset:
    seqs, rts, ams, lms, nn_ = [], [], [], [], []
    for p in plans:
        s, r, a, l = plan_encoding(p, fz, pad_length)
        seqs.append(s); rts.append(r); ams.append(a); lms.append(l); nn_.append(count_nodes(p["Plan"]))
    return DACEDataset(torch.stack(seqs), torch.stack(ams), torch.stack(lms), torch.stack(rts),
                       torch.tensor(nn_, dtype=torch.long), torch.full((len(plans),), db_id, dtype=torch.long))


def build_dace_dataset(plans: Sequence[Dict[str, Any]], fz: NodeFeaturizer, db_id: int = 0,
                       cache_path: Optional[str] = None) -> DACEDataset:
    if cache_path and os.path.exists(cache_path):
        return DACEDataset.load(cache_path)
    ds = encode_plans(plans, fz, db_id)
    if cache_path:
        ds.save(cache_path)
    return ds


__all__ = ["PLAN_PARAMETERS", "PAD_LENGTH", "MAX_RUNTIME", "LOSS_WEIGHT", "load_plans", "count_nodes",
           "NodeFeaturizer", "plan_encoding", "attention_mask", "DACEDataset", "encode_plans", "build_dace_dataset"]
