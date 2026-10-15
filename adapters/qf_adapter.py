"""QueryFormer adapter: QueryFormer node features -> PyG graphs for GNTO.

Everything QueryFormer-side now lives in this repo (``models.qf`` for the code,
``data/qf`` for histograms / sample bitmaps / vocabulary), so no separate
QueryFormer checkout is needed.

Usage::

    from adapters.qf_adapter import (
        load_qf_resources, QueryFormerToPyGDataset, GNTO_QF_Model,
        unnormalize, calc_q_error, evaluate, evaluate_full,
    )
    res = load_qf_resources()                      # encoding / hist / table_sample / cost_norm
    ds = QueryFormerToPyGDataset(df, res["encoding"], res["hist_file"], res["table_sample"],
                                 res["cost_norm"], cache_name="train_full", add_plan_rows=True, add_join_cols=True)
    model = GNTO_QF_Model(res["encoding"])          # = preset "gnto": plan rows + compositional joins

Both GNTO and QueryFormer featurise plans through :class:`models.qf.PlanFeaturizer`,
so the per-node 1165-d features are bit-identical across the two models; GNTO
appends a ``log1p(Plan Rows)`` channel and the join's two column ids (1168-d, ``gnto`` preset).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gnto_root not in sys.path:
    sys.path.insert(0, _gnto_root)

from models.gnto import GNTO, legacy_qf_config  # noqa: E402
from models.qf import PlanFeaturizer, featurize_plans  # noqa: E402
from models.qf import resources as _qf_res  # noqa: E402

CACHE_DIR = os.path.join(_gnto_root, "data", "process", "cache")


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------
def load_qf_resources(zero_table_sample: bool = False, sample: str = "train", **kwargs: Any) -> Dict[str, Any]:
    """QueryFormer featurisation resources from ``data/qf`` (see
    :func:`models.qf.resources.load_qf_resources`).

    ``zero_table_sample=True`` replaces the per-query sample bitmaps with zeros -
    only for plans that are in no sample file (live PG steering); training and
    the job-light / synthetic workloads should use the real bitmaps.
    """
    return _qf_res.load_qf_resources(zero_table_sample=zero_table_sample, sample=sample, **kwargs)


# ---------------------------------------------------------------------------
# dataset: QueryFormer features -> PyG Data
# ---------------------------------------------------------------------------
class QueryFormerToPyGDataset(Dataset):
    """``Data(x=[N,1165(+1)], edge_index=[2,E] parent->child, y=normalised log-latency,
    raw_y=Execution Time ms)`` per plan, cached as a list under ``data/process/cache``.

    Args:
        df: DataFrame with ``json`` and ``id`` columns
        encoding / hist_file / table_sample / cost_norm: from :func:`load_qf_resources`
        cache_name: cache file prefix (``<cache_name>[_planrows][_joincols]_<n>.pt``); None = no cache
        add_plan_rows: append ``log1p(Plan Rows)`` per node (1165 -> 1166 dims)
        add_join_cols: append the join's ``(col_left, col_right)`` column ids per node
            (+2 dims, after plan_rows) for ``node.join_encoding != "id"``
        num_workers: featurisation processes (fork); 1 = in-process
    """

    def __init__(self, df, encoding, hist_file, table_sample, cost_norm,
                 cache_name: Optional[str] = None, add_plan_rows: bool = False,
                 num_workers: Optional[int] = None, add_join_cols: bool = False):
        self.df = df.reset_index(drop=True)
        self.cost_norm = cost_norm
        self.data_list: List[Data] = []

        cache_file = None
        if cache_name:
            suffix = ("_planrows" if add_plan_rows else "") + ("_joincols" if add_join_cols else "")
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_file = os.path.join(CACHE_DIR, f"{cache_name}{suffix}_{len(self.df)}.pt")
            if os.path.exists(cache_file):
                print(f"Loading cached data from {cache_file}...", flush=True)
                self.data_list = torch.load(cache_file, weights_only=False)
                return

        items = featurize_plans(self.df, PlanFeaturizer(encoding, hist_file, table_sample),
                                num_workers=num_workers, desc=f"gnto-featurize[{cache_name or 'nocache'}]",
                                kind="tree")
        for row_idx, it in enumerate(items):
            x = it['features']
            adj = it['adjacency_list']
            edge_index = adj.t().long() if adj.numel() > 0 else torch.zeros((2, 0), dtype=torch.long)
            if add_plan_rows:
                rows = self._plan_rows(json.loads(self.df.iloc[row_idx]['json'])['Plan'])
                rows_t = torch.tensor(rows, dtype=torch.float).view(-1, 1)
                if x.size(0) == rows_t.size(0):
                    x = torch.cat([x, rows_t], dim=1)
            if add_join_cols:
                x = torch.cat([x, it['join_cols'].float()], dim=1)
            exec_time = it['cost']
            y_norm = np.clip((np.log(exec_time + 0.001) - cost_norm.mini) / (cost_norm.maxi - cost_norm.mini), 0.001, 1.0)
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y_norm], dtype=torch.float))
            data.raw_y = torch.tensor([exec_time], dtype=torch.float)
            self.data_list.append(data)

        if cache_file:
            print(f"Saving cache to {cache_file}...", flush=True)
            torch.save(self.data_list, cache_file)

    @staticmethod
    def _plan_rows(node: Dict[str, Any]) -> List[float]:
        """BFS (same order as ``PlanFeaturizer.topo_sort``) ``log1p(Plan Rows)`` per node."""
        out: List[float] = []
        queue = [node]
        while queue:
            n = queue.pop(0)
            out.append(float(np.log1p(float(n['Plan Rows']))))
            queue.extend(n.get('Plans', []))
        return out

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def GNTO_QF_Model(encoding, hidden_dim: int = 64, add_plan_rows: bool = True, join_encoding: str = "columns") -> GNTO:
    """GNTO on QueryFormer features (``legacy_qf_config``): QueryFormerNodeEncoder
    (+ Plan Rows channel, join as string id / column pair / both) -> 3-layer GATv2
    (heads 4/2/1) -> mean readout -> residual head -> sigmoid.  Loads pre-rewrite
    checkpoints via ``load_state_dict`` (``join_encoding="id"``)."""
    return GNTO.from_config(legacy_qf_config(encoding, add_plan_rows=add_plan_rows, hidden_dim=hidden_dim,
                                             join_encoding=join_encoding))


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def unnormalize(y_norm, cost_norm):
    """Normalised prediction -> execution time (ms)."""
    val = np.asarray(y_norm, dtype=np.float64) * (cost_norm.maxi - cost_norm.mini) + cost_norm.mini
    return np.exp(val) - 0.001


def calc_q_error(preds, targets):
    """Q-error percentiles [50, 75, 90, 95, 99] as an array."""
    from models.gnto.metrics import qerror_percentiles
    q = qerror_percentiles(preds, targets, percentiles=(50, 75, 90, 95, 99))
    return np.array([q["q50"], q["q75"], q["q90"], q["q95"], q["q99"]])


@torch.no_grad()
def _run_inference(model, loader, cost_norm, device):
    model.eval()
    preds_all, targets_all, total_loss = [], [], 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        total_loss += torch.nn.functional.mse_loss(out, batch.y.view(-1)).item() * batch.num_graphs
        preds_all.append(unnormalize(out.cpu().numpy(), cost_norm))
        targets_all.append(batch.raw_y.view(-1).cpu().numpy())
    n = len(loader.dataset)
    return total_loss / max(n, 1), np.concatenate(preds_all), np.concatenate(targets_all)


def evaluate(model, loader, cost_norm, device):
    """``(avg_loss, q50, q75, q90, q95, q99)`` on a PyG DataLoader."""
    avg_loss, preds, targets = _run_inference(model, loader, cost_norm, device)
    return (avg_loss, *calc_q_error(preds, targets))


def evaluate_full(model, loader, cost_norm, device, group_ids=None):
    """Q-error percentiles + Spearman / Kendall / pairwise accuracy (+ top-1 regret
    with ``group_ids``); ``preds`` / ``targets`` arrays included."""
    from models.gnto.metrics import evaluation_summary
    avg_loss, preds, targets = _run_inference(model, loader, cost_norm, device)
    summary = {"avg_loss": float(avg_loss)}
    summary.update(evaluation_summary(preds, targets, group_ids=group_ids))
    summary["preds"], summary["targets"] = preds, targets
    return summary


__all__ = ["load_qf_resources", "QueryFormerToPyGDataset", "GNTO_QF_Model", "unnormalize", "calc_q_error",
           "evaluate", "evaluate_full", "CACHE_DIR"]
