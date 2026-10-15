"""Plan JSON -> QueryFormer node features (port of upstream QueryFormer ``model/dataset.py``).

Per node the feature row is (``bin_number = 50``)::

    type(1) | join(1) | filters 3x(col, op, val)(9) | filter mask(3) | hists 3x50(150) | table(1) | sample bitmap(1000)  = 1165

Two ways to build a dataset:

* :class:`PlanTreeDataset` - the upstream class (same constructor, same
  ``__getitem__``), minus the notebook demo helpers.  Eager and padded, so a
  90k-plan split costs ~14 GB of RAM; kept for API compatibility.
* :func:`build_plan_dataset` -> :class:`PlanDataset` - identical features, but
  featurised in parallel across processes, stored **unpadded** and padded to
  30 nodes lazily in ``__getitem__`` (roughly 5x less RAM), and cacheable to a
  single ``.pt`` file.  Plugs into :func:`~models.qf.database_util.collator`
  and :mod:`models.qf.trainer` exactly like ``PlanTreeDataset``.

Both go through :class:`PlanFeaturizer`, which is also what
``adapters.qf_adapter`` uses to build GNTO's PyG graphs, so the two models see
bit-identical node features.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .database_util import (TreeNode, filterDict2Hist, formatFilter, formatJoin, pad_1d_unsqueeze,
                            pad_2d_unsqueeze, pad_attn_bias_unsqueeze, pad_rel_pos_unsqueeze)
from .util import Normalizer

MAX_NODE = 30       # upstream pre_collate(max_node=30)
REL_POS_MAX = 20    # upstream pre_collate(rel_pos_max=20): farther pairs get -inf attention bias
_UNREACH = 60       # upstream floyd_warshall_rewrite "infinity"


# --------------------------------------------------------------------------- #
# node feature (verbatim upstream)
# --------------------------------------------------------------------------- #
def node2feature(node: TreeNode, encoding, hist_file, table_sample) -> np.ndarray:
    """type, join, 3x3 filters, 3 mask, 3x(bins-1) hist, table id, 1000-bit sample."""
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))          # cols, ops, vals
    filts = np.concatenate((filts, pad), axis=1).flatten()     # 3x3 -> 9
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])

    hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    table = np.array([node.table_id])
    if node.table_id == 0:
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]
    return np.concatenate((type_join, filts, mask, hists, table, sample))


def join_columns(join: Optional[str], encoding) -> Tuple[int, int]:
    """Compositional view of a join predicate: ``'mc.movie_id = t.id'`` ->
    ``(col2idx['mc.movie_id'], col2idx['t.id'])``.  ``formatJoin`` already sorts
    the two sides, so the pair is order-independent.  Unknown columns and
    non-join nodes map to the ``NA`` column id, so a join between known
    columns has a representation even if that exact pair never occurred in
    training (the string vocabulary ``join2idx`` cannot do that)."""
    na = encoding.col2idx['NA']
    if not join or ' = ' not in join:
        return na, na
    left, right = join.split(' = ', 1)
    return encoding.col2idx.get(left.strip(), na), encoding.col2idx.get(right.strip(), na)


def shortest_path_matrix(adj: np.ndarray) -> np.ndarray:
    """Vectorised Floyd-Warshall; same values as upstream ``floyd_warshall_rewrite``
    (0 on the diagonal, ``_UNREACH`` for unreachable pairs)."""
    n = adj.shape[0]
    M = np.where(adj.astype(bool), 1, _UNREACH).astype(np.int64)
    np.fill_diagonal(M, 0)
    for k in range(n):
        M = np.minimum(M, M[:, k:k + 1] + M[k:k + 1, :])
    return M


# --------------------------------------------------------------------------- #
# featurizer: plan dict -> tree -> tensors
# --------------------------------------------------------------------------- #
class PlanFeaturizer:
    """Stateless (apart from the vocab inside ``encoding``) plan -> feature converter.

    ``traversePlan`` / ``node2dict`` / ``topo_sort`` / ``calculate_height`` are the
    upstream ``PlanTreeDataset`` methods, unchanged.
    """

    def __init__(self, encoding, hist_file, table_sample):
        self.encoding = encoding
        self.hist_file = hist_file
        self.table_sample = table_sample
        self.treeNodes: List[TreeNode] = []   # upstream kept them for memory collection

    # -- upstream tree construction ----------------------------------------
    def traversePlan(self, plan: Dict[str, Any], idx, encoding=None) -> TreeNode:
        encoding = self.encoding if encoding is None else encoding
        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)

        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        root.join_cols = join_columns(join, encoding)
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx

        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def node2dict(self, treeNode: TreeNode) -> Dict[str, torch.Tensor]:
        adj_list, num_child, features, join_cols = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),
            'join_cols': torch.LongTensor(np.array(join_cols, dtype=np.int64).reshape(-1, 2)),  # [N,2], BFS order
        }

    @staticmethod
    def topo_sort(root_node: TreeNode):
        adj_list = []      # parent -> child
        num_child = []
        features = []
        join_cols = []
        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            features.append(node.feature)
            join_cols.append(getattr(node, 'join_cols', (0, 0)))
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1
        return adj_list, num_child, features, join_cols

    @staticmethod
    def calculate_height(adj_list, tree_size: int) -> np.ndarray:
        if tree_size == 1:
            return np.array([0])
        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)
        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]
        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]
            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order

    # -- convenience --------------------------------------------------------
    def tree_dict(self, plan: Dict[str, Any], idx) -> Dict[str, torch.Tensor]:
        """``{'features' [N,1165], 'heights' [N], 'adjacency_list' [E,2], 'join_cols' [N,2]}`` for one plan
        (the ``Plan`` sub-dict of an EXPLAIN JSON).  Used by the GNTO adapter."""
        root = self.traversePlan(plan, idx)
        d = self.node2dict(root)
        self.treeNodes.clear()
        return d

    def featurize(self, plan: Dict[str, Any], idx) -> Dict[str, torch.Tensor]:
        """Unpadded QueryFormer item: ``features`` [N,1165], ``heights`` [N],
        ``rel_pos`` [N,N] (raw shortest-path lengths).  :func:`pad_item` turns it
        into the padded ``{x, attn_bias, rel_pos, heights}`` the model consumes."""
        d = self.tree_dict(plan, idx)
        n = d['features'].size(0)
        adj = np.zeros((n, n), dtype=bool)
        if n > 1:
            e = d['adjacency_list'].numpy()
            adj[e[:, 0], e[:, 1]] = True
        return {
            'features': d['features'],
            'heights': d['heights'],
            'rel_pos': torch.from_numpy(shortest_path_matrix(adj)).long(),
        }


def pad_item(item: Dict[str, torch.Tensor], max_node: int = MAX_NODE,
             rel_pos_max: int = REL_POS_MAX) -> Dict[str, torch.Tensor]:
    """Unpadded item -> the dict upstream ``PlanTreeDataset.pre_collate`` returns
    (``x`` [1,30,1165], ``attn_bias`` [1,31,31], ``rel_pos`` [1,30,30], ``heights`` [1,30])."""
    feats, heights, rel_pos = item['features'], item['heights'], item['rel_pos']
    n = feats.size(0)
    attn_bias = torch.zeros([n + 1, n + 1], dtype=torch.float)
    attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
    return {
        'x': pad_2d_unsqueeze(feats, max_node),
        'attn_bias': pad_attn_bias_unsqueeze(attn_bias, max_node + 1),
        'rel_pos': pad_rel_pos_unsqueeze(rel_pos, max_node),
        'heights': pad_1d_unsqueeze(heights, max_node),
    }


# --------------------------------------------------------------------------- #
# parallel featurisation
# --------------------------------------------------------------------------- #
_WORKER: Optional[PlanFeaturizer] = None
_KIND = "qf"


def _init_worker(featurizer: PlanFeaturizer, kind: str = "qf") -> None:
    global _WORKER, _KIND
    _WORKER, _KIND = featurizer, kind
    torch.set_num_threads(1)   # forked workers: avoid OpenMP thread-pool deadlocks


def _featurize_row(args: Tuple[Any, str]) -> Dict[str, Any]:
    qid, js = args
    plan = json.loads(js)
    item = _WORKER.featurize(plan['Plan'], qid) if _KIND == "qf" else _WORKER.tree_dict(plan['Plan'], qid)
    item['id'] = qid
    item['cost'] = float(plan['Execution Time'])
    item['card'] = float(plan['Plan']['Actual Rows'])
    return item


def featurize_plans(df: pd.DataFrame, featurizer: PlanFeaturizer, num_workers: Optional[int] = None,
                    chunksize: int = 128, desc: str = "featurize", progress: bool = True,
                    kind: str = "qf") -> List[Dict[str, Any]]:
    """Featurise every row of ``df`` (columns ``id``, ``json``) -> list of items in row
    order, each with ``id`` / ``cost`` (Execution Time, ms) / ``card`` added.
    ``kind='qf'``: unpadded QueryFormer items (features, heights, rel_pos);
    ``kind='tree'``: ``{'features', 'heights', 'adjacency_list'}`` for graph models.
    ``num_workers`` processes (fork; default ``min(16, cpu_count)``).

    The vocabulary must already be complete (``encoding`` frozen / loaded from
    ``encoding.pt``): worker processes cannot grow the parent's vocab."""
    rows = list(zip(df['id'].tolist(), df['json'].tolist()))
    if num_workers is None:
        num_workers = max(1, min(16, os.cpu_count() or 1))
    num_workers = min(num_workers, max(1, len(rows) // chunksize + 1))

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(rows), desc=desc, unit="plan", mininterval=2.0)
        except ImportError:
            pass

    items: List[Dict[str, Any]] = []
    t0 = time.time()
    if num_workers <= 1:
        _init_worker(featurizer, kind)
        for r in rows:
            items.append(_featurize_row(r))
            if bar: bar.update(1)
    else:
        ctx = get_context("fork")
        with ctx.Pool(num_workers, initializer=_init_worker, initargs=(featurizer, kind)) as pool:
            for it in pool.imap(_featurize_row, rows, chunksize=chunksize):
                items.append(it)
                if bar: bar.update(1)
    if bar:
        bar.close()
    print(f"[{desc}] {len(items):,} plans in {time.time() - t0:.1f}s ({num_workers} workers)", flush=True)
    return items


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class PlanDataset(Dataset):
    """Unpadded items + labels; pads lazily.  Same ``__getitem__`` contract as
    ``PlanTreeDataset``: ``(padded_dict, (cost_label, card_label))``."""

    def __init__(self, items: Sequence[Dict[str, Any]], cost_norm: Normalizer,
                 card_norm: Optional[Normalizer] = None, max_node: int = MAX_NODE,
                 rel_pos_max: int = REL_POS_MAX):
        self.items = list(items)
        self.max_node, self.rel_pos_max = max_node, rel_pos_max
        self.ids = [it.get('id') for it in self.items]
        self.costs = [float(it['cost']) for it in self.items]
        self.cards = [float(it['card']) for it in self.items]
        card_norm = card_norm or Normalizer(1, 100)
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.to_predict = 'cost'
        self.gts, self.labels = self.costs, self.cost_labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return pad_item(self.items[idx], self.max_node, self.rel_pos_max), (self.cost_labels[idx], self.card_labels[idx])

    def num_nodes(self) -> List[int]:
        return [int(it['features'].size(0)) for it in self.items]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.items, path)

    @classmethod
    def load(cls, path: str, cost_norm: Normalizer, card_norm: Optional[Normalizer] = None) -> "PlanDataset":
        return cls(torch.load(path, weights_only=False), cost_norm, card_norm)


def build_plan_dataset(df: pd.DataFrame, encoding, hist_file, table_sample, cost_norm: Normalizer,
                       card_norm: Optional[Normalizer] = None, cache_path: Optional[str] = None,
                       num_workers: Optional[int] = None, desc: str = "featurize") -> PlanDataset:
    """Featurise ``df`` (parallel) into a :class:`PlanDataset`; load from / save to
    ``cache_path`` (a list of unpadded items) when given."""
    if cache_path and os.path.exists(cache_path):
        print(f"[{desc}] loading cache {cache_path}", flush=True)
        return PlanDataset.load(cache_path, cost_norm, card_norm)
    items = featurize_plans(df, PlanFeaturizer(encoding, hist_file, table_sample), num_workers=num_workers, desc=desc)
    ds = PlanDataset(items, cost_norm, card_norm)
    if cache_path:
        ds.save(cache_path)
        print(f"[{desc}] cached to {cache_path}", flush=True)
    return ds


class PlanTreeDataset(Dataset):
    """Upstream class (constructor signature unchanged): eager, padded, single-process."""

    def __init__(self, json_df: pd.DataFrame, train, encoding, hist_file, card_norm, cost_norm,
                 to_predict, table_sample):
        self.json_df = json_df
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self._featurizer = PlanFeaturizer(encoding, hist_file, table_sample)
        self.length = len(json_df)

        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        self.cards = [node['Actual Rows'] for node in nodes]
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))

        self.to_predict = to_predict
        if to_predict in ('cost', 'both'):
            self.gts, self.labels = self.costs, self.cost_labels
        elif to_predict == 'card':
            self.gts, self.labels = self.cards, self.card_labels
        else:
            raise Exception('Unknown to_predict type')

        idxs = list(json_df['id'])
        self.collated_dicts = [self.js_node2dict(i, node) for i, node in zip(idxs, nodes)]

    # upstream API kept for callers that drive the featurizer by hand
    @property
    def treeNodes(self):
        return self._featurizer.treeNodes

    def traversePlan(self, plan, idx, encoding=None):
        return self._featurizer.traversePlan(plan, idx, encoding)

    def node2dict(self, treeNode):
        return self._featurizer.node2dict(treeNode)

    def js_node2dict(self, idx, node):
        return pad_item(self._featurizer.featurize(node, idx))

    def pre_collate(self, the_dict, max_node=MAX_NODE, rel_pos_max=REL_POS_MAX):
        n = the_dict['features'].size(0)
        adj = np.zeros((n, n), dtype=bool)
        if n > 1:
            e = the_dict['adjacency_list'].numpy()
            adj[e[:, 0], e[:, 1]] = True
        item = {'features': the_dict['features'], 'heights': the_dict['heights'],
                'rel_pos': torch.from_numpy(shortest_path_matrix(adj)).long()}
        return pad_item(item, max_node, rel_pos_max)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])


__all__ = ["PlanFeaturizer", "PlanTreeDataset", "PlanDataset", "build_plan_dataset", "featurize_plans",
           "pad_item", "node2feature", "join_columns", "shortest_path_matrix", "MAX_NODE", "REL_POS_MAX"]
