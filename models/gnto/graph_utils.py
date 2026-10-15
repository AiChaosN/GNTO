"""Graph-construction helpers shared by the encoders and the data pipelines.

Edge convention (PyTorch Geometric): an entry ``(src, dst)`` in ``edge_index``
means *messages flow from src to dst*.  The existing GNTO data builders
(``QueryFormerToPyGDataset`` and ``legacy.DataPreprocessor.plan_trees_to_graphs``)
store **parent -> child** edges; :func:`plan_to_graph` below lets you choose.
Whatever is stored, :class:`models.GraphEncoder` re-orients it at run time
according to ``GraphEncoderConfig.edge_mode``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops as _pyg_add_self_loops
from torch_geometric.utils import remove_self_loops, to_undirected

EDGE_DIRECTIONS = ("child_to_parent", "parent_to_child", "both")


# --------------------------------------------------------------------------- #
# edge_index transforms
# --------------------------------------------------------------------------- #
def transform_edges(edge_index: Tensor, mode: str, num_nodes: Optional[int] = None) -> Tensor:
    """Re-orient a stored ``edge_index`` (``[2, E]``).

    * ``"as_is"``          - unchanged
    * ``"reverse"``        - ``(src, dst) -> (dst, src)``
    * ``"bidirectional"``  - union of both directions, duplicates removed
                             (idempotent: already-undirected input is unchanged)
    """
    if mode == "as_is":
        return edge_index
    if mode == "reverse":
        return edge_index.flip(0)
    if mode == "bidirectional":
        if edge_index.numel() == 0:
            return edge_index
        return to_undirected(edge_index, num_nodes=num_nodes)
    raise ValueError(f"unknown edge_mode {mode!r}")


def with_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Return ``edge_index`` with exactly one ``(i, i)`` edge per node."""
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = _pyg_add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


def first_node_index(batch: Tensor) -> Tensor:
    """Index of the first node of every graph in a PyG batch vector.

    For graphs built by a DFS pre-order traversal (all GNTO builders) the first
    node is the plan root.
    """
    if batch.numel() == 0:
        return batch.new_zeros(0)
    change = torch.ones_like(batch, dtype=torch.bool)
    change[1:] = batch[1:] != batch[:-1]
    return change.nonzero(as_tuple=False).view(-1)


def add_virtual_nodes(
    x: Tensor,
    edge_index: Tensor,
    batch: Optional[Tensor],
    vnode_feat: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Append one virtual node per graph, connected in both directions to every
    node of that graph.  Virtual nodes are placed *after* the real nodes so the
    caller can slice them off again with ``h[:num_real]``.

    Returns ``(x, edge_index, batch, num_real_nodes)``.
    """
    num_real = x.size(0)
    if batch is None:
        batch = x.new_zeros(num_real, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1 if num_real > 0 else 0
    device = x.device

    v_of_node = num_real + batch                                  # [N]
    node_ids = torch.arange(num_real, device=device)
    e_up = torch.stack([node_ids, v_of_node], dim=0)              # node -> vnode
    e_down = torch.stack([v_of_node, node_ids], dim=0)            # vnode -> node

    x_new = torch.cat([x, vnode_feat.to(x.dtype).unsqueeze(0).expand(num_graphs, -1)], dim=0)
    ei_new = torch.cat([edge_index, e_up, e_down], dim=1)
    batch_new = torch.cat([batch, torch.arange(num_graphs, device=device, dtype=batch.dtype)])
    return x_new, ei_new, batch_new, num_real


# --------------------------------------------------------------------------- #
# EXPLAIN-JSON tree -> (node payloads, edge_index)
# --------------------------------------------------------------------------- #
NodeFn = Callable[[Dict[str, Any], int, Optional[int]], Any]


def plan_to_graph(
    plan: Dict[str, Any],
    node_fn: Optional[NodeFn] = None,
    direction: str = "child_to_parent",
    children_key: str = "Plans",
    self_loops: bool = False,
) -> Tuple[List[Any], Tensor]:
    """DFS pre-order walk over an EXPLAIN-style JSON plan tree.

    ``node_fn(node_dict, node_index, parent_index)`` turns each operator into
    whatever payload you want (feature list, dict, ...); by default the raw
    dict without its children is returned.  The root always gets index 0.

    ``direction`` fixes what is stored in ``edge_index``:
    ``"child_to_parent"`` (data flow, bottom-up), ``"parent_to_child"``
    (legacy builders, top-down) or ``"both"``.
    """
    if direction not in EDGE_DIRECTIONS:
        raise ValueError(f"direction={direction!r}; expected one of {EDGE_DIRECTIONS}")
    if "Plan" in plan and children_key not in plan:
        plan = plan["Plan"]

    payloads: List[Any] = []
    edges: List[Tuple[int, int]] = []

    def default_node_fn(node: Dict[str, Any], _idx: int, _parent: Optional[int]) -> Dict[str, Any]:
        return {k: v for k, v in node.items() if k != children_key}

    fn = node_fn or default_node_fn

    stack: List[Tuple[Dict[str, Any], Optional[int]]] = [(plan, None)]
    while stack:
        node, parent = stack.pop()
        idx = len(payloads)
        payloads.append(fn(node, idx, parent))
        if self_loops:
            edges.append((idx, idx))
        if parent is not None:
            if direction in ("child_to_parent", "both"):
                edges.append((idx, parent))
            if direction in ("parent_to_child", "both"):
                edges.append((parent, idx))
        children = node.get(children_key) or []
        # push in reverse so the left-most child is visited first (pre-order)
        for child in reversed(children):
            stack.append((child, idx))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return payloads, edge_index


def edge_index_from_pairs(pairs: Sequence[Sequence[int]]) -> Tensor:
    """``[(src, dst), ...]`` or ``[[src, dst], ...]`` -> ``[2, E]`` long tensor."""
    if len(pairs) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.as_tensor(pairs, dtype=torch.long)
    if ei.dim() != 2:
        raise ValueError("pairs must be 2-D")
    if ei.size(0) != 2 and ei.size(1) == 2:
        ei = ei.t()
    return ei.contiguous()
