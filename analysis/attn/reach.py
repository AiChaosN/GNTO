"""Edge sets for the reach control: the same attention layer (PyG TransformerConv, the ``transformer``
conv of models/gnto/graph_encoder.py) with three reaches.

    local    parent <-> child edges of the plan (GNTO's topology; three layers reach three hops)
    subplan  every operator receives messages from every operator of its sub-plan (QueryFormer / DACE mask)
    global   every operator receives messages from every other operator of the plan

The plan nodes are in DFS pre-order and the adapter emits both directions of every parent-child edge, so the
parent of an edge is the endpoint with the smaller index.
"""
from __future__ import annotations

import torch

REACHES = ("local", "subplan", "global")


def reach_edges(edge_index: torch.Tensor, n: int, reach: str) -> torch.Tensor:
    if reach == "local" or n < 2:
        return edge_index
    if reach == "global":
        src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        keep = src != dst
        return torch.stack([src[keep], dst[keep]]).contiguous()
    if reach != "subplan":
        raise ValueError(f"unknown reach {reach!r}")
    children = {i: [] for i in range(n)}
    for u, v in edge_index.t().tolist():
        if u < v:
            children[u].append(v)
    edges = []
    for i in range(n):
        stack = list(children[i])
        while stack:
            j = stack.pop()
            edges.append([j, i])                       # descendant j -> ancestor i
            stack.extend(children[j])
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
