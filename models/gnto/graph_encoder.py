"""Structure-aware message passing over the plan DAG.

:class:`GraphEncoder` is a stack of :class:`GraphEncoderLayer`s, each
``conv -> norm -> (+ residual) -> activation -> dropout`` - the same per-layer
recipe as the legacy ``GATv2TreeEncoder_V3``, but with depth, heads, conv
type, edge orientation, self-loops, virtual node, normalisation and
jumping-knowledge all set from :class:`GraphEncoderConfig`.

``forward`` returns ``(node_embeddings, aux)``; ``aux`` carries the virtual
node embeddings (if enabled) and per-layer attention coefficients (if
requested), so the plan-level readout and any analysis code can use them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import dropout_edge

from .config import GraphEncoderConfig
from .graph_utils import add_virtual_nodes, transform_edges, with_self_loops

_ACT = {
    "elu": F.elu,
    "relu": F.relu,
    "gelu": F.gelu,
    "leaky_relu": F.leaky_relu,
    "silu": F.silu,
}
ATTENTION_CONVS = ("gatv2", "gat", "gatv2_nonorm", "gatv2_pos", "transformer")


def child_positions(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor, Tensor]:
    """For parent->child edges: ``(src, dst, pos)`` sorted by parent, where ``pos`` is the
    child's rank among its parent's children (0 = left, 1 = right, 2+ = later)."""
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return src, dst, src
    order = torch.argsort(src * num_nodes + dst)
    src, dst = src[order], dst[order]
    first = torch.ones_like(src, dtype=torch.bool)
    first[1:] = src[1:] != src[:-1]
    idx = torch.arange(src.numel(), device=src.device)
    start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), dim=0).values
    return src, dst, idx - start


class TreeAttnConv(nn.Module):
    """Binary tree convolution with attention over its three slots.

    Like :class:`TreeConv` the node sees ``[h_i ; h_left ; h_right]`` (ordered children,
    zeros if missing), but each slot is gated by a GATv2-style score
    ``alpha_s = sigma(a^T LeakyReLU(W_q h_i + W_k h_s + p_s))`` with a learnable slot
    embedding ``p_s`` (self / left / right).  ``normalize=False`` (``tcnn_attn``) uses a
    sigmoid gate per slot, so the aggregate still grows with the subtree like TCNN;
    ``normalize=True`` (``tcnn_attn_softmax``) uses a softmax over the present slots.
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = False, negative_slope: float = 0.2):
        super().__init__()
        self.normalize, self.negative_slope = normalize, negative_slope
        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.slot = nn.Parameter(torch.zeros(3, in_dim))
        self.att = nn.Parameter(torch.empty(in_dim))
        nn.init.xavier_uniform_(self.att.view(1, -1))
        nn.init.normal_(self.slot, std=0.02)
        self.lin = nn.Linear(3 * in_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n, d = x.shape
        left, right = x.new_zeros(n, d), x.new_zeros(n, d)
        present = x.new_zeros(n, 3)
        present[:, 0] = 1.0
        src, dst, pos = child_positions(edge_index, n)
        m0, m1 = pos == 0, pos == 1
        left[src[m0]] = x[dst[m0]]
        right[src[m1]] = x[dst[m1]]
        present[src[m0], 1] = 1.0
        present[src[m1], 2] = 1.0
        slots = torch.stack([x, left, right], dim=1)                                 # [N,3,d]
        score = F.leaky_relu(self.q(x).unsqueeze(1) + self.k(slots) + self.slot, self.negative_slope)
        score = (score * self.att).sum(dim=-1)                                       # [N,3]
        if self.normalize:
            score = score.masked_fill(present == 0, float("-inf"))
            alpha = torch.softmax(score, dim=1)
        else:
            alpha = torch.sigmoid(score) * present
        return self.lin((slots * alpha.unsqueeze(-1)).flatten(1))


class PosGATv2Conv(nn.Module):
    """GATv2 whose attention scores see the child's position under its parent
    (one-hot left / right / later child, as an edge feature) - GAT given TCNN's
    ordered-children information.  Self-loops added by GATv2Conv get an all-zero feature."""

    def __init__(self, in_dim: int, out_channels: int, heads: int, concat: bool, dropout: float, add_self_loops: bool):
        super().__init__()
        self.conv = GATv2Conv(in_dim, out_channels, heads=heads, concat=concat, dropout=dropout,
                              add_self_loops=add_self_loops, edge_dim=3, fill_value=0.0)

    def forward(self, x: Tensor, edge_index: Tensor, return_attention_weights: bool = False):
        n = x.size(0)
        src, dst, pos = child_positions(edge_index, n)
        ei = torch.stack([src, dst])
        ea = F.one_hot(pos.clamp_max(2), num_classes=3).to(x.dtype)
        if return_attention_weights:
            return self.conv(x, ei, edge_attr=ea, return_attention_weights=True)
        return self.conv(x, ei, edge_attr=ea)


class GATv2NoNormConv(GATv2Conv):
    """GATv2 whose attention coefficients are *not* normalised over a node's neighbours:
    ``alpha_ij = sigmoid(a^T LeakyReLU(W [h_i ; h_j]))`` instead of ``softmax_j(...)``.
    Every neighbour is gated independently in (0, 1), so the aggregated message grows
    with the number of neighbours instead of being a weighted average.  Everything else is GATv2Conv."""

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr, index: Tensor, ptr, dim_size) -> Tensor:
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            x = x + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = torch.sigmoid((x * self.att).sum(dim=-1))
        return F.dropout(alpha, p=self.dropout, training=self.training)


class TreeConv(nn.Module):
    """Binary tree convolution (Bao / Neo): ``h_i' = W [h_i ; h_left(i) ; h_right(i)] + b``.

    ``edge_index`` must hold parent -> child edges (the adapter's stored orientation,
    ``edge_mode="as_is"``); a node's children are ordered by node index, which is the
    plan order the featuriser used.  A missing child contributes zeros; a third or
    later child of an n-ary node is ignored (Bao binarises such nodes).  Information
    flows bottom-up only, so a stack of L layers lets a node see L levels below it.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(3 * in_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n, d = x.shape
        left, right = x.new_zeros(n, d), x.new_zeros(n, d)
        src, dst = edge_index[0], edge_index[1]
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if src.numel():
            order = torch.argsort(src * n + dst)
            src, dst = src[order], dst[order]
            first = torch.ones_like(src, dtype=torch.bool)
            first[1:] = src[1:] != src[:-1]
            idx = torch.arange(src.numel(), device=src.device)
            start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), dim=0).values
            pos = idx - start
            m0, m1 = pos == 0, pos == 1
            left[src[m0]] = x[dst[m0]]
            right[src[m1]] = x[dst[m1]]
        return self.lin(torch.cat([x, left, right], dim=-1))


def _make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class GraphEncoderLayer(nn.Module):
    """One message-passing layer.

    ``out_channels`` is per head for attention convs (``gatv2``/``gat``/
    ``transformer``); the layer output is ``out_channels * heads`` when
    ``concat`` else ``out_channels``.  ``gcn``/``sage``/``none`` ignore
    ``heads`` and output ``out_channels``.  ``none`` is a per-node linear map
    (no neighbourhood aggregation - the "No_GNN" ablation).
    """

    def __init__(self, in_dim: int, out_channels: int, heads: int = 1, concat: bool = True,
                 conv: str = "gatv2", norm: str = "layer", residual: bool = True,
                 activation: str = "elu", dropout: float = 0.0, attn_dropout: float = 0.0,
                 self_loops: bool = True):
        super().__init__()
        self.kind = conv
        self.in_dim = in_dim
        if conv in ("gatv2", "gat", "gatv2_nonorm"):
            cls = {"gatv2": GATv2Conv, "gat": GATConv, "gatv2_nonorm": GATv2NoNormConv}[conv]
            self.conv = cls(in_dim, out_channels, heads=heads, concat=concat,
                            dropout=attn_dropout, add_self_loops=self_loops)
            out_dim = out_channels * heads if concat else out_channels
        elif conv == "transformer":
            self.conv = TransformerConv(in_dim, out_channels, heads=heads, concat=concat,
                                        dropout=attn_dropout)
            out_dim = out_channels * heads if concat else out_channels
        elif conv == "gcn":
            self.conv = GCNConv(in_dim, out_channels, add_self_loops=self_loops)
            out_dim = out_channels
        elif conv == "sage":
            self.conv = SAGEConv(in_dim, out_channels)
            out_dim = out_channels
        elif conv == "gatv2_pos":
            self.conv = PosGATv2Conv(in_dim, out_channels, heads=heads, concat=concat, dropout=attn_dropout,
                                     add_self_loops=self_loops)
            out_dim = out_channels * heads if concat else out_channels
        elif conv == "tcnn":
            self.conv = TreeConv(in_dim, out_channels)
            out_dim = out_channels
        elif conv in ("tcnn_attn", "tcnn_attn_softmax"):
            self.conv = TreeAttnConv(in_dim, out_channels, normalize=(conv == "tcnn_attn_softmax"))
            out_dim = out_channels
        elif conv == "none":
            self.conv = nn.Linear(in_dim, out_channels)
            out_dim = out_channels
        else:
            raise ValueError(f"unknown conv {conv!r}")
        self.out_dim = out_dim
        self.norm = _make_norm(norm, out_dim)
        if residual:
            self.res: Optional[nn.Module] = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        else:
            self.res = None
        self.act = _ACT[activation]
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, return_attention: bool = False
                ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        att = None
        if self.kind == "none":
            h = self.conv(x)
        elif self.kind in ATTENTION_CONVS and return_attention:
            h, att = self.conv(x, edge_index, return_attention_weights=True)
        else:
            h = self.conv(x, edge_index)
        h = self.norm(h)
        if self.res is not None:
            h = h + self.res(x)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h, att


class GraphEncoder(nn.Module):
    """Configurable stack of :class:`GraphEncoderLayer`.

    Layer *i* maps ``d_i -> hidden_dim`` per head (``out_dim`` for the last
    layer); intermediate layers concatenate their heads, the last averages them
    unless ``concat_last``.  With ``heads=[4, 2, 1]``, ``hidden_dim=out_dim=64``
    this is exactly the legacy 3-layer ``GATv2TreeEncoder_V3`` (256 -> 128 -> 64).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, num_layers: int = 3,
                 heads: Union[int, Sequence[int]] = (4, 2, 1), concat_last: bool = False,
                 conv: str = "gatv2", edge_mode: str = "bidirectional", self_loops: bool = True,
                 virtual_node: bool = False, residual: bool = True, norm: str = "layer",
                 activation: str = "elu", dropout: float = 0.0, attn_dropout: float = 0.0,
                 edge_dropout: float = 0.0, jk: str = "last"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        heads_list = [int(heads)] * num_layers if isinstance(heads, int) else [int(h) for h in heads]
        if len(heads_list) != num_layers:
            raise ValueError(f"heads has {len(heads_list)} entries for {num_layers} layers")

        self.in_dim = in_dim
        self.conv_kind = conv
        self.edge_mode = edge_mode
        self.self_loops = self_loops
        self.virtual_node = virtual_node
        self.edge_dropout = edge_dropout
        self.jk = jk

        layers: List[GraphEncoderLayer] = []
        d = in_dim
        for i in range(num_layers):
            last = i == num_layers - 1
            layer = GraphEncoderLayer(
                in_dim=d, out_channels=out_dim if last else hidden_dim, heads=heads_list[i],
                concat=concat_last if last else True, conv=conv, norm=norm, residual=residual,
                activation=activation, dropout=dropout, attn_dropout=attn_dropout,
                self_loops=self_loops)
            layers.append(layer)
            d = layer.out_dim
        self.layers = nn.ModuleList(layers)
        self.layer_dims = [l.out_dim for l in layers]
        self.out_dim = d

        # jumping knowledge: project every non-final layer to out_dim
        self.jk_proj: Optional[nn.ModuleList] = None
        self.jk_out: Optional[nn.Linear] = None
        if jk in ("sum", "max"):
            self.jk_proj = nn.ModuleList([nn.Linear(dim, self.out_dim) for dim in self.layer_dims[:-1]])
        elif jk == "cat":
            self.jk_out = nn.Linear(sum(self.layer_dims), self.out_dim)

        self.vnode_emb: Optional[nn.Parameter] = None
        if virtual_node:
            self.vnode_emb = nn.Parameter(torch.zeros(in_dim))
            nn.init.normal_(self.vnode_emb, std=0.02)

    @classmethod
    def from_config(cls, cfg: GraphEncoderConfig, in_dim: int) -> "GraphEncoder":
        return cls(in_dim=in_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
                   num_layers=cfg.num_layers, heads=cfg.heads_per_layer(),
                   concat_last=cfg.concat_last, conv=cfg.conv, edge_mode=cfg.edge_mode,
                   self_loops=cfg.self_loops, virtual_node=cfg.virtual_node,
                   residual=cfg.residual, norm=cfg.norm, activation=cfg.activation,
                   dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
                   edge_dropout=cfg.edge_dropout, jk=cfg.jk)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def prepare_edges(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Apply ``edge_mode`` (+ explicit self-loops for convs that lack the option)."""
        ei = transform_edges(edge_index, self.edge_mode, num_nodes=num_nodes)
        if self.self_loops and self.conv_kind == "transformer":
            ei = with_self_loops(ei, num_nodes)
        return ei

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None,
                return_attention: bool = False) -> Tuple[Tensor, Dict[str, Any]]:
        num_real = x.size(0)
        ei = self.prepare_edges(edge_index, num_real)
        if self.virtual_node:
            x, ei, _, num_real = add_virtual_nodes(x, ei, batch, self.vnode_emb)
        if self.training and self.edge_dropout > 0:
            ei, _ = dropout_edge(ei, p=self.edge_dropout, force_undirected=False)

        hs: List[Tensor] = []
        atts: List[Optional[Tuple[Tensor, Tensor]]] = []
        h = x
        for layer in self.layers:
            h, att = layer(h, ei, return_attention=return_attention)
            hs.append(h)
            atts.append(att)

        if self.jk == "sum":
            h = hs[-1] + sum(p(hi) for p, hi in zip(self.jk_proj, hs[:-1]))
        elif self.jk == "max":
            stacked = torch.stack([p(hi) for p, hi in zip(self.jk_proj, hs[:-1])] + [hs[-1]], dim=0)
            h = stacked.max(dim=0).values
        elif self.jk == "cat":
            h = self.jk_out(torch.cat(hs, dim=-1))

        aux: Dict[str, Any] = {"edge_index": ei}
        if self.virtual_node:
            aux["virtual"] = h[num_real:]
            h = h[:num_real]
        if return_attention:
            aux["attention"] = atts
        return h, aux
