"""Node encoders: raw per-operator features -> initial node embedding ``h^(0)``.

Three families, all exposing ``out_dim`` and ``forward(x) -> [N, out_dim]``:

* :class:`MLPNodeEncoder`        dense pre-vectorised features (DACE style).
* :class:`PlanNodeEncoder`       heterogeneous flat layout with a DeepSets
                                 predicate encoder.
* :class:`QueryFormerNodeEncoder` port of QueryFormer's ``FeatureEmbed`` with
                                 an optional ``Plan Rows`` channel.  Parameter
                                 names match the legacy ``NodeEncoder_QF*``
                                 classes so old checkpoints load unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import NodeEncoderConfig

_ACT = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


def _norm_layer(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class NodeEncoder(nn.Module):
    """Base class; subclasses set ``self.out_dim`` and ``self.in_dim``."""

    in_dim: Optional[int] = None
    out_dim: int = 0

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def check_input(self, x: Tensor) -> None:
        if self.in_dim is not None and x.size(-1) != self.in_dim:
            raise ValueError(
                f"{type(self).__name__} expects {self.in_dim} input features, got {x.size(-1)}")


# --------------------------------------------------------------------------- #
class MLPNodeEncoder(NodeEncoder):
    """``Linear -> act [-> norm]`` stack over dense features.

    With ``num_layers=2, norm='layer', activation='relu', dropout=0`` the module
    list is ``[Linear, ReLU, LayerNorm, Linear, ReLU]`` - the exact structure of
    the inline encoder in ``adapters.dace_adapter.GNTO_DACE_Model``.
    """

    def __init__(self, in_dim: int, out_dim: int = 64, hidden_dim: int = 64,
                 num_layers: int = 2, norm: str = "layer", activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.in_dim, self.out_dim = in_dim, out_dim
        layers = []
        d = in_dim
        for i in range(num_layers):
            last = i == num_layers - 1
            d_out = out_dim if last else hidden_dim
            layers.append(nn.Linear(d, d_out))
            layers.append(_ACT[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if not last:
                layers.append(_norm_layer(norm, d_out))
            d = d_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        self.check_input(x)
        return self.net(x.float())


# --------------------------------------------------------------------------- #
class PredicateSetEncoder(nn.Module):
    """DeepSets over up to ``slots`` predicates ``(col, op, val, is_col)`` per node.

    ``val`` is a column id when ``is_col == 1`` (join predicate) and a
    normalised numeric literal otherwise.  Empty slots (all-zero) are masked.
    ``pool='sum'`` keeps the "total filtering mass"; ``'mean'``
    reproduces the pre-V4 behaviour.
    """

    def __init__(self, num_cols: int, num_ops: int, col_dim: int = 16, op_dim: int = 8,
                 out_dim: int = 32, pool: str = "sum", dropout: float = 0.0):
        super().__init__()
        if pool not in ("sum", "mean"):
            raise ValueError("pool must be 'sum' or 'mean'")
        self.pool = pool
        self.col_emb = nn.Embedding(num_cols, col_dim)
        self.op_emb = nn.Embedding(num_ops, op_dim)
        self.val_proj = nn.Linear(1, col_dim)
        in_dim = col_dim + op_dim + col_dim + 1
        self.phi = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(out_dim, out_dim))
        self.rho = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, preds: Tensor) -> Tensor:
        """``preds``: ``[N, S, 4]`` -> ``[N, out_dim]``."""
        col = preds[..., 0].round().long().clamp_min(0)
        op = preds[..., 1].round().long().clamp_min(0)
        val = preds[..., 2].float()
        is_col = preds[..., 3].float().clamp(0, 1).unsqueeze(-1)              # [N,S,1]
        present = (preds.abs().sum(dim=-1) > 0).float().unsqueeze(-1)        # [N,S,1]

        col_e = self.col_emb(col.clamp_max(self.col_emb.num_embeddings - 1))
        op_e = self.op_emb(op.clamp_max(self.op_emb.num_embeddings - 1))
        rhs_col = self.col_emb(val.round().long().clamp(0, self.col_emb.num_embeddings - 1))
        rhs_num = self.val_proj(val.unsqueeze(-1))
        rhs = is_col * rhs_col + (1.0 - is_col) * rhs_num
        z = torch.cat([col_e, op_e, rhs, is_col], dim=-1)
        z = self.phi(z) * present                                             # [N,S,out]
        if self.pool == "sum":
            agg = z.sum(dim=1)
        else:
            agg = z.sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
        return self.rho(agg)


class PlanNodeEncoder(NodeEncoder):
    """Heterogeneous flat layout ``[type_id | numeric*K | (col, op, val, is_col)*S]``.

    * type_id           -> embedding
    * numeric (rows, width, cost, ...) -> optional log1p -> BatchNorm -> Linear
    * predicates        -> :class:`PredicateSetEncoder`
    * fusion MLP        -> ``out_dim`` (+ LayerNorm)
    """

    def __init__(self, num_node_types: int, num_cols: int, num_ops: int, out_dim: int = 64,
                 type_dim: int = 16, num_numeric: int = 2, numeric_log1p: bool = True,
                 numeric_norm: str = "batch", numeric_dim: int = 8, pred_slots: int = 3,
                 pred_col_dim: int = 16, pred_op_dim: int = 8, pred_dim: int = 32,
                 pred_pool: str = "sum", hidden_dim: int = 64, dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        self.num_numeric, self.pred_slots = num_numeric, pred_slots
        self.in_dim = 1 + num_numeric + 4 * pred_slots
        self.out_dim = out_dim
        self.numeric_log1p = numeric_log1p

        self.type_emb = nn.Embedding(num_node_types, type_dim)
        self.numeric_norm = nn.BatchNorm1d(num_numeric) if (numeric_norm == "batch" and num_numeric > 0) else nn.Identity()
        self.numeric_proj = nn.Linear(num_numeric, numeric_dim) if num_numeric > 0 else None
        self.pred_enc = PredicateSetEncoder(num_cols, num_ops, pred_col_dim, pred_op_dim,
                                            pred_dim, pred_pool, dropout) if pred_slots > 0 else None
        fuse_in = type_dim + (numeric_dim if num_numeric > 0 else 0) + (pred_dim if pred_slots > 0 else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim), _ACT[activation](), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        self.check_input(x)
        n = x.size(0)
        parts = [self.type_emb(x[:, 0].round().long().clamp(0, self.type_emb.num_embeddings - 1))]
        if self.num_numeric > 0:
            num = x[:, 1:1 + self.num_numeric].float()
            if self.numeric_log1p:
                num = torch.log1p(num.clamp_min(0))
            parts.append(F.relu(self.numeric_proj(self.numeric_norm(num))))
        if self.pred_slots > 0:
            preds = x[:, 1 + self.num_numeric:].reshape(n, self.pred_slots, 4)
            parts.append(self.pred_enc(preds))
        return self.fuse(torch.cat(parts, dim=-1))


# --------------------------------------------------------------------------- #
def qf_vocab_from_encoding(encoding: Any) -> Dict[str, int]:
    """Vocabulary sizes from a QueryFormer ``Encoding`` object."""
    return {
        "types": len(encoding.idx2type),
        "tables": len(encoding.idx2table),
        "joins": len(encoding.idx2join),
        "columns": len(encoding.idx2col),
        "ops": len(encoding.idx2op),
    }


class QueryFormerNodeEncoder(NodeEncoder):
    """QueryFormer ``FeatureEmbed`` port.

    Input row layout (``bin_number=50``): ``type(1) | join(1) | filters(9) |
    filter_mask(3) | hists(150) | table_id(1) | table_sample(1000)`` = 1165,
    plus one trailing ``log1p(Plan Rows)`` column when ``plan_rows=True``, plus
    two trailing join-column ids ``(col_left, col_right)`` when
    ``join_encoding != "id"`` (in that order).

    ``join_encoding``:
    * ``"id"``       ``joinEmbed[join_id]`` - one free vector per join *string*
                     (upstream QueryFormer; a pair never seen in training has none).
    * ``"columns"``  ``leaky_relu(joinColProj(columnEmbed[col_left] + columnEmbed[col_right]))``
                     - composed from the column embeddings the filter encoder
                     already learns; symmetric in the two sides; any pair of known
                     columns is representable.  ``joinEmbed`` is not allocated.
    * ``"both"``     sum of the two.

    Parameter names are identical to the legacy ``NodeEncoder_QF`` /
    ``NodeEncoder_QF_AddPlanrows`` (the two never-used ``linearType`` and
    ``linearJoin`` layers are dropped; ``convert_legacy_state_dict`` discards
    them when loading old checkpoints).
    """

    def __init__(self, embed_size: int = 64, types: int = 20, tables: int = 20, joins: int = 10,
                 columns: int = 30, ops: int = 10, use_hist: bool = True, use_sample: bool = True,
                 bin_number: int = 50, plan_rows: bool = False, join_encoding: str = "id"):
        super().__init__()
        if join_encoding not in ("id", "columns", "both"):
            raise ValueError(f"join_encoding must be 'id', 'columns' or 'both', got {join_encoding!r}")
        self.embed_size, self.bin_number = embed_size, bin_number
        self.use_hist, self.use_sample, self.plan_rows = use_hist, use_sample, plan_rows
        self.join_encoding = join_encoding
        self.join_cols = join_encoding != "id"
        self.base_dim = 1 + 1 + 9 + 3 + 3 * bin_number + 1001
        self.in_dim = self.base_dim + (1 if plan_rows else 0) + (2 if self.join_cols else 0)
        self.out_dim = embed_size

        filt_dim = embed_size + embed_size // 8 + 1
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)
        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size // 8)
        self.joinEmbed = nn.Embedding(joins, embed_size) if join_encoding in ("id", "both") else None
        self.joinColProj = nn.Linear(embed_size, embed_size) if self.join_cols else None
        self.linearFilter = nn.Linear(filt_dim, filt_dim)
        self.linearFilter2 = nn.Linear(filt_dim, filt_dim)
        # only allocate the channels that are actually used
        self.linearSample = nn.Linear(1000, embed_size) if use_sample else None
        self.linearHist = nn.Linear(bin_number, embed_size) if use_hist else None
        n_parts = 4 + (1 if use_hist else 0)
        self.project = nn.Linear(embed_size * n_parts + embed_size // 8 + 1, embed_size)
        self.rows_proj = nn.Linear(1, embed_size) if plan_rows else None

    @classmethod
    def from_encoding(cls, encoding: Any, **kwargs: Any) -> "QueryFormerNodeEncoder":
        return cls(**qf_vocab_from_encoding(encoding), **kwargs)

    # -- pieces --------------------------------------------------------------
    def _filters(self, filters_id: Tensor, mask: Tensor) -> Tensor:
        expand = filters_id.view(-1, 3, 3).transpose(1, 2)          # [N,3,(col,op,val)]
        col = self.columnEmbed(expand[:, :, 0].long())
        op = self.opEmbed(expand[:, :, 1].long())
        val = expand[:, :, 2].unsqueeze(-1)
        z = torch.cat([col, op, val], dim=-1)
        z = F.leaky_relu(self.linearFilter(z))
        z = F.leaky_relu(self.linearFilter2(z))
        z = z * mask.unsqueeze(-1)
        return z.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _hist(self, hists: Tensor, mask: Tensor) -> Tensor:
        expand = hists.view(-1, self.bin_number, 3).transpose(1, 2)  # [N,3,bins]
        z = self.linearHist(expand) * mask.unsqueeze(-1)
        return z.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _join(self, join_id: Tensor, join_cols: Optional[Tensor]) -> Tensor:
        parts = []
        if self.joinEmbed is not None:
            parts.append(self.joinEmbed(join_id.long()).squeeze(1))
        if self.join_cols:
            cols = join_cols.long().clamp(0, self.columnEmbed.num_embeddings - 1)   # [N,2]
            z = self.columnEmbed(cols[:, 0]) + self.columnEmbed(cols[:, 1])          # symmetric
            parts.append(F.leaky_relu(self.joinColProj(z)))
        return parts[0] if len(parts) == 1 else parts[0] + parts[1]

    def _table(self, table_sample: Tensor) -> Tensor:
        table, sample = torch.split(table_sample, (1, 1000), dim=-1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        if self.use_sample:
            emb = emb + self.linearSample(sample)
        return emb

    def forward(self, x: Tensor) -> Tensor:
        self.check_input(x)
        x = x.float()
        rows = join_cols = None
        tail = [(1 if self.plan_rows else 0), (2 if self.join_cols else 0)]
        if any(tail):
            x, extra = torch.split(x, (self.base_dim, sum(tail)), dim=-1)
            if self.plan_rows:
                rows, extra = extra[:, :1], extra[:, 1:]
            if self.join_cols:
                join_cols = extra[:, :2]
        type_id, join_id, filters_id, mask, hists, table_sample = torch.split(
            x, (1, 1, 9, 3, 3 * self.bin_number, 1001), dim=-1)
        parts = [
            self.typeEmbed(type_id.long()).squeeze(1),
            self._filters(filters_id, mask),
            self._join(join_id, join_cols),
            self._table(table_sample),
        ]
        if self.use_hist:
            parts.append(self._hist(hists, mask))
        h = F.leaky_relu(self.project(torch.cat(parts, dim=1)))
        if rows is not None:
            h = h + F.relu(self.rows_proj(rows))
        return h


# --------------------------------------------------------------------------- #
def build_node_encoder(cfg: NodeEncoderConfig) -> NodeEncoder:
    if cfg.kind == "mlp":
        return MLPNodeEncoder(in_dim=cfg.in_dim, out_dim=cfg.out_dim, hidden_dim=cfg.hidden_dim,
                              num_layers=cfg.num_layers, norm=cfg.norm,
                              activation=cfg.activation, dropout=cfg.dropout)
    if cfg.kind == "plan":
        return PlanNodeEncoder(
            num_node_types=cfg.num_node_types, num_cols=cfg.num_cols, num_ops=cfg.num_ops,
            out_dim=cfg.out_dim, type_dim=cfg.type_dim, num_numeric=cfg.num_numeric,
            numeric_log1p=cfg.numeric_log1p, numeric_norm=cfg.numeric_norm,
            numeric_dim=cfg.numeric_dim, pred_slots=cfg.pred_slots,
            pred_col_dim=cfg.pred_col_dim, pred_op_dim=cfg.pred_op_dim, pred_dim=cfg.pred_dim,
            pred_pool=cfg.pred_pool, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout,
            activation=cfg.activation)
    if cfg.kind == "queryformer":
        v = cfg.vocab
        return QueryFormerNodeEncoder(
            embed_size=cfg.embed_size, types=v["types"], tables=v["tables"], joins=v["joins"],
            columns=v["columns"], ops=v["ops"], use_hist=cfg.use_hist,
            use_sample=cfg.use_sample, bin_number=cfg.bin_number, plan_rows=cfg.plan_rows,
            join_encoding=cfg.join_encoding)
    raise ValueError(f"unknown node encoder kind {cfg.kind!r}")
