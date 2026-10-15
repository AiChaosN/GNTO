"""Bridges between checkpoints written by the pre-rewrite classes and the modular GNTO.

The pre-rewrite classes (``NodeEncoder_QF*``, ``GATv2TreeEncoder_V3/V4``,
``PredictionHead_V2`` ...) are not part of this repository; nothing here
imports them.

* :func:`is_legacy_state_dict` detects the old key layout.
* :func:`convert_legacy_state_dict` renames keys from that layout
  (``gnn.gat1.*``, ``prediction_head.*``, ``node_encoder.0.*`` ...) to the
  modular one (``graph_encoder.layers.0.conv.*`` ...).  ``GNTO.load_state_dict``
  calls it automatically, so ``model.load_state_dict(torch.load(old_ckpt))``
  keeps working.
* :func:`legacy_qf_config` / :func:`legacy_dace_config` produce the exact
  :class:`GNTOConfig` of the pre-rewrite ``GNTO_QF_Model`` / ``GNTO_DACE_Model``
  assemblies, so their checkpoints evaluate identically under the new code.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .config import (GNTOConfig, GraphEncoderConfig, HeadConfig, NodeEncoderConfig,
                           QF_IMDB_VOCAB, ReadoutConfig)

_TREE_PREFIXES = r"(?:gnn|tree_encoder|tree_enc)"
_NODE_PREFIXES = r"(?:node_encoder|node_enc)"
_HEAD_PREFIXES = r"(?:head|prediction_head)"

_RULES = [
    # graph encoder layers (1-based in legacy)
    (re.compile(rf"^{_TREE_PREFIXES}\.gat(\d+)\.(.+)$"), lambda m: f"graph_encoder.layers.{int(m.group(1)) - 1}.conv.{m.group(2)}"),
    (re.compile(rf"^{_TREE_PREFIXES}\.norm(\d+)\.(.+)$"), lambda m: f"graph_encoder.layers.{int(m.group(1)) - 1}.norm.{m.group(2)}"),
    (re.compile(rf"^{_TREE_PREFIXES}\.proj_res(\d+)\.(.+)$"), lambda m: f"graph_encoder.layers.{int(m.group(1)) - 1}.res.{m.group(2)}"),
    (re.compile(rf"^{_TREE_PREFIXES}\.jk_p(\d+)\.(.+)$"), lambda m: f"graph_encoder.jk_proj.{int(m.group(1)) - 1}.{m.group(2)}"),
    # legacy V4 registers the gate twice (gate_nn.0.* and pool.gate_nn.0.*)
    (re.compile(rf"^{_TREE_PREFIXES}\.(?:pool\.)?gate_nn\.0\.(.+)$"), lambda m: f"readout.gate_nn.{m.group(1)}"),
    # MLP node encoder written as a bare nn.Sequential (DACE adapter)
    (re.compile(rf"^{_NODE_PREFIXES}\.(\d+)\.(.+)$"), lambda m: f"node_encoder.net.{m.group(1)}.{m.group(2)}"),
    (re.compile(rf"^{_NODE_PREFIXES}\.(.+)$"), lambda m: f"node_encoder.{m.group(1)}"),
    (re.compile(rf"^{_HEAD_PREFIXES}\.(.+)$"), lambda m: f"head.{m.group(1)}"),
]

# parameters that existed in the legacy classes but were never used in forward()
_DROP = re.compile(rf"^{_NODE_PREFIXES}\.(linearType|linearJoin)\.")
_LEGACY_PREFIXES = ("gnn.", "tree_encoder.", "tree_enc.", "prediction_head.", "node_enc.")
_LEGACY_MLP = re.compile(r"^node_encoder\.\d+\.")


def is_legacy_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """True if any key uses the pre-rewrite layout."""
    return any(k.startswith(_LEGACY_PREFIXES) or _DROP.match(k) or _LEGACY_MLP.match(k)
               for k in state_dict)


def convert_legacy_state_dict(state_dict: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Return ``(converted_state_dict, report)``.

    ``report`` lists ``dropped`` (known dead parameters), ``unmapped`` (keys no
    rule recognised - left unchanged) and the number of renamed keys.  Keys
    already in the modular layout pass through untouched.
    """
    out: Dict[str, Tensor] = {}
    report: Dict[str, Any] = {"dropped": [], "unmapped": [], "renamed": 0, "passthrough": 0}
    for key, value in state_dict.items():
        if _DROP.match(key):
            report["dropped"].append(key)
            continue
        if key.startswith(("graph_encoder.", "readout.", "head.", "node_encoder.net.")):
            out[key] = value
            report["passthrough"] += 1
            continue
        new_key: Optional[str] = None
        for pattern, fn in _RULES:
            m = pattern.match(key)
            if m:
                new_key = fn(m)
                break
        if new_key is None:
            report["unmapped"].append(key)
            out[key] = value
            continue
        if new_key in out and not torch.equal(out[new_key], value):
            raise ValueError(f"legacy keys collide on {new_key!r} with different values")
        if new_key != key:
            report["renamed"] += 1
        out[new_key] = value
    return out, report


# --------------------------------------------------------------------------- #
# Config mirrors of the legacy model assemblies
# --------------------------------------------------------------------------- #
def _vocab(encoding_or_vocab: Any) -> Dict[str, int]:
    if encoding_or_vocab is None:
        return dict(QF_IMDB_VOCAB)
    if isinstance(encoding_or_vocab, dict):
        return dict(encoding_or_vocab)
    from .node_encoder import qf_vocab_from_encoding
    return qf_vocab_from_encoding(encoding_or_vocab)


def legacy_qf_config(encoding_or_vocab: Any = None, add_plan_rows: bool = True, hidden_dim: int = 64,
                     attention_pool: bool = False, use_hist: bool = True, use_sample: bool = True,
                     name: Optional[str] = None, join_encoding: str = "id") -> GNTOConfig:
    """Mirror of ``adapters.qf_adapter.GNTO_QF_Model(encoding, hidden_dim, add_plan_rows)``
    (``attention_pool=True`` mirrors the ``GATv2TreeEncoder_V4`` attention-readout variant).
    Edges are consumed exactly as the QueryFormer adapter stores them."""
    return GNTOConfig(
        name=name or ("legacy_qf_v4" if attention_pool else "legacy_qf_v3"),
        notes="config mirror of the legacy GNTO_QF_Model assembly",
        node=NodeEncoderConfig(kind="queryformer", out_dim=64, embed_size=64, vocab=_vocab(encoding_or_vocab),
                               use_hist=use_hist, use_sample=use_sample, bin_number=50, plan_rows=add_plan_rows,
                               join_encoding=join_encoding),
        graph=GraphEncoderConfig(conv="gatv2", num_layers=3, hidden_dim=hidden_dim, out_dim=hidden_dim,
                                 heads=[4, 2, 1], edge_mode="as_is", self_loops=True, residual=True,
                                 norm="layer", activation="elu", dropout=0.0, attn_dropout=0.0, jk="last"),
        readout=ReadoutConfig(pooling="attention" if attention_pool else "mean", gate_hidden=0),
        head=HeadConfig(kind="resmlp", hidden_dim=64, num_blocks=2, dropout=0.0, out_dim=1,
                        out_activation="sigmoid"),
    )


def legacy_dace_config(input_dim: int, hidden_dim: int = 64) -> GNTOConfig:
    """Mirror of ``adapters.dace_adapter.GNTO_DACE_Model(input_dim, hidden_dim)``."""
    return GNTOConfig(
        name="legacy_dace",
        notes="config mirror of the legacy GNTO_DACE_Model assembly",
        node=NodeEncoderConfig(kind="mlp", in_dim=input_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                               num_layers=2, norm="layer", activation="relu", dropout=0.0),
        graph=GraphEncoderConfig(conv="gatv2", num_layers=3, hidden_dim=hidden_dim, out_dim=hidden_dim,
                                 heads=[4, 4, 1], edge_mode="as_is", self_loops=True, residual=True,
                                 norm="layer", activation="elu", dropout=0.1, attn_dropout=0.1, jk="last"),
        readout=ReadoutConfig(pooling="mean"),
        head=HeadConfig(kind="resmlp", hidden_dim=hidden_dim, num_blocks=2, dropout=0.1, out_dim=1,
                        out_activation="sigmoid"),
    )


__all__ = ["is_legacy_state_dict", "convert_legacy_state_dict", "legacy_qf_config", "legacy_dace_config"]
