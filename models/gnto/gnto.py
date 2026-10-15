"""The GNTO model: node encoder -> graph encoder -> readout -> prediction head.

Build from a config (preferred)::

    from models.gnto import GNTO, get_preset
    model = GNTO.from_config(get_preset("gnto", "graph.num_layers=5",
                                         "graph.edge_mode=bidirectional"))
    y = model(batch)                       # [B, 1]
    y, aux = model(batch, return_aux=True) # + node/graph embeddings, attention

or assemble the four components by hand and pass them to ``GNTO(...)``.

Checkpoints written by :meth:`GNTO.save` carry the config, so
:meth:`GNTO.load` rebuilds the architecture without any other information.
Checkpoints saved by the pre-rewrite classes load through the ordinary
:meth:`GNTO.load_state_dict` (keys are converted automatically) or, with a
conversion report, :meth:`GNTO.from_legacy_checkpoint`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .config import GNTOConfig, PRESETS, get_preset
from .graph_encoder import GraphEncoder
from .head import PredictionHead
from .legacy_compat import convert_legacy_state_dict, is_legacy_state_dict
from .node_encoder import NodeEncoder, build_node_encoder
from .readout import Readout

CHECKPOINT_FORMAT = "gnto-modular-1"


class GNTO(nn.Module):
    def __init__(self, node_encoder: NodeEncoder, graph_encoder: GraphEncoder, readout: Readout,
                 head: PredictionHead, config: Optional[GNTOConfig] = None):
        super().__init__()
        if node_encoder.out_dim != graph_encoder.in_dim:
            raise ValueError(f"node_encoder.out_dim={node_encoder.out_dim} != graph_encoder.in_dim={graph_encoder.in_dim}")
        if graph_encoder.out_dim != readout.in_dim:
            raise ValueError(f"graph_encoder.out_dim={graph_encoder.out_dim} != readout.in_dim={readout.in_dim}")
        if readout.out_dim != head.in_dim:
            raise ValueError(f"readout.out_dim={readout.out_dim} != head.in_dim={head.in_dim}")
        self.node_encoder = node_encoder
        self.graph_encoder = graph_encoder
        self.readout = readout
        self.head = head
        self.config = config

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Union[GNTOConfig, Dict[str, Any], str], *overrides: str) -> "GNTO":
        """Build from a :class:`GNTOConfig`, a config dict, or a preset name."""
        if isinstance(cfg, str):
            cfg = get_preset(cfg, *overrides)
        elif isinstance(cfg, dict):
            cfg = GNTOConfig.from_dict(cfg)
            if overrides:
                cfg = cfg.override(*overrides)
        elif overrides:
            cfg = cfg.override(*overrides)
        node = build_node_encoder(cfg.node)
        graph = GraphEncoder.from_config(cfg.graph, in_dim=node.out_dim)
        readout = Readout.from_config(cfg.readout, in_dim=graph.out_dim)
        head = PredictionHead.from_config(cfg.head, in_dim=readout.out_dim)
        return cls(node, graph, readout, head, config=cfg)

    # ---------------------------------------------------------------- forward
    @staticmethod
    def _unpack(data: Any, edge_index: Optional[Tensor], batch: Optional[Tensor]
                ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            return data.x, data.edge_index, getattr(data, "batch", None)
        if edge_index is None:
            raise ValueError("pass a PyG Data/Batch, or (x, edge_index[, batch])")
        return data, edge_index, batch

    def forward(self, data: Any, edge_index: Optional[Tensor] = None, batch: Optional[Tensor] = None,
                return_aux: bool = False) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        x, ei, b = self._unpack(data, edge_index, batch)
        h0 = self.node_encoder(x)
        h, aux = self.graph_encoder(h0, ei, b, return_attention=return_aux)
        if return_aux:
            z, w = self.readout(h, b, aux, return_weights=True)
        else:
            z = self.readout(h, b, aux)
        y = self.head(z)
        if return_aux:
            aux.update(node_input=h0, node_embeddings=h, graph_embedding=z, readout_weights=w)
            return y, aux
        return y

    def encode(self, data: Any, edge_index: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        """Plan embedding ``z_G`` (``[B, d]``) without the prediction head."""
        x, ei, b = self._unpack(data, edge_index, batch)
        h, aux = self.graph_encoder(self.node_encoder(x), ei, b)
        return self.readout(h, b, aux)

    @torch.no_grad()
    def predict(self, data: Any, edge_index: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        """Inference helper: eval mode, no grad, output flattened to ``[B]`` when out_dim == 1."""
        was_training = self.training
        self.eval()
        try:
            y = self.forward(data, edge_index, batch)
        finally:
            self.train(was_training)
        return y.view(-1) if y.size(-1) == 1 else y

    # ------------------------------------------------------------- utilities
    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> Dict[str, int]:
        return {name: sum(p.numel() for p in module.parameters())
                for name, module in (("node_encoder", self.node_encoder), ("graph_encoder", self.graph_encoder),
                                     ("readout", self.readout), ("head", self.head))}

    def summary(self) -> str:
        lines = [self.config.describe() if self.config is not None else "GNTO (no config attached)"]
        for name, count in self.parameter_breakdown().items():
            lines.append(f"  {name:14s} {count:>10,d}")
        lines.append(f"  {'total':14s} {self.num_parameters():>10,d}")
        return "\n".join(lines)

    # ------------------------------------------------------------ checkpoint
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False,
                        convert_legacy: bool = True):
        """``nn.Module.load_state_dict`` that also accepts the pre-rewrite key layout
        (``gnn.gat1.*``, ``prediction_head.*`` ...) and converts it on the fly, so
        checkpoints saved by the old classes load with no extra code."""
        if convert_legacy and is_legacy_state_dict(state_dict):
            state_dict, _ = convert_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def save(self, path: Union[str, Path], extra: Optional[Dict[str, Any]] = None) -> None:
        if self.config is None:
            raise RuntimeError("save() needs the model to carry its config (build with from_config)")
        payload = {"format": CHECKPOINT_FORMAT, "config": self.config.to_dict(),
                   "state_dict": self.state_dict()}
        if extra:
            payload["extra"] = extra
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu",
             strict: bool = True, *overrides: str) -> "GNTO":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not (isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT):
            raise ValueError(
                f"{path} is not a GNTO.save() checkpoint. For a bare legacy state_dict use "
                "GNTO.from_legacy_checkpoint(path, config).")
        model = cls.from_config(payload["config"], *overrides)
        model.load_state_dict(payload["state_dict"], strict=strict)
        model.checkpoint_extra = payload.get("extra")
        return model

    def load_legacy_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = False) -> Dict[str, Any]:
        """Load a state_dict written by the pre-rewrite classes (``GNTO_QF_Model``,
        ``GNTO_DACE_Model``, ``GATv2TreeEncoder_V3/V4`` + ``PredictionHead_V2``...).
        Returns the conversion report (dropped / unmapped / missing keys)."""
        new_sd, report = convert_legacy_state_dict(state_dict)
        result = self.load_state_dict(new_sd, strict=strict, convert_legacy=False)
        report["missing_keys"] = list(result.missing_keys)
        report["unexpected_keys"] = list(result.unexpected_keys)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(f"legacy checkpoint does not match the model: {report}")
        return report

    @classmethod
    def from_legacy_checkpoint(cls, path: Union[str, Path], config: Union[GNTOConfig, str],
                               map_location: Union[str, torch.device] = "cpu",
                               strict: bool = False) -> Tuple["GNTO", Dict[str, Any]]:
        model = cls.from_config(config)
        sd = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd and "format" in sd:
            raise ValueError("this is a GNTO.save() checkpoint; use GNTO.load()")
        report = model.load_legacy_state_dict(sd, strict=strict)
        return model, report


def build_model(cfg: Union[GNTOConfig, Dict[str, Any], str], *overrides: str) -> GNTO:
    """Alias of :meth:`GNTO.from_config` (``build_model("qf_v3", "graph.num_layers=4")``)."""
    return GNTO.from_config(cfg, *overrides)


__all__ = ["GNTO", "build_model", "CHECKPOINT_FORMAT", "PRESETS"]
