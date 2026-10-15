"""Configuration dataclasses for the modular GNTO model.

One :class:`GNTOConfig` fully describes an architecture, so an experiment is
reproducible from a single JSON file::

    cfg = get_preset("gnto")
    cfg = cfg.override("graph.num_layers=5", "graph.edge_mode=bidirectional")
    model = GNTO.from_config(cfg)
    cfg.save("results/my_run/config.json")

Every knob that used to be hard-coded in a ``*_V3`` / ``*_V4`` class is a
field here.  Fields are grouped per component (node encoder, graph encoder,
readout, prediction head); fields that do not apply to the selected ``kind``
are ignored by the builders, which keeps CLI overrides trivial.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# --------------------------------------------------------------------------- #
# Allowed values (validated in __post_init__ so a typo fails at config time,
# not three hours into a training run).
# --------------------------------------------------------------------------- #
NODE_ENCODER_KINDS = ("mlp", "plan", "queryformer")
JOIN_ENCODINGS = ("id", "columns", "both")
CONV_TYPES = ("gatv2", "gat", "gatv2_nonorm", "gatv2_pos", "gcn", "sage", "transformer", "tcnn", "tcnn_attn", "tcnn_attn_softmax", "none")
EDGE_MODES = ("as_is", "reverse", "bidirectional")
NORM_TYPES = ("layer", "batch", "none")
ACTIVATIONS = ("elu", "relu", "gelu", "leaky_relu", "silu")
JK_MODES = ("last", "sum", "max", "cat")
POOLINGS = ("mean", "max", "sum", "attention", "first", "virtual")
HEAD_KINDS = ("resmlp", "mlp")
OUT_ACTIVATIONS = ("none", "sigmoid", "softplus")


def _check(value: Any, allowed: tuple, name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r}; expected one of {allowed}")


# --------------------------------------------------------------------------- #
# Component configs
# --------------------------------------------------------------------------- #
@dataclass
class NodeEncoderConfig:
    """How raw per-operator features become the initial node embedding h^(0).

    ``kind`` selects the encoder family; only the matching field group is used.

    * ``"mlp"``         dense, pre-vectorised features (DACE one-hot + scalars).
    * ``"plan"``        heterogeneous flat layout
                        ``[type_id | numeric*K | (col, op, val, is_col)*S]``
                        with a DeepSets predicate encoder.
    * ``"queryformer"`` QueryFormer ``FeatureEmbed`` layout (1165 dims, plus one
                        trailing ``log1p(Plan Rows)`` column when ``plan_rows``, plus two
                        trailing join-column ids when ``join_encoding != "id"``).

    ``join_encoding`` (queryformer only) - how a join predicate becomes a vector:
    ``"id"`` looks the whole normalised string (``'mc.movie_id = t.id'``) up in a
    16-entry vocabulary (QueryFormer's design; unseen pairs get no representation);
    ``"columns"`` composes it from the two column embeddings shared with the filter
    encoder; ``"both"`` sums the two.
    """
    kind: str = "queryformer"
    out_dim: int = 64
    dropout: float = 0.0
    activation: str = "relu"

    # --- kind == "mlp" -------------------------------------------------------
    in_dim: Optional[int] = None
    hidden_dim: int = 64
    num_layers: int = 2
    norm: str = "layer"

    # --- kind == "plan" ------------------------------------------------------
    num_node_types: int = 32
    num_cols: int = 64
    num_ops: int = 8
    type_dim: int = 16
    num_numeric: int = 2
    numeric_log1p: bool = True
    numeric_norm: str = "batch"       # "batch" | "none"
    numeric_dim: int = 8
    pred_slots: int = 3
    pred_col_dim: int = 16
    pred_op_dim: int = 8
    pred_dim: int = 32
    pred_pool: str = "sum"            # "sum" (DeepSets) | "mean"

    # --- kind == "queryformer" ----------------------------------------------
    embed_size: int = 64
    vocab: Dict[str, int] = field(default_factory=lambda: {
        "types": 20, "tables": 20, "joins": 10, "columns": 30, "ops": 10})
    use_hist: bool = True
    use_sample: bool = True
    bin_number: int = 50
    plan_rows: bool = True
    join_encoding: str = "id"         # "id" | "columns" | "both"; the dataclass default stays "id" so configs saved
                                      # before the field existed still load; every preset uses "columns"

    def __post_init__(self) -> None:
        _check(self.kind, NODE_ENCODER_KINDS, "node.kind")
        _check(self.join_encoding, JOIN_ENCODINGS, "node.join_encoding")
        _check(self.norm, NORM_TYPES, "node.norm")
        _check(self.activation, ACTIVATIONS, "node.activation")
        _check(self.pred_pool, ("sum", "mean"), "node.pred_pool")
        _check(self.numeric_norm, ("batch", "none"), "node.numeric_norm")
        if self.kind == "mlp" and self.in_dim is None:
            raise ValueError("node.in_dim is required when node.kind == 'mlp'")
        for k in ("types", "tables", "joins", "columns", "ops"):
            if self.kind == "queryformer" and k not in self.vocab:
                raise ValueError(f"node.vocab is missing key {k!r}")

    @property
    def qf_input_dim(self) -> int:
        """Expected feature width for kind == 'queryformer'."""
        base = 1 + 1 + 9 + 3 + 3 * self.bin_number + 1001
        return base + (1 if self.plan_rows else 0) + (2 if self.join_encoding != "id" else 0)

    @property
    def qf_join_cols(self) -> bool:
        """Whether the data must carry the two trailing join-column ids."""
        return self.kind == "queryformer" and self.join_encoding != "id"

    @property
    def plan_input_dim(self) -> int:
        """Expected feature width for kind == 'plan'."""
        return 1 + self.num_numeric + 4 * self.pred_slots


@dataclass
class GraphEncoderConfig:
    """Message passing over the plan DAG.

    * ``num_layers`` / ``heads`` — depth and per-layer attention heads.  ``heads``
      may be an int (same for every layer) or a list with exactly ``num_layers``
      entries.  All layers but the last concatenate heads; the last averages
      them unless ``concat_last``.
    * ``edge_mode`` — how the *stored* ``edge_index`` is used:
      ``"as_is"`` (whatever the data pipeline wrote, parent→child for the
      QueryFormer and DataPreprocessor builders), ``"reverse"`` (flip it) or
      ``"bidirectional"`` (both directions, de-duplicated).
    * ``self_loops`` — add ``(i, i)`` edges: inside the convolution for
      ``gatv2``/``gat``/``gcn``, explicitly for ``transformer``; ``sage`` (root
      weight) and ``none`` ignore it.
    * ``virtual_node`` — add one learnable node per graph connected to every
      operator, giving a 1-hop global receptive field.
    * ``jk`` — jumping-knowledge combination of layer outputs.
    """
    conv: str = "gatv2"
    num_layers: int = 3
    hidden_dim: int = 64
    out_dim: int = 64
    heads: Union[int, List[int]] = field(default_factory=lambda: [4, 2, 1])
    concat_last: bool = False
    edge_mode: str = "bidirectional"
    self_loops: bool = True
    virtual_node: bool = False
    residual: bool = True
    norm: str = "layer"
    activation: str = "elu"
    dropout: float = 0.0          # on node features after each layer
    attn_dropout: float = 0.0     # on attention coefficients (GAT/GATv2/Transformer)
    edge_dropout: float = 0.0     # random edge removal during training
    jk: str = "last"

    def __post_init__(self) -> None:
        _check(self.conv, CONV_TYPES, "graph.conv")
        _check(self.edge_mode, EDGE_MODES, "graph.edge_mode")
        _check(self.norm, NORM_TYPES, "graph.norm")
        _check(self.activation, ACTIVATIONS, "graph.activation")
        _check(self.jk, JK_MODES, "graph.jk")
        if self.num_layers < 1:
            raise ValueError("graph.num_layers must be >= 1")
        self.heads_per_layer()  # validates

    def heads_per_layer(self) -> List[int]:
        if isinstance(self.heads, int):
            if self.heads < 1:
                raise ValueError("graph.heads must be >= 1")
            return [self.heads] * self.num_layers
        heads = [int(h) for h in self.heads]
        if len(heads) != self.num_layers:
            raise ValueError(
                f"graph.heads has {len(heads)} entries but graph.num_layers={self.num_layers}; "
                f"give one entry per layer or a single int (e.g. graph.heads=4)")
        if min(heads) < 1:
            raise ValueError("graph.heads entries must be >= 1")
        return heads


@dataclass
class ReadoutConfig:
    """Node embeddings -> one plan embedding."""
    pooling: str = "mean"
    gate_hidden: int = 0   # attention pooling: 0 -> Linear(d,1) gate; >0 -> q^T tanh(W h) gate

    def __post_init__(self) -> None:
        _check(self.pooling, POOLINGS, "readout.pooling")


@dataclass
class HeadConfig:
    """Plan embedding -> prediction."""
    kind: str = "resmlp"
    hidden_dim: int = 64
    num_blocks: int = 2
    dropout: float = 0.0
    out_dim: int = 1
    out_activation: str = "none"

    def __post_init__(self) -> None:
        _check(self.kind, HEAD_KINDS, "head.kind")
        _check(self.out_activation, OUT_ACTIVATIONS, "head.out_activation")
        if self.num_blocks < 0:
            raise ValueError("head.num_blocks must be >= 0")


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
def _parse_override_value(raw: str) -> Any:
    text = raw.strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":          # 'none' stays a string: it is a valid conv / norm / activation value
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _dataclass_from_dict(cls, data: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown field(s) {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class GNTOConfig:
    node: NodeEncoderConfig = field(default_factory=NodeEncoderConfig)
    graph: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    name: str = "gnto"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.readout.pooling == "virtual" and not self.graph.virtual_node:
            raise ValueError("readout.pooling='virtual' requires graph.virtual_node=true")

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GNTOConfig":
        data = dict(data)
        sub = {
            "node": NodeEncoderConfig,
            "graph": GraphEncoderConfig,
            "readout": ReadoutConfig,
            "head": HeadConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, sub_cls in sub.items():
            if key in data:
                value = data.pop(key)
                kwargs[key] = value if isinstance(value, sub_cls) else _dataclass_from_dict(sub_cls, value)
        for key in ("name", "notes"):
            if key in data:
                kwargs[key] = data.pop(key)
        if data:
            raise ValueError(f"GNTOConfig: unknown top-level field(s) {sorted(data)}")
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GNTOConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- editing -----------------------------------------------------------
    def copy(self) -> "GNTOConfig":
        return copy.deepcopy(self)

    def override(self, *assignments: str, **kwargs: Any) -> "GNTOConfig":
        """Return a new config with dotted-path assignments applied.

        ``cfg.override("graph.num_layers=5", "graph.heads=[4,4,4,4,1]",
        "readout.pooling=attention")`` or ``cfg.override(graph__num_layers=5)``.
        Values are parsed as JSON when possible (ints, floats, lists, bools).
        """
        data = self.to_dict()
        items: List[tuple] = []
        for a in assignments:
            if "=" not in a:
                raise ValueError(f"override {a!r} must look like 'section.field=value'")
            key, raw = a.split("=", 1)
            items.append((key.strip(), _parse_override_value(raw)))
        for key, value in kwargs.items():
            items.append((key.replace("__", "."), value))
        for key, value in items:
            parts = key.split(".")
            cursor = data
            for p in parts[:-1]:
                if p not in cursor or not isinstance(cursor[p], dict):
                    raise ValueError(f"override: unknown section {p!r} in {key!r}")
                cursor = cursor[p]
            if parts[-1] not in cursor:
                raise ValueError(f"override: unknown field {key!r}")
            cursor[parts[-1]] = value
        return GNTOConfig.from_dict(data)

    def describe(self) -> str:
        g, n, r, h = self.graph, self.node, self.readout, self.head
        return (
            f"{self.name}: node={n.kind}(out={n.out_dim}) | "
            f"graph={g.conv} L={g.num_layers} heads={g.heads_per_layer()} hidden={g.hidden_dim} "
            f"edges={g.edge_mode} self_loops={g.self_loops} vnode={g.virtual_node} jk={g.jk} | "
            f"readout={r.pooling} | head={h.kind}x{h.num_blocks}({h.hidden_dim})->{h.out_dim} {h.out_activation}"
        )


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
# Vocabulary sizes of QueryFormer's checkpoints/encoding.pt (= data/qf/encoding.pt;
# IMDB / JOB corpus); derive them from a live ``encoding`` object with
# ``models.node_encoder.qf_vocab_from_encoding``.
QF_IMDB_VOCAB: Dict[str, int] = {"types": 13, "tables": 7, "joins": 16, "columns": 21, "ops": 4}


def _qf_base(name: str, notes: str, plan_rows: bool = True, join_encoding: str = "columns") -> GNTOConfig:
    return GNTOConfig(
        name=name,
        notes=notes,
        node=NodeEncoderConfig(kind="queryformer", out_dim=64, embed_size=64,
                               vocab=dict(QF_IMDB_VOCAB), use_hist=True,
                               use_sample=True, bin_number=50, plan_rows=plan_rows,
                               join_encoding=join_encoding),
        graph=GraphEncoderConfig(conv="gatv2", num_layers=3, hidden_dim=64, out_dim=64,
                                 heads=[4, 2, 1], edge_mode="as_is", self_loops=True,
                                 residual=True, norm="layer", activation="elu",
                                 dropout=0.0, attn_dropout=0.0, jk="last"),
        readout=ReadoutConfig(pooling="mean"),
        head=HeadConfig(kind="resmlp", hidden_dim=64, num_blocks=2, dropout=0.0,
                        out_dim=1, out_activation="sigmoid"),
    )


def preset_gnto() -> GNTOConfig:
    """GNTO default: QueryFormer node features + Plan Rows channel, join predicates
    encoded compositionally from their two column ids (``join_encoding="columns"``),
    3-layer GATv2 (heads 4/2/1, edges as stored by the adapter + self-loops), mean
    readout, residual head, sigmoid output (min-max normalised log latency)."""
    return _qf_base("gnto", "default GNTO: qf features + plan rows, join = f(col_left, col_right)")


def preset_gnto_joinid() -> GNTOConfig:
    """Ablation: identical to ``gnto`` but the join predicate is a per-string
    vocabulary id (QueryFormer's original design).  This is the architecture of
    the pre-rewrite ``GNTO_QF_Model(add_plan_rows=True)`` checkpoints,
    which still load into it (see ``GNTO.from_legacy_checkpoint``)."""
    return _qf_base("gnto_joinid", "ablation: join = per-string vocabulary id (legacy GNTO_QF_Model)",
                    join_encoding="id")


def preset_qf_v3() -> GNTOConfig:
    """Legacy name of :func:`preset_gnto_joinid` (kept so old commands / configs work)."""
    cfg = preset_gnto_joinid()
    cfg.name = "qf_v3"
    return cfg


def preset_qf_v4() -> GNTOConfig:
    """Legacy: string-id joins + global attention pooling (GATv2TreeEncoder_V4)."""
    cfg = _qf_base("qf_v4", "legacy V4: join id + attention readout", join_encoding="id")
    cfg.readout = ReadoutConfig(pooling="attention", gate_hidden=0)
    return cfg


def preset_qf_bidir() -> GNTOConfig:
    """``gnto`` with bidirectional parent<->child edges."""
    cfg = _qf_base("qf_bidir", "gnto + bidirectional edges")
    cfg.graph.edge_mode = "bidirectional"
    return cfg


def preset_qf_nostats() -> GNTOConfig:
    """'GNTO (w/o Stats)': ``gnto`` without the histogram / sample channels."""
    cfg = _qf_base("qf_nostats", "gnto without histogram or table-sample channels")
    cfg.node.use_hist = False
    cfg.node.use_sample = False
    return cfg


def preset_dace_zero_shot() -> GNTOConfig:
    """GNTO_DACE_Model equivalent (cross-database, schema-agnostic features).

    ``node.in_dim`` must equal ``num_node_types + 2`` of the DACE workload; the
    default 22 (20 node types + scaled Total Cost / Plan Rows) matches
    ``data/dace/workload1/statistics.json``.
    """
    return GNTOConfig(
        name="dace_zero_shot",
        notes="GNTO_DACE_Model equivalent; set node.in_dim = num_node_types + 2",
        node=NodeEncoderConfig(kind="mlp", in_dim=22, hidden_dim=64, out_dim=64,
                               num_layers=2, norm="layer", activation="relu"),
        graph=GraphEncoderConfig(conv="gatv2", num_layers=3, hidden_dim=64, out_dim=64,
                                 heads=[4, 4, 1], edge_mode="as_is", self_loops=True,
                                 dropout=0.1, attn_dropout=0.1, jk="last"),
        readout=ReadoutConfig(pooling="mean"),
        head=HeadConfig(kind="resmlp", hidden_dim=64, num_blocks=2, dropout=0.1,
                        out_dim=1, out_activation="sigmoid"),
    )


def preset_plan_flat() -> GNTOConfig:
    """Generic encoder for the flat plan layout produced by ``build_dataset`` /
    the LIMAO featurizer: ``[type | rows, width | (col, op, val, is_col) x 3]``."""
    return GNTOConfig(
        name="plan_flat",
        notes="heterogeneous flat-layout node encoder with DeepSets predicates",
        node=NodeEncoderConfig(kind="plan", out_dim=64, num_node_types=32, num_cols=64,
                               num_ops=8, num_numeric=2, pred_slots=3, pred_pool="sum"),
        graph=GraphEncoderConfig(conv="gatv2", num_layers=3, hidden_dim=64, out_dim=64,
                                 heads=4, edge_mode="bidirectional", self_loops=True),
        readout=ReadoutConfig(pooling="attention", gate_hidden=64),
        head=HeadConfig(kind="resmlp", hidden_dim=64, num_blocks=2, out_dim=1),
    )


def preset_tiny() -> GNTOConfig:
    """Small everything; for unit tests and smoke runs."""
    return GNTOConfig(
        name="tiny",
        node=NodeEncoderConfig(kind="mlp", in_dim=8, hidden_dim=16, out_dim=16, num_layers=1),
        graph=GraphEncoderConfig(conv="gatv2", num_layers=2, hidden_dim=16, out_dim=16, heads=2),
        readout=ReadoutConfig(pooling="mean"),
        head=HeadConfig(kind="resmlp", hidden_dim=16, num_blocks=1),
    )


PRESETS: Dict[str, Callable[[], GNTOConfig]] = {
    "gnto": preset_gnto,
    "gnto_joinid": preset_gnto_joinid,
    "qf_v3": preset_qf_v3,
    "qf_v4": preset_qf_v4,
    "qf_bidir": preset_qf_bidir,
    "qf_nostats": preset_qf_nostats,
    "dace_zero_shot": preset_dace_zero_shot,
    "plan_flat": preset_plan_flat,
    "tiny": preset_tiny,
}


def get_preset(name: str, *overrides: str, **kwargs: Any) -> GNTOConfig:
    """``get_preset("qf_v3", "graph.num_layers=5")`` -> fresh, validated config."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    cfg = PRESETS[name]()
    if overrides or kwargs:
        cfg = cfg.override(*overrides, **kwargs)
    return cfg


def list_presets() -> List[str]:
    return sorted(PRESETS)
