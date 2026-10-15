"""GNTO - Graph Neural Tree Optimizer: modular learned cost model.

Quick start::

    from models.gnto import GNTO, get_preset
    cfg = get_preset("gnto", "graph.num_layers=5", "graph.edge_mode=bidirectional")
    model = GNTO.from_config(cfg)
    print(model.summary())

Layout
------
config.py         GNTOConfig + per-component dataclasses, presets, overrides
node_encoder.py   MLPNodeEncoder / PlanNodeEncoder / QueryFormerNodeEncoder
graph_encoder.py  GraphEncoder (conv type, depth, heads, edge_mode, self-loops, virtual node, JK)
readout.py        Readout (mean / max / sum / attention / first / virtual)
head.py           PredictionHead (residual MLP or plain MLP)
gnto.py           GNTO = node_encoder -> graph_encoder -> readout -> head; save / load
losses.py         Huber / MSE / L1, pairwise ranking, combined
metrics.py        Q-error percentiles, Spearman / Kendall, pairwise accuracy, top-k regret
graph_utils.py    edge re-orientation, virtual nodes, EXPLAIN-JSON -> graph
legacy_compat.py  load checkpoints written by the pre-rewrite classes

The pre-rewrite classes (``NodeEncoder_QF``, ``GATv2TreeEncoder_V3`` ...) are
not part of this repository; their checkpoints still load via
``GNTO.load_state_dict`` (keys converted on the fly).
"""
from .config import (
    GNTOConfig, GraphEncoderConfig, HeadConfig, NodeEncoderConfig, ReadoutConfig,
    PRESETS, QF_IMDB_VOCAB, get_preset, list_presets,
    CONV_TYPES, EDGE_MODES, POOLINGS, NODE_ENCODER_KINDS, JK_MODES, JOIN_ENCODINGS,
)
from .node_encoder import (
    NodeEncoder, MLPNodeEncoder, PlanNodeEncoder, PredicateSetEncoder, QueryFormerNodeEncoder,
    build_node_encoder, qf_vocab_from_encoding,
)
from .graph_encoder import GraphEncoder, GraphEncoderLayer
from .readout import Readout
from .head import PredictionHead, ResBlock
from .gnto import GNTO, build_model, CHECKPOINT_FORMAT
from .losses import PairwiseRankingLoss, PointwiseLoss, CombinedLoss, build_loss
from .metrics import (qerror_percentiles, ranking_metrics, pairwise_accuracy, topk_regret,
                            evaluation_summary)
from .graph_utils import (
    transform_edges, with_self_loops, add_virtual_nodes, first_node_index, plan_to_graph,
    edge_index_from_pairs,
)
from .legacy_compat import (is_legacy_state_dict, convert_legacy_state_dict, legacy_qf_config,
                                  legacy_dace_config)

__all__ = [
    # config
    "GNTOConfig", "GraphEncoderConfig", "HeadConfig", "NodeEncoderConfig", "ReadoutConfig",
    "PRESETS", "QF_IMDB_VOCAB", "get_preset", "list_presets",
    "CONV_TYPES", "EDGE_MODES", "POOLINGS", "NODE_ENCODER_KINDS", "JK_MODES", "JOIN_ENCODINGS",
    # components
    "NodeEncoder", "MLPNodeEncoder", "PlanNodeEncoder", "PredicateSetEncoder", "QueryFormerNodeEncoder",
    "build_node_encoder", "qf_vocab_from_encoding",
    "GraphEncoder", "GraphEncoderLayer", "Readout", "PredictionHead", "ResBlock",
    # model
    "GNTO", "build_model", "CHECKPOINT_FORMAT",
    # losses
    "PairwiseRankingLoss", "PointwiseLoss", "CombinedLoss", "build_loss",
    # metrics
    "qerror_percentiles", "ranking_metrics", "pairwise_accuracy", "topk_regret", "evaluation_summary",
    # graph utils
    "transform_edges", "with_self_loops", "add_virtual_nodes", "first_node_index", "plan_to_graph",
    "edge_index_from_pairs",
    # legacy
    "is_legacy_state_dict", "convert_legacy_state_dict", "legacy_qf_config", "legacy_dace_config",
]
