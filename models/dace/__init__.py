"""DACE (ICDE 2024) baseline, ported from the official DACE code (model.py, plan_utils.py).

model.py       DACELora (1-layer Transformer + LoRA MLP head, per-node runtime), dace_loss
plan_utils.py  NodeFeaturizer (22-d node vector shared with the GNTO adapter), plan_encoding,
               DACEDataset, build_dace_dataset (cacheable)
trainer.py     train / predict / qerror_stats reproducing the upstream Lightning protocol
resources.py   data/dace/workload1 paths, the 20 databases, the ``paper`` and random 10/10 splits
"""
from .model import DACELora, dace_loss, q_error, DACE_DEFAULT_HPARAMS
from .plan_utils import (PLAN_PARAMETERS, PAD_LENGTH, MAX_RUNTIME, LOSS_WEIGHT, load_plans, count_nodes,
                         NodeFeaturizer, plan_encoding, attention_mask, DACEDataset, encode_plans,
                         build_dace_dataset)
from .trainer import count_parameters, predict, qerror_stats, train, load_checkpoint, PERCENTILES
from .resources import (DACE_DATA_DIR, WORKLOADS, PAPER_TRAIN_IDS, PAPER_TEST_IDS, statistics_path,
                        plans_path, load_statistics, make_split)

__all__ = [
    "DACELora", "dace_loss", "q_error", "DACE_DEFAULT_HPARAMS",
    "PLAN_PARAMETERS", "PAD_LENGTH", "MAX_RUNTIME", "LOSS_WEIGHT", "load_plans", "count_nodes",
    "NodeFeaturizer", "plan_encoding", "attention_mask", "DACEDataset", "encode_plans", "build_dace_dataset",
    "count_parameters", "predict", "qerror_stats", "train", "load_checkpoint", "PERCENTILES",
    "DACE_DATA_DIR", "WORKLOADS", "PAPER_TRAIN_IDS", "PAPER_TEST_IDS", "statistics_path", "plans_path",
    "load_statistics", "make_split",
]
