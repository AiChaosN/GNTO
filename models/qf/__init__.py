"""QueryFormer (VLDB 2022) baseline, ported from the official QueryFormer code
(``model/`` in https://github.com/zhaoyue-ntu/QueryFormer).

Layout
------
model.py          QueryFormer / FeatureEmbed / Prediction (verbatim upstream)
database_util.py  Encoding, TreeNode, histogram + sample-bitmap loaders, padding, collator (verbatim upstream)
util.py           Normalizer, seed_everything (verbatim upstream)
dataset.py        plan JSON -> 1165-d node features; PlanTreeDataset (upstream API) and the
                  parallel, cacheable, lazily padded PlanDataset used by our scripts
trainer.py        train / evaluate / qerror_stats loops (adapted from upstream ``model/trainer.py``)
resources.py      loads histogram, sample bitmaps, encoding.pt and plan CSVs from ``GNTO/data``

Quick start::

    from models.qf import load_qf_resources, build_plan_dataset, QueryFormer, train, evaluate
    res = load_qf_resources()
    ds = build_plan_dataset(df, res["encoding"], res["hist_file"], res["table_sample"], res["cost_norm"])
    model = QueryFormer(emb_size=64, ffn_dim=128, head_size=12, n_layers=8, dropout=0.1, pred_hid=128)
"""
from .model import QueryFormer, FeatureEmbed, Prediction
from .database_util import (Encoding, TreeNode, Batch, collator, get_hist_file, get_job_table_sample,
                            formatFilter, formatJoin, filterDict2Hist)
from .util import Normalizer, seed_everything
from .dataset import (PlanFeaturizer, PlanTreeDataset, PlanDataset, build_plan_dataset, featurize_plans,
                      pad_item, node2feature, join_columns, MAX_NODE, REL_POS_MAX)
from .trainer import train, evaluate, predict, qerror_stats, iter_batches, count_parameters
from .resources import (QF_DATA_DIR, DATA_DIR, load_encoding, load_qf_resources, harden_encoding,
                        plan_csv_path, workload_plan_csv, workload_sample_prefix, QF_DEFAULT_HPARAMS,
                        COST_NORM_MIN, COST_NORM_MAX)

__all__ = [
    "QueryFormer", "FeatureEmbed", "Prediction",
    "Encoding", "TreeNode", "Batch", "collator", "get_hist_file", "get_job_table_sample",
    "formatFilter", "formatJoin", "filterDict2Hist",
    "Normalizer", "seed_everything",
    "PlanFeaturizer", "PlanTreeDataset", "PlanDataset", "build_plan_dataset", "featurize_plans",
    "pad_item", "node2feature", "join_columns", "MAX_NODE", "REL_POS_MAX",
    "train", "evaluate", "predict", "qerror_stats", "iter_batches", "count_parameters",
    "QF_DATA_DIR", "DATA_DIR", "load_encoding", "load_qf_resources", "harden_encoding",
    "plan_csv_path", "workload_plan_csv", "workload_sample_prefix", "QF_DEFAULT_HPARAMS",
    "COST_NORM_MIN", "COST_NORM_MAX",
]
