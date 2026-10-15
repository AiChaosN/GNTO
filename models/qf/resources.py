"""Where the QueryFormer inputs live inside this repo, and how to load them.

``GNTO/data/``                     plan corpus shared by both models
    train_plan_part{0..19}.csv     100k IMDB plans (id, json); parts 0-17 train, 18-19 val
    job-light_plan.csv             70 JOB-light plans (id, json, sql)
    synthetic_plan.csv             500 synthetic plans (id, json, sql)
    column_min_max_vals.csv        column ranges (also inside encoding.pt)
``GNTO/data/qf/``                  QueryFormer featurisation resources (from the official QueryFormer
                                   repository: ``data/imdb`` + ``checkpoints/``)
    histogram_string.csv           per-column 50-bin histograms
    train.csv / train.bitmaps      per-training-query 1000-row sample bitmaps (MSCN format)
    workloads/<w>.csv|.bitmaps|.sql  same for the job-light / synthetic workloads
    encoding.pt                    trained vocabulary (Encoding object: types / joins / tables / columns)
    cost_model.pt                  upstream pre-trained QueryFormer checkpoint (not used for training)

``encoding.pt`` was pickled with the upstream class path ``model.database_util.Encoding``;
:func:`load_encoding` remaps it to :class:`models.qf.database_util.Encoding` while
unpickling, so no ``model`` shim package is needed on ``sys.path``.
"""
from __future__ import annotations

import os
import pickle
import types
from typing import Any, Dict, Optional

import torch

from . import database_util as _dbu
from .database_util import Encoding, get_hist_file, get_job_table_sample
from .util import Normalizer

GNTO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(GNTO_ROOT, "data")
QF_DATA_DIR = os.path.join(DATA_DIR, "qf")

# upstream Normalizer(-3.61192, 12.290855): min / max of log(exec_time_ms + 0.001) over the corpus
COST_NORM_MIN, COST_NORM_MAX = -3.61192, 12.290855

# hyper-parameters of the QueryFormer baseline as trained in this repository
QF_DEFAULT_HPARAMS: Dict[str, Any] = dict(
    emb_size=64, ffn_dim=128, head_size=12, n_layers=8, dropout=0.1, pred_hid=128,
    use_sample=True, use_hist=True, bin_number=50,
)

_UPSTREAM_MODULES = {
    "model.database_util": "models.qf.database_util",
    "model.util": "models.qf.util",
    "model.dataset": "models.qf.dataset",
    "model.model": "models.qf.model",
}


class _RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_UPSTREAM_MODULES.get(module, module), name)


_remap_pickle = types.SimpleNamespace(Unpickler=_RemapUnpickler, load=pickle.load, loads=pickle.loads,
                                      dump=pickle.dump, dumps=pickle.dumps, __name__="models.qf._remap_pickle")


def torch_load_upstream(path: str, map_location="cpu"):
    """``torch.load`` that understands objects pickled under the upstream ``model.*`` package."""
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=_remap_pickle)


def plan_csv_path(part: int) -> str:
    return os.path.join(DATA_DIR, f"train_plan_part{part}.csv")


def workload_plan_csv(name: str) -> str:
    """``job-light`` / ``synthetic`` -> ``data/<name>_plan.csv``."""
    return os.path.join(DATA_DIR, f"{name}_plan.csv")


def workload_sample_prefix(name: str) -> str:
    """Prefix for ``get_job_table_sample`` (``<prefix>.csv`` + ``<prefix>.bitmaps``)."""
    return os.path.join(QF_DATA_DIR, "workloads", name)


def load_encoding(path: Optional[str] = None) -> Encoding:
    path = path or os.path.join(QF_DATA_DIR, "encoding.pt")
    payload = torch_load_upstream(path)
    enc = payload["encoding"] if isinstance(payload, dict) and "encoding" in payload else payload
    if not isinstance(enc, Encoding):
        raise TypeError(f"{path} does not contain an Encoding (got {type(enc).__name__})")
    return enc


# --------------------------------------------------------------------------- #
# robustness patches (originally in adapters/qf_adapter.py)
# --------------------------------------------------------------------------- #
def _robust_encode_filters(self, filters=(), alias=None):
    """Tolerant replacement for ``Encoding.encode_filters``.

    Upstream assumes every predicate is a numeric ``col op num`` triple.  Live PG
    plans (JOB) contain string literals with spaces, LIKE / IN / IS NULL, >3
    predicates per node ... which crash it.  Parseable triples are encoded
    exactly as upstream does; everything else becomes the NA sentinel; at most
    3 predicates are kept (``node2feature`` pads to 3)."""
    if len(filters) == 0:
        return {'colId': [self.col2idx['NA']], 'opId': [self.op2idx['NA']], 'val': [0.0]}
    res = {'colId': [], 'opId': [], 'val': []}
    for filt in filters:
        filt = ''.join(c for c in filt if c not in '()')
        for f in filt.split(' AND '):
            parts = f.split(' ')
            if len(parts) == 3:
                col, op, num = parts
                column = (alias + '.' + col) if alias else col
                try:
                    cid = self.col2idx[column]
                    oid = self.op2idx[op]
                    val = self.normalize_val(column, float(num))
                    res['colId'].append(cid); res['opId'].append(oid); res['val'].append(val)
                    continue
                except (KeyError, ValueError):
                    pass
            res['colId'].append(self.col2idx['NA']); res['opId'].append(self.op2idx['NA']); res['val'].append(0.0)
    if not res['colId']:
        return {'colId': [self.col2idx['NA']], 'opId': [self.op2idx['NA']], 'val': [0.0]}
    return {k: v[:3] for k, v in res.items()}


def harden_encoding(encoding: Encoding, freeze_vocab: bool = True, robust_filters: bool = True) -> Encoding:
    """In-place: freeze the vocabulary (unseen type / join / table -> id 0 instead of
    growing the embedding tables past what a trained model expects) and/or swap
    in the tolerant filter parser.  Also soft-wraps ``formatFilter`` /
    ``formatJoin`` so odd live-plan nodes yield "no filter" / "no join" instead
    of raising.  Identical output to upstream on the QueryFormer corpus."""
    if freeze_vocab:
        encoding.encode_type = types.MethodType(lambda self, t: self.type2idx.get(t, 0), encoding)
        encoding.encode_join = types.MethodType(lambda self, j: self.join2idx.get(j, 0), encoding)
        encoding.encode_table = types.MethodType(lambda self, t: self.table2idx.get(t, 0), encoding)
    if robust_filters:
        encoding.encode_filters = types.MethodType(_robust_encode_filters, encoding)
        from . import dataset as _ds
        if not getattr(_ds, "_softwrapped", False):
            _off, _oj = _ds.formatFilter, _ds.formatJoin

            def _safe_formatFilter(plan, _o=_off):
                try:
                    return _o(plan)
                except Exception:
                    return [], plan.get("Alias")

            def _safe_formatJoin(plan, _o=_oj):
                try:
                    return _o(plan)
                except Exception:
                    return ""
            _ds.formatFilter, _ds.formatJoin = _safe_formatFilter, _safe_formatJoin
            _ds._softwrapped = True
    return encoding


class _ZeroRow:
    def __getitem__(self, table):
        import numpy as np
        return np.zeros(1000)


class ZeroTableSample:
    """``table_sample`` stand-in: ``ts[query_id][table]`` is an all-zero 1000-bit
    sample for every (query, table) - for plans that are in no sample file
    (live steering).  Same value upstream uses for ``table_id == 0``."""

    def __getitem__(self, query_id):
        return _ZeroRow()


def load_qf_resources(*, freeze_vocab: bool = True, robust_filters: bool = True,
                      zero_table_sample: bool = False, sample: str = "train",
                      qf_dir: Optional[str] = None) -> Dict[str, Any]:
    """Everything needed to featurise plans: ``encoding``, ``hist_file``,
    ``table_sample`` (for ``sample='train'`` the 100k training queries; pass a
    workload name for its file), ``cost_norm`` / ``card_norm`` and the paths."""
    qf_dir = qf_dir or QF_DATA_DIR
    hist_file = get_hist_file(os.path.join(qf_dir, "histogram_string.csv"))
    encoding = harden_encoding(load_encoding(os.path.join(qf_dir, "encoding.pt")), freeze_vocab, robust_filters)
    if zero_table_sample:
        table_sample = ZeroTableSample()
    else:
        prefix = os.path.join(qf_dir, "train") if sample == "train" else os.path.join(qf_dir, "workloads", sample)
        table_sample = get_job_table_sample(prefix)
    return {
        "encoding": encoding,
        "hist_file": hist_file,
        "table_sample": table_sample,
        "cost_norm": Normalizer(COST_NORM_MIN, COST_NORM_MAX),
        "card_norm": Normalizer(1, 100),
        "data_dir": DATA_DIR,
        "qf_dir": qf_dir,
        "plan_dir": DATA_DIR,
    }


__all__ = ["GNTO_ROOT", "DATA_DIR", "QF_DATA_DIR", "COST_NORM_MIN", "COST_NORM_MAX", "QF_DEFAULT_HPARAMS",
           "torch_load_upstream", "load_encoding", "harden_encoding", "ZeroTableSample", "load_qf_resources",
           "plan_csv_path", "workload_plan_csv", "workload_sample_prefix"]
