"""CPU inference latency of the small QueryFormer next to the published one and GNTO.

Reuses exp/exp1_qerror/bench_latency.py unchanged (same plans, same batching, same timing loop) and only
replaces its model list; a QueryFormer checkpoint is rebuilt with the hyper-parameters stored in the
``config.json`` next to it.  Run on an otherwise idle machine.

    python analysis/dependency_study/bench_small_qf.py      # -> results/analysis/dependency_20261003/tables/latency_small_qf/
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "exp", "exp1_qerror"))
import bench_latency as bl                                      # noqa: E402
from models.qf import QF_DEFAULT_HPARAMS, QueryFormer            # noqa: E402

R = bl.R
BASE = R("main_runs")

# The Bao checkpoint was saved from a GPU and its loader passes no map_location; this benchmark is CPU only.
_torch_load = torch.load


def _load_on_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _torch_load(*args, **kwargs)


torch.load = _load_on_cpu


class QFRunner(bl.QFRunner):
    def __init__(self, ckpt, items, fixed_pad):
        cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
        hp = dict(QF_DEFAULT_HPARAMS)
        if os.path.exists(cfg_path):
            hp.update(json.load(open(cfg_path)).get("hparams", {}))
        self.model = QueryFormer(**hp)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.model.eval()
        self.items, self.fixed_pad = items, fixed_pad
        self.params = sum(p.numel() for p in self.model.parameters())


bl.QFRunner = QFRunner
bl.MODELS = [
    ("QueryFormer", "qf", R("exp1_qerror", "qf", "best_model.pt")),
    ("QueryFormer small", "qf", os.path.join(BASE, "capacity", "qf_small_s42", "best_model.pt")),
    ("Bao tree-CNN", "bao", R("exp3_e2e", "bao", "bao_model")),
    ("GNTO default (GATv2 256/128/64, 3 layers)", "gnto", R("exp1_qerror", "gnto", "best_model.pt")),
    ("GNTO GCN (3 layers)", "gnto", R("exp4_ablation", "gnto_gcn", "best_model.pt")),
    ("GNTO None (3 layers)", "gnto", R("exp4_ablation", "gnto_nognn", "best_model.pt")),
]

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", R("analysis", "dependency_20261003", "tables", "latency_small_qf")]
    if "--no-gpu" not in sys.argv:
        sys.argv.append("--no-gpu")
    bl.main()
