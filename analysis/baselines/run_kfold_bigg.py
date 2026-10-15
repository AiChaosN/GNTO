"""BiGG - the plan encoder of Reqo (Chang, Kamali, Kantere; SIGMOD 2026) - as the dependency encoder of GNTO in plan
selection.

``exp/exp3_e2e/run_kfold.py`` is run unchanged (candidates, template folds, inputs, node encoder, head, objective,
optimiser, schedule, batches, epochs); only the dependency encoder and the readout are replaced by the released
``Models.reqo_model.BiGG`` (https://github.com/BaomingChang/Reqo-on-PostgreSQL): direction-specific TransformerConv
layers combined by a learned weight, batch normalisation, GRU readout over the operators in post-order.  Width 64 and
three layers as the other encoders of the comparison; one attention head, dropout 0.1 and alpha 0.3 as in the release.

    python analysis/baselines/run_kfold_bigg.py <the arguments of run_kfold.py>   (--model gnto, graph.conv is ignored)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torch_geometric.utils import sort_edge_index

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQO = os.path.join(os.path.dirname(ROOT), "baselines_src", "Reqo-on-PostgreSQL")
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
sys.path.append(REQO)

import run_kfold as rk                                         # noqa: E402
from Models.reqo_model import BiGG                             # noqa: E402  (released Reqo code)

_GNTO = rk.GNTO
_to_graph = rk.to_graph


class BiGGInGNTO(nn.Module):
    """GNTO's node encoder and head around Reqo's BiGG encoder + GRU readout."""

    def __init__(self, cfg):
        super().__init__()
        base = _GNTO.from_config(cfg.override("graph.conv=none"))
        self.node_encoder, self.head = base.node_encoder, base.head
        params = {"encoder_attention_heads": 1, "encoder_gnn_dropout_rate": 0.1, "encoder_gnn_embedding_dim": self.head.in_dim,
                  "encoder_conv_layers": cfg.graph.num_layers, "encoder_dirgnn_alpha": 0.3}
        self.bigg = BiGG(params, self.node_encoder.out_dim)
        self.config = cfg

    def forward(self, batch):
        return self.head(self.bigg(self.node_encoder(batch.x), batch.edge_index, batch.batch))

    @torch.no_grad()
    def predict(self, batch):
        was = self.training
        self.eval()
        try:
            return self.forward(batch).view(-1)
        finally:
            self.train(was)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config.to_dict(), "encoder": "BiGG (Reqo release)",
                    "extra": extra}, path)


class _Factory:
    """Stands in for ``GNTO`` inside run_kfold: ``GNTO.from_config(cfg)`` builds the BiGG variant."""

    @staticmethod
    def from_config(cfg, *overrides):
        return BiGGInGNTO(cfg.override(*overrides) if overrides else cfg)


def to_graph(plan, fz, y_norm, qid, hint, ms, censored=False, parent_to_child=False):
    """The graph of run_kfold with the operators in post-order and child -> parent edges (as the release)."""
    d = _to_graph(plan, fz, y_norm, qid, hint, ms, censored, parent_to_child=False)
    n = d.x.size(0)
    ei = d.edge_index
    down = ei[:, ei[0] < ei[1]]                                # parent -> child in DFS pre-order
    kids = [[] for _ in range(n)]
    for p, c in down.t().tolist():
        kids[p].append(c)
    order, stack = [], [(0, False)]
    while stack:                                               # iterative post-order
        v, done = stack.pop()
        if done:
            order.append(v)
        else:
            stack.append((v, True))
            stack.extend((c, False) for c in reversed(kids[v]))
    assert len(order) == n
    new = torch.empty(n, dtype=torch.long)
    new[torch.tensor(order)] = torch.arange(n)
    d.x = d.x[torch.tensor(order)]
    d.edge_index = (sort_edge_index(torch.stack([new[down[1]], new[down[0]]])) if down.numel()
                    else torch.zeros((2, 0), dtype=torch.long))
    return d


@torch.no_grad()
def predict(model, items, device, max_runtime):
    """The GRU readout of the release reads the zero padding of a batch, so its output depends on the plans it is
    batched with.  Test plans are therefore batched as in training: the candidates of ``QUERIES_PER_BATCH`` queries."""
    import numpy as np
    from torch_geometric.data import Batch
    model.eval()
    order, groups = [], {}
    for i, d in enumerate(items):
        if d.qid not in groups:
            order.append(d.qid)
            groups[d.qid] = []
        groups[d.qid].append(i)
    out = np.zeros(len(items))
    for k in range(0, len(order), QUERIES_PER_BATCH):
        idx = [i for q in order[k:k + QUERIES_PER_BATCH] for i in groups[q]]
        out[idx] = model.predict(Batch.from_data_list([items[i] for i in idx]).to(device)).cpu().numpy() * max_runtime
    return out


QUERIES_PER_BATCH = 3                                          # --queries-per-batch of the factorial
rk.GNTO, rk.to_graph, rk.predict_gnto = _Factory, to_graph, predict

if __name__ == "__main__":
    rk.main()
