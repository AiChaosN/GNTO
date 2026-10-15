"""Reqo (Chang, Kamali, Kantere; PACMMOD 4(1), SIGMOD 2026) on the plan-selection folds of exp/exp3_e2e/run_kfold.py.

``exp/exp3_e2e/run_kfold.py`` is run unchanged (same candidates, de-duplication, plan representation, template folds,
seeds, capped labels and output contract as Bao-ext / LIMAO-ext / GNTO); four of its functions are replaced so that
the model of a fold is the *released* Reqo network trained with the *released* objective
(https://github.com/BaomingChang/Reqo-on-PostgreSQL, imported from ``../baselines_src``):

* network      type embedding 16 -> BiGG (direction-specific TransformerConv, 256 x 3, one head, batch norm, GRU readout
               over the operators in post-order) -> estimator (512) with a latency head, a variance head and the
               integrated score ``IC`` that Reqo ranks plans by               (configuration of the released main.py)
* objective    uncertainty loss + pairwise ranking loss with margin 3 over the pairs of a shuffled batch of 128
* optimiser    Adam 1e-3, at most 100 epochs
* epoch        ``--reqo-select val`` (default): the early stopping of the release - the epoch with the lowest ratio of
               selected to optimal runtime, patience 20 - evaluated on 15 % of the *training* queries, which are held
               out from training (the release evaluates it on the test fold)
               ``--reqo-select last``: the last epoch, the rule of every other model of the benchmark.
* batch size   128 as released (``--bs 128``, the default).  On these workloads that is 4 to 12 updates per epoch;
               ``--bs 16`` (the batch size of Bao's released loop) is run as a second variant.
* inputs       ``--reqo-inputs tcr``   operator type, estimated cost and rows (log min-max, training fold)
               ``--reqo-inputs full``  + the tables an operator touches and Reqo's per-column predicate encoding
                                       (``Node_info_extract`` of the release; database statistics from the pg_catalog
                                       snapshot ``--reqo-stats``, see ``reqo_db.py``)
* labels       log(ms + 1), min-max with the minimum of the training fold and the 32 s cap as maximum
* selection    the candidate with the lowest integrated score ``IC`` (as ``get_plansubop_and_runtime`` of the release)

``predictions.csv`` holds the latency of the latency head in ``pred`` (so Q-error and rank correlation are comparable
with the other models) and the integrated score in ``score``; ``picks.csv`` follows ``score``.

    python analysis/baselines/run_kfold_reqo.py --hint-plans data/e2e/hint_plans_v2_job112 \
        --plan-representation ordered-binary-v1 --native-timeout-target budget-capped --seed 42 --device cpu \
        --out results/analysis/dependency_20261003/steering/job112/reqo_tcr_s42
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import sort_edge_index

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQO = os.path.join(os.path.dirname(ROOT), "baselines_src", "Reqo-on-PostgreSQL")
sys.path.insert(0, os.path.join(ROOT, "exp", "exp3_e2e"))
sys.path.append(REQO)

import common as ec                                            # noqa: E402  (exp3_e2e/common.py)
import run_kfold as rk                                         # noqa: E402
from Models.reqo_model import BiGG, Estimator                 # noqa: E402  (released Reqo code)
from Utils.loss import DataUncertaintyLoss, PairRankingLoss   # noqa: E402

RELEASE = {"batch_size": 128, "learning_rate": 0.001, "encoder_attention_heads": 1, "encoder_conv_layers": 3,
           "encoder_gnn_embedding_dim": 256, "encoder_gnn_dropout_rate": 0.1, "encoder_dirgnn_alpha": 0.3,
           "encoder_node_type_embedding_dim": 16, "encoder_column_embedding_dim": 8, "estimator_fcn_layers": 4,
           "estimator_estimation_embedding_dim": 512, "estimator_fcn_dropout_rate": 0.1, "pairrankingloss_margin": 3}
STATE = {"inputs": "tcr", "select": "val", "stats": None, "fz": None, "side": {}, "calls": 0, "epochs": []}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
class ReqoFeaturizer:
    """[type one-hot | log min-max cost | log min-max rows | tables | 8 values per column] per operator."""

    def __init__(self, stats, train_plans, db=None):
        self.types = stats["node_types"]["value_dict"]
        self.num_types = len(self.types)
        cost, rows = [], []
        for p in train_plans:
            for n in rk.walk(p["Plan"] if "Plan" in p else p):
                cost.append(float(n.get("Total Cost", 0.0)))
                rows.append(float(n.get("Plan Rows", 0.0)))
        cost, rows = np.log(np.asarray(cost) + 1), np.log(np.asarray(rows) + 1)
        self.lo, self.hi = [float(cost.min()), float(rows.min())], [float(cost.max()), float(rows.max())]
        self.db = db
        self.n_tables = len(db["table_columns_number"]) if db else 0
        self.n_columns = int(sum(db["table_columns_number"])) if db else 0
        self.node_length = self.num_types + 2 + self.n_tables + 8 * self.n_columns

    def rewrite(self, root):                                    # aliases -> tables, qualified columns (release)
        if self.db is None:
            return root
        from reqo_db import rewrite_plan
        return rewrite_plan(root, self.db)

    def node_vector(self, node):
        v = np.zeros(self.node_length, dtype=np.float32)
        t = self.types.get(node.get("Node Type", ""))
        if t is not None:
            v[t] = 1.0
        v[self.num_types] = (math.log(float(node.get("Total Cost", 0.0)) + 1) - self.lo[0]) / ((self.hi[0] - self.lo[0]) or 1.0)
        v[self.num_types + 1] = (math.log(float(node.get("Plan Rows", 0.0)) + 1) - self.lo[1]) / ((self.hi[1] - self.lo[1]) or 1.0)
        if self.db is not None:
            from reqo_db import node_text_encoding
            v[self.num_types + 2:] = node_text_encoding(node, self.db)
        return v


def make_featurizer(stats, a, train_plans=None):
    STATE["fz"] = ReqoFeaturizer(stats, train_plans, STATE["stats"])
    return STATE["fz"]


def to_graph(plan, fz, y_norm, qid, hint, ms, censored=False, parent_to_child=False):
    """Operators in post-order, child -> parent edges (``encoding_generate`` of the release)."""
    root = fz.rewrite(plan["Plan"] if "Plan" in plan else plan)
    xs, edges = [], []

    def rec(node):
        kids = [rec(ch) for ch in node.get("Plans", [])]
        cur = len(xs)
        xs.append(fz.node_vector(node))
        edges.extend([k, cur] for k in kids)
        return cur

    rec(root)
    ei = (sort_edge_index(torch.tensor(edges, dtype=torch.long).t().contiguous()) if edges
          else torch.zeros((2, 0), dtype=torch.long))
    d = Data(x=torch.from_numpy(np.stack(xs)).float(), edge_index=ei, y=torch.tensor([float(ms)], dtype=torch.float))
    d.qid, d.hint, d.ms = qid, hint, ms
    d.censored = torch.tensor([censored], dtype=torch.bool)
    return d


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ReqoNet(nn.Module):
    """``Models.reqo_model.Reqo`` of the release; the table / predicate part exists only with ``--reqo-inputs full``."""

    def __init__(self, fz: ReqoFeaturizer, cfg=RELEASE):
        super().__init__()
        self.num_types, self.n_tables, self.n_columns = fz.num_types, fz.n_tables, fz.n_columns
        self.table_columns_number = tuple(int(n) for n in fz.db["table_columns_number"]) if fz.db else ()
        d_type, d_col = cfg["encoder_node_type_embedding_dim"], cfg["encoder_column_embedding_dim"]
        self.node_type_enc = nn.Linear(self.num_types, d_type)
        self.column_layers = nn.ModuleList([nn.Linear(8, d_col, bias=False) for _ in range(self.n_columns)])
        enc = {k: v for k, v in cfg.items() if k.startswith("encoder_")}
        est = {k: v for k, v in cfg.items() if k.startswith("estimator_")}
        self.bigg = BiGG(enc, d_type + 2 + self.n_tables + self.n_tables * d_col)
        self.estimator = Estimator(est, cfg["encoder_gnn_embedding_dim"])
        self.label_range = None

    def encode_nodes(self, x):                                  # Feature_encoder.forward of the release
        parts = torch.split(x, [self.num_types, 2, self.n_tables, self.n_columns * 8], dim=1)
        out = [torch.relu(self.node_type_enc(parts[0])), parts[1]]
        if self.n_columns:
            col = parts[3].view(x.size(0), self.n_columns, 8).transpose(0, 1)
            w = torch.stack([layer.weight for layer in self.column_layers], dim=0)
            new = torch.relu(torch.bmm(col, w.transpose(1, 2)))
            chunks = torch.split(new.permute(1, 0, 2), self.table_columns_number, dim=1)
            out += [parts[2], torch.cat([t.max(dim=1).values for t in chunks], dim=1)]
        return torch.cat(out, dim=1)

    def forward(self, batch):
        return self.estimator(self.bigg(self.encode_nodes(batch.x), batch.edge_index, batch.batch))

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "label_range": self.label_range, "reqo_config": RELEASE,
                    "inputs": STATE["inputs"], "extra": extra}, path)


def selection_ratio(model, items, device, bs):
    """Sum of the runtimes of the selected candidates over the sum of the optimal ones (release: robustness_results[11])."""
    model.eval()
    score = []
    with torch.no_grad():
        for batch in DataLoader(items, batch_size=bs, shuffle=False):
            score.append(model(batch.to(device))[2].view(-1).cpu().numpy())
    score = np.concatenate(score)
    best, pick = {}, {}
    for d, s in zip(items, score):
        q = int(d.qid)
        best[q] = min(best.get(q, float("inf")), d.ms)
        if q not in pick or s < pick[q][0]:
            pick[q] = (s, d.ms)
    return sum(v[1] for v in pick.values()) / sum(best.values())


def train_reqo(train_items, a, in_dim, device, log):
    fz = STATE["fz"]
    STATE["calls"] += 1
    STATE["bs"] = a.bs                                          # test plans are batched as the training plans
    val_items = []
    if STATE["select"] == "val":
        qids = sorted({int(d.qid) for d in train_items})
        np.random.default_rng(1000 * a.seed + STATE["calls"]).shuffle(qids)
        held = set(qids[:max(1, int(round(0.15 * len(qids))))])
        val_items = [d for d in train_items if int(d.qid) in held]
        train_items = [d for d in train_items if int(d.qid) not in held]
    model = ReqoNet(fz).to(device)
    ms = np.asarray([d.ms for d in train_items], dtype=np.float64)
    lo, hi = float(np.log(ms + 1).min()), float(math.log(a.max_runtime + 1))
    model.label_range = [lo, hi]
    nll, rank = DataUncertaintyLoss(), PairRankingLoss(margin=RELEASE["pairrankingloss_margin"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loader = DataLoader(train_items, batch_size=a.bs, shuffle=True)
    history, best, best_state, stale = [], float("inf"), None, 0
    for epoch in range(a.epochs):
        if val_items and stale >= 20:                           # early stopping of the release
            break
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in loader:
            if batch.num_graphs < 2:                            # batch norm needs more than one plan
                continue
            batch = batch.to(device)
            opt.zero_grad()
            mu, va, iv = model(batch)
            y = batch.y.float().view(-1)
            loss = nll(mu, va.view(-1), y, hi, lo) + rank(iv, y, hi, lo)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
        history.append({"epoch": epoch, "loss": tot / max(n, 1), "time": time.time() - t0, "lr": a.lr})
        if val_items:
            ratio = selection_ratio(model, val_items, device, a.bs)
            history[-1]["val_ratio_to_optimal"] = ratio
            if ratio < best:
                best, stale = ratio, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        if epoch % 10 == 0 or epoch == a.epochs - 1:
            log(f"    epoch {epoch:3d} | loss {history[-1]['loss']:.4f} | {history[-1]['time']:.1f}s"
                + (f" | val ratio {history[-1]['val_ratio_to_optimal']:.2f} (best {best:.2f})" if val_items else ""))
    used = len(history) - 1
    if best_state is not None:
        model.load_state_dict(best_state)
        used = int(np.argmin([h["val_ratio_to_optimal"] for h in history]))
        log(f"    selected epoch {used} of {len(history)} (validation ratio to optimal {best:.2f}, {len(val_items)} plans)")
    STATE["epochs"].append({"selected_epoch": used, "epochs_run": len(history)})
    return model, {"epochs": len(history), "selected_epoch": used, "final_loss": history[-1]["loss"], "history": history}


@torch.no_grad()
def predict_reqo(model, items, device, max_runtime):
    model.eval()
    lo, hi = model.label_range
    score, latency = [], []
    for batch in DataLoader(items, batch_size=STATE.get("bs", RELEASE["batch_size"]), shuffle=False):   # as the release: the training batch size
        mu, _, iv = model(batch.to(device))
        score.append(iv.view(-1).cpu().numpy())
        latency.append((torch.exp(mu.view(-1) * (hi - lo) + lo) - 1).clamp(min=1e-3).cpu().numpy())
    score, latency = np.concatenate(score).astype(np.float64), np.concatenate(latency).astype(np.float64)
    for d, s, t in zip(items, score, latency):
        STATE["side"][(int(d.qid), int(d.hint))] = (float(s), float(t))
    return score                                                # run_kfold picks the candidate with the lowest value


rk.make_featurizer, rk.to_graph, rk.train_gnto, rk.predict_gnto = make_featurizer, to_graph, train_reqo, predict_reqo


def main() -> None:
    argv, out, hint_dir = [], None, None
    it = iter(sys.argv[1:])
    for tok in it:
        if tok == "--reqo-inputs":
            STATE["inputs"] = next(it)
        elif tok == "--reqo-select":
            STATE["select"] = next(it)
        elif tok == "--reqo-stats":
            STATE["stats_path"] = next(it)
        else:
            argv.append(tok)
            if tok == "--out":
                out = next(it)
                argv.append(out)
            elif tok == "--hint-plans":
                hint_dir = next(it)
                argv.append(hint_dir)
    if STATE["inputs"] not in ("tcr", "full") or out is None or hint_dir is None:
        raise SystemExit("usage: --reqo-inputs tcr|full [--reqo-stats DIR] --hint-plans DIR --out DIR <run_kfold.py arguments>")
    if STATE["inputs"] == "full":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reqo_db import load_stats
        STATE["stats"] = load_stats(STATE["stats_path"])
    sys.argv = [sys.argv[0], "--model", "gnto"] + argv
    rk.main()

    # run_kfold stored the integrated score as "pred"; keep it as "score" and report the latency head as "pred"
    packs = rk.load_packs(hint_dir)
    index = {p["query_id"]: i for i, p in enumerate(packs)}
    path = os.path.join(out, "predictions.csv")
    pred = pd.read_csv(path)
    side = [STATE["side"][(index[q], int(h))] for q, h in zip(pred["query_id"], pred["hint_id"])]
    assert np.allclose(pred["pred"].values, [s for s, _ in side])
    pred["score"], pred["pred"] = [s for s, _ in side], [t for _, t in side]
    pred.to_csv(path, index=False)
    spath = os.path.join(out, "summary.json")
    s = json.load(open(spath))
    s["prediction_quality"] = ec.prediction_quality(pred)
    s.update(model="reqo", reqo_inputs=STATE["inputs"], reqo_config=RELEASE, selection_score="integrated score IC",
             reqo_epoch_selection=STATE["select"], reqo_epochs=STATE["epochs"],
             reqo_source="https://github.com/BaomingChang/Reqo-on-PostgreSQL")
    if STATE["inputs"] == "full":
        from reqo_db import STATS
        s["reqo_text_encoding"] = dict(STATS, column_kinds=STATE["stats"]["column_kinds"], catalog=STATE["stats"]["catalog"])
    json.dump(s, open(spath, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
