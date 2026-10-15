"""Reqo / BiGG (Chang, Kamali, Kantere; PACMMOD 4(1), SIGMOD 2026) on the cross-database benchmark.

The released code (https://github.com/BaomingChang/Reqo-on-PostgreSQL, cloned to
``../baselines_src/Reqo-on-PostgreSQL``) is imported unchanged: ``Models.reqo_model.BiGG`` (direction-specific
TransformerConv layers combined by a learned weight, batch normalisation, GRU readout over the operators in
post-order), ``Models.reqo_model.Estimator`` and the losses of ``Utils.loss``.  Two arms:

``--arm bigg``  BiGG as the dependency encoder *inside the GNTO harness*: node encoder, head, loss, optimiser,
                schedule and data are those of ``analysis/dependency_study/run_crossdb_dep.py``; only the encoder and
                the readout are Reqo's.  This is an intervention of the dependency study (how operators are related).
``--arm reqo``  The released network and objective with the inputs that exist on an unseen database: operator type,
                estimated cost and rows (Reqo's table and predicate inputs are database-specific and are dropped).
                Type embedding 16, BiGG 256 x 3, one head, estimator 512, uncertainty loss + pairwise ranking loss
                (margin 3), Adam 1e-3, batch 128 - the configuration of the released ``main.py``.  Cost, rows and
                labels are log min-max normalised as in the release, with the ranges of the *training* plans.

Deviations from the release, common to every model of this benchmark: train-only statistics, 100 epochs, the last
epoch is evaluated (the release selects the epoch on the test fold).

    python analysis/baselines/run_crossdb_reqo.py --arm bigg --features nocost --seed 42 --epochs 100 --select last \
        --device cpu --out results/analysis/dependency_20261003/crossdb/nocost_bigg_L3_true_s42
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import sort_edge_index

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQO = os.path.join(os.path.dirname(ROOT), "baselines_src", "Reqo-on-PostgreSQL")
for p in (os.path.join(ROOT, "exp", "exp2_dace"), os.path.join(ROOT, "exp"), ROOT, os.path.join(ROOT, "analysis", "dependency_study")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.append(REQO)

import common as c                                             # noqa: E402  (exp2_dace/common.py)
import run_gnto as rg                                          # noqa: E402  (frozen training loop)
from models.dace import count_nodes                           # noqa: E402
from models.gnto import GNTO, get_preset                      # noqa: E402
from run_crossdb_dep import SubsetFeaturizer, DerivedFeaturizer, plan_depth      # noqa: E402
from Models.reqo_model import BiGG, Estimator                 # noqa: E402  (released Reqo code)
from Utils.loss import DataUncertaintyLoss, PairRankingLoss   # noqa: E402

RELEASE = {"batch_size": 128, "learning_rate": 0.001, "encoder_attention_heads": 1, "encoder_conv_layers": 3,
           "encoder_gnn_embedding_dim": 256, "encoder_gnn_dropout_rate": 0.1, "encoder_dirgnn_alpha": 0.3,
           "encoder_node_type_embedding_dim": 16, "estimator_fcn_layers": 4, "estimator_estimation_embedding_dim": 512,
           "estimator_fcn_dropout_rate": 0.1, "pairrankingloss_margin": 3}        # main.py of the release


def postorder_graph(root, vec):
    """Operators in post-order with child -> parent edges, as ``encoding_generate`` of the release."""
    xs, edges = [], []

    def rec(node):
        kids = [rec(ch) for ch in node.get("Plans", [])]
        cur = len(xs)
        xs.append(vec(node))
        edges.extend([k, cur] for k in kids)
        return cur

    rec(root)
    x = torch.from_numpy(np.stack(xs)).float()
    ei = (sort_edge_index(torch.tensor(edges, dtype=torch.long).t().contiguous()) if edges
          else torch.zeros((2, 0), dtype=torch.long))
    return x, ei


def build(plans, vec, db_id):
    out = []
    for i, p in enumerate(plans):
        root = p["Plan"]
        x, ei = postorder_graph(root, vec)
        d = Data(x=x, edge_index=ei, y=torch.tensor([float(root.get("Actual Total Time", 0.0))], dtype=torch.float))
        d.n_nodes, d.db_id, d.plan_id = count_nodes(root), db_id, i
        out.append(d)
    return out


class BiGGInGNTO(nn.Module):
    """GNTO's node encoder and head around Reqo's BiGG encoder + GRU readout."""

    def __init__(self, cfg, layers: int, heads: int, width: int):
        super().__init__()
        base = GNTO.from_config(cfg)
        self.node_encoder, self.head = base.node_encoder, base.head
        params = {"encoder_attention_heads": heads, "encoder_gnn_dropout_rate": RELEASE["encoder_gnn_dropout_rate"],
                  "encoder_gnn_embedding_dim": width, "encoder_conv_layers": layers,
                  "encoder_dirgnn_alpha": RELEASE["encoder_dirgnn_alpha"]}
        self.bigg = BiGG(params, self.node_encoder.out_dim)
        assert width == self.head.in_dim, (width, self.head.in_dim)

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
        torch.save({"state_dict": self.state_dict(), "extra": extra}, path)


class ReqoTransfer(nn.Module):
    """The released Reqo network without its database-specific inputs (tables, predicates)."""

    def __init__(self, num_types: int, cfg=RELEASE):
        super().__init__()
        self.num_types = num_types
        self.node_type_enc = nn.Linear(num_types, cfg["encoder_node_type_embedding_dim"])   # as Feature_encoder
        enc = {k: v for k, v in cfg.items() if k.startswith("encoder_")}
        est = {k: v for k, v in cfg.items() if k.startswith("estimator_")}
        self.bigg = BiGG(enc, cfg["encoder_node_type_embedding_dim"] + 2)
        self.estimator = Estimator(est, cfg["encoder_gnn_embedding_dim"])

    def forward(self, batch):
        t = torch.relu(self.node_type_enc(batch.x[:, :self.num_types]))
        x = torch.cat([t, batch.x[:, self.num_types:]], dim=1)
        return self.estimator(self.bigg(x, batch.edge_index, batch.batch))       # mean, variance, integrated score

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


def reqo_node_vector(fz, lo, hi):
    """[operator-type one-hot | log min-max cost | log min-max rows] (``Node_info_extract`` of the release)."""
    def vec(node):
        v = fz.node_vector(node)[:fz.num_types]
        s = [(math.log(float(node["Total Cost"]) + 1) - lo[0]) / (hi[0] - lo[0]),
             (math.log(float(node["Plan Rows"]) + 1) - lo[1]) / (hi[1] - lo[1])]
        return np.concatenate([v, np.asarray(s, dtype=v.dtype)])
    return vec


def reqo_predict(model, items, device, lo, hi, bs=1024):
    model.eval()
    preds, score, targets = [], [], []
    with torch.no_grad():
        for batch in DataLoader(items, batch_size=bs, shuffle=False):
            batch = batch.to(device)
            mu, _, iv = model(batch)
            preds.append((torch.exp(mu.view(-1) * (hi - lo) + lo) - 1).clamp(min=1e-3).cpu().numpy())
            score.append(iv.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float64), np.concatenate(targets).astype(np.float64), np.concatenate(score)


def reqo_train(model, train_list, val_list, device, a, out_dir, log, lo, hi):
    """Training loop of the release (train.py) without its test-fold epoch selection."""
    loader = DataLoader(train_list, batch_size=a.bs, shuffle=True)
    nll, rank = DataUncertaintyLoss(), PairRankingLoss(margin=RELEASE["pairrankingloss_margin"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss,time,val_q50,val_q90,val_q_mean\n")
    for epoch in range(a.epochs):
        t0, total, n = time.time(), 0.0, 0
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            mu, va, iv = model(batch)
            y = batch.y.float().view(-1)
            loss = nll(mu, va.view(-1), y, hi, lo) + rank(iv, y, hi, lo)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        vp, vt, _ = reqo_predict(model, val_list, device, lo, hi)
        vq = c.level_stats(c.qerror(vp, vt))
        with open(log_path, "a") as f:
            f.write(f"{epoch},{total / n},{time.time() - t0},{vq['q50']},{vq['q90']},{vq['q_mean']}\n")
        log(f"[Reqo] epoch {epoch:2d} | loss {total / n:.4f} | {time.time() - t0:.1f}s | val Q50 {vq['q50']:.3f} "
            f"Q90 {vq['q90']:.3f} mean {vq['q_mean']:.3f}")
    ckpt = os.path.join(out_dir, "best_model.pt")
    torch.save({"state_dict": model.state_dict(), "label_log_range": [lo, hi]}, ckpt)
    return {"best_epoch": a.epochs - 1, "best_metric": vq["q_mean"], "select": "last", "epochs": a.epochs, "ckpt": ckpt,
            "trained_here": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["bigg", "reqo"])
    ap.add_argument("--features", default="full", choices=["full", "nocost", "resid"])   # resid: own cost, no root
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--width", type=int, default=64, help="BiGG width of the bigg arm (the reqo arm uses the release: 256)")
    c.add_common_args(ap, default_bs=128)
    a = ap.parse_args()
    if not a.out:
        ap.error("--out is required")
    if a.arm == "reqo" and a.features != "full":
        ap.error("the reqo arm uses the inputs of the release: type, cost and rows")
    a.no_cache = True
    out_dir = c.finalize_args(a, a.arm)
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        raise SystemExit(f"completed output exists: {out_dir}")
    log = c.Logger(os.path.join(out_dir, "log.txt"))
    device = torch.device(a.device)

    bench = c.load_benchmark(a, log)
    c.write_benchmark_manifest(out_dir, bench)
    plans, dbs = bench["plans"], bench["train_dbs"] + bench["test_dbs"]
    sizes = {db: len(plans[db]) for db in plans}
    train_pos = c.index_positions(bench["train_index"], bench["train_dbs"], sizes)
    val_pos = c.index_positions(bench["val_index"], bench["train_dbs"], sizes)
    pool_plans = [p for db in bench["train_dbs"] for p in plans[db]]
    t0 = time.time()
    study = {"arm": a.arm, "features": a.features, "conv": a.arm, "layers": a.layers, "heads": a.heads,
             "virtual_node": False, "topology": "true", "source": "https://github.com/BaomingChang/Reqo-on-PostgreSQL"}

    if a.arm == "bigg":
        if a.features == "resid":
            train_plans = [bench["plans"][db][i] for db, i in bench["train_index"]]
            fz = DerivedFeaturizer(bench["fz"], a.features, train_plans)
        else:
            fz = SubsetFeaturizer(bench["fz"], a.features)
        cfg = get_preset("dace_zero_shot").override(node__in_dim=fz.node_length).override(
            "graph.conv=none", f"graph.num_layers={a.layers}", "graph.heads=1")       # node encoder + head only
        vec, extra_cfg = fz.node_vector, {"gnto_config_for_node_encoder_and_head": cfg.to_dict(), "bigg_width": a.width}
    else:
        def walk(n):
            yield n
            for ch in n.get("Plans", []):
                yield from walk(ch)
        cost = np.log(np.asarray([n["Total Cost"] for i in train_pos for n in walk(pool_plans[i]["Plan"])], dtype=np.float64) + 1)
        rows = np.log(np.asarray([n["Plan Rows"] for i in train_pos for n in walk(pool_plans[i]["Plan"])], dtype=np.float64) + 1)
        lo_x, hi_x = [float(cost.min()), float(rows.min())], [float(cost.max()), float(rows.max())]
        vec = reqo_node_vector(bench["fz"], lo_x, hi_x)
        labels = np.log(np.asarray([pool_plans[i]["Plan"]["Actual Total Time"] for i in train_pos], dtype=np.float64) + 1)
        lo_y, hi_y = float(labels.min()), float(labels.max())
        extra_cfg = {"reqo_config": RELEASE, "input_log_range": [lo_x, hi_x], "label_log_range": [lo_y, hi_y]}
    c.write_config(out_dir, a, model=a.arm, study=study, **extra_cfg)

    per_db = {db: build(plans[db], vec, i) for i, db in enumerate(dbs)}
    pool = [d for db in bench["train_dbs"] for d in per_db[db]]
    train_list, val_list = [pool[i] for i in train_pos], [pool[i] for i in val_pos]
    timing = {"featurize_s": time.time() - t0}

    c.set_seed(a.seed)
    if a.arm == "bigg":
        model = BiGGInGNTO(cfg, a.layers, a.heads, a.width).to(device)
    else:
        model = ReqoTransfer(bench["fz"].num_types).to(device)
    params = {"total": model.num_parameters(), "trainable": model.num_parameters(trainable_only=True)}
    log(f"{a.arm}: features={a.features} -> {out_dir}\nparams: {params['total']:,}")
    t0 = time.time()
    if a.arm == "bigg":
        train_info = rg.train(model, train_list, val_list, device, a, out_dir, log, select=a.select)
        model.load_state_dict(torch.load(train_info["ckpt"], map_location=device, weights_only=False)["state_dict"])
    else:
        train_info = reqo_train(model, train_list, val_list, device, a, out_dir, log, lo_y, hi_y)
    timing["train_s"] = time.time() - t0

    import pandas as pd
    results = {}
    for db in bench["test_dbs"]:
        if a.arm == "bigg":
            preds, targets = rg.predict(model, per_db[db], device)
            score = None
        else:
            preds, targets, score = reqo_predict(model, per_db[db], device, lo_y, hi_y)
        results[db] = c.db_metrics(preds, targets)
        path = c.write_predictions(out_dir, db, preds, targets, [d.n_nodes for d in per_db[db]])
        df = pd.read_csv(path)
        df["depth"] = [plan_depth(p["Plan"]) for p in plans[db]]
        if score is not None:
            df["reqo_score"] = score
        df.to_csv(path, index=False)
        log(f"{db:16s} n={results[db]['n']:5d} {c.format_levels(results[db])}")
    s = c.write_summary(out_dir, a.arm, params, bench, results, train_info, timing, extra={"study": study, **extra_cfg})
    log(f"average over test DBs: {c.format_levels(s['avg'])}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
