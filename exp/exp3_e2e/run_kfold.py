"""Exp3 steerer: a cost model trained on the hint-plan benchmark itself, k-fold by query.

Why this exists.  The Exp1 GNTO checkpoint was trained on a 7-table IMDB vocabulary with
per-query sample bitmaps, neither of which covers these hint plans.

This runner removes both problems:

* **schema-agnostic features** - one-hot node type + RobustScaler-scaled ``Total Cost`` /
  ``Plan Rows`` (``models.dace.NodeFeaturizer``, the Exp2 featurisation).  No table, column
  or join ids, so nothing can be out of vocabulary.
* **in-distribution training** - the benchmark itself is the training set: 112 queries x up
  to 49 hint sets with measured runtimes.  The 35 JOB templates are split into ``--folds``
  folds (``--group template``, the default): a fold's queries are scored by a model that saw
  neither them nor any query of the same template.  ``--group query`` splits single queries
  and is optimistic - a test query's siblings (13a/13b/13c for 13d) sit in training.

``--rank-weight`` adds GNTO's pairwise ranking loss over pairs *within one query* - what
steering actually needs (ordering a query's variants), as opposed to absolute accuracy.
With it the batches are query-grouped (``--queries-per-batch``).  Compare the predictions in
``--rank-space log`` (default): the head is a sigmoid and most runtimes sit far below the
normalisation ceiling, so in linear space a query's candidates differ by ~1e-3 and the
score differences have a compressed dynamic range; log-space differences are ratios.

    python exp/exp3_e2e/run_kfold.py                                    # GNTO, 5 folds
    python exp/exp3_e2e/run_kfold.py --rank-weight 1 --tag ranklog      # + pairwise ranking (log space)
    python exp/exp3_e2e/run_kfold.py --model bao --tag kfold            # Bao on the same folds
    python exp/exp3_e2e/run_kfold.py --model limao --tag tpl            # LIMAO's modular tree-CNN on the same folds
    python exp/exp3_e2e/run_kfold.py --limit-queries 8 --folds 2 --epochs 2 --tag smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GNTO_PATH, PG_DEFAULT_HINT, Logger, add_common_args, candidates, load_packs,  # noqa: E402
                    pack_info, set_seed, steerer_dir, write_config, write_steerer)
from adapters.dace_adapter import plan_to_graph, q_error_loss  # noqa: E402
from models.dace import NodeFeaturizer  # noqa: E402
from models.gnto import GNTO, PairwiseRankingLoss, get_preset  # noqa: E402
from protocol import plan_fingerprint, template_key, validate_complete_pack, assign_folds
from _shared import file_sha256, object_sha256
from adapters.plan_representation import REPRESENTATIONS, represent_plan

PLAN_NUMERICS = ("Total Cost", "Plan Rows")


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def walk(node: Dict[str, Any]):
    yield node
    for child in node.get("Plans", []):
        yield from walk(child)


def build_stats(plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``models.dace`` statistics fitted on *these* plans: node-type vocabulary + a
    RobustScaler (median / IQR) per numeric channel.  Fitted on the training fold only."""
    types, values = set(), {k: [] for k in PLAN_NUMERICS}
    for p in plans:
        for n in walk(p["Plan"] if "Plan" in p else p):
            types.add(n.get("Node Type", ""))
            for k in PLAN_NUMERICS:
                values[k].append(float(n.get(k, 0.0)))
    stats: Dict[str, Any] = {"node_types": {"value_dict": {t: i for i, t in enumerate(sorted(types))}}}
    for k in PLAN_NUMERICS:
        v = np.asarray(values[k], dtype=np.float64)
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        stats[k] = {"center": float(q50), "scale": float(q75 - q25) or 1.0}
    return stats


class SubsetFeaturizer:
    """Wrap ``NodeFeaturizer`` and keep only some channels (feature ablation).

    ``drop`` names numeric channels to remove: ``rows`` (Plan Rows), ``cost`` (Total Cost).
    Both carry PostgreSQL's cardinality estimate - rows directly, cost through the cost model -
    so dropping one alone need not remove the estimate from the input."""

    CHANNELS = {"cost": "Total Cost", "rows": "Plan Rows"}

    def __init__(self, fz: NodeFeaturizer, drop: List[str]):
        self.fz = fz
        unknown = [d for d in drop if d not in self.CHANNELS]
        if unknown:
            raise ValueError(f"unknown feature(s) {unknown}; choose from {list(self.CHANNELS)}")
        numeric = list(PLAN_NUMERICS)                       # order written by NodeFeaturizer
        dropped = {fz.num_types + numeric.index(self.CHANNELS[d]) for d in drop}
        self.keep = [i for i in range(fz.node_length) if i not in dropped]
        self.num_types = fz.num_types
        self.node_length = len(self.keep)

    def node_vector(self, node):
        return self.fz.node_vector(node)[self.keep]


def to_graph(plan: Dict[str, Any], fz: NodeFeaturizer, y_norm: float, qid: int, hint: int, ms: float,
             censored=False, parent_to_child=False) -> Data:
    root = plan["Plan"] if "Plan" in plan else plan
    if hasattr(fz, "bind"):                  # PGFeaturizer resolves alias.column per plan
        fz.bind(root)
    x, edge_index = plan_to_graph(root, fz)
    if parent_to_child:
        # The DACE adapter emits both directions in DFS order. Binary tree
        # convolution requires each original parent->child edge exactly once.
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        if any(len(n.get("Plans", [])) > 2 for n in walk(root)):
            raise ValueError("controlled binary tree convolution requires binary plans; do not drop later children")
    d = Data(x=x, edge_index=edge_index, y=torch.tensor([y_norm], dtype=torch.float))
    d.qid, d.hint, d.ms = qid, hint, ms
    d.censored = torch.tensor([censored], dtype=torch.bool)
    return d


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def query_grouped_batches(items: List[Data], queries_per_batch: int, rng: np.random.Generator) -> List[List[Data]]:
    """Batches that hold every candidate of a few queries, so ranking pairs exist."""
    by_q: Dict[int, List[Data]] = {}
    for d in items:
        by_q.setdefault(d.qid, []).append(d)
    qids = list(by_q)
    rng.shuffle(qids)
    return [[d for q in qids[i:i + queries_per_batch] for d in by_q[q]]
            for i in range(0, len(qids), queries_per_batch)]


def train_gnto(train_items: List[Data], a, in_dim: int, device, log) -> Tuple[GNTO, Dict[str, Any]]:
    cfg = get_preset(a.preset, f"node.in_dim={in_dim}", *a.set)
    model = GNTO.from_config(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    ranker = PairwiseRankingLoss(kind="logistic", min_gap=np.log(1 + a.rank_min_gap)) if a.rank_weight else None
    rng = np.random.default_rng(a.seed)
    history = []
    for epoch in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        if ranker is not None or a.grouped_batches:
            batches = query_grouped_batches(train_items, a.queries_per_batch, rng)
            loader = (DataLoader(b, batch_size=len(b), shuffle=False).__iter__().__next__() for b in batches)
        else:
            loader = DataLoader(train_items, batch_size=a.bs, shuffle=True)
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch).view(-1)
            target = batch.y.view(-1).clamp_min(1e-7)
            delta = torch.log(pred.clamp_min(1e-7)) - torch.log(target)
            # Censored runtimes are lower bounds, never exact latency labels.
            loss = torch.where(batch.censored.view(-1), torch.relu(-delta), delta.abs()).mean()
            if ranker is not None:
                groups = torch.as_tensor(batch.qid, device=device).view(-1)
                log_ms = torch.log(torch.as_tensor(batch.ms, dtype=torch.float, device=device).view(-1).clamp_min(1e-3))
                score = torch.log(pred.clamp_min(1e-6)) if a.rank_space == "log" else pred
                # A censored plan may be the slower member of a pair, but never
                # the known-faster member. Two censored plans cannot be ordered.
                gap = log_ms[None, :] - log_ms[:, None]
                valid = (groups[:, None] == groups[None, :]) & (gap > ranker.min_gap)
                valid &= ~batch.censored.view(-1, 1)
                diff = score[None, :] - score[:, None]
                rank_loss = torch.nn.functional.softplus(-diff[valid]).mean() if valid.any() else score.sum() * 0
                loss = loss + a.rank_weight * rank_loss
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        avg = tot / max(nb, 1)
        sched.step(avg)
        history.append({"epoch": epoch, "loss": avg, "time": time.time() - t0, "lr": opt.param_groups[0]["lr"]})
        if epoch % 10 == 0 or epoch == a.epochs - 1:
            log(f"    epoch {epoch:3d} | loss {avg:.4f} | {history[-1]['time']:.1f}s | lr {history[-1]['lr']:.1e}")
    return model, {"epochs": a.epochs, "final_loss": history[-1]["loss"], "history": history}


@torch.no_grad()
def predict_gnto(model: GNTO, items: List[Data], device, max_runtime: float) -> np.ndarray:
    model.eval()
    out = []
    for batch in DataLoader(items, batch_size=512, shuffle=False):
        out.append(model.predict(batch.to(device)).cpu().numpy() * max_runtime)
    return np.concatenate(out) if out else np.zeros(0)


def run_bao_fold(train_rows, test_rows, a, log, checkpoint=None) -> np.ndarray:
    """Bao's tree-CNN on the same fold (its own 9-d schema-agnostic features)."""
    from adapters.bao_adapter import BaoCostPredictor, _collapse_bitmap_subtree

    def prep(rows):
        plans = []
        for _, _, plan, _ in rows:
            p = copy.deepcopy(plan)
            if a.plan_representation == "native":
                _collapse_bitmap_subtree(p["Plan"] if "Plan" in p else p)
            plans.append(p)
        return plans, np.array([ms for *_, ms in rows], dtype=np.float64)

    tp, ty = prep(train_rows)
    pred = BaoCostPredictor(verbose=False, epochs=a.epochs,
                            extended_tree=a.plan_representation == "ordered-binary-v1")
    pred.fit(tp, ty)
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        pred.save(checkpoint)
    sp, _ = prep(test_rows)
    return np.asarray(pred.predict(sp), dtype=np.float64).reshape(-1)


def run_limao_folds(folds: Dict[int, Tuple[list, list]], a, out: str, log) -> Dict[int, Tuple[np.ndarray, Dict[str, Any]]]:
    """LIMAO's modular tree-CNN (``LIMAOLifeLongRLDB/bao_server``) on the same folds.  Its code
    shares module names with Bao / QueryFormer, so every fold trains in its own process
    (``limao_worker.py``), ``--fold-jobs`` of them at a time.  Returns per fold the predictions
    in test-row order and the worker's report."""
    work = os.path.join(out, "limao_folds")
    os.makedirs(work, exist_ok=True)
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limao_worker.py")
    jobs, order = {}, {}
    for f, (tr, te) in folds.items():
        by_q: Dict[int, List[int]] = {}
        for i, r in enumerate(te):
            by_q.setdefault(r[0], []).append(i)
        order[f] = list(by_q.values())
        inp, outp = os.path.join(work, f"fold{f}_in.pkl"), os.path.join(work, f"fold{f}_out.json")
        pickle.dump({"train_plans": [r[2] for r in tr], "train_ms": [r[3] for r in tr],
                     "test_groups": [[te[i][2] for i in g] for g in order[f]]}, open(inp, "wb"))
        jobs[f] = ([sys.executable, worker, "--input", inp, "--output", outp, "--epochs", str(a.epochs),
                    "--seed", str(a.seed + f), "--threads", str(a.fold_threads),
                    "--plan-representation", a.plan_representation], outp, os.path.join(work, f"fold{f}.log"))
    pending, running, results = sorted(jobs), {}, {}
    if a.reuse_folds:                                    # predictions of an earlier run (same folds, same seed)
        for f in list(pending):
            if os.path.exists(jobs[f][1]):
                rep = json.load(open(jobs[f][1]))
                p = np.zeros(len(folds[f][1]))
                for g, vals in zip(order[f], rep.pop(f"preds_{a.limao_assign}")):
                    p[g] = vals
                rep.pop("preds_first", None), rep.pop("preds_per_plan", None)
                rep["assign"] = a.limao_assign
                results[f] = (p, rep)
                pending.remove(f)
                log(f"  fold {f}: reusing {jobs[f][1]}")
    while pending or running:
        while pending and len(running) < a.fold_jobs:
            f = pending.pop(0)
            cmd, _, logf = jobs[f]
            running[f] = (subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT), time.time())
            log(f"  fold {f}: LIMAO worker started ({len(folds[f][0]):,} train plans)")
        for f in list(running):
            proc, t0 = running[f]
            if proc.poll() is None:
                continue
            del running[f]
            if proc.returncode != 0:
                raise SystemExit(f"LIMAO worker for fold {f} failed, see {jobs[f][2]}")
            rep = json.load(open(jobs[f][1]))
            both = {k: rep.pop(f"preds_{k}") for k in ("first", "per_plan")}
            p = np.zeros(len(folds[f][1]))
            for g, vals in zip(order[f], both[a.limao_assign]):
                p[g] = vals
            rep["assign"] = a.limao_assign
            results[f] = (p, rep)
            log(f"  fold {f} done in {time.time() - t0:.0f}s ({rep['epochs_run']} epochs, modules used in training {rep['module_usage_train']})")
            os.remove(jobs[f][0][3])                     # the input pickle; the fold's predictions stay for --reuse-folds
        time.sleep(2)
    return results


_CATALOG = {}


def make_featurizer(stats, a, train_plans=None):
    if a.features == "pg":
        from adapters.pg_features import CatalogStats, PGFeaturizer
        if a.catalog not in _CATALOG:
            _CATALOG[a.catalog] = CatalogStats(a.catalog)
        return PGFeaturizer(_CATALOG[a.catalog]).fit(train_plans)
    fz = NodeFeaturizer(stats)
    drop = [d.strip() for d in a.drop_features.split(",") if d.strip()]
    return SubsetFeaturizer(fz, drop) if drop else fz


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gnto", choices=["gnto", "bao", "limao"])
    ap.add_argument("--fold-jobs", type=int, default=5, help="limao: folds trained in parallel (one process each)")
    ap.add_argument("--fold-threads", type=int, default=2, help="limao: CPU threads per fold process")
    ap.add_argument("--limao-assign", default="per_plan", choices=["per_plan", "first"],
                    help="limao: modules of a candidate chosen for that candidate (as in training) or taken from the "
                         "first candidate of the query (as LIMAO's server does)")
    ap.add_argument("--reuse-folds", action="store_true", help="limao: reuse <out>/limao_folds/fold*_out.json of an earlier run")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group", default="template", choices=["template", "query"],
                    help="unit of the fold split: single queries, or JOB templates (13a/13b/13c/13d stay together - "
                         "otherwise a test query's siblings with the same shape sit in training)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42, help="fold assignment seed, independent of training seed")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--preset", default="dace_zero_shot", help="GNTO preset (schema-agnostic features)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--rank-weight", type=float, default=0.0, help="weight of the within-query pairwise ranking loss")
    ap.add_argument("--rank-min-gap", type=float, default=0.1, help="ignore pairs closer than this relative runtime gap")
    ap.add_argument("--rank-space", default="log", choices=["log", "linear"],
                    help="compare predictions in log space (scale-free) or as the raw sigmoid output")
    ap.add_argument("--queries-per-batch", type=int, default=3, help="ranking mode: queries per batch")
    ap.add_argument("--grouped-batches", action="store_true", help="use identical query-grouped batches with or without ranking")
    ap.add_argument("--candidate-policy", choices=["complete", "legacy-successful"], default="complete")
    ap.add_argument("--native-timeout-target", choices=["reject", "budget-capped"], default="reject",
                    help="explicit sensitivity policy for native Bao/LIMAO: predict capped latency, not uncensored runtime")
    ap.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plan-representation", choices=REPRESENTATIONS, default="native",
                    help="same ordered binary topology for all controlled encoders; explicit extended adapter for Bao/LIMAO")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-runtime", type=float, default=32000.0, help="target scale in ms (the statement timeout)")
    ap.add_argument("--features", default="dace", choices=["dace", "pg"],
                    help="dace: node type + Total Cost + Plan Rows; pg: what PostgreSQL's optimizer consults "
                         "(catalog stats, predicates, selectivity inputs, operator choices - adapters/pg_features.py)")
    ap.add_argument("--catalog", default=os.path.join(GNTO_PATH, "data", "e2e", "pg_catalog_snapshot.json"),
                    help="pg_catalog snapshot used by --features pg")
    ap.add_argument("--drop-features", default="", help="comma list of numeric channels to remove: rows, cost")
    ap.add_argument("--extra-train", default="",
                    help="directory of extra q_*.json packs (same format, with a 'template' field) added to EVERY "
                         "training fold and never scored - more query templates for training; their table sets "
                         "must not coincide with any benchmark template")
    ap.add_argument("--limit-queries", type=int, help="use only the first N queries (smoke test)")
    ap.add_argument("--train-all", action="store_true",
                    help="no folds: train on every query and save the model + its featurizer stats "
                         "(for scoring plans of queries outside the benchmark)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common_args(ap)
    a = ap.parse_args()
    if a.candidate_policy == "complete" and (a.train_all or a.extra_train):
        ap.error("complete protocol currently supports held-out folds only; train-all/extra-train need separate validation")
    if a.plan_representation != "native" and (a.train_all or a.extra_train):
        ap.error("versioned representation is supported only for held-out benchmark folds")
    torch.set_num_threads(a.threads)

    name = f"{a.model}_kfold"
    out = steerer_dir(a, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        raise SystemExit(f"completed output exists: {out}; choose a new --out or --tag")
    log = Logger(os.path.join(out, "log.txt"))
    set_seed(a.seed)
    device = torch.device(a.device)

    packs = load_packs(a.hint_plans)
    if a.limit_queries:
        packs = packs[:a.limit_queries]
    data_manifest = {"candidate_policy": a.candidate_policy, "deduplicate": a.deduplicate,
                     "plan_representation": a.plan_representation,
                     "packs": {p["query_id"]: file_sha256(os.path.join(a.hint_plans, f"q_{p['query_id']}.json")) for p in packs}}
    if a.features == "pg":
        data_manifest["catalog_sha256"] = file_sha256(a.catalog)
    censored = {}
    for qi, pack in enumerate(packs):
        if a.candidate_policy == "complete":
            validate_complete_pack(pack)
        if a.deduplicate:
            unique = {}
            for row in sorted(pack["results"], key=lambda r: int(r["hint_id"])):
                if "plan" in row:
                    unique.setdefault(plan_fingerprint(row["plan"]), row)
            pack["results"] = list(unique.values())
        for row in pack["results"]:
            censored[(qi, int(row["hint_id"]))] = bool(row.get("censored", False))
    if a.model != "gnto" and any(censored.values()) and a.native_timeout_target != "budget-capped":
        raise SystemExit("native Bao/LIMAO losses do not support censored labels; use controlled tree-conv or implement an explicit censoring policy")
    with open(os.path.join(out, "data_manifest.json"), "w") as f:
        json.dump(data_manifest, f, indent=2)
    rows: List[Tuple[int, str, Dict[str, Any], float]] = []      # (query index, hint, plan, ms)
    qindex = {}
    for i, pack in enumerate(packs):
        qindex[pack["query_id"]] = i
        for h, plan, ms in candidates(pack):
            rows.append((i, h, represent_plan(plan, a.plan_representation), ms))
    extra_rows: List[Tuple[int, str, Dict[str, Any], float]] = []
    if a.extra_train:
        extra_packs = load_packs(a.extra_train)
        for j, pack in enumerate(extra_packs):
            for h, plan, ms in candidates(pack):
                extra_rows.append((-1 - j, h, plan, ms))              # negative ids: never in a test fold
        log(f"extra training data: {len(extra_packs)} queries, "
            f"{len({p.get('template', p['query_id']) for p in extra_packs})} templates, {len(extra_rows):,} plans "
            f"from {a.extra_train}")
    write_config(out, a, steerer=name, n_queries=len(packs), n_plans=len(rows), n_extra_plans=len(extra_rows))
    log(f"{name}: {len(packs)} queries, {len(rows):,} plans, {a.folds} folds -> {out}")

    if a.train_all:
        stats = build_stats([r[2] for r in rows])
        fz = make_featurizer(stats, a, [r[2] for r in rows])
        items = [to_graph(pl, fz, min(ms / a.max_runtime, 1.0), q, h, ms) for q, h, pl, ms in rows]
        model, info = train_gnto(items, a, fz.node_length, device, log)
        model.save(os.path.join(out, "model.pt"), extra={"stats": stats, "max_runtime": a.max_runtime,
                                                         "trained_on": "all", "n_plans": len(rows)})
        json.dump(stats, open(os.path.join(out, "featurizer_stats.json"), "w"), indent=2)
        pd.DataFrame(info["history"]).to_csv(os.path.join(out, "training_log.csv"), index=False)
        log(f"trained on all {len(rows):,} plans -> {out}/model.pt (in_dim {fz.node_length}, "
            f"{fz.num_types} node types, final loss {info['final_loss']:.4f})")
        return

    fold_of = assign_folds(packs, a.group, a.folds, a.split_seed)
    with open(os.path.join(out, "folds.json"), "w") as f:
        json.dump({"split_seed": a.split_seed, "training_seed": a.seed, "group": a.group,
                   "queries": {p["query_id"]: {"fold": fold_of[i], "template": p.get("template", template_key(p["query_id"]))}
                               for i, p in enumerate(packs)}}, f, indent=2)
    preds = np.zeros(len(rows))
    fold_info = []
    params: Dict[str, int] = {}
    limao = {}
    if a.model == "limao":
        limao = run_limao_folds({f: ([r for r in rows if fold_of[r[0]] != f] + extra_rows,
                                     [r for r in rows if fold_of[r[0]] == f]) for f in range(a.folds)}, a, out, log)
    for f in range(a.folds):
        set_seed(a.seed + f)
        tr = [r for r in rows if fold_of[r[0]] != f] + extra_rows
        te_idx = [i for i, r in enumerate(rows) if fold_of[r[0]] == f]
        te = [rows[i] for i in te_idx]
        log(f"  fold {f}: train {len(tr):,} plans / {len({r[0] for r in tr})} queries | "
            f"test {len(te):,} plans / {len({r[0] for r in te})} queries")
        t0 = time.time()
        if a.model == "limao":
            p, rep_f = limao[f]
            params = {"total": rep_f["n_params"], "trainable": rep_f["n_params"]}
            fold_info.append({"fold": f, "n_train": len(tr), "n_test": len(te), **{k: v for k, v in rep_f.items() if k != "log_tail"}})
        elif a.model == "bao":
            p = run_bao_fold(tr, te, a, log, checkpoint=os.path.join(out, f"fold{f}", "bao_model"))
            fold_info.append({"fold": f, "n_train": len(tr), "n_test": len(te), "train_s": time.time() - t0})
        else:
            stats = build_stats([r[2] for r in tr])
            fz = make_featurizer(stats, a, [r[2] for r in tr])
            tree_edges = get_preset(a.preset, *a.set).graph.conv.startswith("tcnn")
            mk = lambda rs: [to_graph(pl, fz, min(ms / a.max_runtime, 1.0), q, h, ms,
                                    censored.get((q, h), False), parent_to_child=tree_edges) for q, h, pl, ms in rs]
            model, info = train_gnto(mk(tr), a, fz.node_length, device, log)
            saved_features = {"stats": stats, "features": a.features, "drop_features": a.drop_features,
                              "plan_representation": a.plan_representation,
                              "parent_to_child": tree_edges,
                              "max_runtime": a.max_runtime, "train_query_ids": [packs[q]["query_id"] for q in sorted({r[0] for r in tr if r[0] >= 0})],
                              "data_manifest_sha256": object_sha256(data_manifest)}
            if a.features == "pg":
                saved_features["pg_state"] = {"type_index": fz.type_index, "center": fz.center.tolist(), "scale": fz.scale.tolist(),
                                               "catalog_sha256": file_sha256(a.catalog)}
            model.save(os.path.join(out, f"fold{f}", "model.pt"), extra=saved_features)
            p = predict_gnto(model, mk(te), device, a.max_runtime)
            fold_info.append({"fold": f, "n_train": len(tr), "n_test": len(te), "in_dim": fz.node_length,
                              "n_node_types": fz.num_types, "train_s": time.time() - t0, **{k: v for k, v in info.items() if k != "history"}})
            params = {"total": model.num_parameters(), "trainable": model.num_parameters(trainable_only=True)}
            pd.DataFrame(info["history"]).to_csv(os.path.join(out, f"training_log_fold{f}.csv"), index=False)
        preds[te_idx] = p
        log(f"  fold {f} done in {time.time() - t0:.1f}s")

    lookup = {(q, h): float(p) for (q, h, _, _), p in zip(rows, preds)}
    picks_rows, pred_rows = [], []
    for pack in packs:
        info = pack_info(pack)
        truths = info.pop("truths")
        i = qindex[pack["query_id"]]
        scores = {h: lookup[(i, h)] for h, _, _ in candidates(pack)}
        for h, s in scores.items():
            pred_rows.append({"query_id": pack["query_id"], "hint_id": h, "pred": s, "actual_ms": truths[h],
                              "censored": censored.get((i, h), False)})
        pick = min(scores, key=scores.get) if scores else PG_DEFAULT_HINT
        info.update(pick_hint=pick, pick_ms=truths[pick], rel_regret=truths[pick] / info["oracle_ms"],
                    pick_censored=censored.get((i, pick), False))
        picks_rows.append(info)

    extra = {"model": a.model, "folds": a.folds, "seed": a.seed, "split_seed": a.split_seed,
             "plan_representation": a.plan_representation,
             "native_tree_adapter": ("extended-tree-v1" if a.model != "gnto" and a.plan_representation != "native"
                                     else "released" if a.model != "gnto" else None),
             "candidate_policy": a.candidate_policy, "deduplicate": a.deduplicate, "epochs": a.epochs,
             "native_timeout_target": a.native_timeout_target,
             "rank_weight": a.rank_weight, "rank_space": a.rank_space, "drop_features": a.drop_features, "features": a.features, "group": a.group, "extra_train": a.extra_train, "preset": a.preset if a.model == "gnto" else None,
             "overrides": a.set, "fold_info": fold_info}
    write_steerer(out, name, pd.DataFrame(picks_rows), pd.DataFrame(pred_rows), log,
                  params=(params if a.model in ("gnto", "limao") else {}), extra=extra)


if __name__ == "__main__":
    main()
