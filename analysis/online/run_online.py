"""Online plan selection: Bao's learning loop on the measured executions of the benchmark.

A selector starts without a model.  The queries of the workload arrive in a random order, pass after pass.  For
every query the selector sees the distinct plans of the 49 hint configurations, executes the plan with the lowest
predicted latency and observes the latency of that plan only: one of its three recorded executions, drawn at random
and capped at the statement timeout.  Before the first training every query runs PostgreSQL's default plan, as in
Bao.  After each pass the model is retrained from scratch on a bootstrap resample of all executions so far (Bao's
Thompson sampling); featurizers are fitted on the executed plans only.

Nothing is tuned here: the models, inputs, objectives and epochs are those of the offline experiment
(``exp/exp3_e2e/run_kfold.py``), whose functions are imported unchanged.

    gnto      GNTO, PG-stat inputs, ranking objective (the default GNTO selector)
    gnto_none the same without message passing (graph.conv=none): the independent-operator selector
    gnto_tcr  GNTO, type / cost / rows inputs, ranking objective

Schedules (``--schedule``).  ``sync``: a pass starts with the model retrained on all previous passes; retraining is
not on the clock.  ``async``: Bao's background trainer.  A clock advances by the execution time of the chosen plans;
a retraining starts at the end of a pass whenever the trainer is free and is ready after the wall time it takes on
CPU in this process (``train_s``); a pass is served by the newest model that was ready when it started, and by the
default plans before the first one.  Retraining never adds to the clock, staleness is the price.
    bao       Bao's tree convolution network with its inputs and training procedure (adapters/bao_adapter.py)
    limao     LIMAO's modular network, retrained in a worker process (exp/exp3_e2e/limao_worker.py)

The executions are replayed from ``data/e2e/<benchmark>`` (three runs per plan on a warm cache); no query is
executed by this script.  The reported latency of a choice is the median of the three runs of the chosen plan, the
value that the offline experiment uses.

    python analysis/online/run_online.py --model gnto --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E2E = os.path.join(ROOT, "exp", "exp3_e2e")
sys.path.insert(0, ROOT)
sys.path.insert(0, E2E)
_argv, sys.argv = sys.argv, sys.argv[:1]
import run_kfold as rk                                         # noqa: E402
from common import PG_DEFAULT_HINT, load_packs, set_seed       # noqa: E402
from protocol import plan_fingerprint                          # noqa: E402
from adapters.plan_representation import represent_plan       # noqa: E402
sys.argv = _argv

REPRESENTATION = "ordered-binary-v1"
OFFLINE = dict(epochs=100, bs=128, lr=1e-3, preset="dace_zero_shot", set=["graph.conv=gatv2", "graph.heads=1"],
               rank_weight=1.0, rank_min_gap=0.1, rank_space="log", queries_per_batch=3, grouped_batches=True,
               plan_representation=REPRESENTATION, drop_features="")


def load_rows(hint_dir: str, timeout_ms: float):
    packs = load_packs(hint_dir)
    rows = []
    for q, pack in enumerate(packs):
        unique = {}
        for r in sorted(pack["results"], key=lambda r: int(r["hint_id"])):
            if "plan" in r:
                unique.setdefault(plan_fingerprint(r["plan"]), r)
        for r in unique.values():
            runs = [(min(float(m["wall_ms"]), timeout_ms), bool(m.get("timeout", False))) for m in r["measurements"]]
            rows.append(SimpleNamespace(q=q, hint=int(r["hint_id"]), plan=represent_plan(r["plan"], REPRESENTATION),
                                        ms=float(r["evaluation_ms"]), runs=runs))
    return packs, rows


def fit_predict(model: str, entries, rows, a, log, work: str) -> np.ndarray:
    """Train on the executions ``entries`` = [(row index, observed ms, timed out)], score every candidate plan."""
    if model == "bao":
        train = [(rows[k].q, rows[k].hint, rows[k].plan, ms) for k, ms, _ in entries]
        return rk.run_bao_fold(train, [(r.q, r.hint, r.plan, r.ms) for r in rows], a, log)
    if model == "limao":
        by_q = {}
        for i, r in enumerate(rows):
            by_q.setdefault(r.q, []).append(i)
        groups = list(by_q.values())
        inp, outp = os.path.join(work, "limao_in.pkl"), os.path.join(work, "limao_out.json")
        pickle.dump({"train_plans": [rows[k].plan for k, _, _ in entries], "train_ms": [ms for _, ms, _ in entries],
                     "test_groups": [[rows[i].plan for i in g] for g in groups]}, open(inp, "wb"))
        subprocess.run([sys.executable, os.path.join(E2E, "limao_worker.py"), "--input", inp, "--output", outp,
                        "--epochs", str(a.epochs), "--seed", str(a.seed), "--threads", str(a.threads),
                        "--plan-representation", REPRESENTATION],
                       check=True, stdout=open(os.path.join(work, "limao_worker.log"), "w"), stderr=subprocess.STDOUT)
        rep = json.load(open(outp))
        pred = np.zeros(len(rows))
        for g, vals in zip(groups, rep["preds_per_plan"]):
            pred[g] = vals
        return pred
    plans = [rows[k].plan for k, _, _ in entries]
    fz = rk.make_featurizer(rk.build_stats(plans), a, plans)
    train = [rk.to_graph(rows[k].plan, fz, min(ms / a.max_runtime, 1.0), rows[k].q, rows[k].hint, ms, cens)
             for k, ms, cens in entries]
    net, _ = rk.train_gnto(train, a, fz.node_length, torch.device("cpu"), log)
    cand = [rk.to_graph(r.plan, fz, 0.0, r.q, r.hint, r.ms) for r in rows]
    return rk.predict_gnto(net, cand, torch.device("cpu"), a.max_runtime)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["gnto", "gnto_none", "gnto_tcr", "bao", "limao"])
    ap.add_argument("--schedule", default="sync", choices=["sync", "async"],
                    help="sync: retrain between passes; async: background trainer on a clock of execution time (see above)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--hint-plans", default=os.path.join(ROOT, "data", "e2e", "hint_plans_v2_job112"))
    ap.add_argument("--max-runtime", type=float, default=32000.0, help="statement timeout of the benchmark in ms")
    ap.add_argument("--explore", default="bootstrap", choices=["bootstrap", "none"],
                    help="bootstrap: retrain on a resample of the executions (Thompson sampling of Bao); none: all executions")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit-queries", type=int, help="smoke test")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results", "analysis", "online_20261005"))
    a = ap.parse_args()
    for k, v in OFFLINE.items():
        setattr(a, k, v)
    a.features = "pg" if a.model in ("gnto", "gnto_none") else "dace"
    if a.model == "gnto_none":
        a.set = ["graph.conv=none", "graph.heads=1"]
    a.catalog = os.path.join(a.hint_plans, "pg_catalog_snapshot.json")
    torch.set_num_threads(a.threads)
    tag = os.path.basename(a.hint_plans.rstrip("/")).split("_")[-1]
    out = os.path.join(a.out_root, tag, f"{a.model}_s{a.seed}" + ("" if a.explore == "bootstrap" else "_greedy")
                       + ("" if a.schedule == "sync" else "_async"))
    if os.path.exists(os.path.join(out, "summary.json")):
        raise SystemExit(f"completed output exists: {out}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(msg: str) -> None:
        if msg.startswith("    epoch"):
            return
        logf.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        logf.flush()

    packs, rows = load_rows(a.hint_plans, a.max_runtime)
    if a.limit_queries:
        rows = [r for r in rows if r.q < a.limit_queries]
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r.q, []).append(i)
    nq = len(by_q)
    default = {q: next(i for i in idx if rows[i].hint == PG_DEFAULT_HINT) for q, idx in by_q.items()}
    best = {q: min(idx, key=lambda i: rows[i].ms) for q, idx in by_q.items()}
    base = np.array([rows[default[q]].ms for q in range(nq)])
    json.dump({**{k: v for k, v in vars(a).items()}, "n_queries": nq, "n_plans": len(rows),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(out, "config.json"), "w"), indent=2)
    log(f"{a.model} seed {a.seed}: {nq} queries, {len(rows)} distinct plans, {a.passes} passes, explore={a.explore}, "
        f"schedule={a.schedule}")

    rng = np.random.default_rng(a.seed)
    entries, preds, passes, choices, seen = [], None, [], [], set()
    clock, served, pending, n_retrain = 0.0, -1, None, 0      # async: pending = (ready at clock, predictions, data through pass)
    for p in range(a.passes):
        pick = dict(default) if preds is None else {q: min(idx, key=lambda i: preds[i]) for q, idx in by_q.items()}
        for q in rng.permutation(nq):
            k = pick[int(q)]
            ms, timed_out = rows[k].runs[int(rng.integers(len(rows[k].runs)))]
            entries.append((k, ms, timed_out))
            seen.add(k)
            choices.append(dict(pass_=p, q=int(q), query_id=packs[int(q)]["query_id"], hint=rows[k].hint, ms=rows[k].ms,
                                default_ms=rows[default[int(q)]].ms, best_ms=rows[best[int(q)]].ms, observed_ms=ms))
        got = np.array([rows[pick[q]].ms for q in range(nq)])
        sp = base / got
        clock += float(got.sum() / 1e3)
        rec = dict(pass_=p, model_pass=served, clock_s=clock, geo_speedup=float(np.exp(np.log(sp).mean())), total_s=float(got.sum() / 1e3),
                   default_total_s=float(base.sum() / 1e3), faster=int((sp > 1.05).sum()), slower=int((sp < 1 / 1.05).sum()),
                   slower_2x=int((sp < 0.5).sum()), optimal=int(sum(rows[pick[q]].ms <= rows[best[q]].ms * 1.05 for q in range(nq))),
                   plans_executed=len(seen))
        t0 = time.time()
        rec["retrained"] = False
        if p + 1 < a.passes:
            if a.schedule == "async" and pending is not None and pending[0] <= clock:
                preds, served, pending = pending[1], pending[2], None           # the background trainer has finished
            if a.schedule == "sync" or pending is None:
                a.seed_train = 1000 * a.seed + p
                saved, a.seed = a.seed, a.seed_train
                set_seed(a.seed)
                idx = rng.integers(len(entries), size=len(entries)) if a.explore == "bootstrap" else np.arange(len(entries))
                new = fit_predict(a.model, [entries[i] for i in idx], rows, a, log, out)
                a.seed = saved
                n_retrain += 1
                rec["retrained"] = True
                if a.schedule == "sync":
                    preds, served = new, p
                else:
                    pending = (clock + (time.time() - t0), new, p)
        rec["train_s"] = time.time() - t0
        rec["ready_at_s"] = pending[0] if pending is not None else float("nan")
        passes.append(rec)
        log(f"pass {p:2d}: G {rec['geo_speedup']:.3f} | total {rec['total_s']:.1f}s (default {rec['default_total_s']:.1f}s) | "
            f"faster {rec['faster']} slower {rec['slower']} (2x: {rec['slower_2x']}) optimal {rec['optimal']} | "
            f"{rec['plans_executed']} plans executed | model of pass {rec['model_pass']} | clock {clock:.0f}s | "
            f"retrain {rec['train_s']:.0f}s")
        pd.DataFrame(passes).rename(columns={"pass_": "pass"}).to_csv(os.path.join(out, "passes.csv"), index=False)
    pd.DataFrame(choices).rename(columns={"pass_": "pass"}).to_csv(os.path.join(out, "choices.csv"), index=False)
    df = pd.DataFrame(passes)
    last = df.tail(5)
    oracle = np.array([rows[best[q]].ms for q in range(nq)])
    json.dump({"model": a.model, "seed": a.seed, "passes": a.passes, "explore": a.explore, "schedule": a.schedule,
               "n_retrain": n_retrain, "n_queries": nq,
               "last5_geo_speedup": float(np.exp(np.log(last.geo_speedup).mean())), "last5_total_s": float(last.total_s.mean()),
               "default_total_s": float(base.sum() / 1e3), "oracle_total_s": float(oracle.sum() / 1e3),
               "oracle_geo_speedup": float(np.exp(np.log(base / oracle).mean())),
               "cumulative_total_s": float(df.total_s.sum()), "cumulative_default_s": float(df.default_total_s.sum()),
               "first_pass_at_least_default": next((int(r.pass_) for r in df.itertuples() if r.pass_ > 0 and r.geo_speedup >= 1.0), None),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(out, "summary.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
