"""Exp1 add-on: inference latency of the cost models, CPU and GPU, on real plans.

What is timed.  The model forward pass on plans that are already featurised (``forward``),
and the same plus batching / padding / moving to the device (``collate+forward``).  Turning
EXPLAIN JSON into the 1165-d node features is identical for QueryFormer and GNTO (one shared
``PlanFeaturizer``), so it is excluded.  Bao has its own featuriser; its ``forward`` includes
flattening the binary trees (``prepare_trees``), which is what its network does on every call.

Workloads
* ``val``  - the first ``--n-val`` plans of the Exp1 validation split (corpus plans, <= 2 joins)
* ``job``  - the Exp3 steering benchmark: 112 JOB queries, each with up to 49 hint-set plans.
             One batch = the candidate set of one query, i.e. what a steerer scores per query.

QueryFormer is padded to the largest plan of the batch (the favourable case for it); the row
``qf_pad30`` is the upstream fixed padding to 30 nodes and exists for the corpus plans only
(JOB plans exceed 30 nodes).

    python exp/exp1_qerror/bench_latency.py                 # -> results/exp1_qerror/tables/latency/
    python exp/exp1_qerror/bench_latency.py --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch as PygBatch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import GNTO_PATH  # noqa: E402  (also puts the repo root on sys.path)
from adapters.qf_adapter import QueryFormerToPyGDataset  # noqa: E402
from models.gnto import GNTO  # noqa: E402
from models.qf import (QF_DEFAULT_HPARAMS, PlanFeaturizer, QueryFormer, collator, featurize_plans,  # noqa: E402
                       load_qf_resources, pad_item, plan_csv_path)

R = lambda *p: os.path.join(GNTO_PATH, "results", *p)  # noqa: E731
MODELS = [  # name, kind, checkpoint
    ("QueryFormer", "qf", R("exp1_qerror", "qf", "best_model.pt")),
    ("QueryFormer (pad 30, upstream)", "qf_pad30", R("exp1_qerror", "qf", "best_model.pt")),
    ("Bao tree-CNN", "bao", R("exp3_e2e", "bao", "bao_model")),
    ("GNTO default (GATv2 256/128/64, 3 layers)", "gnto", R("exp1_qerror", "gnto", "best_model.pt")),
    ("GNTO GCN (3 layers)", "gnto", R("exp4_ablation", "gnto_gcn", "best_model.pt")),
    ("GNTO tree-conv (3 layers)", "gnto", R("exp4_ablation", "gnto_tcnn", "best_model.pt")),
    ("GNTO w64 L16 (identity residual)", "gnto", R("exp4_ablation", "gnto_w64_L16", "best_model.pt")),
    ("GNTO w32 L48 (identity residual)", "gnto", R("exp4_ablation", "gnto_w32_L48", "best_model.pt")),
]


# --------------------------------------------------------------------------- #
# workloads
# --------------------------------------------------------------------------- #
def load_val(n: int) -> pd.DataFrame:
    return pd.read_csv(plan_csv_path(18)).head(n).reset_index(drop=True)


def load_job(limit_queries: int | None) -> tuple[pd.DataFrame, List[List[int]]]:
    """Rows (id, json) of every finished hint plan + the row indices of each query."""
    sys.path.insert(0, os.path.join(GNTO_PATH, "exp", "exp3_e2e"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp3_common", os.path.join(GNTO_PATH, "exp", "exp3_e2e", "common.py"))
    e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e3)
    packs = e3.load_packs(e3.HINT_PLANS)
    if limit_queries:
        packs = packs[:limit_queries]
    rows, groups = [], []
    for pack in packs:
        g = []
        for _, plan, _ in e3.candidates(pack):
            g.append(len(rows))
            rows.append({"id": len(rows), "json": json.dumps(plan)})
        if g:
            groups.append(g)
    return pd.DataFrame(rows), groups


def n_nodes(plan: Dict[str, Any]) -> int:
    return 1 + sum(n_nodes(c) for c in plan.get("Plans", []))


# --------------------------------------------------------------------------- #
# per-model "prepare a batch" + "forward"
# --------------------------------------------------------------------------- #
class QFRunner:
    def __init__(self, ckpt: str, items: List[dict], fixed_pad: int | None):
        self.model = QueryFormer(**QF_DEFAULT_HPARAMS)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.model.eval()
        self.items, self.fixed_pad = items, fixed_pad
        self.params = sum(p.numel() for p in self.model.parameters())

    def to(self, device):
        self.model.to(device)

    def collate(self, idxs: Sequence[int], device):
        its = [self.items[i] for i in idxs]
        pad = self.fixed_pad or max(int(it["features"].size(0)) for it in its)
        batch, _ = collator(([pad_item(it, pad) for it in its], [0] * len(its)))
        return batch.to(device)

    def forward(self, batch):
        return self.model(batch)[0]


class GNTORunner:
    def __init__(self, ckpt: str, data_list: List[Any]):
        self.model = GNTO.load(ckpt)
        self.model.eval()
        self.data = data_list
        self.params = self.model.num_parameters()

    def to(self, device):
        self.model.to(device)

    def collate(self, idxs: Sequence[int], device):
        return PygBatch.from_data_list([self.data[i] for i in idxs]).to(device)

    def forward(self, batch):
        return self.model(batch)


class BaoRunner:
    def __init__(self, ckpt: str, plans: List[dict]):
        from adapters.bao_adapter import BaoCostPredictor, _collapse_bitmap_subtree
        pred = BaoCostPredictor(verbose=False)
        pred.load(ckpt)
        reg = pred._reg
        self.net = reg._BaoRegression__net
        self.net.eval()
        prepared = []
        for p in plans:
            q = copy.deepcopy(p)
            _collapse_bitmap_subtree(q["Plan"])
            prepared.append(q)
        self.trees = reg._BaoRegression__tree_transform.transform(prepared)
        self.params = sum(p.numel() for p in self.net.parameters())

    def to(self, device):
        if str(device).startswith("cuda"):
            self.net.cuda()

    def collate(self, idxs: Sequence[int], device):
        return [self.trees[i] for i in idxs]     # the network flattens the trees itself

    def forward(self, batch):
        return self.net(batch)


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def time_runner(runner, batches: List[List[int]], device, repeats: int, warmup: int) -> Dict[str, float]:
    prepared = [runner.collate(b, device) for b in batches]
    for _ in range(warmup):
        for pb in prepared[:4]:
            runner.forward(pb)
    sync(device)
    fwd, full = [], []
    for _ in range(repeats):
        for pb in prepared:
            sync(device)
            t0 = time.perf_counter()
            runner.forward(pb)
            sync(device)
            fwd.append(time.perf_counter() - t0)
    for _ in range(max(1, repeats // 2)):
        for b in batches:
            sync(device)
            t0 = time.perf_counter()
            runner.forward(runner.collate(b, device))
            sync(device)
            full.append(time.perf_counter() - t0)
    peak = float("nan")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        runner.forward(max(prepared, key=lambda pb: 0) if False else prepared[int(np.argmax([len(b) for b in batches]))])
        sync(device)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
    sizes = np.array([len(b) for b in batches] * repeats, dtype=np.float64)
    fwd, full = np.array(fwd) * 1e3, np.array(full) * 1e3
    return {
        "plans_per_batch": float(np.mean([len(b) for b in batches])),
        "fwd_ms_batch_median": float(np.median(fwd)), "fwd_ms_batch_p95": float(np.percentile(fwd, 95)),
        "fwd_ms_per_plan": float(np.median(fwd / sizes)),
        "fwd_plans_per_s": float(sizes.sum() / (fwd.sum() / 1e3)),
        "full_ms_batch_median": float(np.median(full)),
        "full_ms_per_plan": float(np.median(full / np.array([len(b) for b in batches] * max(1, repeats // 2)))),
        "peak_gpu_mb": peak,
    }


def chunks(idx: Sequence[int], bs: int) -> List[List[int]]:
    return [list(idx[i:i + bs]) for i in range(0, len(idx), bs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-val", type=int, default=2560, help="validation plans used (20 batches of 128)")
    ap.add_argument("--n-single", type=int, default=200, help="plans timed one at a time")
    ap.add_argument("--threads", default="1,16", help="CPU thread counts")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--only", default="", help="comma list of substrings of model names to keep")
    ap.add_argument("--out", default=R("exp1_qerror", "tables", "latency"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_val, a.n_single, a.repeats, a.warmup, a.out = 256, 20, 1, 1, a.out + "_smoke"
    os.makedirs(a.out, exist_ok=True)
    threads = [int(t) for t in a.threads.split(",")]
    models = [m for m in MODELS if not a.only or any(s.strip().lower() in m[0].lower() for s in a.only.split(","))]

    # ---- featurise both workloads once (shared by every model) -----------------
    workloads: Dict[str, Dict[str, Any]] = {}
    val_df = load_val(a.n_val)
    job_df, job_groups = load_job(8 if a.smoke else None)
    for name, df, res in (("val", val_df, load_qf_resources()),
                          ("job", job_df, load_qf_resources(zero_table_sample=True))):
        enc, hist, ts, cn = res["encoding"], res["hist_file"], res["table_sample"], res["cost_norm"]
        plans = [json.loads(j) for j in df["json"]]
        workloads[name] = {
            "plans": plans,
            "nodes": np.array([n_nodes(p["Plan"]) for p in plans]),
            "qf_items": featurize_plans(df, PlanFeaturizer(enc, hist, ts), num_workers=8, desc=f"qf-{name}", progress=False),
            "pyg": QueryFormerToPyGDataset(df, enc, hist, ts, cn, cache_name=None, add_plan_rows=True,
                                           add_join_cols=True, num_workers=8).data_list,
        }
    rng = np.random.default_rng(0)
    batching = {
        "val": {"batch 128": chunks(range(len(val_df)), 128),
                "single plan": chunks(rng.choice(len(val_df), min(a.n_single, len(val_df)), replace=False).tolist(), 1)},
        "job": {"one query (<= 49 candidates)": job_groups,
                "single plan": chunks(rng.choice(len(job_df), min(a.n_single, len(job_df)), replace=False).tolist(), 1)},
    }
    for w, d in workloads.items():
        print(f"[{w}] {len(d['plans']):,} plans, nodes per plan mean {d['nodes'].mean():.1f} / p50 "
              f"{np.median(d['nodes']):.0f} / max {d['nodes'].max()}", flush=True)

    settings = [("cpu", t) for t in threads] + ([] if a.no_gpu or not torch.cuda.is_available() else [("cuda", 0)])
    rows = []
    for mname, kind, ckpt in models:
        for w, d in workloads.items():
            if kind == "qf_pad30" and d["nodes"].max() > 30:
                continue
            if kind.startswith("qf"):
                runner = QFRunner(ckpt, d["qf_items"], 30 if kind == "qf_pad30" else None)
            elif kind == "bao":
                runner = BaoRunner(ckpt, d["plans"])
            else:
                runner = GNTORunner(ckpt, d["pyg"])
            for dev, th in settings:                     # CPU settings first, GPU last (Bao's .cuda() is one-way)
                if dev == "cpu":
                    torch.set_num_threads(th)
                device = torch.device(dev)
                runner.to(device)
                for bname, batches in batching[w].items():
                    t = time_runner(runner, batches, device, a.repeats, a.warmup)
                    rows.append({"workload": w, "model": mname, "params": runner.params, "device": dev,
                                 "threads": th if dev == "cpu" else "", "batch": bname,
                                 "nodes_per_plan": float(d["nodes"].mean()), **t})
                    print(f"{w:4s} | {mname:42s} | {dev:4s} {str(th) if dev == 'cpu' else '':>2s} | {bname:28s} | "
                          f"fwd {t['fwd_ms_batch_median']:8.3f} ms/batch  {t['fwd_ms_per_plan']:7.4f} ms/plan  "
                          f"{t['fwd_plans_per_s']:9.0f} plans/s | +collate {t['full_ms_batch_median']:8.3f} ms", flush=True)
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "latency.csv"), index=False)
    env = {"cpu": platform.processor() or "", "torch": torch.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "threads": threads,
           "n_val": len(val_df), "n_job_plans": len(job_df), "n_job_queries": len(job_groups),
           "repeats": a.repeats, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        env["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    json.dump(env, open(os.path.join(a.out, "env.json"), "w"), indent=2)

    with open(os.path.join(a.out, "latency.md"), "w") as f:
        f.write("# Inference latency (forward pass on featurised plans)\n\n")
        f.write(f"{env['cpu']}; {env['gpu']}; torch {env['torch']}; median over {a.repeats} passes.\n")
        f.write("`+collate` adds batching / padding / device transfer.\n")
        for w in workloads:
            for bname in batching[w]:
                sub = df[(df.workload == w) & (df.batch == bname)]
                if sub.empty:
                    continue
                f.write(f"\n## {w} - {bname} ({sub.plans_per_batch.iloc[0]:.1f} plans per batch, "
                        f"{sub.nodes_per_plan.iloc[0]:.1f} nodes per plan)\n\n")
                f.write("| model | params | setting | ms / batch | p95 | ms / plan | plans / s | +collate ms / batch | peak GPU MB |\n")
                f.write("|---|---|---|---|---|---|---|---|---|\n")
                for _, r in sub.iterrows():
                    setting = f"CPU, {r.threads} thread{'s' if r.threads != 1 else ''}" if r.device == "cpu" else "GPU"
                    peak = "" if np.isnan(r.peak_gpu_mb) else f"{r.peak_gpu_mb:.1f}"
                    f.write(f"| {r.model} | {int(r.params):,} | {setting} | {r.fwd_ms_batch_median:.3f} | {r.fwd_ms_batch_p95:.3f} | "
                            f"{r.fwd_ms_per_plan:.4f} | {r.fwd_plans_per_s:,.0f} | {r.full_ms_batch_median:.3f} | {peak} |\n")
    print(f"-> {a.out}/latency.csv, latency.md, env.json")


if __name__ == "__main__":
    main()
