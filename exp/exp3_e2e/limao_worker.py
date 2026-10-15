"""Train LIMAO's cost model on one fold and score the fold's test queries (subprocess worker).

LIMAO (``LIMAOLifeLongRLDB/bao_server``) is Bao's tree-CNN split into a module hub: every
plan is cut into three masked sub-trees (operators above the first hash / nested-loop join,
the hash-join part, the nested-loop part), a k-prototypes *module assigner* picks one of
1 / 2 / 3 tree-CNN modules for each part, and an attention merger combines the three
embeddings.  Its code imports ``model`` / ``net`` / ``featurize`` by bare name - the same
names Bao and QueryFormer use - so it runs in its own process with ``bao_server`` first on
``sys.path``.  Nothing in the LIMAO repository is modified.

Offline differences from LIMAO's online use:
* no buffer-cache information exists for the pre-collected plans, so the ``Buffers`` channel
  is present (the module assigner was fitted on 10-d node vectors) but zero;
* ``Bitmap Heap Scan`` subtrees are collapsed into one leaf, exactly as for the Bao steerer;
* training is LIMAO's own ``fit`` (batch 16, Adam, MSE on min-max scaled log runtime, its
  convergence rule, at most ``--epochs`` epochs); module assignments are memoised by content,
  which changes the run time and not the result;
* scoring is reported in two ways.  ``first``: LIMAO's own ``predict``, called once per query
  with that query's candidate plans - its server chooses the three modules from the *first*
  candidate and applies them to all candidates.  ``per_plan``: every candidate is scored with
  the modules the assigner gives to that candidate, which is how the model is trained.
  ``per_plan`` is the default of ``run_kfold.py``.

Called by ``run_kfold.py --model limao``; not meant to be run by hand.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import pickle
import random
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU: the network is small and the GPU is shared

import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_GNTO = os.path.dirname(os.path.dirname(_HERE))
LIMAO_SERVER = os.path.abspath(os.path.join(_GNTO, "..", "LIMAOLifeLongRLDB", "bao_server"))


def collapse_bitmap(node) -> None:
    if node.get("Node Type") == "Bitmap Heap Scan":
        node["Node Type"] = "Bitmap Index Scan"
        node.pop("Plans", None)
        return
    for child in node.get("Plans", []) or []:
        collapse_bitmap(child)


def prepare(plan: dict, representation="native") -> dict:
    p = copy.deepcopy(plan)
    if "Plan" not in p:
        p = {"Plan": p}
    if representation == "native":
        collapse_bitmap(p["Plan"])
    p["Buffers"] = {}                               # cache state unknown offline -> zero channel
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--plan-representation", choices=["native", "ordered-binary-v1"], default="native")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)

    if not os.path.isdir(LIMAO_SERVER):
        raise SystemExit(f"LIMAO not found at {LIMAO_SERVER}")
    sys.path.insert(0, _GNTO)
    sys.path.insert(0, LIMAO_SERVER)
    import module_assigner as ma_mod
    # kproto_processor.pkl was written by running module_assigner.py as a script
    sys.modules["__main__"].Kproto_MultiArrayProcessor = ma_mod.Kproto_MultiArrayProcessor
    import model as limao_model
    import net as limao_net
    if a.plan_representation == "ordered-binary-v1":
        import featurize as limao_featurize
        from adapters.native_tree_compat import install_extended_tree_builder
        install_extended_tree_builder(limao_featurize)
    prep = lambda p: prepare(p, a.plan_representation)

    data = pickle.load(open(a.input, "rb"))
    reg = limao_model.BaoRegression(verbose=True, have_cache_data=True)

    assigner, cache, usage = reg.module_assigner, {}, {}
    raw_predict = assigner.predict

    def memo_predict(group, tree):
        key = (group, b"".join(np.ascontiguousarray(x).tobytes() for x in assigner._extract_arrays(tree)))
        if key not in cache:
            cache[key] = raw_predict(group, tree)
        usage[(group, cache[key])] = usage.get((group, cache[key]), 0) + 1
        return cache[key]
    assigner.predict = memo_predict

    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reg.fit([prep(p) for p in data["train_plans"]], np.asarray(data["train_ms"], dtype=np.float32), epochs=a.epochs)
    train_s = time.time() - t0
    m = re.search(r"convergence condition at epoch (\d+)", buf.getvalue())
    epochs_run = int(m.group(1)) + 1 if m else a.epochs
    # Restore the serializable assigner before saving the actual fitted model.
    assigner.predict = raw_predict
    checkpoint = os.path.splitext(a.output)[0] + "_model"
    reg.save(checkpoint)
    assigner.predict = memo_predict
    train_usage = {f"{g}{i}": n for (g, i), n in sorted(usage.items())}
    usage.clear()

    preds, preds_pp = [], []
    t0 = time.time()
    net = reg._BaoRegression__net
    tt, pipe = reg._BaoRegression__tree_transform, reg._BaoRegression__pipeline
    net.eval()
    with contextlib.redirect_stdout(io.StringIO()):
        for group in data["test_groups"]:
            out = reg.predict([prep(p) for p in group], None)
            preds.append([float(v) for v in np.asarray(out).reshape(-1)])
    usage_first = {f"{g}{i}": n for (g, i), n in sorted(usage.items())}
    usage.clear()
    with torch.no_grad():
        for group in data["test_groups"]:
            a3, b3, c3 = tt.transform_subtrees([prep(p) for p in group])
            ia = [assigner.predict("a", x) for x in a3]
            ib = [assigner.predict("b", x) for x in b3]
            ic = [assigner.predict("c", x) for x in c3]
            out = pipe.inverse_transform(net(a3, b3, c3, ia, ib, ic).cpu().numpy())
            preds_pp.append([float(v) for v in np.asarray(out).reshape(-1)])
    json.dump({"preds_first": preds, "preds_per_plan": preds_pp, "train_s": train_s, "score_s": time.time() - t0, "epochs_run": epochs_run,
               "checkpoint": checkpoint, "plan_representation": a.plan_representation,
               "native_tree_adapter": "extended-tree-v1" if a.plan_representation != "native" else "released",
               "n_params": int(sum(p.numel() for p in net.parameters())),
               "modules": {"other": limao_net.NUM_OTHER_HUB, "hash_join": limao_net.NUM_HASHJOIN_HUB,
                           "nested_loop": limao_net.NUM_NESTEDLOOP_HUB},
               "module_usage_train": train_usage,
               "module_usage_test_first": usage_first,
               "module_usage_test_per_plan": {f"{g}{i}": n for (g, i), n in sorted(usage.items())},
               "log_tail": buf.getvalue()[-400:]}, open(a.output, "w"))


if __name__ == "__main__":
    main()
