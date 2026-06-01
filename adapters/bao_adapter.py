"""
Bao Adapter - 适配 Bao (SIGMOD 2021) tree-CNN 成本预测模型,作为 GNTO 的对比 baseline。

Bao 用自己的 TreeFeaturizer (节点 one-hot + 归一化 cost/rows) +
BaoNet (BinaryTreeConv) 做成本预测,不走 PyG。

用法:
    from adapters.bao_adapter import (
        BaoCostPredictor, plans_from_qf_df,
        evaluate_bao, calc_q_error
    )
    pred = BaoCostPredictor(verbose=True)
    pred.fit(train_plans, train_targets, epochs=100)
    metrics = evaluate_bao(pred, val_plans, val_targets)
"""

import os
import sys
import json
import importlib.util

import numpy as np

_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bao_path():
    """Find Bao repo (BaoForPostgreSQL/bao_server)."""
    candidates = [
        os.path.join(_gnto_root, "..", "BaoForPostgreSQL"),
        "/home/AiChaosN/Project/Workspace/01_Research/BaoForPostgreSQL",
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(os.path.join(p, "bao_server")):
            return p
    raise FileNotFoundError(
        "BaoForPostgreSQL not found. Expected as sibling directory of GNTO "
        "or at /home/AiChaosN/Project/Workspace/01_Research/BaoForPostgreSQL"
    )


_BAO_MODULES = ("featurize", "net", "model")


def _load_bao_modules():
    """Load Bao's featurize/net/model from BaoForPostgreSQL/bao_server by file
    path (under unique names like ``bao_model``) so they don't collide with
    QueryFormer's own ``model`` namespace package on sys.path. Returns the
    BaoRegression class.
    """
    bao_root = get_bao_path()
    bao_server = os.path.join(bao_root, "bao_server")
    if bao_server not in sys.path:
        sys.path.insert(0, bao_server)

    loaded = {}
    for name in _BAO_MODULES:
        unique_name = f"bao_{name}"
        if unique_name in sys.modules:
            loaded[name] = sys.modules[unique_name]
            continue
        path = os.path.join(bao_server, f"{name}.py")
        spec = importlib.util.spec_from_file_location(unique_name, path)
        mod = importlib.util.module_from_spec(spec)
        # Bao's model.py does `import net` / `from featurize import TreeFeaturizer`,
        # so register short aliases too — but only the Bao-local ones.
        sys.modules[unique_name] = mod
        sys.modules[name] = mod  # short alias for intra-Bao imports
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded["model"].BaoRegression


# ---------------------------------------------------------------------------
# Plan normalization: collapse PG bitmap subtrees so Bao's TreeFeaturizer can
# handle them. Bao only knows joins + {Seq, Index, Index Only, Bitmap Index}
# Scan + single-child "transparent" nodes — IMDB plans include
# Bitmap Heap Scan → BitmapAnd/Or → Bitmap Index Scan*, which trips
# TreeBuilderError. We rewrite each Bitmap Heap Scan into a synthetic
# Bitmap Index Scan leaf carrying its Relation Name + aggregate cost/rows.
# ---------------------------------------------------------------------------
def _collapse_bitmap_subtree(node):
    """In-place rewrite: turn any Bitmap Heap Scan into a Bitmap Index Scan leaf;
    recurse into remaining children."""
    if not isinstance(node, dict):
        return
    if node.get("Node Type") == "Bitmap Heap Scan":
        node["Node Type"] = "Bitmap Index Scan"
        node.pop("Plans", None)
        return
    for child in node.get("Plans", []) or []:
        _collapse_bitmap_subtree(child)


# ---------------------------------------------------------------------------
# Plan extraction: QueryFormer CSV DataFrame → list[plan_dict], targets array
# ---------------------------------------------------------------------------
def plans_from_qf_df(df, target="execution_time"):
    """Convert a QueryFormer-format CSV DataFrame (columns: id, json) into
    a list of plan dicts (top-level objects with "Plan") and a numpy array of
    targets.

    Args:
        df: pandas DataFrame with a "json" column containing PG EXPLAIN JSON.
        target: "execution_time" | "total_cost" | "actual_total_time"

    Returns:
        (plans, targets) where plans is a list of dicts and targets is a
        float numpy array (length == len(df)).
    """
    plans = []
    targets = []
    for json_str in df["json"]:
        plan = json.loads(json_str) if isinstance(json_str, str) else json_str
        if "Plan" in plan:
            _collapse_bitmap_subtree(plan["Plan"])
        plans.append(plan)
        if target == "execution_time":
            t = plan.get("Execution Time", 0.0)
        elif target == "actual_total_time":
            t = plan["Plan"].get("Actual Total Time", 0.0)
        elif target == "total_cost":
            t = plan["Plan"].get("Total Cost", 0.0)
        else:
            raise ValueError(f"Unknown target: {target}")
        targets.append(float(t))
    return plans, np.array(targets, dtype=np.float64)


# ---------------------------------------------------------------------------
# BaoCostPredictor: thin wrapper around Bao's BaoRegression
# ---------------------------------------------------------------------------
class BaoCostPredictor:
    """Wraps Bao's BaoRegression so it integrates with GNTO experiments.

    The underlying model (BaoNet) is a 3-layer BinaryTreeConv + DynamicPooling
    + small MLP, trained with MSE on log1p-MinMax-scaled targets.

    Args:
        verbose: forward-printed training logs.
        epochs: max epochs (Bao's own early-stopping condition still applies).
    """

    def __init__(self, verbose=False, epochs=100):
        BaoRegression = _load_bao_modules()
        self._BaoRegression = BaoRegression
        self._reg = BaoRegression(verbose=verbose)
        self._verbose = verbose
        self._epochs_target = epochs  # informational; Bao hardcodes 100 internally

    def fit(self, plans, targets, epochs=None):
        """Train on plans (list of dicts) and targets (1D numpy array of times)."""
        if epochs is not None and epochs != 100 and self._verbose:
            print(f"[bao_adapter] Note: Bao's BaoRegression hardcodes 100 epochs; "
                  f"requested {epochs} ignored. Patch upstream if needed.")
        targets = np.asarray(targets, dtype=np.float32)
        self._reg.fit(plans, targets)

    def predict(self, plans):
        """Return raw-scale predictions (1D numpy array)."""
        preds = self._reg.predict(plans)
        return np.asarray(preds).reshape(-1)

    def save(self, path):
        self._reg.save(path)

    def load(self, path):
        self._reg.load(path)

    @property
    def num_items_trained_on(self):
        return self._reg.num_items_trained_on()


# ---------------------------------------------------------------------------
# Evaluation utilities (mirrors qf_adapter.calc_q_error API)
# ---------------------------------------------------------------------------
def calc_q_error(preds, targets, percentiles=(50, 75, 90, 95, 99)):
    """Q-Error percentiles, mirroring qf_adapter.calc_q_error."""
    from utils.metrics import qerror_percentiles
    q = qerror_percentiles(preds, targets, percentiles=percentiles)
    return np.array([q[f"q{int(pc)}"] for pc in percentiles])


def evaluate_bao(predictor, plans, targets, group_ids=None):
    """Run inference and report (preds, q-error percentiles + ranking metrics).

    With group_ids (one per plan, same length as plans), also reports per-query
    Top-1 regret for Bao-style steering evaluation.
    """
    from utils.metrics import evaluation_summary
    preds = predictor.predict(plans)
    out = {"preds": preds}
    out.update(evaluation_summary(preds, targets, group_ids=group_ids))
    return out
