"""
Zero-Shot Cost Model Adapter (Hilprecht & Binnig, VLDB 2022)

Status: SCAFFOLD. Full integration is non-trivial because the upstream codebase
(``zero-shot-cost-estimation``) uses DGL graphs (not PyG) and a custom cross-DB
JSON schema (not raw PG EXPLAIN JSON). See ``docs/zero_shot_integration.md``
for the concrete steps still required.

Once the conversion pipeline is in place, the public API mirrors bao_adapter:

    from adapters.zeroshot_adapter import ZeroShotPredictor, evaluate_zeroshot
    pred = ZeroShotPredictor(verbose=True)
    pred.fit(train_plans, train_targets, epochs=100)
    metrics = evaluate_zeroshot(pred, val_plans, val_targets)

For now the class raises ``NotImplementedError`` on use, but the path-discovery
and module-loading helpers are wired up so the missing pieces are localized to
two methods.

Upstream: https://github.com/DataManagementLab/zero-shot-cost-estimation
Paper: VLDB 2022, https://www.vldb.org/pvldb/vol15/p2361-hilprecht.pdf
"""

from __future__ import annotations

import os
import sys
import json
import importlib.util

import numpy as np

_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_zeroshot_path():
    """Locate the zero-shot-cost-estimation repo."""
    candidates = [
        os.path.join(_gnto_root, "..", "zero-shot-cost-estimation"),
        "/home/AiChaosN/Project/Workspace/01_Research/zero-shot-cost-estimation",
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(os.path.join(p, "models", "zero_shot_models")):
            return p
    raise FileNotFoundError(
        "zero-shot-cost-estimation not found. Expected as sibling directory of "
        "GNTO or at /home/AiChaosN/Project/Workspace/01_Research/"
    )


def _check_dgl():
    """Zero-shot uses DGL. Confirm it's available so we fail early with a useful message."""
    try:
        import dgl  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Plan conversion: PG EXPLAIN JSON -> zero-shot's cross-DB plan schema
# ---------------------------------------------------------------------------
def convert_pg_plan_to_zeroshot(pg_plan: dict, db_stats: dict = None) -> dict:
    """Convert a PG EXPLAIN JSON plan into the schema zero-shot's preprocessing
    pipeline expects.

    BLOCKER: zero-shot's plan schema is produced by their own
    ``cross_db_benchmark/parse_plan.py`` and pulls statistics from a DB stats
    JSON file (table cardinalities, attribute histograms, etc.). To use IMDB
    plans here we need either:

      (a) Run their ``run_benchmark.py setup`` against the same PG 12.5 + IMDB
          we're using, and feed those JSONs to ``train.py``; OR
      (b) Reimplement their parse_plan + statistics extraction inside this
          adapter, so we can pass GNTO's existing
          ``data/train_plan_part*.csv`` content in directly.

    Option (a) is faster but couples us to their CLI; option (b) keeps the
    GNTO adapter pattern but is ~300 LOC of reproduction. Recommend (a) for
    the rebuttal experiments, (b) later for tighter integration.
    """
    raise NotImplementedError(
        "Zero-shot plan conversion not yet implemented. See docstring; choose "
        "option (a) — run upstream's run_benchmark.py to produce parsed plans."
    )


# ---------------------------------------------------------------------------
# Predictor wrapper (skeleton)
# ---------------------------------------------------------------------------
class ZeroShotPredictor:
    """Wraps zero-shot-cost-estimation's MSCNZeroShotModel/MAESTROZeroShotModel.

    Public API matches BaoCostPredictor: ``fit(plans, targets)`` /
    ``predict(plans)`` / ``save(path)`` / ``load(path)``.

    Currently raises NotImplementedError to make the missing-integration
    boundary explicit. To finish:

    1. Implement ``convert_pg_plan_to_zeroshot`` (or run upstream's
       run_benchmark.py to produce parsed plans on disk).
    2. In ``fit``, build a DGL DataLoader using their PlanDataset, instantiate
       the chosen ZeroShotModel, run their training loop (or import it).
    3. In ``predict``, run one forward pass per batch and inverse-transform.
    """

    def __init__(self, model_variant: str = "MSCNZeroShotModel", verbose: bool = False):
        if not _check_dgl():
            raise ImportError(
                "Zero-shot requires DGL (pip install dgl). It's listed in "
                "zero-shot-cost-estimation/requirements.txt but not in GNTO's."
            )
        self._zs_root = get_zeroshot_path()
        self._variant = model_variant
        self._verbose = verbose
        self._model = None

    def fit(self, plans, targets, epochs: int = 100):
        raise NotImplementedError(
            "ZeroShotPredictor.fit: see docs/zero_shot_integration.md for the "
            "TODO list. Bottleneck is data-format conversion (see "
            "convert_pg_plan_to_zeroshot)."
        )

    def predict(self, plans):
        raise NotImplementedError("Pending fit() implementation.")

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


def evaluate_zeroshot(predictor, plans, targets, group_ids=None):
    """Mirror evaluate_bao's signature so the recompute script can drop this in."""
    from utils.metrics import evaluation_summary
    preds = predictor.predict(plans)
    out = {"preds": preds}
    out.update(evaluation_summary(preds, targets, group_ids=group_ids))
    return out
