"""Where the DACE benchmark lives inside this repo (``data/dace/workload1``) and the splits."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

GNTO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DACE_DATA_DIR = os.path.join(GNTO_ROOT, "data", "dace", "workload1")

WORKLOADS: List[str] = [
    "accidents", "airline", "baseball", "basketball", "carcinogenesis",
    "consumer", "credit", "employee", "fhnk", "financial",
    "geneea", "genome", "hepatitis", "imdb_full", "movielens",
    "seznam", "ssb", "tournament", "tpc_h", "walmart",
]

# fixed 10 / 10 split of the 20 databases (training ids / zero-shot test ids)
PAPER_TRAIN_IDS = [3, 10, 11, 12, 13, 14, 15, 16, 18, 19]
PAPER_TEST_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 17]


def statistics_path(data_dir: str = DACE_DATA_DIR) -> str:
    return os.path.join(data_dir, "statistics.json")


def plans_path(db: str, data_dir: str = DACE_DATA_DIR) -> str:
    return os.path.join(data_dir, f"{db}_filted.json")


def load_statistics(data_dir: str = DACE_DATA_DIR) -> Dict[str, Any]:
    with open(statistics_path(data_dir)) as f:
        return json.load(f)


def make_split(split: str = "paper", split_seed: int = 1) -> Tuple[List[int], List[int], str]:
    """``(train_db_ids, test_db_ids, name)``; ``split='rand'`` draws a seeded random 10 / 10 split
    (``random.Random(split_seed).sample``; ``rand1`` / ``rand2`` are split seeds 1 and 2)."""
    if split == "paper":
        return list(PAPER_TRAIN_IDS), list(PAPER_TEST_IDS), "paper"
    if split == "rand":
        rng = random.Random(split_seed)
        train = sorted(rng.sample(range(len(WORKLOADS)), 10))
        return train, [i for i in range(len(WORKLOADS)) if i not in train], f"rand{split_seed}"
    raise ValueError(f"split must be 'paper' or 'rand', got {split!r}")


__all__ = ["GNTO_ROOT", "DACE_DATA_DIR", "WORKLOADS", "PAPER_TRAIN_IDS", "PAPER_TEST_IDS",
           "statistics_path", "plans_path", "load_statistics", "make_split"]
