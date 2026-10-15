"""GNTO trained on the zero-shot model's training plans.

The released zero-shot data holds the first 5,000 of the 10,000 plans of ``imdb_full`` and ``tpc_h``, so with these
databases in the training set the zero-shot model trains on 50,000 of the 60,000 plans.  This wrapper runs the frozen
exp2 GNTO runner unchanged on the same plans: ``imdb_full`` and ``tpc_h`` are trimmed to their first 5,000 plans when
they are training databases; test databases keep every plan.  Split, statistics (fitted on the training plans),
epochs, selection rule and outputs are those of exp2.

    python analysis/baselines/run_gnto_subset.py --split paper --split-seed 1 --seed 42 --statistics train \
        --epochs 100 --select last --device cpu --threads 2 --out results/analysis/zs_subset_20261008/gnto_paper1_s42
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "exp", "exp2_dace"), os.path.join(ROOT, "exp"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as c  # noqa: E402  (exp2_dace/common.py)

SUBSET = {"imdb_full": 5000, "tpc_h": 5000}      # the released zero-shot plans: the first half


def main() -> None:
    argv = sys.argv[:]
    if "--no-cache" not in argv:
        argv.append("--no-cache")                  # the exp2 cache keys on the full plan files
    split = argv[argv.index("--split") + 1] if "--split" in argv else "paper"
    split_seed = int(argv[argv.index("--split-seed") + 1]) if "--split-seed" in argv else 1
    train_ids, _, _ = c.make_split(split, split_seed)
    train_dbs = {c.WORKLOADS[i] for i in train_ids}
    trimmed = {db: n for db, n in SUBSET.items() if db in train_dbs}
    original = c.load_plans

    def load_plans_subset(path):
        plans = original(path)
        db = os.path.basename(path).replace("_filted.json", "")
        return plans[:trimmed[db]] if db in trimmed else plans

    c.load_plans = load_plans_subset
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
        os.makedirs(out, exist_ok=True)
        json.dump({"training_plans": "zero-shot subset", "trimmed_to_first_n": trimmed}, open(os.path.join(out, "subset.json"), "w"), indent=2)
    import run_gnto as runner
    sys.argv = argv
    runner.main()


if __name__ == "__main__":
    main()
