"""Exp3 reference steerers on the pre-collected hint plans:

* ``pg_default`` - always the optimizer's own plan (hint set 0, all toggles on)
* ``oracle``     - the truly fastest of the measured variants (upper bound)
* ``pg_cost``    - argmin of PostgreSQL's own cost estimate over the variants

Usage::

    python exp/exp3_e2e/run_baselines.py            # -> results/exp3_e2e/{pg_default,oracle,pg_cost}/
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, Logger, add_common_args, load_packs, steer, write_config, write_steerer  # noqa: E402
from protocol import validate_complete_pack, plan_fingerprint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--candidate-policy", choices=["complete", "legacy-successful"], default="complete")
    ap.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    packs = load_packs(a.hint_plans)
    for pack in packs:
        if a.candidate_policy == "complete":
            validate_complete_pack(pack)
        if a.deduplicate:
            unique = {}
            for r in sorted(pack["results"], key=lambda r: int(r["hint_id"])):
                if "plan" in r:
                    unique.setdefault(plan_fingerprint(r["plan"]), r)
            pack["results"] = list(unique.values())
    for name, fn, ms in (("pg_default", None, True), ("oracle", "oracle", True),
                         ("pg_cost", lambda plan: float(plan["Plan"]["Total Cost"]), False)):
        out = os.path.join(a.out, name) if a.out else os.path.join(RESULTS, name + (f"_{a.tag}" if a.tag else ""))
        if os.path.exists(os.path.join(out, "summary.json")):
            raise SystemExit(f"completed output exists: {out}")
        log = Logger(os.path.join(out, "log.txt"))
        write_config(out, a, steerer=name)
        picks, preds = steer(packs, fn, name, log)
        write_steerer(out, name, picks, preds, log, preds_are_ms=ms)


if __name__ == "__main__":
    main()
