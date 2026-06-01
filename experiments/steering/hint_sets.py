"""
Bao-style 48 hint sets reconstructed from the 6-toggle GUC space.

Bao steers the PG optimizer by enumerating combinations of six boolean
``SET enable_<op> TO on/off`` flags. The full 2^6=64 space is filtered down
to 48 by requiring at least one join method and at least one scan method
to be enabled (otherwise PG cannot plan the query). Upstream Bao paper:

  Marcus et al., "Bao: Making Learned Query Optimization Practical",
  SIGMOD 2021.

Use ``HINT_SETS`` for the canonical ordered list of (hint_id, hints) pairs.
"""

from __future__ import annotations
from typing import List, Tuple

JOIN_TOGGLES = ("enable_hashjoin", "enable_mergejoin", "enable_nestloop")
SCAN_TOGGLES = ("enable_seqscan", "enable_indexscan", "enable_indexonlyscan")
ALL_TOGGLES = JOIN_TOGGLES + SCAN_TOGGLES


def _enumerate() -> List[Tuple[int, dict]]:
    """All 2^6 combinations, filtered to those with at least one join AND one scan enabled."""
    out = []
    for mask in range(1 << len(ALL_TOGGLES)):
        flags = {t: bool(mask & (1 << i)) for i, t in enumerate(ALL_TOGGLES)}
        has_join = any(flags[t] for t in JOIN_TOGGLES)
        has_scan = any(flags[t] for t in SCAN_TOGGLES)
        if has_join and has_scan:
            out.append((mask, flags))
    return out


def hints_to_sql(flags: dict) -> str:
    """Render a flag dict into a chain of ``SET ... TO on/off;`` statements."""
    parts = [f"SET {t} TO {'on' if flags[t] else 'off'};" for t in ALL_TOGGLES]
    return " ".join(parts)


def make_hint_sets() -> List[dict]:
    """Return a list of 48 (or 49) hint-set dicts with stable hint_id ordering.

    Each entry: {"hint_id": int, "flags": dict[guc -> bool], "sql": str}.
    hint_id 0 corresponds to "all six toggles on" (PG default).
    """
    enumerated = _enumerate()
    # canonical ordering: place "all on" first (matches Bao arm 0), then by mask
    all_on_mask = (1 << len(ALL_TOGGLES)) - 1
    enumerated.sort(key=lambda mf: (mf[0] != all_on_mask, mf[0]))

    return [
        {"hint_id": i, "flags": flags, "sql": hints_to_sql(flags)}
        for i, (_, flags) in enumerate(enumerated)
    ]


HINT_SETS = make_hint_sets()


if __name__ == "__main__":
    print(f"Total hint sets: {len(HINT_SETS)}")
    for h in HINT_SETS[:5]:
        print(f"  [{h['hint_id']:2d}] {h['sql']}")
    print("  ...")
    for h in HINT_SETS[-3:]:
        print(f"  [{h['hint_id']:2d}] {h['sql']}")
