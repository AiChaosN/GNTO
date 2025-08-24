
"""Generate join order permutations for a simple equi-join workload.

This is a placeholder to show the interface. In practice, you will want to:
- Parse a SQL into join graph (tables, predicates)
- Enumerate left-deep / bushy per your study
- Return concrete SQLs or pg_hint_plan hints for each permutation
"""
from __future__ import annotations
from typing import List, Tuple, Iterable
import itertools

def enumerate_left_deep(tables: List[str]) -> Iterable[Tuple[str, ...]]:
    for perm in itertools.permutations(tables):
        yield perm
