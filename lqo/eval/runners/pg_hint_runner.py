
from __future__ import annotations
from typing import Optional

def make_join_order_hint(tables_in_order: list[str]) -> str:
    """pg_hint_plan hint for join order (Leading)."""
    inside = ' '.join(tables_in_order)
    return f"/*+ Leading({inside}) */"
