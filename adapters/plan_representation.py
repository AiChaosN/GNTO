"""Explicit, reversible model-input representation; never edits measured plans.

ordered-binary-v1 keeps each original operator and its fields exactly once. For
ordered children [a, b, c, ...], a node gets [a, GROUP(b, c, ...)] with a right
chain of synthetic grouping nodes. Binary/unary plans are value-identical.
The extra depth and grouping nodes are part of the experimental protocol, not
PostgreSQL operators. All controlled encoders must use the same representation.
"""
from __future__ import annotations

import copy

REPRESENTATIONS = ("native", "ordered-binary-v1")
GROUP_TYPE = "__GNTO_ORDERED_CHILDREN_V1__"
GROUP_MARKER = "_gnto_ordered_children_v1"


def represent_plan(plan, representation="native"):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown plan representation: {representation}")
    if representation == "native":
        return plan
    result = copy.deepcopy(plan)

    def group(children):
        if len(children) == 1:
            return children[0]
        return {"Node Type": GROUP_TYPE, GROUP_MARKER: True,
                "Total Cost": 0.0, "Plan Rows": 0.0,
                "Plans": [children[0], group(children[1:])]}

    def visit(node):
        if GROUP_MARKER in node or node.get("Node Type") == GROUP_TYPE:
            raise ValueError("reserved grouping marker in input; do not transform twice")
        if "Plans" not in node:
            return
        children = node["Plans"]
        for child in children:
            visit(child)
        if len(children) > 2:
            node["Plans"] = [children[0], group(children[1:])]

    visit(result.get("Plan", result))
    return result


def restore_plan(plan):
    """Invert ordered-binary-v1 for provenance/tests, including original fields."""
    result = copy.deepcopy(plan)

    def children(node):
        if node.get(GROUP_MARKER):
            if node.get("Node Type") != GROUP_TYPE or len(node.get("Plans", [])) != 2:
                raise ValueError("invalid grouping node")
            return [item for child in node["Plans"] for item in children(child)]
        if "Plans" in node:
            node["Plans"] = [item for child in node["Plans"] for item in children(child)]
        return [node]

    root = result.get("Plan", result)
    if root.get(GROUP_MARKER):
        raise ValueError("a plan root cannot be a grouping node")
    children(root)
    return result
