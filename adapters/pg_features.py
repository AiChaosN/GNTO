"""Plan-node features that mirror what PostgreSQL's optimizer itself consults.

The vector holds the optimizer's *inputs* (catalog statistics, predicates, operator choices)
in addition to its *outputs* (``Plan Rows``, ``Total Cost``).

Everything is available at planning time - nothing from ``EXPLAIN ANALYZE`` (``Actual *``,
buffer counters, hash batches, sort methods, rows removed) - and everything is expressed by
*value*, never by table / column / index id, so an unseen schema element cannot be out of
vocabulary.

Per node, from the plan:
    operator            Node Type (fold vocabulary), Join Type, Parent Relationship, Strategy,
                        Inner Unique, Parallel Aware, backward scan, partial aggregate, Memoize
    optimizer estimates Startup Cost, Total Cost, Plan Rows, Plan Width, Workers Planned
Per scan node, from the catalog snapshot (``pg_class`` / ``pg_index``):
    relation            reltuples, relpages, all-visible fraction
    index               index pages, unique, number of key columns
Per node, from its conditions (Filter / Index Cond / Recheck Cond / Join Filter / Hash Cond /
Merge Cond) resolved against ``pg_stats``:
    shape               predicate counts by operator class, OR present, join-key comparisons,
                        distinct constant-filtered columns (independence-assumption exposure)
    column statistics   null_frac, n_distinct, avg_width, |correlation|, MCV coverage
    selectivity inputs  PG-style selectivity per constant predicate (MCV hit / non-MCV share /
                        histogram fraction / null_frac) and their product and minimum

Global cost GUCs (``seq_page_cost`` ...) are constant within one server, so they carry no
signal on a single-server benchmark and are not part of the vector.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

COND_KEYS = ("Filter", "Index Cond", "Recheck Cond", "Join Filter", "Hash Cond", "Merge Cond")
JOIN_TYPES = ("Inner", "Left", "Right", "Full", "Semi", "Anti", "Right Semi", "Right Anti")
PARENT_REL = ("Outer", "Inner", "Subquery", "InitPlan", "SubPlan", "Member")
STRATEGIES = ("Plain", "Sorted", "Hashed", "Mixed")

_IDENT = r"[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?"
_CMP = re.compile(r"\(*(" + _IDENT + r")\)*(?:::[\w ]+(?:\[\])?)?\s*(<=|>=|<>|!~~\*|~~\*|!~~|~~|=|<|>)\s*"
                  r"(ANY\s*\(|ALL\s*\()?\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?|" + _IDENT + r")")
_NULL = re.compile(r"\(*(" + _IDENT + r")\)*(?:::[\w ]+)?\s+IS\s+(NOT\s+)?NULL")

NUMERIC = [
    # optimizer estimates (log1p)
    "startup_cost", "total_cost", "plan_rows", "plan_width", "workers_planned",
    # relation / index (log1p where sizes)
    "rel_tuples", "rel_pages", "rel_allvisible_frac", "idx_pages", "idx_natts",
    # predicate shape
    "n_pred", "n_eq_const", "n_range", "n_like", "n_not_like", "n_neq", "n_null_test", "n_join_key",
    "n_const_cols", "n_or",
    # column statistics (mean / max over referenced columns)
    "col_null_frac", "col_log_ndistinct", "col_avg_width", "col_abs_corr_max", "col_mcv_coverage",
    # selectivity inputs
    "log_sel_product", "log_sel_min", "n_sel_unknown",
]
FLAGS = ["inner_unique", "parallel_aware", "backward_scan", "partial_agg", "memoize",
         "has_relation", "has_index", "idx_unique"]


def _walk(node):
    yield node
    for c in node.get("Plans", []):
        yield from _walk(c)


def _parse_array(text: Optional[str]) -> List[str]:
    """``{a,"b c",3}`` (pg_stats text form) -> list of strings."""
    if not text or text in ("{}", "NULL"):
        return []
    body = text[1:-1]
    out, cur, quoted, esc = [], [], False, False
    for ch in body:
        if esc:
            cur.append(ch); esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            quoted = not quoted
        elif ch == "," and not quoted:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _num(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class CatalogStats:
    """The ``pg_catalog`` snapshot (``data/e2e/pg_catalog_snapshot.json``) with parsed MCV lists
    and histograms, plus PostgreSQL-style selectivity estimates for single predicates."""

    def __init__(self, path: str):
        snap = json.load(open(path))
        self.tables: Dict[str, dict] = snap["tables"]
        self.indexes: Dict[str, dict] = snap["indexes"]
        self.columns: Dict[str, dict] = {}
        for key, c in snap["columns"].items():
            mcv = _parse_array(c.get("mcv"))
            hist = _parse_array(c.get("hist"))
            hist_num = [x for x in (_num(h) for h in hist) if x is not None]
            nd = c.get("n_distinct") or 0.0
            reltuples = self.tables.get(key.split(".")[0], {}).get("reltuples", 0.0)
            nd_abs = -nd * reltuples if nd < 0 else nd
            self.columns[key] = {**c, "mcv_list": mcv, "mcf_list": c.get("mcf") or [], "hist_list": hist,
                                 "hist_num": hist_num if len(hist_num) == len(hist) else [],
                                 "nd_abs": max(nd_abs, 1.0), "mcv_cov": float(sum(c.get("mcf") or []))}

    def column(self, rel: Optional[str], col: str) -> Optional[dict]:
        return self.columns.get(f"{rel}.{col}") if rel else None

    @staticmethod
    def _strip(v: str) -> str:
        return v[1:-1].replace("''", "'") if v.startswith("'") and v.endswith("'") else v

    def eq_selectivity(self, c: dict, value: str) -> float:
        v = self._strip(value)
        if v in c["mcv_list"]:
            return float(c["mcf_list"][c["mcv_list"].index(v)])
        rest = max(1.0 - (c.get("null_frac") or 0.0) - c["mcv_cov"], 0.0)
        return rest / max(c["nd_abs"] - len(c["mcv_list"]), 1.0)

    def any_selectivity(self, c: dict, value: str) -> float:
        """``col = ANY ('{a,b,c}')``: PG sums the per-element equality selectivities (capped at 1)."""
        items = _parse_array(self._strip(value))
        return min(sum(self.eq_selectivity(c, "'" + v + "'") for v in items), 1.0) if items else 0.0

    @staticmethod
    def _like_regex(pattern: str, case_insensitive: bool) -> "re.Pattern":
        out, esc = [], False
        for ch in pattern:
            if esc:
                out.append(re.escape(ch)); esc = False
            elif ch == "\\":
                esc = True
            elif ch == "%":
                out.append(".*")
            elif ch == "_":
                out.append(".")
            else:
                out.append(re.escape(ch))
        return re.compile("^" + "".join(out) + "$", re.S | (re.I if case_insensitive else 0))

    def like_selectivity(self, c: dict, value: str, case_insensitive: bool = False) -> Optional[float]:
        """Like PG's pattern selectivity with statistics: apply the pattern to the MCVs (weighted by
        their frequencies) and to the histogram entries (uniform share of the non-MCV rows)."""
        rx = self._like_regex(self._strip(value), case_insensitive)
        mcv_hit = sum(f for m, f in zip(c["mcv_list"], c["mcf_list"]) if rx.match(m))
        hist = c["hist_list"]
        if len(hist) < 10 and not c["mcv_list"]:
            return None                                        # PG falls back to a default constant here
        rest = max(1.0 - (c.get("null_frac") or 0.0) - c["mcv_cov"], 0.0)
        hist_frac = (sum(1 for h in hist if rx.match(h)) / len(hist)) if len(hist) >= 10 else 0.0
        return mcv_hit + rest * hist_frac

    def range_selectivity(self, c: dict, op: str, value: str) -> Optional[float]:
        x = _num(self._strip(value))
        h = c["hist_num"]
        if x is None or len(h) < 2:
            return None
        pos = bisect.bisect_left(h, x) / (len(h) - 1)          # share of histogram below x
        below = min(max(pos, 0.0), 1.0)
        frac = below if op in ("<", "<=") else 1.0 - below
        mcv_part = sum(f for m, f in zip(c["mcv_list"], c["mcf_list"])
                       if _num(m) is not None and ((_num(m) < x) if op in ("<", "<=") else (_num(m) > x)))
        rest = max(1.0 - (c.get("null_frac") or 0.0) - c["mcv_cov"], 0.0)
        return mcv_part + rest * frac


class PGFeaturizer:
    """Fit on training plans (node-type vocabulary + per-channel robust scalers), then turn a
    plan node into a fixed-length vector.  ``bind(root)`` must be called once per plan so
    ``alias.column`` references resolve to relations."""

    def __init__(self, catalog: CatalogStats):
        self.cat = catalog
        self.type_index: Dict[str, int] = {}
        self.center = np.zeros(len(NUMERIC)); self.scale = np.ones(len(NUMERIC))
        self._alias: Dict[str, str] = {}

    # ------------------------------------------------------------------ layout
    @property
    def num_types(self) -> int:
        return len(self.type_index)

    @property
    def node_length(self) -> int:
        return self.num_types + len(JOIN_TYPES) + len(PARENT_REL) + len(STRATEGIES) + len(FLAGS) + len(NUMERIC)

    # ----------------------------------------------------------------- binding
    def bind(self, root: Dict[str, Any]) -> None:
        self._alias = {}
        for n in _walk(root):
            if "Relation Name" in n:
                self._alias[n.get("Alias", n["Relation Name"])] = n["Relation Name"]
                self._alias.setdefault(n["Relation Name"], n["Relation Name"])

    def _resolve(self, ident: str, node: Dict[str, Any]) -> Tuple[Optional[str], str]:
        if "." in ident:
            alias, col = ident.split(".", 1)
            return self._alias.get(alias), col
        return node.get("Relation Name"), ident

    # ------------------------------------------------------------------ raw
    def raw(self, node: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
        num = {k: 0.0 for k in NUMERIC}; flag = {k: 0.0 for k in FLAGS}
        cat = {"type": node.get("Node Type", ""), "join": node.get("Join Type", ""),
               "parent": node.get("Parent Relationship", ""), "strategy": node.get("Strategy", "")}
        for k, src in (("startup_cost", "Startup Cost"), ("total_cost", "Total Cost"), ("plan_rows", "Plan Rows"),
                       ("plan_width", "Plan Width"), ("workers_planned", "Workers Planned")):
            num[k] = math.log1p(max(float(node.get(src, 0.0) or 0.0), 0.0))
        flag["inner_unique"] = float(bool(node.get("Inner Unique")))
        flag["parallel_aware"] = float(bool(node.get("Parallel Aware")))
        flag["backward_scan"] = float(node.get("Scan Direction") == "Backward")
        flag["partial_agg"] = float(node.get("Partial Mode", "Simple") != "Simple")
        flag["memoize"] = float(node.get("Node Type") == "Memoize")

        rel = node.get("Relation Name")
        if rel and rel in self.cat.tables:
            t = self.cat.tables[rel]
            flag["has_relation"] = 1.0
            num["rel_tuples"] = math.log1p(max(t["reltuples"], 0.0))
            num["rel_pages"] = math.log1p(t["relpages"])
            num["rel_allvisible_frac"] = t["relallvisible"] / t["relpages"] if t["relpages"] else 0.0
        idx = node.get("Index Name")
        if idx and idx in self.cat.indexes:
            i = self.cat.indexes[idx]
            flag["has_index"] = 1.0; flag["idx_unique"] = float(i["unique"])
            num["idx_pages"] = math.log1p(i["relpages"]); num["idx_natts"] = float(i["natts"])

        stats_cols, sels, const_cols = [], [], set()
        for key in COND_KEYS:
            text = node.get(key)
            if not text:
                continue
            num["n_or"] += text.count(" OR ")
            for m in _NULL.finditer(text):
                num["n_pred"] += 1; num["n_null_test"] += 1
                r, c = self._resolve(m.group(1), node); cs = self.cat.column(r, c)
                if cs:
                    stats_cols.append(cs); const_cols.add((r, c))
                    nf = cs.get("null_frac") or 0.0
                    sels.append(1.0 - nf if m.group(2) else nf)
                else:
                    num["n_sel_unknown"] += 1
            for m in _CMP.finditer(text):
                lhs, op, anyall, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
                num["n_pred"] += 1
                r, c = self._resolve(lhs, node); cs = self.cat.column(r, c)
                if cs:
                    stats_cols.append(cs)
                rhs_is_const = rhs.startswith("'") or _num(rhs) is not None
                if not rhs_is_const:                                   # column = column: join key
                    num["n_join_key"] += 1
                    r2, c2 = self._resolve(rhs, node); cs2 = self.cat.column(r2, c2)
                    if cs2:
                        stats_cols.append(cs2)
                    continue
                const_cols.add((r, c))
                sel = None
                if op == "=":
                    num["n_eq_const"] += 1
                    if cs:
                        sel = self.cat.any_selectivity(cs, rhs) if anyall else self.cat.eq_selectivity(cs, rhs)
                elif op in ("<", "<=", ">", ">="):
                    num["n_range"] += 1
                    if cs:
                        sel = self.cat.range_selectivity(cs, op, rhs)
                elif op in ("~~", "~~*"):
                    num["n_like"] += 1
                    if cs:
                        sel = self.cat.like_selectivity(cs, rhs, op == "~~*")
                elif op in ("!~~", "!~~*"):
                    num["n_not_like"] += 1
                    if cs:
                        ls = self.cat.like_selectivity(cs, rhs, op == "!~~*")
                        sel = None if ls is None else max(1.0 - ls - (cs.get("null_frac") or 0.0), 0.0)
                elif op == "<>":
                    num["n_neq"] += 1
                    if cs and not anyall:
                        sel = 1.0 - self.cat.eq_selectivity(cs, rhs) - (cs.get("null_frac") or 0.0)
                if sel is None:
                    num["n_sel_unknown"] += 1
                else:
                    sels.append(min(max(sel, 1e-9), 1.0))
        num["n_const_cols"] = float(len(const_cols))
        if stats_cols:
            num["col_null_frac"] = float(np.mean([c.get("null_frac") or 0.0 for c in stats_cols]))
            num["col_log_ndistinct"] = float(np.mean([math.log1p(c["nd_abs"]) for c in stats_cols]))
            num["col_avg_width"] = float(np.mean([c.get("avg_width") or 0.0 for c in stats_cols]))
            num["col_abs_corr_max"] = float(max(abs(c.get("correlation") or 0.0) for c in stats_cols))
            num["col_mcv_coverage"] = float(np.mean([c["mcv_cov"] for c in stats_cols]))
        if sels:
            num["log_sel_product"] = float(np.sum(np.log(sels)))
            num["log_sel_min"] = float(np.log(min(sels)))
        return num, flag, cat

    # ------------------------------------------------------------------ fit / transform
    def fit(self, plans: Iterable[Dict[str, Any]]) -> "PGFeaturizer":
        types, rows = set(), []
        for p in plans:
            root = p["Plan"] if "Plan" in p else p
            self.bind(root)
            for n in _walk(root):
                num, _, cat = self.raw(n)
                types.add(cat["type"]); rows.append([num[k] for k in NUMERIC])
        self.type_index = {t: i for i, t in enumerate(sorted(types))}
        a = np.asarray(rows, dtype=np.float64)
        q25, q50, q75 = np.percentile(a, [25, 50, 75], axis=0)
        self.center = q50
        iqr = q75 - q25
        std = a.std(axis=0)
        self.scale = np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0))
        return self

    def node_vector(self, node: Dict[str, Any]) -> np.ndarray:
        num, flag, cat = self.raw(node)
        parts = [np.zeros(self.num_types, dtype=np.float32)]
        if cat["type"] in self.type_index:
            parts[0][self.type_index[cat["type"]]] = 1.0
        for vocab, key in ((JOIN_TYPES, "join"), (PARENT_REL, "parent"), (STRATEGIES, "strategy")):
            v = np.zeros(len(vocab), dtype=np.float32)
            if cat[key] in vocab:
                v[vocab.index(cat[key])] = 1.0
            parts.append(v)
        parts.append(np.array([flag[k] for k in FLAGS], dtype=np.float32))
        raw = np.array([num[k] for k in NUMERIC], dtype=np.float64)
        parts.append(((raw - self.center) / self.scale).astype(np.float32))
        return np.concatenate(parts)
