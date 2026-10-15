"""Reqo's database-specific operator inputs (tables, per-column predicate encoding) for the plan-selection workloads.

The encoding itself is the released code, imported unchanged: ``replace_aliases_and_columns`` and
``Node_info_extract`` / ``Text_extraction`` of ``Utils/query_plan_feature_extraction.py``.  What the release reads from a
live database (``Utils/postgresql_database_statistic_generator.py``) is taken here from the pg_catalog snapshot that the
benchmark stores next to its plans - the same snapshot GNTO's PG-stat inputs use:

* tables and columns     the snapshot's ``tables`` (sorted by name, as in the release) and ``columns``; the columns of
                         seven never-analysed dimension tables (``SMALL_TABLES``) are added from the standard schemas
* numeric / date columns ``[min, max, 1 / n_distinct]`` from the bounds of the pg_stats histogram and the most common
                         values (the release scans the column), ``n_distinct`` from pg_stats
* text columns           the release trains a one-dimensional word2vec model per column on a single "sentence" of all
                         tokens; gensim trains on the first 10,000 tokens of a sentence only, so nearly every token
                         keeps its seeded random initial value in (-0.5, 0.5).  We use a deterministic hash of the token
                         into the same interval and average over the tokens of a literal as the release does.  Tokens
                         are always treated as known (the release returns -2 if no token occurs in the column).

Five regular expressions of the release are double-escaped and never match (see ``_RepairedRe``); they are repaired,
otherwise the table and predicate inputs would be empty.  A node whose predicate text the released parser cannot handle
keeps its table inputs and gets zero column inputs; the number of such nodes is counted in ``STATS['failed_nodes']``.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import string
import sys
import types
from contextlib import redirect_stdout
from datetime import datetime

import json
import numpy as np

REQO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "baselines_src", "Reqo-on-PostgreSQL")
if "gensim" not in sys.modules:                                # the release imports gensim for its word2vec models only
    _g, _m = types.ModuleType("gensim"), types.ModuleType("gensim.models")
    _m.Word2Vec = type("Word2Vec", (), {})
    _g.models = _m
    sys.modules["gensim"], sys.modules["gensim.models"] = _g, _m
_threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
sys.path.append(REQO)
import Utils.query_plan_feature_extraction as qf              # noqa: E402  (released Reqo code)
from Utils.postgresql_database_statistic_generator import postgresql_nodes_types  # noqa: E402
for _k, _v in _threads.items():                                # the release pins both to 1 at import time
    if _v is None:
        os.environ.pop(_k, None)

STATS = {"failed_nodes": 0, "nodes": 0}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2}(\.\d+)?)?$")


def _token_value(word: str) -> float:
    n = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big")
    return n / (1 << 64) - 0.5


def str_value_encoding(values, model_path):
    """Stand-in for the one-dimensional word2vec lookup of the release (see module docstring)."""
    table = str.maketrans("", "", string.punctuation)
    words = [w for v in values for w in v.lower().translate(table).split()]
    return float(np.mean([_token_value(w) for w in words])) if words else -2


qf.str_value_encoding = str_value_encoding


class _RepairedRe:
    """Five regular expressions of the release are double-escaped (``r'\\\\b' + alias + r'\\\\.'``,
    ``r'\\\\b\\\\w+\\\\b'`` ...) and therefore never match: aliases are not resolved, columns are not qualified and no
    table is recognised, which leaves Reqo's table and predicate inputs empty for every aliased query.  The patterns
    are repaired here (double backslash -> backslash) without touching the released file."""

    @staticmethod
    def _fix(pattern):
        return pattern.replace("\\\\", "\\") if isinstance(pattern, str) else pattern

    def __getattr__(self, name):
        return getattr(re, name)

    def compile(self, pattern, flags=0):
        return re.compile(self._fix(pattern), flags)

    def findall(self, pattern, string, flags=0):
        return re.findall(self._fix(pattern), string, flags)

    def match(self, pattern, string, flags=0):
        return re.match(self._fix(pattern), string, flags)

    def search(self, pattern, string, flags=0):
        return re.search(self._fix(pattern), string, flags)

    def sub(self, pattern, repl, string, count=0, flags=0):
        return re.sub(self._fix(pattern), repl, string, count=count, flags=flags)


qf.re = _RepairedRe()


def parse_pg_array(text):
    """Elements of a PostgreSQL array literal ``{a,"b c",...}`` as strings."""
    if text is None or text in ("None", "") or not str(text).startswith("{"):
        return []
    s, out, cur, quoted, i = str(text)[1:-1], [], [], False, 0
    while i < len(s):
        ch = s[i]
        if quoted:
            if ch == "\\" and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 1
            elif ch == '"':
                quoted = False
            else:
                cur.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == ",":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


def _to_date(v):
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            pass
    return None


# Dimension tables that PostgreSQL never analysed (a handful of rows), so the snapshot has no pg_stats rows for them.
# The release would list their columns from information_schema; (column, min, max) for keys, None for text.
SMALL_TABLES = {
    "role_type": [("id", 1, 12), ("role", None)], "kind_type": [("id", 1, 7), ("kind", None)],
    "company_type": [("id", 1, 4), ("kind", None)], "link_type": [("id", 1, 18), ("link", None)],
    "comp_cast_type": [("id", 1, 4), ("kind", None)],
    "nation": [("n_nationkey", 0, 24), ("n_name", None), ("n_regionkey", 0, 4), ("n_comment", None)],
    "region": [("r_regionkey", 0, 4), ("r_name", None), ("r_comment", None)],
}


def load_stats(catalog_path: str):
    cat = json.load(open(catalog_path))
    tables = sorted(cat["tables"])
    for t, cols in SMALL_TABLES.items():                       # only where the snapshot knows the table but no column
        if t in cat["tables"] and not any(c.split(".", 1)[0] == t for c in cat["columns"]):
            for name, *rng in cols:
                n = rng[1] - rng[0] + 1 if rng[0] is not None else max(r[2] - r[1] + 1 for r in cols if r[1] is not None)
                cat["columns"][f"{t}.{name}"] = {"n_distinct": float(n), "mcv": None,
                                                 "hist": "{%d,%d}" % (rng[0], rng[1]) if rng[0] is not None else '{"text"}'}
    tables_index, tables_index_all, columns_index, columns_list, attribute_range, per_table = {}, {}, {}, [], {}, []
    kinds = {"numeric": 0, "date": 0, "text": 0, "other": 0}
    for i, t in enumerate(tables):
        tables_index[t] = tables_index_all[t] = i
        cols = [c for c in cat["columns"] if c.split(".", 1)[0] == t]
        per_table.append(len(cols))
        reltuples = float(cat["tables"][t].get("reltuples", 0) or 0)
        for full in cols:
            c = cat["columns"][full]
            tables_index_all[full] = i
            columns_index[full] = len(columns_index)
            columns_list.append(full.split(".", 1)[1])
            nd = float(c.get("n_distinct") or 0)
            nd = -nd * reltuples if nd < 0 else nd
            share = 1.0 / nd if nd > 0 else 0.0
            values = [v for v in parse_pg_array(c.get("hist")) + parse_pg_array(c.get("mcv")) if v != "NULL"]
            try:
                nums = [float(v) for v in values]
            except ValueError:
                nums = None
            if values and nums is not None:
                attribute_range[full] = [min(nums), max(nums), share]
                kinds["numeric"] += 1
            elif values and all(_DATE.match(v) for v in values):
                dts = [_to_date(v) for v in values]
                attribute_range[full] = [min(dts), max(dts), share]
                kinds["date"] += 1
            elif values:
                attribute_range[full] = ["hash", share]
                kinds["text"] += 1
            else:
                attribute_range[full] = [None, None, 0.0]
                kinds["other"] += 1
    return {"tables_index": tables_index, "tables_index_all": tables_index_all, "columns_index": columns_index,
            "columns_list": sorted(set(columns_list)), "attribute_range": attribute_range,
            "table_columns_number": per_table, "nodes": postgresql_nodes_types(), "column_kinds": kinds,
            "catalog": catalog_path}


def rewrite_plan(root, db):
    """Aliases -> table names and qualified column names (``replace_aliases_and_columns`` of the release)."""
    return qf.replace_aliases_and_columns(root, db["columns_list"])


_FAKE = [["Total Cost", "Plan Rows"], [0.0, 0.0], [1.0, 1.0]]


def node_text_encoding(node, db):
    """[tables | 8 values per column] of one operator of a rewritten plan (``Node_info_extract`` of the release)."""
    STATS["nodes"] += 1
    width = len(db["tables_index"]) + 8 * len(db["columns_index"])
    try:
        with redirect_stdout(io.StringIO()):
            v = qf.Node_info_extract(node, db["tables_index"], db["tables_index_all"], db["columns_index"],
                                     db["attribute_range"], db["nodes"], _FAKE)
        return np.asarray(v[len(db["nodes"]) + 2:], dtype=np.float32)
    except Exception:
        STATS["failed_nodes"] += 1
        out = np.zeros(width, dtype=np.float32)
        rel = node.get("Relation Name")
        if rel in db["tables_index"]:
            out[db["tables_index"][rel]] = 1.0
        return out
